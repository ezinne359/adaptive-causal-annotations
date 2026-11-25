"""Cross-fitted annotation pipeline with separate sampling and treatment weights.

All outcome fitting receives observed labels only. Treatment weights never
receive outcomes. Random/adaptive policies share uniforms and weighting models.
"""
import numpy as np
from .allocation import rf, weight_models, sbw, pilot_predictions, solve
from .scores import score_summary
from .propensity import drain_audits

WEIGHTS = ('plugin', 'balance', 'sbw')
POLICIES = ('random', 'adaptive')

class ZeroLabelError(ValueError):
    pass


def make_split(n, seed=20261022, pilot_fraction=.30):
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    fold = np.zeros(n, dtype=int)
    fold[order[n//2:]] = 1
    pilot = np.zeros(n, dtype=bool)
    for f in (0, 1):
        ids = np.flatnonzero(fold == f)
        pilot[rng.permutation(ids)[:int(pilot_fraction*len(ids))]] = True
    return fold, pilot


def fit_weights(x, z, fold, propensity='logistic'):
    drain_audits()
    weights, clipping, _ = weight_models(x, z, fold, propensity)
    audit = {"propensity_fits": drain_audits()}; failures = {}
    try:
        weights['sbw'], audit['sbw'] = sbw(x, z, fold, .20)
        clipping['sbw'] = np.zeros(len(z), dtype=bool)
    except ValueError as exc:
        # An infeasible frozen tolerance is a method failure, not permission to retune.
        failures['sbw'] = str(exc)
    for name, w in weights.items():
        if not np.isfinite(w).all() or ((2*z-1)*w < 0).any():
            raise ValueError(f'Invalid {name} treatment weights')
    return weights, clipping, audit, failures


def designs(x, z, observed_pilot_y, pilot, pilot_r, fold, budget, uniforms, propensity='logistic'):
    """Same pooled, cross-fitted allocation as the completed synthetic study."""
    e, v, fallback = pilot_predictions(x, z, observed_pilot_y, pilot, pilot_r, fold, 'cross', propensity)
    kappa = pilot.mean()
    for policy in POLICIES:
        if policy == 'random':
            q = np.full(len(z), budget)
            failed = np.zeros(len(z), dtype=bool)
        else:
            failed = fallback
            scale = np.sqrt(v)/np.where(z == 1, e, 1-e)
            if failed.any():
                scale[failed] = np.median(scale[~failed]) if (~failed).any() else 1.
            q = solve(scale, ~pilot, budget, kappa, pooled=True)
        stage = np.where(pilot, budget, q)
        pooled = kappa*budget+(1-kappa)*q
        r = uniforms < stage
        np.testing.assert_array_equal(r[pilot], pilot_r[pilot])
        np.testing.assert_allclose(stage.sum(), len(z)*budget, rtol=0, atol=1e-8)
        yield policy, r, pooled, stage, int(failed.sum())


def outcome_predictions(x, z, observed_y, proxy, r, fold, method):
    """Two-fold predictions; a singleton fits normally, zero labels fail explicitly."""
    n = len(z)
    base = np.zeros((2, n)); aug = np.zeros((2, n)); counts = []
    for f in (0, 1):
        test = fold == f
        for arm in (0, 1):
            train = (~test) & r & (z == arm)
            count = int(train.sum()); counts.append(count)
            if count == 0:
                raise ZeroLabelError(f'zero_outcome_labels: prediction_fold={f}, arm={arm}')
            if not np.isfinite(observed_y[train]).all():
                raise ValueError('Missing observed training labels')
            xa = np.column_stack((x[train], proxy[train]))
            model_aug = rf().fit(xa, observed_y[train])
            # Preserve the two application outcome-model definitions.
            target = model_aug.predict(xa) if method == 'llm-eff' else observed_y[train]
            model_base = rf().fit(x[train], target)
            base[arm, test] = model_base.predict(x[test])
            aug[arm, test] = model_aug.predict(np.column_stack((x[test], proxy[test])))
    return base, aug, counts


def progress_scores(z, r, observed_y, base, aug, w, pi):
    """Corrected two-component score with separate treatment and sampling weights.

    base1-base0 + w*(aug_Z-base_Z) + R*w/pi*(Y-aug_Z).
    The first correction uses all proxies; only the second uses observed labels.
    With aug==base this equals synthetic signed_weight_scores exactly.
    """
    if base.shape != (2, len(z)) or aug.shape != base.shape:
        raise ValueError('Both counterfactual predictions must cover all clients')
    if not (np.isfinite(base).all() and np.isfinite(aug).all() and np.isfinite(w).all()):
        raise ValueError('Nonfinite model predictions or treatment weights')
    if not np.isfinite(pi).all() or (pi <= 0).any() or (pi > 1).any():
        raise ValueError('Invalid annotation probabilities')
    i = np.arange(len(z)); factual_base = base[z, i]; factual_aug = aug[z, i]
    scores = base[1]-base[0] + w*(factual_aug-factual_base)
    if not np.isfinite(observed_y[r]).all():
        raise ValueError('Missing observed scoring labels')
    scores[r] += w[r]/pi[r]*(observed_y[r]-factual_aug[r])
    return scores


def benchmark(x, z, y, proxy, fold, method, weights):
    r = np.ones(len(z), dtype=bool)
    base, aug, _ = outcome_predictions(x, z, y, proxy, r, fold, method)
    return {name: score_summary(progress_scores(z, r, y, base, aug, w, np.ones(len(z))))
            for name, w in weights.items()}


def draw(x, z, y, proxy, fold, pilot, weights, clipping, weight_failures, refs, method, budget, trial, propensity='logistic'):
    seed = 20263000 + 100000*int(round(10*budget)) + trial
    u = np.random.RandomState(seed).uniform(size=len(z))
    pilot_r = pilot & (u < budget)
    pilot_y = np.where(pilot_r, y, np.nan)
    rows = []
    for policy, r, pi, stage, fallback in designs(x, z, pilot_y, pilot, pilot_r, fold, budget, u, propensity):
        observed = np.where(r, y, np.nan)
        failure = ''; counts = []
        try:
            base, aug, counts = outcome_predictions(x, z, observed, proxy, r, fold, method)
        except ZeroLabelError as exc:
            failure = str(exc)
        for name in WEIGHTS:
            reason = weight_failures.get(name, '') or failure
            row = dict(method=method, trial=trial, annotation_seed=seed, budget=budget,
                       policy=policy, weighting=name, comparison_status='failed' if reason else 'completed',
                       failure_reason=reason, annotations=int(r.sum()), expected_annotations=float(stage.sum()),
                       pilot_annotations=int(pilot_r.sum()), pilot_fraction=float(pilot.mean()),
                       fallback_units=fallback, singleton_cells=counts.count(1),
                       stage_pi_lower_fraction=float(np.mean(stage[~pilot] <= .01+1e-10)),
                       stage_pi_upper_fraction=float(np.mean(stage[~pilot] >= .99-1e-10)))
            if name in weights:
                w = weights[name]
                row['weight_clipping_fraction'] = float(clipping[name].mean())
                row['max_treatment_weight'] = float(np.abs(w).max())
                for arm in (0, 1):
                    mask = r & (z == arm); a = np.abs(w[mask]/pi[mask])
                    row[f'ess_arm{arm}'] = float(a.sum()**2/(a@a)) if a@a else 0.
            if not reason:
                a = score_summary(progress_scores(z, r, observed, base, aug, weights[name], pi))
                ref = refs[name]['estimate']; common = refs['plugin']['estimate']
                row.update(a, benchmark=ref, common_plugin_benchmark=common,
                           squared_error=(a['estimate']-ref)**2,
                           common_benchmark_squared_error=(a['estimate']-common)**2,
                           coverage=float(a['lower'] <= ref <= a['upper']), width=a['upper']-a['lower'])
            rows.append(row)
    return rows
