"""Population-average AIPW scores, separate from the legacy replication.

Both outcome predictions must cover ALL units. This fixes score assembly only:
it does not repair nuisance fitting, adaptive sampling, or establish validity
of Wald intervals for an arbitrary fitted/adaptive pipeline.
"""
import numpy as np
from scipy.stats import norm


def _arrays(z, r, y, mu1, mu0):
    arrays = [np.asarray(a, dtype=float) for a in (z, r, y, mu1, mu0)]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Inputs must be equally sized one-dimensional arrays.')
    z, r, y, mu1, mu0 = arrays
    if not np.isin(z, [0, 1]).all() or not np.isin(r, [0, 1]).all():
        raise ValueError('Treatment and annotation indicators must be binary.')
    if not all(np.isfinite(a).all() for a in (mu1, mu0)) or not np.isfinite(y[r == 1]).all():
        raise ValueError('Predictions and observed outcomes must be finite.')
    return z, r, y, mu1, mu0


def signed_weight_scores(z, r, y, mu1, mu0, signed_joint_weights):
    """Weights represent Z/(e*pi1) - (1-Z)/((1-e)*pi0).

    Do not pass a representer fitted to the binary variable Z*R: its negative
    arm describes Z*R=0, which mixes unannotated treated units with controls.
    """
    z, r, y, mu1, mu0 = _arrays(z, r, y, mu1, mu0)
    weights = np.asarray(signed_joint_weights, dtype=float)
    if weights.shape != z.shape or not np.isfinite(weights).all():
        raise ValueError('Weights must be finite and have one value per unit.')
    if (weights[z == 1] < 0).any() or (weights[z == 0] > 0).any():
        raise ValueError('Signed weights must be positive for treatment and negative for control.')
    scores = mu1 - mu0
    observed = r == 1
    factual_mu = z * mu1 + (1 - z) * mu0
    # Never evaluate hidden outcomes, including NaNs, in the residual term.
    scores[observed] += weights[observed] * (y[observed] - factual_mu[observed])
    return scores


def aipw_scores(z, r, y, mu1, mu0, propensity, annotation_probability):
    z, r, y, mu1, mu0 = _arrays(z, r, y, mu1, mu0)
    e = np.asarray(propensity, dtype=float)
    pi = np.asarray(annotation_probability, dtype=float)
    if e.shape != z.shape or pi.shape != z.shape:
        raise ValueError('Probabilities must have one value per unit.')
    if not np.isfinite(e).all() or not np.isfinite(pi).all():
        raise ValueError('Probabilities must be finite.')
    if ((e <= 0) | (e >= 1)).any() or ((pi <= 0) | (pi > 1)).any():
        raise ValueError('Require 0 < propensity < 1 and 0 < annotation probability <= 1.')
    weights = (z / e - (1 - z) / (1 - e)) / pi
    return signed_weight_scores(z, r, y, mu1, mu0, weights)


def joint_probability_scores(z, r, y, mu1, mu0, treated_annotated, control_annotated):
    """Use separately estimated P(Z=1,R=1|X) and P(Z=0,R=1|X).

    These are disjoint events, not complementary events when labels are missing.
    For an adaptive design the conditioning must also respect its history.
    """
    z, r, y, mu1, mu0 = _arrays(z, r, y, mu1, mu0)
    q1 = np.asarray(treated_annotated, dtype=float)
    q0 = np.asarray(control_annotated, dtype=float)
    if q1.shape != z.shape or q0.shape != z.shape:
        raise ValueError('Joint probabilities must have one value per unit.')
    if (not np.isfinite(q1).all() or not np.isfinite(q0).all()
            or (q1 <= 0).any() or (q0 <= 0).any() or (q1 + q0 > 1 + 1e-12).any()):
        raise ValueError('Joint probabilities must be positive, finite, and sum to at most one.')
    return signed_weight_scores(z, r, y, mu1, mu0, z / q1 - (1-z) / q0)


def combine_batch_features(first, second):
    """Stack unchanged feature matrices; treatment prefixes belong only in forest queries."""
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError('Both batches must use the same feature columns in the same order.')
    return np.concatenate([first, second], axis=0)


def score_summary(scores, alpha=0.05):
    """IID influence-score Wald summary; nuisance fitting needs separate validation."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError('Need at least two finite scores.')
    estimate = float(scores.mean())
    variance = float(scores.var(ddof=1) / len(scores))
    se = np.sqrt(variance)
    half_width = norm.ppf(1 - alpha / 2) * se
    return dict(estimate=estimate, variance=variance, se=float(se),
                lower=float(estimate - half_width), upper=float(estimate + half_width))
