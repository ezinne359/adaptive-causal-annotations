# Corrected annotation implementation

This module carries the tested revision fixes into the public repository.
It does not overwrite historical notebook outputs or reproduce the paper figures.

## Changes

- Compute population-average influence scores with both potential-outcome
  predictions on every unit. Keep signed treatment weights and annotation
  probabilities separate. Missing outcomes may be NaN and are never evaluated.
- Fit treatment representers to Z, not Z*R. The latter mixes unlabeled treated
  observations with controls. ForestRiesz receives dummy outcomes for weight fitting.
- Keep covariate arrays unchanged. Treatment prefixes appear only in forest queries.
- Cross-fit outcome predictions. Train each arm only on observed labels in the
  opposite fold. Singleton cells fit; zero-label cells raise an explicit error.
- Read the treated class probability for allocation. A signed Riesz weight is
  not a propensity probability.
- Use shared pilot draws and uniforms for random/adaptive comparisons. Calibrate
  the expected stage-two budget while retaining pooled probabilities for scoring.
- Provide a centered, convergence-audited C=1 logistic fit and an optional forest
  propensity (200 trees, minimum leaf size 10). Clipping remains [.01,.99].
- Retain stable balancing weights, with constant covariates removed before scaling
  and an explicit infeasibility failure at the fixed tolerance.

## API and migration

`pipeline.make_split` returns two folds and a pilot partition. `fit_weights`
fits treatment weights using X and Z only; its propensity option is `logistic`
by default or `forest`. `outcome_predictions` takes observed outcomes (NaN for
unobserved labels) and returns arrays of shape (2,n) in control/treated order.
The proxy must be an (n,k) array. `designs` learns allocation from pilot labels.
`progress_scores` assembles the two-component score; at full labels it reduces
to `mu1-mu0 + signed_weight*(Y-mu_factual)`.

`draw` is a retrospective experiment helper: it accepts fully labeled outcomes
only to reveal the sampled subset and evaluate repeated annotation draws. It is
not an interface for deploying a real labeling service. It returns comparison
failures rather than silently dropping zero-label draws. `benchmark` returns
full-label estimates, not known causal truth. Do not report benchmark coverage
as population causal coverage. `score_summary` gives nominal score-based Wald
intervals; it does not establish inference validity for arbitrary adaptive fits.

The outcome modes retain the revision definitions: `llm-residual` fits the base
forest to observed outcomes; `llm-eff` projects augmented-model training predictions
onto X. The same proxy-augmented outcome forest supplies the annotation correction.

A migration must pass both counterfactual predictions for all units and preserve
feature column order. The historical arm-only prediction arrays cannot be passed
to `scores.aipw_scores`. Do not compare new scores against old full-data benchmarks;
recompute each benchmark with the same nuisance definitions as its sampled estimator.
No automatic migration of old notebooks is included in this PR.

## Validation and scope

Tests cover score algebra, hidden-label invariance, held-out prediction invariance,
feature preservation, shared pilot sampling, budget calibration, zero/singleton
cells, constant covariates in SBW, logistic objective/translation invariance, and
forest classifier parity. All test inputs are generated; no private data or
experiment artifacts are required. Forest versus logistic is an explicit model
choice, not a bug fix or evidence of superior causal accuracy.
