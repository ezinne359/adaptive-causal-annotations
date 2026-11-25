"""Converged C=1 L2 logistic propensity, with an unpenalized intercept.

Centering improves conditioning without changing the fitted objective or
rescaling its penalty. Every fit must pass an explicit gradient audit.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

_AUDITS = []


def drain_audits():
    result = list(_AUDITS)
    _AUDITS.clear()
    return result


class ConvergedPropensity:
    def fit(self, x, z):
        x = np.asarray(x, dtype=float); z = np.asarray(z, dtype=float)
        if x.ndim != 2 or z.shape != (len(x),) or not np.isfinite(x).all():
            raise ValueError('Invalid propensity covariates or treatment shape')
        if not np.array_equal(np.unique(z), [0., 1.]):
            raise ValueError('Both treatment groups are required for propensity fitting')
        self.mean_ = x.mean(axis=0)
        design = np.column_stack((x-self.mean_, np.ones(len(x))))
        penalty = np.r_[np.ones(x.shape[1]), 0.]

        def objective(beta):
            logits = design@beta
            return float(np.logaddexp(0, logits).sum()-z@logits + .5*np.sum(penalty*beta**2))

        def gradient(beta):
            return design.T@(expit(design@beta)-z)+penalty*beta

        def hessian(beta):
            prob = expit(design@beta)
            return design.T@((prob*(1-prob))[:, None]*design)+np.diag(penalty)

        initial = np.zeros(design.shape[1]); initial[-1] = np.log(z.mean()/(1-z.mean()))
        result = minimize(objective, initial, jac=gradient, hess=hessian, method='trust-exact',
                          options={'gtol':1e-8, 'maxiter':200})
        beta = result.x.copy()
        # Newton polishing handles termination at floating-point objective precision.
        for _ in range(5):
            if np.max(np.abs(gradient(beta))) < 1e-8:
                break
            candidate = beta-np.linalg.solve(hessian(beta), gradient(beta))
            if objective(candidate) > objective(beta)+1e-9:
                break
            beta = candidate
        error = float(np.max(np.abs(gradient(beta))))
        if not np.isfinite(beta).all() or not np.isfinite(error) or error >= 1e-6:
            raise RuntimeError(f'Propensity convergence audit failed: max gradient {error}')
        self.beta_ = beta; self.classes_ = np.array([0, 1])
        self.audit_ = dict(n=len(x),n_control=int((z==0).sum()),iterations=int(result.nit),
                           objective=objective(beta),max_abs_gradient=error,
                           optimizer_reported_success=bool(result.success),audit_passed=True,
                           solver='centered trust-exact; C=1 L2; unpenalized intercept')
        _AUDITS.append(self.audit_)
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        p = expit((x-self.mean_)@self.beta_[:-1]+self.beta_[-1])
        return np.column_stack((1-p, p))


class ForestPropensity:
    """Fixed PNAS-style binary forest; probability clipping is applied by callers."""
    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, x, z):
        from sklearn.ensemble import RandomForestClassifier
        x = np.asarray(x, dtype=float); z = np.asarray(z)
        if x.ndim != 2 or z.shape != (len(x),) or not np.isfinite(x).all():
            raise ValueError('Invalid propensity covariates or treatment shape')
        if not np.array_equal(np.unique(z), [0, 1]):
            raise ValueError('Both treatment groups are required for propensity fitting')
        self.model_ = RandomForestClassifier(n_estimators=200, min_samples_leaf=10,
                                             random_state=self.seed, n_jobs=1).fit(x, z)
        self.classes_ = self.model_.classes_
        self.audit_ = dict(n=len(x), n_control=int((z==0).sum()),
                          learner='forest', n_estimators=200, min_samples_leaf=10,
                          seed=self.seed, audit_passed=True)
        _AUDITS.append(self.audit_)
        return self

    def predict_proba(self, x):
        p = self.model_.predict_proba(x)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('Invalid forest propensity predictions')
        return p


def propensity_model(learner='logistic', seed=42):
    if learner == 'forest':
        return ForestPropensity(seed)
    if learner == 'logistic':
        return ConvergedPropensity()
    raise ValueError(f'Unknown propensity learner: {learner}')
