"""Allocation and weighting functions used in the revision experiments."""
import numpy as np
import cvxpy as cp
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor
from .propensity import propensity_model
from forestriesz import ForestRieszATE
CONFIG = {"lower": .01, "upper": .99}


def rf(oob=False):
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=4,
        min_samples_split=10, random_state=42, n_jobs=1, oob_score=oob)

def weight_models(x,z,fold,propensity='logistic'):
    weights={key:np.zeros(len(z)) for key in ['plugin','balance']}
    clipping={key:np.zeros(len(z),bool) for key in weights}
    policy_e=np.zeros(len(z))
    for f in range(2):
        train=fold==f; test=~train
        model=propensity_model(propensity,42+1009*(1-f)).fit(x[train],z[train])
        col=int(np.flatnonzero(model.classes_==1)[0])
        policy_e[train]=np.clip(model.predict_proba(x[train])[:,col],.01,.99)
        e_raw=model.predict_proba(x[test])[:,col]
        e=np.clip(e_raw,.01,.99)
        weights['plugin'][test]=z[test]/e-(1-z[test])/(1-e)
        clipping['plugin'][test]=(e_raw!=e)
        forest=ForestRieszATE(criterion='het',n_estimators=1000,min_samples_leaf=2,
            min_var_fraction_leaf=.001,min_var_leaf_on_val=True,min_impurity_decrease=.01,
            max_samples=.8,max_depth=None,warm_start=False,inference=False,
            subforest_size=1,honest=True,verbose=0,n_jobs=1,random_state=123)
        # Only the treatment representer is used: no hidden outcomes enter the forest.
        forest.fit(x[train],z[train,None],np.zeros((train.sum(),1)))
        raw,_=forest.predict_riesz_and_reg(np.c_[z[test],x[test]])
        if not np.isfinite(raw).all() or ((2*z[test]-1)*raw<=0).any():
            raise ValueError('Invalid learned Riesz signs or nonfinite weights')
        clipped=np.sign(raw)*np.clip(np.abs(raw),1/.99,1/.01)
        weights['balance'][test]=clipped
        clipping['balance'][test]=np.abs(raw-clipped)>1e-12
    return weights,clipping,policy_e

def solve(scale, calibration, budget, kappa, pooled=False, arm=None):
    """Return q(X,Z), calibrated on stage two, with its exact budget bounds."""
    scale=np.maximum(np.asarray(scale,float),1e-12)
    answer=np.zeros(len(scale))
    groups=[np.ones(len(scale),bool)] if arm is None else [arm==0,arm==1]
    for group in groups:
        use=calibration&group
        if not use.any():raise ValueError('Empty calibration group')
        def probs(c):
            value=c*scale
            if pooled:value=(value-kappa*budget)/(1-kappa)
            return np.clip(value,.01,.99)
        lo,hi=0.,1.
        while probs(hi)[use].mean()<budget:hi*=2
        for _ in range(100):
            mid=(lo+hi)/2
            if probs(mid)[use].mean()<budget:lo=mid
            else:hi=mid
        answer[group]=probs((lo+hi)/2)[group]
        np.testing.assert_allclose(answer[use].mean(),budget,rtol=0,atol=1e-12)
    return answer

def pilot_predictions(x,z,observed_y,pilot,r,fold,mode,propensity='logistic'):
    e=np.zeros(len(z));v=np.zeros(len(z));fallback=np.zeros(len(z),bool)
    groups=[np.ones(len(z),bool)] if mode=='all' else [fold==f for f in [0,1]]
    for group_index,test in enumerate(groups):
        train=pilot if mode=='all' else pilot&(test if mode=='own' else ~test)
        model=propensity_model(propensity,42+1009*group_index).fit(x[train],z[train])
        e[test]=np.clip(model.predict_proba(x[test])[:,list(model.classes_).index(1)],.01,.99)
        if any(np.sum(train&r&(z==arm))<2 for arm in [0,1]):
            fallback[test]=True
            v[test]=1
            continue
        for arm in [0,1]:
            labels=train&r&(z==arm)
            model=rf(oob=True).fit(x[labels],observed_y[labels])
            beta,_=nnls(np.c_[np.ones(labels.sum()),x[labels]],(observed_y[labels]-model.oob_prediction_)**2)
            target=test&(z==arm)
            v[target]=np.maximum(np.c_[np.ones(target.sum()),x[target]]@beta,1e-8)
    return e,v,fallback

def sbw(x, z, fold, tolerance):
    """Direct calibration within each evaluation fold; outcomes are never inputs."""
    result = np.zeros(len(z)); audit = []
    for f in [0, 1]:
        ids = np.flatnonzero(fold == f); n = len(ids)
        features = x[ids]
        sd = features.std(axis=0, ddof=0)
        # Constant columns have zero imbalance; do not divide by zero.
        features = (features[:, sd > 1e-12]-features[:, sd > 1e-12].mean(axis=0))/sd[sd > 1e-12]
        for arm in [0, 1]:
            in_arm = z[ids] == arm; a = features[in_arm]; m = len(a)
            if m < 2:
                raise ValueError('Insufficient treatment-arm observations for SBW')
            w = cp.Variable(m)
            constraints = [w >= 0, cp.sum(w) == 1, a.T @ w <= tolerance, a.T @ w >= -tolerance]
            problem = cp.Problem(cp.Minimize(cp.sum_squares(w)), constraints)
            problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10, max_iter=300)
            if problem.status != 'optimal':
                raise ValueError(f'SBW solver status {problem.status}')
            values = np.asarray(w.value)
            if values.min() < -1e-8:
                raise ValueError('Negative SBW weight')
            values = np.maximum(values, 0)
            imbalance = float(np.max(np.abs(a.T @ values)))
            normalization = float(abs(values.sum()-1))
            if imbalance > tolerance+1e-7 or normalization > 1e-7:
                raise ValueError('SBW constraints failed numerical audit')
            result[ids[in_arm]] = (2*arm-1)*n*values
            audit.append(dict(fold=f, arm=arm, tolerance=tolerance, status=problem.status,
                              imbalance=imbalance, normalization_error=normalization,
                              max_weight=float(n*values.max()), zero_fraction=float(np.mean(values<1e-8))))
    return result, audit
