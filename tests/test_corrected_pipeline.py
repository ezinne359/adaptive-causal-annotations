"""Regression checks for the legacy failures and parity with synthetic scores."""
import unittest
import numpy as np
from corrected_annotation.pipeline import outcome_predictions, progress_scores, designs, make_split, ZeroLabelError
from corrected_annotation.scores import signed_weight_scores, combine_batch_features
from corrected_annotation.allocation import sbw

class CorrectedProgressTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.RandomState(342)
        self.x=rng.normal(size=(80,3)); self.z=np.tile([0,1],40)
        self.y=rng.normal(size=80);self.l=rng.normal(size=(80,1))
        self.fold=np.repeat([0,1],40);self.r=np.tile([True,True,False,False],20)

    def test_synthetic_score_parity_and_hidden_labels(self):
        base=np.vstack((self.y*.1,self.y*.2));w=2*(2*self.z-1);pi=np.full(80,.5)
        obs=np.where(self.r,self.y,np.nan)
        actual=progress_scores(self.z,self.r,obs,base,base,w,pi)
        expected=signed_weight_scores(self.z,self.r,obs,base[1],base[0],w/pi)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        poisoned=np.where(self.r,self.y,1e12)
        np.testing.assert_array_equal(actual,progress_scores(self.z,self.r,poisoned,base,base,w,pi))

    def test_no_hidden_labels_in_fitting_and_no_feature_mutation(self):
        x=self.x.copy();obs=np.where(self.r,self.y,np.nan)
        for method in ['llm-eff','llm-residual']:
            a=outcome_predictions(x,self.z,obs,self.l,self.r,self.fold,method)
            b=outcome_predictions(x,self.z,np.where(self.r,self.y,1e12),self.l,self.r,self.fold,method)
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_array_equal(x,self.x)
        np.testing.assert_array_equal(combine_batch_features(x[:40],x[40:]),x)

    def test_heldout_outcomes_do_not_change_heldout_predictions(self):
        a=outcome_predictions(self.x,self.z,self.y,self.l,self.r,self.fold,'llm-eff')
        y=self.y.copy();y[self.fold==0]+=1000
        b=outcome_predictions(self.x,self.z,y,self.l,self.r,self.fold,'llm-eff')
        np.testing.assert_array_equal(a[0][:,self.fold==0],b[0][:,self.fold==0])
        np.testing.assert_array_equal(a[1][:,self.fold==0],b[1][:,self.fold==0])

    def test_singleton_and_zero_labels(self):
        r=np.zeros(80,bool);r[[0,1,40,41]]=True
        a=outcome_predictions(self.x,self.z,self.y,self.l,r,self.fold,'llm-eff')
        self.assertEqual(a[2],[1,1,1,1])
        r[0]=False
        with self.assertRaises(ZeroLabelError):
            outcome_predictions(self.x,self.z,self.y,self.l,r,self.fold,'llm-eff')

    def test_exact_budget_shared_pilot_and_no_policy_label_leak(self):
        fold,pilot=make_split(80);u=np.random.RandomState(33).uniform(size=80);budget=.4
        r=pilot&(u<budget);obs=np.where(r,self.y,np.nan)
        a=list(designs(self.x,self.z,obs,pilot,r,fold,budget,u))
        b=list(designs(self.x,self.z,np.where(r,self.y,1e12),pilot,r,fold,budget,u))
        for aa,bb in zip(a,b):
            np.testing.assert_array_equal(aa[1],bb[1]);np.testing.assert_array_equal(aa[2],bb[2])
            self.assertAlmostEqual(aa[3].sum(),80*budget)
            np.testing.assert_array_equal(aa[1][pilot],r[pilot])
        np.testing.assert_array_equal(a[0][1][pilot],a[1][1][pilot])

    def test_sbw_constant_covariate(self):
        # Paired treatment groups have identical covariates: exact balance is feasible.
        x=np.repeat(self.x[:40],2,axis=0);x=np.column_stack((x,np.ones(80)))
        w,audit=sbw(x,self.z,self.fold,.20)
        self.assertTrue(np.isfinite(w).all());self.assertEqual(len(audit),4)
        self.assertTrue(all(a['imbalance']<=.20+1e-7 for a in audit))

class PropensityConvergenceTests(unittest.TestCase):
    def test_intercept_only(self):
        from corrected_annotation.propensity import ConvergedPropensity
        x=np.ones((100,3))*700000;z=np.r_[np.zeros(30),np.ones(70)]
        m=ConvergedPropensity().fit(x,z)
        np.testing.assert_allclose(m.predict_proba(x)[:,1],.7,atol=1e-12)
        self.assertLess(m.audit_['max_abs_gradient'],1e-6)

    def test_translation_invariance_and_same_l2_objective(self):
        from corrected_annotation.propensity import ConvergedPropensity
        from sklearn.linear_model import LogisticRegression
        rng=np.random.RandomState(142);x=rng.normal(size=(300,5));z=rng.binomial(1,1/(1+np.exp(-x[:,0]+x[:,1])))
        a=ConvergedPropensity().fit(x,z)
        shift=np.array([700000,0,0,0,0]);b=ConvergedPropensity().fit(x+shift,z)
        ref=LogisticRegression(C=1,penalty='l2',max_iter=10000,tol=1e-11).fit(x,z)
        np.testing.assert_allclose(a.predict_proba(x),b.predict_proba(x+shift),atol=1e-9,rtol=0)
        np.testing.assert_allclose(a.predict_proba(x),ref.predict_proba(x),atol=1e-6,rtol=0)
        self.assertLess(b.audit_['max_abs_gradient'],1e-6)

if __name__=='__main__': unittest.main()
