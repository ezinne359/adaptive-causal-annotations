import unittest
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from corrected_annotation.propensity import ForestPropensity,drain_audits

class ForestPropensityTests(unittest.TestCase):
    def test_exact_classifier_parity_and_one_arm_failure(self):
        rng=np.random.RandomState(9);x=rng.normal(size=(70,3));z=np.tile([0,1],35)
        actual=ForestPropensity(1051).fit(x,z)
        expected=RandomForestClassifier(n_estimators=200,min_samples_leaf=10,random_state=1051,n_jobs=1).fit(x,z)
        np.testing.assert_array_equal(actual.predict_proba(x),expected.predict_proba(x))
        with self.assertRaises(ValueError): ForestPropensity().fit(x,np.zeros(70))
        self.assertTrue(drain_audits()[-1]['audit_passed'])
