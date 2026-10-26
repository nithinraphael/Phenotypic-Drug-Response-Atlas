import unittest
from unittest.mock import patch

import numpy as np
from thunor.curve_fit import HillCurveLL4

from analysis.fitting import fit_hill


class FittingTests(unittest.TestCase):
    def setUp(self):
        self.x = np.geomspace(0.001, 100, 30)
        self.y = HillCurveLL4.fit_fn(self.x, 2.5, 0.1, 1.2, 0.3)

    def test_clean_sigmoid_parameters_and_effect_range(self):
        fit = fit_hill(self.x,self.y)
        self.assertEqual(fit.status,'ok')
        np.testing.assert_allclose([fit.hill_slope,fit.lower_asymptote,fit.upper_asymptote,fit.ec50],[2.5,0.1,1.2,0.3],rtol=1e-5)
        np.testing.assert_allclose(fit.predict(np.array([fit.ec20,fit.ec80])).ok,[0.98,0.32],atol=1e-5)
        self.assertGreater(fit.r2,0.99999)

    def test_invalid_inputs_return_no_fabricated_parameters(self):
        fixtures = [(None,None),([],[]),([1,2],[1]),(['x'],[1]),([0,1,2,3,4],[1,0.9,0.8,0.4,0.2]),
                    ([np.nan,1,2,3,4],[1,2,3,4,5]),(self.x,np.full(30,np.inf)),([1,1,2,2,3,3],[1,1,0.5,0.5,0,0])]
        for x,y in fixtures:
            with self.subTest(x=str(x)[:30]):
                fit=fit_hill(x,y)
                self.assertEqual(fit.status,'failed')
                self.assertIsNone(fit.ec50)
                self.assertIsNone(fit.predictions)
                self.assertTrue(fit.predict(np.array([1.0])).is_error())

    def test_constant_response_r2_undefined(self):
        fit=fit_hill(self.x,np.ones(30))
        self.assertEqual(fit.status,'failed')
        self.assertIsNone(fit.r2)

    def test_replicates_preserved(self):
        fit=fit_hill(np.repeat(self.x,3),np.repeat(self.y,3))
        self.assertEqual(fit.diagnostics['n_observations'],90)
        self.assertEqual(fit.diagnostics['n_concentrations'],30)
        self.assertAlmostEqual(fit.ec50,0.3,places=4)

    def test_noise_and_unclamped_values(self):
        y=self.y+np.random.default_rng(42).normal(0,0.02,30)
        fit=fit_hill(self.x,y)
        self.assertEqual(fit.status,'ok')
        self.assertAlmostEqual(fit.ec50,0.3,delta=0.03)
        self.assertGreater(fit.upper_asymptote,1)

    def test_failed_convergence(self):
        with patch('analysis.fitting.fit_drc',return_value=None):
            self.assertEqual(fit_hill(self.x,self.y).status,'failed')

    def test_degenerate_asymptotes(self):
        with patch('analysis.fitting.fit_drc',return_value=HillCurveLL4([2,1,1,0.3])):
            self.assertEqual(fit_hill(self.x,self.y).status,'failed')

    def test_negative_slope_normalization_preserves_predictions(self):
        model=HillCurveLL4([-2.5,1.2,0.1,0.3])
        with patch('analysis.fitting.fit_drc',return_value=model):
            fit=fit_hill(self.x,self.y)
        np.testing.assert_allclose(fit.predictions,model.fit(self.x))
        self.assertGreater(fit.hill_slope,0)

    def test_extreme_slope_and_out_of_range_ec50_flags(self):
        with patch('analysis.fitting.fit_drc',return_value=HillCurveLL4([25,0,1,1000])):
            fit=fit_hill(self.x,self.y)
        self.assertIn('EC50 outside observed doses',fit.diagnostics['flags'])
        self.assertIn('Extreme slope; inspect identifiability',fit.diagnostics['flags'])
        self.assertLess(fit.r2,0)

    def test_steep_synthetic_observations_remain_honest(self):
        y=HillCurveLL4.fit_fn(self.x,25,0.1,1.2,0.3)
        fit=fit_hill(self.x,y)
        self.assertIn(fit.status,('ok','failed'))
        if fit.status=='ok':
            self.assertTrue(np.all(np.isfinite(fit.predictions)))
            self.assertIn('Transition sampled by fewer than two concentrations',fit.diagnostics['flags'])

    def test_nonfinite_and_nonpositive_parameters_rejected(self):
        for params in [[2,0,1,0],[np.inf,0,1,.3],[2,np.nan,1,.3]]:
            with self.subTest(params=params),patch('analysis.fitting.fit_drc',return_value=HillCurveLL4(params)):
                fit=fit_hill(self.x,self.y)
                self.assertEqual(fit.status,'failed')
                self.assertIsNone(fit.predictions)
                self.assertIsNone(fit.ec50)

    def test_increasing_response_has_no_inhibition_interval(self):
        with patch('analysis.fitting.fit_drc',return_value=HillCurveLL4([2,1.2,.1,.3])):
            fit=fit_hill(self.x,self.y)
        self.assertEqual(fit.status,'ok')
        self.assertIsNone(fit.ec20)
        self.assertIsNone(fit.ec80)
        self.assertIn('Increasing response; inhibition range undefined',fit.diagnostics['flags'])
