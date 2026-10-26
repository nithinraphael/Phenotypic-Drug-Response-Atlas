import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
from expression import Error,Ok
from thunor.curve_fit import HillCurveLL4

from analysis.benchmark import aggregate_predictions,concentration_splits,evaluate_curve,rmse


def observations():
    x=np.repeat(np.geomspace(.001,100,9),2)
    return pl.DataFrame({'column_name':[str(i) for i in range(len(x))], 'screen_id':['s']*len(x),
                         'broad_id':['d']*len(x),'dose':x,'viability':HillCurveLL4.fit_fn(x,2,0.1,1.2,0.3)})


class BenchmarkTests(unittest.TestCase):
    def test_dose_groups_disjoint_and_reproducible(self):
        x=observations()['dose'].to_numpy()
        splits=concentration_splits(x).ok
        again=concentration_splits(x).ok
        for (train,test),(train2,test2) in zip(splits,again):
            self.assertFalse(set(x[train]) & set(x[test]))
            np.testing.assert_array_equal(train,train2)
            np.testing.assert_array_equal(test,test2)
        self.assertEqual(sum(len(test) for train,test in splits),len(x))

    def test_rmse_exact_and_invalid(self):
        self.assertAlmostEqual(rmse([1.,2.],[1.,4.]).ok,np.sqrt(2))
        for y,p in [([],[]),([1],[1,2]),([np.inf],[0]),([1],[np.nan]),(['a'],['b'])]:
            self.assertTrue(rmse(y,p).is_error())

    def test_training_unaffected_by_held_out_responses(self):
        data=observations()
        held_out=concentration_splits(data['dose'].to_numpy()).ok[0][1]
        changed=data.with_row_index('i').with_columns(pl.when(pl.col('i').is_in(held_out.tolist())).then(pl.col('viability')+10).otherwise(pl.col('viability')).alias('viability')).drop('i')
        a=evaluate_curve(data,'curve',models=('Hill LL4',)).ok.filter(pl.col('fold')==0)
        b=evaluate_curve(changed,'curve',models=('Hill LL4',)).ok.filter(pl.col('fold')==0)
        np.testing.assert_allclose(a['predicted'].to_numpy(),b['predicted'].to_numpy(),rtol=0,atol=0)
        self.assertFalse(a['observed'].equals(b['observed']))

    def test_failed_predictions_counted_and_common_metrics(self):
        with patch('analysis.benchmark.isolated_predict',return_value=Error('Failed optimization')):
            predictions=evaluate_curve(observations(),'curve').ok
        summary=aggregate_predictions(predictions)
        segmented=summary.filter(pl.col('model')=='Segmented').row(0,named=True)
        self.assertEqual(segmented['failed_predictions'],18)
        self.assertEqual(segmented['failed_folds'],3)
        self.assertIsNone(segmented['rmse'])
        self.assertTrue(summary['common_rmse'].is_null().all())
        self.assertEqual(summary['common_observations'].sum(),0)

    def test_comparison_uses_identical_observations(self):
        predictions=evaluate_curve(observations(),'curve',models=('Hill LL4','Mean baseline')).ok
        summary=aggregate_predictions(predictions)
        self.assertEqual(summary['held_out_observations'].to_list(),[18,18])
        self.assertEqual(summary['successful_predictions'].to_list(),[18,18])
        self.assertEqual(summary['common_observations'].to_list(),[18,18])
        for row in summary.iter_rows(named=True):
            self.assertAlmostEqual(row['rmse'],row['common_rmse'])

    def test_insufficient_training_fold_remains_missing(self):
        data=observations().head(8)
        prediction=evaluate_curve(data,'curve',models=('Hill LL4',),n_splits=2).ok
        self.assertTrue(prediction['predicted'].is_null().all())
        self.assertTrue(prediction['failure'].is_not_null().all())

    def test_multiple_screens_rejected(self):
        data=observations().with_columns(pl.when(pl.col('column_name')=='0').then(pl.lit('other')).otherwise(pl.col('screen_id')).alias('screen_id'))
        self.assertTrue(evaluate_curve(data,'curve').is_error())
