import unittest
import polars as pl

from analysis.prism import eligible_observations,join_measurements,load_prism


class PrismTests(unittest.TestCase):
    def setUp(self):
        self.tr=pl.DataFrame({'column_name':['a','b','c'],'dose':[0.1,0.1,1.],
                              'screen_id':['s']*3,'well':['A01','B01','A02'],'broad_id':['drug']*3})
        self.responses=pl.DataFrame({'column_name':['a','b','c'],'response':[-1.,0.,1.]})

    def test_log2_transformation_and_replicates(self):
        joined=join_measurements(self.tr,self.responses).ok
        self.assertEqual(joined['viability'].to_list(),[0.5,1.,2.])
        self.assertEqual(joined.height,3)
        self.assertEqual(joined['well'].to_list(),['A01','B01','A02'])
        self.assertEqual(joined['screen_id'].unique().to_list(),['s'])

    def test_duplicate_join_rejected(self):
        self.assertTrue(join_measurements(pl.concat([self.tr,self.tr.head(1)]),self.responses).is_error())
        self.assertTrue(join_measurements(self.tr,pl.concat([self.responses,self.responses.head(1)])).is_error())

    def test_unknown_treatment_rejected(self):
        self.assertTrue(join_measurements(self.tr.head(2),self.responses).is_error())

    def test_invalid_observations_audited(self):
        joined=join_measurements(self.tr,self.responses.with_columns(pl.when(pl.col('column_name')=='a').then(None).otherwise(pl.col('response')).alias('response'))).ok
        valid,excluded=eligible_observations(joined)
        self.assertEqual(valid.height,2)
        self.assertEqual(excluded.height,1)
        self.assertIn('exclusion_reason',excluded.columns)
        self.assertEqual(joined.height,3)

    def test_missing_files(self):
        self.assertTrue(load_prism('/tmp/nonexistent-atlas-input').is_error())
