import unittest
import numpy as np
import polars as pl

from analysis.catalog import build_catalog,response_category


def fixture():
    return pl.DataFrame({'broad_id':['b','a','c','d'],'depmap_id':['cell']*4,'screen_id':['s']*4,
                         'name':['B','A','C','D'],'ec50':[1.]*4,'slope':[2.,-2.,4.,30.],
                         'r2':[0.9,0.9,0.7,0.95],'lower_limit':[0.]*4,'upper_limit':[1.]*4})


class CatalogTests(unittest.TestCase):
    def test_category_boundaries(self):
        for slope,r2,expected in [(2,0.9,'Steep response, strong fit'),(-2,0.75,'Steep response, moderate fit'),
                                   (2,0.8999,'Steep response, moderate fit'),(2,0.7499,'Other qualifying fit')]:
            self.assertEqual(response_category(slope,r2),expected)
        for value in [None,np.nan,np.inf]:
            self.assertEqual(response_category(value,0.9),'Fit unavailable')
            self.assertEqual(response_category(2,value),'Fit unavailable')
        self.assertEqual(response_category(2,0.9,False),'Fit failed')

    def test_filter_sort_full_and_shortlist(self):
        result=build_catalog(fixture())
        self.assertTrue(result.is_ok())
        df,audit,excluded=result.ok
        self.assertEqual(df['broad_id'].to_list(),['a','b'])
        self.assertEqual(audit['excluded'].sum(),2)
        self.assertEqual(excluded.height,2)
        self.assertEqual(df['Screening Score'].to_list(),[1.8,1.8])
        self.assertNotIn('p_value',df.columns)
        self.assertIn('Response Category',df.columns)
        self.assertEqual(df.head(1).height,1)
        self.assertEqual(pl.read_csv(__import__('io').StringIO(df.write_csv())).height,2)

    def test_duplicates_rejected(self):
        self.assertTrue(build_catalog(pl.concat([fixture(),fixture().head(1)])).is_error())

    def test_missing_identifier_excluded(self):
        df=fixture().with_columns(pl.when(pl.col('broad_id')=='a').then(None).otherwise(pl.col('depmap_id')).alias('depmap_id'))
        result=build_catalog(df).ok
        self.assertEqual(result[0].height,1)
        self.assertEqual(result[1]['excluded'][0],1)

    def test_missing_slope_never_becomes_zero(self):
        df=fixture().with_columns(pl.lit(None,dtype=pl.Float64).alias('slope'))
        self.assertEqual(build_catalog(df).ok[0].height,0)

    def test_tissue_metadata_preserves_missing_and_rejects_ambiguity(self):
        metadata=pl.DataFrame({'depmap_id':['cell'],'primary_tissue':['test tissue']})
        table=build_catalog(fixture(),metadata).ok[0]
        self.assertEqual(table['primary_tissue'].to_list(),['test tissue','test tissue'])
        missing=pl.DataFrame(schema={'depmap_id':pl.String,'primary_tissue':pl.String})
        self.assertTrue(build_catalog(fixture(),missing).ok[0]['primary_tissue'].is_null().all())
        ambiguous=pl.concat([metadata,metadata.with_columns(pl.lit('other').alias('primary_tissue'))])
        self.assertTrue(build_catalog(fixture(),ambiguous).is_error())
