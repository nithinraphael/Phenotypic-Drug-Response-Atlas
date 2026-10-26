import unittest
import numpy as np
import polars as pl

from analysis.validation import load_configured_validation,match_records,prepare_records


def source():
    return dict(dataset='Synthetic test dataset',version='test',source_file='test.csv',source_sha256='test-hash',source_url='https://example.test')


def records():
    return pl.DataFrame({'source_record_id':['1','2'],'raw_compound_id':[' a ','b'],'raw_cell_line_id':['x','x'],
                         'potency_metric':['EC50','IC50'],'metric_definition':['fitted midpoint','control inhibition'],
                         'potency_value':[1000.,2.],'concentration_unit':['nM','uM'],'quality_status':['pass','pass'],
                         'censored':[False,False]})


def maps():
    return pl.DataFrame({'raw_id':['a','b'],'canonical_id':['A','B']}),pl.DataFrame({'raw_id':['x'],'canonical_id':['X']})


class ValidationTests(unittest.TestCase):
    def test_conversion_identifiers_counts_provenance(self):
        prepared,counts=prepare_records(records(),*maps(),source()).ok
        self.assertEqual(prepared['potency_value_um'].to_list(),[1.,2.])
        self.assertEqual(prepared['compound_id'].to_list(),['A','B'])
        self.assertEqual(counts['raw_records'],2)
        self.assertEqual(counts['excluded_records'],0)
        self.assertEqual(prepared['source_version'].unique().to_list(),['test'])
        self.assertEqual(match_records(prepared,prepared).ok.height,2)

    def test_ambiguous_mapping_excluded(self):
        cm,lm=maps()
        cm=pl.concat([cm,pl.DataFrame({'raw_id':['a'],'canonical_id':['OTHER']})])
        table,counts=prepare_records(records(),cm,lm,source()).ok
        self.assertEqual(counts['ambiguous_records'],1)
        self.assertIsNone(table['potency_value_um'][0])
        self.assertEqual(match_records(table,table).ok.height,1)

    def test_duplicate_source_rows_excluded(self):
        duplicate=pl.concat([records(),records().head(1)])
        table,counts=prepare_records(duplicate,*maps(),source()).ok
        self.assertEqual(counts['excluded_records'],2)
        self.assertEqual(match_records(table,table).ok.height,1)

    def test_invalid_values_and_censoring_excluded(self):
        for value in [None,0.,-1.,np.inf,np.nan]:
            df=records().with_columns(pl.lit(value,dtype=pl.Float64).alias('potency_value'))
            table,counts=prepare_records(df,*maps(),source()).ok
            self.assertEqual(counts['excluded_records'],2)
        df=records().with_columns(pl.lit(True).alias('censored'))
        self.assertEqual(prepare_records(df,*maps(),source()).ok[1]['excluded_records'],2)

    def test_distinct_metrics_do_not_match(self):
        table=prepare_records(records(),*maps(),source()).ok[0]
        other=table.with_columns(pl.lit('different definition').alias('metric_definition'))
        self.assertEqual(match_records(table,other).ok.height,0)

    def test_missing_mapping_excluded(self):
        cm,lm=maps()
        self.assertEqual(prepare_records(records(),cm.head(1),lm,source()).ok[1]['excluded_records'],1)

    def test_legacy_dictionary_values_not_accepted(self):
        legacy=pl.DataFrame({'Compound':['Example'],'PRISM_EC50_uM':[1.0],'GDSC2_IC50_uM':[2.0]})
        self.assertTrue(prepare_records(legacy,*maps(),source()).is_error())
        self.assertTrue(load_configured_validation('/tmp/no-atlas-validation-config').is_error())

    def test_missing_provenance_rejected(self):
        self.assertTrue(prepare_records(records(),*maps(),{}).is_error())

    def test_conversion_overflow_excluded(self):
        df=records().with_columns(pl.lit(1e308).alias('potency_value'),pl.lit('M').alias('concentration_unit'))
        table,counts=prepare_records(df,*maps(),source()).ok
        self.assertEqual(counts['excluded_records'],2)
        self.assertTrue(table['potency_value_um'].is_null().all())

    def test_calculated_correlation_and_exclusions(self):
        from analysis.validation import compare_potencies
        matched=pl.DataFrame({'source_dataset':['A']*4,'source_dataset_external':['B']*4,'potency_metric':['EC50']*4,
                              'metric_definition':['test definition']*4,'potency_value_um':[1.,2.,3.,0.],
                              'potency_value_um_external':[2.,4.,6.,8.]})
        result=compare_potencies(matched).ok.row(0,named=True)
        self.assertAlmostEqual(result['coefficient'],1.)
        self.assertEqual(result['eligible_pairs'],3)
        self.assertEqual(result['excluded_pairs'],1)
        result=compare_potencies(matched.head(2)).ok.row(0,named=True)
        self.assertIsNone(result['coefficient'])
