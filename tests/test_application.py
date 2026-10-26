import unittest
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
from expression import Error

from notebook import app


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs,cls.definitions=app.run()

    def test_five_views_and_real_data(self):
        d=self.definitions
        self.assertEqual(len(self.outputs),22)
        for name in ['overview_view','drc_view','imaging_view','validation_view','methods_view']:
            self.assertIsNotNone(d[name])
        self.assertEqual(d['counts']['raw_curve_records'],701004)
        self.assertEqual(d['counts']['cell_line_ids'],480)
        self.assertEqual(d['ranked_result'].ok[0].height,51763)
        self.assertEqual(d['image_metadata'].ok.height,240)
        self.assertEqual(d['measurement_result'].ok.height,d['measurements'].height+d['measurement_exclusions'].height)
        self.assertTrue(d['external_result'].is_error())
        self.assertTrue(d['projection'].is_error())
        self.assertTrue(d['tab_prediction'].is_error())

    def test_compound_change_rebuilds_dependent_selection(self):
        d=self.definitions
        rows=d['compound_rows']
        chosen=rows.filter(pl.col('name')=='floxuridine').row(0,named=True)
        label=f"{chosen['name']} · {chosen['broad_id']}"
        replacement=dict(compound_rows=rows,compound_options={label:chosen['broad_id']},default_compound=label,
                         compound_select=SimpleNamespace(value=chosen['broad_id'],selected_key=label))
        outputs,changed=app.run(defs=replacement)
        self.assertTrue(changed['measurement_result'].is_ok())
        self.assertEqual(changed['measurement_result'].ok['broad_id'].unique().to_list(),[chosen['broad_id']])
        self.assertIn(changed['cell_select'].value,set(changed['tested']['depmap_id']))
        self.assertIn('floxuridine',changed['dose_figure'].layout.title.text)

    def test_missing_prism_preserves_other_views(self):
        with patch('analysis.prism.load_prism',return_value=Error('Missing PRISM files: fixture.csv')):
            from analysis.provenance import sha256
            with patch('analysis.provenance.sha256',return_value='different test cache identity'):
                outputs,changed=app.run()
        self.assertTrue(changed['prism_result'].is_error())
        self.assertIsNone(changed['measurements'])
        self.assertIn('fixture.csv',changed['overview_view'].text)
        self.assertIsNotNone(changed['imaging_view'])
        self.assertIsNotNone(changed['methods_view'])

    def test_development_notebook_runs_without_writing_reports(self):
        from pathlib import Path
        from dev_notebook import app as diagnostics
        before=set(Path('reports').rglob('*'))
        outputs,definitions=diagnostics.run()
        self.assertEqual(len(outputs),4)
        self.assertIn('Not evaluated',definitions['analysis_output'].text)
        self.assertEqual(before,set(Path('reports').rglob('*')))

    def test_cell_line_change_reloads_current_observations(self):
        d=self.definitions
        label=next(k for k,v in d['cell_options'].items() if v!=d['cell_select'].value)
        replacement={name:d[name] for name in ['tested','names','cell_options','default_cell']}
        replacement['cell_select']=SimpleNamespace(value=d['cell_options'][label],selected_key=label)
        outputs,changed=app.run(defs=replacement)
        self.assertTrue(changed['measurement_result'].is_ok())
        self.assertEqual(changed['measurement_result'].ok[''].unique().to_list(),[d['cell_options'][label]])
        self.assertIn(label,changed['dose_figure'].layout.title.text)
