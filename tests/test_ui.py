import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from analysis.fitting import FitResult,fit_hill
from analysis.microscopy import load_microscopy_rgb,select_control
from analysis.models import predict_tabpfn,tabpfn_config
from components import build_drc_plot,build_generalization_plot,build_umap_plot,build_validation_plot


class UITests(unittest.TestCase):
    def test_failed_fit_plots_only_observations_and_cursor(self):
        figure=build_drc_plot(np.array([.1,1]),np.array([1,.2]),np.array([.1,1]),FitResult('failed','test'),active_dose=.1,drug='A',cell_line='X')
        self.assertEqual(len(figure.data),1)
        self.assertIn('A · X',figure.layout.title.text)
        self.assertEqual(len(figure.layout.shapes),1)
        self.assertEqual(figure.layout.shapes[0].x0,.1)

    def test_cursor_is_not_model_dependency(self):
        tree=ast.parse(Path('notebook.py').read_text())
        prediction_cells=[node for node in tree.body if isinstance(node,ast.FunctionDef) and any(isinstance(x,ast.Name) and x.id=='predict_selected' for x in ast.walk(node))]
        self.assertEqual(len(prediction_cells),1)
        self.assertNotIn('concentration_cursor',[arg.arg for arg in prediction_cells[0].args.args])
        self.assertIn('measurements',[arg.arg for arg in prediction_cells[0].args.args])

    def test_umap_highlights_all_matching_fields(self):
        df=pl.DataFrame({'Image_Metadata_Compound':['A','A','A','B'],'Image_Metadata_Concentration':[1.,1.,2.,1.],
                         'umap_1':[0.,1.,2.,3.],'umap_2':[0.,1.,2.,3.]})
        figure=build_umap_plot(df,'A',1.1)
        self.assertEqual(len(figure.data[-1].x),2)
        self.assertIn('requested 1.1',figure.data[-1].name)
        self.assertEqual(figure.data[-2].name,'Dose-ordered condition centroids')

    def test_missing_model_and_image_are_explicit(self):
        self.assertTrue(tabpfn_config('/tmp/no-atlas-model-config').is_error())
        self.assertTrue(predict_tabpfn([1],[1],[1],None).is_error())
        self.assertTrue(load_microscopy_rgb('/tmp/no-atlas-images.zip',{}).is_error())

    def test_control_matches_plate_and_replicate(self):
        df=pl.DataFrame({'Image_Metadata_Compound':['DMSO']*3,'Image_Metadata_Plate_DAPI':['other','p','p'],
                         'Replicate':[1,2,1],'image_field_id':['a','b','c']})
        treated={'Image_Metadata_Plate_DAPI':'p','Replicate':1}
        self.assertEqual(select_control(df,treated).ok['image_field_id'],'c')
        self.assertTrue(select_control(df,dict(treated,Replicate=3)).is_error())

    def test_public_outputs_contain_no_invalid_inference(self):
        for filename in ['notebook.py','dev_notebook.py','components.py']:
            text=Path(filename).read_text()
            for invalid in ['se_slopes','false_discovery_control','norm.cdf','ci_band','Evidence Tier','fdr_q']:
                self.assertNotIn(invalid,text)
        tree=ast.parse(Path('notebook.py').read_text())
        self.assertEqual(sum(isinstance(n,ast.Constant) and n.value in ('Overview','Dose–Response','Cell Imaging','Validation','Methods') for n in ast.walk(tree)),5)

    def test_plot_values_equal_input_tables(self):
        summary=pl.DataFrame({'model':['test'],'common_rmse':[0.123]})
        self.assertEqual(list(build_generalization_plot(summary).data[0].y),[0.123])
        matched=pl.DataFrame({'compound_id':['test'],'potency_metric':['EC50'],'potency_value_um':[1.],'potency_value_um_external':[2.]})
        figure=build_validation_plot(matched,dict(reference='A',external='B'))
        self.assertEqual(list(figure.data[0].x),[1.])
        self.assertEqual(list(figure.data[0].y),[2.])
