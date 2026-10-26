import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from analysis.fitting import fit_hill
from components import build_drc_plot
from public_ui import audited_benchmark,candidate_figure,dose_figure,metrics,unavailable


class PublicInterfaceTests(unittest.TestCase):
    def test_visual_styling_preserves_numerical_results(self):
        x=np.geomspace(.001,100,20)
        y=.1+1/(1+(x/.3)**2)
        fit=fit_hill(x,y)
        original=build_drc_plot(x,y,x,fit,active_dose=.3,drug='Test',cell_line='Test cell')
        styled=dose_figure(original)
        self.assertEqual(len(original.data),len(styled.data))
        for before,after in zip(original.data,styled.data):
            np.testing.assert_array_equal(before.x,after.x)
            np.testing.assert_array_equal(before.y,after.y)
        for before,after in zip(original.layout.shapes,styled.layout.shapes):
            self.assertEqual(before.x0,after.x0)
            self.assertEqual(before.x1,after.x1)
        self.assertEqual(original.data[0].name,'Measured relative viability')
        self.assertEqual(styled.data[0].name,'Measured viability')
        self.assertEqual(len(styled.layout.annotations),1)
        self.assertEqual(styled.layout.template.layout.xaxis.type,'log')
        self.assertEqual(styled.layout.template.layout.xaxis.title.text,'Concentration (μM)')
        self.assertEqual(styled.layout.template.layout.yaxis.title.text,'Relative viability')

    def test_discovery_chart_uses_source_fit_values(self):
        data=pl.DataFrame({'ec50':[.1,1.],'slope':[-2.,3.],'r2':[.9,.8],
                           'name':['A','B'],'ccle_name':['X','Y'],'Screening Score':[1.8,2.4]})
        figure=candidate_figure(data)
        self.assertEqual(list(figure.data[0].x),[.1,1.])
        self.assertEqual(list(figure.data[0].y),[2.,3.])

    def test_saved_benchmark_requires_actual_artifacts(self):
        self.assertTrue(audited_benchmark('/tmp/no-atlas-benchmark').is_error())
        self.assertTrue(audited_benchmark().is_ok())

    def test_public_html_escapes_source_text(self):
        self.assertNotIn('<script>',unavailable('Example','<script>').text)
        self.assertIn('&lt;script&gt;',metrics([('Label','<script>')]).text)
