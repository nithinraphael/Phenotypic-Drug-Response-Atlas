import tempfile
import unittest
from pathlib import Path

import polars as pl

from analysis.benchmark import aggregate_predictions,evaluate_curve
from analysis.catalog import build_catalog
from analysis.provenance import manifest
from analysis.reporting import export_report
from test_benchmark import observations
from test_catalog import fixture


class ReportTests(unittest.TestCase):
    def test_consistent_outputs_provenance_and_no_overwrite(self):
        catalog,audit,excluded=build_catalog(fixture()).ok
        predictions=evaluate_curve(observations(),'test',models=('Mean baseline',)).ok
        metrics=aggregate_predictions(predictions)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';source.write_text('synthetic test input')
            m=manifest('test',[source],{'seed':42})
            output=Path(directory)/'report'
            self.assertTrue(export_report(output,catalog,audit,predictions,metrics,m).is_ok())
            self.assertTrue(pl.read_csv(output/'eligible_catalog.csv').equals(catalog))
            self.assertTrue((output/'manifest.json').exists())
            self.assertTrue(export_report(output,catalog,audit,predictions,metrics,m).is_error())
            self.assertTrue(export_report(Path(directory)/'bad',catalog,audit,predictions,metrics.with_columns(pl.lit(0.).alias('rmse')),m).is_error())
            source.write_text('changed')
            self.assertTrue(export_report(Path(directory)/'stale',catalog,audit,predictions,metrics,m).is_error())
