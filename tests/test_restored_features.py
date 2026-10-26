import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.microscopy import load_metadata
from analysis.provenance import sha256
from external_evidence import source_inventory
from image_projection import load_projection
from public_ui import morphology_figure,selected_image_field


class RestoredFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = load_metadata().ok
        cls.projection = load_projection(cls.metadata)

    def test_real_projection_preserves_field_identity_and_model(self):
        self.assertTrue(self.projection.is_ok())
        frame = self.projection.ok
        self.assertEqual(frame.height,240)
        self.assertEqual(frame['image_field_id'].to_list(),self.metadata['image_field_id'].to_list())
        self.assertTrue(np.isfinite(frame.select('umap_1','umap_2').to_numpy()).all())
        self.assertEqual(frame['model_id'].unique().to_list(),['DINO ViT-S/8 ImageNet baseline'])

    def test_projection_verifies_additional_pipeline_and_model_inputs(self):
        from analysis.provenance import verify_manifest
        import image_projection
        original = verify_manifest
        checked = []
        def record(metadata, inputs, config):
            checked.extend(inputs)
            return original(metadata, inputs, config)
        with patch.object(image_projection.provenance,'verify_manifest',side_effect=record):
            self.assertTrue(load_projection(self.metadata).is_ok())
        self.assertIn('image_projection.py',checked)
        self.assertIn('data/bbbc021/verified_model/vision_transformer.py',checked)

    def test_map_point_ids_match_metadata_in_both_color_modes(self):
        frame = self.projection.ok
        for mode in ('Concentration','Compound'):
            fig = morphology_figure(frame,'taxol',0.1,mode)
            for trace in fig.data:
                if trace.customdata is None:
                    self.assertEqual(trace.name,'Dose-ordered condition centroids')
                    continue
                self.assertEqual(len(trace.customdata),len(trace.x))
                positions = dict(zip(frame['image_field_id'],frame.select('umap_1','umap_2').rows()))
                for x,y,info in zip(trace.x,trace.y,trace.customdata):
                    self.assertEqual((x,y),positions[info[0]])

    def test_source_counts_and_authentication_are_distinct_from_matching(self):
        rows = source_inventory().to_dicts()
        self.assertEqual(rows[0]['Records'],135242)
        self.assertEqual(rows[0]['Compounds'],198)
        self.assertEqual(rows[0]['Cells'],809)
        self.assertIn('authenticated',rows[0]['Provenance'])
        self.assertEqual(rows[1]['Records'],16729)
        self.assertIn('unverified',rows[1]['Provenance'])
        self.assertTrue(all(row['Comparison'].startswith('Not evaluated') for row in rows))

    def test_native_selection_payload_resolves_server_side_field_ids(self):
        for mode in ('Compound','Concentration'):
            fig=morphology_figure(self.projection.ok,'taxol',0.1,mode)
            for curve,trace in enumerate(fig.data):
                point=[{'curveNumber':curve,'pointIndex':0}]
                self.assertEqual(selected_image_field(point,fig),trace.customdata[0][0] if trace.customdata is not None else None)
            for points in ([],[{}],[{'curveNumber':999,'pointIndex':0}],[{'curveNumber':0,'pointIndex':-1}]):
                self.assertIsNone(selected_image_field(points,fig))

    def test_changed_external_file_never_reuses_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)/'source.csv'
            raw.write_text('drug,cell\nsynthetic_A,synthetic_X\n')
            config = Path(directory)/'config.json'
            config.write_text(json.dumps([dict(dataset='Synthetic test fixture',path=str(raw),sha256=sha256(raw),compound_key='drug',cell_key='cell',release='test-only',provenance='test-only')]))
            self.assertEqual(source_inventory(config)['Records'][0],1)
            raw.write_text('changed')
            row = source_inventory(config).row(0,named=True)
            self.assertIsNone(row['Records'])
            self.assertIn('hash mismatch',row['Provenance'])

    def test_missing_projection_does_not_load_legacy_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            result = load_projection(self.metadata,directory)
            self.assertTrue(result.is_error())
            self.assertIn('No verified projection',result.error)
