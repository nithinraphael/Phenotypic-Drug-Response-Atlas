import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from analysis.embeddings import join_coordinates,load_projection
from analysis.provenance import manifest,write_json


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.metadata=pl.DataFrame({'image_field_id':['plate/a','plate/b']})

    def test_shuffled_ids_join_correctly(self):
        result=join_coordinates(self.metadata,np.array(['plate/b','plate/a']),np.array([[20.,21.],[10.,11.]])).ok
        self.assertEqual(result['umap_1'].to_list(),[10.,20.])

    def test_duplicate_missing_extra_nonfinite_positional_rejected(self):
        for ids,xy in [(np.array(['plate/a','plate/a']),np.ones((2,2))),
                       (np.array(['plate/a','other']),np.ones((2,2))),
                       (np.array(['plate/a']),np.ones((1,2))),
                       (np.array(['plate/a','plate/b']),np.full((2,2),np.nan)),
                       (np.array([0,1]),np.ones((2,2)))]:
            self.assertTrue(join_coordinates(self.metadata,ids,xy).is_error())

    def test_cache_provenance_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            raw=root/'metadata.csv';raw.write_text('source')
            cache=root/'projection.npz'
            np.savez(cache,image_field_id=np.array(['plate/a','plate/b']),embeddings=np.ones((2,3)),coordinates=np.ones((2,2)))
            config=dict(model_id='synthetic test model',checkpoint_hash='test',preprocessing_hash='test',embedding_dimension=3,
                        umap_version='test',random_seed=42,n_neighbors=2,min_dist=0.1,metric='euclidean',input_normalization='test',n_fields=2)
            write_json(cache.with_suffix('.json'),manifest('test',[raw,cache],config))
            self.assertTrue(load_projection(self.metadata,cache,[raw],config).is_ok())
            self.assertTrue(load_projection(self.metadata,cache,[raw],dict(config,checkpoint_hash='changed')).is_error())
            self.assertTrue(load_projection(self.metadata,cache,[raw],dict(config,preprocessing_hash='changed')).is_error())
            raw.write_text('modified')
            self.assertTrue(load_projection(self.metadata,cache,[raw],config).is_error())

    def test_legacy_cache_without_sidecar_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.npz';np.savez(path,keys=['a','b'],u=np.ones((2,2)))
            self.assertTrue(load_projection(self.metadata,path,[],{}).is_error())

    def test_pickle_backed_cache_rejected_without_unpickling(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)/'projection.npz'
            np.savez(cache,image_field_id=np.array(['plate/a','plate/b'],dtype=object),embeddings=np.ones((2,3)),coordinates=np.ones((2,2)))
            config=dict(model_id='synthetic test model',checkpoint_hash='test',preprocessing_hash='test',embedding_dimension=3,
                        umap_version='test',random_seed=42,n_neighbors=2,min_dist=0.1,metric='euclidean',input_normalization='test',n_fields=2)
            write_json(cache.with_suffix('.json'),manifest('test',[cache],config))
            result=load_projection(self.metadata,cache,[],config)
            self.assertTrue(result.is_error())
            self.assertIn('Pickle-backed',result.error)
