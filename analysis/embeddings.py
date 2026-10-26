import json
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from expression import Error, Ok

from analysis.provenance import verify_manifest


PROVENANCE_FIELDS = {'model_id', 'checkpoint_hash', 'preprocessing_hash', 'embedding_dimension',
                     'umap_version', 'random_seed', 'n_neighbors', 'min_dist', 'metric', 'input_normalization', 'n_fields'}


def image_ids(metadata):
    return metadata.with_columns(pl.concat_str(['Image_Metadata_Plate_DAPI', 'Image_FileName_DAPI'], separator='/').alias('image_field_id'))


def join_coordinates(metadata, identifiers, coordinates):
    ids, xy = np.asarray(identifiers), np.asarray(coordinates)
    if ids.ndim != 1 or ids.dtype.kind not in 'US' or xy.ndim != 2 or xy.shape != (len(ids), 2):
        return Error('Invalid UMAP identifiers or coordinate dimensions')
    if xy.dtype.kind not in 'iuf' or not np.all(np.isfinite(xy)):
        return Error('Nonfinite or nonnumeric UMAP coordinates')
    if len(set(ids.tolist())) != len(ids) or metadata['image_field_id'].n_unique() != metadata.height:
        return Error('Duplicate image-field identifiers')
    expected, found = set(metadata['image_field_id']), set(ids.tolist())
    if expected != found:
        return Error(f'UMAP identifier mismatch: {len(expected-found)} missing, {len(found-expected)} extra')
    positions = pl.DataFrame({'image_field_id': ids.tolist(), 'umap_1': xy[:, 0], 'umap_2': xy[:, 1]})
    return Ok(metadata.join(positions, on='image_field_id', how='left', validate='1:1', maintain_order='left'))


def load_projection(metadata, path, inputs, expected_config):
    path = Path(path)
    sidecar = path.with_suffix('.json')
    if not path.is_file() or not sidecar.is_file():
        return Error('UMAP unavailable: cache lacks verified model and preprocessing provenance')
    if not PROVENANCE_FIELDS.issubset(expected_config):
        return Error('Incomplete embedding/UMAP provenance configuration')
    verified = verify_manifest(json.loads(sidecar.read_text()), [*inputs, path], expected_config)
    if verified.is_error():
        return verified
    if not zipfile.is_zipfile(path):
        return Error('Embedding cache is not a valid NPZ archive')
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                return Error('Unsupported NPZ archive entry')
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    return Error('Unsupported array serialization version')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                if dtype.hasobject:
                    return Error('Pickle-backed embedding arrays rejected')
    with np.load(path, allow_pickle=False) as cache:
        if not {'image_field_id', 'embeddings', 'coordinates'}.issubset(cache.files):
            return Error('Unsupported embedding cache schema; positional caches rejected')
        ids, embeddings, xy = cache['image_field_id'], cache['embeddings'], cache['coordinates']
    if embeddings.shape != (len(ids), expected_config['embedding_dimension']) or expected_config['n_fields'] != len(ids):
        return Error('Embedding dimensions or field counts mismatch')
    if embeddings.dtype.kind not in 'iuf' or not np.all(np.isfinite(embeddings)):
        return Error('Invalid embedding values')
    joined = join_coordinates(metadata, ids, xy)
    return joined.map(lambda frame: frame.with_columns(pl.lit(expected_config['model_id']).alias('model_id')))


def configured_projection(metadata, root='data/bbbc021'):
    config_path = Path(root) / 'projection_config.json'
    if not config_path.is_file():
        return Error('UMAP unavailable: legacy cache model/checkpoint and preprocessing are unverified')
    config = json.loads(config_path.read_text())
    required = {'checkpoint_path', 'cache_path', 'analysis_config'}
    if not required.issubset(config):
        return Error('Incomplete projection configuration')
    from analysis.provenance import sha256
    checkpoint = Path(config['checkpoint_path'])
    if not checkpoint.is_file() or sha256(checkpoint) != config['analysis_config'].get('checkpoint_hash'):
        return Error('Embedding checkpoint missing or hash mismatch')
    return load_projection(metadata, config['cache_path'],
                           [Path(root) / 'metadata.csv', Path(root) / 'plate.zip', checkpoint], config['analysis_config'])
