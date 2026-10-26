import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import tifffile
from expression import Error, Ok

from analysis.embeddings import image_ids

CHANNELS = ('DAPI', 'Tubulin', 'Actin')


def load_metadata(path='data/bbbc021/metadata.csv', plate='Week1_22123'):
    if not Path(path).is_file():
        return Error(f'Microscopy metadata missing: {path}')
    df = image_ids(pl.read_csv(path).filter(pl.col('Image_Metadata_Plate_DAPI') == plate))
    if df['image_field_id'].n_unique() != df.height:
        return Error('Duplicate microscopy field identifiers')
    return Ok(df)


def select_control(metadata, treated):
    matches = metadata.filter((pl.col('Image_Metadata_Compound') == 'DMSO')
                              & (pl.col('Image_Metadata_Plate_DAPI') == treated['Image_Metadata_Plate_DAPI'])
                              & (pl.col('Replicate') == treated['Replicate'])).sort('image_field_id')
    if matches.is_empty():
        return Error('No vehicle control in the same plate and replicate group')
    return Ok(matches.row(0, named=True))


def load_microscopy_rgb(path, row):
    if not Path(path).is_file():
        return Error(f'Microscopy archive missing: {path}')
    entries = [f"{row['Image_Metadata_Plate_DAPI']}/{row[f'Image_FileName_{c}']}" for c in CHANNELS]
    with zipfile.ZipFile(path) as archive:
        missing = set(entries) - set(archive.namelist())
        if missing:
            return Error('Image files absent from archive: ' + ', '.join(sorted(missing)))
        channels = []
        for name in entries:
            with archive.open(name) as stream:
                channels.append(tifffile.imread(stream))
    if len({ch.shape for ch in channels}) != 1 or any(ch.ndim != 2 for ch in channels):
        return Error('Incompatible image channel shapes')
    def normalize(channel):
        low, high = np.percentile(channel, [1, 99.5])
        return np.zeros(channel.shape, dtype=np.uint8) if high == low else (np.clip((channel.astype(float)-low)/(high-low), 0, 1)*255).astype(np.uint8)
    return Ok(np.stack([normalize(channels[2]), normalize(channels[1]), normalize(channels[0])], axis=-1))
