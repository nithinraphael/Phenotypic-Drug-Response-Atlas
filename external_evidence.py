import json
from pathlib import Path

import polars as pl

from analysis.provenance import sha256


def source_inventory(path='config/external_sources.json'):
    config = json.loads(Path(path).read_text())
    rows = []
    for entry in config:
        file = Path(entry['path'])
        verified = file.is_file() and sha256(file) == entry['sha256']
        data = pl.read_csv(file, infer_schema=False) if verified else None
        rows.append(dict(Dataset=entry['dataset'], Records=data.height if data is not None else None,
                         Compounds=data[entry['compound_key']].drop_nulls().n_unique() if data is not None else None,
                         Cells=data[entry['cell_key']].drop_nulls().n_unique() if data is not None else None,
                         Release=entry['release'], Provenance=entry['provenance'] if verified else 'Missing file or hash mismatch',
                         Comparison='Not evaluated: identifier crosswalks and metric compatibility unresolved',
                         Source=entry.get('source_url'), SHA256=entry['sha256']))
    return pl.DataFrame(rows)
