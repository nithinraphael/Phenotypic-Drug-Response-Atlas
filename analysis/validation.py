import json
import math
from pathlib import Path

import polars as pl
from expression import Error, Ok

from analysis.provenance import sha256

UNAVAILABLE = 'External validation unavailable: source measurements have not been configured or verified.'
REQUIRED = {'source_record_id', 'raw_compound_id', 'raw_cell_line_id', 'potency_metric', 'metric_definition',
            'potency_value', 'concentration_unit', 'quality_status', 'censored'}


def normalize_identifier(value):
    return value.strip() if isinstance(value, str) else value


def prepare_records(records, compound_map, cell_map, source):
    if not REQUIRED.issubset(records.columns):
        return Error('Missing validation source columns: ' + ', '.join(sorted(REQUIRED-set(records.columns))))
    if not all(source.get(key) for key in ('dataset', 'version', 'source_file', 'source_sha256', 'source_url')):
        return Error('Missing external source provenance')
    for mapping in (compound_map, cell_map):
        if not {'raw_id', 'canonical_id'}.issubset(mapping.columns):
            return Error('Mapping requires raw_id and canonical_id')
    df = records.with_columns(pl.col('raw_compound_id', 'raw_cell_line_id', 'source_record_id').str.strip_chars())
    maps = []
    for mapping in (compound_map, cell_map):
        maps.append(mapping.with_columns(pl.col('raw_id', 'canonical_id').str.strip_chars()))
    duplicate_records = df.group_by('source_record_id').len().filter(pl.col('len') > 1)['source_record_id'].to_list()
    compound_map, cell_map = maps
    ambiguous_compounds = compound_map.group_by('raw_id').agg(pl.col('canonical_id').n_unique().alias('n')).filter(pl.col('n') != 1)['raw_id'].to_list()
    ambiguous_cells = cell_map.group_by('raw_id').agg(pl.col('canonical_id').n_unique().alias('n')).filter(pl.col('n') != 1)['raw_id'].to_list()
    cm = compound_map.filter(~pl.col('raw_id').is_in(ambiguous_compounds)).unique().rename({'raw_id':'raw_compound_id','canonical_id':'compound_id'})
    lm = cell_map.filter(~pl.col('raw_id').is_in(ambiguous_cells)).unique().rename({'raw_id':'raw_cell_line_id','canonical_id':'cell_line_id'})
    df = df.join(cm, on='raw_compound_id', how='left', validate='m:1').join(lm, on='raw_cell_line_id', how='left', validate='m:1')
    factors = {'M': 1e6, 'mM': 1e3, 'uM': 1.0, 'μM': 1.0, 'nM': 1e-3, 'pM': 1e-6}
    rows = []
    for row in df.iter_rows(named=True):
        reasons = []
        if row['source_record_id'] in duplicate_records:
            reasons.append('duplicate source record')
        if not row['source_record_id']:
            reasons.append('missing source record identifier')
        ambiguous = row['raw_compound_id'] in ambiguous_compounds or row['raw_cell_line_id'] in ambiguous_cells
        if ambiguous:
            reasons.append('ambiguous identifier mapping')
        if not row['compound_id'] or not row['cell_line_id']:
            reasons.append('unmapped identifier')
        value, unit = row['potency_value'], row['concentration_unit']
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            reasons.append('invalid potency')
        if unit not in factors:
            reasons.append('unsupported concentration unit')
        converted = value*factors[unit] if isinstance(value, (int, float)) and unit in factors else None
        if converted is not None and not math.isfinite(converted):
            reasons.append('nonfinite converted potency')
        if row['quality_status'] != 'pass' or row['censored'] is not False or not row['metric_definition'] or not row['potency_metric']:
            reasons.append('quality, censoring or metric definition unresolved')
        row.update(source_dataset=source['dataset'], source_version=source['version'], source_file=source['source_file'],
                   source_sha256=source['source_sha256'], source_url=source['source_url'],
                   potency_value_um=converted if not reasons else None,
                   transformation=f'{unit} to uM' if unit in factors else None,
                   matching_status='ambiguous' if ambiguous else ('unmatched' if not row['compound_id'] or not row['cell_line_id'] else 'mapped'),
                   exclusion_reason='; '.join(reasons) or None)
        rows.append(row)
    prepared = pl.DataFrame(rows, infer_schema_length=None) if rows else df.with_columns(pl.lit(None, dtype=pl.Float64).alias('potency_value_um'), pl.lit(None,dtype=pl.String).alias('exclusion_reason'))
    counts = {'source_dataset': source['dataset'], 'raw_records': records.height, 'unique_compounds': records['raw_compound_id'].drop_nulls().n_unique(),
              'unique_cell_lines': records['raw_cell_line_id'].drop_nulls().n_unique(),
              'valid_potency_measurements': prepared.filter(pl.col('potency_value_um').is_not_null()).height,
              'records_with_identifiers': prepared.filter(pl.col('compound_id').is_not_null() & pl.col('cell_line_id').is_not_null()).height,
              'ambiguous_records': sum(r.get('matching_status') == 'ambiguous' for r in rows),
              'excluded_records': sum(r.get('exclusion_reason') is not None for r in rows)}
    return Ok((prepared, counts))


def match_records(reference, external):
    keys = ['compound_id', 'cell_line_id', 'potency_metric', 'metric_definition']
    a = reference.filter(pl.col('exclusion_reason').is_null())
    b = external.filter(pl.col('exclusion_reason').is_null())
    if a.select(keys).is_duplicated().any() or b.select(keys).is_duplicated().any():
        return Error('Ambiguous repeated conditions; specify assay grouping before comparison')
    return Ok(a.join(b, on=keys, suffix='_external', how='inner', validate='1:1'))


def load_configured_validation(path='config/validation.json'):
    if not Path(path).is_file():
        return Error(UNAVAILABLE)
    config = json.loads(Path(path).read_text())
    if not {'reference', 'external'}.issubset(config):
        return Error('Validation configuration requires reference and external sources')
    tables, counts = [], []
    for name in ('reference', 'external'):
        entry = config[name]
        if not {'source', 'records', 'compound_mapping', 'cell_mapping', 'file_hashes'}.issubset(entry):
            return Error('Incomplete validation input configuration')
        paths = [entry[k] for k in ('records', 'compound_mapping', 'cell_mapping')]
        source_path = entry['source'].get('source_file')
        if not source_path:
            return Error('Missing original external source file')
        for file in [source_path, *paths]:
            if not Path(file).is_file() or sha256(file) != entry['file_hashes'].get(file):
                return Error(f'Validation input missing or hash mismatch: {file}')
        if sha256(source_path) != entry['source'].get('source_sha256'):
            return Error('Source provenance hash mismatch')
        result = prepare_records(pl.read_csv(entry['records']), pl.read_csv(entry['compound_mapping']), pl.read_csv(entry['cell_mapping']), entry['source'])
        if result.is_error():
            return result
        table, count = result.ok
        tables.append(table)
        counts.append(count)
    matched = match_records(*tables)
    if matched.is_error():
        return matched
    for count in counts:
        count['matched_records'] = matched.ok.height
    return Ok((matched.ok, tables, counts, config))


def compare_potencies(matched):
    import numpy as np
    from scipy.stats import spearmanr
    keys = ['source_dataset', 'source_dataset_external', 'potency_metric', 'metric_definition']
    required = set(keys + ['potency_value_um', 'potency_value_um_external'])
    if not required.issubset(matched.columns):
        return Error('Matched comparison lacks dataset or potency provenance')
    rows = []
    for group in matched.partition_by(keys, maintain_order=True):
        values = group.select('potency_value_um', 'potency_value_um_external').to_numpy()
        valid = np.all(np.isfinite(values) & (values > 0), axis=1)
        x, y = np.log10(values[valid]).T
        coefficient = float(spearmanr(x, y).statistic) if len(x) >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1 else None
        rows.append({**group.select(keys).row(0, named=True), 'eligible_pairs': len(x),
                     'excluded_pairs': len(values)-len(x), 'correlation_type': 'Spearman rank',
                     'coefficient': coefficient, 'status': 'calculated' if coefficient is not None else 'Not evaluated: insufficient or constant pairs'})
    schema = {**{k:pl.String for k in keys}, 'eligible_pairs':pl.Int64,'excluded_pairs':pl.Int64,
              'correlation_type':pl.String,'coefficient':pl.Float64,'status':pl.String}
    return Ok(pl.DataFrame(rows,schema=schema))
