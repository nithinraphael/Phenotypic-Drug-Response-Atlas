from pathlib import Path

import numpy as np
import polars as pl
from expression import Error, Ok

FILES = ('cell_lines', 'treatments', 'curves', 'responses')


def load_prism(root='data/prism'):
    missing = [str(Path(root) / f'{name}.csv') for name in FILES if not (Path(root) / f'{name}.csv').is_file()]
    if missing:
        return Error('Missing PRISM files: ' + ', '.join(missing))
    return Ok({name: pl.scan_csv(Path(root) / f'{name}.csv', null_values=['NA', '']) for name in FILES})


def join_measurements(treatments, melted):
    if treatments['column_name'].n_unique() != treatments.height or melted['column_name'].n_unique() != melted.height:
        return Error('Duplicate treatment or response identifiers; join would multiply observations')
    if not set(melted['column_name']).issubset(set(treatments['column_name'])):
        return Error('Response column missing treatment metadata')
    joined = melted.join(treatments, on='column_name', how='left', validate='1:1')
    assert joined.height == melted.height
    joined = joined.with_columns(pl.col('response').alias('response_raw'), pl.col('response').cast(pl.Float64, strict=False))
    return Ok(joined.with_columns(pl.lit(2.0).pow(pl.col('response')).alias('viability')).sort('dose'))


def fetch_measurements(prism, broad_id, cell_line_id, screen_id):
    tr = prism['treatments'].filter((pl.col('broad_id') == broad_id) & (pl.col('screen_id') == screen_id)
                                  & (pl.col('perturbation_type') == 'experimental_treatment')).collect()
    columns = prism['responses'].collect_schema().names()
    selected = [c for c in tr['column_name'] if c in columns[1:]]
    if not selected:
        return Error('No response columns for selected compound and screen')
    row = prism['responses'].select([columns[0], *selected]).filter(pl.col(columns[0]) == cell_line_id).collect()
    if row.height != 1:
        return Error(f'Expected one response row for {cell_line_id}; found {row.height}')
    melted = row.unpivot(index=columns[0], on=selected, variable_name='column_name', value_name='response')
    return join_measurements(tr.filter(pl.col('column_name').is_in(selected)), melted)


def eligible_observations(df):
    valid = (pl.col('dose').is_finite() & (pl.col('dose') > 0) & pl.col('viability').is_finite()).fill_null(False)
    return df.filter(valid), df.filter(~valid).with_columns(pl.lit('Missing/nonfinite response or nonpositive dose').alias('exclusion_reason'))


def dataset_counts(prism):
    curves = prism['curves']
    counts = curves.select(pl.len().alias('raw_curve_records'), pl.col('broad_id').drop_nulls().n_unique().alias('compound_ids'),
                           pl.col('depmap_id').drop_nulls().n_unique().alias('cell_line_ids'),
                           pl.struct(['broad_id', 'depmap_id']).n_unique().alias('compound_cell_pairs'),
                           pl.struct(['broad_id', 'depmap_id', 'screen_id']).n_unique().alias('compound_cell_screen_records')).collect().row(0, named=True)
    valid_ids = curves.filter(pl.col('depmap_id').is_not_null() & pl.col('broad_id').is_not_null())
    counts.update(valid_ids.select(pl.struct(['broad_id', 'depmap_id']).n_unique().alias('compound_cell_pairs'), pl.struct(['broad_id', 'depmap_id', 'screen_id']).n_unique().alias('compound_cell_screen_records')).collect().row(0, named=True))
    counts['compound_display_names'] = curves.select(pl.col('name').drop_nulls().n_unique()).collect().item()
    counts['curve_rows_missing_cell_identifier'] = curves.filter(pl.col('depmap_id').is_null()).select(pl.len()).collect().item()
    columns = prism['responses'].collect_schema().names()
    counts['response_rows'] = prism['responses'].select(pl.len()).collect().item()
    counts['response_columns'] = len(columns) - 1
    counts['observed_response_values'] = prism['responses'].select(pl.sum_horizontal(pl.exclude(columns[0]).count())).collect().item()
    tr_ids = prism['treatments'].select('column_name').collect()['column_name']
    counts['unmapped_response_columns'] = len(set(columns[1:]) - set(tr_ids))
    counts['duplicate_treatment_ids'] = len(tr_ids) - tr_ids.n_unique()
    return counts
