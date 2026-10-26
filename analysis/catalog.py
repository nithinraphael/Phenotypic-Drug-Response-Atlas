import math

import polars as pl
from expression import Error, Ok

KEYS = ['broad_id', 'depmap_id', 'screen_id']
CATEGORY_NOTE = 'Categories use predefined slope and fit-quality thresholds. They do not indicate statistical significance.'


def response_category(slope, r2, converged=True):
    if not converged:
        return 'Fit failed'
    if slope is None or r2 is None or not math.isfinite(slope) or not math.isfinite(r2):
        return 'Fit unavailable'
    if abs(slope) >= 2 and r2 >= 0.90:
        return 'Steep response, strong fit'
    if abs(slope) >= 2 and 0.75 <= r2 < 0.90:
        return 'Steep response, moderate fit'
    return 'Other qualifying fit'


def build_catalog(curves, cell_lines=None):
    df = curves.collect() if isinstance(curves, pl.LazyFrame) else curves
    if not set(KEYS + ['name', 'ec50', 'slope', 'r2', 'lower_limit', 'upper_limit']).issubset(df.columns):
        return Error('Missing PRISM curve columns')
    df = df.with_row_index('source_row_index')
    audit, excluded_records = [], []
    criteria = {
        'missing identifiers': pl.all_horizontal(pl.col(k).is_not_null() for k in KEYS),
        'invalid fitted parameters': pl.all_horizontal(pl.col(k).is_finite() for k in ['ec50', 'slope', 'r2', 'lower_limit', 'upper_limit']),
        'nonpositive EC50': pl.col('ec50') > 0,
        'R² below 0.75 or above 1': pl.col('r2').is_between(0.75, 1),
        'absolute slope not in (1, 20]': (pl.col('slope').abs() > 1) & (pl.col('slope').abs() <= 20),
    }
    for reason, criterion in criteria.items():
        valid = criterion.fill_null(False)
        excluded_records.append(df.filter(~valid).select('source_row_index', *KEYS).with_columns(pl.lit(reason).alias('exclusion_reason')))
        kept = df.filter(valid)
        audit.append({'reason': reason, 'excluded': df.height - kept.height, 'remaining': kept.height})
        df = kept
    if df.select(KEYS).is_duplicated().any():
        return Error('Duplicate compound–cell-line–screen curve identifiers')
    if cell_lines is not None:
        annotation = cell_lines.select('depmap_id', 'primary_tissue').drop_nulls('depmap_id').unique()
        if annotation['depmap_id'].n_unique() != annotation.height:
            return Error('Ambiguous cell-line tissue annotations')
        df = df.join(annotation, on='depmap_id', how='left', validate='m:1')
    df = df.with_columns((pl.col('slope').abs() * pl.col('r2')).alias('Screening Score'))
    df = df.sort(['Screening Score', *KEYS], descending=[True, False, False, False])
    categories = [response_category(s, r) for s, r in df.select('slope', 'r2').iter_rows()]
    flags = ['Source fit; convergence and transition coverage unverified' + ('; extreme slope' if abs(s) >= 19 else '')
             for s in df['slope']]
    return Ok((df.with_columns(pl.Series('Response Category', categories, dtype=pl.String),
                               pl.Series('Fit diagnostics', flags, dtype=pl.String)), pl.DataFrame(audit), pl.concat(excluded_records)))
