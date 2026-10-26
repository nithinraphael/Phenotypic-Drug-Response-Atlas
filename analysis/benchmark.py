import importlib.metadata

import numpy as np
import polars as pl
from expression import Error, Ok
from sklearn.model_selection import KFold

from analysis.fitting import fit_hill, validate_observations
from analysis.models import isolated_predict


PREDICTION_SCHEMA = {'model': pl.String, 'model_version': pl.String, 'checkpoint_sha256': pl.String, 'curve_id': pl.String, 'fold': pl.Int64, 'observation_id': pl.String,
                     'dose': pl.Float64, 'observed': pl.Float64, 'predicted': pl.Float64, 'failure': pl.String}
SUMMARY_SCHEMA = {'model': pl.String, 'model_version': pl.String, 'package_version': pl.String, 'checkpoint_sha256': pl.String, 'curves': pl.Int64, 'folds': pl.Int64,
                  'held_out_observations': pl.Int64, 'successful_predictions': pl.Int64, 'failed_predictions': pl.Int64,
                  'failed_folds': pl.Int64, 'rmse': pl.Float64, 'common_observations': pl.Int64, 'common_rmse': pl.Float64}


def concentration_splits(doses, n_splits=3, seed=42):
    x = np.asarray(doses)
    unique = np.unique(x)
    if n_splits < 2 or len(unique) < n_splits:
        return Error('Insufficient concentrations for requested folds')
    return Ok([(np.flatnonzero(np.isin(x, unique[train])), np.flatnonzero(np.isin(x, unique[test])))
               for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(unique)])


def rmse(observed, predicted):
    y, p = np.asarray(observed), np.asarray(predicted)
    if y.shape != p.shape or y.ndim != 1 or not y.size or y.dtype.kind not in 'iuf' or p.dtype.kind not in 'iuf':
        return Error('RMSE requires matching nonempty numeric vectors')
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        return Error('RMSE requires finite paired observations')
    return Ok(float(np.sqrt(np.mean((y - p) ** 2))))


def evaluate_curve(data, curve_id, models=('Hill LL4', 'Mean baseline', 'Segmented'), n_splits=3, seed=42, tabpfn=None):
    checked = validate_observations(data['dose'].to_numpy(), data['viability'].to_numpy())
    if checked.is_error():
        return checked
    if data['column_name'].n_unique() != data.height or data['screen_id'].n_unique() != 1 or data['broad_id'].n_unique() != 1:
        return Error('Benchmark requires unique observations within one compound and screen')
    x, y = checked.ok
    splits = concentration_splits(x, n_splits, seed)
    if splits.is_error():
        return splits
    rows = []
    for fold, (train, test) in enumerate(splits.ok):
        for model in models:
            if model == 'Hill LL4':
                fit = fit_hill(x[train], y[train])
                predictions = fit.predict(x[test])
            elif model == 'Mean baseline':
                predictions = Ok(np.full(len(test), y[train].mean()))
            elif model == 'Segmented':
                predictions = isolated_predict('Segmented', x[train], y[train], x[test], {'random_seed': seed})
            elif model == 'TabPFN':
                predictions = Error('TabPFN not configured') if tabpfn is None else isolated_predict('TabPFN', x[train], y[train], x[test], tabpfn)
            else:
                return Error(f'Unknown benchmark model: {model}')
            if predictions.is_ok() and rmse(y[test], predictions.ok).is_error():
                predictions = Error('Invalid held-out predictions')
            for i, index in enumerate(test):
                package = {'Hill LL4':'thunor', 'Mean baseline':'numpy', 'Segmented':'piecewise-regression', 'TabPFN':'tabpfn'}
                version = tabpfn['model_version'] if model == 'TabPFN' and tabpfn else importlib.metadata.version(package[model])
                checkpoint = tabpfn['checkpoint_sha256'] if model == 'TabPFN' and tabpfn else None
                rows.append(dict(model=model, model_version=version, checkpoint_sha256=checkpoint, curve_id=curve_id, fold=fold, observation_id=str(data['column_name'][int(index)]),
                                 dose=float(x[index]), observed=float(y[index]),
                                 predicted=float(predictions.ok[i]) if predictions.is_ok() else None,
                                 failure=predictions.error if predictions.is_error() else None))
    return Ok(pl.DataFrame(rows, schema=PREDICTION_SCHEMA))


def aggregate_predictions(predictions):
    if predictions.is_empty():
        return pl.DataFrame(schema=SUMMARY_SCHEMA)
    models = predictions['model'].unique().sort().to_list()
    successes = predictions.filter(pl.col('failure').is_null() & pl.col('predicted').is_not_null())
    common = successes.group_by(['curve_id', 'fold', 'observation_id']).len().filter(pl.col('len') == len(models)).drop('len')
    rows = []
    package = {'Hill LL4': 'thunor', 'Mean baseline': 'numpy', 'Segmented': 'piecewise-regression', 'TabPFN': 'tabpfn'}
    for model in models:
        all_rows = predictions.filter(pl.col('model') == model)
        valid = successes.filter(pl.col('model') == model)
        matched = valid.join(common, on=['curve_id', 'fold', 'observation_id'], how='inner', maintain_order='left')
        rows.append(dict(model=model, model_version=all_rows['model_version'].unique().item(),
                         package_version=importlib.metadata.version(package[model]), checkpoint_sha256=all_rows['checkpoint_sha256'].unique().item(),
                         curves=all_rows['curve_id'].n_unique(), folds=all_rows.select('curve_id', 'fold').unique().height,
                         held_out_observations=all_rows.height, successful_predictions=valid.height,
                         failed_predictions=all_rows.height-valid.height,
                         failed_folds=all_rows.filter(pl.col('failure').is_not_null()).select('curve_id', 'fold').unique().height,
                         rmse=rmse(valid['observed'].to_numpy(), valid['predicted'].to_numpy()).ok if valid.height else None,
                         common_observations=matched.height,
                         common_rmse=rmse(matched['observed'].to_numpy(), matched['predicted'].to_numpy()).ok if matched.height else None))
    return pl.DataFrame(rows, schema=SUMMARY_SCHEMA)
