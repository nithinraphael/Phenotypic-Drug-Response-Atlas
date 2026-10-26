import argparse
from pathlib import Path

import polars as pl

from analysis.benchmark import aggregate_predictions, evaluate_curve
from analysis.catalog import build_catalog
from analysis.fitting import fit_hill
from analysis.models import predict_tabpfn, tabpfn_config
from analysis.prism import dataset_counts, eligible_observations, fetch_measurements, load_prism
from analysis.provenance import manifest, write_json
from analysis.reporting import export_report
from components import build_drc_plot, build_generalization_plot


def main():
    parser = argparse.ArgumentParser(description='Reproduce source-fit screening and a selected-curve concentration benchmark')
    parser.add_argument('--output', required=True)
    parser.add_argument('--compound-id', default='BRD-K88510285-001-11-1')
    parser.add_argument('--cell-id', default='ACH-000943')
    parser.add_argument('--screen')
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--include-tabpfn', action='store_true', help='Evaluate the explicitly configured local TabPFN model')
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Output already exists; use a new run directory')
    loaded = load_prism()
    if loaded.is_error():
        parser.error(loaded.error)
    tables = loaded.ok
    columns = ['broad_id','depmap_id','screen_id','name','ccle_name','ec50','slope','r2','lower_limit','upper_limit']
    ranked = build_catalog(tables['curves'].select(columns), tables['cell_lines'].collect())
    if ranked.is_error():
        parser.error(ranked.error)
    catalog, exclusions, excluded_curve_records = ranked.ok
    available = tables['curves'].filter((pl.col('broad_id') == args.compound_id) & (pl.col('depmap_id') == args.cell_id)).select('screen_id').unique().collect()['screen_id'].sort().to_list()
    if not available or (args.screen is not None and args.screen not in available):
        parser.error('Requested condition/screen unavailable')
    screen = args.screen or available[0]
    fetched = fetch_measurements(tables, args.compound_id, args.cell_id, screen)
    if fetched.is_error():
        parser.error(fetched.error)
    observations, excluded_observations = eligible_observations(fetched.ok)
    configured = tabpfn_config() if args.include_tabpfn else None
    if configured is not None and configured.is_error():
        parser.error(configured.error)
    model_config = configured.ok if configured is not None else None
    selected_models = ('Hill LL4', 'Mean baseline', 'Segmented') + (('TabPFN',) if model_config else ())
    evaluated = evaluate_curve(observations, f'{args.compound_id}/{args.cell_id}/{screen}', models=selected_models, n_splits=args.folds, tabpfn=model_config)
    if evaluated.is_error():
        parser.error(evaluated.error)
    predictions = evaluated.ok
    metrics = aggregate_predictions(predictions)
    import numpy as np
    fit = fit_hill(observations['dose'].to_numpy(), observations['viability'].to_numpy())
    grid = np.geomspace(observations['dose'].min(), observations['dose'].max(), 120)
    conditioned = predict_tabpfn(observations['dose'].to_numpy(), observations['viability'].to_numpy(), grid, model_config) if model_config else None
    name = catalog.filter(pl.col('broad_id') == args.compound_id)['name'].first()
    figures = {'held_out_benchmark':build_generalization_plot(metrics),
               'dose_response':build_drc_plot(observations['dose'].to_numpy(),observations['viability'].to_numpy(),grid,fit, pred_tabpfn=conditioned.ok if conditioned is not None and conditioned.is_ok() else None,
                   model_label=f"TabPFN {model_config['model_version']} conditioned prediction" if model_config else None, drug=name,cell_line=args.cell_id)}
    inputs = [*sorted(Path('data/prism').glob('*.csv')), *sorted(Path('analysis').glob('*.py')), Path('components.py'),Path('pyproject.toml'),Path('uv.lock')]
    if model_config:
        inputs.extend([Path('config/tabpfn.json'), Path(model_config['checkpoint_path'])])
    config = dict(compound_id=args.compound_id, cell_line_id=args.cell_id, screen_id=screen,
                  screen_selection='explicit' if args.screen else 'lexicographically first available; other screens not pooled',
                  random_seed=42, split_strategy='shuffled KFold on unique concentrations', n_splits=args.folds,
                  models=list(selected_models), tabpfn=model_config, primary_metric='pooled held-out observation RMSE',
                  comparison_metric='pooled RMSE on common successful observations', catalog='finite EC50>0; R² in [0.75,1]; abs(slope) in (1,20]')
    metadata = manifest('screening_and_selected_curve_benchmark', inputs, config,
                        dataset_counts=dataset_counts(tables), eligible_records=catalog.height,
                        observations=observations.height, excluded_observations=excluded_observations.height)
    exported = export_report(args.output,catalog,exclusions,predictions,metrics,metadata,figures)
    if exported.is_error():
        parser.error(exported.error)
    root = exported.ok
    excluded_curve_records.write_parquet(root/'excluded_curve_records.parquet')
    if conditioned is not None and conditioned.is_ok():
        pl.DataFrame({'dose_um':grid,'predicted_viability':conditioned.ok}).write_csv(root/'conditioned_tabpfn_predictions.csv')
    fetched.ok.write_csv(root/'selected_observations.csv')
    excluded_observations.write_csv(root/'observation_exclusions.csv')
    diagnostics = dict(status=fit.status, reason=fit.reason, ec50=fit.ec50,hill_slope=fit.hill_slope,
                       lower_asymptote=fit.lower_asymptote,upper_asymptote=fit.upper_asymptote,r2=fit.r2,rmse=fit.rmse,
                       ec20=fit.ec20,ec80=fit.ec80,diagnostics=fit.diagnostics,
                       tabpfn_status='ok' if conditioned is not None and conditioned.is_ok() else (conditioned.error if conditioned is not None else 'Not evaluated'))
    write_json(root/'selected_fit.json', diagnostics)
    from analysis.provenance import sha256
    write_json(root/'outputs.json',{p.name:sha256(p) for p in root.iterdir() if p.is_file() and p.name!='outputs.json'})
    print(metrics)
    print(root)


if __name__ == '__main__':
    main()
