from pathlib import Path
import tarfile

from expression import Error, Ok

from analysis.provenance import sha256, write_json


def export_report(directory, catalog, audit, predictions, benchmark, provenance, figures=None):
    root = Path(directory)
    if root.exists():
        return Error(f'Report directory already exists: {root}; choose a new run directory')
    if provenance.get('status') != 'completed' or not provenance.get('inputs'):
        return Error('Completed analysis with input provenance required')
    for path, digest in provenance['inputs'].items():
        if not Path(path).is_file() or sha256(path) != digest:
            return Error(f'Report input missing or changed: {path}')
    from analysis.benchmark import aggregate_predictions
    calculated = aggregate_predictions(predictions)
    if not calculated.equals(benchmark):
        return Error('Benchmark table does not reconcile with held-out predictions')
    root.mkdir(parents=True)
    catalog.write_csv(root / 'eligible_catalog.csv')
    catalog.head(100).write_csv(root / 'shortlist.csv')
    audit.write_csv(root / 'catalog_exclusions.csv')
    predictions.write_csv(root / 'held_out_predictions.csv')
    benchmark.write_csv(root / 'model_benchmark.csv')
    write_json(root / 'manifest.json', provenance)
    with tarfile.open(root / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for path in provenance['inputs']:
            if Path(path).suffix in ('.py', '.toml', '.lock', '.json'):
                archive.add(path, arcname=path)
    text = ['# Results', f"Generated: {provenance['created_at']}",
            f'Eligible source-fit records: {catalog.height}. Display shortlist: {min(100,catalog.height)}.',
            'Screening Score = absolute fitted Hill slope × R². Descriptive ranking; no statistical significance.',
            'Source catalog fits are not raw-data refits; convergence and transition coverage remain unverified.',
            'Held-out concentration evaluation uses pooled observation RMSE. Curves with more observations receive more weight.',
            'Common RMSE uses only observations predicted successfully by every evaluated model.',
            'No transition-zone evaluation or uncertainty intervals.',
            'External comparison and embedding analysis omitted: provenance not configured.',
            'Static curves do not establish irreversible switching; PRISM viability and BBBC021 morphology are separate assays.',
            'See manifest.json for source hashes, software versions, configuration and code revision.']
    (root / 'REPORT.md').write_text('\n\n'.join(text) + '\n')
    for name, figure in (figures or {}).items():
        figure.write_html(root / f'{name}.html', include_plotlyjs=True)
    outputs = {p.name: sha256(p) for p in root.iterdir() if p.is_file()}
    write_json(root / 'outputs.json', outputs)
    return Ok(root)
