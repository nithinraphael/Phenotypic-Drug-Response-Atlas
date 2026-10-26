import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium', app_title='Atlas diagnostics')


@app.cell
def _():
    from pathlib import Path
    import json
    import marimo as mo
    from analysis import fitting, prism, provenance
    return Path, fitting, json, mo, prism, provenance


@app.cell
def _(Path, json, mo, provenance):
    diagnostics = mo.vstack([
        mo.md('# Atlas diagnostics'),
        mo.md('Public interface: `marimo run notebook.py`. Calculations below use the same analysis modules. Execution writes no reports.'),
        mo.ui.table([provenance.versions()], selection=None),
        mo.ui.table(json.loads(Path('data_inventory.json').read_text()), selection=None),
        mo.accordion({'Scientific audit': mo.md(Path('SCIENTIFIC_AUDIT.md').read_text()),
                      'Remaining blockers': mo.md(Path('REMAINING_BLOCKERS.md').read_text())}),
        mo.md('Explicit report generation: `.venv/bin/python -m analysis.run --output reports/runs/NEW_RUN`. Use a new directory for every run. Historical experiments are quarantined in legacy/unverified and require review before reuse.'),
    ])
    diagnostics
    return (diagnostics,)


@app.cell
def _(mo):
    diagnostic_compound = mo.ui.text(value='BRD-K88510285-001-11-1', label='Compound BRD identifier')
    diagnostic_cell = mo.ui.text(value='ACH-000943', label='Cell DepMap identifier')
    diagnostic_screen = mo.ui.text(value='HTS002', label='Screen identifier')
    diagnostic_run = mo.ui.run_button(label='Inspect selected observations and fit')
    mo.vstack([diagnostic_compound, diagnostic_cell, diagnostic_screen, diagnostic_run])
    return diagnostic_cell, diagnostic_compound, diagnostic_run, diagnostic_screen


@app.cell
def _(diagnostic_cell, diagnostic_compound, diagnostic_run, diagnostic_screen, fitting, mo, prism):
    if not diagnostic_run.value:
        analysis_output = mo.md('Not evaluated. Select identifiers and run diagnostics.')
    else:
        sources = prism.load_prism()
        measurements = prism.fetch_measurements(sources.ok, diagnostic_compound.value, diagnostic_cell.value, diagnostic_screen.value) if sources.is_ok() else sources
        if measurements.is_error():
            analysis_output = mo.md(measurements.error)
        else:
            valid, excluded = prism.eligible_observations(measurements.ok)
            fit = fitting.fit_hill(valid['dose'].to_numpy(), valid['viability'].to_numpy())
            analysis_output = mo.vstack([
                mo.ui.table(measurements.ok, selection=None),
                mo.md(f'Fit status: {fit.status}. {fit.reason or fit.diagnostics}'),
                mo.ui.table(excluded, selection=None),
            ])
    analysis_output
    return (analysis_output,)


if __name__ == '__main__':
    app.run()
