import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium', app_title='Phenotypic Drug-Response Atlas')


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import polars as pl
    from expression import Error, Ok
    from analysis import benchmark, catalog, embeddings, fitting, microscopy, models, prism, provenance, validation
    import components
    return Path, Error, Ok, benchmark, catalog, components, embeddings, fitting, microscopy, mo, models, np, pl, prism, provenance, validation


@app.cell
def _(Path, mo, prism, provenance, catalog, pl):
    source_identity = tuple((str(p), provenance.sha256(p)) for p in sorted(Path('data/prism').glob('*.csv')))

    @mo.cache
    def load_analysis(identity):
        loaded = prism.load_prism()
        if loaded.is_error():
            return loaded, None, None, None
        tables = loaded.ok
        counts = prism.dataset_counts(tables)
        columns = ['broad_id', 'depmap_id', 'screen_id', 'name', 'ec50', 'slope', 'r2', 'lower_limit', 'upper_limit', 'ccle_name']
        ranked = catalog.build_catalog(tables['curves'].select(columns), tables['cell_lines'].collect())
        return loaded, counts, ranked, tables['cell_lines'].collect()

    prism_result, counts, ranked_result, cell_metadata = load_analysis(source_identity)
    return cell_metadata, counts, prism_result, ranked_result, source_identity


@app.cell
def _(catalog, counts, mo, pl, prism_result, ranked_result):
    if prism_result.is_error():
        overview_view = mo.md(prism_result.error)
    elif ranked_result.is_error():
        overview_view = mo.md(ranked_result.error)
    else:
        eligible_catalog, catalog_exclusions, excluded_curve_records = ranked_result.ok
        shortlist = eligible_catalog.head(100)
        overview_view = mo.vstack([
            mo.md('### Candidate screening'),
            mo.ui.table(pl.DataFrame([counts]), selection=None),
            mo.md(f'Showing {shortlist.height} highest-ranked curves from {eligible_catalog.height:,} eligible source-fit records.'),
            mo.md('Screening Score = absolute fitted Hill slope × R². Extreme slopes can be unstable. Source-fit convergence and transition coverage have not been reverified.'),
            mo.md(catalog.CATEGORY_NOTE),
            mo.ui.table(shortlist.select(
                pl.col('name').alias('Compound'), pl.col('ccle_name').alias('Cell line'), pl.col('primary_tissue').alias('Tissue'),
                pl.col('ec50').alias('EC50 (μM)'), pl.col('slope').alias('Hill slope'), pl.col('r2').alias('R²'),
                'Screening Score', 'Response Category', 'Fit diagnostics', 'broad_id', 'depmap_id', 'screen_id'), selection=None),
            mo.hstack([
                mo.download(eligible_catalog.write_csv().encode(), filename='eligible_catalog.csv', label='Export full eligible catalog'),
                mo.download(shortlist.write_csv().encode(), filename='shortlist.csv', label='Export displayed shortlist'),
            ]),
            mo.accordion({'Sequential filter exclusions': mo.ui.table(catalog_exclusions, selection=None)}),
        ])
    return (overview_view,)


@app.cell
def _(mo, pl, prism_result):
    compound_rows = (prism_result.ok['curves'].select('broad_id', 'name').unique().collect().sort(['name', 'broad_id'])
                     if prism_result.is_ok() else pl.DataFrame(schema={'broad_id':pl.String,'name':pl.String}))
    compound_options = {f"{r['name']} · {r['broad_id']}": r['broad_id'] for r in compound_rows.iter_rows(named=True)}
    default_compound = next((label for label in compound_options if label.startswith('bortezomib ·')), next(iter(compound_options), None))
    compound_select = mo.ui.dropdown(compound_options, value=default_compound, searchable=True, label='Compound')
    return compound_select, compound_rows


@app.cell
def _(cell_metadata, compound_select, mo, pl, prism_result):
    tested = (prism_result.ok['curves'].filter(pl.col('broad_id') == compound_select.value).select('depmap_id').drop_nulls().unique().collect()
              if prism_result.is_ok() else pl.DataFrame(schema={'depmap_id':pl.String}))
    names = dict(cell_metadata.select('depmap_id','ccle_name').iter_rows()) if cell_metadata is not None else {}
    cell_options = {f'{names.get(key) or key} · {key}': key for key in sorted(tested['depmap_id'].drop_nulls())}
    default_cell = next((label for label in cell_options if 'RKO' in label), next(iter(cell_options), None))
    cell_select = mo.ui.dropdown(cell_options, value=default_cell, searchable=True, label='Cell line')
    return (cell_select,)


@app.cell
def _(cell_select, compound_select, mo, pl, prism_result):
    screens = (prism_result.ok['curves'].filter((pl.col('broad_id') == compound_select.value) & (pl.col('depmap_id') == cell_select.value))
               .select('screen_id').unique().collect()['screen_id'].drop_nulls().sort().to_list() if prism_result.is_ok() else [])
    screen_select = mo.ui.dropdown(screens, value=screens[0] if screens else None, label='Source screen')
    return (screen_select,)


@app.cell
def _(Error, cell_select, compound_select, mo, prism, prism_result, screen_select, source_identity):
    @mo.cache
    def fetch_selected(identity, broad_id, cell_id, screen):
        return prism.fetch_measurements(prism_result.ok, broad_id, cell_id, screen)

    measurement_result = (fetch_selected(source_identity, compound_select.value, cell_select.value, screen_select.value)
                          if prism_result.is_ok() and screen_select.value else Error('No available PRISM condition'))
    if measurement_result.is_ok():
        measurements, measurement_exclusions = prism.eligible_observations(measurement_result.ok)
    else:
        measurements, measurement_exclusions = None, None
    return measurement_exclusions, measurement_result, measurements


@app.cell
def _(fitting, measurements, mo, np):
    @mo.cache
    def fit_selected(doses, responses):
        return fitting.fit_hill(np.asarray(doses), np.asarray(responses))

    hill_fit = fit_selected(tuple(measurements['dose']), tuple(measurements['viability'])) if measurements is not None and measurements.height else fitting.FitResult('failed', 'No valid observations')
    prediction_grid = np.geomspace(measurements['dose'].min(), measurements['dose'].max(), 120) if measurements is not None and measurements.height else np.array([])
    return hill_fit, prediction_grid


@app.cell
def _(measurements, mo):
    cursor_doses = sorted(measurements['dose'].unique().to_list()) if measurements is not None and measurements.height else []
    concentration_cursor = mo.ui.dropdown(cursor_doses, value=cursor_doses[len(cursor_doses)//2] if cursor_doses else None, label='Concentration cursor (μM)')
    run_prediction = mo.ui.run_button(label='Run configured TabPFN prediction')
    run_benchmark = mo.ui.run_button(label='Evaluate held-out concentrations')
    return concentration_cursor, run_benchmark, run_prediction


@app.cell
def _(Path, models, provenance):
    model_config = models.tabpfn_config()
    model_config_identity = provenance.sha256('config/tabpfn.json') if Path('config/tabpfn.json').is_file() else None
    return model_config, model_config_identity


@app.cell
def _(Error, measurements, mo, model_config, model_config_identity, models, np, prediction_grid, run_prediction):
    @mo.cache
    def predict_selected(doses, responses, grid, config_identity, configuration):
        return models.predict_tabpfn(np.asarray(doses), np.asarray(responses), np.asarray(grid), configuration)

    if model_config.is_error():
        tab_prediction = Error(model_config.error)
    elif not run_prediction.value:
        tab_prediction = Error('TabPFN not evaluated for this selection')
    elif measurements is None or not measurements.height:
        tab_prediction = Error('No valid observations for prediction')
    else:
        tab_prediction = predict_selected(tuple(measurements['dose']), tuple(measurements['viability']), tuple(prediction_grid), model_config_identity, model_config.ok)
    return (tab_prediction,)


@app.cell
def _(cell_select, components, compound_select, concentration_cursor, hill_fit, measurement_exclusions, measurement_result, measurements, mo, model_config, prediction_grid, run_benchmark, run_prediction, screen_select, tab_prediction):
    if measurements is None or not measurements.height:
        dose_content = mo.md(measurement_result.error if measurement_result.is_error() else 'No valid measurements; inspect input exclusions.')
    else:
        dose_figure = components.build_drc_plot(measurements['dose'].to_numpy(), measurements['viability'].to_numpy(), prediction_grid,
            hill_fit, tab_prediction.ok if tab_prediction.is_ok() else None,
            concentration_cursor.value, compound_select.selected_key or '', cell_select.selected_key or '',
            f"TabPFN {model_config.ok['model_version']} conditioned on observations" if model_config.is_ok() else None)
        dose_content = mo.vstack([
            mo.ui.plotly(dose_figure),
            mo.md(f'Fit: {hill_fit.status}. ' + (hill_fit.reason or f'EC50 {hill_fit.ec50:.4g} μM; slope {hill_fit.hill_slope:.4g}; R² {hill_fit.r2:.4f}; RMSE {hill_fit.rmse:.4f}.')),
            mo.md('; '.join(hill_fit.diagnostics.get('flags', [])) or 'No diagnostic flags recorded.'),
            mo.md(tab_prediction.error if tab_prediction.is_error() else 'TabPFN predictions conditioned on this curve’s measured examples.'),
            mo.download(measurement_result.ok.write_csv().encode(), filename='selected_measurements.csv', label='Export current observations, including missing responses'),
            mo.accordion({'Excluded observations': measurement_exclusions}),
        ])
    drc_view = mo.vstack([
        mo.md('### Dose–Response'),
        mo.hstack([compound_select, cell_select, screen_select], wrap=True),
        concentration_cursor,
        mo.hstack([run_prediction, run_benchmark], wrap=True),
        dose_content,
        mo.md('EC50 is the fitted response midpoint. EC20–EC80 describes inhibition of the fitted effect span, not absolute viability or uncertainty. Measurements remain separated by compound identifier and screen.'),
    ])
    return (drc_view,)


@app.cell
def _(Error, benchmark, measurements, run_benchmark, cell_select, compound_select, screen_select, model_config):
    if run_benchmark.value and measurements is not None and measurements.height:
        evaluation = benchmark.evaluate_curve(measurements, f'{compound_select.value}/{cell_select.value}/{screen_select.value}',
            models=('Hill LL4', 'Mean baseline', 'Segmented') + (('TabPFN',) if model_config.is_ok() else ()),
            tabpfn=model_config.ok if model_config.is_ok() else None)
    else:
        evaluation = Error('Not evaluated: run held-out concentration evaluation for the selected curve')
    return (evaluation,)


@app.cell
def _(microscopy, mo, provenance, Path):
    image_identity = tuple((str(p), provenance.sha256(p)) for p in [Path('data/bbbc021/metadata.csv')] if p.is_file())
    @mo.cache
    def read_image_metadata(identity):
        return microscopy.load_metadata()
    image_metadata = read_image_metadata(image_identity)
    imaging_options = sorted(image_metadata.ok['Image_Metadata_Compound'].unique().to_list()) if image_metadata.is_ok() else []
    imaging_compound = mo.ui.dropdown(imaging_options, value='taxol' if 'taxol' in imaging_options else next(iter(imaging_options), None), label='Imaging compound')
    return image_metadata, imaging_compound


@app.cell
def _(image_metadata, imaging_compound, mo, pl):
    image_doses = sorted(image_metadata.ok.filter(pl.col('Image_Metadata_Compound') == imaging_compound.value)['Image_Metadata_Concentration'].drop_nulls().unique().to_list()) if image_metadata.is_ok() else []
    imaging_dose = mo.ui.dropdown(image_doses, value=image_doses[len(image_doses)//2] if image_doses else None, label='Imaging concentration (μM)')
    return (imaging_dose,)


@app.cell
def _(image_metadata, imaging_compound, imaging_dose, mo, pl):
    field_rows = image_metadata.ok.filter((pl.col('Image_Metadata_Compound') == imaging_compound.value) & (pl.col('Image_Metadata_Concentration') == imaging_dose.value)) if image_metadata.is_ok() else None
    field_options = {f"Well {r['Image_Metadata_Well_DAPI']} · field {r['ImageNumber']} · replicate {r['Replicate']}":r['image_field_id'] for r in field_rows.iter_rows(named=True)} if field_rows is not None else {}
    field_select = mo.ui.dropdown(field_options, value=next(iter(field_options), None), label='Image field')
    return (field_select,)


@app.cell
def _(Error, embeddings, image_metadata, mo):
    @mo.cache
    def read_projection(metadata):
        return embeddings.configured_projection(metadata)
    projection = read_projection(image_metadata.ok) if image_metadata.is_ok() else Error(image_metadata.error)
    return (projection,)


@app.cell
def _(components, field_select, image_metadata, imaging_compound, imaging_dose, microscopy, mo, pl, projection):
    image_parts = []
    if image_metadata.is_error():
        image_parts.append(mo.md(image_metadata.error))
    elif field_select.value:
        treated = image_metadata.ok.filter(pl.col('image_field_id') == field_select.value).row(0, named=True)
        rgb = microscopy.load_microscopy_rgb('data/bbbc021/plate.zip', treated)
        condition_label = 'Vehicle control (DMSO)' if treated['Image_Metadata_Compound'] == 'DMSO' else f"{treated['Image_Metadata_Compound']} · {treated['Image_Metadata_Concentration']} μM"
        image_parts.append(mo.md(f"{condition_label} · {treated['image_field_id']}"))
        image_parts.append(mo.image(rgb.ok, width=440, alt=f"Treated field: {treated['image_field_id']}") if rgb.is_ok() else mo.md(rgb.error))
        control = microscopy.select_control(image_metadata.ok, treated)
        if control.is_ok():
            control_rgb = microscopy.load_microscopy_rgb('data/bbbc021/plate.zip', control.ok)
            image_parts.append(mo.md(f"Vehicle control (DMSO) · {control.ok['image_field_id']}. Same plate and replicate group; acquisition matching beyond metadata unverified. First field by stable identifier."))
            image_parts.append(mo.image(control_rgb.ok, width=440, alt='Vehicle control DMSO') if control_rgb.is_ok() else mo.md(control_rgb.error))
        else:
            image_parts.append(mo.md(control.error))
    else:
        image_parts.append(mo.md('No microscopy field for the selected condition.'))
    image_parts.append(mo.md(projection.error) if projection.is_error() else mo.ui.plotly(components.build_umap_plot(projection.ok, imaging_compound.value, imaging_dose.value)))
    imaging_view = mo.vstack([
        mo.md('### Cell Imaging'),
        mo.md(f"Available metadata fields on Week1_22123: {image_metadata.ok.height if image_metadata.is_ok() else 'unavailable'}. Fields analyzed with verified embeddings: {projection.ok.height if projection.is_ok() else 'unavailable'}."),
        mo.hstack([imaging_compound, imaging_dose, field_select], wrap=True),
        *image_parts,
        mo.md('Contrast-adjusted RGB: red actin, green tubulin, blue DAPI. Raw intensities are unchanged. BBBC021 and PRISM observations are not paired.'),
        mo.md('UMAP can distort distances and global geometry. Lines connect concentration-group centroids for visualization; they do not represent measured temporal trajectories. Clusters alone do not establish mechanisms.'),
    ])
    return (imaging_view,)


@app.cell
def _(validation):
    external_result = validation.load_configured_validation()
    return (external_result,)


@app.cell
def _(benchmark, components, evaluation, external_result, mo, validation, pl):
    validation_parts = [mo.md('### External dataset comparison')]
    if external_result.is_error():
        validation_parts.append(mo.md(external_result.error))
    else:
        matched, external_tables, external_counts, external_config = external_result.ok
        validation_parts.extend([
            mo.md(f'Matched conditions with identical metric definitions: {matched.height}'),
            mo.ui.table(pl.DataFrame(external_counts), selection=None),
            mo.ui.table(validation.compare_potencies(matched).ok, selection=None),
            mo.ui.table(matched, selection=None),
            mo.download(matched.write_csv().encode(), filename='matched_external_measurements.csv', label='Export matched measurements'),
            mo.ui.plotly(components.build_validation_plot(matched, {'reference':external_config['reference']['source']['dataset'],'external':external_config['external']['source']['dataset']})),
        ])
    validation_parts.append(mo.md('### Held-out concentration prediction'))
    if evaluation.is_error():
        validation_parts.append(mo.md(evaluation.error))
    else:
        metrics = benchmark.aggregate_predictions(evaluation.ok)
        validation_parts.extend([mo.ui.table(metrics.with_columns(pl.col('rmse', 'common_rmse').cast(pl.String).fill_null('Not evaluated')), selection=None), mo.ui.plotly(components.build_generalization_plot(metrics)),
                                  mo.download(evaluation.ok.write_csv().encode(), filename='held_out_predictions.csv', label='Export predictions and failures')])
    validation_parts.append(mo.md('Pooled observation RMSE weights curves by observation count. Common RMSE compares identical successful observations across models; failures remain counted. This evaluates held-out concentrations within the selected curve, not unseen compound–cell-line combinations. No transition-zone benchmark or significance test.'))
    validation_view = mo.vstack(validation_parts)
    return (validation_view,)


@app.cell
def _(Path, mo, provenance):
    methods_view = mo.vstack([mo.md(Path('METHODS.md').read_text()), mo.ui.table([provenance.versions()], selection=None)])
    return (methods_view,)


@app.cell
def _(drc_view, imaging_view, methods_view, mo, overview_view, validation_view):
    application_view = mo.vstack([
        mo.md('# Phenotypic Drug-Response Atlas'),
        mo.md('Drug response and cell morphology. Static dose responses do not establish irreversible cell-state switching.'),
        mo.ui.tabs({'Overview': overview_view, 'Dose–Response': drc_view, 'Cell Imaging': imaging_view, 'Validation': validation_view, 'Methods': methods_view}),
    ])
    application_view
    return (application_view,)


if __name__ == '__main__':
    app.run()
