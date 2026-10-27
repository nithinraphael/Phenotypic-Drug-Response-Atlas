import marimo

__generated_with = '0.25.2'
app = marimo.App(width='full', css_file='atlas.css', app_title='Phenotypic Drug-Response Atlas')


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import polars as pl
    from expression import Error, Ok
    from analysis import benchmark, catalog, embeddings, fitting, microscopy, models, prism, provenance, validation
    import components
    import public_ui as ui
    import image_projection
    import external_evidence
    import single_cells
    import single_cell_ui
    return Path, Error, Ok, benchmark, catalog, components, embeddings, fitting, microscopy, mo, models, np, pl, prism, provenance, validation, ui, image_projection, external_evidence, single_cells, single_cell_ui


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
def _(catalog, counts, mo, pl, prism_result, ranked_result, ui):
    candidate_table = mo.ui.table(pl.DataFrame(), selection='single', show_data_types=False)
    if prism_result.is_error():
        overview_view = ui.unavailable('PRISM data unavailable', prism_result.error)
    elif ranked_result.is_error():
        overview_view = ui.unavailable('Candidates unavailable', ranked_result.error)
    else:
        eligible_catalog, catalog_exclusions, excluded_curve_records = ranked_result.ok
        shortlist = eligible_catalog.head(100)
        candidate_table = mo.ui.table(shortlist.select(
            pl.col('name').alias('Compound'), pl.col('ccle_name').alias('Cell line'),
            pl.col('ec50').alias('EC50 (μM)'), pl.col('slope').abs().alias('|Hill slope|'), pl.col('r2').alias('R²'),
            'Screening Score', 'Response Category', 'broad_id', 'depmap_id', 'screen_id'),
            selection='single', page_size=6, pagination=True, show_column_summaries=False, show_data_types=False,
            hidden_columns=['broad_id','depmap_id','screen_id'], max_height=370,
            format_mapping={'EC50 (μM)':'{:.4g}', '|Hill slope|':'{:.2f}', 'R²':'{:.3f}', 'Screening Score':'{:.2f}'},
            header_tooltip={'Screening Score':'Absolute Hill slope × R². A descriptive ranking, without calibrated statistical meaning.'},
            label='Ranked candidates')
        overview_view = mo.vstack([
            ui.metrics([('Observed curve records',f"{counts['raw_curve_records']:,}"),
                        ('Compounds by name',f"{counts['compound_display_names']:,}"),
                        ('Identified cell lines',f"{counts['cell_line_ids']:,}"),
                        ('Eligible source fits',f'{eligible_catalog.height:,}')]),
            mo.md('## Steep responses across the screen'),
            mo.md('Explore the highest-ranked PRISM fits by potency and steepness. Each point is a compound, cell line and source screen.'),
            mo.ui.plotly(ui.candidate_figure(eligible_catalog),config=ui.PLOT_CONFIG),
            mo.md('## Ranked candidates'),
            mo.md(f'Select a row, then open Drug Explorer to inspect its measured response. Showing 100 of {eligible_catalog.height:,} eligible source fits.'),
            candidate_table,
            mo.hstack([
                mo.download(lambda: eligible_catalog.write_csv().encode(),filename='eligible_catalog.csv',label='Download full catalog'),
                mo.download(lambda: shortlist.write_csv().encode(),filename='shortlist.csv',label='Download shortlist'),
            ],wrap=True),
            mo.accordion({'Ranking and interpretation':mo.md(
                'Screening score = absolute fitted Hill slope × R². Extreme slopes can be unstable. Source-fit convergence and dose coverage have not been reverified. '+catalog.CATEGORY_NOTE+
                ' Static responses cannot establish irreversible cell-state switching.')}),
        ],gap=1.25)
    return candidate_table, overview_view


@app.cell
def _(candidate_table, mo, pl, prism_result):
    compound_rows = (prism_result.ok['curves'].select('broad_id', 'name').unique().collect().sort(['name', 'broad_id'])
                     if prism_result.is_ok() else pl.DataFrame(schema={'broad_id':pl.String,'name':pl.String}))
    compound_options = {f"{r['name']} · {r['broad_id']}": r['broad_id'] for r in compound_rows.iter_rows(named=True)}
    default_compound = next((label for label in compound_options if label.startswith('bortezomib ·')), next(iter(compound_options), None))
    if len(candidate_table.value):
        default_compound = next((label for label,value in compound_options.items() if value == candidate_table.value['broad_id'][0]), default_compound)
    compound_select = mo.ui.dropdown(compound_options, value=default_compound, searchable=True, label='Compound', full_width=True)
    return compound_select, compound_rows


@app.cell
def _(candidate_table, cell_metadata, compound_select, mo, pl, prism_result):
    tested = (prism_result.ok['curves'].filter(pl.col('broad_id') == compound_select.value).select('depmap_id').drop_nulls().unique().collect()
              if prism_result.is_ok() else pl.DataFrame(schema={'depmap_id':pl.String}))
    names = dict(cell_metadata.select('depmap_id','ccle_name').iter_rows()) if cell_metadata is not None else {}
    cell_options = {f'{names.get(key) or key} · {key}': key for key in sorted(tested['depmap_id'].drop_nulls())}
    default_cell = next((label for label in cell_options if 'RKO' in label), next(iter(cell_options), None))
    if len(candidate_table.value) and candidate_table.value['broad_id'][0] == compound_select.value:
        default_cell = next((label for label,value in cell_options.items() if value == candidate_table.value['depmap_id'][0]), default_cell)
    cell_select = mo.ui.dropdown(cell_options, value=default_cell, searchable=True, label='Cell line', full_width=True)
    return (cell_select,)


@app.cell
def _(candidate_table, cell_select, compound_select, mo, pl, prism_result):
    screens = (prism_result.ok['curves'].filter((pl.col('broad_id') == compound_select.value) & (pl.col('depmap_id') == cell_select.value))
               .select('screen_id').unique().collect()['screen_id'].drop_nulls().sort().to_list() if prism_result.is_ok() else [])
    selected_screen = screens[0] if screens else None
    if len(candidate_table.value) and candidate_table.value['broad_id'][0] == compound_select.value and candidate_table.value['depmap_id'][0] == cell_select.value:
        selected_screen = candidate_table.value['screen_id'][0]
    screen_select = mo.ui.dropdown(screens, value=selected_screen, label='Screen', full_width=True)
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
    concentration_cursor = mo.ui.dropdown(cursor_doses, value=cursor_doses[len(cursor_doses)//2] if cursor_doses else None, label='Selected concentration (μM)', full_width=True)
    run_prediction = mo.ui.run_button(label='Add TabPFN prediction')
    run_benchmark = mo.ui.run_button(label='Evaluate this condition')
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
        with mo.status.spinner(title=f"Running TabPFN {model_config.ok['model_version']} prediction"):
            tab_prediction = predict_selected(tuple(measurements['dose']), tuple(measurements['viability']), tuple(prediction_grid), model_config_identity, model_config.ok)
    return (tab_prediction,)


@app.cell
def _(cell_select, components, compound_rows, compound_select, concentration_cursor, hill_fit, measurement_result, measurements, mo, model_config, pl, prediction_grid, run_prediction, screen_select, tab_prediction, ui):
    dose_figure = None
    if measurements is None or not measurements.height:
        dose_content = ui.unavailable('No response available', measurement_result.error if measurement_result.is_error() else 'No valid observations for this condition.')
    else:
        selected_drug_name = compound_rows.filter(pl.col('broad_id') == compound_select.value)['name'][0]
        dose_figure = ui.dose_figure(components.build_drc_plot(measurements['dose'].to_numpy(), measurements['viability'].to_numpy(), prediction_grid,
            hill_fit, tab_prediction.ok if tab_prediction.is_ok() else None,
            concentration_cursor.value, selected_drug_name, cell_select.selected_key or '',
            f"TabPFN {model_config.ok['model_version']} prediction" if model_config.is_ok() else None))
        fit_summary = ui.metrics([('EC50 (μM)',f'{hill_fit.ec50:.4g}'), ('Absolute Hill slope',f'{hill_fit.hill_slope:.2f}'),
            ('R²',f'{hill_fit.r2:.3f}'), ('Measured observations',str(measurements.height))]) if hill_fit.status == 'ok' else ui.unavailable('Hill fit unavailable',hill_fit.reason)
        dose_content = mo.vstack([
            mo.ui.plotly(dose_figure,config=ui.PLOT_CONFIG),
            fit_summary,
            mo.md('Circles show measured viability. The solid line is a Hill fit; the dashed blue line, when enabled, is a TabPFN prediction conditioned on these observations.'),
            ui.controls([concentration_cursor,run_prediction]),
            mo.md('TabPFN has not been evaluated for this condition. Select “Add TabPFN prediction” to run it.' if tab_prediction.is_error() and model_config.is_ok() and not run_prediction.value else (tab_prediction.error if tab_prediction.is_error() else f"TabPFN {model_config.ok['model_version']}: {len(tab_prediction.ok)} predictions, conditioned on {measurements.height} measured examples. Features: {model_config.ok['feature_definition']}.")),
            mo.md('Built with PriorLabs-TabPFN. [Model license](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt).'),
            mo.accordion({'Prediction model and license':mo.md('Pretrained [TabPFN v2 by Prior Labs](https://huggingface.co/Prior-Labs/TabPFN-v2-reg). '
                'Weights use the [Prior Labs license](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt), including attribution requirements. '
                'The configured checkpoint is verified by SHA-256; inference uses CPU and log10 concentration. No task-specific weight training is performed.')}),
            mo.download(lambda: pl.DataFrame({'dose_um':prediction_grid,'predicted_relative_viability':tab_prediction.ok}).with_columns(
                pl.lit(compound_select.value).alias('compound_id'),pl.lit(cell_select.value).alias('cell_line_id'),pl.lit(screen_select.value).alias('screen_id'),
                pl.lit(model_config.ok['model_version']).alias('model_version'),pl.lit(model_config.ok['checkpoint_sha256']).alias('checkpoint_sha256'),
                pl.lit(model_config.ok['feature_definition']).alias('feature_definition')).write_csv().encode(),
                filename='tabpfn_predictions.csv',label='Download TabPFN predictions') if tab_prediction.is_ok() else mo.md(''),
            mo.hstack([
                mo.download(lambda: measurement_result.ok.write_csv().encode(), filename='selected_measurements.csv',label='Download observations'),
                mo.download(lambda: dose_figure.to_html(include_plotlyjs=True).encode(),filename='dose_response.html',label='Download interactive figure'),
            ],wrap=True),
            mo.accordion({'Fit interpretation and uncertainty':mo.md(
                'EC50 is the midpoint of the fitted response span. The shaded EC20 to EC80 range describes 20% to 80% inhibition of that span. '
                'It is not an uncertainty interval. Confidence intervals are unavailable because independent replicate units have not been established. '
                + ('Fit cautions: '+ '; '.join(hill_fit.diagnostics.get('flags',[])) if hill_fit.diagnostics.get('flags') else '')
            )}),
        ],gap=1.25)
    drc_view = mo.vstack([
        mo.md('## Drug Explorer'),
        mo.md('Inspect a single PRISM compound, cell line and screen. Source screens remain separate.'),
        ui.controls([compound_select,cell_select,screen_select]),
        dose_content,
    ],gap=1.25)
    return dose_figure, drc_view


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
def _(microscopy, mo, provenance, Path, get_map_selection, pl):
    image_identity = tuple((str(p), provenance.sha256(p)) for p in [Path('data/bbbc021/metadata.csv')] if p.is_file())
    @mo.cache
    def read_image_metadata(identity):
        return microscopy.load_metadata()
    image_metadata = read_image_metadata(image_identity)
    imaging_options = sorted(image_metadata.ok['Image_Metadata_Compound'].unique().to_list()) if image_metadata.is_ok() else []
    mapped_field = image_metadata.ok.filter(pl.col('image_field_id') == get_map_selection()) if image_metadata.is_ok() and get_map_selection() else None
    mapped_row = mapped_field.row(0,named=True) if mapped_field is not None and mapped_field.height == 1 else None
    imaging_compound = mo.ui.dropdown(imaging_options, value=mapped_row['Image_Metadata_Compound'] if mapped_row else ('taxol' if 'taxol' in imaging_options else next(iter(imaging_options), None)), label='Imaging compound', full_width=True)
    return image_metadata, imaging_compound, mapped_row


@app.cell
def _(image_metadata, imaging_compound, mo, pl, mapped_row):
    image_doses = sorted(image_metadata.ok.filter(pl.col('Image_Metadata_Compound') == imaging_compound.value)['Image_Metadata_Concentration'].drop_nulls().unique().to_list()) if image_metadata.is_ok() else []
    imaging_dose = mo.ui.dropdown(image_doses, value=mapped_row['Image_Metadata_Concentration'] if mapped_row and mapped_row['Image_Metadata_Compound'] == imaging_compound.value else (image_doses[len(image_doses)//2] if image_doses else None), label='Concentration (μM)', full_width=True)
    return (imaging_dose,)


@app.cell
def _(image_metadata, imaging_compound, imaging_dose, mo, pl, mapped_row):
    field_rows = image_metadata.ok.filter((pl.col('Image_Metadata_Compound') == imaging_compound.value) & (pl.col('Image_Metadata_Concentration') == imaging_dose.value)) if image_metadata.is_ok() else None
    field_options = {f"Well {r['Image_Metadata_Well_DAPI']} · field {r['ImageNumber']} · replicate {r['Replicate']}":r['image_field_id'] for r in field_rows.iter_rows(named=True)} if field_rows is not None else {}
    default_field = next((label for label,value in field_options.items() if mapped_row and value == mapped_row['image_field_id']),next(iter(field_options),None))
    field_select = mo.ui.dropdown(field_options, value=default_field, label='Image field', full_width=True)
    return (field_select,)


@app.cell
def _(Error, image_projection, image_metadata, mo):
    @mo.cache
    def read_projection(metadata):
        return image_projection.load_projection(metadata)
    projection = read_projection(image_metadata.ok) if image_metadata.is_ok() else Error(image_metadata.error)
    return (projection,)


@app.cell
def _(components, field_select, image_metadata, imaging_compound, imaging_dose, microscopy, mo, pl, projection, ui, map_color, set_map_selection):
    image_parts = []
    map_plot = None
    if image_metadata.is_error():
        image_parts.append(ui.unavailable('Microscopy unavailable',image_metadata.error))
    elif field_select.value:
        treated = image_metadata.ok.filter(pl.col('image_field_id') == field_select.value).row(0,named=True)
        rgb = microscopy.load_microscopy_rgb('data/bbbc021/plate.zip',treated)
        condition_label = 'Vehicle control (DMSO)' if treated['Image_Metadata_Compound'] == 'DMSO' else f"{treated['Image_Metadata_Compound']} at {treated['Image_Metadata_Concentration']:g} μM"
        treated_panel = mo.vstack([mo.md(f'### {condition_label}'),mo.image(rgb.ok,alt=condition_label) if rgb.is_ok() else mo.md(rgb.error),
            mo.md(f"Well {treated['Image_Metadata_Well_DAPI']}, field {treated['ImageNumber']}")])
        control = microscopy.select_control(image_metadata.ok,treated)
        if control.is_ok():
            control_rgb = microscopy.load_microscopy_rgb('data/bbbc021/plate.zip',control.ok)
            control_panel = mo.vstack([mo.md('### Vehicle control (DMSO)'),mo.image(control_rgb.ok,alt='Vehicle control DMSO') if control_rgb.is_ok() else mo.md(control_rgb.error),
                mo.md(f"Well {control.ok['Image_Metadata_Well_DAPI']}, field {control.ok['ImageNumber']}")])
        else:
            control_panel = ui.unavailable('Control unavailable',control.error)
        image_parts.append(mo.Html(f'<div class="atlas-compare"><div>{control_panel.text}</div><div>{treated_panel.text}</div></div>'))
        image_parts.append(mo.md('Control and treated fields share a plate and replicate group. Further acquisition matching has not been verified. RGB contrast is adjusted independently for display.'))
    else:
        image_parts.append(ui.unavailable('No field selected','Choose an available imaging compound and concentration.'))
    if projection.is_error():
        image_parts.append(ui.unavailable('Embedding map unavailable',projection.error))
    else:
        _map_figure = ui.morphology_figure(projection.ok,imaging_compound.value,imaging_dose.value,map_color.value)
        map_plot = mo.ui.plotly(_map_figure,config=ui.PLOT_CONFIG,
            on_change=lambda points: set_map_selection(field) if (field := ui.selected_image_field(points,_map_figure)) else None)
        image_parts.extend([mo.md('### Image-field morphology map'),map_color,
            map_plot.style({'display':'grid','height':'600px'}),
            mo.md(f"{projection.ok.height} image fields. Select a point to inspect its microscopy field. DINO ViT-S/8 uses ImageNet weights on fluorescence composites; this exploratory baseline is not Cell-DINO or an individual-cell measurement. "
                '[Cell-DINO weights](https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/) and compatible channel preprocessing are still required.'),
            mo.download(lambda: projection.ok.select('image_field_id','Image_Metadata_Compound','Image_Metadata_Concentration','umap_1','umap_2','model_id').write_csv().encode(),filename='image_field_projection.csv',label='Download map coordinates'),
        ])
    image_field_view = mo.vstack(image_parts,gap=1.25)
    return image_field_view, map_plot


@app.cell
def _(mo, ui, imaging_compound, imaging_dose, field_select, single_cell_view, image_field_view):
    imaging_view = mo.vstack([
        mo.md('## Cell Explorer'),
        mo.md('Compare BBBC021 microscopy fields from MCF-7 cells. These imaging conditions are separate from the PRISM viability screen.'),
        ui.controls([imaging_compound,imaging_dose,field_select]),
        single_cell_view,
        mo.accordion({'Image-field analysis':image_field_view}),
        mo.accordion({'Channels and map interpretation':mo.md('Red: actin. Green: tubulin. Blue: DAPI. Raw image intensities are unchanged. '
            'UMAP can distort distances and global geometry. Lines joining dose-group centroids do not represent measured time trajectories; clusters alone do not establish mechanisms.')}),
    ],gap=1.25)
    return (imaging_view,)


@app.cell
def _(validation, external_evidence):
    external_result = validation.load_configured_validation()
    external_inventory = external_evidence.source_inventory()
    return external_result, external_inventory


@app.cell
def _(Path, benchmark, components, evaluation, external_result, external_inventory, mo, validation, pl, ui, run_benchmark, compound_select, cell_select, screen_select, methods_view, saved_benchmark):
    validation_parts = [mo.md('## Validation'),mo.md('Evaluate prediction error within a measured curve, and inspect the scope and provenance of each comparison.')]
    current_scope = f'{compound_select.value} / {cell_select.value} / {screen_select.value}'
    validation_parts.extend([mo.md('### Held-out concentration prediction'),mo.md(f'Current Drug Explorer condition: `{current_scope}`.'),run_benchmark])
    if run_benchmark.value and evaluation.is_error():
        validation_parts.append(ui.unavailable('Current evaluation unavailable',evaluation.error))
    if evaluation.is_ok():
        metrics = benchmark.aggregate_predictions(evaluation.ok)
        evaluated_rows = evaluation.ok
        benchmark_scope = 'Current condition'
    elif saved_benchmark.is_ok():
        metrics,evaluated_rows,recorded_manifest = saved_benchmark.ok
        recorded_config = recorded_manifest['config']
        benchmark_scope = f"Recorded condition: {recorded_config['compound_id']} / {recorded_config['cell_line_id']} / {recorded_config['screen_id']}"
    else:
        metrics = None
        evaluated_rows = None
        benchmark_scope = 'No verified benchmark is available. Evaluate the selected condition to calculate held-out prediction error.'
    validation_parts.append(mo.md(benchmark_scope))
    if metrics is not None:
        validation_parts.extend([
            mo.ui.plotly(ui.figure(components.build_generalization_plot(metrics)),config=ui.PLOT_CONFIG),
            mo.md(f"Common RMSE uses {metrics['common_observations'].min()} observations successfully predicted by every evaluated model. This small sample does not establish general model performance."),
            mo.accordion({'Model coverage and error':mo.ui.table(metrics.with_columns(pl.col('rmse','common_rmse').cast(pl.String).fill_null('Not evaluated')),selection=None,show_data_types=False,show_column_summaries=False)}),
            mo.download(lambda: evaluated_rows.write_csv().encode(),filename='held_out_predictions.csv',label='Download predictions and failures'),
        ])
    validation_parts.extend([mo.md('### External dataset comparison'),
        ui.unavailable('Cross-dataset comparison unavailable',external_result.error+' Reviewed compound/cell-line crosswalks and comparable potency definitions are required. Correlations, intervals and replication classifications have not been calculated.') if external_result.is_error() else mo.md('Verified source measurements are available.'),
        mo.accordion({'External source records and blockers':mo.vstack([
            mo.ui.table(external_inventory,selection=None,show_data_types=False,show_column_summaries=False),
            mo.md('[GDSC release 8.2 archive](https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.2/) · '
                '[GDSC assay and metric definitions](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/). '
                'GDSC natural-log IC50 uses μM. PRISM EC50 describes a fitted response midpoint; equal potency definitions have not been established. CTRPv2 processing provenance remains unresolved.'),
            mo.download(lambda: external_inventory.write_csv().encode(),filename='external_source_inventory.csv',label='Download external source inventory'),
        ])}),
    ])
    if external_result.is_ok():
        matched,external_tables,external_counts,external_config = external_result.ok
        validation_parts.extend([
            mo.md(f'{matched.height:,} matched conditions with identical potency definitions.'),
            mo.ui.plotly(ui.figure(components.build_validation_plot(matched,{'reference':external_config['reference']['source']['dataset'],'external':external_config['external']['source']['dataset']})),config=ui.PLOT_CONFIG),
            mo.accordion({'Comparison methods and counts':mo.vstack([mo.ui.table(pl.DataFrame(external_counts),selection=None),mo.ui.table(validation.compare_potencies(matched).ok,selection=None)])}),
            mo.download(lambda: matched.write_csv().encode(),filename='matched_external_measurements.csv',label='Download matched measurements'),
        ])
    validation_parts.extend([
        mo.md('### Reproducibility'),
        mo.md('Raw observations remain unchanged. Reported results carry source hashes, model identity, analysis settings and fold-level predictions.'),
        mo.download(lambda: Path('METHODS.md').read_bytes(),filename='methods.md',label='Download methods and citations'),
        mo.accordion({'Methods and citations':methods_view,'Evaluation scope':mo.md('All measurements at a held-out concentration stay in the same fold. Fits use training observations only. '
            "Individual RMSE uses each model's successful predictions; common RMSE compares the same successful observations across all models. "
            'Failures remain counted. These results concern concentrations within an observed curve; they do not measure unseen compound and cell-line combinations.')}),
    ])
    validation_view = mo.vstack(validation_parts,gap=1.25)
    return (validation_view,)


@app.cell
def _(Path, mo, provenance):
    methods_view = mo.md(Path('METHODS.md').read_text())
    return (methods_view,)


@app.cell
def _(drc_view, get_tab, imaging_view, mo, overview_view, set_tab, validation_view):
    public_tabs = mo.ui.tabs({'Discoveries':overview_view,'Drug Explorer':drc_view,'Cell Explorer':imaging_view,'Validation':validation_view},value=get_tab(),on_change=set_tab,lazy=False)
    application_view = mo.vstack([
        mo.Html('<header class="atlas-header"><h1>Phenotypic Drug-Response Atlas</h1><p>Explore drug sensitivity and cell morphology, with measurements and model predictions kept in view.</p></header>'),
        public_tabs,
    ],gap=1.5)
    application_view
    return application_view, public_tabs


@app.cell
def _(mo, ui):
    @mo.cache
    def read_recorded_benchmark():
        return ui.audited_benchmark()
    saved_benchmark = read_recorded_benchmark()
    return (saved_benchmark,)


@app.cell
def _(mo):
    get_tab, set_tab = mo.state('Discoveries')
    return get_tab, set_tab


@app.cell
def _(mo):
    get_map_selection, set_map_selection = mo.state(None)
    map_color = mo.ui.dropdown(['Concentration','Compound'],value='Concentration',label='Map color')
    return get_map_selection, set_map_selection, map_color


@app.cell
def _(Path, mo, provenance, single_cells):
    _root = str(mo.cli_args().get('cell-atlas') or 'data/bbbc021/single_cells/atlas-v1')
    _path = Path(_root)/'manifest.json'
    cell_atlas_identity = provenance.sha256(_path) if _path.is_file() else None
    @mo.cache
    def read_cell_atlas(identity, root):
        return single_cells.load_atlas(root)
    cell_atlas = read_cell_atlas(cell_atlas_identity,_root)
    return cell_atlas, cell_atlas_identity


@app.cell
def _(mo):
    get_cell_selection, set_cell_selection = mo.state(None)
    cell_color = mo.ui.dropdown(['Compound','Dose','Cell area','Nuclear fraction'],value='Compound',label='Color cells by',full_width=True)
    return get_cell_selection, set_cell_selection, cell_color


@app.cell
def _(cell_atlas, field_select, get_cell_selection, mo, pl):
    _cells = cell_atlas.ok['cells'].filter(pl.col('image_field_id')==field_select.value).sort('mask_label') if cell_atlas.is_ok() else pl.DataFrame()
    _options = {f"Cell {row['mask_label']} · {row['area_px']:,} px²":row['cell_id'] for row in _cells.iter_rows(named=True)}
    _default = next((label for label,value in _options.items() if value==get_cell_selection()),next(iter(_options),None))
    cell_selector = mo.ui.dropdown(_options,value=_default,label='Cell in selected field',searchable=True,full_width=True)
    return (cell_selector,)


@app.cell
def _(cell_atlas, cell_selector, cell_color, imaging_compound, imaging_dose, mo, pl, ui, single_cell_ui, set_cell_selection, set_map_selection):
    cell_plot = None
    if cell_atlas.is_ok():
        _data = cell_atlas.ok['cells']
        _figure = single_cell_ui.cell_figure(_data,imaging_compound.value,imaging_dose.value,cell_selector.value,cell_color.value)
        def _choose_cell(points):
            _id = ui.selected_image_field(points,_figure)
            if _id:
                _row = _data.filter(pl.col('cell_id')==_id)
                if _row.height==1:
                    set_cell_selection(_id)
                    set_map_selection(_row['image_field_id'][0])
        cell_plot = mo.ui.plotly(_figure,config=ui.PLOT_CONFIG,on_change=_choose_cell)
    return (cell_plot,)


@app.cell
def _(cell_atlas, cell_selector, field_select, single_cell_ui, mo):
    cell_detail_view = single_cell_ui.cell_view(cell_atlas.ok,cell_selector.value,
        cell_atlas.ok['manifest']['inputs']['data/bbbc021/metadata.csv'],field_select.value) if cell_atlas.is_ok() else mo.md('')
    return (cell_detail_view,)


@app.cell
def _(cell_atlas, imaging_compound, single_cell_ui, mo, pl, ui):
    cell_area_plot, cell_change_plot, cell_well_table = None,None,None
    cell_population_view = mo.md('')
    if cell_atlas.is_ok():
        _atlas = cell_atlas.ok
        _area,_change,_wells,_changes = single_cell_ui.population_figures(_atlas,imaging_compound.value)
        cell_area_plot = mo.ui.plotly(_area,config=ui.PLOT_CONFIG)
        cell_change_plot = mo.ui.plotly(_change,config=ui.PLOT_CONFIG)
        cell_well_table = mo.ui.table(_wells,selection=None,show_data_types=False,show_column_summaries=False)
        _lead = _changes.filter(pl.col('cosine_change').is_finite()).sort('cosine_change',descending=True).head(1)
        _lead_text = (f"Largest adjacent-dose difference: {_lead['lower_dose_um'][0]:g} to {_lead['upper_dose_um'][0]:g} μM "
            f"(cosine distance {_lead['cosine_change'][0]:.4f}). This is a descriptive candidate interval for follow-up, not a confirmed biological transition.") if _lead.height else 'No adjacent-dose comparison is available for this condition.'
        cell_population_view = mo.vstack([
            mo.md('### Dose-dependent morphology'),
            mo.md(_lead_text),
            mo.hstack([cell_area_plot.style({'display':'grid','height':'510px'}),cell_change_plot.style({'display':'grid','height':'510px'})],widths=[1,1],gap=2),
            mo.md('Boxes describe cell distributions, not uncertainty. Profile differences average fields and wells equally. Cells are not independent biological replicates; no significance test is reported.'),
            mo.accordion({'Sampled wells and cell counts':cell_well_table}),
            mo.download(lambda:_wells.write_csv().encode(),filename='cell_well_summaries.csv',label='Download well summaries'),
        ],gap=1.25)
    return cell_population_view, cell_area_plot, cell_change_plot, cell_well_table


@app.cell
def _(cell_atlas, cell_atlas_identity, cell_selector, cell_color, cell_plot, cell_detail_view, cell_population_view, single_cell_ui, mo, ui, pl):
    if cell_atlas.is_error():
        single_cell_view = ui.unavailable('Individual-cell analysis unavailable',cell_atlas.error)
    else:
        _atlas = cell_atlas.ok
        single_cell_view = mo.vstack([
            mo.md('## Individual-cell atlas'),
            mo.md(f"{_atlas['cells'].height:,} retained cell crops from {_atlas['manifest']['fields']} sampled fields. Each point links to its source image and mask. "
                  'Generic DINO ImageNet features describe predicted cell crops; they do not establish cell identities or mechanisms.'),
            ui.controls([cell_selector,cell_color]),
            cell_plot.style({'display':'grid','height':'600px'}),
            cell_detail_view,
            cell_population_view,
            mo.download(lambda:_atlas['cells'].with_columns(pl.lit(cell_atlas_identity).alias('atlas_manifest_sha256')).write_csv().encode(),filename='cell_atlas.csv',label='Download cell atlas table'),
            single_cell_ui.methods(_atlas),
        ],gap=1.25)
    return (single_cell_view,)


if __name__ == '__main__':
    app.run()
