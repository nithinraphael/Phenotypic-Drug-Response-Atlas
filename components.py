import numpy as np
import plotly.graph_objects as go
import polars as pl

from analysis.fitting import fit_hill
from analysis.microscopy import load_microscopy_rgb
from analysis.models import predict_tabpfn


def build_drc_plot(doses, viabs, grid_x, fit, pred_tabpfn=None, active_dose=None, drug='', cell_line='', model_label=None):
    fig = go.Figure()
    fig.add_scatter(x=doses, y=viabs, mode='markers', name='Measured relative viability')
    if fit.status == 'ok':
        predicted = fit.predict(grid_x)
        if predicted.is_ok():
            fig.add_scatter(x=grid_x, y=predicted.ok, mode='lines', name='Hill LL4 fit')
        fig.add_vline(x=fit.ec50, line_dash='dash', annotation_text=f'EC50 {fit.ec50:.4g} μM')
        if fit.ec20 is not None and fit.ec80 is not None:
            fig.add_vrect(x0=fit.ec20, x1=fit.ec80, opacity=0.12, fillcolor='orange', line_width=0,
                          annotation_text='EC20–EC80, fitted effect span')
    if pred_tabpfn is not None:
        fig.add_scatter(x=grid_x, y=pred_tabpfn, mode='lines', name=model_label)
    if active_dose is not None:
        fig.add_vline(x=active_dose, line_dash='dot', annotation_text=f'Selected {active_dose:.4g} μM')
    fig.update_layout(title=f'{drug} · {cell_line}', template='plotly_white', height=430,
                      xaxis_title='Concentration (μM)', xaxis_type='log', yaxis_title='Relative viability',
                      legend=dict(orientation='h', y=-0.25))
    return fig


def build_umap_plot(data, target_compound, active_dose=None, model_label='Verified image representation'):
    if 'model_id' in data.columns:
        model_label = data['model_id'].unique().item()
    fig = go.Figure()
    bg = data.filter(pl.col('Image_Metadata_Compound') != target_compound)
    fg = data.filter(pl.col('Image_Metadata_Compound') == target_compound)
    fig.add_scatter(x=bg['umap_1'], y=bg['umap_2'], mode='markers', name='Other conditions', marker=dict(color='lightgray'))
    fig.add_scatter(x=fg['umap_1'], y=fg['umap_2'], mode='markers', name=target_compound,
                    marker=dict(color=fg['Image_Metadata_Concentration'].to_list(), colorscale='Viridis'))
    centroids = fg.group_by('Image_Metadata_Concentration').agg(pl.col('umap_1','umap_2').mean()).sort('Image_Metadata_Concentration')
    fig.add_scatter(x=centroids['umap_1'], y=centroids['umap_2'], mode='lines+markers', name='Dose-ordered condition centroids')
    if active_dose is not None and fg.height:
        avail = fg['Image_Metadata_Concentration'].unique().drop_nulls().to_numpy()
        if len(avail):
            nearest = float(avail[np.argmin(np.abs(avail-active_dose))])
            selected = fg.filter(pl.col('Image_Metadata_Concentration') == nearest)
            fig.add_scatter(x=selected['umap_1'], y=selected['umap_2'], mode='markers',
                            marker=dict(symbol='star',size=14),name=f'All fields at {nearest:g} μM (requested {active_dose:g})')
    fig.update_layout(title=f'{model_label} · {target_compound}', template='plotly_white',
                      xaxis_title='UMAP 1',yaxis_title='UMAP 2', height=430)
    return fig


def build_generalization_plot(results):
    fig = go.Figure()
    fig.add_bar(x=results['model'].to_list(), y=results['common_rmse'].to_list(), name='Common held-out observations')
    fig.update_layout(title='Held-out concentration prediction',yaxis_title='Pooled observation RMSE',template='plotly_white')
    return fig


def build_validation_plot(matched_data, comparison_config):
    fig = go.Figure()
    for metric in matched_data['potency_metric'].unique().sort():
        sub = matched_data.filter(pl.col('potency_metric') == metric)
        fig.add_scatter(x=sub['potency_value_um'].to_list(), y=sub['potency_value_um_external'].to_list(),
                        mode='markers', name=metric, text=sub['compound_id'].to_list())
    fig.update_layout(template='plotly_white', xaxis_type='log',yaxis_type='log',
                      xaxis_title=f"{comparison_config['reference']} potency (μM)",
                      yaxis_title=f"{comparison_config['external']} potency (μM)")
    return fig
