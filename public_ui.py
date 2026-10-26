import json
from html import escape
from pathlib import Path

import marimo as mo
import plotly.graph_objects as go
import polars as pl
from expression import Error, Ok

from analysis import benchmark, provenance

COLORS = {'measured':'#176b62', 'prediction':'#315bbb', 'range':'#b66b13', 'ink':'#163c3b'}
PLOT_CONFIG = {'responsive':True, 'displaylogo':False, 'toImageButtonOptions':{'format':'svg','filename':'atlas_figure','width':1200,'height':750,'scale':1}}


def metrics(items):
    return mo.Html('<div class="atlas-metrics">' + ''.join(f'<div><strong>{escape(str(value))}</strong><span>{escape(label)}</span></div>' for label,value in items) + '</div>')


def controls(items):
    return mo.Html('<div class="atlas-controls">' + ''.join(item.text for item in items) + '</div>')


def unavailable(title, text):
    return mo.Html(f'<section class="atlas-empty"><h3>{escape(title)}</h3><p>{escape(text)}</p></section>')


def figure(source):
    fig = go.Figure(source)
    fig.update_layout(template='plotly_white', height=510, autosize=True,
                      font=dict(family='PT Sans, system-ui, sans-serif',size=14,color=COLORS['ink']),
                      paper_bgcolor='white',plot_bgcolor='white',margin=dict(l=60,r=24,t=65,b=95),
                      title=dict(font=dict(size=20)),legend=dict(orientation='h',y=-0.2,x=0,font=dict(size=12)))
    for trace in fig.data:
        if trace.type == 'bar':
            trace.marker.color = [COLORS['prediction'] if 'TabPFN' in str(label) else COLORS['measured'] if 'Hill' in str(label) else '#738984' for label in trace.x]
    fig.update_xaxes(gridcolor='#e8eeeb',zeroline=False,automargin=True)
    fig.update_yaxes(gridcolor='#e8eeeb',zeroline=False,automargin=True)
    for axis in ('xaxis', 'yaxis'):
        fig.layout.template.layout[axis].update(fig.layout[axis].to_plotly_json())
    return fig


def dose_figure(source):
    fig = figure(source)
    for trace in fig.data:
        if trace.name == 'Measured relative viability':
            trace.update(name='Measured viability',marker=dict(color=COLORS['measured'],size=9,line=dict(color='white',width=1)))
        elif trace.name == 'Hill LL4 fit':
            trace.update(line=dict(color=COLORS['measured'],width=3))
        else:
            trace.update(line=dict(color=COLORS['prediction'],width=2.5,dash='dash'))
    selected = next((a.text for a in fig.layout.annotations if a.text.startswith('Selected')), None)
    fig.layout.annotations = ()
    if selected:
        fig.add_annotation(x=1,y=1.02,xref='paper',yref='paper',xanchor='right',yanchor='bottom',text=selected,showarrow=False,font=dict(size=12))
    title = fig.layout.title.text.split(' · ',1)
    fig.update_layout(title=dict(text=title[0] + ('<br><sup>'+title[1]+'</sup>' if len(title)>1 else ''),font=dict(size=18)),margin=dict(l=55,r=20,t=95,b=120))
    for shape in fig.layout.shapes or ():
        if shape.type == 'rect':
            shape.update(fillcolor=COLORS['range'],showlegend=True,name='Fitted effect span')
    return fig


def candidate_figure(catalog):
    data = catalog.head(100)
    fig = go.Figure(go.Scatter(x=data['ec50'].to_list(),y=data['slope'].abs().to_list(),mode='markers',
        marker=dict(size=9,color=data['r2'].to_list(),colorscale=[[0,'#b6d7c9'],[1,'#176b62']],showscale=True,colorbar=dict(title='R²',thickness=10)),
        customdata=data.select('name','ccle_name','r2','Screening Score').rows(),
        hovertemplate='%{customdata[0]} · %{customdata[1]}<br>EC50 %{x:.4g} μM<br>|Slope| %{y:.3g}<br>R² %{customdata[2]:.3f}<br>Screening score %{customdata[3]:.3f}<extra></extra>'))
    fig.update_layout(title='Potency and steepness',xaxis_type='log',xaxis_title='EC50 (μM)',yaxis_title='Absolute fitted Hill slope',showlegend=False)
    return figure(fig)


def morphology_figure(data, compound, dose, color_by='Concentration'):
    from components import build_umap_plot
    fig = build_umap_plot(data, compound, dose)
    if color_by == 'Compound':
        fig.data = fig.data[2:]
        for group in data.partition_by('Image_Metadata_Compound', maintain_order=True):
            fig.add_scatter(x=group['umap_1'],y=group['umap_2'],mode='markers',name=group['Image_Metadata_Compound'][0])
    else:
        fig.data[1].marker.update(showscale=True,colorbar=dict(title='μM',thickness=12))
    for trace in fig.data:
        if trace.name == 'Dose-ordered condition centroids':
            continue
        if trace.name == 'Other conditions':
            group = data.filter(pl.col('Image_Metadata_Compound') != compound)
        elif trace.name.startswith('All fields at'):
            group = data.filter(pl.col('Image_Metadata_Compound') == compound)
            nearest = min(group['Image_Metadata_Concentration'].unique(),key=lambda value:abs(value-dose))
            group = group.filter(pl.col('Image_Metadata_Concentration') == nearest)
        else:
            group = data.filter(pl.col('Image_Metadata_Compound') == trace.name)
        trace.update(customdata=group.select('image_field_id','Image_Metadata_Compound','Image_Metadata_Concentration').rows(),
                     hovertemplate='%{customdata[1]} · %{customdata[2]:g} μM<br>%{customdata[0]}<extra></extra>')
    fig.update_layout(clickmode='event+select')
    fig = figure(fig)
    title = fig.layout.title.text
    if ' ImageNet baseline · ' in title:
        title = title.replace(' ImageNet baseline · ','<br><sup>ImageNet baseline · ')+'</sup>'
    fig.update_layout(title=dict(text=title,font=dict(size=18)),height=600,margin=dict(t=95,b=145))
    return fig


def selected_image_field(points, plot):
    if not points:
        return None
    curve = points[-1].get('curveNumber')
    index = points[-1].get('pointIndex',points[-1].get('pointNumber'))
    if not isinstance(curve,int) or not isinstance(index,int) or not 0 <= curve < len(plot.data):
        return None
    fields = plot.data[curve].customdata
    return fields[index][0] if fields is not None and 0 <= index < len(fields) else None


def audited_benchmark(root='reports/runs/step18-1-2026-10-26'):
    root = Path(root)
    required = ['manifest.json','outputs.json','model_benchmark.csv','held_out_predictions.csv']
    if any(not (root/name).is_file() for name in required):
        return Error('No traceable benchmark is available.')
    metadata = json.loads((root/'manifest.json').read_text())
    checked = provenance.verify_manifest(metadata,list(metadata.get('inputs',{})),metadata.get('config'))
    if checked.is_error():
        return checked
    hashes = json.loads((root/'outputs.json').read_text())
    for name in required[2:]:
        if provenance.sha256(root/name) != hashes.get(name):
            return Error('Benchmark artifact hash mismatch')
    rows = pl.read_csv(root/'held_out_predictions.csv')
    table = pl.read_csv(root/'model_benchmark.csv',schema_overrides={'model_version':pl.String,'package_version':pl.String,'checkpoint_sha256':pl.String})
    if not benchmark.aggregate_predictions(rows).equals(table):
        return Error('Benchmark metrics do not match the saved predictions')
    return Ok((table,rows,metadata))
