import io
import json

import marimo as mo
import numpy as np
import plotly.graph_objects as go
import polars as pl
import tifffile
from scipy import ndimage as ndi

import public_ui as ui
import single_cells


def cell_figure(cells, compound, dose, selected, color):
    figure = go.Figure()
    groups = cells.partition_by('compound',maintain_order=True) if color == 'Compound' else [cells.filter(pl.col('compound')!=compound),cells.filter(pl.col('compound')==compound)]
    for group in groups:
        if group.is_empty():
            continue
        foreground = group['compound'].n_unique()==1 and group['compound'][0]==compound
        marker = dict(size=5,opacity=.65)
        if color != 'Compound':
            column = {'Dose':'dose_um','Cell area':'area_px','Nuclear fraction':'nuclear_fraction'}[color]
            marker.update(color=group[column].to_list() if foreground else '#bcc9c5',colorscale='Viridis',showscale=foreground,
                          colorbar=dict(title={'Dose':'μM','Cell area':'px²','Nuclear fraction':'Fraction'}[color],thickness=12))
        info = [(*row,'Vehicle control (DMSO)' if row[1]=='DMSO' else f'{row[1]} · {row[2]:g} μM') for row in group.select('cell_id','compound','dose_um','well_id','mask_label','area_px').rows()]
        figure.add_trace(go.Scattergl(x=group['umap_1'],y=group['umap_2'],mode='markers',name=group['compound'][0] if color=='Compound' or foreground else 'Other compounds',
            marker=marker,customdata=info,
            hovertemplate='%{customdata[6]}<br>Well %{customdata[3]} · cell %{customdata[4]}<br>Area %{customdata[5]:,.0f} px²<extra></extra>'))
    highlighted = cells.filter((pl.col('compound')==compound)&(pl.col('dose_um')==dose))
    figure.add_trace(go.Scattergl(x=highlighted['umap_1'],y=highlighted['umap_2'],mode='markers',name='Selected condition',
        marker=dict(symbol='circle-open',size=8,color='#b66b13',line=dict(width=1)),customdata=highlighted.select('cell_id').rows(),hoverinfo='skip'))
    cell = cells.filter(pl.col('cell_id')==selected) if selected else cells.head(0)
    if cell.height:
        figure.add_trace(go.Scattergl(x=cell['umap_1'],y=cell['umap_2'],mode='markers',name='Selected cell',marker=dict(symbol='diamond',size=14,color='#b66b13',line=dict(color='white',width=1)),customdata=cell.select('cell_id').rows(),hoverinfo='skip'))
    figure.update_layout(title='Cell crops · DINO ViT-S/8<br><sup>Shared UMAP · ImageNet baseline</sup>',xaxis_title='UMAP 1',yaxis_title='UMAP 2',clickmode='event+select')
    return ui.figure(figure).update_layout(height=600,margin=dict(t=90,b=120))


def population_figures(atlas, compound):
    cells = atlas['cells'].filter(pl.col('compound')==compound)
    wells = atlas['wells'].filter(pl.col('compound')==compound).sort('dose_um','well_id')
    changes = atlas['changes'].filter(pl.col('compound')==compound).sort('upper_dose_um')
    distribution = go.Figure()
    for group in cells.sort('dose_um').partition_by('dose_um',maintain_order=True):
        dose = group['dose_um'][0]
        distribution.add_box(y=group['area_px'].to_list(),name='Vehicle control' if compound=='DMSO' else f'{dose:g} μM',boxpoints=False,marker_color=ui.COLORS['measured'],
                             hovertemplate='Area %{y:,.0f} px²<extra></extra>')
    distribution.update_layout(title='Retained-cell area distributions',xaxis_title='Measured concentration',yaxis_title='Cell area (px²)',showlegend=False)
    change = go.Figure()
    change.add_scatter(x=changes['upper_dose_um'].to_list(),y=changes['cosine_change'].to_list(),mode='lines+markers',line=dict(color=ui.COLORS['range']),
                       customdata=changes.select('lower_dose_um','upper_dose_um','lower_wells','upper_wells').rows(),
                       hovertemplate='%{customdata[0]:g} to %{customdata[1]:g} μM<br>Cosine change %{y:.4f}<br>Wells: %{customdata[2]} → %{customdata[3]}<extra></extra>')
    change.update_layout(title='Adjacent-dose profile differences',xaxis_title='Upper measured concentration (μM)',xaxis_type='log',yaxis_title='Cosine distance between well profiles',showlegend=False)
    if compound == 'DMSO':
        change.update_xaxes(type='linear')
    return ui.figure(distribution),ui.figure(change),wells,changes


@mo.cache
def field_images(image_field_id, metadata_identity, mask_file, cache_identity):
    from analysis.microscopy import load_metadata
    metadata = load_metadata().ok
    row = metadata.filter(pl.col('image_field_id')==image_field_id).row(0,named=True)
    raw = single_cells.read_field(row)
    with np.load(mask_file,allow_pickle=False) as labels:
        masks = labels['labels']
    return raw,masks


def crop_tiff(cell, padding=8):
    channels,mask = single_cells.read_crop(cell)
    buffer = io.BytesIO()
    tifffile.imwrite(buffer,channels,photometric='minisblack',metadata=dict(axes='CYX',cell_id=cell['cell_id'],image_field_id=cell['image_field_id'],
        channel_order=['DAPI','Tubulin','Actin'],cell_bbox_yxyx=[cell['min_y'],cell['min_x'],cell['max_y'],cell['max_x']],crop_padding_px=padding))
    return buffer.getvalue()


def cell_view(atlas, selected, metadata_identity, field_id=None):
    if not selected:
        field = atlas['fields'].filter(pl.col('image_field_id')==field_id)
        if field.height:
            return ui.unavailable('No retained cell crops',f"{field['cells'][0]} segmented objects; {field['retained'][0]} retained crops. Status: {field['status'][0]}. Inspect the original field below.")
        return ui.unavailable('No analyzed cell in this field','The current atlas samples the first field in each well. Choose a cell on the map, or select an analyzed image field. Image-level viewing remains available below.')
    cell = atlas['cells'].filter(pl.col('cell_id')==selected).row(0,named=True)
    raw,masks = field_images(cell['image_field_id'],metadata_identity,cell['mask_file'],atlas['manifest']['created_at'])
    crop,crop_mask = single_cells.read_crop(cell)
    full = single_cells.overlay(raw,masks,cell['mask_label'])
    thumbnail = single_cells.rgb(crop)
    thumbnail[crop_mask & ~ndi.binary_erosion(crop_mask)] = [255,215,55]
    field = atlas['fields'].filter(pl.col('image_field_id')==cell['image_field_id']).row(0,named=True)
    return mo.vstack([
        mo.md(f"### Cell {cell['mask_label']} · well {cell['well_id']} · {cell['compound']}"),
        mo.hstack([mo.vstack([mo.image(full,alt=f"Segmentation overlay for {cell['image_field_id']}, selected cell {cell['mask_label']}"),
                             mo.md('Gold: cell outlines. Cyan: predicted nuclei. Orange: selected cell.')]),
                   mo.vstack([mo.image(thumbnail,width=260,alt=f"Three-channel crop of cell {cell['mask_label']}"),
                              mo.md(f"Area **{cell['area_px']:,} px²** · centroid ({cell['x']:.1f}, {cell['y']:.1f}) px. "
                                    f"Predicted nuclei with majority overlap: {cell['nuclear_count']}."),
                              mo.download(lambda:crop_tiff(cell,atlas['manifest']['config']['core']['settings']['crop_padding_px']),filename=f"cell_{cell['mask_label']}_channels.tif",label='Download raw cell channels')])],widths=[2,1],gap=2),
        ui.metrics([('Segmented objects',f"{field['cells']:,}"),('Retained cell crops',f"{field['retained']:,}"),
                    ('Objects touching edge',f"{field['border_objects']:,}"),('Without nuclear support',f"{field['non_nucleated_objects']:,}")]),
        mo.md('Masks are predictions; merged or disconnected regions can remain. Crops retain the original DAPI, tubulin and actin channels. Display contrast is adjusted; raw intensities are unchanged.'),
    ],gap=1.25)


def methods(atlas):
    manifest = atlas['manifest']
    return mo.accordion({'Cell analysis methods and exclusions':mo.vstack([
        mo.md(f"{manifest['cells']:,} retained cells from {manifest['fields']} of {manifest['available_fields']} available fields. "
              f"The first field identifier per well is sampled without inspecting its appearance. Boundary objects, areas below {manifest['config']['core']['settings']['minimum_area_px']} pixels and masks without a majority-overlapping predicted nucleus are excluded from embeddings; all exclusions are retained. "
              'These rules can underrepresent damaged or weakly stained cells. Counts describe sampled images and do not measure viability.'),
        mo.md('Cellpose-SAM 4.0.8 supplies cell masks after the InstanSeg pilot produced non-nuclear fragments. InstanSeg 0.1.1 supplies nuclear predictions for QC. '
              'Generic DINO ViT-S/8 ImageNet weights embed aspect-preserving, isolated cell crops. This is exploratory transfer, not Cell-DINO or a validated biological state classifier. '
              'UMAP uses cosine distance on normalized 384-dimensional features, 15 neighbors, min_dist 0.1 and seed 42. UMAP distances may be distorted; no temporal paths are inferred.'),
        mo.md('Profiles average cell embeddings within fields and weight fields/wells equally. Adjacent-dose cosine distances rank observed changes descriptively. '
              'Cells are nested within fields and wells. Biological independence is unverified, so no p-values, confidence intervals or biological tipping-point claims are reported. '
              'Physical pixel calibration is unavailable; areas and coordinates use pixels. Segmentation IoU/Dice accuracy cannot be established without matched manual labels.'),
        mo.md('[BBBC021](https://bbbc.broadinstitute.org/BBBC021) · [InstanSeg](https://github.com/instanseg/instanseg) · '
              '[Cellpose-SAM](https://github.com/MouseLand/cellpose) · [DINO](https://github.com/facebookresearch/dino). '
              'Pre-segmented Zenodo masks were inspected but no traceable mapping to this plate was established. CellProfiler was not executed as a baseline.'),
        mo.download(lambda:json.dumps(manifest,indent=2).encode(),filename='cell_atlas_manifest.json',label='Download cell-analysis provenance'),
    ])})
