import argparse
import hashlib
import importlib.metadata
import io
import json
import resource
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import tifffile
from expression import Error, Ok
from scipy import ndimage as ndi
from PIL import Image, ImageOps

from analysis import microscopy, provenance

ROOT = Path('data/bbbc021/single_cells')


def read_field(row):
    with zipfile.ZipFile('data/bbbc021/plate.zip') as archive:
        return np.stack([tifffile.imread(archive.open(f"{row['Image_Metadata_Plate_DAPI']}/{row['Image_FileName_'+channel]}"))
                         for channel in microscopy.CHANNELS])


def rgb(channels, limits=None):
    image = np.moveaxis(channels[[2,1,0]],0,-1).astype(float)
    low, high = np.percentile(image,[1,99.5],axis=(0,1)) if limits is None else limits
    return (255*np.clip((image-low)/np.maximum(high-low,1),0,1)).astype(np.uint8)


def describe_cells(cells, nuclei, channels, row, minimum_area=20):
    if cells.ndim != 2 or nuclei.shape != cells.shape or channels.shape != (3,*cells.shape):
        return Error('Cell/nuclear labels and three image channels must have identical spatial dimensions')
    if cells.dtype.kind not in 'iu' or nuclei.dtype.kind not in 'iu' or np.any(cells < 0) or np.any(nuclei < 0):
        return Error('Instance masks must contain nonnegative integer labels')
    records = []
    nucleus_area = np.bincount(nuclei.ravel())
    for label, bounds in enumerate(ndi.find_objects(cells),1):
        if bounds is None:
            continue
        yy, xx = bounds
        selected = cells[bounds] == label
        y, x = np.nonzero(selected)
        area = len(y)
        nuclear_ids, nuclear_pixels = np.unique(nuclei[bounds][selected],return_counts=True)
        nuclear_count = int(np.sum((nuclear_ids>0)&(nuclear_pixels>.5*nucleus_area[nuclear_ids])))
        border = yy.start == 0 or xx.start == 0 or yy.stop == cells.shape[0] or xx.stop == cells.shape[1]
        reasons = ['Touches field boundary'] if border else []
        if area < minimum_area:
            reasons.append(f'Area below {minimum_area} pixels')
        if not nuclear_count:
            reasons.append('No majority-overlapping predicted nucleus')
        records.append(dict(cell_id=f"{row['image_field_id']}::cell={label}", image_field_id=row['image_field_id'], mask_label=label,
                            compound=row['Image_Metadata_Compound'],dose_um=row['Image_Metadata_Concentration'],
                            plate_id=row['Image_Metadata_Plate_DAPI'],well_id=row['Image_Metadata_Well_DAPI'],replicate=row['Replicate'],
                            x=float(x.mean()+xx.start),y=float(y.mean()+yy.start),min_y=yy.start,min_x=xx.start,max_y=yy.stop,max_x=xx.stop,
                            area_px=area,nuclear_count=nuclear_count,nuclear_fraction=float(np.mean(nuclei[bounds][selected]>0)),
                            mean_dapi=float(channels[0][bounds][selected].mean()),mean_tubulin=float(channels[1][bounds][selected].mean()),
                            mean_actin=float(channels[2][bounds][selected].mean()),border=border,eligible=not reasons,
                            exclusion_reason='; '.join(reasons) or None))
    if not records:
        return Error('No cell instances detected; no substitute masks or crops generated')
    return Ok(pl.DataFrame(records,schema_overrides={'exclusion_reason':pl.String}))


def crop_cell(channels, mask, cell, padding=8):
    y0,x0 = max(0,cell['min_y']-padding),max(0,cell['min_x']-padding)
    y1,x1 = min(mask.shape[0],cell['max_y']+padding),min(mask.shape[1],cell['max_x']+padding)
    return channels[:,y0:y1,x0:x1],mask[y0:y1,x0:x1] == cell['mask_label'],(y0,x0,y1,x1)


def population_summary(cells, vectors):
    if cells.height != len(vectors) or vectors.ndim != 2 or not np.all(np.isfinite(vectors)):
        return Error('Population analysis requires one finite embedding per identified retained cell')
    keys = ['compound','dose_um','plate_id','well_id','replicate']
    indexed = cells.with_row_index('_position')
    summaries, profiles = [], []
    for group in indexed.partition_by(keys,maintain_order=True):
        record = group.select(keys).row(0,named=True)
        summaries.append(dict(**record,cells=group.height,fields=group['image_field_id'].n_unique(),
                              median_area_px=group['area_px'].median(),q25_area_px=group['area_px'].quantile(.25,interpolation='linear'),
                              q75_area_px=group['area_px'].quantile(.75,interpolation='linear'),
                              median_nuclear_fraction=group['nuclear_fraction'].median()))
        field_profiles = [vectors[field['_position'].to_numpy()].mean(axis=0) for field in group.partition_by('image_field_id')]
        profile = np.mean(field_profiles,axis=0)
        profiles.append(dict(**record,profile=profile))
    changes = []
    for compound in sorted(cells['compound'].unique()):
        if compound == 'DMSO':
            continue
        doses = sorted(cells.filter(pl.col('compound')==compound)['dose_um'].unique())
        means = [np.mean([r['profile'] for r in profiles if r['compound']==compound and r['dose_um']==dose],axis=0) for dose in doses]
        for index in range(1,len(doses)):
            a,b = means[index-1:index+1]
            denominator = np.linalg.norm(a)*np.linalg.norm(b)
            distance = float(np.clip(1-np.dot(a,b)/denominator,0,2)) if denominator > 0 else None
            changes.append(dict(compound=compound,lower_dose_um=doses[index-1],upper_dose_um=doses[index],
                                cosine_change=distance,lower_wells=sum(r['compound']==compound and r['dose_um']==doses[index-1] for r in profiles),
                                upper_wells=sum(r['compound']==compound and r['dose_um']==doses[index] for r in profiles)))
    schema = dict(compound=pl.String,lower_dose_um=pl.Float64,upper_dose_um=pl.Float64,cosine_change=pl.Float64,lower_wells=pl.Int64,upper_wells=pl.Int64)
    return Ok((pl.DataFrame(summaries),pl.DataFrame(changes,schema=schema)))


def load_atlas(root=ROOT/'atlas-v1'):
    root = Path(root)
    if not (root/'manifest.json').is_file():
        return Error('No verified cell atlas is available. Run the documented pretrained cell-analysis pipeline.')
    metadata = json.loads((root/'manifest.json').read_text())
    verified = provenance.verify_manifest(metadata,list(metadata.get('inputs',{})),metadata.get('config'))
    if verified.is_error():
        return verified
    for name,digest in metadata['outputs'].items():
        if not (root/name).is_file() or provenance.sha256(root/name) != digest:
            return Error(f'Cell atlas artifact missing or changed: {name}')
    cells = pl.read_parquet(root/'cells.parquet')
    with np.load(root/'embeddings.npz',allow_pickle=False) as arrays:
        ids,vectors,coordinates = arrays['cell_id'],arrays['embeddings'],arrays['coordinates']
    if ids.dtype.kind not in 'US' or cells.height != len(ids) or cells['cell_id'].n_unique() != cells.height or len(set(ids.tolist())) != len(ids) or set(ids.tolist()) != set(cells['cell_id']):
        return Error('Cell embedding identifiers do not match retained cell records')
    if vectors.dtype.kind not in 'f' or coordinates.dtype.kind not in 'f' or vectors.shape != (len(ids),384) or coordinates.shape != (len(ids),2) or not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(coordinates)):
        return Error('Invalid cell embedding or projection dimensions/values')
    positions = pl.DataFrame(dict(cell_id=ids.tolist(),umap_1=coordinates[:,0],umap_2=coordinates[:,1]))
    return Ok(dict(cells=cells.join(positions,on='cell_id',validate='1:1',maintain_order='left'),
                   fields=pl.read_parquet(root/'fields.parquet'),wells=pl.read_parquet(root/'wells.parquet'),
                   changes=pl.read_parquet(root/'changes.parquet'),manifest=metadata,root=root))


def read_crop(cell):
    with zipfile.ZipFile(cell['crop_archive']) as archive:
        with np.load(io.BytesIO(archive.read(f"{cell['mask_label']}.npz")),allow_pickle=False) as crop:
            return crop['channels'],crop['mask']


def overlay(channels, masks, label=None):
    image = rgb(channels)
    outlined = (image*.7).astype(np.uint8)
    for plane,color in [(masks[1],[255,215,55]),(masks[0],[40,220,255])]:
        outlined[(plane>0)&(plane!=ndi.minimum_filter(plane,3))] = color
    if label is not None:
        selected = masks[1] == label
        outlined[selected] = (.65*image[selected]+.35*np.array([255,100,30])).astype(np.uint8)
        outlined[selected & ~ndi.binary_erosion(selected)] = [255,245,140]
    return outlined


def generate(output, settings_path='config/single_cells.json', fields_per_well=1, device=None):
    import torch
    import torchvision.transforms as T
    import umap
    from instanseg import InstanSeg
    from cellpose.models import CellposeModel

    start = time.perf_counter()
    output = Path(output)
    if output.exists():
        return Error('Output already exists; use a new run directory. Verified field caches can be reused.')
    settings = json.loads(Path(settings_path).read_text())
    for path,digest in settings['verified_files'].items():
        if not Path(path).is_file() or provenance.sha256(path) != digest:
            return Error(f'Model/source missing or hash mismatch: {path}')
    metadata = microscopy.load_metadata()
    if metadata.is_error():
        return metadata
    selected = metadata.ok.sort('image_field_id').group_by('Image_Metadata_Plate_DAPI','Image_Metadata_Well_DAPI',maintain_order=True).head(fields_per_well)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        return Error('CUDA unavailable; run with --device cpu')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    nuclei_model = InstanSeg(torch.jit.load(settings['instanseg_checkpoint'],map_location='cpu'),device=device,verbosity=0,channels_last=False)
    nuclei_model.instanseg.eval()
    cells_model = CellposeModel(gpu=device=='cuda',device=torch.device(device),pretrained_model=settings['cellpose_checkpoint'],use_bfloat16=False)
    dino = json.loads(Path('config/dino_baseline.json').read_text())
    encoder = torch.hub.load(dino['source_directory'],'dino_vits8',source='local',pretrained=False)
    encoder.load_state_dict(torch.load(dino['checkpoint_path'],map_location='cpu',weights_only=True),strict=True)
    encoder.eval().to(device)
    transform = T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    inputs = ['data/bbbc021/metadata.csv','data/bbbc021/plate.zip','single_cells.py',settings_path,'config/dino_baseline.json',*settings['verified_files']]
    identities = {path:provenance.sha256(path) for path in dict.fromkeys(inputs)}
    core = dict(settings=settings,inputs=identities,device=device,instanseg_version=importlib.metadata.version('instanseg-torch'),cellpose_version=importlib.metadata.version('cellpose'),software=provenance.versions())
    tables,embeddings,field_rows,cache_inputs = [],[],[],[]
    for index,row in enumerate(selected.iter_rows(named=True)):
        signature = hashlib.sha256(json.dumps(dict(core=core,field=row['image_field_id']),sort_keys=True).encode()).hexdigest()
        cache = ROOT/'fields'/signature
        receipt = cache/'manifest.json'
        if receipt.is_file():
            stored = json.loads(receipt.read_text())
            if any(not (cache/name).is_file() or provenance.sha256(cache/name)!=digest for name,digest in stored['outputs'].items()):
                return Error(f'Field cache failed integrity check: {row["image_field_id"]}')
            cells = pl.read_parquet(cache/'cells.parquet')
            with np.load(cache/'embeddings.npz',allow_pickle=False) as a:
                vectors = a['embeddings']
        else:
            cache.mkdir(parents=True,exist_ok=True)
            image = read_field(row)
            field_start = time.perf_counter()
            with torch.inference_mode():
                predicted_nuclei = nuclei_model.eval_medium_image(image.astype(np.float32),pixel_size=None,tile_size=512,batch_size=1,target='nuclei',return_image_tensor=False).squeeze().cpu().numpy().astype(np.int32)
                predicted_cells,_,_ = cells_model.eval(np.moveaxis(image,0,-1),batch_size=1,channel_axis=-1,normalize=True,bsize=256)
            masks = np.stack([predicted_nuclei,predicted_cells.astype(np.int32)])
            np.savez_compressed(cache/'masks.npz',labels=masks)
            described = describe_cells(masks[1],masks[0],image,row,settings['minimum_area_px'])
            if described.is_error():
                field_rows.append(dict(image_field_id=row['image_field_id'],status=described.error,cells=0,retained=0,compound=row['Image_Metadata_Compound'],dose_um=row['Image_Metadata_Concentration'],well_id=row['Image_Metadata_Well_DAPI']))
                print(f'{index+1}/{selected.height}: {described.error}',flush=True)
                continue
            all_cells = described.ok
            all_cells.write_parquet(cache/'all_cells.parquet')
            cells = all_cells.filter(pl.col('eligible')).with_columns(pl.lit(str(cache/'crops.zip')).alias('crop_archive'),pl.lit(str(cache/'masks.npz')).alias('mask_file'))
            vectors_list, batch = [],[]
            embed_start = time.perf_counter()
            field_limits = np.percentile(np.moveaxis(image[[2,1,0]],0,-1),[1,99.5],axis=(0,1))
            with zipfile.ZipFile(cache/'crops.zip','w') as crop_archive,torch.inference_mode():
                for cell in cells.iter_rows(named=True):
                    crop,mask,bounds = crop_cell(image,masks[1],cell,settings['crop_padding_px'])
                    buffer = io.BytesIO()
                    np.savez_compressed(buffer,channels=crop,mask=mask)
                    crop_archive.writestr(f"{cell['mask_label']}.npz",buffer.getvalue())
                    composite = rgb(crop,field_limits)
                    composite[~mask] = 0
                    padded = ImageOps.pad(Image.fromarray(composite),(224,224),method=Image.Resampling.BILINEAR,color=(0,0,0))
                    batch.append(transform(padded))
                    if len(batch) == settings['embedding_batch_size']:
                        vectors_list.append(encoder(torch.stack(batch).to(device)).cpu().numpy());batch=[]
                if batch:
                    vectors_list.append(encoder(torch.stack(batch).to(device)).cpu().numpy())
            vectors = np.concatenate(vectors_list) if vectors_list else np.empty((0,384),dtype=np.float32)
            if len(vectors) and (not np.all(np.isfinite(vectors)) or np.any(np.linalg.norm(vectors,axis=1)==0)):
                return Error('Nonfinite or zero cell embeddings; projection not produced')
            vectors = vectors/np.linalg.norm(vectors,axis=1,keepdims=True) if len(vectors) else vectors
            cells.write_parquet(cache/'cells.parquet')
            np.savez_compressed(cache/'embeddings.npz',cell_id=np.array(cells['cell_id'].to_list(),dtype=str),embeddings=vectors)
            field = dict(image_field_id=row['image_field_id'],compound=row['Image_Metadata_Compound'],dose_um=row['Image_Metadata_Concentration'],well_id=row['Image_Metadata_Well_DAPI'],status='predicted; visual QC required',cells=all_cells.height,retained=cells.height,
                         nuclei=int(np.count_nonzero(np.unique(masks[0]))),border_objects=int(all_cells['border'].sum()),
                         non_nucleated_objects=all_cells.filter(pl.col('nuclear_count')==0).height,segmentation_seconds=embed_start-field_start,embedding_seconds=time.perf_counter()-embed_start)
            provenance.write_json(receipt,dict(signature=signature,field=field,outputs={p.name:provenance.sha256(p) for p in cache.iterdir() if p.is_file()}))
            stored = json.loads(receipt.read_text())
        tables.append(cells);embeddings.append(vectors);field_rows.append(stored['field'])
        cache_inputs.extend(str(cache/name) for name in ['manifest.json',*stored['outputs']])
        print(f'{index+1}/{selected.height}: {row["Image_Metadata_Compound"]} {row["Image_Metadata_Concentration"]} μM; {cells.height} retained cells',flush=True)
    if not tables or sum(table.height for table in tables) < 16:
        return Error('Insufficient retained cells for the configured shared projection')
    cells = pl.concat(tables)
    vectors = np.concatenate(embeddings)
    coordinates = umap.UMAP(n_neighbors=15,min_dist=.1,metric='cosine',random_state=42,n_jobs=1).fit_transform(vectors)
    population = population_summary(cells,vectors)
    if population.is_error():
        return population
    output.mkdir(parents=True)
    cells.write_parquet(output/'cells.parquet')
    pl.DataFrame(field_rows,infer_schema_length=None).write_parquet(output/'fields.parquet')
    population.ok[0].write_parquet(output/'wells.parquet');population.ok[1].write_parquet(output/'changes.parquet')
    np.savez_compressed(output/'embeddings.npz',cell_id=np.array(cells['cell_id'].to_list()),embeddings=vectors,coordinates=coordinates)
    outputs = {p.name:provenance.sha256(p) for p in output.iterdir()}
    config = dict(core=core,fields_per_well=fields_per_well,selection='First stable image identifier per well; independent of image appearance',umap=dict(n_neighbors=15,min_dist=.1,metric='cosine',random_state=42,input='L2-normalized 384-dimensional cell embeddings'),
                  inference='Pretrained Cellpose-SAM cells after inadequate InstanSeg pilot; pretrained InstanSeg nuclei for QC; generic DINO ImageNet cell-crop embeddings',pixel_size_um=None,spatial_units='pixels',statistics='Descriptive per-well/field summaries only; cells are not independent biological replicates')
    manifest = provenance.manifest('Single-cell microscopy atlas',[*inputs,*cache_inputs],config,outputs=outputs,cells=cells.height,fields=selected.height,available_fields=metadata.ok.height,
                                   runtime_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
    provenance.write_json(output/'manifest.json',manifest)
    return Ok(dict(cells=cells.height,fields=selected.height,runtime_seconds=manifest['runtime_seconds'],peak_gpu_allocated_bytes=manifest['peak_gpu_allocated_bytes']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'atlas-v1'))
    parser.add_argument('--fields-per-well',type=int,choices=[1,2,3,4],default=1)
    parser.add_argument('--device',choices=['cpu','cuda'])
    args = parser.parse_args()
    result = generate(args.output,fields_per_well=args.fields_per_well,device=args.device)
    print(result.ok if result.is_ok() else result.error)
    raise SystemExit(0 if result.is_ok() else 1)
