import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from scipy import ndimage as ndi

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import microscopy,provenance
from single_cells import crop_cell,load_atlas,population_summary,read_crop,read_field

parser=argparse.ArgumentParser()
parser.add_argument('--root',default='data/bbbc021/single_cells/atlas-v1')
parser.add_argument('--output',default='artifacts/step19/data-validation.json')
args=parser.parse_args()
loaded=load_atlas(args.root)
assert loaded.is_ok(),loaded.error
atlas=loaded.ok;cells=atlas['cells'];manifest=atlas['manifest']
assert cells['cell_id'].n_unique()==cells.height==manifest['cells']
assert cells['eligible'].all() and not cells['border'].any()
assert cells['exclusion_reason'].null_count()==cells.height
with np.load(Path(args.root)/'embeddings.npz',allow_pickle=False) as arrays:
 ids=arrays['cell_id'];vectors=arrays['embeddings'];coordinates=arrays['coordinates']
assert vectors.shape==(cells.height,384) and coordinates.shape==(cells.height,2)
np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
positions={cell_id:index for index,cell_id in enumerate(ids)}
aligned=vectors[[positions[cell_id] for cell_id in cells['cell_id']]]
wells,changes=population_summary(cells,aligned).ok
assert_frame_equal(wells.sort('compound','dose_um','well_id'),atlas['wells'].sort('compound','dose_um','well_id'),check_exact=False,abs_tol=1e-6,rel_tol=1e-6)
assert_frame_equal(changes.sort('compound','upper_dose_um'),atlas['changes'].sort('compound','upper_dose_um'),check_exact=False,abs_tol=1e-6,rel_tol=1e-6)
metadata=microscopy.load_metadata().ok
sampled_crops=0;fragmented=0;field_results=[]
rng=np.random.default_rng(42)
for group in cells.partition_by('image_field_id',maintain_order=True):
 field_id=group['image_field_id'][0]
 source=metadata.filter(pl.col('image_field_id')==field_id).row(0,named=True)
 assert group['compound'].unique().to_list()==[source['Image_Metadata_Compound']]
 assert group['dose_um'].unique().to_list()==[source['Image_Metadata_Concentration']]
 assert group['well_id'].unique().to_list()==[source['Image_Metadata_Well_DAPI']]
 with np.load(group['mask_file'][0],allow_pickle=False) as arrays:labels=arrays['labels']
 assert labels.dtype.kind in 'iu' and labels.shape[0]==2 and np.all(labels>=0)
 areas=np.bincount(labels[1].ravel())
 assert all(areas[label]==area for label,area in group.select('mask_label','area_px').iter_rows())
 all_cells=pl.read_parquet(Path(group['mask_file'][0]).parent/'all_cells.parquet')
 field=atlas['fields'].filter(pl.col('image_field_id')==field_id).row(0,named=True)
 assert all_cells.height==field['cells']==np.count_nonzero(np.unique(labels[1]))
 assert group.height==field['retained']==all_cells['eligible'].sum()
 assert all_cells.filter(~pl.col('eligible'))['exclusion_reason'].null_count()==0
 for row in group.iter_rows(named=True):
  region=labels[1][row['min_y']:row['max_y'],row['min_x']:row['max_x']]==row['mask_label']
  fragmented+=ndi.label(region,structure=np.ones((3,3)))[1]>1
 raw=read_field(source)
 assert raw.dtype==np.uint16 and raw.shape==(3,*labels.shape[1:])
 for index in rng.choice(group.height,size=min(3,group.height),replace=False):
  cell=group.row(int(index),named=True)
  cached,mask=read_crop(cell)
  expected,expected_mask,bounds=crop_cell(raw,labels[1],cell,manifest['config']['core']['settings']['crop_padding_px'])
  np.testing.assert_array_equal(cached,expected);np.testing.assert_array_equal(mask,expected_mask)
  y,x=np.nonzero(labels[1]==cell['mask_label'])
  np.testing.assert_allclose([cell['x'],cell['y']],[x.mean(),y.mean()])
  assert (int(y.min()),int(x.min()),int(y.max())+1,int(x.max())+1)==(cell['min_y'],cell['min_x'],cell['max_y'],cell['max_x'])
  sampled_crops+=1
 field_results.append(dict(image_field_id=field_id,objects=field['cells'],retained=group.height))
result=dict(status='passed',analysis_manifest_sha256=provenance.sha256(Path(args.root)/'manifest.json'),cells=cells.height,
 sampled_fields=manifest['fields'],fields_with_retained_cells=len(field_results),unique_wells=atlas['wells'].height,
 raw_crops_compared=sampled_crops,fragmented_retained_masks=int(fragmented),embedding_dimensions=384,
 coordinate_dimensions=2,norm_tolerance=1e-5,population_recalculation_tolerance=1e-6,
 segmentation_accuracy='Not estimable: no matched manual reference masks',field_results=field_results)
provenance.write_json(args.output,result)
print(json.dumps({key:value for key,value in result.items() if key!='field_results'},indent=2))
