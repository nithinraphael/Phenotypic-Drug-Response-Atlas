import importlib.metadata
import json
from pathlib import Path
import numpy as np
import polars as pl
import umap
from analysis import provenance
from single_cells import population_summary

root=Path('data/bbbc021/single_cells/atlas-pilot')
assert not root.exists()
settings=json.load(open('config/single_cells.json'))
inputs=['data/bbbc021/metadata.csv','data/bbbc021/plate.zip','single_cells.py','config/single_cells.json','config/dino_baseline.json',*settings['verified_files']]
identities={p:provenance.sha256(p) for p in dict.fromkeys(inputs)}
core=dict(settings=settings,inputs=identities,device='cuda',instanseg_version=importlib.metadata.version('instanseg-torch'),cellpose_version=importlib.metadata.version('cellpose'),software=provenance.versions())
tables=[];vectors=[];fields=[];cache_inputs=[]
for receipt in sorted(Path('data/bbbc021/single_cells/fields').glob('*/manifest.json')):
 stored=json.load(receipt.open());cache=receipt.parent
 import hashlib
 signature=hashlib.sha256(json.dumps(dict(core=core,field=stored['field']['image_field_id']),sort_keys=True).encode()).hexdigest()
 assert signature==stored['signature']
 for name,digest in stored['outputs'].items():assert provenance.sha256(cache/name)==digest
 tables.append(pl.read_parquet(cache/'cells.parquet'))
 with np.load(cache/'embeddings.npz',allow_pickle=False) as arrays:vectors.append(arrays['embeddings'])
 fields.append(stored['field']);cache_inputs.extend(str(cache/name) for name in ['manifest.json',*stored['outputs']])
cells=pl.concat(tables);vectors=np.concatenate(vectors)
coordinates=umap.UMAP(n_neighbors=15,min_dist=.1,metric='cosine',random_state=42,n_jobs=1).fit_transform(vectors)
wells,changes=population_summary(cells,vectors).ok
root.mkdir(parents=True)
cells.write_parquet(root/'cells.parquet');pl.DataFrame(fields).write_parquet(root/'fields.parquet');wells.write_parquet(root/'wells.parquet');changes.write_parquet(root/'changes.parquet')
np.savez_compressed(root/'embeddings.npz',cell_id=np.array(cells['cell_id'].to_list()),embeddings=vectors,coordinates=coordinates)
outputs={p.name:provenance.sha256(p) for p in root.iterdir()}
provenance.write_json(root/'manifest.json',provenance.manifest('Real-data pilot for desktop UI verification',[*inputs,*cache_inputs,'artifacts/step19/build_pilot.py'],dict(core=core,scope='Only fields completed before pilot assembly; not the final atlas'),outputs=outputs,cells=cells.height,fields=len(fields),available_fields=240))
print(cells.height,len(fields),root)
