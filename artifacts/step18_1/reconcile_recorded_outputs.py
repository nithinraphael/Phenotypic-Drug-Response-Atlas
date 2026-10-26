import json,shutil
from pathlib import Path
import polars as pl
from analysis import benchmark,provenance,reporting
old=Path('reports/runs/final-2026-10-26');new=Path('reports/runs/step18-1-2026-10-26')
m=json.loads((old/'manifest.json').read_text());hashes=json.loads((old/'outputs.json').read_text())
for name in ['eligible_catalog.csv','catalog_exclusions.csv','held_out_predictions.csv']:
 assert provenance.sha256(old/name)==hashes[name]
for path,digest in m['inputs'].items():
 if path != 'analysis/benchmark.py': assert provenance.sha256(path)==digest,path
assert provenance.versions()==m['software_versions']
predictions=pl.read_csv(old/'held_out_predictions.csv')
metrics=benchmark.aggregate_predictions(predictions)
prior=pl.read_csv(old/'model_benchmark.csv',schema_overrides={'model_version':pl.String,'package_version':pl.String,'checkpoint_sha256':pl.String})
from polars.testing import assert_frame_equal
assert_frame_equal(metrics,prior,check_exact=False,rel_tol=1e-14,abs_tol=1e-14)
inputs=[*m['inputs'],str(old/'manifest.json'),str(old/'outputs.json'),str(old/'held_out_predictions.csv')]
manifest=provenance.manifest('Stable reaggregation of recorded concentration predictions',inputs,m['config'],inference_source=str(old),processing='Existing predictions reused unchanged; left-order join fixes floating-point reduction order. No new model inference or fitting.')
r=reporting.export_report(new,pl.read_csv(old/'eligible_catalog.csv'),pl.read_csv(old/'catalog_exclusions.csv'),predictions,metrics,manifest)
assert r.is_ok(),r.error
print(new)
cache=Path('data/bbbc021/dino_baseline_projection.npz');sidecar=cache.with_suffix('.json');old_projection=json.loads(sidecar.read_text())
for path,digest in old_projection['inputs'].items(): assert provenance.sha256(path)==digest,path
for path,digest in old_projection['analysis_code_hashes'].items():
 if path!='analysis/benchmark.py':assert provenance.sha256(path)==digest,path
original=cache.with_name('dino_baseline_projection_initial_manifest.json');shutil.copyfile(sidecar,original)
updated=provenance.manifest('DINO image-field projection; unchanged output reverified',list(old_projection['inputs'])+[str(original)],old_projection['config'],original_created_at=old_projection['created_at'],processing='Image inputs, extraction code, checkpoint, preprocessing and projection bytes unchanged. Reverified after unrelated benchmark join-order repair.')
provenance.write_json(sidecar,updated)
print('Projection lineage preserved; output unchanged')
