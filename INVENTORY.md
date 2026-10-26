# Baseline inventory

Baseline git working tree was clean. No tests existed. Python 3.13.15; packages recorded in environment.json.

Entrypoints: notebook.py (public marimo), dev_notebook.py (3,739-line experimental marimo), hello.py (placeholder). components.py combines fitting, model loading, image reading and plotting. scripts/download_data.py contains a syntax error (stray c).

Available: six PRISM CSVs; two unversioned parquet caches; BBBC021 metadata, compounds, mechanisms, one 839 MB plate archive, 240 keyed 384-dimensional embeddings and 240 keyed UMAP coordinates. Neither cache identifies the generating checkpoint/preprocessing. Cell-DINO checkpoint absent. GDSC2 fitted CSV and processed CTRPv2 CSV plus Cellosaurus table exist, without acquisition manifests or verified cross-dataset identifier mappings. No local configured TabPFN checkpoint.

Reports and figures contain unsupported inference, manually entered validation/performance values and synthetic empirical figures. Originals preserved in legacy/unverified with hashes. No raw observations altered. Legacy caches are not trusted or migrated by guessing provenance.

Known baseline runtime defect: notebook cell returns undefined def_cl. Fit failure fabricates parameters; cursor triggers TabPFN fitting; authentication silently selects model version. No baseline scientific tests available.

Subsequent cache inspection found /home/ronaldo/.cache/tabpfn/tabpfn-v2-regressor.ckpt and generic DINO/DINOv2 weights. model_inventory.json records hashes. The TabPFN hash was authenticated against the publisher record and explicitly configured as v2 CPU; the DINO files do not establish embedding-cache provenance.
