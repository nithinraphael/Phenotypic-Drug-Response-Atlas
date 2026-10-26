# Data provenance

Local bytes are identified by SHA-256. Unknown acquisition dates and release versions remain unknown. No raw data were modified during repair. The input inventory is machine-readable in data_inventory.json; execution software/device information is in environment.json.

## Source identities and use

| Dataset | Source identity | Version / acquisition | Use and limits |
|---|---|---|---|
| PRISM | Original download script: Figshare article 9393293; secondary-screen source filenames | Exact local acquisition/release unrecorded; not retroactively assigned | Local cell-line, treatment, response and curve CSVs drive descriptive screening/refits; pooling file retained |
| BBBC021 | https://bbbc.broadinstitute.org/BBBC021 and Broad BBBC021_v1 source filenames | v1; acquisition date unknown | Metadata, compounds, mechanisms and plate Week1_22123 archive available; selected microscopy display only |
| GDSC2 | Local GDSC2_fitted_dose_response.csv, with dataset-specific model/drug identifiers | Original download URL/release/acquisition unknown | Excluded from automated comparison until authenticated and mapped |
| CTRPv2 | Local ctrp_matched_responses.csv and Cellosaurus sample_info.tsv | Source column CTRPv2_2015 is a local label, not independently authenticated release provenance | Processed potency definitions, upstream raw records and matching provenance unresolved; excluded |
| Legacy PRISM parquet caches | ranked_catalog_cache.parquet and summary_cache.parquet | No input/config/algorithm manifests | Not consumed; catalog and counts recomputed |
| Legacy embeddings/UMAP | embeddings_cache.npz and umap_cache.npz | No model/checkpoint/preprocessing manifest | Not consumed; correct identifiers alone cannot recover scientific provenance |

## File hashes

| Local file | Bytes | SHA-256 |
|---|---:|---|
| data/bbbc021/compounds.csv | 8179 | `e4b6868ea77eb1cef67cfe75dd792a2074ffde609c79d69c42deec131b1b88ec` |
| data/bbbc021/embeddings_cache.npz | 425996 | `ab4369bf6e586954688f221a597a4de2ea5ab59a41e3fa94a4ff04e486e93312` |
| data/bbbc021/mechanisms.csv | 4404 | `cddb6c6b72c043fd25f3a41d84624d4817927a1d76fe87dc829674181dd62719` |
| data/bbbc021/metadata.csv | 3965784 | `7fd8a9363970548736a03eba3e14bd2b737c967f0e0fdf436bc91004676f50e7` |
| data/bbbc021/plate.zip | 839436312 | `9e8607c2b8a4d57b75a2fae29cc5a7eff5bd1ca32d95fdf1b5b01e7b620592d3` |
| data/bbbc021/umap_cache.npz | 10462 | `d400072599d519cdf603fef38b6367c7b7c40445dc9db61604af9bf2b05057da` |
| data/ctrp/ctrp_matched_responses.csv | 2533919 | `caaa4af92d2ec61c9280b3ec77dfa36da62deddc08b5f630079910338ba56227` |
| data/ctrp/sample_info.tsv | 4590972 | `e01feeb3752af8339222ca978fad048b889ddc32b939ba5f52ebb47a47eef3e1` |
| data/gdsc/GDSC2_fitted_dose_response.csv | 20355833 | `0883b3ef6e440603094c43c74044e1d333fce61e92e3efa374ea3a54f0b50302` |
| data/prism/cell_lines.csv | 40979 | `b93436b5f4bcf5fd14589697be4ca8f99ffd99d639223080380bc514c3718d27` |
| data/prism/curves.csv | 264340589 | `88d1013506e0cd6f191a51c5f3fdd3fb2be54f8afb4e19a5d1f8538e81fbfec8` |
| data/prism/pooling.csv | 264680 | `73c4aa555eef63b277a3badbc7ea4dcba05c9c12e2190e373c5a1fda322377f1` |
| data/prism/ranked_catalog_cache.parquet | 6546 | `14151100a280a38967ea0f0d087bf3080ff89519aec6273e586d35136cee9cb2` |
| data/prism/responses.csv | 237567060 | `59fab6f8126cfc57d84a59e11cb67add1c7850da72cc3df6f4f9d6ef0c1808e7` |
| data/prism/summary_cache.parquet | 2179 | `43cdf99f47895914440b2a8b6c6dfc261fa28f17bd1c1b99b0e5c828f8a11b36` |
| data/prism/treatments.csv | 15690740 | `84c0043e3cdbd37a4f9b66fb62900053b5ac695f4edb6bd0a0c9d4b6168918d6` |

## Transformations and identifiers

PRISM response = log2 fold change; viability = 2^response. Broad’s reprocessing script explicitly names log2_fold_change, sets DoseUnit=uM and exponentiates to ViabilityMatrix: https://github.com/broadinstitute/repurposing/blob/master/depmap_reprocessing_script_for_secondary.R. Source documentation accessed during repair on 2026-10-26. This supports the convention, not independent authentication of the local release.

Treatment column_name is a unique condition key. Measurement loading selects one BRD compound ID, DepMap cell ID and screen ID, then joins one-to-one. Wells, detection/compound plates and original response values remain attached. Duplicate IDs are rejected. Raw matrix cell rows bearing failed STR identifiers are not relabeled as verified cell lines. The finite-positive-dose subset is separate from original exported observations.

The curve catalog is a source-fit table, not a set of independently reproduced fits. Its zero-based source_row_index points to the data row in curves.csv (excluding the header); source hashes identify that file. Missing identifiers, nonfinite parameters, nonpositive EC50, R² outside [0.75,1], and absolute slopes outside (1,20] are recorded sequentially. Excluded source row identifiers/reasons are exported in Parquet. No metadata or measurements are synthesized.

BBBC021 stable image_field_id = Image_Metadata_Plate_DAPI + / + Image_FileName_DAPI. All 240 plate fields reference 720 channel files present in plate.zip. Images are decoded only when selected. This is an archive-membership check, not a verification of every TIFF payload. RGB displays use independent channel percentiles and never overwrite raw intensities. Same-plate, same-replicate-group control selection does not establish full acquisition equivalence.

External linear potency conversion supports M, mM, uM/μM, nM and pM. Log scales are not guessed from field names. Configured records require reviewed identifier maps, explicit metric definitions and source-row IDs. Ambiguous/unmapped identifiers, duplicate records, failed/unknown QC, censoring and invalid values remain excluded. Matches cannot mix potency definitions.

## Derived artifacts

Corrected run: reports/runs/final-2026-10-26. Its manifest.json records all analysis input hashes, software versions, code revision, dirty-worktree status, code hashes, UTC time, selection and evaluation configuration. source_snapshot.tar.gz preserves the exact analysis source and dependency files. outputs.json hashes generated tables/figures. Interactive HTML figures use the same Plotly builders as the app. selected_fit.json contains numerical fit diagnostics; held_out_predictions.csv records fold failures. TabPFN v2 produced 120 conditioned grid predictions and seven held-out predictions; external validation remains unavailable. The model checkpoint/configuration hashes and exact model/package identities appear in the run manifest.

Historical reports/figures/development code are quarantined under legacy/unverified. index.json identifies originals and hashes. The superseded repair run is retained separately; its early cell-line count included null and is not the current result. Old UI session metadata was quarantined to prevent stale state reuse.

## Cache identity and invalidation

Legacy scientific caches are never accepted because they merely exist. Future embedding caches require a completed sidecar manifest, exact source metadata/archive/checkpoint and artifact hashes, configuration, schema version, software versions and analysis-source hashes. Model identity, preprocessing, dimensions and UMAP settings form the expected configuration. Identifier sets, uniqueness, dtype and finiteness are checked; object/pickle arrays are rejected before loading.

In-memory marimo caches separate loading, fitting and prediction from annotations. Source bytes and relevant configuration identify cached results. TabPFN predictions key on observations, grid, package/model/checkpoint/device/precision/features/seed configuration; the worker rechecks checkpoint hash. Cursor position is excluded. Reopen the app after changing data/configuration; external files are not continuously watched.

A hash proves file identity, not experimental validity, authoritative origin or channel compatibility. New cache manifests must follow verified generation, never serve as provenance backfill for unknown artifacts.

## Cached model inventory

model_inventory.json records three cached weight files discovered outside data/. TabPFN v2 matches the publisher checkpoint hash and is explicitly configured (see config/tabpfn.json). Two generic DINO/DINOv2 weight files were found, but existing embedding model/preprocessing attribution remains unverified. No Cell-DINO checkpoint was found. Cached weights were not overwritten or downloaded.
