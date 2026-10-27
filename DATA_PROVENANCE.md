# Data provenance

Local bytes are identified by SHA-256. Unknown acquisition dates and release versions remain unknown. No raw data were modified during repair. The input inventory is machine-readable in data_inventory.json; execution software/device information is in environment.json.

## Source identities and use

| Dataset | Source identity | Version / acquisition | Use and limits |
|---|---|---|---|
| PRISM | Original download script: Figshare article 9393293; secondary-screen source filenames | Exact local acquisition/release unrecorded; not retroactively assigned | Local cell-line, treatment, response and curve CSVs drive descriptive screening/refits; pooling file retained |
| BBBC021 | https://bbbc.broadinstitute.org/BBBC021 and Broad BBBC021_v1 source filenames | v1; acquisition date unknown | Metadata, compounds, mechanisms and plate Week1_22123 archive available; microscopy and newly computed DINO baseline image-field projection |
| GDSC2 | Official Sanger release archive; local table authenticated against downloaded CSV | Release 8.2 / 25Feb20; authenticated 2026-10-26; original local acquisition date unknown | 135,242 records, 198 drug IDs, 809 Sanger model IDs; cross-PRISM mappings remain unresolved |
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

BBBC021 stable image_field_id = Image_Metadata_Plate_DAPI + / + Image_FileName_DAPI. All 240 plate fields reference 720 channel files present in plate.zip. The original repair checked archive membership. Step 18.1 decoded all 720 TIFF channel files for fresh embeddings; normal display decodes only selected fields. RGB displays use independent channel percentiles and never overwrite raw intensities. Same-plate, same-replicate-group control selection does not establish full acquisition equivalence.

External linear potency conversion supports M, mM, uM/μM, nM and pM. Log scales are not guessed from field names. Configured records require reviewed identifier maps, explicit metric definitions and source-row IDs. Ambiguous/unmapped identifiers, duplicate records, failed/unknown QC, censoring and invalid values remain excluded. Matches cannot mix potency definitions.

## Derived artifacts

Corrected run: reports/runs/final-2026-10-26. Its manifest.json records all analysis input hashes, software versions, code revision, dirty-worktree status, code hashes, UTC time, selection and evaluation configuration. source_snapshot.tar.gz preserves the exact analysis source and dependency files. outputs.json hashes generated tables/figures. Interactive HTML figures use the same Plotly builders as the app. selected_fit.json contains numerical fit diagnostics; held_out_predictions.csv records fold failures. TabPFN v2 produced 120 conditioned grid predictions and seven held-out predictions; external validation remains unavailable. The model checkpoint/configuration hashes and exact model/package identities appear in the run manifest.

Historical reports/figures/development code are quarantined under legacy/unverified. index.json identifies originals and hashes. The superseded repair run is retained separately; its early cell-line count included null and is not the current result. Old UI session metadata was quarantined to prevent stale state reuse.

## Cache identity and invalidation

Legacy scientific caches are never accepted because they merely exist. Future embedding caches require a completed sidecar manifest, exact source metadata/archive/checkpoint and artifact hashes, configuration, schema version, software versions and analysis-source hashes. Model identity, preprocessing, dimensions and UMAP settings form the expected configuration. Identifier sets, uniqueness, dtype and finiteness are checked; object/pickle arrays are rejected before loading.

In-memory marimo caches separate loading, fitting and prediction from annotations. Source bytes and relevant configuration identify cached results. TabPFN predictions key on observations, grid, package/model/checkpoint/device/precision/features/seed configuration; the worker rechecks checkpoint hash. Cursor position is excluded. Reopen the app after changing data/configuration; external files are not continuously watched.

A hash proves file identity, not experimental validity, authoritative origin or channel compatibility. New cache manifests must follow verified generation, never serve as provenance backfill for unknown artifacts.

## Cached model inventory

model_inventory.json records three cached weight files discovered outside data/. TabPFN v2 matches the publisher checkpoint hash and is explicitly configured (see config/tabpfn.json). Two generic DINO/DINOv2 weight files were found, but existing embedding model/preprocessing attribution remains unverified. No Cell-DINO checkpoint was found. Cached weights were not overwritten. Step 18.1 downloaded an official DINO copy separately to verify its identity.

## Step 18.1 recovered provenance

GDSC source: https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.2/GDSC2_fitted_dose_response_25Feb20.csv. Original local bytes remain unchanged. Official and local tables have identical record identifiers and metadata after NA/empty normalization; all numeric columns agree within absolute tolerance 1e-12. Verification: artifacts/step18_1/gdsc-source-verification.json. Source hashes, URLs and status: config/external_sources.json. CTRPv2's 16,729 processed rows remain unverified; its 9 compound labels are not established chemical identifiers. No crosswalk or comparable-potency family was recovered. No matches, correlations, confidence intervals or replication classifications were generated.

DINO ViT-S/8 baseline weights were independently downloaded from https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth and exactly match cached SHA-256 55c8b267f479b614ec20858947ce361ba804846c193e20be74263a096bb6d75e. Official architecture/helper source files also match the local cache. config/dino_baseline.json records each executable source/checkpoint hash; data/bbbc021/verified_model retains source and license. No Cell-DINO weights were found.

image_projection.py recomputed all 240 image-field embeddings (384 dimensions) and UMAP coordinates. Exact RGB/channel transforms, device, batch size, software, seed and UMAP settings are in data/bbbc021/dino_baseline_projection.json and projection_config.json. Original caches were untouched. Stable plate-qualified filenames join coordinates to metadata. This is exploratory ImageNet transfer, not Cell-DINO, biological validation or single-cell analysis.

A pre-existing unordered benchmark join caused floating-point reduction differences across repeated aggregation. The join now preserves observation order. Original prediction artifacts were hash-verified and reused in reports/runs/step18-1-2026-10-26; the new manifest identifies the original run and current aggregation code. No models were refitted for this reaggregation. Metrics agree within 1e-14 and all counts are unchanged. The image projection was reverified after this unrelated analysis-code change; its initial manifest is retained and output bytes are unchanged.

TabPFN uses the existing publisher-verified v2 checkpoint, log10 micromolar doses and explicit CPU configuration. The current publisher license (Prior Labs License 1.1) is retained in artifacts/step18_1/TabPFN-LICENSE.txt; the public interface includes the required attribution. Prediction CSVs identify compound, cell, screen, model, checkpoint and feature representation.

## Step 19 cell atlas

Uses only local BBBC021 metadata and plate images; no PRISM join. Source hashes, all field artifacts, model/code identities, preprocessing and versions are recorded in `data/bbbc021/single_cells/atlas-v1/manifest.json` (SHA256 `e905e3a48971962671b7263b65db8a79eeb09f8de85efc47a104e557febcdc8c`). Checkpoint source URLs/publisher hash verification are in `config/single_cells.json` and `artifacts/step19/cellpose-model-record.json`.

Pre-segmented Zenodo record 22763120 was inspected through metadata and two final archive parts, whose published MD5s were checked. A traceable image mapping was not established, so its masks were not used. No full 2.9 GB dataset download. Official InstanSeg and pinned Cellpose-SAM weights were explicitly acquired; existing generic DINO weights reused.

One stable first field per each of 60 wells; 5,857 retained objects. Raw channels DAPI/Tubulin/Actin retain uint16 pixels. Display percentile adjustment, isolated mask, aspect-preserving padding to 224 and ImageNet normalization apply only to DINO/display inputs. Spatial units are pixels. All masks and excluded rows remain in per-field caches. IDs include image-field identifier and original mask label.

Each field cache keys source hashes, generator/configuration, model/checkpoint, package versions and preprocessing. A final receipt hashes all field and combined artifacts; loaders reject mismatches or positional-only coordinate assignment. Shared UMAP: cosine, L2 input, 15 neighbors, min_dist 0.1, seed 42. Crops and feature vectors are derived image-level cell-object measurements, with unverified biological replicate independence.
