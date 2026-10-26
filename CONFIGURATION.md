# Optional analysis configuration

No example numbers are installed as experimental results. TabPFN is configured explicitly as v2 CPU using verified cached weights; external-comparison and projection configurations remain absent. Reopen the application after changing source files or configuration; it does not watch external data mutations.

## TabPFN

`config/tabpfn.json` requires:

- `model_family`: TabPFN.
- `model_version`: the verified release identity of the configured checkpoint, not the Python package version.
- `package_version`: exact installed TabPFN package version.
- `checkpoint_path`: existing local checkpoint; no auto-download or authentication-dependent model choice.
- `checkpoint_sha256`: SHA-256 of that checkpoint, checked again by the worker.
- `device`: cpu or cuda; no automatic substitution.
- `precision`: auto or autocast, passed explicitly to the installed API.
- `feature_definition`: log10_dose or log10_and_raw_dose.
- `random_seed`: integer.

The redundant two-feature representation is retained as an explicit option, not a proven improvement. To compare representations, run identical concentration-held-out splits with separate configurations. No representation comparison has been executed during repair. The explicitly configured v2 model is evaluated only in the declared selected-curve run.

Local checkpoint identity still requires an authoritative release record; a self-entered model label alone is not evidence. The app records the supplied identity and verifies file integrity; it cannot establish the truth of a mislabeled checkpoint.

## Embeddings and UMAP

`data/bbbc021/projection_config.json` requires `checkpoint_path`, `cache_path`, and `analysis_config`. Analysis configuration requires model_id, checkpoint_hash, preprocessing_hash, embedding_dimension, umap_version, random_seed, n_neighbors, min_dist, metric, input_normalization and n_fields.

The NPZ must contain numeric `embeddings` (N × D), numeric `coordinates` (N × 2), and Unicode `image_field_id` (N). Identifiers must equal `plate/DAPI_filename.tif`, matching the selected metadata subset exactly, with no duplicates. Pickle-backed arrays are rejected.

The same-stem JSON sidecar must be a completed `analysis.provenance.manifest` for metadata.csv, plate.zip, the checkpoint and the NPZ itself, with the exact analysis configuration and installed software/code hashes. Hashes cover the input archive, not selected images by row order. Model architecture, biological channel order and preprocessing must be scientifically verified before generating this manifest. Recording hashes alone does not validate input compatibility. No automatic extraction or provenance backfill for legacy caches is provided.

## External comparison

`config/validation.json` contains `reference` and `external` entries. Each requires:

- `source`: dataset, version, source_file, source_sha256 and source_url.
- `records`: normalized source-measurement CSV.
- `compound_mapping` and `cell_mapping`: reviewed CSV crosswalks with raw_id and canonical_id.
- `file_hashes`: SHA-256 entries for the original source file, normalized records and both mappings.

Normalized measurements require source_record_id, raw_compound_id, raw_cell_line_id, potency_metric, metric_definition, potency_value, concentration_unit, quality_status and censored. Preserve optional original identifiers, assay duration, r2, slope, boundary flags and name fields. Values must be linear concentrations; log metrics must be transformed upstream using verified definitions and recorded processing provenance. Supported units: M, mM, uM, μM, nM, pM. Only quality_status=pass, uncensored positive finite values enter comparisons.

Mappings are identifier-based, whitespace-normalized and case-sensitive. Ambiguities are excluded; repeated source records are excluded; repeated mapped conditions reject automated comparison until an explicit assay-group policy is specified. Matching requires canonical compound, canonical cell, potency_metric and metric_definition equality. Never assign EC50 and IC50 the same definition merely to force a join. Review definition equivalence against source methods; string equality cannot establish biological equivalence.

Per-source counts accompany matches. Correlations are calculated separately by source pair, metric and definition, using Spearman ranks; fewer than three nonconstant pairs produce Not evaluated. No correlation p-values, intervals, causal explanations or subjective concordance categories are produced.

Configured TabPFN weights match publisher SHA-256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`, repository revision `230d61748baee2db4beb5cf5b99c7cd3a9b71df0`: https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/230d61748baee2db4beb5cf5b99c7cd3a9b71df0/tabpfn-v2-regressor.ckpt. Their absolute local cache path is machine-specific; a different machine must supply an explicitly verified checkpoint path. No v3.5 fallback is implemented.
