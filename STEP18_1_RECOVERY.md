# Step 18.1 recovery

Status: partial recovery. TabPFN and a separately labeled DINO baseline/UMAP are operational. Cell-DINO and cross-dataset quantitative validation remain blocked by missing evidence.

## Recovered evidence and behavior

- GDSC2 release 8.2 / 25Feb20 authenticated against the official Sanger CSV. All 135,242 local records match after identifier sorting, NA/empty normalization and 1e-12 absolute numerical tolerance. The official file contains 198 drug IDs and 809 Sanger model IDs. Original local bytes were unchanged.
- CTRPv2 has 16,729 local processed rows, 9 compound labels and 1,153 sample identifiers. These counts describe local files; release identity, transformations and chemical identity are unverified. No verified Step 17.1 crosswalk/results were found. The historical validation summary was hand-entered and remains quarantined.
- Fresh DINO ViT-S/8 ImageNet baseline embeddings were computed for all 240 BBBC021 image fields, each with 384 dimensions. Official weight/source downloads match cached identities. Legacy embeddings were not reused or relabeled. Field identifiers, input hashes, transforms, CPU float32/batch size 2, software and UMAP settings are recorded. This is exploratory image-level transfer, not Cell-DINO or single-cell analysis.
- The interactive map supports compound/dose colors, all-field highlighting, point-to-microscopy linking and coordinate export. PRISM controls stay separate. Native marimo omits custom data from selection payloads, so the app resolves trace/point indices against server-side field IDs. Retaining the widget keeps its callback registered. A native grid wrapper preserves the plot's 600-pixel interactive height on resize; the mobile title wraps. No CSS or JavaScript was added.
- Real TabPFN v2 inference produced 120 finite predictions for bortezomib BRD-K88510285-001-11-1 / A375 ACH-000219 / HTS002. Exported predicted relative viability ranges from 0.014616452666717294 to 1.3742628256643687; predictions are not clipped or treated as measurements. The browser verified equality between plotted and exported values, conditioning on observed examples, cursor persistence and tab navigation. The interface provides loading/error/success states, Prior Labs attribution and checkpoint/feature identity in CSV exports.
- A pre-existing unordered common-prediction join caused small floating-point differences across repeated metric aggregation and report reconciliation. Left observation order is now preserved. Hash-verified stored predictions were reaggregated into reports/runs/step18-1-2026-10-26; counts are identical and metrics agree within 1e-14. No models were rerun for this recorded artifact. Original reports and the initial image-projection manifest remain preserved.

## Still unavailable

Cell-DINO requires an approved checkpoint path/URL and verified support for the available fluorescence channels. No checkpoint was found or supplied. The official access route is https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/. The DINO baseline is explicitly identified throughout and is not a silent Cell-DINO substitute.

External comparisons require reviewed compound/cell-line crosswalks, verified CTRP processing provenance, quality/censoring rules and compatible potency definitions. GDSC IC50 and PRISM EC50 are not automatically interchangeable. Correlations, confidence intervals and replication classifications remain unavailable. Source availability/counts and exact blockers are visible and downloadable.

CUDA was unavailable. GTX 1650 memory/performance, other browsers and physical touch/screen-reader behavior were not tested. The baseline is not biologically validated. The recorded benchmark is one curve with only two common successful held-out predictions and does not establish model superiority or unseen-combination generalization.

## Verification

- Scientific/unit/integration suite: 73 passed, 0 failed, 0 skipped. Log: artifacts/step18_1/test-results-verified.log.
- Chromium: 18 check groups passed at 1440×1050 and 390×844. Real model inference, map selection, microscopy matching, exports, controls, zoom/reset and tab persistence executed. No JavaScript errors or horizontal page overflow. Results: artifacts/step18_1/browser-results.json.
- Source inventory table hydration separately verified; final screenshot shows both source rows and blockers. Log: artifacts/step18_1/external-table-check.log.
- Requested command uv run marimo check notebook.py: exit 0. All 16 installed marimo rules also passed for notebook.py, dev_notebook.py and audit_notebook.py.
- Complete notebook: uv run python notebook.py, exit 0. Log: artifacts/step18_1/notebook-final-source.log.
- Compileall and git diff --check passed. All 16 original data/cache files retain their recorded hashes: artifacts/step18_1/raw-data-preservation.json.
- Scoped Impeccable finish review: disposition ship after recapturing the hydrated source table. It covers visual fidelity and labels of the changed states, not independent scientific validation.

Initial checks exposed outdated projection expectations, a temporary component-hash mismatch, unstable aggregation, the native selection payload/callback retention and mobile container problems. Those failures were corrected; diagnostic logs remain in artifacts/step18_1.

## Files and provenance

IMPLEMENTATION_CHANGES.md lists changed files and compatibility effects. METHODS.md, DATA_PROVENANCE.md and CONFIGURATION.md document the running methods. recovery-manifest.json records partial feature status and current file identities; scientific cache manifests and source verification are linked under artifacts/step18_1 and data/bbbc021. Existing development notebooks and original raw data remain unchanged.

Official GDSC source: https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-8.2/GDSC2_fitted_dose_response_25Feb20.csv. Official DINO weights: https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth. Model code: https://github.com/facebookresearch/dino. TabPFN v2 license: https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt.
