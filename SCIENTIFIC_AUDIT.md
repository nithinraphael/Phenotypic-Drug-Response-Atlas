# Scientific audit

## Executive assessment

The original interface mixed measured data with fabricated statistical inference, manually entered external comparisons, fixed benchmark values and synthetic figure observations. These were scientific implementation errors, not presentation problems. Production now uses shared calculations, explicit failures and provenance checks. Optional analyses lacking evidence are unavailable. Raw data were preserved. This repair does not establish biological validity or reproduce every historical conclusion.

## Critical findings and corrections

| Original behavior | Corrected behavior |
|---|---|
| Slope SE approximated as max((1−R²)/sqrt(8), 0.05); normal-tail p-values and BH correction on a selected shortlist | Procedure removed everywhere executable. No replacement significance test. Full eligible catalog ranked descriptively before shortlist selection |
| Residual-width band called a 95% confidence interval; R²-derived EC50 bounds | Both removed. No parameter or response uncertainty reported without established resampling units |
| Manually entered GDSC2/CTRP values written as an independent-validation CSV | Generator removed; legacy CSV quarantined. Configured source ingestion requires provenance, mappings, metric definitions and explicit exclusions |
| Fixed, inconsistent RMSE arrays and unconditional model superiority | Actual concentration-held-out predictions and failures; pooled RMSE plus comparison on common successful observations |
| Full-data fit determined transition-zone evaluation | Transition-zone benchmark omitted; all retained fitting is training-only within each fold |
| Missing RMSE initialized to zero; failed segmented fits became a mean predictor | Missing metrics remain missing; failures counted. Training mean is a separately named baseline |
| Synthetic observations, sinusoidal “TabPFN” curve and constructed confidence band exported as empirical figure | Synthetic figure pipeline removed; real observations and shared fitted results drive interactive report plots |
| Manual Taxol microscopy record inserted into viability catalog | Removed. BBBC021 morphology and PRISM viability remain separate |
| Fit failures fabricated EC50, zero slope, mean prediction and confidence width | Failed FitResult contains reason and missing parameters/predictions; observations remain visible |
| R² denominator used arbitrary epsilon | Exact SSE/SST; constant-response R² unavailable; negative R² retained |
| Name-based selection pooled different BRD identifiers and screens | Compound ID, cell ID and screen selectors; one-to-one joins retain well/plate/source identifiers |
| Authentication silently selected different TabPFN releases | Explicit local configuration and checkpoint hash; no authentication-driven substitution |
| Cursor moved through cell that also fitted TabPFN | Prediction and annotation cells separated; cache includes observations, grid and model identity |
| UMAP attached positionally and labeled Cell-DINO | Identifier joins and strict model/preprocessing/input provenance checks; legacy map disabled |
| Five-channel Cell-DINO input zero-padded from three channels | Unsupported extraction removed; compatibility remains unresolved |
| First DMSO control without a matching rule | Same plate and replicate group, deterministic field selection; further acquisition matching unverified |
| Ordinary notebook execution overwrote reports | Explicit new-directory report command; no automatic overwrites |

## Actual source-data findings

The local files reproduce 701,004 curve-table rows. They contain 1,502 distinct BRD identifiers and 1,448 display names; these are different counts. There are 480 nonmissing DepMap IDs. The previous 481 included null, not another identified cell line. Excluding missing cell identifiers leaves 644,798 unique compound–cell pairs and 690,192 compound–cell–screen records. Missing IDs account for 10,812 source rows, including failed STR-profiling rows whose original row_name remains in the raw file.

The response matrix has 489 source rows, 39,925 condition columns and 14,527,780 nonmissing response values. All condition columns map to treatment identifiers; treatment identifiers are unique. Selected-condition joins enforce one-to-one cardinality. Missing responses remain in exports and are excluded explicitly from numerical calculations.

The Broad Institute reprocessing script identifies log2 fold changes and converts them using 2^response, with dose units uM. This transformation was retained. Exact local release identity cannot be authenticated because original acquisition manifests were absent; current file hashes provide reproducibility, not proof of source release. Sources: [Broad script](https://github.com/broadinstitute/repurposing/blob/master/depmap_reprocessing_script_for_secondary.R), [PRISM resource](https://depmap.org/repurposing/).

The complete descriptive filter retains 51,763 source-fit records. Sequential exclusions: 10,812 missing identifiers; zero additional nonfinite-parameter exclusions; zero nonpositive-EC50 exclusions; 632,384 R² exclusions; 6,045 slope-range exclusions. Every excluded row is recorded by zero-based source row index, identifiers and reason in excluded_curve_records.parquet. Source-fit convergence and raw-data agreement were not verified for all 51,763 records.

## Fitting and interpretations

Shared LL4 uses installed Thunor 1.0.3. Its parameterization and optimizer code were inspected locally. Internal flat-model inference is disabled. Numerical retries remain within the same LL4 model. Parameter validation, failure status, exact R², RMSE, effect-span EC20/EC80 and numerical diagnostics replace arbitrary defaults. No response clamping is applied.

For the reproduced bortezomib BRD-K88510285-001-11-1 / RKO ACH-000943 / HTS002 condition, 24 referenced observations include seven usable concentrations and seventeen missing responses. The refit gives EC50 0.0300963 μM, slope 18.0837 and R² 0.965740. These are newly calculated values, not identical to the source catalog fit. Sparse transition sampling and an ill-conditioned Jacobian are flagged. No biological switching or parameter certainty is inferred from that steep fit.

Categories are “Steep response, strong fit,” “Steep response, moderate fit,” and “Other qualifying fit.” Failed or unavailable fits are explicit. Screening Score = absolute fitted Hill slope × R². Neither categories nor scores imply statistical significance, mechanisms or probabilities. Extreme slopes can rank highly while remaining unstable.

## Benchmark findings

The executed benchmark covers one selected curve, three concentration-group folds and seven held-out observations. Hill LL4 predicts four observations and fails one fold; segmented regression predicts two observations and fails two folds; the separate mean baseline predicts seven. Their individual pooled RMSE values use different successful samples and must not be compared as though samples match.

Common-observation RMSE uses two observations: Hill 0.255813, mean baseline 0.816801, segmented 0.714270, TabPFN v2 0.726735. These tiny-sample descriptive results do not establish superiority or reproduce the former global claims. Full predictions, exclusions, failures, configurations and package versions are in reports/runs/final-2026-10-26. The explicitly configured TabPFN v2 checkpoint produced seven held-out predictions without fold failures (individual pooled RMSE 0.556963; common-observation RMSE 0.726735). It also produced 120 finite predictions conditioned on all seven usable observations for the dose-response figure. No claim about unseen compound–cell combinations is supported.

## TabPFN provenance

A cached 44,390,977-byte v2 regressor checkpoint was found outside the project data directory. Its SHA-256 matches the publisher record: `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`, repository revision `230d61748baee2db4beb5cf5b99c7cd3a9b71df0`. Source: [Prior Labs checkpoint](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/230d61748baee2db4beb5cf5b99c7cd3a9b71df0/tabpfn-v2-regressor.ckpt). config/tabpfn.json identifies this explicitly as v2, CPU, auto precision, log10 dose and seed 42 with installed package 9.2.0. It is not a substitute chosen in response to authentication, and is not labeled v3.5. No new checkpoint was downloaded. Predictions and the small benchmark were executed; feature-representation alternatives remain unevaluated.

## External-data findings

GDSC2 fitted CSV and processed CTRPv2 CSV exist, with a Cellosaurus table. Acquisition versions, upstream transformations and reviewed chemical/cell-line crosswalks are absent. The processed CTRP potency scale cannot be inferred safely from column names alone. No name-only matching or EC50/IC50 substitution was performed. Quantitative comparison, historical matched-pair counts, correlations and candidate concordance claims are unavailable. Generic ingestion/matching/correlation functions are tested with labeled synthetic fixtures; that is not external experimental validation.

## Imaging findings

The available plate has 240 metadata fields and 720 referenced channel files; all 720 names exist in plate.zip. Legacy embeddings are 240 keyed 384-dimensional vectors; the UMAP file contains 240 keys and two-dimensional coordinates. Keys alone do not establish model identity. The historical default was DINO ViT-S/8 baseline, but the cache cannot be conclusively attributed to that model or Cell-DINO. No checkpoint/preprocessing hashes were recorded for the embedding cache; the Cell-DINO checkpoint is absent. Two cached generic DINO/DINOv2 weight files were found, but their presence does not establish the provenance of existing embeddings. Attribution and UMAP are suppressed.

Microscopy display remains available. Channels and 24-hour MCF-7 scope agree with [BBBC021 v1](https://bbbc.broadinstitute.org/BBBC021). Display contrast adjustment preserves raw intensities. Future projections require plate-qualified image identifiers, source/checkpoint/preprocessing hashes, dimensions and complete UMAP settings. Shuffled-ID tests demonstrate correct joins; missing/duplicate IDs and stale/pickle-backed caches are rejected.

## Removed claims and static audit

Unsupported FDR labels, significance tiers, fixed correlations, 5,572 matched-pair claims, subjective concordance fractions, model-superiority claims, publication-quality language and causal explanations were removed from executable output paths. Historical occurrences are retained only in quarantined artifacts. Documentation of the flawed formulas is historical explanation; synthetic fixture values are tests; ordinary input-validation variable names are not scientific-validation claims. No fixed scientific result arrays remain in production chart builders.

## Unresolved questions and required evidence

- Formal slope inference: establish independent replicate units, a defined null, identifiable model, calibrated test and complete predeclared hypothesis family.
- Uncertainty: establish plate/screen/biological replication and defensible resampling design. Numerical parameter fits alone are insufficient.
- Entire catalog: source optimization status, parameter bounds, raw-data agreement and dose coverage require per-record verification. Catalog status remains source-fit, not independently refitted.
- Source authenticity: recover release/file identifiers and authoritative checksums for local PRISM/external files. Hashes identify local bytes only.
- External comparisons: recover raw records, linear/log potency definitions, units, censoring/bound rules, assay scope and reviewed identifier maps.
- Image representations: recover exact checkpoint/preprocessing and compatible channels, or explicitly recompute using a verified model. Do not relabel old caches by guesswork.
- Perturbation stability and valid bootstrap intervals remain unevaluated.
- Browser interaction and pretrained execution require further tests in an environment with browser/model assets. Programmatic UI tests are not browser E2E tests.

## Step 18.1 follow-up

GDSC release 8.2 source identity was recovered through the official archive and compared across all 135,242 local records. CTRP processing provenance and reviewed compound/cell-line crosswalks were not recovered; historical hand-entered validation remains quarantined. Correlations, confidence intervals and replication classes remain unsupported.

Official DINO ViT-S/8 weights and source code were authenticated. Fresh image-field embeddings and UMAP were computed for all 240 available fields, with full transforms and stable identifiers. This ImageNet baseline is explicitly distinct from Cell-DINO and individual-cell analysis. No approved Cell-DINO weights or three-channel compatibility evidence were available.

Real TabPFN v2 inference, source-labeled exports, license attribution, conditioning semantics and reactive separation are exposed publicly. A native marimo selection payload omitted custom data; server-side trace/point resolution repairs microscopy linking.

Scientific regression tests exposed unstable ordering in the existing common-prediction join. Left-order preservation fixes repeated floating-point reductions. Stored predictions were reaggregated with preserved original lineage; no prior report was overwritten.
