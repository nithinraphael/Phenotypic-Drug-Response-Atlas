# Methods

## Data sources

PRISM secondary-screen files are local CSVs referenced by the original downloader to Figshare article 9393293. Exact acquisition date and release-file identity were not recorded; local SHA-256 hashes identify the analyzed files. Counts are computed at runtime, including raw curve records, compound identifiers, cell-line identifiers, condition groups and nonmissing response values. Null cell-line identifiers are excluded from screening. Distinct BRD identifiers can share a display name.

BBBC021 v1: MCF-7 cells, 24-hour compound exposure; DNA (DAPI), tubulin and actin channels. The local archive contains plate Week1_22123 only. The interface reports available metadata fields separately from fields with verified embeddings. Original acquisition dates are unknown. See DATA_PROVENANCE.md and reports/runs for hashes and processed counts.

GDSC2 release 8.2 (25Feb20) is authenticated against the official Sanger CSV. All 135,242 local records agree after sorting by NLME_CURVE_ID, normalizing NA/empty strings and allowing 1e-12 absolute numeric tolerance. GDSC defines LN_IC50 as natural-log micromolar IC50. The local CTRPv2 subset contains 16,729 processed records; its acquisition release and potency transformations remain unverified. Neither dataset has reviewed compound/cell-line crosswalks to PRISM. Quantitative comparison remains unavailable. Source counts and hashes are downloadable.

## PRISM processing

The response matrix stores log2 fold changes. Relative viability = 2^response, following the Broad Institute's public reprocessing script: https://github.com/broadinstitute/repurposing/blob/master/depmap_reprocessing_script_for_secondary.R (lines 11, 23 and 35). Doses are micromolar. Values above one are retained.

Compound identifiers, cell-line identifiers and screen identifiers select measurements. Response columns join one-to-one with treatment identifiers; wells, detection plates and source column identifiers remain attached. Different screens and BRD identifiers are never pooled. Replicates within a selected screen remain separate, equally weighted observations; their biological independence is not established. Nonpositive or nonfinite doses and missing/nonfinite responses are excluded from fitting with retained exclusion records. Raw exports retain missing observations. No bootstrap uncertainty is reported.

## Nonlinear fitting

Thunor 1.0.3 HillCurveLL4 and fit_drc implement:

`y(x) = L + (U − L) / [1 + exp(h × (ln x − ln EC50))]`.

EC50 is the midpoint of the fitted span; L and U are asymptotes. The fitted slope is normalized to a positive magnitude by swapping asymptotes when necessary, preserving predictions. Increasing responses are flagged and receive no inhibition-range annotation.

Thunor uses SciPy nonlinear least squares, log-EC50 parameterization, dose scaling and unconstrained Levenberg–Marquardt optimization for LL4. Its internal numerical retry uses the same LL4 in linear EC50 coordinates; no alternate model or fabricated parameter is substituted. The internal flat-model significance gate is explicitly disabled. At least four distinct concentrations and five observations are required. Invalid, nonfinite, degenerate or unconverged fits return no parameters. Constant response has undefined R² and is reported as failed.

R² = 1 − SSE/SST; RMSE = sqrt(mean squared residual). Negative R² is retained. No residual-width confidence band is drawn. EC20 = EC50 × 0.25^(1/h); EC80 = EC50 × 4^(1/h), for inhibition relative to the fitted effect span. These are neither absolute viability thresholds nor confidence limits. Out-of-range EC values, extreme slopes, ill-conditioned Jacobians and sparsely sampled transitions are flagged. Perturbation stability is not evaluated.

The source-fit catalog uses predefined descriptive filters: finite parameters, EC50 > 0, 0.75 ≤ R² ≤ 1 and 1 < |slope| ≤ 20. These preserve the historical screening policy, with sequential exclusion counts; the slope cap is an application choice, not a biological bound. Ranking uses Screening Score = |slope| × R², then stable identifier tie-breaking. Source-fit convergence, dose coverage and raw-data agreement are unverified. Refits are separate from source-fit catalog values.

Categories use predefined slope and fit-quality thresholds. They do not indicate statistical significance. |slope| ≥ 2 and R² ≥ 0.90: steep response, strong fit; |slope| ≥ 2 and 0.75 ≤ R² < 0.90: steep response, moderate fit. Other eligible fits retain a descriptive category. Missing parameters remain unavailable.

## Machine learning and evaluation

TabPFN runs only when config/tabpfn.json specifies a local checkpoint, its SHA-256, explicit model/package versions, device, precision, random seed and feature representation. Authentication never selects a model. Supported representations are log10(dose) alone or explicitly configured log10(dose) plus dose; neither is assumed superior. The available v2 checkpoint is explicitly configured on CPU and its SHA-256 matches the publisher record. The selected-curve prediction was executed; representation comparisons have not run. Package version and model release are distinct identifiers.

TabPFN conditions on the selected curve's observed examples; predictions are not zero-shot measurements. Predictions are cached by observations, prediction grid and model configuration. Cursor annotations occupy a separate reactive cell and do not retrain the model. The public interface includes Prior Labs attribution and links to the v2 weight license. Prediction exports identify the condition, model version, checkpoint hash and features. A model worker failure disables that layer while retaining observations and the Hill fit. No CUDA/CPU substitution occurs.

Held-out concentration evaluation uses reproducible shuffled K-fold splits of distinct concentrations (seed 42). All observations at a concentration stay together. Each fold fits only training observations; no full-data region determines evaluation. Fold eligibility is checked separately. Model failures produce missing predictions with recorded reasons. Segmented regression failures never become mean predictions; the mean-training-response baseline is separate. Segmented regression uses one breakpoint, 20 package bootstrap restarts and the fixed seed, without reporting inferential intervals.

Primary metric: pooled held-out observation RMSE; more observations receive more weight. Individual-model RMSE includes its successful predictions; common RMSE compares the same observations successfully predicted by every requested model. Counts and failure rates accompany metrics. Common-observation joins preserve left observation order so repeated floating-point reductions are stable. No automatic superiority claim, transition-zone analysis or unseen-combination generalization claim is made.

## Imaging

Images are displayed as contrast-adjusted RGB composites (actin red, tubulin green, DAPI blue) using each channel's 1st and 99.5th percentiles. Raw image data are unchanged. The control is the first DMSO field by stable identifier in the same plate and replicate group. Matching beyond those metadata is unverified; absent controls are explicit.

Legacy embeddings and UMAP coordinates remain unused. Fresh DINO ViT-S/8 ImageNet baseline embeddings cover all 240 image fields in plate Week1_22123. Official weights were downloaded and matched the cached SHA-256; official architecture sources were verified byte-for-byte. Contrast-adjusted RGB composites use actin, tubulin and DAPI, then resize the short edge to 256, center-crop to 224 and apply ImageNet channel means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225]. Inference used CPU float32, batch size 2, seed 42 and four Torch threads. CUDA can be explicitly selected; this run had no available CUDA device. The model is a 384-dimensional image-field representation, not a single-cell measurement or Cell-DINO representation. Transfer from natural-image pretraining to these composites is exploratory and has not been biologically validated.

UMAP uses L2-normalized embeddings, cosine distance, n_neighbors=15, min_dist=0.1, seed 42 and n_jobs=1; its installed version and input hashes are recorded in the projection manifest. Stable plate/filename identifiers join coordinates to metadata. The loader checks all manifest inputs, including extraction code and architecture sources. An unrelated benchmark join-order repair required re-verification; the initial projection manifest is preserved and the cache bytes are unchanged. Points link to microscopy; compound/dose colors remain within BBBC021. UMAP can distort distances; static dose-ordered centroids are not time trajectories.

Cell-DINO weights are absent and require access through Meta. The available fluorescence channels do not establish compatibility with the five-channel Cell Painting checkpoint. No zero-padding or silent model substitution is performed. The available map is labeled DINO baseline throughout.

## External comparison

Cross-dataset comparison remains unavailable until source records, metric definitions and reviewed mappings are configured. GDSC release identity is recovered; CTRPv2 processing provenance remains unresolved. Ingestion requires source identifiers, unit conversion, quality and censoring metadata; ambiguous mappings, repeated conditions and invalid values are excluded or rejected explicitly. Matches require canonical compound/cell identifiers and identical potency metric definitions. EC50 and IC50 remain separate. No legacy handwritten validation values are consumed. No correlation, confidence interval or replication classification is currently supported. The historical validation table was hand-entered and is not a verified Step 17.1 result. GDSC source authentication alone does not establish cross-dataset matches.

## Limitations

Static dose–response curves do not establish bistability, hysteresis or irreversible state switching. PRISM viability and BBBC021 morphology are distinct assays without paired experimental conditions. Model predictions are not observations. High R² does not establish a mechanism. Heuristic rankings do not constitute statistical significance. Assay differences limit cross-platform potency interpretation. PRISM/CTRP release identity and independent experimental units remain unresolved; file hashes provide reproducibility, not independent source authentication.
