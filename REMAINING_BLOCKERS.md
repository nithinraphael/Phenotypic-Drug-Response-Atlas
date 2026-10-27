# Remaining blockers

| Missing input / evidence | Affected feature | Why unresolved / required action | Public behavior |
|---|---|---|---|
| Authoritative acquisition/release records for local PRISM files | Source release attribution | Recover upstream file/version identifiers and checksums. Current hashes identify local bytes only; original script supplies the expected source family | Descriptive local-data analysis available; exact release/acquisition not claimed |
| Independent replicate structure and calibrated slope test | Formal inference / uncertainty | Identify biological versus technical units, screen/plate structure, identifiable model, null, valid resampling/calibration and complete testing family | Significance and uncertainty layers removed |
| Per-record source convergence/bounds, raw-data agreement and dose coverage for the full catalog | Atlas-wide fit verification | Refit or audit source fits against raw observations with their screen structure. This repair reproduces a selected-condition refit, not 51,763 refits | Catalog clearly labeled source fits; convergence/coverage unverified |
| TabPFN v3.5 checkpoint; broader model/feature evaluation | v3.5 and representation comparisons | Cached v2 weights match the publisher SHA-256 and are configured explicitly on CPU. v3.5 is not configured; feature alternatives need the same evaluation protocol | Explicit v2 only, on request; no authentication-dependent model substitution |
| Provenance for legacy cached embeddings | UMAP / morphology distances | Recover exact generating model, weights, image inputs and preprocessing, or deliberately recompute with verified channel-compatible model. Current dropdown cannot establish historical provenance | Legacy embeddings/UMAP rejected; newly computed, explicitly labeled DINO baseline map is available |
| Cell-DINO checkpoint and evidence supporting three available channels | Cell-DINO extraction | Checkpoint absent; zero-padding missing biological channels is not established as compatible | Unsupported extraction removed; raw microscopy available |
| CTRP source version/transformations and external identifier crosswalks | External quantitative comparison | GDSC2 release 8.2 is authenticated and its natural-log micromolar IC50 definition recovered. CTRP processing provenance and reviewed crosswalks remain absent | Exact requested unavailable message; no historical correlations/counts shown |
| Reviewed chemical and cell-line crosswalks; assay/bound/censoring metadata | External matching and potency interpretation | Resolve synonyms/salts/stereochemistry, dataset IDs, metric equivalence and repeated conditions. No display-name-only joins | Ambiguous/unconfigured records excluded; EC50 and IC50 remain separate |
| Additional acquisition/control metadata | Fully matched microscopy controls | Current matching uses same plate and replicate group. Confirm finer batch/acquisition compatibility before asserting equivalence | Explicitly limited control matching; no unrelated-image fallback |
| Defined numerical perturbation study | Fit perturbation stability | Current diagnostics assess sampling and Jacobian conditioning; no perturbation experiment executed | Marked Not evaluated |

The executed benchmark covers one curve, three folds and only two common successful predictions. It cannot establish model superiority, population performance or generalization to unseen compound–cell-line combinations. A declared broader evaluation sample with suitable concentration coverage and failure analysis is required.

Raw files remain unchanged. Historical outputs remain quarantined. None of these missing inputs is replaced by synthetic observations or guessed provenance.


## Step 18 tooling update

Browser tooling was found in the existing cache. Desktop/mobile Chromium interactions, real prediction and evaluation controls, and native SVG/HTML downloads have now been executed. Full marimo CLI checks pass outside the sandbox, so the earlier CLI and browser-tooling rows are superseded for the public workflow. Cross-browser/screen-reader testing and static Kaleido PNG generation remain unexecuted. External-data provenance, embedding attribution and statistical-uncertainty limitations are unchanged. See STEP18_REDESIGN.md.

## Step 18.1 status

Recovered: official GDSC2 release identity, DINO baseline checkpoint/source identity, fresh 240-field DINO/UMAP cache, and real TabPFN inference/export connections. No approved Cell-DINO checkpoint path or URL was supplied. Meta's Cell-DINO access form and supported channel preprocessing are required; the baseline is never labeled Cell-DINO. CUDA is unavailable on this execution host, so GPU memory use and GTX 1650 behavior remain untested. Cross-dataset correlations, confidence intervals and replication classifications remain withheld until chemical/cell-line crosswalks, CTRP transformations and comparable metric definitions are verified.

## Step 19 limitations

- Matched manually reviewed BBBC021 cell masks/image mapping are missing. Segmentation IoU/Dice and rigorous error rates cannot be calculated; UI labels masks as predictions. Zenodo mapping remains unverified.
- CellProfiler is not configured/executed. No baseline comparison claimed; requires a verified pipeline and matched inputs.
- Optical pixel size is missing. Micron measurements suppressed; pixels retained.
- Independent biological replication is unverified. Inferential transition tests/intervals suppressed; per-well descriptive summaries retained.
- Generic ImageNet DINO supports exploratory crops only. Cell-DINO-specific claims remain disabled.
- 180 additional image fields were not processed. Optional explicit new-run command supports them; current scope is 60 fields.
- Predicted masks can merge or fragment cells; 258 retained masks have disconnected components. Manual review/reference masks are required before treating object-level populations as accurate cell counts.
