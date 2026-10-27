# Implementation changes

| File | Functions / previous behavior | Replacement and reason | Compatibility impact |
|---|---|---|---|
| notebook.py | Embedded pseudo-inference; fixed KPIs; pooled screens; automatic model fallback; positional UMAP; synthetic failure parameters | Thin shared-analysis interface; computed counts; identifier/screen selectors; unavailable states; separate prediction and cursor cells; five tabs retained | Old UI state and unsupported result layers removed; full catalog export now distinct from shortlist |
| components.py | fit_hill, predict_tabpfn, fixed validation/benchmark charts; residual confidence band | Fitting/model functions imported from shared modules; plotting consumes results; no uncertainty invention; all nearest-dose fields highlighted | fit_hill returns FitResult; build_drc_plot accepts fit; benchmark/validation builders require data |
| dev_notebook.py | Duplicate fitting, row bootstraps, inference, generated manual reports, synthetic figures and inserted Taxol metrics | Shared calculations in diagnostic notebook, configurable observation/fit inspection, software/audit display; no automatic artifact writing | Original experiments preserved in compressed archive, require deliberate recovery and scientific review before reuse |
| hello.py | Placeholder greeting | Starts the same marimo app | Entry point now useful |
| analysis/__init__.py | Absent | Package marker | New package |
| analysis/fitting.py | Three independent fit implementations and invented fallbacks | Validated LL4, finite parameters, explicit failed FitResult, exact R², diagnostics, effect-span EC values | Failed parameters are missing; no inferential intervals |
| analysis/prism.py | Name-based many-screen measurement selection | File checks, one-to-one treatment joins, raw-response preservation, per-ID/screen observations, counted exclusions and dataset summaries | Null/invalid observations remain exported but are excluded from fitting |
| analysis/catalog.py | Selected subset before inference; fixed tiers; polluted cache | Full source catalog ranking, finite/identifier filters, descriptive categories, stable ties, source row references and sequential exclusions | Source fit status explicitly unverified; legacy parquet ignored |
| analysis/benchmark.py | Fixed RMSE, zero defaults, full-data transition-zone leakage, segmented mean fallback | Concentration-group folds; training-only fits; retained failures; separate mean baseline; pooled and common-observation RMSE | No transition-zone analysis or model-superiority assertion |
| analysis/models.py | Authentication-selected pretrained release; repeated implicit initialization | Explicit local config/hash/package/device/features; isolated workers; shape/finiteness checks | No configured checkpoint means unavailable; no network download |
| analysis/model_worker.py | In-process model errors could kill application | Optional TabPFN / segmented workers with seed and explicit output; no segmented fallback | Worker failure is recorded; raw worker logs are not rendered |
| analysis/microscopy.py | First DMSO anywhere; unchecked archive members | Plate/replicate control rule; stable field IDs; explicit missing members; closed image streams; contrast adjustment only | No unrelated image substitution |
| analysis/embeddings.py | Positional UMAP and unverified Cell-DINO attribution | Strict provenance/hash/schema checks; safe arrays; ID joins; actual configured model label | All legacy caches disabled; no unsupported five-channel extraction |
| analysis/validation.py | Handwritten empirical dictionaries and fixed plots | Normalized source ingestion, reviewed crosswalks, unit conversion, exclusions/counts, compatible-metric joins and calculated correlations | Unconfigured sources unavailable; EC50/IC50 separated |
| analysis/provenance.py | File-existence cache checks | SHA-256, exact software/code/config identities, UTC manifests | Changed input/software/implementation invalidates derived caches |
| analysis/reporting.py | Notebook overwrote reports on execution | Reconciled tables/predictions, immutable new run directory, manifests, source snapshot and shared plot builders | Existing reports never overwritten; HTML plots available without Chrome |
| analysis/run.py | No reproducible report command | Explicit selected-condition analysis CLI; source-fit screening; observations, exclusions, diagnostics, benchmark and outputs | Small selected-curve benchmark, not an atlas-wide performance result |
| scripts/download_data.py | Syntax error; network on import; no provenance | Explicit main; preserves existing inputs; download records/checksums | Network only on explicit command; no new datasets downloaded during repair |
| pyproject.toml / uv.lock | Lower bounds, NumPy transitive | Exact installed direct versions, explicit NumPy; lock metadata updated offline | No package upgrades or model-version inference |
| tests/test_fitting.py, test_catalog.py, test_prism.py, test_benchmark.py | No tests | Numerical, edge-case, cardinality, leakage, category, failure and metric tests | Synthetic observations confined to tests |
| tests/test_validation.py, test_embeddings.py, test_provenance.py, test_reporting.py | No tests | Ambiguity/unit/cache identity/shuffling/pickle/reconciliation tests | No heavyweight models needed |
| tests/test_ui.py, test_application.py, test_models.py | No state smoke tests | Chart/table consistency, cursor dependency, controls, real-data app execution, changed selections and missing-data checks | Programmatic tests, not browser E2E coverage |
| tests/check_notebooks.py | CLI check stalls in this runtime | Runs all default installed marimo lint rules directly | Documents CLI timeout separately; no rules skipped |
| METHODS.md, CONFIGURATION.md, README.md | Incorrect methods / empty README | Running methods, limitations and explicit configuration / commands | Unsupported claims removed |
| INVENTORY.md, environment.json, data_inventory.json, model_inventory.json, STATIC_AUDIT.json | No baseline/provenance inventory | Inventory, actual package/device status and 16 local-file hashes | Local reproducibility does not authenticate release origin |
| SCIENTIFIC_AUDIT.md, DATA_PROVENANCE.md, TEST_RESULTS.md, REMAINING_BLOCKERS.md, this file | Absent | Required audit evidence and unresolved inputs | No claim of full validation |
| legacy/unverified/* | Publicly available questionable reports/figures | Original bytes quarantined with hashes; compressed development source preserved | Historical results excluded from public app |
| RESEARCH_REPORT.md, reports/README.md, figures/README.md | Historical conclusions / undocumented artifacts | Status pointers to current reproducible runs and quarantine | Old reports must not be cited as verified findings |
| config/tabpfn.json | Authentication selected a model implicitly | Explicit publisher-hash-matched v2 checkpoint, CPU, precision, single log-dose feature and seed | Machine-specific path; absent/incompatible checkpoint gives unavailable status |
| reports/runs/* | Absent | Explicitly executed analytical outputs and manifests | Each run identifies its sample and failures; no global benchmark implication |

Additional evidence files: TEST_EXECUTION.log records actual named test outcomes; reports/LATEST.md points only to the current completed run. __marimo__/session/notebook.py.json was moved into quarantine to discard stale UI state. config/tabpfn.json and model_inventory.json record the verified v2 identity and the separately unverified generic DINO caches. No credentials are stored in these files.


## Step 18: public atlas redesign

| File | Change | Reason / compatibility |
|---|---|---|
| notebook.py | Four public tabs, persistent tab selection, linked candidate selectors, curated scientific views, deferred downloads, verified recorded benchmark display | Public navigation changes; all numerical implementations and scientific limits remain intact |
| public_ui.py | Presentation helpers, role colors, large plots, escaped state messages and benchmark artifact verification | No new statistical algorithm; Plotly template axis metadata prevents native remount from losing log scales/units |
| atlas.css | Small native-control, typography, canvas and responsive-grid stylesheet | Desktop/mobile visual system; no application JavaScript |
| audit_notebook.py | Byte-identical copy of the pre-redesign public notebook | Retains all former tables, controls, calculations and methods; root location preserves imports and relative data paths |
| tests/test_application.py, tests/test_ui.py | Updated navigation expectations and HTML test doubles | Existing scientific assertions retained |
| tests/test_public_ui.py | Numerical styling invariance, source-fit values, provenance and escaped HTML tests | Prevents a presentation change from changing research values |
| tests/browser_step18.py | Live Chromium interaction and responsive regression script | Uses cached Playwright/Chromium outside app dependencies; no training or synthetic research data |
| tests/check_notebooks.py | Includes preserved audit notebook | All notebook checks cover the development entry too |
| README.md | Four-tab workflow and development entry commands | Public and audit workspace entry points are explicit |
| PRODUCT.md, DESIGN.md, .impeccable/* | Product brief, direction record and extracted design tokens | Design documentation only; no new scientific claim |
| STEP18_REDESIGN.md, artifacts/step18/* | Actual test, screenshot, preservation and scoped review evidence | Verification artifacts kept separate from audited scientific reports |
| TEST_RESULTS.md, REMAINING_BLOCKERS.md | Append current UI checks and mark resolved tooling limitations | Earlier scientific limitations and repair results retained |

Scientific analysis modules, components.py, dependencies, model configuration, raw observations and audited run outputs were not changed.

## Step 18.1

| File | Change and reason | Compatibility |
|---|---|---|
| notebook.py | Connect verified image projection; native point selection updates compound/dose/field; show exact external ingestion failure and source inventory; add TabPFN loading, attribution, success and CSV export | Four public tabs retained; no Cell-DINO substitution or unverified comparisons |
| public_ui.py | Wrap existing UMAP builder with dose/compound color and stable field metadata; resolve native trace/point indices to field IDs; load newly reconciled recorded benchmark | Existing figure/numerical builders unchanged |
| image_projection.py | Explicit, bounded DINO baseline extraction and UMAP generation; complete input verification before cache use | New verified cache only; legacy caches unchanged; CPU run, explicit CUDA option |
| external_evidence.py | Hash-check source records and compute availability/counts without inferring cross-dataset matches | Source inventory is separate from validation results |
| config/dino_baseline.json | Verified checkpoint and official executable source hashes, baseline identity/license | Machine-local checkpoint path |
| config/external_sources.json | Official GDSC release URL/hash and unresolved local CTRP provenance | Does not enable comparison by itself |
| analysis/benchmark.py | Preserve left observation order in common-prediction join, fixing nondeterministic floating-point reduction order | Counts unchanged; metrics unchanged within 1e-14; prior reports preserved |
| reports/runs/step18-1-2026-10-26 | Reaggregate hash-verified stored predictions with current code and explicit lineage | No model fitting/inference rerun for this artifact |
| tests/test_application.py | Expect verified baseline projection and correct model identity | Existing scientific assertions retained |
| tests/test_benchmark.py | Repeated aggregation determinism regression | No model performance winner assumed |
| tests/test_restored_features.py | Real field identities, native selection payload, color layers, manifest inputs, source counts and hash rejection | Synthetic data confined to explicitly labeled tests |
| tests/browser_step18.py | Real TabPFN inference/export, UMAP selection, microscopy link, inventories and responsive interactions | Uses separately cached Playwright/Chromium |
| METHODS.md, DATA_PROVENANCE.md, CONFIGURATION.md, model_inventory.json, REMAINING_BLOCKERS.md | Record recovered provenance, actual computation, license and unresolved evidence | Prior claims narrowed to executed scope |

Step 18.1 documentation also updates README.md, DESIGN.md, reports/LATEST.md, TEST_RESULTS.md and STEP18_1_RECOVERY.md. These point to current methods, responsive map behavior, preserved report lineage, actual verification and remaining blockers. artifacts/step18_1 contains execution logs, screenshots, source authentication, license, preservation audit and the partial recovery manifest. No atlas.css change remains.

## Step 19: individual-cell analysis

- `single_cells.py`: new explicit pretrained inference command, stable cell metadata, raw crops, per-field receipts, strict cache loading, equal-weight descriptive summaries and shared UMAP. No prior scientific implementation replaced. Optional inference dependencies stay outside the base environment.
- `single_cell_ui.py`: cell map, linked source overlays/crops, dose summaries, QC and provenance/raw exports. Public rendering consumes verified analytical outputs.
- `notebook.py`: integrates single-cell controls and selection state into Cell Explorer; isolates existing image-field analysis in a disclosure; optional `--cell-atlas` selects a verified run. Image-field calculations remain independent of cell/color changes.
- `config/single_cells.json`, `config/segmentation-requirements.txt`: explicit verified pretrained identities and pinned optional inference environment. Existing base dependency manifest/lock unchanged.
- `tests/test_single_cells.py`, `tests/test_application.py`: scientific behavior and real atlas application coverage. `tests/validate_cell_atlas.py` checks actual raw pixels, IDs, masks, embeddings and population recalculation. `tests/browser_step19.py` tests desktop interactions/exports without running inference.
- `METHODS.md`, `CONFIGURATION.md`, `README.md`, `DESIGN.md`, audit/provenance/blocker/test documents and `STEP19_RESULTS.md`: actual sampling, processing, reproducibility and limitations.
- `artifacts/step19`: execution logs, model-source receipts, integrity results and desktop verification artifacts. Generated masks/crops/weights stay under ignored data paths; preserve them when moving the project.

Compatibility: base dependencies, PRISM analysis, prior caches and development notebooks preserved. Cell Explorer requires the verified local single-cell run; absent or incompatible artifacts produce an unavailable state.
