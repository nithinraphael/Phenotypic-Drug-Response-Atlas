# Public atlas redesign

Public entry: notebook.py. Four tabs: Discoveries, Drug Explorer, Cell Explorer, Validation.

## Preserved research

Shared analysis, components.py, model configuration, dependencies, raw files and audited run outputs remain unchanged. The complete pre-redesign interface is preserved byte-for-byte in audit_notebook.py. dev_notebook.py remains the diagnostic entry. See artifacts/step18/preservation.json.

## Public interface

- Computed KPI cards, a source-fit potency/steepness figure and a compact selectable shortlist.
- Candidate selection initializes the PRISM compound, cell and screen. Imaging selectors remain independent because PRISM and BBBC021 observations are unpaired.
- Larger dose-response figures distinguish measured points, the Hill fit, dashed TabPFN predictions and the fitted effect span. Native SVG and interactive HTML downloads retain the plotted data.
- Side-by-side microscopy with explicit control matching and channel information. Unverified embedding maps, external comparisons and uncertainty intervals remain unavailable.
- Recorded benchmark results load only after input/artifact checks and metric reconciliation. Current-condition evaluation runs on request, with its scope shown separately from recorded results.
- Methods, citations, coverage and limitations use disclosures; no raw processing tables or environment dumps appear in the public workflow.

## Reactive and display repairs

Tab state persists when selections update. Native Plotly can restore only saved axis ranges on remount; axis type and titles now also live in the standard Plotly template so the concentration scale remains logarithmic across navigation, zoom/reset and mobile resizing. No custom JavaScript or scientific algorithm change was needed. Full catalog and figure downloads use deferred native callbacks. Model prediction remains cached separately from cursor annotations. Plots stay mounted to retain native interaction state.

## Verification

65 automated tests pass. Full marimo CLI checks pass outside the filesystem/network sandbox for notebook.py, dev_notebook.py and audit_notebook.py; all 16 default rules also pass directly for each notebook. The earlier sandbox-only CLI timeout is resolved by running the checks without that sandbox restriction.

17 browser-check groups passed in Chromium at 1440 × 1050 and 390 × 844: tab navigation; candidate/compound/cell/screen selection; real TabPFN inference; cursor and prediction persistence; current-observation, full-catalog, shortlist, HTML, SVG and methods downloads; chart zoom/reset; imaging condition/field selection; assay separation; live concentration benchmark; disclosures; responsive layout; no horizontal overflow or browser JavaScript errors. Log-axis type and both scientific axis titles are asserted after navigation and resizing.

Evidence: artifacts/step18/scientific-tests.log, browser-execution.log, browser-results.json and desktop/mobile PNG captures. The independent finish reviewer scored the scientific-axis and canvas fixes resolved; that verdict is scoped to those material fixes. The documenter checked all 14 primary captures and extracted DESIGN.md and .impeccable/design.json from the finished interface. Supplemental captures cover lower-page limitations and methods.

No pretrained weights were trained, no scientific findings or confidence intervals were invented, and no raw data or audited results were overwritten. The existing v2 model was conditioned through its unchanged prediction/evaluation implementation during browser testing. This UI verification does not establish scientific validity for unavailable analyses or general model performance.
