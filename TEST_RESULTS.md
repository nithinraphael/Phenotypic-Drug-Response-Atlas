# Test results

## Environment

Python 3.13.15; CPU execution; CUDA unavailable. Exact package versions are recorded in environment.json and the run manifest. Direct dependencies are pinned; uv.lock was updated offline without upgrading packages. pytest was absent; tests use standard-library unittest.

## Automated scientific and application tests

Command:

```sh
MPLCONFIGDIR=/tmp/atlas-matplotlib .venv/bin/python -m unittest discover -s tests -v > TEST_EXECUTION.log 2>&1
```

Final result: **61 executed, 61 passed, 0 failed, 0 skipped**, 35.488 seconds. Full named results: TEST_EXECUTION.log.

| Module | Executed |
|---|---:|
| test_application | 5 |
| test_benchmark | 7 |
| test_catalog | 6 |
| test_embeddings | 5 |
| test_fitting | 12 |
| test_models | 2 |
| test_prism | 5 |
| test_provenance | 1 |
| test_reporting | 1 |
| test_ui | 7 |
| test_validation | 10 |

Coverage includes known-parameter sigmoid recovery, constant/invalid inputs, repeated concentrations, noisy/steep fits, signed-slope normalization, degenerate/nonfinite parameters, out-of-range flags; category boundaries and missing values; ranking/ties/metadata/exclusions; log2 conversion and join cardinality; reproducible concentration splits, held-out response perturbation, failure counts and common-observation aggregation; source mappings/units/ambiguities/metric separation; shuffled/missing/duplicate embedding IDs, checkpoint/preprocessing/input mismatches and pickle rejection; report reconciliation/no overwrite; explicit model identity/device/worker failures; current compound/cell selections, missing PRISM state, five views, control matching and chart/table equality.

Programmatic public app execution completes all 22 cells with real local inputs. Development diagnostics completes all four cells and writes no reports unless an explicit report command is run. Tests confirm compound/cell changes reload corresponding observations and rebuild dependent selections. Cursor exclusion from model computation is verified from the actual reactive cell dependency signature; browser-driven event timing remains untested.

## Reproduced analysis

Command:

```sh
MPLCONFIGDIR=/tmp/atlas-matplotlib .venv/bin/python -m analysis.run --output reports/runs/final-2026-10-26 --include-tabpfn
```

Completed using actual local PRISM observations and explicitly configured, publisher-hash-matched TabPFN v2 weights. Scope: BRD-K88510285-001-11-1 / ACH-000943 / HTS002; seven usable observations, seventeen missing responses, three concentration-group folds. All model failures remain in held_out_predictions.csv.

| Model | Successful predictions | Failed folds | Individual RMSE | Common RMSE (2 observations) |
|---|---:|---:|---:|---:|
| Hill LL4 | 4/7 | 1/3 | 0.662869 | 0.255813 |
| Mean baseline | 7/7 | 0/3 | 0.746265 | 0.816801 |
| Segmented | 2/7 | 2/3 | 0.714270 | 0.714270 |
| TabPFN | 7/7 | 0/3 | 0.556963 | 0.726735 |

Individual RMSE values use different successful observations; they cannot establish a fair model ranking. Common metrics have only two observations. No statistical model-superiority claim follows. TabPFN also produced 120 finite grid predictions conditioned on the observed curve; these drive the exported dose-response chart. No feature-representation comparison was run.

Additional executed integrity checks: all sixteen original data-file hashes unchanged; all final output hashes match outputs.json; final manifest matches current input/software/configuration/analysis-code identities; all 39,925 response columns map to unique treatment identifiers; 240 plate fields reference 720 available image-channel files. Archive membership does not verify every TIFF payload.

## Other checks

| Command / check | Actual result |
|---|---|
| `.venv/bin/python tests/check_notebooks.py` | Passed: all 16 default installed marimo lint rules on each notebook, no diagnostics |
| `timeout 12 .venv/bin/python -m marimo check notebook.py dev_notebook.py` | Timed out, exit 124; not reported as passed. CLI async filesystem path stalls in this runtime; direct rule checks above complete |
| `.venv/bin/python -m compileall -q analysis tests notebook.py components.py dev_notebook.py hello.py scripts/download_data.py` | Passed syntax/import-bytecode compilation; not scientific validation |
| `git diff --check` | Passed |
| `uv lock --offline` | Completed, 137 packages resolved from existing cache; dependency versions unchanged |
| Impeccable detector on notebook.py, dev_notebook.py, components.py | Empty findings; mechanical scan only, not browser or scientific validation |
| Static audit | Classified matching lines in STATIC_AUDIT.json; executable pseudo-inference removed; historical/test/documentation occurrences retained deliberately |

## Checks not executed

- Browser E2E, responsive visual comparison and screen-reader tests: no browser binary or Playwright/Selenium installed. Programmatic UI tests are not a substitute for these.
- Static Kaleido SVG/PNG export: no Chrome/Chromium available. Self-contained interactive HTML figures were generated using shared chart builders; browser toolbar export remains available.
- DINO/Cell-DINO embedding extraction and empirical UMAP validation: cache checkpoint/preprocessing provenance unresolved; Cell-DINO checkpoint absent. Cached generic DINO weights do not authenticate old embeddings.
- External experimental validation: source releases, potency transformations and reviewed ID mappings are missing. Generic ingestion tests do not count as experimental validation.
- Formal slope inference/bootstrap intervals: independent experimental units and calibrated procedure not established; inference intentionally disabled.
- Atlas-wide model benchmarking, all-catalog refitting and perturbation stability: not executed. Current benchmark is one curve and cannot establish general performance.
- TabPFN v3.5 and alternative feature representation: not configured/evaluated. Executed pretrained analysis is explicitly v2, log10 dose, CPU.

No unittest tests were skipped. The items above are unavailable or unexecuted analyses, not passed tests. Early development smoke runs exposed an unsupported nested marimo embedding path; diagnostics was repaired to call shared analysis directly, and its final smoke test passes.


## Step 18 public-interface verification

The earlier results above remain the repair baseline. The redesigned public app now has four tabs; the old five-view workspace is preserved as audit_notebook.py.

- `MPLCONFIGDIR=/tmp/atlas-matplotlib .venv/bin/python -m unittest discover -s tests -v`: **65 passed, 0 failed, 0 skipped**. Current log: artifacts/step18/scientific-tests.log.
- `timeout 30 .venv/bin/marimo check notebook.py dev_notebook.py audit_notebook.py` executed outside the sandbox: **passed, exit 0**. The prior CLI timeout was specific to sandboxed asynchronous file operations.
- `.venv/bin/python tests/check_notebooks.py`: all 16 default rules pass on all three notebooks.
- `uv run --no-project --offline --with playwright python tests/browser_step18.py`: **17 browser-check groups passed** using the already cached Chromium binary. Desktop 1440 × 1050; mobile 390 × 844. Explicit assertions preserve logarithmic concentration axes and units after zoom, tab navigation and resizing.
- Real configured TabPFN predictions and current-condition held-out evaluation ran through the unchanged implementations. CSV, HTML, SVG and methods downloads were verified against their current scope. These checks do not establish model superiority.
- `MPLCONFIGDIR=/tmp/atlas-matplotlib .venv/bin/python audit_notebook.py`: preserved workspace executed successfully from the project root. Diagnostics remains unchanged.
- `git diff --check` and compilation passed. Raw input hashes and shared scientific/dependency sources remain unchanged; preservation.json records the original notebook copy hash.

Browser/render evidence and scoped design review are in artifacts/step18. All prior unexecuted-analysis limitations still apply except the browser/native-SVG and marimo-CLI checks now demonstrated here. Cross-browser, screen-reader and static Kaleido PNG evaluation were not run.
