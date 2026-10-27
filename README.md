# Phenotypic Drug-Response Atlas

marimo application for descriptive PRISM screening, per-screen dose–response fitting and BBBC021 microscopy viewing. Four public tabs: Discoveries, Drug Explorer, Cell Explorer, Validation. Methods and citations are available within Validation.

```sh
uv sync --locked
uv run marimo run notebook.py
```

Use `uv run marimo edit dev_notebook.py` for diagnostics. The complete pre-redesign interface, including raw tables and filter audits, is preserved at `audit_notebook.py` and can be opened with `uv run marimo edit audit_notebook.py`. `hello.py` launches the same public application.

Raw inputs belong under `data/prism` and `data/bbbc021`. No downloads occur during application import or execution. `uv run python scripts/download_data.py` explicitly downloads the configured PRISM files and one BBBC021 plate; it preserves existing files and writes an acquisition manifest. Existing files are not retroactively assigned a download date.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/check_notebooks.py
.venv/bin/python -m analysis.run --output reports/runs/NEW_RUN
```

The report command reproduces the full descriptive source-fit catalog and a concentration-held-out benchmark for a declared compound/cell/screen. Defaults select bortezomib BRD-K88510285-001-11-1, RKO ACH-000943 and its first available screen; override with `--compound-id`, `--cell-id`, `--screen`, `--folds`. Add `--include-tabpfn` to explicitly include configured pretrained predictions and evaluation. It refuses existing output directories. Reports contain observations, exclusions, predictions/failures, plots, hashes, configuration, software versions and code snapshot. Interactive HTML plots use the same chart builders as the application; browser toolbar supports image exports.

TabPFN v2 is explicitly configured on CPU using publisher-hash-matched cached weights; predictions run only on request. UMAP and external comparisons require additional verified local inputs and show unavailable states, with no replacement measurements or silent model substitutions. See [CONFIGURATION.md](CONFIGURATION.md), [METHODS.md](METHODS.md), [SCIENTIFIC_AUDIT.md](SCIENTIFIC_AUDIT.md), [TEST_RESULTS.md](TEST_RESULTS.md) and [REMAINING_BLOCKERS.md](REMAINING_BLOCKERS.md).

Historical research reports, figures and development code are quarantined under `legacy/unverified`. Their results are not accepted empirical evidence. Original raw data remain unchanged.

Step 18.1 restores a verified, explicitly labeled DINO ImageNet baseline map for 240 BBBC021 image fields, native microscopy linking and TabPFN prediction exports. GDSC release 8.2 is authenticated; cross-dataset comparisons still require reviewed identifier crosswalks and CTRP processing provenance. Cell-DINO requires approved weights and compatible channel preprocessing. See [METHODS.md](METHODS.md) and [REMAINING_BLOCKERS.md](REMAINING_BLOCKERS.md).

Cell Explorer also loads a verified individual-cell atlas: 5,857 retained predicted objects from 60 BBBC021 fields, with linked microscopy/crops, generic DINO embeddings, shared UMAP, dose summaries and raw exports. See [STEP19_RESULTS.md](STEP19_RESULTS.md) for actual inference, QC and limitations; [CONFIGURATION.md](CONFIGURATION.md) for explicit optional generation. Segmentation and embeddings never run automatically in the public interface.
