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
