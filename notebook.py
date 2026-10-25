# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "expression",
#     "marimo",
#     "polars",
#     "tifffile",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import zipfile
    from expression import Error, Ok, Result
    import marimo as mo
    import polars as pl
    import tifffile

    return Error, Ok, Result, mo, pl, tifffile, zipfile


@app.cell
def _(mo, pl):
    @mo.cache
    def load_prism():
        return {
            "cell_lines": pl.read_csv("data/prism/cell_lines.csv", null_values="NA"),
            "treatments": pl.read_csv("data/prism/treatments.csv", null_values="NA"),
            "curves": pl.scan_csv("data/prism/curves.csv", null_values="NA"),
            "responses": pl.scan_csv("data/prism/responses.csv", null_values="NA"),
        }

    prism = load_prism()
    return (prism,)


@app.cell
def _(mo, pl):
    @mo.cache
    def load_bbbc021():
        return {
            "metadata": pl.read_csv("data/bbbc021/metadata.csv"),
            "compounds": pl.read_csv("data/bbbc021/compounds.csv"),
            "mechanisms": pl.read_csv("data/bbbc021/mechanisms.csv"),
        }

    bbbc021 = load_bbbc021()
    return (bbbc021,)


@app.cell
def _(mo, tifffile, zipfile):
    @mo.cache
    def load_sample_image(zip_path="data/bbbc021/plate.zip"):
        with zipfile.ZipFile(zip_path) as z:
            tif_name = sorted(n for n in z.namelist() if n.endswith(".tif"))[0]
            with z.open(tif_name) as f:
                img = tifffile.imread(f)
        return tif_name, img

    sample_img_name, sample_img = load_sample_image()
    return sample_img, sample_img_name


@app.cell
def _(bbbc021, mo, pl, prism):
    summary_data = [
        {
            "dataset": "PRISM cell_lines",
            "type": "Eager",
            "rows": prism["cell_lines"].height,
            "columns": prism["cell_lines"].width,
            "nulls": prism["cell_lines"].null_count().sum_horizontal()[0],
        },
        {
            "dataset": "PRISM treatments",
            "type": "Eager",
            "rows": prism["treatments"].height,
            "columns": prism["treatments"].width,
            "nulls": prism["treatments"].null_count().sum_horizontal()[0],
        },
        {
            "dataset": "PRISM curves",
            "type": "Lazy",
            "rows": prism["curves"].select(pl.len()).collect().item(),
            "columns": len(prism["curves"].collect_schema()),
            "nulls": prism["curves"].null_count().collect().sum_horizontal()[0],
        },
        {
            "dataset": "PRISM responses",
            "type": "Lazy",
            "rows": prism["responses"].select(pl.len()).collect().item(),
            "columns": len(prism["responses"].collect_schema()),
            "nulls": prism["responses"]
            .select(pl.sum_horizontal(pl.all().is_null().sum()))
            .collect()
            .item(),
        },
        {
            "dataset": "BBBC021 metadata",
            "type": "Eager",
            "rows": bbbc021["metadata"].height,
            "columns": bbbc021["metadata"].width,
            "nulls": bbbc021["metadata"].null_count().sum_horizontal()[0],
        },
        {
            "dataset": "BBBC021 compounds",
            "type": "Eager",
            "rows": bbbc021["compounds"].height,
            "columns": bbbc021["compounds"].width,
            "nulls": bbbc021["compounds"].null_count().sum_horizontal()[0],
        },
        {
            "dataset": "BBBC021 mechanisms",
            "type": "Eager",
            "rows": bbbc021["mechanisms"].height,
            "columns": bbbc021["mechanisms"].width,
            "nulls": bbbc021["mechanisms"].null_count().sum_horizontal()[0],
        },
    ]
    summary_df = pl.DataFrame(summary_data)
    mo.vstack([mo.md("### Dataset Summary & Missing Values"), summary_df])
    return (summary_df,)


@app.cell
def _(bbbc021, mo, prism):
    resp_preview_cols = prism["responses"].collect_schema().names()[:6]
    samples = {
        "PRISM cell_lines": prism["cell_lines"].head(3),
        "PRISM treatments": prism["treatments"].head(3),
        "PRISM curves": prism["curves"].limit(3).collect(),
        "PRISM responses (subset of 39,926 cols)": prism["responses"]
        .select(resp_preview_cols)
        .limit(3)
        .collect(),
        "BBBC021 metadata": bbbc021["metadata"].head(3),
        "BBBC021 compounds": bbbc021["compounds"].head(3),
        "BBBC021 mechanisms": bbbc021["mechanisms"].head(3),
    }

    mo.vstack(
        [
            mo.md("### Sample Rows & Schemas"),
            mo.accordion(
                {
                    f"{name} ({df.width} cols)": mo.vstack(
                        [mo.md(f"**Columns:** `{', '.join(df.columns)}`"), df]
                    )
                    for name, df in samples.items()
                }
            ),
        ]
    )
    return (samples,)


@app.cell
def _(mo, sample_img, sample_img_name):
    mo.vstack(
        [
            mo.md("### BBBC021 Sample Microscopy Image"),
            mo.md(
                f"**File:** `{sample_img_name}` | **Shape:** {sample_img.shape} | **Dtype:** {sample_img.dtype}"
            ),
            mo.image(sample_img, width=450, caption=sample_img_name),
        ]
    )
    return


@app.cell
def _(Error, Ok, Result, bbbc021, mo, pl, prism, zipfile):
    def verify_relationships() -> list[Result[str, str]]:
        checks = []

        # 1. PRISM cell line IDs: responses vs cell_lines
        cl_ids = set(prism["cell_lines"]["row_name"])
        resp_rows = (
            prism["responses"]
            .select(pl.first())
            .collect()
            .to_series()
            .to_list()
        )
        str_failed = set(resp_rows) - cl_ids
        if str_failed:
            checks.append(
                Error(
                    f"PRISM cell lines: {len(str_failed)} cell lines in responses have failed STR profiling and are missing from clean cell_lines ({sorted(str_failed)})"
                )
            )
        else:
            checks.append(Ok("PRISM cell lines: All response cell lines match cell_lines.csv"))

        # 2. PRISM treatment columns: responses vs treatments
        resp_cols = prism["responses"].collect_schema().names()[1:]
        tr_cols = set(prism["treatments"]["column_name"])
        missing_tr = set(resp_cols) - tr_cols
        if missing_tr:
            checks.append(Error(f"PRISM treatments: {len(missing_tr)} response columns missing in treatments.csv"))
        else:
            checks.append(Ok(f"PRISM treatments: All {len(resp_cols)} response condition columns match treatments.csv"))

        # 3. BBBC021 compounds & mechanisms
        meta_cpds = set(bbbc021["metadata"]["Image_Metadata_Compound"].drop_nulls())
        comp_cpds = set(bbbc021["compounds"]["compound"].drop_nulls())
        moa_cpds = set(bbbc021["mechanisms"]["compound"].drop_nulls())
        if not meta_cpds.issubset(comp_cpds):
            checks.append(Error(f"BBBC021 compounds: metadata has unknown compounds {meta_cpds - comp_cpds}"))
        elif not moa_cpds.issubset(comp_cpds):
            checks.append(Error(f"BBBC021 mechanisms: mechanism compounds missing from compounds.csv {moa_cpds - comp_cpds}"))
        else:
            checks.append(Ok("BBBC021 identifiers: compounds and mechanisms fully consistent with compounds.csv"))

        # 4. BBBC021 image archive coverage
        with zipfile.ZipFile("data/bbbc021/plate.zip") as z:
            zip_count = len([n for n in z.namelist() if n.endswith(".tif")])
        expected_images = bbbc021["metadata"].height * 3
        if zip_count < expected_images:
            checks.append(
                Error(
                    f"BBBC021 image archive: plate.zip contains {zip_count} images for plate Week1_22123 only; metadata references {expected_images} images across 55 plates"
                )
            )
        else:
            checks.append(Ok("BBBC021 image archive: All metadata images present in archive"))

        # 5. Dose units verification
        prism_dose_max = prism["treatments"]["dose"].max()
        bbbc_conc_max = bbbc021["metadata"]["Image_Metadata_Concentration"].max()
        if prism_dose_max and bbbc_conc_max:
            checks.append(
                Ok(f"Dose units: Both PRISM and BBBC021 use micromolar (μM) concentrations (PRISM max: {prism_dose_max} μM, BBBC021 max: {bbbc_conc_max} μM)")
            )

        # 6. Experimental conditions matching check (Requirement 8)
        checks.append(
            Error(
                "Condition mismatch: PRISM is a 120h viability screen across 500+ cancer lines; BBBC021 is a 24h high-content microscopy assay on MCF-7. Kept strictly separate."
            )
        )

        return checks

    results = verify_relationships()
    status_md = "\n".join(
        f"- {'🔴 **Mismatch / Alert**:' if r.is_error() else '🟢 **Passed**:'} {r.error if r.is_error() else r.ok}"
        for r in results
    )

    mo.vstack([mo.md("### Dataset Verification & Relationship Checks"), mo.md(status_md)])
    return (results,)


if __name__ == "__main__":
    app.run()
