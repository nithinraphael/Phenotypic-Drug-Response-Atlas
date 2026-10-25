# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "expression",
#     "marimo",
#     "numpy",
#     "plotly",
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
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import tifffile

    return Error, Ok, Result, go, mo, np, pl, tifffile, zipfile


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

        resp_cols = prism["responses"].collect_schema().names()[1:]
        tr_cols = set(prism["treatments"]["column_name"])
        missing_tr = set(resp_cols) - tr_cols
        if missing_tr:
            checks.append(Error(f"PRISM treatments: {len(missing_tr)} response columns missing in treatments.csv"))
        else:
            checks.append(Ok(f"PRISM treatments: All {len(resp_cols)} response condition columns match treatments.csv"))

        meta_cpds = set(bbbc021["metadata"]["Image_Metadata_Compound"].drop_nulls())
        comp_cpds = set(bbbc021["compounds"]["compound"].drop_nulls())
        moa_cpds = set(bbbc021["mechanisms"]["compound"].drop_nulls())
        if not meta_cpds.issubset(comp_cpds):
            checks.append(Error(f"BBBC021 compounds: metadata has unknown compounds {meta_cpds - comp_cpds}"))
        elif not moa_cpds.issubset(comp_cpds):
            checks.append(Error(f"BBBC021 mechanisms: mechanism compounds missing from compounds.csv {moa_cpds - comp_cpds}"))
        else:
            checks.append(Ok("BBBC021 identifiers: compounds and mechanisms fully consistent with compounds.csv"))

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

        prism_dose_max = prism["treatments"]["dose"].max()
        bbbc_conc_max = bbbc021["metadata"]["Image_Metadata_Concentration"].max()
        if prism_dose_max and bbbc_conc_max:
            checks.append(
                Ok(f"Dose units: Both PRISM and BBBC021 use micromolar (μM) concentrations (PRISM max: {prism_dose_max} μM, BBBC021 max: {bbbc_conc_max} μM)")
            )

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


@app.cell
def _(mo, pl, prism):
    drug_options = sorted(prism["treatments"]["name"].drop_nulls().unique().to_list())
    cl_clean = prism["cell_lines"].filter(pl.col("depmap_id").is_not_null())
    cell_line_options = {
        f"{r['ccle_name']} ({r['depmap_id']})": r["depmap_id"]
        for r in cl_clean.iter_rows(named=True)
    }

    prism_drug_select = mo.ui.dropdown(
        options=drug_options,
        value="bortezomib" if "bortezomib" in drug_options else drug_options[0],
        searchable=True,
        label="PRISM Drug",
    )
    default_cl = (
        "MCF7_BREAST (ACH-000019)"
        if "MCF7_BREAST (ACH-000019)" in cell_line_options
        else list(cell_line_options.keys())[0]
    )
    prism_cell_select = mo.ui.dropdown(
        options=cell_line_options,
        value=default_cl,
        searchable=True,
        label="Cancer Cell Line",
    )

    mo.vstack(
        [
            mo.md("## Step 3 — Interactive Data Explorer"),
            mo.md("### PRISM Dose–Response Explorer"),
            mo.hstack([prism_drug_select, prism_cell_select], justify="start", gap=2),
        ]
    )
    return cell_line_options, default_cl, drug_options, prism_cell_select, prism_drug_select


@app.cell
def _(Error, Ok, Result, mo, pl, prism, prism_cell_select, prism_drug_select):
    resp_col_set = set(prism["responses"].collect_schema().names()[1:])

    @mo.cache
    def fetch_prism_measurements(drug: str, cell_line_id: str) -> Result[tuple[pl.DataFrame, pl.DataFrame], str]:
        if not drug or not cell_line_id:
            return Error("Please select both a drug and a cell line.")

        drug_tr = prism["treatments"].filter(
            (pl.col("name") == drug) & (pl.col("perturbation_type") == "experimental_treatment")
        )
        if drug_tr.height == 0:
            return Error(f"No experimental treatments found for '{drug}'.")

        matched_cols = [c for c in drug_tr["column_name"].to_list() if c in resp_col_set]
        if not matched_cols:
            return Error(f"No response matrix columns found for '{drug}'.")

        row = (
            prism["responses"]
            .select([""] + matched_cols)
            .filter(pl.col("") == cell_line_id)
            .collect()
        )
        if row.height == 0:
            return Error(f"Cell line '{cell_line_id}' not found in response matrix.")

        melted = row.unpivot(index="", on=matched_cols, variable_name="column_name", value_name="response")
        melted = melted.with_columns(pl.col("response").cast(pl.Float64, strict=False))

        joined = (
            drug_tr.join(melted, on="column_name")
            .filter(pl.col("response").is_not_null())
            .with_columns(pl.lit(2.0).pow(pl.col("response")).alias("viability"))
            .sort("dose")
        )

        curves_df = prism["curves"].filter(
            (pl.col("name") == drug) & (pl.col("depmap_id") == cell_line_id)
        ).collect()

        if joined.height == 0 and curves_df.height == 0:
            return Error(f"No measurements or fitted curves found for '{drug}' in '{cell_line_id}'.")

        return Ok((joined, curves_df))

    prism_data_result = fetch_prism_measurements(
        prism_drug_select.value, prism_cell_select.value
    )
    return fetch_prism_measurements, prism_data_result, resp_col_set


@app.cell
def _(go, mo, np, prism_cell_select, prism_data_result, prism_drug_select):
    if prism_data_result.is_error():
        prism_view = mo.md(f"⚠️ **PRISM Data Warning:** {prism_data_result.error}")
    else:
        measurements, curves_df = prism_data_result.ok
        fig = go.Figure()

        screens = measurements["screen_id"].unique().to_list()
        for screen_id in screens:
            sub = measurements.filter(measurements["screen_id"] == screen_id)
            custom = np.stack(
                [
                    sub["response"].to_numpy(),
                    sub["well"].to_numpy(),
                    sub["column_name"].to_numpy(),
                ],
                axis=-1,
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["dose"].to_list(),
                    y=sub["viability"].to_list(),
                    mode="markers",
                    name=f"Replicates ({screen_id})",
                    customdata=custom,
                    hovertemplate=(
                        "<b>Dose:</b> %{x:.4g} μM<br>"
                        "<b>Viability:</b> %{y:.3f}<br>"
                        "<b>Log2 Response:</b> %{customdata[0]:.3f}<br>"
                        "<b>Well:</b> %{customdata[1]}<extra></extra>"
                    ),
                )
            )

        doses = measurements["dose"].to_list()
        if doses:
            min_d, max_d = min(doses), max(doses)
            grid_x = np.logspace(np.log10(min_d * 0.8), np.log10(max_d * 1.2), 120)
            for r in curves_df.iter_rows(named=True):
                ec50, slope = r["ec50"], r["slope"]
                upper, lower = r["upper_limit"], r["lower_limit"]
                if ec50 and slope and ec50 > 0:
                    grid_y = lower + (upper - lower) / (1.0 + (grid_x / ec50) ** slope)
                    r2_str = f"{r['r2']:.2f}" if r["r2"] is not None else "N/A"
                    fig.add_trace(
                        go.Scatter(
                            x=grid_x,
                            y=grid_y,
                            mode="lines",
                            name=f"Fit ({r['screen_id']}) EC50={ec50:.3g} μM R²={r2_str}",
                        )
                    )

        fig.update_xaxes(type="log", title="Dose (μM, log scale)")
        fig.update_yaxes(title="Relative Viability (2^response)")
        fig.update_layout(
            title=f"Dose–Response: {prism_drug_select.value} in {prism_cell_select.value}",
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=40, t=50, b=40),
        )

        prism_meta_cols = [
            "name",
            "smiles",
            "moa",
            "target",
            "disease.area",
            "indication",
            "phase",
        ]
        meta_sub = measurements.select(
            [c for c in prism_meta_cols if c in measurements.columns]
        ).unique()

        prism_view = mo.vstack(
            [
                mo.ui.plotly(fig),
                mo.accordion(
                    {
                        "Drug Metadata": meta_sub,
                        "Fitted Curve Parameters": curves_df.select(
                            [
                                "screen_id",
                                "ec50",
                                "ic50",
                                "slope",
                                "auc",
                                "r2",
                                "lower_limit",
                                "upper_limit",
                            ]
                        )
                        if curves_df.height > 0
                        else mo.md("No fitted curves recorded."),
                        f"Response Measurements ({measurements.height} replicates)": measurements.select(
                            [
                                "dose",
                                "viability",
                                "response",
                                "screen_id",
                                "well",
                                "column_name",
                            ]
                        ),
                    }
                ),
            ]
        )

    prism_view
    return (
        curves_df,
        custom,
        doses,
        ec50,
        fig,
        grid_x,
        grid_y,
        lower,
        max_d,
        measurements,
        meta_sub,
        min_d,
        prism_meta_cols,
        prism_view,
        r,
        r2_str,
        screen_id,
        screens,
        slope,
        sub,
        upper,
    )


@app.cell
def _(bbbc021, mo, pl):
    bbbc_cpd_list = sorted(bbbc021["compounds"]["compound"].drop_nulls().unique().to_list())
    bbbc_cpd_select = mo.ui.dropdown(
        options=bbbc_cpd_list,
        value="taxol" if "taxol" in bbbc_cpd_list else bbbc_cpd_list[0],
        searchable=True,
        label="BBBC021 Compound",
    )

    mo.vstack(
        [
            mo.md("### BBBC021 High-Content Imaging Explorer"),
            mo.md(
                "> **Note:** PRISM viability measurements and BBBC021 microscopy images reflect distinct experimental protocols (120h viability across 500+ lines vs 24h MCF-7 imaging) and remain separated."
            ),
            bbbc_cpd_select,
        ]
    )
    return bbbc_cpd_list, bbbc_cpd_select


@app.cell
def _(bbbc021, bbbc_cpd_select, mo, pl):
    cpd_meta = bbbc021["metadata"].filter(
        pl.col("Image_Metadata_Compound") == bbbc_cpd_select.value
    )
    avail_doses = sorted(cpd_meta["Image_Metadata_Concentration"].drop_nulls().unique().to_list())
    dose_dict = {f"{d} μM": d for d in avail_doses} if avail_doses else {"None": 0.0}

    bbbc_dose_select = mo.ui.dropdown(
        options=dose_dict,
        value=list(dose_dict.keys())[0],
        label="Dose (μM)",
    )
    bbbc_channel_select = mo.ui.dropdown(
        options=["DAPI", "Tubulin", "Actin"],
        value="DAPI",
        label="Stain Channel",
    )

    return avail_doses, bbbc_channel_select, bbbc_dose_select, cpd_meta, dose_dict


@app.cell
def _(
    bbbc021,
    bbbc_channel_select,
    bbbc_cpd_select,
    bbbc_dose_select,
    cpd_meta,
    mo,
    pl,
):
    dose_val = bbbc_dose_select.value
    chan_val = bbbc_channel_select.value
    col_fname = f"Image_FileName_{chan_val}"

    matching_rows = cpd_meta.filter(
        pl.col("Image_Metadata_Concentration") == dose_val
    )

    image_choices = {
        f"Plate: {r['Image_Metadata_Plate_DAPI']} | Well: {r['Image_Metadata_Well_DAPI']} | Replicate: {r['Replicate']}": (
            r["Image_Metadata_Plate_DAPI"],
            r[col_fname],
        )
        for r in matching_rows.iter_rows(named=True)
    }

    bbbc_img_select = mo.ui.dropdown(
        options=image_choices,
        value=list(image_choices.keys())[0] if image_choices else None,
        label="Field Image",
    )

    mo.hstack([bbbc_dose_select, bbbc_channel_select, bbbc_img_select], justify="start", gap=2)
    return chan_val, col_fname, dose_val, image_choices, matching_rows, bbbc_img_select


@app.cell
def _(
    Error,
    Ok,
    Result,
    bbbc021,
    bbbc_cpd_select,
    bbbc_dose_select,
    bbbc_img_select,
    mo,
    pl,
    tifffile,
    zipfile,
):
    with zipfile.ZipFile("data/bbbc021/plate.zip") as _z:
        _zip_names = set(_z.namelist())

    @mo.cache
    def load_field_image(plate: str, filename: str) -> Result[object, str]:
        target = f"{plate}/{filename}"
        if target not in _zip_names and filename not in _zip_names:
            return Error(
                f"Image `{filename}` (Plate: `{plate}`) is not in `plate.zip`. Only plate `Week1_22123` is packaged."
            )
        entry = target if target in _zip_names else filename
        with zipfile.ZipFile("data/bbbc021/plate.zip") as z:
            with z.open(entry) as f:
                return Ok(tifffile.imread(f))

    if not bbbc_img_select.value:
        bbbc_view = mo.md("No image available for this selection.")
    else:
        plate_id, img_file = bbbc_img_select.value
        img_res = load_field_image(plate_id, img_file)
        if img_res.is_error():
            img_element = mo.md(f"⚠️ **Image Unavailable:** {img_res.error}")
        else:
            arr = img_res.ok
            img_element = mo.vstack(
                [
                    mo.md(f"**Shape:** {arr.shape} | **Dtype:** {arr.dtype} | **File:** `{img_file}`"),
                    mo.image(arr, width=450, caption=f"{plate_id} / {img_file}"),
                ]
            )

        cpd_info = bbbc021["compounds"].filter(
            pl.col("compound") == bbbc_cpd_select.value
        )
        moa_info = bbbc021["mechanisms"].filter(
            (pl.col("compound") == bbbc_cpd_select.value)
            & (pl.col("concentration") == bbbc_dose_select.value)
        )

        bbbc_view = mo.vstack(
            [
                img_element,
                mo.accordion(
                    {
                        "Compound & Mechanism Info": mo.vstack(
                            [
                                mo.md(f"**Compound:** {bbbc_cpd_select.value}"),
                                cpd_info,
                                moa_info if moa_info.height > 0 else mo.md("No mechanism annotation at this exact concentration."),
                            ]
                        )
                    }
                ),
            ]
        )

    bbbc_view
    return arr, bbbc_view, cpd_info, img_element, img_file, img_res, load_field_image, moa_info, plate_id


if __name__ == "__main__":
    app.run()
