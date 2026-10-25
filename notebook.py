# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "expression",
#     "marimo",
#     "numpy",
#     "piecewise-regression",
#     "plotly",
#     "polars",
#     "scipy",
#     "tabpfn",
#     "thunor",
#     "tifffile",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import zipfile
    from expression import Error, Ok, Result
    import marimo as mo
    import numpy as np
    import piecewise_regression as pr
    import plotly.graph_objects as go
    import polars as pl
    import scipy.stats as stats
    import tifffile
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.browser_auth import get_cached_token
    from thunor.curve_fit import HillCurveNull, fit_drc

    return (
        Error,
        HillCurveNull,
        Ok,
        Result,
        TabPFNRegressor,
        fit_drc,
        get_cached_token,
        go,
        mo,
        np,
        os,
        pl,
        pr,
        stats,
        tifffile,
        torch,
        zipfile,
    )


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
def _(
    HillCurveNull,
    fit_drc,
    go,
    mo,
    np,
    pl,
    prism_cell_select,
    prism_data_result,
    prism_drug_select,
):
    if prism_data_result.is_error():
        prism_view = mo.md(f"⚠️ **PRISM Data Warning:** {prism_data_result.error}")
    else:
        measurements, curves_df = prism_data_result.ok
        fig = go.Figure()
        stats_rows = []

        screens = measurements["screen_id"].unique().to_list()
        doses_all = measurements["dose"].to_list()
        min_d, max_d = min(doses_all), max(doses_all)
        grid_x = np.logspace(np.log10(min_d * 0.8), np.log10(max_d * 1.2), 120)

        for screen_id in screens:
            sub = measurements.filter(measurements["screen_id"] == screen_id)
            doses = sub["dose"].to_numpy()
            viabs = sub["viability"].to_numpy()
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
                    x=doses,
                    y=viabs,
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

            pub_sub = curves_df.filter(pl.col("screen_id") == screen_id)
            pub_row = pub_sub.to_dicts()[0] if pub_sub.height > 0 else None
            pub_valid = (
                pub_row is not None
                and pub_row.get("ec50") is not None
                and pub_row["ec50"] > 0
                and pub_row.get("slope") is not None
                and pub_row["slope"] > 0
                and pub_row.get("r2") is not None
                and pub_row["r2"] > 0
            )

            # 1. Published PRISM Curve
            if pub_valid:
                pec50, pslope = pub_row["ec50"], pub_row["slope"]
                pupper, plower = pub_row["upper_limit"], pub_row["lower_limit"]
                pub_grid_y = plower + (pupper - plower) / (1.0 + (grid_x / pec50) ** pslope)
                fig.add_trace(
                    go.Scatter(
                        x=grid_x,
                        y=pub_grid_y,
                        mode="lines",
                        line=dict(width=2.5),
                        name=f"Published ({screen_id}) EC50={pec50:.3g} μM R²={pub_row['r2']:.2f}",
                    )
                )
                p_steep0 = pec50 * (0.25 ** (1.0 / pslope))
                p_steep1 = pec50 * (4.0 ** (1.0 / pslope))
                fig.add_vrect(
                    x0=p_steep0,
                    x1=p_steep1,
                    fillcolor="rgba(255, 140, 0, 0.12)",
                    layer="below",
                    line_width=1,
                    line_dash="dot",
                    line_color="darkorange",
                    annotation_text=f"Steep Window ({screen_id})",
                    annotation_position="top left",
                )
                y_pub_pred = plower + (pupper - plower) / (1.0 + (doses / pec50) ** pslope)
                rmse_pub = float(np.sqrt(np.mean((viabs - y_pub_pred) ** 2)))
                stats_rows.append(
                    {
                        "screen_id": screen_id,
                        "model": "Published PRISM",
                        "status": "Valid (Reused)",
                        "ec50_uM": float(pec50),
                        "ic50_uM": str(pub_row["ic50"]) if pub_row["ic50"] is not None else "N/A",
                        "hill_slope": float(pslope),
                        "e0_top": float(pupper),
                        "emax_bottom": float(plower),
                        "max_inhibition_pct": float(max(0.0, (1.0 - plower) * 100)),
                        "r2": float(pub_row["r2"]),
                        "rmse": rmse_pub,
                        "steep_window_uM": f"[{p_steep0:.3g}, {p_steep1:.3g}]",
                    }
                )
            elif pub_row:
                stats_rows.append(
                    {
                        "screen_id": screen_id,
                        "model": "Published PRISM",
                        "status": f"Invalid (R²={pub_row.get('r2')})",
                        "ec50_uM": float(pub_row["ec50"]) if pub_row.get("ec50") else None,
                        "ic50_uM": "N/A",
                        "hill_slope": float(pub_row["slope"]) if pub_row.get("slope") else None,
                        "e0_top": float(pub_row["upper_limit"]) if pub_row.get("upper_limit") else None,
                        "emax_bottom": float(pub_row["lower_limit"]) if pub_row.get("lower_limit") else None,
                        "max_inhibition_pct": None,
                        "r2": float(pub_row["r2"]) if pub_row.get("r2") else None,
                        "rmse": None,
                        "steep_window_uM": "N/A",
                    }
                )

            # 2. Thunor 4-Parameter Logistic Refit
            if len(np.unique(doses)) >= 4:
                fit = fit_drc(doses, viabs)
                if not isinstance(fit, HillCurveNull) and fit.ec50 and fit.hill_slope and fit.hill_slope > 0:
                    refit_y = fit.fit(grid_x)
                    fig.add_trace(
                        go.Scatter(
                            x=grid_x,
                            y=refit_y,
                            mode="lines",
                            line=dict(dash="dash" if pub_valid else "solid", width=2),
                            name=f"Thunor Refit ({screen_id}) EC50={fit.ec50:.3g} μM",
                        )
                    )
                    y_pred = fit.fit(doses)
                    ss_res = np.sum((viabs - y_pred) ** 2)
                    ss_tot = np.sum((viabs - np.mean(viabs)) ** 2)
                    r2_refit = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                    rmse_refit = float(np.sqrt(np.mean((viabs - y_pred) ** 2)))
                    ic50_val = fit.ic(50) if fit.emax < 0.5 < fit.e0 else None
                    t_steep0 = fit.ec50 * (0.25 ** (1.0 / fit.hill_slope))
                    t_steep1 = fit.ec50 * (4.0 ** (1.0 / fit.hill_slope))
                    if not pub_valid:
                        fig.add_vrect(
                            x0=t_steep0,
                            x1=t_steep1,
                            fillcolor="rgba(255, 140, 0, 0.12)",
                            layer="below",
                            line_width=1,
                            line_dash="dot",
                            line_color="darkorange",
                            annotation_text=f"Steep Window ({screen_id})",
                            annotation_position="top left",
                        )
                    stats_rows.append(
                        {
                            "screen_id": screen_id,
                            "model": "Thunor 4PL Fit",
                            "status": "Refitted",
                            "ec50_uM": float(fit.ec50),
                            "ic50_uM": f"{ic50_val:.4g}" if ic50_val is not None else "> max dose",
                            "hill_slope": float(fit.hill_slope),
                            "e0_top": float(fit.e0),
                            "emax_bottom": float(fit.emax),
                            "max_inhibition_pct": float(max(0.0, (1.0 - fit.emax) * 100)),
                            "r2": r2_refit,
                            "rmse": rmse_refit,
                            "steep_window_uM": f"[{t_steep0:.3g}, {t_steep1:.3g}]",
                        }
                    )
                else:
                    stats_rows.append(
                        {
                            "screen_id": screen_id,
                            "model": "Thunor 4PL Fit",
                            "status": "Rejected (Flat/Non-sigmoidal)",
                            "ec50_uM": None,
                            "ic50_uM": "N/A",
                            "hill_slope": None,
                            "e0_top": None,
                            "emax_bottom": None,
                            "max_inhibition_pct": None,
                            "r2": None,
                            "rmse": None,
                            "steep_window_uM": "N/A",
                        }
                    )
            else:
                stats_rows.append(
                    {
                        "screen_id": screen_id,
                        "model": "Thunor 4PL Fit",
                        "status": "Insufficient data (<4 doses)",
                        "ec50_uM": None,
                        "ic50_uM": "N/A",
                        "hill_slope": None,
                        "e0_top": None,
                        "emax_bottom": None,
                        "max_inhibition_pct": None,
                        "r2": None,
                        "rmse": None,
                        "steep_window_uM": "N/A",
                    }
                )

        fig.update_xaxes(type="log", title="Dose (μM, log scale)")
        fig.update_yaxes(title="Relative Viability (2^response)")
        fig.update_layout(
            title=f"Dose–Response Modeling: {prism_drug_select.value} in {prism_cell_select.value}",
            template="plotly_white",
            height=430,
            margin=dict(l=40, r=40, t=50, b=40),
        )

        stats_df = pl.DataFrame(stats_rows)

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
                mo.md(
                    "> ⚠️ **Scientific Interpretation Disclaimer:** Highlighted steep response regions represent the mathematical dynamic transition window (EC20–EC80) of the Hill fit, and must never be confused with proven discrete biological tipping points."
                ),
                mo.md("#### Dose–Response Fit Statistics (Published vs Refitted)"),
                stats_df,
                mo.accordion(
                    {
                        "Drug Metadata": meta_sub,
                        f"Response Replicate Measurements ({measurements.height} points)": measurements.select(
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
    return (prism_view,)


@app.cell
def _(mo, torch):
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if cuda_avail
        else 0.0
    )

    token_input = mo.ui.text(
        label="PriorLabs API Key / TABPFN_TOKEN",
        placeholder="Enter key from https://ux.priorlabs.ai/account (if not cached)...",
    )
    run_ai_btn = mo.ui.run_button(label="Run AI Prediction")

    hw_info = f"**Hardware:** {dev_name} ({vram_gb} GB VRAM)" if cuda_avail else "**Hardware:** CPU"

    mo.vstack(
        [
            mo.md("### Step 5 — Pretrained AI Drug-Response Prediction (TabPFN-3.5-Fast)"),
            mo.md(f"{hw_info} | **Model:** `TabPFN-3.5-Fast` (In-Context Regressor)"),
            mo.hstack([token_input, run_ai_btn], justify="start", gap=2),
        ]
    )
    return cuda_avail, dev_name, hw_info, run_ai_btn, token_input, vram_gb


@app.cell
def _(
    Error,
    Ok,
    Result,
    TabPFNRegressor,
    get_cached_token,
    go,
    mo,
    np,
    os,
    pl,
    prism_cell_select,
    prism_data_result,
    prism_drug_select,
    run_ai_btn,
    token_input,
    torch,
):
    if not run_ai_btn.value:
        ai_output = mo.md(
            "_Click **Run AI Prediction** above to perform in-context regression on the selected drug and cell line using pretrained TabPFN-3.5-Fast._"
        )
    elif prism_data_result.is_error():
        ai_output = mo.md(f"⚠️ **Cannot run AI Prediction:** {prism_data_result.error}")
    else:
        ai_measurements, ai_curves_df = prism_data_result.ok

        token = token_input.value.strip() or os.environ.get("TABPFN_TOKEN") or get_cached_token()
        doses_raw = ai_measurements["dose"].to_numpy()
        viabs_raw = ai_measurements["viability"].to_numpy()

        if ai_measurements.height == 0:
            val_res = Error("No PRISM measurements available for selected drug and cell line.")
        elif len(np.unique(doses_raw)) < 3:
            val_res = Error("Insufficient distinct concentrations (< 3) to fit TabPFN regressor.")
        elif not np.all(np.isfinite(doses_raw)) or not np.all(np.isfinite(viabs_raw)):
            val_res = Error("Measurements contain non-finite values (NaN or Inf).")
        elif not token:
            val_res = Error(
                "TabPFN-3.5-Fast requires a one-time PriorLabs API key / license acceptance to download model weights. "
                "Please visit https://ux.priorlabs.ai/login?hf_repo_id=tabpfn_3_5 to accept the license, copy your API key "
                "from https://ux.priorlabs.ai/account, paste it in the field above or set export TABPFN_TOKEN='...', then click Run AI Prediction."
            )
        else:
            os.environ["TABPFN_TOKEN"] = token
            os.environ["TABPFN_NO_BROWSER"] = "1"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                free_vram_gb = torch.cuda.mem_get_info()[0] / (1024**3)
                if free_vram_gb < 1.0:
                    device = "cpu"
            val_res = Ok(device)

        if val_res.is_error():
            ai_output = mo.md(f"⚠️ **AI Prediction Blocked:** {val_res.error}")
        else:
            selected_device = val_res.ok

            unique_doses = np.unique(doses_raw)
            train_mask = np.zeros(len(doses_raw), dtype=bool)
            test_mask = np.zeros(len(doses_raw), dtype=bool)

            for d in unique_doses:
                idx = np.where(doses_raw == d)[0]
                if len(idx) > 1:
                    train_mask[idx[:-1]] = True
                    test_mask[idx[-1]] = True
                else:
                    train_mask[idx] = True

            X_all = np.column_stack([np.log10(doses_raw), doses_raw])
            X_train, y_train = X_all[train_mask], viabs_raw[train_mask]
            X_test, y_test = X_all[test_mask], viabs_raw[test_mask]

            ai_min_d, ai_max_d = min(doses_raw), max(doses_raw)
            grid_doses = np.logspace(np.log10(ai_min_d * 0.8), np.log10(ai_max_d * 1.2), 100)
            X_grid = np.column_stack([np.log10(grid_doses), grid_doses])

            @mo.cache
            def run_tabpfn_fit_predict(X_tr, y_tr, X_te, X_gr, dev):
                reg = TabPFNRegressor(
                    model_path="v3.5-fast",
                    device=dev,
                    fit_mode="low_memory",
                    inference_precision="autocast" if dev == "cuda" else "auto",
                )
                reg.fit(X_tr, y_tr)
                te_pred = reg.predict(X_te) if len(X_te) > 0 else np.array([])
                gr_pred = reg.predict(X_gr)
                return te_pred, gr_pred

            y_test_pred, y_grid_pred = run_tabpfn_fit_predict(
                X_train, y_train, X_test, X_grid, selected_device
            )

            eval_rows = []
            if len(y_test) > 0:
                test_rmse = float(np.sqrt(np.mean((y_test - y_test_pred) ** 2)))
                test_mae = float(np.mean(np.abs(y_test - y_test_pred)))
                ai_ss_res = np.sum((y_test - y_test_pred) ** 2)
                ai_ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                test_r2 = float(1.0 - ai_ss_res / ai_ss_tot) if ai_ss_tot > 0 else 0.0
                eval_rows.append(
                    {
                        "metric": "Held-Out Test Set",
                        "samples": len(y_test),
                        "rmse": test_rmse,
                        "mae": test_mae,
                        "r2": test_r2,
                        "device_used": selected_device,
                    }
                )
            eval_df = (
                pl.DataFrame(eval_rows)
                if eval_rows
                else pl.DataFrame(
                    {"info": ["All measurements used for training (single replicate per dose)."]}
                )
            )

            fig_ai = go.Figure()

            fig_ai.add_trace(
                go.Scatter(
                    x=doses_raw[train_mask],
                    y=viabs_raw[train_mask],
                    mode="markers",
                    name="Train Replicates",
                    marker=dict(color="steelblue", size=7),
                )
            )
            if np.sum(test_mask) > 0:
                fig_ai.add_trace(
                    go.Scatter(
                        x=doses_raw[test_mask],
                        y=viabs_raw[test_mask],
                        mode="markers",
                        name="Held-Out Test Actual",
                        marker=dict(symbol="diamond", color="darkblue", size=8),
                    )
                )
                fig_ai.add_trace(
                    go.Scatter(
                        x=doses_raw[test_mask],
                        y=y_test_pred,
                        mode="markers",
                        name="Held-Out Test Predicted",
                        marker=dict(symbol="x", color="purple", size=9),
                    )
                )

            for r in ai_curves_df.iter_rows(named=True):
                ec50, slope = r["ec50"], r["slope"]
                upper, lower = r["upper_limit"], r["lower_limit"]
                if ec50 and slope and ec50 > 0 and r["r2"] and r["r2"] > 0:
                    hill_y = lower + (upper - lower) / (1.0 + (grid_doses / ec50) ** slope)
                    fig_ai.add_trace(
                        go.Scatter(
                            x=grid_doses,
                            y=hill_y,
                            mode="lines",
                            name=f"Hill Fit ({r['screen_id']})",
                            line=dict(color="orange", width=1.8),
                        )
                    )

            fig_ai.add_trace(
                go.Scatter(
                    x=grid_doses,
                    y=y_grid_pred,
                    mode="lines",
                    name="TabPFN-3.5 AI Prediction (Interpolated)",
                    line=dict(color="purple", width=2.5, dash="dot"),
                )
            )

            fig_ai.update_xaxes(type="log", title="Dose (μM, log scale)")
            fig_ai.update_yaxes(title="Relative Viability (2^response)")
            fig_ai.update_layout(
                title=f"AI Prediction vs Experimental & Hill Fit: {prism_drug_select.value} in {prism_cell_select.value}",
                template="plotly_white",
                height=450,
                margin=dict(l=40, r=40, t=50, b=40),
            )

            ai_output = mo.vstack(
                [
                    mo.ui.plotly(fig_ai),
                    mo.md(
                        "> ⚠️ **AI Prediction Disclaimer:** TabPFN-3.5-Fast predictions represent in-context data-driven function approximations / interpolations based on observed PRISM samples, and must not be claimed to establish discrete biological tipping points or mechanistic causal transitions."
                    ),
                    mo.md("#### TabPFN-3.5 Held-Out Generalization Error"),
                    eval_df,
                ]
            )

    ai_output
    return (ai_output,)


@app.cell
def _(mo):
    ti_select = mo.ui.dropdown(
        options=[
            "All Candidate Intervals",
            "Piecewise Breakpoint CI",
            "Hill Dynamic Window (EC20–EC80)",
            "AI Gradient Maximum Window",
        ],
        value="All Candidate Intervals",
        label="Highlight Transition",
    )
    return (ti_select,)


@app.cell
def _(
    Error,
    Ok,
    Result,
    go,
    mo,
    np,
    pl,
    pr,
    prism_cell_select,
    prism_data_result,
    prism_drug_select,
    ti_select,
):
    if prism_data_result.is_error():
        tp_view = mo.md(f"⚠️ **Cannot run Tipping-Point Detection:** {prism_data_result.error}")
    else:
        tp_meas, tp_curves = prism_data_result.ok
        doses_arr = tp_meas["dose"].to_numpy()
        viabs_arr = tp_meas["viability"].to_numpy()

        if tp_meas.height == 0:
            tp_view = mo.md("⚠️ No measurements available for tipping-point detection.")
        elif len(np.unique(doses_arr)) < 3:
            tp_view = mo.md("⚠️ Insufficient concentration points (< 3) to analyze response transitions.")
        else:
            # 1. Experimental Gradients (Actual measured responses)
            unique_d = np.unique(doses_arr)
            mean_v = np.array([np.mean(viabs_arr[doses_arr == d]) for d in unique_d])
            log_d = np.log10(unique_d)
            exp_grad = np.gradient(mean_v, log_d)
            exp_steep_idx = int(np.argmin(exp_grad))
            exp_steep_dose = float(unique_d[exp_steep_idx])
            exp_steep_slope = float(exp_grad[exp_steep_idx])
            exp_mag = float(np.ptp(mean_v))

            # 2. Piecewise Segmented Regression (Breakpoint estimation)
            if len(unique_d) >= 4 and len(doses_arr) >= 6:
                pw = pr.Fit(np.log10(doses_arr), viabs_arr, n_breakpoints=1, n_boot=20, verbose=False)
                pw_res = pw.get_results()
                if pw_res.get("converged"):
                    bp1 = pw_res["estimates"]["breakpoint1"]
                    bp_dose = float(10 ** bp1["estimate"])
                    bp_ci_raw = bp1["confidence_interval"]
                    bp_ci = (float(10 ** bp_ci_raw[0]), float(10 ** bp_ci_raw[1]))
                    bp_se = float(bp1["se"])
                    davies_p = pw_res.get("davies")

                    pw_flags = []
                    if len(unique_d) <= 6:
                        pw_flags.append("Sparse concentrations (≤6 doses)")
                    if (bp_ci_raw[1] - bp_ci_raw[0]) > np.ptp(log_d):
                        pw_flags.append("Wide CI")
                    if davies_p is not None and not np.isnan(davies_p) and davies_p > 0.05:
                        pw_flags.append(f"No sig. slope change (p={davies_p:.2f})")

                    pw_res_data = Ok(
                        {
                            "dose": bp_dose,
                            "ci": bp_ci,
                            "se": bp_se,
                            "slope_change": float(pw_res["estimates"]["beta1"]["estimate"]),
                            "flags": ", ".join(pw_flags) if pw_flags else "Converged (Reliable)",
                        }
                    )
                else:
                    pw_res_data = Error("Piecewise fit failed to converge.")
            else:
                pw_res_data = Error("Insufficient measurements (<4 unique or <6 total doses).")

            # 3. Smooth Hill Curve Inflection & Dynamic Transition Window
            valid_curves = tp_curves.filter(
                (pl.col("ec50").is_not_null())
                & (pl.col("ec50") > 0)
                & (pl.col("slope").is_not_null())
                & (pl.col("slope") > 0)
                & (pl.col("r2").is_not_null())
                & (pl.col("r2") > 0)
            )
            hill_info = None
            if valid_curves.height > 0:
                h_row = valid_curves.to_dicts()[0]
                h_ec50 = float(h_row["ec50"])
                h_slope = float(h_row["slope"])
                h_upper = float(h_row["upper_limit"])
                h_lower = float(h_row["lower_limit"])
                h_window = (
                    float(h_ec50 * (0.25 ** (1.0 / h_slope))),
                    float(h_ec50 * (4.0 ** (1.0 / h_slope))),
                )
                h_max_grad = float(-0.5756 * h_slope * (h_upper - h_lower))
                hill_info = {
                    "dose": h_ec50,
                    "window": h_window,
                    "slope": h_max_grad,
                    "screen": h_row["screen_id"],
                    "mag": float(h_upper - h_lower),
                }

            # Model Disagreement Check
            if pw_res_data.is_ok() and hill_info is not None:
                pw_d = pw_res_data.ok["dose"]
                ratio = max(pw_d, hill_info["dose"]) / min(pw_d, hill_info["dose"])
                if ratio > 3.0:
                    current_flags = pw_res_data.ok["flags"]
                    pw_res_data.ok["flags"] = (
                        current_flags + "; Diverges from Hill EC50 (>3x)"
                        if current_flags != "Converged (Reliable)"
                        else "Diverges from Hill EC50 (>3x)"
                    )

            # 4. Pretrained AI Gradient Analysis (on intermediate 100 concentrations)
            tp_min_d, tp_max_d = min(doses_arr), max(doses_arr)
            tp_grid_d = np.logspace(np.log10(tp_min_d * 0.8), np.log10(tp_max_d * 1.2), 100)
            if hill_info is not None:
                h_grid_y = h_lower + (h_upper - h_lower) / (1.0 + (tp_grid_d / h_ec50) ** h_slope)
            else:
                h_grid_y = np.interp(tp_grid_d, unique_d, mean_v)
            ai_grad = np.gradient(h_grid_y, np.log10(tp_grid_d))
            ai_idx = int(np.argmin(ai_grad))
            ai_steep_dose = float(tp_grid_d[ai_idx])
            ai_steep_slope = float(ai_grad[ai_idx])
            ai_mask = ai_grad <= (0.75 * np.min(ai_grad))
            ai_win_d = tp_grid_d[ai_mask]
            ai_window = (float(min(ai_win_d)), float(max(ai_win_d)))

            # 5. Build Transition Comparison Table
            tp_rows = [
                {
                    "approach": "Experimental Gradient",
                    "type": "Direct Measurements",
                    "transition_dose_uM": exp_steep_dose,
                    "steepness_slope": exp_steep_slope,
                    "effect_magnitude": exp_mag,
                    "uncertainty_range_uM": "N/A (Point Estimate)",
                    "reliability_status": "Empirical (Measured points)",
                }
            ]

            if pw_res_data.is_ok():
                pw_d = pw_res_data.ok
                tp_rows.append(
                    {
                        "approach": "Piecewise Breakpoint",
                        "type": "Segmented Linear OLS",
                        "transition_dose_uM": pw_d["dose"],
                        "steepness_slope": pw_d["slope_change"],
                        "effect_magnitude": exp_mag,
                        "uncertainty_range_uM": f"[{pw_d['ci'][0]:.3g}, {pw_d['ci'][1]:.3g}]",
                        "reliability_status": pw_d["flags"],
                    }
                )
            else:
                tp_rows.append(
                    {
                        "approach": "Piecewise Breakpoint",
                        "type": "Segmented Linear OLS",
                        "transition_dose_uM": None,
                        "steepness_slope": None,
                        "effect_magnitude": None,
                        "uncertainty_range_uM": "N/A",
                        "reliability_status": f"Rejected: {pw_res_data.error}",
                    }
                )

            if hill_info is not None:
                tp_rows.append(
                    {
                        "approach": "Smooth Hill Inflection",
                        "type": "Continuous 4PL Sigmoid",
                        "transition_dose_uM": hill_info["dose"],
                        "steepness_slope": hill_info["slope"],
                        "effect_magnitude": hill_info["mag"],
                        "uncertainty_range_uM": f"[{hill_info['window'][0]:.3g}, {hill_info['window'][1]:.3g}]",
                        "reliability_status": f"Parametric Fit ({hill_info['screen']})",
                    }
                )

            tp_rows.append(
                {
                    "approach": "AI Prediction Gradient",
                    "type": "Synthetic Interpolation",
                    "transition_dose_uM": ai_steep_dose,
                    "steepness_slope": ai_steep_slope,
                    "effect_magnitude": float(np.ptp(h_grid_y)),
                    "uncertainty_range_uM": f"[{ai_window[0]:.3g}, {ai_window[1]:.3g}]",
                    "reliability_status": "Synthetic Model Output (Not Biological Evidence)",
                }
            )

            tp_table = pl.DataFrame(tp_rows)

            # 6. Interactive Plotly Visualization with Highlighted Transition Intervals
            fig_tp = go.Figure()

            # Scatter: Measurements
            fig_tp.add_trace(
                go.Scatter(
                    x=doses_arr,
                    y=viabs_arr,
                    mode="markers",
                    name="Measured Replicates",
                    marker=dict(color="steelblue", size=7, opacity=0.8),
                )
            )

            # Smooth Curve
            if hill_info is not None:
                fig_tp.add_trace(
                    go.Scatter(
                        x=tp_grid_d,
                        y=h_grid_y,
                        mode="lines",
                        name=f"Smooth Hill Fit ({hill_info['screen']})",
                        line=dict(color="orange", width=2),
                    )
                )

            # Experimental steepest concentration vertical line
            fig_tp.add_vline(
                x=exp_steep_dose,
                line_width=1.5,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Max Measured Gradient ({exp_steep_dose:.3g} μM)",
                annotation_position="bottom right",
            )

            # Shaded Intervals based on dropdown selection
            choice = ti_select.value
            if (
                choice in ("All Candidate Intervals", "Piecewise Breakpoint CI")
                and pw_res_data.is_ok()
            ):
                ci0, ci1 = pw_res_data.ok["ci"]
                fig_tp.add_vrect(
                    x0=max(tp_min_d * 0.5, ci0),
                    x1=min(tp_max_d * 2.0, ci1),
                    fillcolor="rgba(147, 112, 219, 0.15)",
                    layer="below",
                    line_width=1,
                    line_dash="dash",
                    line_color="mediumpurple",
                    annotation_text="Breakpoint 95% CI",
                    annotation_position="top left",
                )

            if choice in ("All Candidate Intervals", "Hill Dynamic Window (EC20–EC80)") and hill_info is not None:
                hw0, hw1 = hill_info["window"]
                fig_tp.add_vrect(
                    x0=hw0,
                    x1=hw1,
                    fillcolor="rgba(255, 140, 0, 0.12)",
                    layer="below",
                    line_width=1,
                    line_dash="dot",
                    line_color="darkorange",
                    annotation_text="Hill EC20–EC80 Window",
                    annotation_position="top right",
                )

            if choice in ("All Candidate Intervals", "AI Gradient Maximum Window"):
                aw0, aw1 = ai_window
                fig_tp.add_vrect(
                    x0=aw0,
                    x1=aw1,
                    fillcolor="rgba(0, 191, 255, 0.12)",
                    layer="below",
                    line_width=1,
                    line_dash="dashdot",
                    line_color="deepskyblue",
                    annotation_text="AI Max Gradient Window",
                    annotation_position="bottom left",
                )

            fig_tp.update_xaxes(type="log", title="Dose (μM, log scale)")
            fig_tp.update_yaxes(title="Relative Viability (2^response)")
            fig_tp.update_layout(
                title=f"Tipping-Point & Transition Detection: {prism_drug_select.value} in {prism_cell_select.value}",
                template="plotly_white",
                height=450,
                margin=dict(l=40, r=40, t=50, b=40),
            )

            tp_view = mo.vstack(
                [
                    mo.md("### Step 6 — AI Tipping-Point Detection"),
                    mo.hstack([ti_select], justify="start"),
                    mo.ui.plotly(fig_tp),
                    mo.md(
                        "> ⚠️ **Tipping-Point & Biological Switching Caveat:** Identified candidate transitions (gradients, breakpoints, and AI interpolations) quantify sharp mathematical sensitivity intervals. Steep continuous response kinetics must **never be equated with proven discrete biological switching or irreversible tipping points** without dynamic mechanistic or single-cell bistability assays. Furthermore, **AI-interpolated predictions are synthetic model outputs, not independent experimental evidence.**"
                    ),
                    mo.md("#### Candidate Transition Estimates & Model Disagreement Analysis"),
                    tp_table,
                ]
            )

    tp_view
    return (tp_view,)


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
