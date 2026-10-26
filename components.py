import os
import zipfile
from expression import Error, Ok, Result
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from scipy import stats
from tabpfn import TabPFNRegressor
from tabpfn.browser_auth import get_cached_token
from thunor.curve_fit import HillCurveNull, fit_drc
import tifffile
import torch


def fit_hill(doses: np.ndarray, viabs: np.ndarray):
    f = fit_drc(doses, viabs)
    if isinstance(f, HillCurveNull) or f.ec50 is None or f.hill_slope is None:
        return Error("Curve fit did not converge")

    pred = f.fit_fn(doses, *f.popt)
    ss_res = np.sum((viabs - pred) ** 2)
    ss_tot = np.sum((viabs - np.mean(viabs)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

    ic50 = float(f.ec50)
    slope = float(f.hill_slope)
    if slope != 0:
        ec20 = ic50 * ((20.0 / 80.0) ** (1.0 / slope))
        ec80 = ic50 * ((80.0 / 20.0) ** (1.0 / slope))
        lo_ec, hi_ec = float(min(ec20, ec80)), float(max(ec20, ec80))
    else:
        lo_ec, hi_ec = float(np.min(doses)), float(np.max(doses))

    se_res = float(np.std(viabs - pred))
    ci_band = float(1.96 * se_res / np.sqrt(max(1, len(doses))))

    return Ok({
        "fit": f,
        "ec50": ic50,
        "slope": slope,
        "r2": r2,
        "lo_ec": lo_ec,
        "hi_ec": hi_ec,
        "ci_band": ci_band,
    })


def predict_tabpfn(doses: np.ndarray, viabs: np.ndarray, grid_x: np.ndarray):
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / (1024**3) >= 1.0 else "cpu"
    token = os.environ.get("TABPFN_TOKEN") or get_cached_token()
    m_path = "v3.5-fast" if token else "tabpfn-v2-regressor.ckpt"

    reg = TabPFNRegressor(
        model_path=m_path,
        device=dev,
        fit_mode="low_memory",
        inference_precision="autocast" if dev == "cuda" else "auto",
    )
    x_tr = np.column_stack([np.log10(doses), doses])
    x_gr = np.column_stack([np.log10(grid_x), grid_x])
    reg.fit(x_tr, viabs)
    return reg.predict(x_gr)


def build_drc_plot(
    doses: np.ndarray,
    viabs: np.ndarray,
    grid_x: np.ndarray,
    pred_hill: np.ndarray,
    pred_tabpfn: np.ndarray | None,
    ec50: float,
    lo_ec: float,
    hi_ec: float,
    ci_band: float,
    active_dose: float | None = None,
    drug: str = "",
    cell_line: str = "",
):
    fig = go.Figure()

    ci_upper = np.clip(pred_hill + ci_band, 0, None)
    ci_lower = np.clip(pred_hill - ci_band, 0, None)
    fig.add_trace(go.Scatter(x=grid_x, y=ci_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="none"))
    fig.add_trace(go.Scatter(x=grid_x, y=ci_lower, mode="lines", fill="tonexty", fillcolor="rgba(31, 119, 180, 0.12)", line=dict(width=0), name="95% Confidence Interval", hoverinfo="none"))

    if pred_tabpfn is not None:
        fig.add_trace(go.Scatter(x=grid_x, y=pred_tabpfn, mode="lines", line=dict(color="#2ca02c", dash="dash", width=2), name="TabPFN Prior"))

    fig.add_trace(go.Scatter(x=grid_x, y=pred_hill, mode="lines", line=dict(color="#1f77b4", width=2.5), name="Hill Fit (LL4)"))
    fig.add_trace(go.Scatter(
        x=doses,
        y=viabs,
        mode="markers",
        marker=dict(size=8, color="#1f77b4", line=dict(width=1, color="white"), opacity=0.85),
        name="PRISM Measured Viability",
        text=[f"Concentration: {d:.4g} μM<br>Relative Viability: {v:.3f}" for d, v in zip(doses, viabs)],
        hoverinfo="text",
    ))

    fig.add_vrect(x0=lo_ec, x1=hi_ec, fillcolor="#ff7f0e", opacity=0.12, line_width=0, annotation_text="EC20–EC80 Range", annotation_position="top left")
    fig.add_vline(x=ec50, line_dash="dash", line_color="#d62728", annotation_text=f"EC50: {ec50:.4g} μM", annotation_position="bottom left")

    if active_dose is not None:
        fig.add_vline(x=active_dose, line_dash="dot", line_color="#4b5563", annotation_text=f"Selected: {active_dose:.4g} μM", annotation_position="top right")

    fig.update_xaxes(type="log", title_text="Concentration (μM, log scale)", gridcolor="#f3f4f6")
    fig.update_yaxes(title_text="Relative Viability", range=[-0.05, 1.35], gridcolor="#f3f4f6")
    fig.update_layout(
        title=f"Dose–Response: {drug} in {cell_line}",
        template="plotly_white",
        height=420,
        margin=dict(l=50, r=40, t=50, b=40),
        legend=dict(orientation="h", y=-0.22, x=0),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12),
    )
    return fig


def load_microscopy_rgb(zip_path: str, plate: str, dapi: str, tubulin: str, actin: str):
    with zipfile.ZipFile(zip_path) as z:
        d = tifffile.imread(z.open(f"{plate}/{dapi}")).astype("float32")
        t = tifffile.imread(z.open(f"{plate}/{tubulin}")).astype("float32")
        a = tifffile.imread(z.open(f"{plate}/{actin}")).astype("float32")

    def _norm(ch):
        p1, p99 = np.percentile(ch, (1.0, 99.5))
        clipped = np.clip((ch - p1) / (p99 - p1 + 1e-7), 0.0, 1.0)
        return (clipped * 255).astype("uint8")

    return np.stack([_norm(a), _norm(t), _norm(d)], axis=-1)


def build_umap_plot(s8_df, target_compound: str, active_dose: float | None = None):
    fig = go.Figure()

    bg = s8_df.filter(s8_df["Image_Metadata_Compound"] != target_compound)
    fg = s8_df.filter(s8_df["Image_Metadata_Compound"] == target_compound).sort("Image_Metadata_Concentration")

    fig.add_trace(go.Scatter(
        x=bg["umap_1"].to_list(),
        y=bg["umap_2"].to_list(),
        mode="markers",
        marker=dict(size=4.5, color="#e5e7eb", opacity=0.35),
        name="Other Conditions",
        hoverinfo="none",
    ))

    if fg.height > 0:
        d_grp = fg.group_by("Image_Metadata_Concentration", maintain_order=True).agg(
            [pl.col("umap_1").mean().alias("umap_1"), pl.col("umap_2").mean().alias("umap_2")]
        )
        fig.add_trace(go.Scatter(
            x=d_grp["umap_1"].to_list(),
            y=d_grp["umap_2"].to_list(),
            mode="lines",
            line=dict(color="#4b5563", width=1.5, dash="dot"),
            name="Concentration Trajectory",
            hoverinfo="none",
        ))

        fig.add_trace(go.Scatter(
            x=fg["umap_1"].to_list(),
            y=fg["umap_2"].to_list(),
            mode="markers",
            marker=dict(
                size=9,
                color=fg["Image_Metadata_Concentration"].to_list(),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Concentration (μM)", len=0.7),
            ),
            name=f"{target_compound} Fields",
            text=[
                f"Compound: {target_compound}<br>Concentration: {r['Image_Metadata_Concentration']} μM<br>Well: {r['Image_Metadata_Well_DAPI']}"
                for r in fg.iter_rows(named=True)
            ],
            hoverinfo="text",
        ))

        if active_dose is not None:
            avail = np.array(fg["Image_Metadata_Concentration"].unique().to_list())
            if len(avail) > 0:
                nearest = avail[np.argmin(np.abs(avail - active_dose))]
                closest = fg.filter(fg["Image_Metadata_Concentration"] == nearest).row(0, named=True)
                fig.add_trace(go.Scatter(
                    x=[closest["umap_1"]],
                    y=[closest["umap_2"]],
                    mode="markers",
                    marker=dict(size=14, color="#e11d48", symbol="star", line=dict(width=1.5, color="black")),
                    name=f"Selected Condition ({nearest} μM)",
                    text=[f"Active: {target_compound} @ {nearest} μM"],
                    hoverinfo="text",
                ))

    fig.update_xaxes(title_text="UMAP Dimension 1", gridcolor="#f3f4f6")
    fig.update_yaxes(title_text="UMAP Dimension 2", gridcolor="#f3f4f6")
    fig.update_layout(
        title=f"Cell-State Space (Cell-DINO ViT-S/8) — {target_compound}",
        template="plotly_white",
        height=420,
        margin=dict(l=50, r=40, t=50, b=40),
        legend=dict(orientation="h", y=-0.22, x=0),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12),
    )
    return fig


def build_validation_plot():
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Potency Concordance: PRISM vs External Consortia",
            "Hill Slope Concordance: PRISM vs CTRPv2",
        ),
        horizontal_spacing=0.14,
    )

    prism_p = [-1.54, -1.58, -1.42, -1.46, -1.32, -2.05, -2.08, -2.03]
    ext_p = [-2.36, -2.33, -1.94, -2.01, -1.32, -1.77, -0.78, -1.82]
    colors = ["#2563eb", "#2563eb", "#d97706", "#2563eb", "#2563eb", "#dc2626", "#dc2626", "#dc2626"]
    labels = [
        "Bortezomib (RKO, GDSC2)",
        "Bortezomib (A375, GDSC2)",
        "Docetaxel (MDA-MB-231, GDSC2)",
        "Docetaxel (KYSE-270, GDSC2)",
        "Paclitaxel (A375, GDSC2)",
        "Paclitaxel (MKN1, GDSC2)",
        "Paclitaxel (NUGC-3, CTRPv2)",
        "Docetaxel (KU-19-19, GDSC2)",
    ]

    fig.add_trace(
        go.Scatter(
            x=prism_p,
            y=ext_p,
            mode="markers+text",
            text=labels,
            textposition="top left",
            marker=dict(size=10, color=colors, line=dict(width=1, color="black")),
            name="Potency Concordance",
            showlegend=False,
        ),
        row=1, col=1,
    )
    lims = [-2.5, 0.0]
    fig.add_trace(
        go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="gray", dash="dash", width=1.5), name="Identity (y=x)"),
        row=1, col=1,
    )

    prism_slopes = [14.00, 6.67, 4.06, 18.15, 3.45, 3.25, 17.48, 19.31]
    ctrp_slopes = [3.65, 4.22, 1.25, 1.89, 5.00, 2.24, 5.00, 5.00]
    slope_labels = [
        "Bortezomib (RKO)",
        "Bortezomib (A375)",
        "Docetaxel (MDA-MB-231)",
        "Docetaxel (KYSE-270)",
        "Paclitaxel (A375)",
        "Paclitaxel (MKN1)",
        "Paclitaxel (NUGC-3)",
        "Docetaxel (KU-19-19)",
    ]

    fig.add_trace(
        go.Scatter(
            x=prism_slopes,
            y=ctrp_slopes,
            mode="markers+text",
            text=slope_labels,
            textposition="bottom right",
            marker=dict(size=10, color=colors, line=dict(width=1, color="black")),
            name="CTRPv2 Slope",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="black", annotation_text="Standard Mass Action (|nH|=1)", row=1, col=2)
    fig.add_hline(y=5.0, line_dash="dash", line_color="#b91c1c", annotation_text="CTRP Parameter Clamp (|nH|=5.0)", row=1, col=2)

    fig.update_xaxes(title_text="PRISM log10(EC50, μM)", row=1, col=1, gridcolor="#f3f4f6")
    fig.update_yaxes(title_text="External log10(EC50 or IC50, μM)", row=1, col=1, gridcolor="#f3f4f6")
    fig.update_xaxes(title_text="PRISM Hill Slope (|nH|)", row=1, col=2, gridcolor="#f3f4f6")
    fig.update_yaxes(title_text="CTRPv2 Hill Slope (|nH|)", row=1, col=2, gridcolor="#f3f4f6")
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=50, r=40, t=50, b=40),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11),
    )
    return fig


def build_generalization_plot():
    models = ["Parametric Hill (LL4)", "Pretrained TabPFN", "Segmented Breakpoint"]
    rmses_overall = [0.0952, 0.1148, 0.1853]
    rmses_trans = [0.1214, 0.1382, 0.2461]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Overall Dose RMSE",
        x=models,
        y=rmses_overall,
        marker_color="#1f77b4",
        text=[f"{r:.4f}" for r in rmses_overall],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Transition-Zone RMSE [EC20, EC80]",
        x=models,
        y=rmses_trans,
        marker_color="#ff7f0e",
        text=[f"{r:.4f}" for r in rmses_trans],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        title="Dose-Level Generalization: Out-of-Sample Prediction Error",
        yaxis_title="RMSE (Relative Viability)",
        template="plotly_white",
        height=360,
        margin=dict(l=50, r=40, t=50, b=40),
        legend=dict(orientation="h", y=-0.22, x=0),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12),
    )
    return fig
