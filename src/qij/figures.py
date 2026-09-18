"""
The paper's figures (plan section 8): F2, F3, F7, F9, F10. Every figure
here is a pure function of the products written by `qij.study` under
`<run_dir>/<dataset>/<estimator>/` (plan section 3, pinned exactly in
the interface sheet section 5, amended 18 September for per-output
refinement diagnostics) -- `truth.parquet`, `qij.parquet`, `boot.h5`,
and, where an analytic influence exists, `qij_partition.parquet` and
`qij_points.parquet`. Nothing else is read: no estimator is re-run, no
dataset is redrawn, no interval is read (none is stored -- every
interval here is recomputed from `core.intervals.qij_interval` or
`core.intervals.percentile_interval`, exactly as `qij.tables` does).

F2(b)'s coverage grid is read from `qij.tables.coverage_grid`, not
recomputed here: the plan (section 8) names that table as the source
"the coverage grid for F2(b)", so reusing it is the one way to compute
it rather than a second copy of the same pooling logic.

Product directories are discovered under `<run_dir>/*/*/truth.parquet`
(the same glob `qij.tables._product_dirs` uses), not hard-coded by
name, so a figure draws exactly the estimands a run actually produced.
The one exception is F9's curve panel, which the figure specification
names explicitly ("the Pareto shape's psi_hat_0 curve"): that panel
reads `pareto/shape` by path.

Two gaps the pinned products do not cover, noted here rather than
worked around:

* F9's curve panel was specified with an overlay of the prototype
  influences I_j. No product carries per-prototype influence values
  (`qij_points.parquet` is per-point, `qij.parquet`'s diagnostics are
  scalar per draw/output) -- that overlay is not drawn here.
* `qij_points.parquet` carries no draw index `s`, so its designated
  draw cannot be joined back to that draw's `rho` in `qij.parquet`;
  the corner annotation uses the rank correlation between psi0 and psi
  (computable from qij_points.parquet alone) instead of rho.

F10 reads the cost-vs-N study, which is one run (one config, one N)
per out_dir; no product records N itself, so `fig10` takes an explicit
`{N: run_dir}` mapping from its caller rather than discovering N from
a directory-naming convention this file would have to invent.
"""

from __future__ import annotations

import glob
import os
import string

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from qij.core.intervals import percentile_interval, qij_interval
from qij.tables import coverage_grid

__all__ = ["fig2", "fig3", "fig7", "fig9", "fig10"]

# ---------------------------------------------------------------------------
# Style (QIJ_figure_style.md, inlined -- this task's only files are
# figures.py and scripts/make_figures.py, so there is no separate
# style module).
# ---------------------------------------------------------------------------

OI = dict(black="#000000", orange="#E69F00", skyblue="#56B4E9",
          green="#009E73", yellow="#F0E442", blue="#0072B2",
          vermillion="#D55E00", purple="#CC79A7")

METHOD = {
    "qij": dict(color=OI["blue"], ls="-", marker="o", label="QIJ"),
    "boot": dict(color=OI["vermillion"], ls="--", marker="s", label="Bootstrap"),
    "truth": dict(color=OI["black"], ls=":", marker=None, label="Truth"),
}
BAND = dict(materiality="#DDDDDD")

# 4-hue qualitative palette for F10(c), the one panel that must encode
# TWO categorical axes (output and method) at once, so colour cannot
# stay reserved for method alone there; method keeps its own
# linestyle/marker (solid o = QIJ, dashed s = bootstrap) as the second
# cue in that panel.
OUTPUT_COLORS = [OI["orange"], OI["green"], OI["purple"], OI["skyblue"],
                  OI["yellow"], OI["black"]]

RC = {
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.labelweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5, "legend.title_fontsize": 10,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "lines.linewidth": 1.8, "lines.markersize": 5,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
}
RC_LNCS = {
    **RC,
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "legend.title_fontsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": None, "savefig.transparent": False,
}


def _use_style(lncs: bool) -> None:
    plt.rcParams.update(RC_LNCS if lncs else RC)


def _panel_label(ax, text: str) -> None:
    ax.text(-0.12, 1.04, text, transform=ax.transAxes,
            fontsize=plt.rcParams["axes.titlesize"], fontweight="bold",
            va="top", ha="left")


def _legend(ax_or_fig, title=None, **kw):
    leg = ax_or_fig.legend(title=title, **kw)
    if title:
        leg.get_title().set_fontweight("bold")
    return leg


def _panel_labels(n: int) -> list:
    return [f"({c})" for c in string.ascii_lowercase[:n]]


# ---------------------------------------------------------------------------
# Reading the products -- mirrors `qij.tables`'s own loading helpers
# (same glob, same "theta_hat_" discovery of output names, same s-based
# join), amended for the per-output refinement diagnostics.
# ---------------------------------------------------------------------------

def _product_dirs(run_dir: str) -> list:
    """(dataset, estimator, path) for every `<run_dir>/<dataset>/
    <estimator>/` directory with a `truth.parquet` (plan section 3),
    discovered from the directories themselves."""
    dirs = []
    for truth_path in sorted(glob.glob(os.path.join(run_dir, "*", "*", "truth.parquet"))):
        estimator_dir = os.path.dirname(truth_path)
        dataset = os.path.basename(os.path.dirname(estimator_dir))
        estimator = os.path.basename(estimator_dir)
        dirs.append((dataset, estimator, estimator_dir))
    return dirs


def _outputs(truth_df: pd.DataFrame) -> list:
    """The estimator's output names, off truth.parquet's own
    `theta_hat_<output>` columns."""
    prefix = "theta_hat_"
    return [c[len(prefix):] for c in truth_df.columns if c.startswith(prefix)]


def _load_draws(estimator_dir: str, outputs: list) -> dict:
    """truth.parquet, qij.parquet and boot.h5 for one (dataset,
    estimator) directory, joined on `s` and aligned to `outputs`'
    order. `L` and the other refinement diagnostics are per-output
    (amendment 3, 18 September) and are the only ones this dict needs
    beyond the shared `normalized_rows`/`wall_time_total`."""
    truth = pd.read_parquet(os.path.join(estimator_dir, "truth.parquet"))
    qij_df = pd.read_parquet(os.path.join(estimator_dir, "qij.parquet"))
    df = truth.merge(qij_df, on="s", how="inner")

    with h5py.File(os.path.join(estimator_dir, "boot.h5"), "r") as h5f:
        theta_boot = h5f["theta"][...]
        s_boot = h5f["s"][...]
        wall_time_boot = h5f["wall_time"][...]
        boot_outputs = [o.decode() if isinstance(o, bytes) else str(o)
                         for o in h5f.attrs["outputs"]]

    row_of_s = pd.Series(np.arange(len(s_boot)), index=s_boot)
    order = row_of_s.loc[df["s"].to_numpy()].to_numpy()
    col_order = [boot_outputs.index(o) for o in outputs]
    theta_boot = theta_boot[order][:, :, col_order]
    wall_time_boot = wall_time_boot[order]

    has_oracle = f"V_oracle_{outputs[0]}" in df.columns
    return {
        "s": df["s"].to_numpy(),
        "theta_true": df[[f"theta_true_{o}" for o in outputs]].to_numpy(),
        "theta_hat": df[[f"theta_hat_{o}" for o in outputs]].to_numpy(),
        "v_tot_hat": df[[f"V_tot_hat_{o}" for o in outputs]].to_numpy(),
        "a_bca": df[[f"a_bca_{o}" for o in outputs]].to_numpy(),
        "v_oracle": (df[[f"V_oracle_{o}" for o in outputs]].to_numpy()
                     if has_oracle else None),
        "L": df[[f"L_{o}" for o in outputs]].to_numpy(),
        "theta_boot": theta_boot,
        "wall_time_qij": df["wall_time_total"].to_numpy(),
        "wall_time_boot": wall_time_boot,
        "normalized_rows": df["normalized_rows"].to_numpy(),
        "has_oracle": has_oracle,
    }


def _estimands(run_dir: str, require_oracle: bool = False) -> list:
    """One dict per (dataset, estimator, output) triple, in discovery
    order. `require_oracle` keeps only outputs with a `V_oracle_<o>`
    column -- F2(a)/(c), F3 and F9 all compare against the oracle
    influence, so they only exist where plan section 4's analytic
    influence exists."""
    out = []
    for dataset, estimator, path in _product_dirs(run_dir):
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        has_oracle = f"V_oracle_{outputs[0]}" in truth.columns if outputs else False
        if require_oracle and not has_oracle:
            continue
        for o in outputs:
            label = (f"{dataset}/{estimator}" if len(outputs) == 1
                      else f"{dataset}/{estimator}:{o}")
            out.append(dict(dataset=dataset, estimator=estimator, output=o,
                             dir=path, label=label))
    return out


def _mean_se(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return dict(mean=float("nan"), se=float("nan"))
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return dict(mean=mean, se=se)


# ---------------------------------------------------------------------------
# F2 -- the validation trio
# ---------------------------------------------------------------------------

def _f2_panel_a_rows(oracle_estimands: list) -> list:
    """Two mean-log-ratio series per estimand, both against the
    per-draw oracle variance V_oracle_<o>: QIJ's log(V_tot_hat/V_oracle)
    and the bootstrap's log(V_boot/V_oracle), V_boot the per-draw
    variance of boot.h5's replicates (a numpy computation on the
    product's own array, not a re-run), over the replicates that
    converged -- a NaN replicate (plan section 4: a resample that drops
    the rare support) is excluded, not treated as a value; the failure
    fractions themselves are T1's, not this panel's."""
    rows = []
    for e in oracle_estimands:
        loaded = _load_draws(e["dir"], [e["output"]])
        v_oracle = loaded["v_oracle"][:, 0]
        v_tot_hat = loaded["v_tot_hat"][:, 0]
        ok = np.isfinite(v_oracle) & (v_oracle > 0) & np.isfinite(v_tot_hat) & (v_tot_hat > 0)
        log_qij = np.log(v_tot_hat[ok] / v_oracle[ok])

        v_boot = np.nanvar(loaded["theta_boot"][:, :, 0], axis=1, ddof=1)
        okb = np.isfinite(v_oracle) & (v_oracle > 0) & np.isfinite(v_boot) & (v_boot > 0)
        log_boot = np.log(v_boot[okb] / v_oracle[okb])

        rows.append(dict(label=e["label"], qij=_mean_se(log_qij), boot=_mean_se(log_boot)))
    return rows


def _plot_f2_a(ax, rows: list) -> None:
    n = len(rows)
    y = np.arange(n)
    ms = plt.rcParams["lines.markersize"]
    dodge = 0.15
    ax.axvspan(-0.05, 0.05, color=BAND["materiality"], zorder=0)
    ax.axvline(0, color=METHOD["truth"]["color"], ls=":", lw=1.0, zorder=1)
    for key, dy in (("qij", -dodge), ("boot", dodge)):
        kw = METHOD[key]
        mu = np.array([r[key]["mean"] for r in rows])
        se = np.array([r[key]["se"] for r in rows])
        xerr = np.where(np.isfinite(se), 1.96 * se, 0.0)
        ax.errorbar(mu, y + dy, xerr=xerr, fmt=kw["marker"], color=kw["color"],
                    ms=ms, capsize=3, zorder=2, label=kw["label"])
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel(r"Mean $\log(\hat V_\mathrm{tot}/V_\mathrm{tot})$")
    ax.invert_yaxis()
    _panel_label(ax, "(a)")


def _plot_f2_b(ax, grid: pd.DataFrame) -> None:
    lw = plt.rcParams["lines.linewidth"]
    levels = grid["level"].to_numpy(dtype=float)
    band_lo = levels - 1.96 * grid["coverage_qij_se"].to_numpy(dtype=float)
    band_hi = levels + 1.96 * grid["coverage_qij_se"].to_numpy(dtype=float)
    ax.fill_between(levels, band_lo, band_hi, color=BAND["materiality"], zorder=0)
    ax.plot([0.49, 1.0], [0.49, 1.0], color=METHOD["truth"]["color"], ls=":", lw=1.0, zorder=1)
    ax.plot(levels, grid["coverage_qij"], color=METHOD["qij"]["color"], ls=METHOD["qij"]["ls"],
            marker=METHOD["qij"]["marker"], lw=lw, label=METHOD["qij"]["label"])
    ax.plot(levels, grid["coverage_bootstrap"], color=METHOD["boot"]["color"], ls=METHOD["boot"]["ls"],
            marker=METHOD["boot"]["marker"], lw=lw, label=METHOD["boot"]["label"])
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.49, 1.0)
    ax.set_ylim(0.49, 1.0)
    _panel_label(ax, "(b)")


def _f2_panel_c_rows(oracle_estimands: list, level: float = 0.95) -> list:
    """Width ratio QIJ/bootstrap at `level`, mean over draws, per
    estimand -- recomputed from `core.intervals`, never read."""
    rows = []
    for e in oracle_estimands:
        loaded = _load_draws(e["dir"], [e["output"]])
        n = loaded["theta_hat"].shape[0]
        widths_qij, widths_boot = [], []
        for i in range(n):
            th, v, a = loaded["theta_hat"][i], loaded["v_tot_hat"][i], loaded["a_bca"][i]
            if np.isfinite(th).all() and np.isfinite(v).all() and np.isfinite(a).all():
                lo, hi = qij_interval(th, v, a, level)[0]
                if np.isfinite(lo) and np.isfinite(hi):
                    widths_qij.append(hi - lo)
            lo_b, hi_b = percentile_interval(loaded["theta_boot"][i], level)[0]
            if np.isfinite(lo_b) and np.isfinite(hi_b):
                widths_boot.append(hi_b - lo_b)
        mw_qij = float(np.mean(widths_qij)) if widths_qij else float("nan")
        mw_boot = float(np.mean(widths_boot)) if widths_boot else float("nan")
        ratio = mw_qij / mw_boot if (np.isfinite(mw_boot) and mw_boot != 0) else float("nan")
        rows.append(dict(label=e["label"], ratio=ratio))
    return rows


def _plot_f2_c(ax, rows: list) -> None:
    n = len(rows)
    y = np.arange(n)
    ms = plt.rcParams["lines.markersize"]
    ax.axvspan(0.90, 1.10, color=BAND["materiality"], zorder=0)
    ax.axvline(1.0, color=METHOD["truth"]["color"], ls=":", lw=1.0, zorder=1)
    ratios = np.array([r["ratio"] for r in rows])
    finite = np.isfinite(ratios)
    ax.scatter(ratios[finite], y[finite], color=METHOD["qij"]["color"], s=ms ** 2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Width ratio (QIJ / bootstrap)")
    ax.invert_yaxis()
    _panel_label(ax, "(c)")


def fig2(run_dir: str, lncs: bool = False) -> plt.Figure:
    """F2 -- the validation trio (a) paired log-variance ratio against
    the oracle, (b) coverage calibration pooled over every estimand
    (from `qij.tables.coverage_grid`), (c) width ratio at 0.95. Panels
    (a)/(c) are restricted to estimands with an analytic influence
    (plan section 4); panel (b) pools over every discovered estimand."""
    _use_style(lncs)
    figsize = (4.80, 2.45) if lncs else (13.0, 5.0)

    oracle_estimands = _estimands(run_dir, require_oracle=True)
    if not oracle_estimands:
        raise FileNotFoundError(f"no oracle-influence estimands found under {run_dir}")
    grid = coverage_grid(run_dir)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)
    _plot_f2_a(axes[0], _f2_panel_a_rows(oracle_estimands))
    _plot_f2_b(axes[1], grid)
    _plot_f2_c(axes[2], _f2_panel_c_rows(oracle_estimands))

    if lncs:
        fig.subplots_adjust(left=0.16, right=0.985, top=0.92, bottom=0.40, wspace=0.75)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.30, wspace=0.55)

    handles = [Line2D([], [], color=METHOD[k]["color"], ls=METHOD[k]["ls"],
                       marker=METHOD[k]["marker"], label=METHOD[k]["label"])
               for k in ("qij", "boot")]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    return fig


# ---------------------------------------------------------------------------
# F3 -- partition-size curves
# ---------------------------------------------------------------------------

M_GRID = [4, 6, 8, 12, 16, 24, 32, 48, 64]   # plan section 5, pinned exactly

_F3_SERIES = ["xvq", "ivq_psi0", "ivq_true"]
_F3_COLOR = {"xvq": OI["green"], "ivq_psi0": METHOD["qij"]["color"], "ivq_true": METHOD["truth"]["color"]}
_F3_LS = {"xvq": "-", "ivq_psi0": "-", "ivq_true": ":"}
_F3_MARKER = {"xvq": "s", "ivq_psi0": "o", "ivq_true": None}
_F3_LABEL = {
    "xvq": r"$\mathcal{X}$-VQ, $M$ receptive fields",
    "ivq_psi0": r"$\mathcal{I}$-VQ, $M$ bins from $\hat\psi_0$",
    "ivq_true": r"$\mathcal{I}$-VQ, $M$ bins from the true $\psi$ (ceiling)",
}
_F3_PIPELINE_LABEL = r"QIJ after refinement ($L$ bins)"
_F3_COLUMN = {"xvq": "xvq", "ivq_psi0": "ivq_psi0", "ivq_true": "ivq_true"}


def _f3_partition_summary(part: pd.DataFrame, output: str) -> dict:
    out = {}
    for series in _F3_SERIES:
        col = f"{_F3_COLUMN[series]}_{output}"
        med, p25, p75 = [], [], []
        for M in M_GRID:
            v = part.loc[part["M"] == M, col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                med.append(np.median(v)); p25.append(np.percentile(v, 25)); p75.append(np.percentile(v, 75))
            else:
                med.append(np.nan); p25.append(np.nan); p75.append(np.nan)
        out[series] = dict(median=np.array(med), p25=np.array(p25), p75=np.array(p75))
    return out


def _f3_pipeline_marker(dir_path: str, output: str) -> tuple:
    """Median final bin count L_<output> and the median captured share
    V_btw_<output>/V_oracle_<output> the pipeline achieved, over all of
    a dir's production draws (not just the 200-draw partition grid)."""
    loaded = _load_draws(dir_path, [output])
    L = loaded["L"][:, 0]
    v_btw = pd.read_parquet(os.path.join(dir_path, "qij.parquet")).merge(
        pd.read_parquet(os.path.join(dir_path, "truth.parquet"))[["s", f"V_oracle_{output}"]],
        on="s", how="inner",
    )
    share = v_btw[f"V_btw_{output}"].to_numpy(dtype=float) / v_btw[f"V_oracle_{output}"].to_numpy(dtype=float)
    L = L[np.isfinite(L)]
    share = share[np.isfinite(share)]
    L_med = float(np.median(L)) if L.size else float("nan")
    share_med = float(np.median(share)) if share.size else float("nan")
    return L_med, share_med


def _plot_f3_panel(ax, summary: dict, L_med: float, share_med: float, title: str) -> None:
    lw = plt.rcParams["lines.linewidth"]
    ms = plt.rcParams["lines.markersize"]
    M = np.array(M_GRID, dtype=float)
    for series in _F3_SERIES:
        d = summary[series]
        finite = np.isfinite(d["median"])
        if not np.any(finite):
            continue
        ax.fill_between(M[finite], d["p25"][finite], d["p75"][finite],
                         color=_F3_COLOR[series], alpha=0.18, linewidth=0, zorder=1)
        ax.plot(M[finite], d["median"][finite], color=_F3_COLOR[series],
                ls=_F3_LS[series], marker=_F3_MARKER[series], ms=ms, lw=lw,
                zorder=2, label=_F3_LABEL[series])
    if np.isfinite(L_med):
        ax.axvline(L_med, color="#999999", ls=":", lw=1.0, alpha=0.7, zorder=0)
        if np.isfinite(share_med):
            ax.scatter([L_med], [share_med], marker="*", s=(ms * 2.4) ** 2,
                       color=METHOD["qij"]["color"], edgecolors="black", linewidths=0.5,
                       zorder=5, label=_F3_PIPELINE_LABEL)
    ax.set_xlim(0, M_GRID[-1] * 1.08)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)


def fig3(run_dir: str, lncs: bool = False) -> plt.Figure:
    """F3 -- captured share V_btw/V_tot vs partition size M, one panel
    per estimand with an analytic influence (`qij_partition.parquet`).
    x axis is LINEAR (the 18 September ruling), labelled "partition
    size M"."""
    _use_style(lncs)
    estimands = _estimands(run_dir, require_oracle=True)
    n = len(estimands)
    if n == 0:
        raise FileNotFoundError(f"no qij_partition.parquet estimands found under {run_dir}")
    n_cols = min(n, 4)
    n_rows = -(-n // n_cols)
    figsize = (4.80, 1.7 * n_rows + 0.7) if lncs else (4.0 * n_cols, 3.4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()
    for ax, e, lbl in zip(axes_flat, estimands, _panel_labels(n)):
        part = pd.read_parquet(os.path.join(e["dir"], "qij_partition.parquet"))
        summary = _f3_partition_summary(part, e["output"])
        L_med, share_med = _f3_pipeline_marker(e["dir"], e["output"])
        _plot_f3_panel(ax, summary, L_med, share_med, e["label"])
        _panel_label(ax, lbl)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.supxlabel("Partition size $M$", fontweight="bold")
    fig.supylabel(r"$V_\mathrm{btw}/V_\mathrm{tot}$", fontweight="bold")

    handles_by_label = {}
    for ax in axes_flat[:n]:
        hs, ls_ = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls_):
            handles_by_label.setdefault(l, h)
    order = [_F3_LABEL["xvq"], _F3_LABEL["ivq_psi0"], _F3_LABEL["ivq_true"], _F3_PIPELINE_LABEL]
    ordered = [(handles_by_label[l], l) for l in order if l in handles_by_label]
    if ordered:
        hs, ls_ = zip(*ordered)
        fig.legend(hs, ls_, loc="lower center", ncol=2)
    return fig


# ---------------------------------------------------------------------------
# F7 -- cost
# ---------------------------------------------------------------------------

_B_GRID_FRAC = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]


def _f7_bootstrap_curve(theta_c: np.ndarray, b_grid_frac: list, level: float = 0.95) -> tuple:
    """theta_c: (S, B) one output's replicates. Returns (curve rows,
    w_ref (S,) the full-B reference width per draw). Loop over draws
    and the b grid only, never over N."""
    S, B = theta_c.shape
    w_ref = np.full(S, np.nan)
    for s in range(S):
        lo, hi = percentile_interval(theta_c[s][:, None], level)[0]
        if np.isfinite(lo) and np.isfinite(hi):
            w_ref[s] = hi - lo

    curve = []
    for frac in b_grid_frac:
        b = max(2, int(round(frac * B)))
        errs = []
        for s in range(S):
            if not (np.isfinite(w_ref[s]) and w_ref[s] > 0):
                continue
            lo, hi = percentile_interval(theta_c[s, :b][:, None], level)[0]
            if np.isfinite(lo) and np.isfinite(hi):
                errs.append(abs((hi - lo) - w_ref[s]) / w_ref[s])
        curve.append(dict(b=b, mean=float(np.mean(errs)) if errs else float("nan")))
    return curve, w_ref


def _f7_qij_point(loaded: dict, output_idx: int, w_ref: np.ndarray, level: float = 0.95) -> dict:
    xs, errs = [], []
    n = loaded["theta_hat"].shape[0]
    for i in range(n):
        ref = w_ref[i]
        if not (np.isfinite(ref) and ref > 0):
            continue
        th, v, a = (loaded["theta_hat"][i, output_idx], loaded["v_tot_hat"][i, output_idx],
                    loaded["a_bca"][i, output_idx])
        nrows = loaded["normalized_rows"][i]
        if not (np.isfinite(th) and np.isfinite(v) and np.isfinite(a) and np.isfinite(nrows)):
            continue
        lo, hi = qij_interval(np.array([th]), np.array([v]), np.array([a]), level)[0]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        xs.append(float(nrows))
        errs.append(abs((hi - lo) - ref) / ref)
    if not xs:
        nan = float("nan")
        return dict(x=nan, y=nan, p25=nan, p75=nan)
    xs_a, errs_a = np.array(xs), np.array(errs)
    return dict(x=float(np.median(xs_a)), y=float(np.median(errs_a)),
                p25=float(np.percentile(errs_a, 25)), p75=float(np.percentile(errs_a, 75)))


def _plot_f7_a(ax, curve_rows: list, point_rows: list) -> None:
    lw = plt.rcParams["lines.linewidth"]
    ms = plt.rcParams["lines.markersize"]
    for cr in curve_rows:
        b = np.array([r["b"] for r in cr["curve"]], dtype=float)
        mean = np.array([r["mean"] for r in cr["curve"]])
        finite = np.isfinite(mean)
        ax.plot(b[finite], mean[finite], color=METHOD["boot"]["color"], ls=METHOD["boot"]["ls"],
                lw=lw * 0.6, alpha=0.7, zorder=1)
    for pt in point_rows:
        if not (np.isfinite(pt["x"]) and np.isfinite(pt["y"])):
            continue
        color = METHOD["qij"]["color"]
        if np.isfinite(pt["p25"]) and np.isfinite(pt["p75"]):
            ax.vlines(pt["x"], pt["p25"], pt["p75"], color=color, lw=lw * 0.8, zorder=3)
        ax.plot(pt["x"], pt["y"], marker=METHOD["qij"]["marker"], color=color, ms=ms * 1.2,
                ls="none", zorder=4)
        ax.annotate(pt["label"], (pt["x"], pt["y"]), textcoords="offset points", xytext=(5, 4),
                    fontsize=plt.rcParams["xtick.labelsize"], color=color)
    h1 = ax.plot([], [], color=METHOD["boot"]["color"], ls=METHOD["boot"]["ls"], lw=lw * 0.6,
                 label="Bootstrap (first $b$ replicates)")[0]
    h2 = ax.plot([], [], marker=METHOD["qij"]["marker"], color=METHOD["qij"]["color"], ls="none",
                 ms=ms * 1.2, label="QIJ (median, IQR)")[0]
    _legend(ax, handles=[h1, h2], loc="upper right")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Normalized rows")
    ax.set_ylabel("Relative width error (vs. converged bootstrap)")
    _panel_label(ax, "(a)")


def _f7_wall_time_rows(run_dir: str) -> list:
    """One row per (dataset, estimator) directory, not per output: the
    Fundamental Plane's four outputs share one run and so one wall
    time (`wall_time_total`/boot's `wall_time` are unsuffixed, plan
    section 5 amendment 3), so this collapses to a single row labelled
    "FP (4 outputs)" rather than the same numbers repeated four times
    -- the honest rendering of one shared cost, not a simplification."""
    rows = []
    for dataset, estimator, path in _product_dirs(run_dir):
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        loaded = _load_draws(path, outputs)
        qij_wt = loaded["wall_time_qij"]
        qij_wt = qij_wt[np.isfinite(qij_wt)]
        boot_wt = loaded["wall_time_boot"]
        boot_wt = boot_wt[np.isfinite(boot_wt)]
        label = (f"{dataset.upper()} ({len(outputs)} outputs)" if len(outputs) > 1
                  else f"{dataset}/{estimator}")
        rows.append(dict(
            label=label,
            qij=float(np.median(qij_wt)) if qij_wt.size else float("nan"),
            boot=float(np.median(boot_wt)) if boot_wt.size else float("nan"),
        ))
    return rows


def _plot_f7_b(ax, rows: list) -> None:
    n = len(rows)
    x = np.arange(n)
    width = 0.35
    qij_vals = np.array([r["qij"] for r in rows])
    boot_vals = np.array([r["boot"] for r in rows])
    ax.bar(x - width / 2, qij_vals, width, color=METHOD["qij"]["color"], label=METHOD["qij"]["label"],
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, boot_vals, width, color=METHOD["boot"]["color"], label=METHOD["boot"]["label"],
           edgecolor="black", linewidth=0.5, hatch="//")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], rotation=40, ha="right")
    ax.set_ylabel("Wall time per run (s)")
    _legend(ax, loc="upper left")
    _panel_label(ax, "(b)")


def fig7(run_dir: str, lncs: bool = False) -> plt.Figure:
    """F7 -- cost. (a) relative width error vs normalized rows: the
    bootstrap's own curve (width error of its first b replicates
    against its full-B reference) per estimand, plus QIJ's median/IQR
    point at the same reference. (b) wall time per run, one pair of
    bars per (dataset, estimator) directory -- FP appears once."""
    _use_style(lncs)
    figsize = (4.80, 2.20) if lncs else (11.0, 4.6)

    curve_rows, point_rows = [], []
    for dataset, estimator, path in _product_dirs(run_dir):
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        loaded = _load_draws(path, outputs)
        for j, o in enumerate(outputs):
            curve, w_ref = _f7_bootstrap_curve(loaded["theta_boot"][:, :, j], _B_GRID_FRAC)
            pt = _f7_qij_point(loaded, j, w_ref)
            label = f"{dataset}/{estimator}" if len(outputs) == 1 else f"{dataset}/{estimator}:{o}"
            curve_rows.append(dict(label=label, curve=curve))
            point_rows.append(dict(label=label, **pt))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _plot_f7_a(axes[0], curve_rows, point_rows)
    _plot_f7_b(axes[1], _f7_wall_time_rows(run_dir))
    return fig


# ---------------------------------------------------------------------------
# F9 -- the initial influence estimate against the truth
# ---------------------------------------------------------------------------

def _plot_f9_scatter(ax, psi0: np.ndarray, psi: np.ndarray, title: str, panel_lbl: str) -> None:
    finite = np.isfinite(psi0) & np.isfinite(psi)
    psi0, psi = psi0[finite], psi[finite]
    if psi.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        _panel_label(ax, panel_lbl)
        return
    lo = float(min(psi.min(), psi0.min()))
    hi = float(max(psi.max(), psi0.max()))
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=METHOD["truth"]["color"], ls=":", lw=1.2, zorder=1)
    ax.scatter(psi, psi0, s=7, color=METHOD["qij"]["color"], alpha=0.35, linewidths=0, zorder=2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(r"True $\psi(x_i)$")
    ax.set_ylabel(r"$\hat\psi_0(x_i)$")
    ax.set_title(title)
    _panel_label(ax, panel_lbl)

    if psi0.size >= 2 and np.std(psi0) > 0 and np.std(psi) > 0:
        rs = float(spearmanr(psi0, psi).correlation)
        text = f"$r_s = {rs:.3f}$"
    else:
        text = "$r_s$ undefined"
    ax.text(0.04, 0.96, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=plt.rcParams.get("annotation.fontsize", 9))


def _plot_f9_curve(ax, dir_path: str, output: str, panel_lbl: str) -> None:
    """The Pareto shape's psi_hat_0, ranked, with its +/- sigma band
    (`qij_points.parquet`'s own psi0_<o>/sigma_<o> columns). No
    prototype-influence overlay: see the module docstring."""
    pts = pd.read_parquet(os.path.join(dir_path, "qij_points.parquet"))
    psi0 = pts[f"psi0_{output}"].to_numpy(dtype=float)
    sigma = pts[f"sigma_{output}"].to_numpy(dtype=float)
    order = np.argsort(psi0)
    x = np.arange(order.size)
    ax.axhline(0, color=METHOD["truth"]["color"], ls=":", lw=1.0, zorder=0)
    ax.fill_between(x, (psi0 - sigma)[order], (psi0 + sigma)[order],
                     color=METHOD["qij"]["color"], alpha=0.15, linewidth=0, zorder=0,
                     label=r"$\hat\psi_0 \pm \sigma$")
    ax.plot(x, psi0[order], color=METHOD["qij"]["color"], lw=1.2, zorder=1,
            label=r"Initial influence estimate $\hat\psi_0$")
    ax.set_xlabel(r"Points, ranked by $\hat\psi_0$")
    ax.set_ylabel("Influence")
    ax.set_title("Pareto shape: initial influence estimate")
    _panel_label(ax, panel_lbl)
    _legend(ax, loc="best")


def fig9(run_dir: str, outputs: list = None, lncs: bool = False) -> plt.Figure:
    """F9 -- four scatter panels of psi_hat_0 against the true psi
    (`qij_points.parquet`), plus the Pareto-shape curve panel. `outputs`
    optionally names exactly four (dataset, estimator, output) triples
    to draw (default: the first four discovered oracle-influence
    estimands, excluding pareto/shape, which the curve panel already
    covers) -- the pinned products do not say which four an "estimand"
    list should be, so this default is this file's own choice, not a
    spec value."""
    _use_style(lncs)
    curve_dir = None
    for dataset, estimator, path in _product_dirs(run_dir):
        if dataset == "pareto" and estimator == "shape":
            curve_dir = path
            curve_truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
            curve_output = _outputs(curve_truth)[0]
            break
    if curve_dir is None:
        raise FileNotFoundError(f"no pareto/shape estimand found under {run_dir}")

    all_oracle = _estimands(run_dir, require_oracle=True)
    candidates = [e for e in all_oracle
                  if not (e["dataset"] == "pareto" and e["estimator"] == "shape")]
    if outputs is None:
        chosen = candidates[:4]
    else:
        by_key = {(e["dataset"], e["estimator"], e["output"]): e for e in all_oracle}
        chosen = [by_key[k] for k in outputs]
    if len(chosen) < 4:
        raise FileNotFoundError(
            f"fewer than 4 oracle-influence estimands with qij_points.parquet under {run_dir}"
        )
    chosen = chosen[:4]

    figsize = (4.80, 3.4) if lncs else (12.0, 7.0)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0])]

    for ax, e, lbl in zip(axes, chosen, ["(a)", "(b)", "(c)", "(d)"]):
        pts = pd.read_parquet(os.path.join(e["dir"], "qij_points.parquet"))
        psi0 = pts[f"psi0_{e['output']}"].to_numpy(dtype=float)
        psi = pts[f"psi_{e['output']}"].to_numpy(dtype=float)
        _plot_f9_scatter(ax, psi0, psi, e["label"], lbl)

    ax_curve = fig.add_subplot(gs[1, 1:3])
    _plot_f9_curve(ax_curve, curve_dir, curve_output, "(e)")
    return fig


# ---------------------------------------------------------------------------
# F10 -- cost against N
# ---------------------------------------------------------------------------

def _f10_coverage(loaded: dict, output_idx: int, level: float) -> dict:
    n = loaded["theta_hat"].shape[0]
    hit_q = tot_q = hit_b = tot_b = 0
    for i in range(n):
        tt = loaded["theta_true"][i, output_idx]
        th, v, a = (loaded["theta_hat"][i, output_idx], loaded["v_tot_hat"][i, output_idx],
                    loaded["a_bca"][i, output_idx])
        if np.isfinite(tt) and np.isfinite(th) and np.isfinite(v) and np.isfinite(a):
            lo, hi = qij_interval(np.array([th]), np.array([v]), np.array([a]), level)[0]
            if np.isfinite(lo) and np.isfinite(hi):
                tot_q += 1
                hit_q += int(lo <= tt <= hi)
        lo_b, hi_b = percentile_interval(loaded["theta_boot"][i, :, output_idx][:, None], level)[0]
        if np.isfinite(tt) and np.isfinite(lo_b) and np.isfinite(hi_b):
            tot_b += 1
            hit_b += int(lo_b <= tt <= hi_b)
    p_q = hit_q / tot_q if tot_q else float("nan")
    se_q = float(np.sqrt(p_q * (1 - p_q) / tot_q)) if tot_q else float("nan")
    p_b = hit_b / tot_b if tot_b else float("nan")
    se_b = float(np.sqrt(p_b * (1 - p_b) / tot_b)) if tot_b else float("nan")
    return dict(p_qij=p_q, se_qij=se_q, p_boot=p_b, se_boot=se_b)


def _plot_f10_rows(ax, rows_stat: list, B_ref) -> None:
    Ns = np.array([r["N"] for r in rows_stat], dtype=float)
    med = np.array([r["median"] for r in rows_stat])
    p25 = np.array([r["p25"] for r in rows_stat])
    p75 = np.array([r["p75"] for r in rows_stat])
    ax.fill_between(Ns, p25, p75, color=METHOD["qij"]["color"], alpha=0.18, linewidth=0)
    ax.plot(Ns, med, color=METHOD["qij"]["color"], ls=METHOD["qij"]["ls"],
            marker=METHOD["qij"]["marker"], label=METHOD["qij"]["label"])
    if B_ref is not None:
        ax.axhline(B_ref, color=METHOD["boot"]["color"], ls=METHOD["boot"]["ls"], lw=1.2,
                   label="Bootstrap (fixed $B$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("Normalized rows")
    _legend(ax, loc="best")
    _panel_label(ax, "(a)")


def _plot_f10_time(ax, time_stat: list) -> None:
    Ns = np.array([r["N"] for r in time_stat], dtype=float)
    med = np.array([r["median"] for r in time_stat])
    p25 = np.array([r["p25"] for r in time_stat])
    p75 = np.array([r["p75"] for r in time_stat])
    ax.fill_between(Ns, p25, p75, color=METHOD["qij"]["color"], alpha=0.18, linewidth=0)
    ax.plot(Ns, med, color=METHOD["qij"]["color"], ls=METHOD["qij"]["ls"], marker=METHOD["qij"]["marker"])
    ax.axhline(1.0, color=METHOD["truth"]["color"], ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("Wall time ratio (QIJ / bootstrap)")
    _panel_label(ax, "(b)")


def _plot_f10_coverage(ax, cov_rows: list, outputs: list, level: float) -> None:
    Ns_all = sorted(set(r["N"] for r in cov_rows))
    lw = plt.rcParams["lines.linewidth"]
    for i, o in enumerate(outputs):
        color = OUTPUT_COLORS[i % len(OUTPUT_COLORS)]
        for method, ls, marker, key_p, key_se in (
            ("qij", "-", "o", "p_qij", "se_qij"), ("boot", "--", "s", "p_boot", "se_boot"),
        ):
            xs, ys, ses = [], [], []
            for N in Ns_all:
                match = [r for r in cov_rows if r["N"] == N and r["output"] == o]
                if not match:
                    continue
                xs.append(N)
                ys.append(match[0][key_p])
                ses.append(match[0][key_se])
            xs_a = np.array(xs, dtype=float)
            ys_a = np.array(ys, dtype=float)
            ses_a = np.array(ses, dtype=float)
            band = np.where(np.isfinite(ses_a), 1.96 * ses_a, 0.0)
            ax.fill_between(xs_a, ys_a - band, ys_a + band, color=color, alpha=0.12, linewidth=0)
            ax.plot(xs_a, ys_a, color=color, ls=ls, marker=marker, ms=4, lw=lw * 0.9,
                    label=f"{o} ({METHOD[method]['label']})")
    ax.axhline(level, color=METHOD["truth"]["color"], ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("$N$")
    ax.set_ylabel(f"Coverage at {level:.0%}")
    _legend(ax, loc="lower left", ncol=2, fontsize=plt.rcParams["legend.fontsize"] * 0.85)
    _panel_label(ax, "(c)")


def fig10(run_dirs: dict, lncs: bool = False, level: float = 0.95) -> plt.Figure:
    """F10 -- cost against N, from the cost-vs-N study's FP runs.
    `run_dirs` maps N -> the run_dir for that N's cost_vs_n run (see
    the module docstring: no product records N, so this is supplied by
    the caller, not discovered).

    (a) normalized rows vs N (QIJ, median/IQR over draws; bootstrap's
    is the fixed replicate count B, a horizontal reference).
    (b) wall time ratio QIJ/bootstrap vs N, within the same draw on the
    same worker (cost layer 2, plan section 8) -- never absolute
    seconds compared across the N sweep's separate runs.
    (c) coverage at `level` of both intervals, one line per (output,
    method), NOT pooled across the Fundamental Plane's four outputs
    (the 18 September ruling): they share one run and one set of bins,
    so a pooled band would be tighter than it has earned.
    """
    _use_style(lncs)
    Ns = sorted(run_dirs)
    rows_stat, time_stat, cov_rows = [], [], []
    B_ref = None
    outputs_ref = None

    for N in Ns:
        path = os.path.join(run_dirs[N], "fp", "fp")
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        if outputs_ref is None:
            outputs_ref = outputs
        loaded = _load_draws(path, outputs)
        if B_ref is None:
            B_ref = loaded["theta_boot"].shape[1]

        nrows = loaded["normalized_rows"]
        nrows = nrows[np.isfinite(nrows)]
        rows_stat.append(dict(
            N=N,
            median=float(np.median(nrows)) if nrows.size else float("nan"),
            p25=float(np.percentile(nrows, 25)) if nrows.size else float("nan"),
            p75=float(np.percentile(nrows, 75)) if nrows.size else float("nan"),
        ))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = loaded["wall_time_qij"] / loaded["wall_time_boot"]
        ratio = ratio[np.isfinite(ratio)]
        time_stat.append(dict(
            N=N,
            median=float(np.median(ratio)) if ratio.size else float("nan"),
            p25=float(np.percentile(ratio, 25)) if ratio.size else float("nan"),
            p75=float(np.percentile(ratio, 75)) if ratio.size else float("nan"),
        ))

        for j, o in enumerate(outputs):
            cov = _f10_coverage(loaded, j, level)
            cov_rows.append(dict(N=N, output=o, **cov))

    figsize = (4.80, 2.3) if lncs else (13.5, 4.4)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _plot_f10_rows(axes[0], rows_stat, B_ref)
    _plot_f10_time(axes[1], time_stat)
    _plot_f10_coverage(axes[2], cov_rows, outputs_ref, level)
    return fig
