"""
The paper's figures (`QIJ_figure_spec_final.md`, 19 September 2026): Figure
A (accuracy, replaces F2), Figure B (refined influence against the truth,
replaces F3 and F9), Figure C (cost, replaces F7), Figure D (cost against
sample size, was F10, design unchanged, now with an optional IMF sweep
pair). Every figure here is a pure function of the products `qij.study`
writes under `<run_dir>/<dataset>/<estimator>/` -- `truth.parquet`,
`qij.parquet`, `boot.h5`, `qij_points.parquet`, `qij_prototypes.parquet` --
and, for Figure C, the separate timing run's own products. Nothing else is
read: no estimator is re-run, no dataset is redrawn, no interval is read
(none is stored -- every interval here is recomputed from
`core.intervals.qij_interval` or `core.intervals.percentile_interval`).
QIJ's second-order interval was built, checked against plan §34's decision
rule on the rev-7 products and rejected there (it lowered coverage on
every coordinate at every level, plan §36.1, §36.2 ruling 6); this module
never carried a way to plot it, and now there is only ever one QIJ
interval to plot, so no figure here takes an `interval` parameter or
names one in a corner annotation.

One coordinate set, one grid (spec, 19 September). Every figure that shows
more than one estimand shows exactly these six, in exactly these
positions, taken from the spec's table:

    (1,1) MVT nu           (1,2) MVT P_tail        (1,3) FP a
    (2,1) FP scatter        (2,2) IMF slope          (2,3) IMF p

Pareto, FP b/c and the IMF's M*/gamma_shape/gamma_scale appear only in
Table T1 (`tables.py`), never in these four figures -- `_COORDS` below is
this file's one definition of the six, and every figure iterates it rather
than discovering estimands from the run directory the way the old F2/F3/F7
did (plan section 12: one way to do each thing).

THE AXIS FOR FIGURE B, and why it comes from `qij_prototypes.parquet` and
not `qij_points.parquet`. The spec's table gives, for each coordinate, "the
axis for the influence figure": radius in whitened coordinates for MVT
(no single native coordinate is meaningful for an elliptically symmetric
draw), log sigma -- data coordinate 0 -- for FP, log stellar mass for IMF;
in every case the value plotted is the MEAN OVER THE PROTOTYPE'S RECEPTIVE
FIELD, because the true influence of an empirical estimator is not defined
AT a single point, let alone at a prototype that may not itself be a data
point. `qij_points.parquet` carries no per-point data-space coordinate at
all (`study.py`'s `_points_frame`: `s, i, bmu`, then per output `psi0,
psi, sigma, psi_hat, bin_label`) -- there is no column to group by `bmu`
and average, which is what the interface notes for this task describe.
What IS available, and turns out to be exactly the quantity needed: `w_0,
w_1, ...` in `qij_prototypes.parquet`, "the prototype's position in T's
own native coordinates" (`qij.py`: `W_X = inverse(xvq.centers)`). Because
`fit_xvq`'s prototypes are k-means centroids -- the mean, in the
quantizer's own coordinates, of exactly the points in that receptive field
-- and because every `inverse` this package ever uses (`mvt_vq_transform`,
or the identity for every other dataset) is affine, `inverse` commutes
with the receptive-field mean: `w_j` IS ALREADY `mean_{i in RF_j}
(native coordinates of x_i)`, not merely a point somewhere near it. So for
FP (whose native coordinate 0 already IS log sigma, `datasets._fp_pool`'s
`[log_sigma, log_I_e, log_R_half]`) and for IMF (`log10` of `w_0`, the
receptive field's mean raw mass -- see the note below), `w_j` supplies the
spec's axis exactly, with no further computation needed at all.

For MVT this is only a PROXY, and that is flagged in the function
docstrings and in this task's own report rather than worked around
silently: the spec wants radius in the 𝒳-VQ's WHITENED coordinates, but
`qij_prototypes.parquet` stores only the NATIVE-coordinate `w_j` --
`xvq.centers`, the actual whitened centroids `fit_xvq` computed, are never
written to any product, and the per-draw mean/std `mvt_vq_transform` used
to whiten are not recoverable downstream without redrawing the dataset
(which this module, like its predecessor, refuses to do). What this file
plots instead is the Euclidean norm of the NATIVE `w_j`. For this
particular dataset that is a reasonable proxy -- the multivariate t drawn
here is exchangeable across its 10 coordinates with population mean 0 and
one common per-coordinate variance, so `mvt_vq_transform`'s empirical
per-coordinate standardization is, in expectation, close to a single
scalar rescaling that would not change the ORDER of prototypes along the
radius axis, only its units -- but it is not what the spec literally asks
for, and a future revision that wants the exact quantity should have
`study.py` write `xvq.centers` into `qij_prototypes.parquet` alongside
`w_*`.

For IMF, `log10(w_0)` is `log10` of the receptive field's mean LINEAR
mass, not the mean of the receptive field's LOG mass -- the two differ by
Jensen's inequality, and the gap is the same structural point as MVT's:
the 𝒳-VQ clusters IMF draws on raw mass (no `vq_transform` is registered
for `imf`, `study.py`'s `_VQ_TRANSFORM` table), so the only receptive-field
mean available downstream is a mean in linear space. Log is applied to
that mean afterward, purely as this axis's display unit.

BOX-RULE EXCLUSION is not reimplemented here. The IMF estimator returns
NaN for a fit resting on its box, and that NaN already propagates through
`theta_hat`, `V_tot_hat`, the bootstrap replicates and the bin
constituents in `truth.parquet`/`qij.parquet`/`boot.h5`; every statistic
in this file is computed with `np.nanmedian`/`np.nanpercentile`/
`np.nanvar(ddof=...)` or an explicit `np.isfinite` mask, so a box-rule draw
drops out of every figure's numbers on its own.
"""

from __future__ import annotations

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from qij.core.intervals import percentile_interval, qij_interval

__all__ = ["fig_a", "fig_b", "fig_c", "fig_d"]

# ---------------------------------------------------------------------------
# Style (QIJ_figure_style.md, inlined -- this task's only files are
# figures.py and scripts/make_figures.py, so there is no separate style
# module). Palette and rcParams unchanged from the module this replaces;
# the LNCS sizes below are the ones fixed in the style guide's section 8 on
# 19 September, verified against the revision's own llncs.cls.
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
SECONDARY = "#444444"   # a neutral third hue for a twin-axis quantity that
                         # is itself a ratio (wall time, width) rather than
                         # a second method -- never confused with QIJ/boot.

# 4-hue qualitative palette for the panels that must encode a per-output
# categorical axis at once (Figure D's coverage panels): method keeps its
# own linestyle/marker (solid o = QIJ, dashed s = bootstrap) as the second
# cue there, same convention as the module this replaces used for F10(c).
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

# Final LNCS sizes (style guide section 8, 19 September, verified against
# the revision's llncs.cls: text width 347.12pt = 4.80in). Draft sizes
# (lncs=False) are larger only so a screen render is legible while
# iterating; they carry no meaning for the paper and are never placed in
# the tex.
FIGSIZE_A_LNCS = (4.80, 2.20)
FIGSIZE_A_DRAFT = (12.0, 5.0)
FIGSIZE_B_LNCS = (4.80, 3.00)
FIGSIZE_B_DRAFT = (12.0, 7.0)
FIGSIZE_C_LNCS = (4.80, 3.00)
FIGSIZE_C_DRAFT = (12.0, 7.0)
FIGSIZE_D1_LNCS = (4.80, 2.20)     # 1x2: FP only
FIGSIZE_D1_DRAFT = (11.0, 4.4)
FIGSIZE_D2_LNCS = (4.80, 3.60)     # 2x2: FP row + IMF sweep row
FIGSIZE_D2_DRAFT = (11.0, 8.0)

# `plt.rcParams` has no real "annotation.fontsize" key -- the module this
# replaces called `plt.rcParams.get("annotation.fontsize", 9)` for its
# corner text, which silently always returned the hardcoded default
# because that key does not exist and `dict.get` does not validate it, so
# every corner annotation rendered at 9pt even in LNCS mode, ignoring the
# style guide's "annotation: 7pt at final size" row. Fixed here with a
# real module-level switch that `_use_style` sets.
_STATE = {"annotation_fontsize": 9}


def _use_style(lncs: bool) -> None:
    plt.rcParams.update(RC_LNCS if lncs else RC)
    _STATE["annotation_fontsize"] = 7 if lncs else 9


def _panel_label(ax, text: str, dy: float = 1.16, dx: float = -0.13) -> None:
    """`dy` (axes-fraction) is a parameter, not a constant, because the
    same absolute clearance above a panel's title means a very different
    RELATIVE offset depending on how tall that panel's own axes are: a
    2x3 grid's row is a fraction of the figure a 1x3 row is not, so a
    `dy` generous enough to clear a wide, centred title in a narrow 2x3
    panel (Figures B, C) would push the label off the TOP of a 1x3
    figure (Figure A, which has no title to clear in the first place).
    `dx` is likewise tuned per grid: a centred title's rendered width is
    a much bigger share of a narrow panel's own width, so how far left
    the label has to sit to clear it differs the same way."""
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=plt.rcParams["axes.titlesize"], fontweight="bold",
            va="top", ha="left")


def _legend(ax_or_fig, title=None, **kw):
    leg = ax_or_fig.legend(title=title, **kw)
    if title:
        leg.get_title().set_fontweight("bold")
    return leg


# ---------------------------------------------------------------------------
# The six coordinates, fixed by the spec's table -- every figure iterates
# this list and nothing else discovers an estimand.
# ---------------------------------------------------------------------------

_COORDS = [
    dict(dataset="mvt", estimator="nu", output="nu",
         label=r"MVT $\nu$", axis="radius", row=0, col=0),
    dict(dataset="mvt", estimator="tail", output="P_tail",
         label=r"MVT $P_\mathrm{tail}$", axis="radius", row=0, col=1),
    dict(dataset="fp", estimator="fp", output="a",
         label=r"FP $a$", axis="logsigma", row=0, col=2),
    dict(dataset="fp", estimator="fp", output="scatter",
         label="FP scatter", axis="logsigma", row=1, col=0),
    dict(dataset="imf", estimator="imf", output="slope",
         label="IMF slope", axis="logmass", row=1, col=1),
    dict(dataset="imf", estimator="imf", output="p",
         label=r"IMF $p$", axis="logmass", row=1, col=2),
]

_AXIS_LABEL = {
    "radius": r"radius $\|x\|$",
    "logsigma": r"$\log\sigma$",
    "logmass": r"$\log_{10}$ mass",
}

_LEVEL = 0.95   # every figure's one coverage/width level (spec: "at 0.95")


# ---------------------------------------------------------------------------
# Reading the products
# ---------------------------------------------------------------------------

def _product_dir(run_dir: str, dataset: str, estimator: str) -> str:
    """`<run_dir>/<dataset>/<estimator>`, checked rather than assumed to
    exist -- `_COORDS` names datasets/estimators the spec fixed, not ones
    discovered from what a particular run happened to write, so a run
    missing one is a clear error here rather than a KeyError three calls
    later."""
    path = os.path.join(run_dir, dataset, estimator)
    if not os.path.exists(os.path.join(path, "truth.parquet")):
        raise FileNotFoundError(
            f"no truth.parquet under {path} (dataset={dataset!r}, "
            f"estimator={estimator!r}, needed by the spec's fixed six coordinates)"
        )
    return path


def _outputs(truth_df: pd.DataFrame) -> list:
    prefix = "theta_hat_"
    return [c[len(prefix):] for c in truth_df.columns if c.startswith(prefix)]


def _load_draws(estimator_dir: str, outputs: list) -> dict:
    """truth.parquet, qij.parquet and boot.h5 for one (dataset, estimator)
    directory, joined on `s` and aligned to `outputs`' order -- the same
    shape `qij.tables._load_draws` builds, kept as this module's own copy
    (plan section 12: each downstream module owns its own loader) but
    without the oracle-influence columns tables.py also carries: no figure
    in this file compares against V_oracle any more (Figure A(a) compares
    against V_MC, the Monte-Carlo variance of theta_hat over draws, per
    the 19 September spec), so there is nothing here to load it for.
    `qij.parquet` may also carry `bin_mass_<o>`/`bin_influence_<o>`/
    `bin_d2T_<o>` (the second-order interval's inputs, kept in the
    product regardless of what reads them, plan §36.2 ruling 6) --
    nothing here loads them, and the merge below is indifferent to a
    source frame carrying extra columns, so a run that has them and a
    run that does not load identically."""
    truth = pd.read_parquet(os.path.join(estimator_dir, "truth.parquet"))
    qij_df = pd.read_parquet(os.path.join(estimator_dir, "qij.parquet"))
    df = truth.merge(qij_df, on="s", how="inner")

    with h5py.File(os.path.join(estimator_dir, "boot.h5"), "r") as h5f:
        theta_boot = h5f["theta"][...]
        s_boot = h5f["s"][...]
        wall_time_boot = h5f["wall_time"][...]
        n_failed_boot = h5f["n_failed"][...]
        boot_outputs = [o.decode() if isinstance(o, bytes) else str(o)
                         for o in h5f.attrs["outputs"]]

    row_of_s = pd.Series(np.arange(len(s_boot)), index=s_boot)
    order = row_of_s.loc[df["s"].to_numpy()].to_numpy()
    col_order = [boot_outputs.index(o) for o in outputs]
    theta_boot = theta_boot[order][:, :, col_order]
    wall_time_boot = wall_time_boot[order]
    n_failed_boot = n_failed_boot[order]

    return dict(
        s=df["s"].to_numpy(),
        outputs=outputs,
        theta_true=df[[f"theta_true_{o}" for o in outputs]].to_numpy(),
        theta_hat=df[[f"theta_hat_{o}" for o in outputs]].to_numpy(),
        v_tot_hat=df[[f"V_tot_hat_{o}" for o in outputs]].to_numpy(),
        a_bca=df[[f"a_bca_{o}" for o in outputs]].to_numpy(),
        theta_boot=theta_boot,
        wall_time_qij=df["wall_time_total"].to_numpy(),
        wall_time_boot=wall_time_boot,
        n_failed_boot=n_failed_boot,
        B=theta_boot.shape[1],
        n_failed_qij=df["n_failed"].to_numpy(),
        evals_total=df["evals_total"].to_numpy(),
        normalized_rows=df["normalized_rows"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# The interval helper -- the one place [lo, hi] arrays get built for the
# QIJ interval. Every figure that draws it calls this, never `qij_interval`
# directly, so there is exactly one place doing the per-draw loop.
# ---------------------------------------------------------------------------

def _qij_lo_hi(loaded: dict, j: int, level: float) -> np.ndarray:
    """(n, 2) [lo, hi] for output index `j`, over every draw in `loaded`,
    at `level`, from `core.intervals.qij_interval`. A draw whose
    `theta_hat`/`V_tot_hat`/`a_bca` is not all finite for this output
    (a box-rule or QIJ-side failure, plan §36.2 ruling 5) is left NaN
    rather than passed in -- `qij_interval` would produce NaN from it
    anyway, but the finiteness check is made explicit here rather than
    relied on implicitly."""
    n = loaded["theta_hat"].shape[0]
    lo_hi = np.full((n, 2), np.nan)
    for i in range(n):
        th, v, a = loaded["theta_hat"][i, j], loaded["v_tot_hat"][i, j], loaded["a_bca"][i, j]
        if np.isfinite(th) and np.isfinite(v) and np.isfinite(a):
            lo_hi[i] = qij_interval(np.array([th]), np.array([v]), np.array([a]), level)[0]
    return lo_hi


def _boot_lo_hi(loaded: dict, j: int, level: float, b: int = None) -> np.ndarray:
    """(n, 2) [lo, hi], the bootstrap's percentile interval at `level` from
    the first `b` replicates (all of them if `b` is None) -- the prefix
    that gives Figure C's cost-against-replicates curve."""
    n = loaded["theta_hat"].shape[0]
    arr = loaded["theta_boot"][:, :b, j] if b is not None else loaded["theta_boot"][:, :, j]
    lo_hi = np.full((n, 2), np.nan)
    for i in range(n):
        lo_hi[i] = percentile_interval(arr[i][:, None], level)[0]
    return lo_hi


def _coverage_se(theta_true: np.ndarray, lo_hi: np.ndarray) -> dict:
    """Empirical coverage and its Monte Carlo SE sqrt(p(1-p)/n), over the
    draws where both `theta_true` and the interval are finite (a failed
    evaluation or a box-rule NaN is excluded, not counted as a miss)."""
    lo, hi = lo_hi[:, 0], lo_hi[:, 1]
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(theta_true)
    covered = (theta_true >= lo) & (theta_true <= hi)
    ind = np.where(ok, covered.astype(float), np.nan)
    valid = ind[np.isfinite(ind)]
    n = int(valid.size)
    if n == 0:
        return dict(p=float("nan"), se=float("nan"), n=0)
    p = float(valid.mean())
    return dict(p=p, se=float(np.sqrt(p * (1.0 - p) / n)), n=n)


def _coverage_pair(loaded: dict, j: int, level: float) -> tuple:
    """(qij coverage dict, bootstrap coverage dict) at `level`, for output
    index `j` -- the one call both Figure A(b) and Figure D's coverage
    panels make."""
    lo_hi_qij = _qij_lo_hi(loaded, j, level)
    lo_hi_boot = _boot_lo_hi(loaded, j, level)
    theta_true = loaded["theta_true"][:, j]
    return _coverage_se(theta_true, lo_hi_qij), _coverage_se(theta_true, lo_hi_boot)


def _width_ratio(loaded: dict, j: int, level: float) -> np.ndarray:
    """(n,) QIJ/bootstrap interval width at `level`, per draw, NaN where
    either interval failed."""
    lo_hi_qij = _qij_lo_hi(loaded, j, level)
    lo_hi_boot = _boot_lo_hi(loaded, j, level)
    w_qij = lo_hi_qij[:, 1] - lo_hi_qij[:, 0]
    w_boot = lo_hi_boot[:, 1] - lo_hi_boot[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = w_qij / w_boot
    bad = ~np.isfinite(w_boot) | (w_boot == 0)
    return np.where(bad, np.nan, ratio)


def _mean_se(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return dict(mean=float("nan"), se=float("nan"))
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return dict(mean=mean, se=se)


def _stat(x: np.ndarray) -> dict:
    """NaN-aware median/p25/p75 -- box-rule and failed-evaluation NaNs
    drop out on their own (module docstring)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(median=float("nan"), p25=float("nan"), p75=float("nan"))
    return dict(median=float(np.median(x)), p25=float(np.percentile(x, 25)),
                p75=float(np.percentile(x, 75)))


# ---------------------------------------------------------------------------
# Figure A -- accuracy (replaces F2): three panels, six rows each.
# ---------------------------------------------------------------------------

def _row_panel(ax, rows: list, key_fn, xlabel: str, ref_line, band, xlim,
               panel_lbl: str, show_labels: bool) -> None:
    """The one dot-and-whisker layout Figure A's three panels share: a
    horizontal reference (`ref_line`, e.g. 0, 1.0 or the nominal level),
    an optional shaded tolerance band, and two dodged markers (QIJ above,
    bootstrap below its row) with asymmetric or symmetric whiskers from
    `key_fn`. `key_fn(row, method)` returns `(center, lo_err, hi_err)`."""
    n = len(rows)
    y = np.arange(n)
    ms = plt.rcParams["lines.markersize"]
    dodge = 0.16
    if band is not None:
        ax.axvspan(band[0], band[1], color=BAND["materiality"], zorder=0)
    ax.axvline(ref_line, color=METHOD["truth"]["color"], ls=":", lw=1.0, zorder=1)
    for key, dy in (("qij", -dodge), ("boot", dodge)):
        kw = METHOD[key]
        c, lo_e, hi_e = key_fn(rows, key)
        xerr = np.vstack([lo_e, hi_e])
        ax.errorbar(c, y + dy, xerr=xerr, fmt=kw["marker"], color=kw["color"],
                    ms=ms, capsize=2.5, lw=plt.rcParams["lines.linewidth"] * 0.7,
                    zorder=2, label=kw["label"])
    ax.set_yticks(y)
    if show_labels:
        ax.set_yticklabels([r["label"] for r in rows])
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(xlabel, labelpad=2)
    if xlim is not None:
        ax.set_xlim(*xlim)
    # Four x-ticks, not matplotlib's default ~6-7: at 2.20in total height
    # split three ways, a fifth or sixth label collides with its neighbour
    # or clips at the panel's own right edge (both observed before this
    # was added).
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.invert_yaxis()
    _panel_label(ax, panel_lbl)


def _a_rows(run_dir: str) -> tuple:
    """One pass over the six fixed coordinates, building the three
    panels' rows together so `_load_draws` runs once per coordinate."""
    rows_a, rows_b, rows_c = [], [], []
    for c in _COORDS:
        path = _product_dir(run_dir, c["dataset"], c["estimator"])
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        j = outputs.index(c["output"])
        loaded = _load_draws(path, outputs)

        # (a) V_MC: the Monte-Carlo variance of theta_hat over the draws in
        # the truth product (spec: replaces the old oracle-based ratio, so
        # this panel needs no analytic influence and exists for every
        # coordinate). nanvar excludes box-rule/failed draws by itself.
        theta_hat_j = loaded["theta_hat"][:, j]
        v_mc = float(np.nanvar(theta_hat_j, ddof=1))
        v_boot = np.nanvar(loaded["theta_boot"][:, :, j], axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_qij = np.log(loaded["v_tot_hat"][:, j] / v_mc)
            log_boot = np.log(v_boot / v_mc)
        rows_a.append(dict(label=c["label"], qij=_mean_se(log_qij), boot=_mean_se(log_boot)))

        # (b) coverage at 0.95
        cov_qij, cov_boot = _coverage_pair(loaded, j, _LEVEL)
        rows_b.append(dict(label=c["label"], qij=cov_qij, boot=cov_boot))

        # (c) width ratio QIJ/bootstrap at 0.95, median with 5-95% whiskers
        ratio = _width_ratio(loaded, j, _LEVEL)
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size:
            rows_c.append(dict(label=c["label"], median=float(np.median(ratio)),
                                p05=float(np.percentile(ratio, 5)),
                                p95=float(np.percentile(ratio, 95))))
        else:
            rows_c.append(dict(label=c["label"], median=float("nan"),
                                p05=float("nan"), p95=float("nan")))
    return rows_a, rows_b, rows_c


def fig_a(run_dir: str, lncs: bool = False) -> plt.Figure:
    """Figure A -- accuracy (spec section "Figure A"). Three panels side by
    side, the six fixed coordinates as rows in each, reading down in the
    spec table's order: MVT nu, MVT P_tail, FP a, FP scatter, IMF slope,
    IMF p.

    (a) log(V_hat_tot/V_MC) [QIJ] and log(V_boot/V_MC) [bootstrap], V_MC
    the Monte-Carlo variance of theta_hat over draws (NOT the oracle
    variance -- the 19 September spec's own change from the module this
    replaces); mean over draws with 95% whiskers, +/-0.05 materiality band.
    Caption note for the IMF p row (not rendered here, this is prose for
    the tex): V_MC there is inflated by interior fits far out on the
    ridge, so that row's coverage and width (panels b, c) are the
    informative numbers, not panel (a).
    (b) coverage of both intervals at 0.95, MC-SE whiskers, nominal line.
    (c) width ratio QIJ/bootstrap at 0.95, median with 5-95% whiskers,
    +/-10% band, unity line.

    Row labels are drawn once, on panel (a), and omitted from (b)/(c) --
    repeating six labels three times each does not fit in 2.20in of
    height and adds nothing panel (a) did not already say.
    """
    _use_style(lncs)
    figsize = FIGSIZE_A_LNCS if lncs else FIGSIZE_A_DRAFT
    rows_a, rows_b, rows_c = _a_rows(run_dir)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)

    def key_a(rows, method):
        c = np.array([r[method]["mean"] for r in rows])
        se = np.array([r[method]["se"] for r in rows])
        e = np.where(np.isfinite(se), 1.96 * se, 0.0)
        return c, e, e

    _row_panel(axes[0], rows_a, key_a, r"$\log(\hat V_\mathrm{tot}/V_\mathrm{MC})$",
               ref_line=0.0, band=(-0.05, 0.05), xlim=None, panel_lbl="(a)", show_labels=True)

    def key_b(rows, method):
        c = np.array([r[method]["p"] for r in rows])
        se = np.array([r[method]["se"] for r in rows])
        e = np.where(np.isfinite(se), 1.96 * se, 0.0)
        return c, e, e

    _row_panel(axes[1], rows_b, key_b, "Coverage",
               ref_line=_LEVEL, band=None, xlim=(0.45, 1.02), panel_lbl="(b)", show_labels=False)

    def key_c(rows, method):
        # width ratio has only one series (QIJ/bootstrap is already a
        # comparison), plotted at the "qij" dodge slot; the "boot" slot is
        # empty so `_row_panel`'s shared two-marker loop still works.
        if method == "boot":
            nan = np.full(len(rows), np.nan)
            return nan, nan, nan
        c = np.array([r["median"] for r in rows])
        lo = c - np.array([r["p05"] for r in rows])
        hi = np.array([r["p95"] for r in rows]) - c
        return c, lo, hi

    _row_panel(axes[2], rows_c, key_c, "Width ratio",
               ref_line=1.0, band=(0.90, 1.10), xlim=None, panel_lbl="(c)", show_labels=False)
    # Panel (c) plots one series (the ratio already compares the two
    # methods); its "boot" errorbar call above drew nothing (all-NaN), so
    # nothing further is needed here -- the figure-level legend below
    # still correctly labels panels (a)/(b), which do have both series.

    if lncs:
        fig.subplots_adjust(left=0.20, right=0.965, top=0.88, bottom=0.32, wspace=0.55)
    else:
        fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.24, wspace=0.45)

    handles = [Line2D([], [], color=METHOD[k]["color"], ls="none",
                       marker=METHOD[k]["marker"], label=METHOD[k]["label"])
               for k in ("qij", "boot")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=plt.rcParams["legend.fontsize"])
    return fig


# ---------------------------------------------------------------------------
# Figure B -- the refined influence against the truth (replaces F3, F9)
# ---------------------------------------------------------------------------

def _b_axis(protos: pd.DataFrame, axis_kind: str) -> np.ndarray:
    """The panel's data axis, one value per prototype, already the
    receptive-field mean (module docstring: `w_j` IS that mean, by
    construction, for every dataset this package has). `radius` uses every
    `w_*` column (MVT is 10-dimensional); `logsigma` and `logmass` use
    `w_0` alone, the spec's "data coordinate 0"."""
    if axis_kind == "radius":
        w_cols = sorted((c for c in protos.columns if c.startswith("w_")),
                         key=lambda c: int(c.split("_")[1]))
        W = protos[w_cols].to_numpy(dtype=float)
        return np.sqrt(np.sum(W ** 2, axis=1))
    if axis_kind == "logsigma":
        return protos["w_0"].to_numpy(dtype=float)
    if axis_kind == "logmass":
        return np.log10(protos["w_0"].to_numpy(dtype=float))
    raise ValueError(axis_kind)


def _b_series(pts: pd.DataFrame, protos: pd.DataFrame, output: str) -> tuple:
    """The two y-series, one point per prototype: the true influence and
    the refined estimate, each averaged over that prototype's receptive
    field (`qij_points.parquet` grouped by `bmu`), reindexed to
    `qij_prototypes.parquet`'s own prototype order `j` so a panel's x
    (from `_b_axis`) and y arrays line up index-for-index."""
    g = pts.groupby("bmu")[[f"psi_{output}", f"psi_hat_{output}"]].mean()
    g = g.reindex(protos["j"].to_numpy())
    return (g[f"psi_{output}"].to_numpy(dtype=float),
            g[f"psi_hat_{output}"].to_numpy(dtype=float))


def _plot_b_panel(ax, x, y_true, y_hat, mass, rho, title, xlabel, panel_lbl) -> None:
    finite = np.isfinite(x) & np.isfinite(y_true) & np.isfinite(y_hat)
    x, y_true, y_hat, mass = x[finite], y_true[finite], y_hat[finite], mass[finite]
    ms2 = plt.rcParams["lines.markersize"] ** 2
    size = ms2 * (0.6 + 8.0 * mass) if mass.size else ms2  # RF mass, optional per spec

    ax.scatter(x, y_true, s=size, facecolors="none", edgecolors=METHOD["truth"]["color"],
               linewidths=0.8, zorder=2, label="True $\\psi$ (RF mean)")
    ax.scatter(x, y_hat, s=size, facecolors=METHOD["qij"]["color"], edgecolors="none",
               alpha=0.85, zorder=3, label=r"$\hat\psi$ (RF mean)")
    ax.set_title(title)
    ax.set_xlabel(xlabel, labelpad=2)
    # No per-panel "Influence" ylabel -- fig_b sets it once with
    # `fig.supylabel`; six repeats of the same word cost horizontal room
    # this grid does not have (2x3 panels at ~1.1in wide each).
    _panel_label(ax, panel_lbl, dy=1.34)

    fs = _STATE["annotation_fontsize"]
    if y_true.size >= 2 and np.std(y_true) > 0 and np.std(y_hat) > 0:
        rs = float(spearmanr(y_true, y_hat).correlation)
        rho_text = f"{rho:.3f}" if np.isfinite(rho) else "n/a"
        text = f"$r_s={rs:.3f}$\n$\\rho={rho_text}$"
    else:
        text = "$r_s$ undefined"
    ax.text(0.04, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=fs)


def fig_b(run_dir: str, lncs: bool = False) -> plt.Figure:
    """Figure B -- the refined influence against the truth (spec section
    "Figure B", replaces F3 and F9). 2x3 panels at the spec's fixed grid
    positions. Each panel: one point per prototype of the designated
    draw's X-VQ; x the receptive-field mean of the panel's data axis
    (`_b_axis`, from `qij_prototypes.parquet`); y two series, the true
    influence (hollow marker) and the refined estimate psi_hat (filled
    marker), both receptive-field means (`_b_series`, from
    `qij_points.parquet`); marker area proportional to receptive-field
    mass; corner text the Spearman rank correlation between the two
    series and rho (`qij.parquet`'s `rho_<output>` for the designated
    draw). No interval is drawn here, so this figure takes no `interval`
    parameter.

    Backward compatibility: a run written before `qij_points.parquet`
    carried `psi_hat_<output>`/`bmu`/`bin_label_<output>` raises a single
    clear sentence naming the run and the missing column, not a pandas
    KeyError three lines into `_b_series`.
    """
    _use_style(lncs)
    figsize = FIGSIZE_B_LNCS if lncs else FIGSIZE_B_DRAFT
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=False)

    for c in _COORDS:
        path = _product_dir(run_dir, c["dataset"], c["estimator"])
        o = c["output"]
        pts = pd.read_parquet(os.path.join(path, "qij_points.parquet"))
        required = ["bmu", f"psi_hat_{o}", f"psi_{o}"]
        missing = [col for col in required if col not in pts.columns]
        if missing:
            raise ValueError(
                f"qij_points.parquet at {path} is missing column(s) {missing} -- "
                f"this run of {c['dataset']}/{c['estimator']} predates QIJ's "
                f"refined-influence points product (psi_hat_<output>, bmu, "
                f"bin_label_<output>); rerun the study to regenerate it before "
                f"rendering Figure B."
            )
        protos = pd.read_parquet(os.path.join(path, "qij_prototypes.parquet"))
        x = _b_axis(protos, c["axis"])
        y_true, y_hat = _b_series(pts, protos, o)
        mass = protos["p"].to_numpy(dtype=float)

        # rho (the FD-to-prediction scale, glossary) for the designated
        # draw alone -- qij_points.parquet's own `s` column names which
        # draw that is, so no separate "designated draw" constant is
        # needed here.
        rho_col = f"rho_{o}"
        rho = float("nan")
        qij_full = pd.read_parquet(os.path.join(path, "qij.parquet"))
        if rho_col in qij_full.columns:
            designated_s = int(pts["s"].iloc[0])
            match = qij_full.loc[qij_full["s"] == designated_s, rho_col]
            if len(match):
                rho = float(match.iloc[0])

        ax = axes[c["row"], c["col"]]
        _plot_b_panel(ax, x, y_true, y_hat, mass, rho, c["label"], _AXIS_LABEL[c["axis"]],
                      f"({chr(ord('a') + c['row'] * 3 + c['col'])})")

    if lncs:
        fig.subplots_adjust(left=0.11, right=0.99, top=0.84, bottom=0.22,
                             hspace=1.65, wspace=0.35)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.14,
                             hspace=0.75, wspace=0.30)

    fig.supylabel("Influence", fontsize=plt.rcParams["axes.labelsize"], fontweight="bold")

    handles = [
        Line2D([], [], marker="o", mfc="none", mec=METHOD["truth"]["color"], ls="none",
               label="True $\\psi$ (RF mean)"),
        Line2D([], [], marker="o", mfc=METHOD["qij"]["color"], mec="none", ls="none",
               label=r"$\hat\psi$ (RF mean)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=plt.rcParams["legend.fontsize"])
    return fig


# ---------------------------------------------------------------------------
# Figure C -- cost (replaces F7)
# ---------------------------------------------------------------------------

_B_GRID_FRAC = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]


def _c_bootstrap_curve(theta_c: np.ndarray, b_grid_frac: list, level: float) -> tuple:
    """theta_c: (S, B) one output's replicates. Returns (curve rows, w_ref
    (S,) the full-B reference width per draw). `curve` rows now carry
    p25/p75 as well as the mean (spec: "mean over draws with the
    interquartile band" -- the module this replaces plotted the mean
    alone). Loop over draws and the b grid only, never over N."""
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
        errs = np.array(errs)
        curve.append(dict(b=b, mean=float(np.mean(errs)) if errs.size else float("nan"),
                           p25=float(np.percentile(errs, 25)) if errs.size else float("nan"),
                           p75=float(np.percentile(errs, 75)) if errs.size else float("nan")))
    return curve, w_ref


def _c_qij_point(loaded: dict, j: int, w_ref: np.ndarray, level: float) -> dict:
    lo_hi = _qij_lo_hi(loaded, j, level)
    n = loaded["theta_hat"].shape[0]
    xs, errs = [], []
    for i in range(n):
        ref = w_ref[i]
        if not (np.isfinite(ref) and ref > 0):
            continue
        lo, hi = lo_hi[i]
        nrows = loaded["normalized_rows"][i]
        if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(nrows)):
            continue
        xs.append(float(nrows))
        errs.append(abs((hi - lo) - ref) / ref)
    if not xs:
        nan = float("nan")
        return dict(x=nan, y=nan, p25=nan, p75=nan)
    xs_a, errs_a = np.array(xs), np.array(errs)
    return dict(x=float(np.median(xs_a)), y=float(np.median(errs_a)),
                p25=float(np.percentile(errs_a, 25)), p75=float(np.percentile(errs_a, 75)))


def _plot_c_panel(ax, curve: list, pt: dict, wall_times: dict, title: str, panel_lbl: str) -> None:
    lw = plt.rcParams["lines.linewidth"]
    ms = plt.rcParams["lines.markersize"]
    b = np.array([r["b"] for r in curve], dtype=float)
    mean = np.array([r["mean"] for r in curve])
    p25 = np.array([r["p25"] for r in curve])
    p75 = np.array([r["p75"] for r in curve])
    finite = np.isfinite(mean)
    ax.fill_between(b[finite], p25[finite], p75[finite], color=METHOD["boot"]["color"],
                     alpha=0.18, linewidth=0, zorder=1)
    ax.plot(b[finite], mean[finite], color=METHOD["boot"]["color"], ls=METHOD["boot"]["ls"],
            lw=lw * 0.7, zorder=2)

    if np.isfinite(pt["x"]) and np.isfinite(pt["y"]):
        color = METHOD["qij"]["color"]
        if np.isfinite(pt["p25"]) and np.isfinite(pt["p75"]):
            ax.vlines(pt["x"], pt["p25"], pt["p75"], color=color, lw=lw * 0.8, zorder=3)
        ax.plot(pt["x"], pt["y"], marker=METHOD["qij"]["marker"], color=color, ms=ms * 1.1,
                ls="none", zorder=4)

    # Two lines, not one: at panel widths of ~1.2in, "QIJ x s / boot y s"
    # on one line is wider than the panel and spills into the y-tick
    # labels on the left; stacked, each line is about half as wide.
    fs = _STATE["annotation_fontsize"]
    ax.text(0.96, 0.97, f"QIJ {wall_times['qij']:.2g}s\nboot {wall_times['boot']:.2g}s",
            transform=ax.transAxes, ha="right", va="top", fontsize=fs, color="#555555",
            linespacing=1.15)

    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    # No per-panel x/y label -- every one of the six panels shares the
    # same two axis quantities (unlike Figure B, where the x quantity
    # differs by row), so `fig_c` sets them once with `fig.supxlabel`/
    # `fig.supylabel` rather than repeating identical text six times in
    # a grid with no room to spare.
    _panel_label(ax, panel_lbl, dy=1.34)


def fig_c(run_dir: str, timing_dir: str, lncs: bool = False) -> plt.Figure:
    """Figure C -- cost (spec section "Figure C", replaces F7). 2x3 panels
    at the spec's fixed grid positions. Each panel: the bootstrap's own
    relative width error of its 0.95 interval after the first b
    replicates, against its full-B reference width, mean over draws with
    the interquartile band (`boot.h5`'s replicates, prefix quantiles); the
    QIJ marker at the coordinate's median normalized rows with
    interquartile whiskers, its width error against the SAME converged-
    bootstrap reference (`run_dir`'s `qij.parquet`); in the corner,
    "QIJ x s / bootstrap y s", the two wall times FROM THE SEPARATE
    TIMING RUN (one worker, one thread) -- never
    `run_dir`'s own wall time, which was not captured under that
    discipline (cost layer 3, `study.py`'s module docstring). The Fundamental
    Plane's four outputs share one run and hence one wall-time pair; both
    FP panels ((1,3) FP a and (2,1) FP scatter) print the identical pair,
    which is correct, not a bug -- the tex caption should say so, since
    this figure cannot.
    """
    _use_style(lncs)
    figsize = FIGSIZE_C_LNCS if lncs else FIGSIZE_C_DRAFT
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=False)

    timing_cache = {}
    for c in _COORDS:
        path = _product_dir(run_dir, c["dataset"], c["estimator"])
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        j = outputs.index(c["output"])
        loaded = _load_draws(path, outputs)
        curve, w_ref = _c_bootstrap_curve(loaded["theta_boot"][:, :, j], _B_GRID_FRAC, _LEVEL)
        pt = _c_qij_point(loaded, j, w_ref, _LEVEL)

        key = (c["dataset"], c["estimator"])
        if key not in timing_cache:
            tpath = _product_dir(timing_dir, c["dataset"], c["estimator"])
            ttruth = pd.read_parquet(os.path.join(tpath, "truth.parquet"))
            touts = _outputs(ttruth)
            tloaded = _load_draws(tpath, touts)
            timing_cache[key] = dict(
                qij=float(np.nanmedian(tloaded["wall_time_qij"])),
                boot=float(np.nanmedian(tloaded["wall_time_boot"])),
            )

        ax = axes[c["row"], c["col"]]
        panel_lbl = f"({chr(ord('a') + c['row'] * 3 + c['col'])})"
        _plot_c_panel(ax, curve, pt, timing_cache[key], c["label"], panel_lbl)

    if lncs:
        fig.subplots_adjust(left=0.11, right=0.99, top=0.86, bottom=0.30,
                             hspace=1.55, wspace=0.32)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.20,
                             hspace=0.55, wspace=0.28)

    # Explicit y for the sup-label (rather than matplotlib's default,
    # which sits at the very bottom of the figure): the fig-level legend
    # ALSO wants that spot, so the label is pinned just under the tick
    # labels and the legend given the strip below it, stacked on
    # purpose rather than colliding.
    sup_y = 0.155 if lncs else 0.115
    fig.supxlabel("Normalized rows", y=sup_y, fontsize=plt.rcParams["axes.labelsize"], fontweight="bold")
    fig.supylabel("Rel. width error", fontsize=plt.rcParams["axes.labelsize"], fontweight="bold")

    h1 = Line2D([], [], color=METHOD["boot"]["color"], ls=METHOD["boot"]["ls"],
                lw=plt.rcParams["lines.linewidth"] * 0.7, label="Bootstrap (first $b$)")
    h2 = Line2D([], [], marker=METHOD["qij"]["marker"], color=METHOD["qij"]["color"], ls="none",
                ms=plt.rcParams["lines.markersize"] * 1.1, label="QIJ (median, IQR)")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2,
               fontsize=plt.rcParams["legend.fontsize"])
    return fig


# ---------------------------------------------------------------------------
# Figure D -- cost against N (was F10, design unchanged; adds the IMF
# sweep pair when an IMF cost-vs-N run is supplied).
# ---------------------------------------------------------------------------

def _d_series_fp(run_dirs: dict, level: float) -> dict:
    """One pass over the FP cost-vs-N sweep's N values, building every
    series Figure D's top row needs: normalized rows, the QIJ/bootstrap
    wall-time ratio, coverage of both intervals per output, and the width
    ratio -- one `_load_draws` per N, not one per series."""
    Ns = sorted(run_dirs)
    rows_stat, time_stat, width_stat, cov_rows = [], [], [], []
    outputs_ref, B_ref = None, None
    for N in Ns:
        path = run_dirs[N]
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        if outputs_ref is None:
            outputs_ref = outputs
        loaded = _load_draws(path, outputs)
        if B_ref is None:
            B_ref = loaded["B"]

        rows_stat.append(dict(N=N, **_stat(loaded["normalized_rows"])))
        with np.errstate(divide="ignore", invalid="ignore"):
            tratio = loaded["wall_time_qij"] / loaded["wall_time_boot"]
        time_stat.append(dict(N=N, **_stat(tratio)))

        wratios = []
        for j, o in enumerate(outputs):
            cov_qij, cov_boot = _coverage_pair(loaded, j, level)
            cov_rows.append(dict(N=N, output=o, p_qij=cov_qij["p"], se_qij=cov_qij["se"],
                                  p_boot=cov_boot["p"], se_boot=cov_boot["se"]))
            wratios.append(_width_ratio(loaded, j, level))
        width_stat.append(dict(N=N, **_stat(np.concatenate(wratios))))
    return dict(rows_stat=rows_stat, time_stat=time_stat, width_stat=width_stat,
                cov_rows=cov_rows, outputs=outputs_ref, B_ref=B_ref)


def _d_series_imf(imf_run_dirs: dict, level: float,
                   coverage_outputs=("slope", "Mstar", "p")) -> dict:
    """The IMF sweep's series (spec: "coverage ... (slope, M*, p)" --
    gamma_shape/gamma_scale are excluded here exactly as Pareto and FP
    b/c are excluded from the fixed six, per the spec's own naming) plus
    the two failure fractions the sweet-spot panel needs: the share of
    bootstrap replicates that hit the IMF's box constraint (NaN) at each
    N, and QIJ's own evaluation failure fraction at the same N -- "QIJ's
    zero" in the spec is a claim about the data, so it is computed here,
    not hardcoded."""
    Ns = sorted(imf_run_dirs)
    time_stat, cov_rows, boot_fail, qij_fail = [], [], [], []
    for N in Ns:
        path = imf_run_dirs[N]
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        loaded = _load_draws(path, outputs)

        with np.errstate(divide="ignore", invalid="ignore"):
            tratio = loaded["wall_time_qij"] / loaded["wall_time_boot"]
        time_stat.append(dict(N=N, **_stat(tratio)))

        for o in coverage_outputs:
            if o not in outputs:
                continue
            j = outputs.index(o)
            cov_qij, cov_boot = _coverage_pair(loaded, j, level)
            cov_rows.append(dict(N=N, output=o, p_qij=cov_qij["p"], se_qij=cov_qij["se"],
                                  p_boot=cov_boot["p"], se_boot=cov_boot["se"]))

        n_draws = len(loaded["s"])
        boot_total = loaded["B"] * n_draws
        boot_fail.append(dict(N=N, frac=float(np.sum(loaded["n_failed_boot"])) / boot_total
                               if boot_total else float("nan")))
        evals_total = float(np.sum(loaded["evals_total"]))
        qij_fail.append(dict(N=N, frac=float(np.sum(loaded["n_failed_qij"])) / evals_total
                              if evals_total > 0 else float("nan")))
    return dict(time_stat=time_stat, cov_rows=cov_rows, boot_fail=boot_fail, qij_fail=qij_fail,
                coverage_outputs=[o for o in coverage_outputs])


def _raise_primary_axis(ax, ax2) -> None:
    """`ax.twinx()` stacks the new axes `ax2` ABOVE `ax` by default, so
    anything `ax2` draws -- including a line whose only job is to sit
    under a legend -- paints over a legend attached to `ax`. Every panel
    below puts its combined legend on `ax`, so this is called right after
    each `twinx()` to swap the stacking order back, once, rather than
    fighting z-order per-artist."""
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)


def _plot_d_cost(ax, series: dict, panel_lbl: str) -> None:
    """(a) normalized rows vs N (QIJ median/IQR, bootstrap's fixed-B
    reference) on the left axis; the QIJ/bootstrap wall-time ratio
    (median/IQR) on a twin right axis in a neutral third colour -- it is a
    ratio, not a second method, so it gets neither the QIJ nor the
    bootstrap hue (style guide section 1: never let one hue mean two
    things)."""
    rows_stat, time_stat, B_ref = series["rows_stat"], series["time_stat"], series["B_ref"]
    Ns = np.array([r["N"] for r in rows_stat], dtype=float)
    med = np.array([r["median"] for r in rows_stat])
    p25 = np.array([r["p25"] for r in rows_stat])
    p75 = np.array([r["p75"] for r in rows_stat])
    ax.fill_between(Ns, p25, p75, color=METHOD["qij"]["color"], alpha=0.18, linewidth=0)
    ax.plot(Ns, med, color=METHOD["qij"]["color"], ls=METHOD["qij"]["ls"],
            marker=METHOD["qij"]["marker"], label="Normalized rows (QIJ)")
    if B_ref is not None:
        ax.axhline(B_ref, color=METHOD["boot"]["color"], ls=METHOD["boot"]["ls"], lw=1.2,
                   label="Bootstrap (fixed $B$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("Normalized rows")

    ax2 = ax.twinx()
    tN = np.array([r["N"] for r in time_stat], dtype=float)
    tmed = np.array([r["median"] for r in time_stat])
    tp25 = np.array([r["p25"] for r in time_stat])
    tp75 = np.array([r["p75"] for r in time_stat])
    ax2.fill_between(tN, tp25, tp75, color=SECONDARY, alpha=0.12, linewidth=0)
    ax2.plot(tN, tmed, color=SECONDARY, ls="-.", marker="D", ms=3.2,
             label="Wall-time ratio (QIJ/boot)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Wall-time ratio", color=SECONDARY, labelpad=9)
    ax2.tick_params(axis="y", colors=SECONDARY)
    _raise_primary_axis(ax, ax2)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend(ax, handles=h1 + h2, loc="lower right", fontsize=plt.rcParams["legend.fontsize"] * 0.7,
            handlelength=1.4, labelspacing=0.3, borderpad=0.4)
    ax.set_title("Cost vs $N$")
    _panel_label(ax, panel_lbl, dx=-0.22)


def _plot_d_coverage(ax, cov_rows: list, outputs: list, level: float, panel_lbl: str,
                      width_stat: list = None) -> None:
    """Coverage of both intervals vs N, one thin line per (output,
    method) with a per-output MC-error band -- pooling across outputs
    would be tighter than the FP's shared run and bins have earned (the
    same ruling the module this replaces made for old F10(c)). When
    `width_stat` is given (the FP row only; the spec names width ratio
    for panel (b), not for the IMF coverage panel (c)), the median width
    ratio is added on a twin axis."""
    Ns_all = sorted(set(r["N"] for r in cov_rows))
    lw = plt.rcParams["lines.linewidth"]
    for i, o in enumerate(outputs):
        color = OUTPUT_COLORS[i % len(OUTPUT_COLORS)]
        for method, ls, marker, kp, ks in (
            ("qij", "-", "o", "p_qij", "se_qij"), ("boot", "--", "s", "p_boot", "se_boot"),
        ):
            xs, ys, ses = [], [], []
            for N in Ns_all:
                match = [r for r in cov_rows if r["N"] == N and r["output"] == o]
                if not match:
                    continue
                xs.append(N)
                ys.append(match[0][kp])
                ses.append(match[0][ks])
            xs_a, ys_a, ses_a = np.array(xs, dtype=float), np.array(ys, dtype=float), np.array(ses, dtype=float)
            band = np.where(np.isfinite(ses_a), 1.96 * ses_a, 0.0)
            ax.fill_between(xs_a, ys_a - band, ys_a + band, color=color, alpha=0.10, linewidth=0)
            ax.plot(xs_a, ys_a, color=color, ls=ls, marker=marker, ms=3, lw=lw * 0.8)
    ax.axhline(level, color=METHOD["truth"]["color"], ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("$N$")
    ax.set_ylabel(f"Coverage at {level:.0%}")

    if width_stat is not None:
        ax2 = ax.twinx()
        wN = np.array([r["N"] for r in width_stat], dtype=float)
        wmed = np.array([r["median"] for r in width_stat])
        ax2.plot(wN, wmed, color=SECONDARY, ls="-.", marker="D", ms=3.2,
                 label="Width ratio (QIJ/boot)")
        ax2.axhline(1.0, color=SECONDARY, ls=":", lw=0.8)
        ax2.set_ylabel("Width ratio", color=SECONDARY, labelpad=9)
        ax2.tick_params(axis="y", colors=SECONDARY)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
        _raise_primary_axis(ax, ax2)

    # Compact legend: one swatch per output (colour identifies it) plus
    # the method linestyle/marker convention, stated once.
    out_handles = [Line2D([], [], color=OUTPUT_COLORS[i % len(OUTPUT_COLORS)], ls="-",
                          label=o) for i, o in enumerate(outputs)]
    method_handles = [Line2D([], [], color="#666666", ls=METHOD[k]["ls"],
                              marker=METHOD[k]["marker"], ms=3, label=METHOD[k]["label"])
                       for k in ("qij", "boot")]
    _legend(ax, handles=out_handles + method_handles, loc="lower left",
            ncol=2, fontsize=plt.rcParams["legend.fontsize"] * 0.7)
    ax.set_title("Reliability vs $N$")
    _panel_label(ax, panel_lbl, dx=-0.22)


def _plot_d_sweet_spot(ax, series_imf: dict, panel_lbl: str) -> None:
    """The IMF sweet-spot panel: wall-time ratio vs N on the left axis
    (where QIJ's cost advantage over the bootstrap grows), and, on a twin
    right axis, the fraction of bootstrap replicates that hit the box
    constraint at that N against QIJ's own (near-zero, per the package
    plan's rare-support finding, but measured here rather than assumed)."""
    time_stat = series_imf["time_stat"]
    Ns = np.array([r["N"] for r in time_stat], dtype=float)
    med = np.array([r["median"] for r in time_stat])
    p25 = np.array([r["p25"] for r in time_stat])
    p75 = np.array([r["p75"] for r in time_stat])
    ax.fill_between(Ns, p25, p75, color=METHOD["qij"]["color"], alpha=0.15, linewidth=0)
    ax.plot(Ns, med, color=METHOD["qij"]["color"], ls="-", marker="o", ms=3.5,
            label="Wall-time ratio (QIJ/boot)")
    ax.axhline(1.0, color=METHOD["truth"]["color"], ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("Wall-time ratio")

    ax2 = ax.twinx()
    bN = [r["N"] for r in series_imf["boot_fail"]]
    bfrac = [r["frac"] for r in series_imf["boot_fail"]]
    qN = [r["N"] for r in series_imf["qij_fail"]]
    qfrac = [r["frac"] for r in series_imf["qij_fail"]]
    ax2.plot(bN, [100.0 * f for f in bfrac], color=METHOD["boot"]["color"], ls="--", marker="s",
             ms=3.2, label="Bootstrap at box constraint")
    ax2.plot(qN, [100.0 * f for f in qfrac], color=METHOD["qij"]["color"], ls=":", marker="o",
             ms=2.8, label="QIJ at box constraint")
    ax2.set_ylim(bottom=0)
    # Percent, not a bare fraction: "0.16%" is half the character width of
    # "0.0016" at this panel's tiny right-margin allowance, and the
    # earlier fraction-formatted ticks clipped against the figure edge
    # even with generous labelpad.
    ax2.set_ylabel("At constraint (%)", color=METHOD["boot"]["color"], labelpad=7)
    ax2.tick_params(axis="y", colors=METHOD["boot"]["color"])
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))
    _raise_primary_axis(ax, ax2)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend(ax, handles=h1 + h2, loc="upper left", fontsize=plt.rcParams["legend.fontsize"] * 0.8)
    ax.set_title("IMF sweet spot")
    _panel_label(ax, panel_lbl, dx=-0.22)


def fig_d(run_dirs: dict, imf_run_dirs: dict = None,
          lncs: bool = False, level: float = _LEVEL) -> plt.Figure:
    """Figure D -- cost against sample size (spec section "Figure D",
    design unchanged from F10). `run_dirs` maps N -> the Fundamental
    Plane cost-vs-N product directory for that N (as F10 took it: no
    product records N, so the caller supplies the mapping,
    `scripts/make_figures.py`'s `_cost_dirs` builds it by globbing).

    (a) normalized rows and the QIJ/bootstrap wall-time ratio against N.
    (b) coverage of both intervals against N, one thin line per FP output
    with per-output MC bands, and the width ratio.

    `imf_run_dirs`, optional, is the same kind of mapping for an IMF
    cost-vs-N sweep; when given, a second row is added: (c) coverage of
    both intervals for the IMF's slope/M*/p against N, and (d) the
    wall-time ratio against N together with the fraction of bootstrap
    replicates hitting the box constraint at each N and QIJ's own (the
    "sweet-spot" panel). Layout and figsize both key off whether
    `imf_run_dirs` is given (1x2 at (4.80, 2.20) without it, 2x2 at
    (4.80, 3.60) with it, style guide section 8, 19 September) -- there is
    no partial state where the IMF row exists without its own figsize.
    """
    _use_style(lncs)
    has_imf = bool(imf_run_dirs)
    if has_imf:
        figsize = FIGSIZE_D2_LNCS if lncs else FIGSIZE_D2_DRAFT
        nrows = 2
    else:
        figsize = FIGSIZE_D1_LNCS if lncs else FIGSIZE_D1_DRAFT
        nrows = 1
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, squeeze=False, constrained_layout=False)

    series_fp = _d_series_fp(run_dirs, level)
    _plot_d_cost(axes[0, 0], series_fp, "(a)")
    _plot_d_coverage(axes[0, 1], series_fp["cov_rows"], series_fp["outputs"], level, "(b)",
                      width_stat=series_fp["width_stat"])

    if has_imf:
        series_imf = _d_series_imf(imf_run_dirs, level)
        _plot_d_coverage(axes[1, 0], series_imf["cov_rows"], series_imf["coverage_outputs"],
                          level, "(c)", width_stat=None)
        _plot_d_sweet_spot(axes[1, 1], series_imf, "(d)")

    if lncs:
        if has_imf:
            fig.subplots_adjust(left=0.09, right=0.80, top=0.90, bottom=0.10, hspace=0.85, wspace=1.05)
        else:
            fig.subplots_adjust(left=0.10, right=0.80, top=0.86, bottom=0.22, wspace=1.00)
    else:
        fig.subplots_adjust(left=0.07, right=0.92, top=0.93, bottom=0.10, hspace=0.5, wspace=0.5)

    return fig
