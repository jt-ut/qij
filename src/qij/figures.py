"""
The paper's figures (`QIJ_figure_spec_final.md`, 19 September 2026): Figure
A (accuracy, replaces F2), Figure B (refined influence against the truth,
replaces F3 and F9), Figure C (precision per evaluation, redesigned 19
September, replaces F7), Figure D (cost against sample size, was F10,
design unchanged, now with an optional IMF sweep pair). Every figure here
is a pure function of the products `qij.study` writes under
`<run_dir>/<dataset>/<estimator>/` -- `truth.parquet`, `qij.parquet`,
`boot.h5`, `qij_points.parquet`, `qij_prototypes.parquet`. Nothing else is
read: no estimator is re-run, no dataset is redrawn, no interval is read
(none is stored -- every interval here is recomputed from
`core.intervals.qij_interval` or `core.intervals.percentile_interval`).
Figure C no longer needs a separate timing run: the 19 September redesign
measures both methods against the truth in the same evaluation-cost unit,
not against wall time (Figure D's subject).
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
from scipy.stats import pearsonr
from scipy.stats.mstats import mjci

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
    "boot": dict(color=OI["vermillion"], ls="--", marker="s", label="Boot"),
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
    "axes.titlesize": 9, "axes.labelsize": 8.5, "figure.titlesize": 9,
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
# Figure A is taller than the style guide's original 2.20in. That height was
# set for three panels with one-line x-labels and no title; with six rows, a
# title and two-line labels it left about 1.2in of panel for six rows, so each
# row was barely twice the height of its own label and the type dominated the
# plotting area. The fonts are already at LNCS body size and must not shrink
# below it, so the height is the free variable. Still half the 7.60in text
# height, so the figure and its caption sit on one page.
FIGSIZE_A_LNCS = (4.80, 3.20)
FIGSIZE_A_DRAFT = (12.0, 5.0)
FIGSIZE_B_LNCS = (4.80, 4.20)
FIGSIZE_B_DRAFT = (12.0, 7.0)
FIGSIZE_C_LNCS = (4.80, 4.20)
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


def _use_style() -> None:
    # ONE size, the paper's. A figure is built at the inches it is placed
    # at, so \includegraphics takes it 1:1 and its type matches the body
    # text. A larger "draft" render scaled down by LaTeX would put its
    # labels at a fraction of their intended size, differently for every
    # figure, which is the mismatch the style guide exists to prevent.
    plt.rcParams.update(RC_LNCS)
    _STATE["annotation_fontsize"] = 7


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
         label=r"MVT $\nu$", param=r"$\nu$", group="MVT", axis="radius", row=0, col=0),
    dict(dataset="mvt", estimator="tail", output="P_tail",
         label=r"MVT $S_{99}$", param=r"$S_{99}$", group="MVT", axis="radius", row=1, col=0),
    dict(dataset="fp", estimator="fp", output="a",
         label=r"FP $a$", param=r"$a$", group="FP", axis="logsigma", row=0, col=1),
    dict(dataset="fp", estimator="fp", output="scatter",
         label=r"FP $s$", param=r"$s$", group="FP", axis="logsigma", row=1, col=1),
    dict(dataset="imf", estimator="imf", output="slope",
         label=r"IMF $\alpha$", param=r"$\alpha$", group="IMF", axis="logmass", row=0, col=2),
    dict(dataset="imf", estimator="imf", output="p",
         label=r"IMF $p$", param=r"$p$", group="IMF", axis="logmass", row=1, col=2),
]

_AXIS_LABEL = {
    "radius": r"$\mathbf{\|x\|}$",
    "logsigma": r"$\mathbf{\log\sigma}$",
    "logmass": r"$\mathbf{\log_{10} M_\odot}$",
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


def _comparison_mask(lo_hi_qij: np.ndarray, lo_hi_boot: np.ndarray) -> np.ndarray:
    """(n,) boolean: True for a draw where BOTH methods' intervals are
    finite -- the comparison-set ruling (figure spec, "Comparison set";
    plan §36.11, 19 September 2026). A draw whose full-data fit failed
    has no QIJ interval and is dropped from the bootstrap's side too,
    even though the bootstrap may still form and score one from its own
    converged replicates on that same draw (an IMF box-rule draw, e.g.
    s = 936, is exactly this case); a draw where QIJ's stage-1 collapsed,
    or where the bootstrap has no converged replicate at all, is dropped
    the same way. This is the conjunction of the SAME finiteness test
    `_coverage_se` already applies to each method's own interval, not a
    second convention -- every site here that compares the two methods
    (`_coverage_pair`, `_width_ratio`, Figure C's curve/point/floor)
    calls this once rather than re-deriving the test."""
    qij_ok = np.isfinite(lo_hi_qij[:, 0]) & np.isfinite(lo_hi_qij[:, 1])
    boot_ok = np.isfinite(lo_hi_boot[:, 0]) & np.isfinite(lo_hi_boot[:, 1])
    return qij_ok & boot_ok


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
    panels make. Comparison-set ruling (figure spec, "Comparison set"):
    a draw is scored for EITHER method only where BOTH methods' `level`
    intervals are finite (`_comparison_mask`), so the two coverages are
    never pooled from differently sized sets of draws -- the defect the
    ruling fixes (the IMF's QIJ coverage stood on 969 draws, its
    bootstrap coverage on up to 1000, because the bootstrap still forms
    and scores an interval on a draw whose full-data fit failed)."""
    lo_hi_qij = _qij_lo_hi(loaded, j, level)
    lo_hi_boot = _boot_lo_hi(loaded, j, level)
    common = _comparison_mask(lo_hi_qij, lo_hi_boot)
    theta_true = np.where(common, loaded["theta_true"][:, j], np.nan)
    return _coverage_se(theta_true, lo_hi_qij), _coverage_se(theta_true, lo_hi_boot)


def _width_ratio(loaded: dict, j: int, level: float) -> np.ndarray:
    """(n,) QIJ/bootstrap interval width at `level`, per draw, NaN where
    either interval failed. `_comparison_mask` is applied explicitly here
    too, even though the ratio's own NaN propagation (a NaN numerator or
    denominator already NaNs the quotient) would exclude the same draws
    on its own: one rule, called the same way at every comparison site,
    rather than two conventions that merely happen to agree."""
    lo_hi_qij = _qij_lo_hi(loaded, j, level)
    lo_hi_boot = _boot_lo_hi(loaded, j, level)
    common = _comparison_mask(lo_hi_qij, lo_hi_boot)
    w_qij = lo_hi_qij[:, 1] - lo_hi_qij[:, 0]
    w_boot = lo_hi_boot[:, 1] - lo_hi_boot[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = w_qij / w_boot
    bad = ~common | ~np.isfinite(w_boot) | (w_boot == 0)
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

def _data_span(lo, hi, include=(), pad: float = 0.06):
    """A plotted range derived from the data, never written into the code.

    `lo`/`hi` are the extremes actually drawn -- a marker's whisker ends,
    a band's edges -- and `include` names anything that must stay in view
    however the data fall: a reference line at the nominal level, a unity
    line, the edges of a tolerance band. Those are part of the figure's
    claim, so a range that cropped one would be wrong, but they are not
    allowed to SET the range on their own either.

    Returns None when nothing finite was passed, which leaves matplotlib's
    own autoscale in place rather than inventing a range.

    A hardcoded range goes stale the moment the numbers change and is
    silently wrong in between: it was `(0.45, 1.02)` here for coverage
    that lived in 0.88-0.96, so two thirds of every coverage panel was
    empty, and it would have stayed that way through any rerun.
    """
    vals = [v for v in list(np.ravel(lo)) + list(np.ravel(hi)) + list(include)
            if v is not None and np.isfinite(v)]
    if not vals:
        return None
    lo_v, hi_v = float(min(vals)), float(max(vals))
    span = hi_v - lo_v
    if span <= 0.0:
        # A single value, or all of them equal: pad by something with the
        # right scale rather than by zero, which would give a degenerate axis.
        step = abs(hi_v) * pad if hi_v else pad
        return lo_v - step, hi_v + step
    return lo_v - pad * span, hi_v + pad * span


def _row_panel(ax, rows: list, key_fn, xlabel: str, ref_line, band,
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
    # The range comes from what is drawn -- both series' whisker ends --
    # widened only far enough to keep the reference line and the tolerance
    # band in view (`_data_span`).
    lo_all, hi_all = [], []
    for key in ("qij", "boot"):
        c, lo_e, hi_e = key_fn(rows, key)
        lo_all.append(np.asarray(c) - np.asarray(lo_e))
        hi_all.append(np.asarray(c) + np.asarray(hi_e))
    keep = [ref_line] + (list(band) if band is not None else [])
    span = _data_span(np.concatenate(lo_all), np.concatenate(hi_all), include=keep)
    if span is not None:
        ax.set_xlim(*span)

    ax.set_yticks(y)
    if show_labels:
        # The parameter symbol alone, flush left so six labels of different
        # lengths share an edge instead of raggedly right-aligning; the
        # dataset is written once per pair of rows, outside them. Repeating
        # "MVT"/"FP"/"IMF" on every row cost more left margin than the
        # symbols themselves and said nothing the grouping does not.
        ax.set_yticklabels([r["label"] for r in rows])
    else:
        ax.set_yticklabels([])
    # The panel's tag lives in its x-label (author's rule, 19 September):
    # "(tag) <what the axis is>". Nothing is drawn above the axes, which is
    # where the old tag sat, detached from its panel by the full height of
    # the title gap. Both of Figure A's panels carry one -- unlike Figure B,
    # where a column shares one x quantity and only the bottom row is
    # labelled.
    ax.set_xlabel(xlabel, labelpad=2, fontweight="bold")
    # Few x-ticks, and never one hard against either end of the axis.
    # The three panels sit almost edge to edge (see `fig_a`'s wspace), so a
    # tick label at a panel's right edge would run into its neighbour's
    # left-edge label; `prune="both"` drops exactly those two. What is left
    # still spans the range, and the data, not the axis furniture, gets the
    # width.
    ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    # (4) Bold row labels. They are the figure's only y-axis text and they
    # name the coordinates, so they carry as much as the panel titles do.
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.invert_yaxis()


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
        rows_a.append(dict(label=c["label"], param=c["param"], group=c["group"],
                            qij=_mean_se(log_qij), boot=_mean_se(log_boot)))

        # (b) coverage at 0.95 -- `_coverage_pair` masks to the
        # comparison-set ruling's common set (both methods' intervals
        # finite) before scoring either, so the two coverages plotted in
        # panel (a) below are always over the same draws.
        cov_qij, cov_boot = _coverage_pair(loaded, j, _LEVEL)
        rows_b.append(dict(label=c["label"], param=c["param"], group=c["group"],
                            qij=cov_qij, boot=cov_boot))

        # (c) width ratio QIJ/bootstrap at 0.95, median with 5-95%
        # whiskers -- `_width_ratio` applies the same common-set mask.
        ratio = _width_ratio(loaded, j, _LEVEL)
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size:
            rows_c.append(dict(label=c["label"], param=c["param"], group=c["group"],
                               median=float(np.median(ratio)),
                                p05=float(np.percentile(ratio, 5)),
                                p95=float(np.percentile(ratio, 95))))
        else:
            rows_c.append(dict(label=c["label"], param=c["param"], group=c["group"],
                               median=float("nan"),
                                p05=float("nan"), p95=float("nan")))
    return rows_a, rows_b, rows_c


_A_TITLE = "QIJ vs. Bootstrap: 95% CIs"


def fig_a(run_dir: str) -> plt.Figure:
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
    _use_style()
    figsize = FIGSIZE_A_LNCS
    # rows_a (the variance ratio) is still built -- one pass over the
    # coordinates makes all three -- but no longer plotted.
    _rows_a, rows_b, rows_c = _a_rows(run_dir)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)


    def key_b(rows, method):
        c = np.array([r[method]["p"] for r in rows])
        se = np.array([r[method]["se"] for r in rows])
        e = np.where(np.isfinite(se), 1.96 * se, 0.0)
        return c, e, e

    _row_panel(axes[0], rows_b, key_b, "(a) Empirical Coverage",
               ref_line=_LEVEL, band=None, panel_lbl="(a)", show_labels=True)

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

    _row_panel(axes[1], rows_c, key_c, "(b) Width: QIJ/Boot",
               ref_line=1.0, band=(0.90, 1.10), panel_lbl="(b)", show_labels=False)
    # Panel (c) plots one series (the ratio already compares the two
    # methods); its "boot" errorbar call above drew nothing (all-NaN), so
    # nothing further is needed here -- the figure-level legend below
    # still correctly labels panels (a)/(b), which do have both series.

    fig.suptitle(_A_TITLE, fontweight="bold")
    # The three panels are equal width (subplots does that) and fill the
    # figure: the only reserved space is the left margin for the bold row
    # labels, which appear once, on panel (a). `wspace` is a fraction of the
    # average panel width, so 0.11 leaves a hairline between panels rather
    # than the half-panel gutter the default gives.
    fig.subplots_adjust(left=0.155, right=0.985, top=0.905, bottom=0.175, wspace=0.11)

    # (3) The legend is an inset on the middle panel rather than a band under
    # the figure: that band cost about a fifth of the height and, at the LNCS
    # size, height is the scarce dimension. Panel (b) is the one with room --
    # its data sit in a narrow strip near the nominal line, so the lower-left
    # corner is empty in a way (a)'s and (c)'s are not.
    handles = [Line2D([], [], color=METHOD[k]["color"], ls="none",
                       marker=METHOD[k]["marker"], label=METHOD[k]["label"])
               for k in ("qij", "boot")]
    axes[0].legend(handles=handles, loc="lower left", frameon=True,
                   framealpha=0.9, borderpad=0.3, handletextpad=0.4,
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


# Figure B plots one point per data point -- N = 2000 per panel -- so the
# markers are small and translucent and density reads as depth. Rasterized, so
# twelve thousand vector circles do not go into the PDF.
_B_POINT_SIZE = 4.0
_B_POINT_ALPHA = 0.60

# When a panel's range is set by the bulk rather than by the extremes.
#
# Three conditions, all required. A robust outlier test alone is not enough
# and was measured to misfire both ways on these panels: on MVT S_99 the
# modified z-score flags 1992 of 4000 points, the entire tail signal included,
# because MAD collapses on a two-valued distribution; on FP a it flags 339
# points that cost the axis nothing. A test answers "is this point
# anomalous"; an axis needs "do a few points cost most of the range".
#
#   1. flagged by the MAD-based modified z-score, |0.6745 (x - med)/MAD|,
#      at a deliberately loose threshold -- these are candidates, not verdicts
#   2. FEW: no more than this fraction of the points, so a panel can never
#      clamp away a mode
#   3. WORTH IT: dropping them shrinks the plotted range by at least this
#      factor, so a panel with well-behaved tails is left alone
#
# Points outside the resulting range are clipped by the axes in the ordinary
# way. They are not redrawn on the border and not counted in the corner:
# both were tried and read as clutter on a figure whose subject is the
# agreement along the diagonal.
_B_OUTLIER_Z = 10.0
_B_OUTLIER_MAX_FRAC = 0.02
_B_RANGE_TRIGGER = 2.0


def _plot_b_panel(ax, psi, psi_hat, title, panel_lbl) -> None:
    """One pairwise panel: the true influence against the method's estimate,
    one point per DATA POINT of the designated draw (spec, Figure B, pairwise
    form). Agreement is the diagonal; a wrong scale is a tilt; a bin whose
    points were pooled shows as a vertical stack, several true values sharing
    one psi_hat.

    Per point rather than per receptive field because within-field error is
    exactly what has to be visible, and on its own scale with equal x and y so
    the diagonal is at 45 degrees and a tilt reads as one.
    """
    finite = np.isfinite(psi) & np.isfinite(psi_hat)
    psi, psi_hat = psi[finite], psi_hat[finite]

    pooled = np.concatenate([psi, psi_hat])
    lo, hi = float(pooled.min()), float(pooled.max())
    med = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - med)))
    if mad > 0:
        flagged = 0.6745 * np.abs(pooled - med) / mad > _B_OUTLIER_Z
        kept = pooled[~flagged]
        if (flagged.mean() <= _B_OUTLIER_MAX_FRAC and kept.size
                and (hi - lo) > _B_RANGE_TRIGGER * (kept.max() - kept.min())):
            lo, hi = float(kept.min()), float(kept.max())
    span = _data_span(np.array([lo]), np.array([hi]), pad=0.05)
    if span is not None:
        ax.set_xlim(*span)
        ax.set_ylim(*span)
        ax.plot(span, span, color=METHOD["truth"]["color"], ls=":", lw=1.0, zorder=1)

    ax.scatter(psi_hat, psi, s=_B_POINT_SIZE, facecolors=METHOD["qij"]["color"],
               edgecolors="none", alpha=_B_POINT_ALPHA, zorder=2, rasterized=True)
    ax.set_title(f"{panel_lbl} {title}")
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    fs = _STATE["annotation_fontsize"] - 1.0
    if psi.size >= 2 and np.std(psi) > 0 and np.std(psi_hat) > 0:
        r = float(pearsonr(psi_hat, psi)[0])
        text = f"$r={r:.3f}$"
    else:
        text = "$r$ undefined"
    ax.text(0.04, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=fs,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0))


def fig_b(run_dir: str) -> plt.Figure:
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
    _use_style()
    figsize = FIGSIZE_B_LNCS
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
        ax = axes[c["row"], c["col"]]
        _plot_b_panel(ax, pts[f"psi_{o}"].to_numpy(dtype=float),
                      pts[f"psi_hat_{o}"].to_numpy(dtype=float),
                      c["label"], f"{chr(ord('a') + c['col'] * 2 + c['row'])}.")

    # Every panel plots the same two quantities, so the axes are named once.
    fig.subplots_adjust(left=0.115, right=0.99, top=0.945, bottom=0.115,
                         hspace=0.42, wspace=0.34)
    fig.supxlabel(r"Estimated $\mathbf{\hat\psi}$", fontsize=plt.rcParams["axes.labelsize"],
                  fontweight="bold")
    fig.supylabel(r"True $\mathbf{\psi}$", fontsize=plt.rcParams["axes.labelsize"],
                  fontweight="bold")

    return fig


# ---------------------------------------------------------------------------
# Figure C -- precision per evaluation (replaces F7; redesigned by the
# author, 19 September 2026). The reference is the TRUTH, not the
# bootstrap's own converged self: `_c_w_true` is the one place that true
# width, and its Monte-Carlo uncertainty, is computed; both methods'
# curves are measured against it, in the same cost unit (a full-data
# evaluation), so the two are finally on one axis.
# ---------------------------------------------------------------------------

_B_GRID_FRAC = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]


def _c_w_true(truth_df: pd.DataFrame, output: str, level: float) -> dict:
    """The TRUE width for one estimand (spec, Figure C): the distance
    between the (1-level)/2 and 1-(1-level)/2 quantiles of `theta_hat`
    over `truth.parquet`'s own finite draws. Read directly from the truth
    product, not through `_load_draws`'s inner join onto `qij.parquet`/
    `boot.h5` -- a draw that failed QIJ or was dropped from the bootstrap
    product is still a draw of the truth, and this quantity must not
    shrink to whatever the other two products happened to keep. NaN draws
    (the IMF's box rule) drop out of `np.percentile` on the finite mask
    the same way every other statistic in this file excludes them.

    The uncertainty on `w_true` is NOT a resample -- this package does not
    resample its own reference (revision plan §35) -- but the Maritz-
    Jarrett standard error of each quantile (`scipy.stats.mstats.mjci`),
    an analytic, distribution-free estimator built from the SAME order
    statistics: it weights every `theta_hat` by how much probability mass
    a Beta(m, n-m+1) distribution -- the continuous limit of the binomial
    distribution of an order statistic's rank -- places on that point
    being the target quantile. Deterministic function of n and the target
    probability, no randomness, no second sample. The two quantiles' SEs
    are then combined in quadrature as though independent; their true
    covariance is positive (`Cov(X_p, X_q) = p(1-q) / (n f(x_p) f(x_q))`
    for p < q, both order statistics of the same draws), so treating it
    as zero mildly OVER-states the width's SE -- conservative in exactly
    the direction a "this difference is not resolvable" band should be.
    `mjci` itself returns NaN when a run has too few draws for a stable
    tail estimate (needs roughly 1/level draws per tail); that NaN
    propagates to no band being drawn, not an error, matching how every
    other under-powered statistic in this file behaves."""
    alpha = 1.0 - level
    p_lo, p_hi = alpha / 2.0, 1.0 - alpha / 2.0
    x = truth_df[f"theta_hat_{output}"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return dict(w=float("nan"), se=float("nan"))
    lo, hi = np.percentile(x, [100.0 * p_lo, 100.0 * p_hi])
    se_lo, se_hi = mjci(x, prob=[p_lo, p_hi])
    se = float(np.sqrt(se_lo ** 2 + se_hi ** 2))
    return dict(w=float(hi - lo), se=se)


def _interval_score(lo, hi, y, level: float):
    """The interval (Winkler) score of [lo, hi] against the truth y, a
    PROPER scoring rule for an interval: the width, plus a penalty of
    (2/alpha) times how far outside the interval the truth fell.

        S = (hi - lo) + (2/a)(lo - y) if y < lo
                      + (2/a)(y - hi) if y > hi

    One number for what Figure A needs two panels to say. A too-narrow
    interval pays the miss penalty; a too-wide one pays in width; and
    because the rule is proper, neither shrinking nor inflating can game
    it. Lower is better. Reported here divided by w_true, which makes it
    dimensionless and comparable across estimands.
    """
    a = 1.0 - level
    pen = 0.0
    if y < lo:
        pen = (2.0 / a) * (lo - y)
    elif y > hi:
        pen = (2.0 / a) * (y - hi)
    return (hi - lo) + pen


def _c_bootstrap_curve(theta_c: np.ndarray, w_true: float, theta_true: float,
                        b_grid_frac: list, level: float, mask: np.ndarray = None) -> list:
    """theta_c: (S, B) one output's replicates. For each prefix `b` of the
    grid, the percentile width from the first b replicates of every draw,
    relative error against the SINGLE scalar `w_true` (not each draw's
    own converged width -- the 19 September redesign's point). Rows carry
    the MEDIAN over draws and the interquartile band (spec: "the curve is
    the median over draws ... the band the interquartile range"). Loop
    over draws and the b grid only, never over N.

    `mask`: (S,) boolean, the comparison-set ruling's fixed common set
    (`_comparison_mask`, computed ONCE by `fig_c` from the FULL bootstrap
    and the QIJ interval, both at `level`) -- held fixed across every
    prefix `b` here, never recomputed from that prefix's own interval, so
    a draw excluded from the comparison cannot drift back in at a smaller
    b just because its own low-replicate interval happens to be finite
    there (CAREFUL, task note: the curve would otherwise be over a moving
    population). A draw INSIDE the common set can still drop out of one
    b's own curve point below if that prefix itself has too few converged
    replicates to form an interval (`percentile_interval` returns NaN) --
    that is the bootstrap's own low-b behaviour, a second and unrelated
    reason to skip a point, not a second comparison-set test."""
    S, B = theta_c.shape
    if mask is None:
        mask = np.ones(S, dtype=bool)
    curve = []
    for frac in b_grid_frac:
        b = max(2, int(round(frac * B)))
        errs = []
        for s in range(S):
            if not mask[s]:
                continue
            lo, hi = percentile_interval(theta_c[s, :b][:, None], level)[0]
            if np.isfinite(lo) and np.isfinite(hi):
                errs.append(_interval_score(lo, hi, theta_true, level) / w_true)
        errs = np.array(errs)
        if errs.size:
            m = float(np.mean(errs))
            se = float(np.std(errs, ddof=1) / np.sqrt(errs.size))
        else:
            m = se = float("nan")
        curve.append(dict(b=b, median=m, p25=m - se, p75=m + se))
    return curve


def _c_qij_point(loaded: dict, j: int, w_true: float, theta_true: float, level: float,
                  mask: np.ndarray = None) -> dict:
    """The QIJ marker: x the median over draws of `normalized_rows`, y the
    median over draws of the QIJ interval's relative width error against
    the same scalar `w_true` the bootstrap curve uses, interquartile
    whiskers on y.

    `mask`: the same fixed common set `_c_bootstrap_curve` uses
    (`_comparison_mask`, comparison-set ruling), so the marker is a
    median over the exact same draws as the curve and the ideal floor,
    never QIJ's own larger valid set."""
    lo_hi = _qij_lo_hi(loaded, j, level)
    n = loaded["theta_hat"].shape[0]
    if mask is None:
        mask = np.ones(n, dtype=bool)
    xs, errs = [], []
    for i in range(n):
        if not mask[i]:
            continue
        lo, hi = lo_hi[i]
        nrows = loaded["normalized_rows"][i]
        if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(nrows)):
            continue
        xs.append(float(nrows))
        errs.append(_interval_score(lo, hi, theta_true, level) / w_true)
    if not xs:
        nan = float("nan")
        return dict(x=nan, y=nan, p25=nan, p75=nan)
    xs_a, errs_a = np.array(xs), np.array(errs)
    m = float(np.mean(errs_a))
    se = float(np.std(errs_a, ddof=1) / np.sqrt(errs_a.size)) if errs_a.size > 1 else float("nan")
    return dict(x=float(np.median(xs_a)), y=m, p25=m - se, p75=m + se)


def _plot_c_panel(ax, curve: list, pt: dict, ideal: float, title: str, panel_lbl: str) -> None:
    lw = plt.rcParams["lines.linewidth"]
    ms = plt.rcParams["lines.markersize"]
    b = np.array([r["b"] for r in curve], dtype=float)
    med = np.array([r["median"] for r in curve])
    p25 = np.array([r["p25"] for r in curve])
    p75 = np.array([r["p75"] for r in curve])
    finite = np.isfinite(med)

    # Reference: a zero line (perfect agreement with the truth) and, when
    # `w_true`'s own Monte-Carlo SE is finite, a light band up to its
    # relative size -- an apparent width error smaller than that could be
    # noise in the reference itself rather than either method's error.
    ax.axhline(0.0, color=METHOD["truth"]["color"], ls=":", lw=1.0, zorder=1)
    if np.isfinite(ideal):
        # The achievable floor, not zero: a correctly calibrated interval
        # still misses its 5% and pays the score for it.
        ax.axhline(ideal, color=METHOD["truth"]["color"], ls=":", lw=1.0, zorder=1)

    ax.fill_between(b[finite], p25[finite], p75[finite], color=METHOD["boot"]["color"],
                     alpha=0.18, linewidth=0, zorder=1)
    ax.plot(b[finite], med[finite], color=METHOD["boot"]["color"], ls=METHOD["boot"]["ls"],
            lw=lw * 0.7, zorder=2)

    if np.isfinite(pt["x"]) and np.isfinite(pt["y"]):
        color = METHOD["qij"]["color"]
        if np.isfinite(pt["p25"]) and np.isfinite(pt["p75"]):
            ax.vlines(pt["x"], pt["p25"], pt["p75"], color=color, lw=lw * 0.8, zorder=3)
        ax.plot(pt["x"], pt["y"], marker=METHOD["qij"]["marker"], color=color, ms=ms * 1.1,
                ls="none", zorder=4)

    # The range comes from what is drawn -- the boot band's edges and the
    # QIJ whiskers (or its bare marker, when it has no whiskers) -- widened
    # only far enough to keep the zero line and the materiality band in
    # view (`_data_span`), never a hardcoded `bottom=0`.
    lo_parts, hi_parts = [p25[finite]], [p75[finite]]
    if np.isfinite(pt["y"]):
        pt_lo = pt["p25"] if np.isfinite(pt["p25"]) else pt["y"]
        pt_hi = pt["p75"] if np.isfinite(pt["p75"]) else pt["y"]
        lo_parts.append(np.array([pt_lo]))
        hi_parts.append(np.array([pt_hi]))
    keep = [ideal] if np.isfinite(ideal) else []
    span = _data_span(np.concatenate(lo_parts), np.concatenate(hi_parts), include=keep)
    if span is not None:
        ax.set_ylim(*span)

    ax.set_xscale("log")
    ax.set_title(f"{panel_lbl} {title}")
    # No per-panel x/y label -- every one of the six panels shares the
    # same two axis quantities (unlike Figure B, where the x quantity
    # differs by row), so `fig_c` sets them once with `fig.supxlabel`/
    # `fig.supylabel` rather than repeating identical text six times in
    # a grid with no room to spare.


def fig_c(run_dir: str) -> plt.Figure:
    """Figure C -- precision per evaluation (spec section "Figure C",
    redesigned 19 September 2026, replaces F7). 2x3 panels at the spec's
    fixed grid positions. Each panel measures BOTH methods against the
    TRUTH, not the bootstrap against its own converged self (the design
    this replaces): a single true width `w_true` per estimand (`_c_w_true`,
    the 2.5/97.5 quantile spread of `theta_hat` over `truth.parquet`'s own
    finite draws), then

      Boot: the percentile width from the first b of `boot.h5`'s
      replicates, relative error against `w_true`, median over draws with
      the interquartile band -- it plateaus at the bootstrap's own limit
      (near 0 on the clean estimands, near 0.5 on IMF p) and does not
      reach zero at b = B; nothing here draws it as though it should.

      QIJ: one marker at the coordinate's median normalized rows over
      draws, y its median relative width error against the SAME `w_true`,
      interquartile whiskers.

      Reference: a zero line plus a light band for `w_true`'s own Monte-
      Carlo uncertainty (`_c_w_true`'s Maritz-Jarrett SE, relative to
      `w_true`), so a reader can see what difference is resolvable.

    x is cost in full-data evaluations (the bootstrap's b replicates and
    QIJ's normalized rows, the same unit), log scale. No wall times are
    printed here -- that is Figure D's subject and Table T1's within-draw
    ratio -- so this figure takes no `timing_dir`.

    Comparison-set ruling (figure spec, "Comparison set"): the bootstrap
    curve, the QIJ marker and the ideal floor are all computed over the
    same set of draws, `common` below -- `_comparison_mask` applied ONCE
    per coordinate, from the FULL bootstrap (every replicate) and the
    QIJ interval, both at `_LEVEL`, never recomputed from a prefix `b`'s
    own interval (a draw must not enter the curve at one b and leave it
    at another, or the curve is over a moving population). `w_true`
    (`_c_w_true`) is deliberately NOT restricted to `common`: it is read
    from `truth.parquet`'s own finite draws by design (that function's
    own docstring), the reference both methods are measured against, and
    must not shrink to whatever the comparison set happens to keep.
    """
    _use_style()
    figsize = FIGSIZE_C_LNCS
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=False)

    for c in _COORDS:
        path = _product_dir(run_dir, c["dataset"], c["estimator"])
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        j = outputs.index(c["output"])
        ax = axes[c["row"], c["col"]]
        # Tags run DOWN each column, matching the grid's dataset pairing.
        panel_lbl = f"{chr(ord('a') + c['col'] * 2 + c['row'])}."

        w_true = _c_w_true(truth, c["output"], _LEVEL)
        if not (np.isfinite(w_true["w"]) and w_true["w"] > 0):
            # No usable true width for this estimand's draws -- an empty,
            # titled panel rather than a crash or a silent divide by zero.
            ax.set_title(f"{panel_lbl} {c['label']}")
            continue

        loaded = _load_draws(path, outputs)
        theta_true = float(np.nanmedian(loaded["theta_true"][:, j]))

        # The comparison set, fixed ONCE from each method's full result: a
        # draw on which either method has no interval is dropped for both
        # (plan §36.11). Taken from the full-B bootstrap, never from a
        # prefix's own interval, so the curve is over one population at every
        # point along it rather than a moving one.
        common = _comparison_mask(_qij_lo_hi(loaded, j, _LEVEL),
                                  _boot_lo_hi(loaded, j, _LEVEL))

        # The reference is the TRUE 95% interval itself -- [q2.5, q97.5] of
        # theta_hat over the draws -- scored against theta_true. Its width is
        # w_true by definition and theta_true lies inside it, so it never pays
        # a penalty and its normalised score is exactly 1.000 on every
        # coordinate. The line is therefore the y-axis's own unit: "paid
        # exactly the true width, never missed".
        #
        # It replaces a pivotal fixed-width interval that was drawn here as an
        # "ideal" and was not one. That interval is CALIBRATED -- it covers 95%
        # by construction -- but calibration does not minimise this score: the
        # minimum belongs to the interval that is narrow on the draws where the
        # estimator is precise and wide where it is not, and both real methods
        # adapt that way. So both beat it, on IMF p by a wide margin, and a
        # reference that the things it bounds can beat teaches the reader
        # nothing except mistrust.
        #
        # 1.000 is not a hard lower bound either -- a deliberately narrow
        # interval can score under it if its misses stay small -- and the
        # caption must not claim otherwise. What it is, is the price of being
        # exactly right, which is the thing worth measuring against.
        ideal = _interval_score(*np.percentile(
            loaded["theta_hat"][:, j][np.isfinite(loaded["theta_hat"][:, j])],
            [2.5, 97.5]), theta_true, _LEVEL) / w_true["w"]
        curve = _c_bootstrap_curve(loaded["theta_boot"][:, :, j], w_true["w"],
                                    theta_true, _B_GRID_FRAC, _LEVEL, mask=common)
        pt = _c_qij_point(loaded, j, w_true["w"], theta_true, _LEVEL, mask=common)
        _plot_c_panel(ax, curve, pt, float(ideal), c["label"], panel_lbl)

    fig.subplots_adjust(left=0.125, right=0.99, top=0.945, bottom=0.115,
                         hspace=0.42, wspace=0.34)
    fig.supxlabel("Full-data evaluations", fontsize=plt.rcParams["axes.labelsize"],
                  fontweight="bold")
    fig.supylabel("Interval score / $w_{true}$", fontsize=plt.rcParams["axes.labelsize"], fontweight="bold")

    # (3) The legend is an inset on panel (a), MVT nu, upper right -- the
    # same placement the module this replaces used, and it still holds:
    # at large b/normalized-rows the curve and the QIJ marker have both
    # settled near their low plateau on this coordinate, so the upper
    # right corner (large x, large y) is the one region no series ever
    # reaches, on every build. As in Figure A, one legend for the figure.
    h1 = Line2D([], [], color=METHOD["boot"]["color"], ls=METHOD["boot"]["ls"],
                lw=plt.rcParams["lines.linewidth"] * 0.7, label="Boot (mean $\\pm$ SE)")
    h2 = Line2D([], [], marker=METHOD["qij"]["marker"], color=METHOD["qij"]["color"], ls="none",
                ms=plt.rcParams["lines.markersize"] * 1.1, label="QIJ (mean $\\pm$ SE)")
    axes[0, 0].legend(handles=[h1, h2], loc="upper right", ncol=1,
                      frameon=True, framealpha=0.9, borderpad=0.3,
                      handletextpad=0.4, labelspacing=0.3,
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
                   label="Boot (fixed $B$)")
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
    cov_lo, cov_hi = [], []      # the band edges actually drawn, for the range
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
            cov_lo.append(ys_a - band)
            cov_hi.append(ys_a + band)
    ax.axhline(level, color=METHOD["truth"]["color"], ls=":", lw=1.0)
    ax.set_xscale("log")
    # Derived from the coverage series and their Monte-Carlo bands, with the
    # nominal level kept in view; not the old fixed (0.5, 1.02), which put
    # half the panel below anything ever plotted in it.
    cov_span = _data_span(np.concatenate(cov_lo) if cov_lo else [],
                          np.concatenate(cov_hi) if cov_hi else [], include=[level])
    if cov_span is not None:
        ax.set_ylim(*cov_span)
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
             ms=3.2, label="Boot at box constraint")
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
          level: float = _LEVEL) -> plt.Figure:
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
    _use_style()
    has_imf = bool(imf_run_dirs)
    if has_imf:
        figsize = FIGSIZE_D2_LNCS
        nrows = 2
    else:
        figsize = FIGSIZE_D1_LNCS
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

    if has_imf:
        fig.subplots_adjust(left=0.09, right=0.80, top=0.90, bottom=0.10,
                            hspace=0.85, wspace=1.05)
    else:
        fig.subplots_adjust(left=0.10, right=0.80, top=0.86, bottom=0.22, wspace=1.00)
    return fig
