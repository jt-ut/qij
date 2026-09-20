"""
The paper's figures (`QIJ_figure_spec_final.md`, 19 September 2026): Figure
A (accuracy, replaces F2), Figure B (refined influence against the truth,
replaces F3 and F9), Figure C (precision per evaluation, redesigned 19
September, replaces F7), Figure D (wall time, redesigned 20 September,
replaces F10 entirely -- a 2x2 of estimator objects, each panel carrying a
seconds axis and a QIJ/Boot ratio axis over the same markers; the old
"when QIJ pays"/cost-against-N/accuracy-against-N panels are gone, as are
the normalized-rows and bound-hit-fraction panels before them).
Every figure here is a pure function of the products `qij.study` writes
under `<run_dir>/<dataset>/<estimator>/` -- `truth.parquet`, `qij.parquet`,
`boot.h5`, `qij_points.parquet`, `qij_prototypes.parquet`. Nothing else is
read: no estimator is re-run, no dataset is redrawn, no interval is read
(none is stored -- every interval here is recomputed from
`core.intervals.qij_interval` or `core.intervals.percentile_interval`).
NO figure needs a separate timing run, and none needs a cost-against-N
sweep. Figure C's 19 September redesign measures both methods against the
truth in the same evaluation-cost unit rather than against wall time, and
Figure D's 20 September redesign reads the per-draw wall times already in
the main products (`qij.parquet`'s `wall_time_total`, `boot.h5`'s
`wall_time`), where a draw's QIJ and its bootstrap ran back to back in one
process. `<run_dir>` is every figure's only input.
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
# module). Palette from the module this replaces; the sizes below are the
# ones fixed in the style guide's section 8 on 19 September, verified
# against the revision's own llncs.cls.
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

# The one rcParams dict, at the paper's sizes: titles 9pt (both
# `axes.titlesize` and `figure.titlesize`), axis labels 8.5, ticks and
# legend 7.5, as the style guide's section 8 fixes them. Fixed point sizes
# are correct here only because a figure is built at the inches it is
# placed at and included 1:1 -- see `_use_style`.
RC = {
    "font.family": "DejaVu Sans",
    "axes.titlesize": 9, "axes.titleweight": "bold", "figure.titlesize": 9,
    "axes.labelsize": 8.5, "axes.labelweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.title_fontsize": 8,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": None,
    "savefig.transparent": False,
    # MUST be False, and this is load-bearing -- see the note below.
    #
    # Every figure here lays itself out by hand with `subplots_adjust` and
    # passes `constrained_layout=False` to `plt.subplots`, which correctly
    # leaves `fig._layout_engine` as None AT CONSTRUCTION. That opt-out is
    # not durable across saves. `Figure.savefig` -> `print_figure` wraps the
    # save in `figure._cm_set(layout_engine='none')`; on exit that context
    # manager "restores" the captured value by calling
    # `set_layout_engine(layout=None)`, and THAT method's contract for None
    # is not "no engine" but "take it from rcParams" -- so with this True it
    # installs a live ConstrainedLayoutEngine. Measured on matplotlib 3.9.4:
    # engine None at creation, ConstrainedLayoutEngine after the first
    # savefig, and on the SECOND savefig the engine executes and silently
    # discards every `subplots_adjust` (Figure D's axes went from
    # top=0.800/bottom=0.275 to top=0.936/bottom=0.072).
    #
    # That is why a figure's two outputs disagreed: whichever of the PDF and
    # PNG was written SECOND from the same figure object got the relaid-out
    # geometry. Order-dependent, not format-dependent. Figures whose titles
    # are axes children moved with the axes and merely shifted; Figure D's
    # `fig.text` titles are pinned to fractions computed once from the
    # correct positions, so they were left sitting inside the axes.
    "figure.constrained_layout.use": False,
}

# The figure sizes (style guide section 8, 19 September, verified against
# the revision's llncs.cls: text width 347.12pt = 4.80in). There is one
# size per figure, the size it is placed at.
# Figure A is taller than the style guide's original 2.20in. That height was
# set for three panels with one-line x-labels and no title; with six rows, a
# title and two-line labels it left about 1.2in of panel for six rows, so each
# row was barely twice the height of its own label and the type dominated the
# plotting area. The fonts are already at LNCS body size and must not shrink
# below it, so the height is the free variable. Still half the 7.60in text
# height, so the figure and its caption sit on one page.
FIGSIZE_A = (4.80, 3.20)
FIGSIZE_B = (4.80, 4.20)
FIGSIZE_C = (4.80, 4.20)
# 2x2, full width (redesigned 20 September, replacing the 1x3 this height
# was first set for): one panel per estimator object, each carrying a
# seconds axis along its top and the shared ratio axis along its bottom,
# so the figure needs room for two rows of tick labels between the panel
# rows and a third below the last. Grown 2.35 -> 2.45in for exactly that,
# empirically, until nothing collided -- the same way Figures B and C were
# sized. Half the 7.60in text height, so figure and caption share a page.
FIGSIZE_D = (4.80, 2.45)

# Corner-annotation size, the style guide's "annotation: 7pt at final
# size" row. It is a module constant rather than an rcParam because
# `plt.rcParams` has no real "annotation.fontsize" key -- the module this
# replaces called `plt.rcParams.get("annotation.fontsize", 9)` for its
# corner text, which silently always returned the hardcoded default
# because that key does not exist and `dict.get` does not validate it, so
# every corner annotation rendered at 9pt, ignoring the style guide.
_ANNOTATION_FONTSIZE = 7


def _use_style() -> None:
    # ONE size, the paper's. A figure is built at the inches it is placed
    # at, so \includegraphics takes it 1:1 and its type matches the body
    # text. A larger render scaled down by LaTeX would put its labels at a
    # fraction of their intended size, differently for every figure, which
    # is the mismatch the style guide exists to prevent.
    plt.rcParams.update(RC)


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
    # The IMF column is the Chabrier form (20 September 2026): a lognormal
    # below a fixed 1 Msun break joined to a power law above, free in
    # (m_c, sigma, x). Two of its three coordinates are plotted, chosen to
    # show two different data regimes rather than two views of one: `m_c`,
    # the characteristic mass, is a location fixed by the data-rich peak,
    # while `x` is the slope of the sparse high-mass tail. `sigma` is a
    # peak parameter like `m_c` and would repeat that regime.
    #
    # `x` is the index on xi(log m) ~ m^-x, Salpeter x = 1.35 -- NOT the
    # index on dN/dm, where Salpeter is 2.35. The paper must say which.
    dict(dataset="imf", estimator="chabrier", output="m_c",
         label=r"IMF $m_{\rm c}$", param=r"$m_{\rm c}$", group="IMF", axis="logmass", row=0, col=2),
    dict(dataset="imf", estimator="chabrier", output="x",
         label=r"IMF $x$", param=r"$x$", group="IMF", axis="logmass", row=1, col=2),
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
        v_btw=df[[f"V_btw_{o}" for o in outputs]].to_numpy(),
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
    at `level`, from `core.intervals.qij_interval`: the normal interval
    `theta_hat +/- z sqrt(V_btw)` on the measured between-bin variance.
    There is no acceleration term and no natural-parameter support clip
    any more -- an ablation on the main run's products found V_win_hat
    under 0.6% of the total variance, the median |a_bca| between 0.0002
    and 0.037, and the clip changing no endpoint on any draw (see
    `core.intervals.qij_interval`). A draw whose `theta_hat`/`V_btw` is
    not both finite for this output (a box-rule or QIJ-side failure,
    plan §36.2 ruling 5) is left NaN rather than passed in --
    `qij_interval` would produce NaN from it anyway, but the finiteness
    check is made explicit here rather than relied on implicitly."""
    n = loaded["theta_hat"].shape[0]
    lo_hi = np.full((n, 2), np.nan)
    for i in range(n):
        th, v = loaded["theta_hat"][i, j], loaded["v_btw"][i, j]
        if np.isfinite(th) and np.isfinite(v):
            lo_hi[i] = qij_interval(np.array([th]), np.array([v]), level)[0]
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


def _pretty_ticks(lo: float, hi: float, ref: float = None, target: int = 5) -> list:
    """Round tick values across [lo, hi], in R's `pretty()` spirit, ANCHORED
    on `ref` so the panel's reference line always lands on a tick.

    The step is the smallest of {1, 2, 2.5, 5} x 10^k giving at most `target`
    intervals across the range (5, not 4: at 4 the coverage panel's 0.042-wide
    range just missed a 0.01 step at 4.2 intervals and fell back to 0.02,
    which labelled only two of its four round values) -- the same family `pretty()` and matplotlib's
    `MaxNLocator` choose from. What neither of those guarantees is the
    anchoring: ticks are generated as `ref + k*step`, not `0 + k*step`, so a
    reference at 0.95 or 1.0 is always labelled.

    That is what was wrong before. The range came from `_data_span`, which is
    data-derived and arbitrary (coverage landed on 0.9276-0.9656), and the
    default locator divided it into a step of 0.015 -- not a round number, and
    it stepped straight over the 0.95 the panel's own reference line marks.
    Both 0.95 and 1.0 happen to be multiples of their step here, so an
    unanchored pretty locator would look right today; it would silently stop
    being right if a reference moved off a round multiple.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return []
    span = hi - lo
    anchor = float(ref) if ref is not None and np.isfinite(ref) else lo
    exp = np.floor(np.log10(span / max(target, 1)))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = mult * 10.0 ** exp
        if span / step <= target:
            break
    k0 = int(np.ceil((lo - anchor) / step))
    k1 = int(np.floor((hi - anchor) / step))
    return [anchor + k * step for k in range(k0, k1 + 1)]


def _row_panel(ax, rows: list, key_fn, xlabel: str, ref_line, band,
               panel_lbl: str, show_labels: bool, bars_from: float = None) -> None:
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
    # `bars_from` turns the panel into diverging horizontal bars measured
    # from that value instead of dodged dot-and-whisker markers. It exists
    # for the width-ratio panel, whose quantity is a RATIO: a bar encodes
    # distance from a baseline, and for a ratio the meaningful baseline is 1,
    # not 0. Drawn from zero every bar would be ~99% of full length and the
    # differences -- which live in the third decimal -- would be invisible.
    # From 1 the bar's direction reads directly: left of the line QIJ's
    # interval is narrower than the bootstrap's, right of it wider.
    if bars_from is not None:
        # Every draw is plotted, with the median marked in black on it --
        # Figure D's idiom (`_d_strip`/`_d_glyph`), reused here rather than
        # re-invented so the two figures read the same way.
        #
        # Three summary forms were tried first and all were worse. A bar from
        # 1 to the median showed no spread. The same bar with 5-95% whiskers
        # was unreadable: the whiskers ran about eight times the bar's length.
        # A box spanning the IQR with a median tick was legible but summarised
        # a 1000-draw distribution twice over, and choosing its whiskers meant
        # choosing a convention to explain -- Tukey's 1.5x IQR spans 3.8-4.0x
        # the box here and brings 3-10 outlier points per row, 5-95% spans
        # 2.3-2.5x. With this many draws the distribution can simply be shown.
        for i, r in enumerate(rows):
            vals = np.asarray(r.get("ratio", []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                _d_strip(ax, vals, float(i), METHOD["qij"]["color"], rng_seed=i)
                _d_glyph(ax, vals, float(i), "o", METHOD["qij"]["color"])
    else:
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
        if bars_from is not None:
            # The range comes from the clouds themselves, trimmed at the 1st
            # and 99th percentile so that a handful of extreme draws cannot
            # set the axis for all six rows -- the points outside still draw,
            # they just do not get a vote on the limits.
            lo_all.append(np.array([np.percentile(np.asarray(r["ratio"], dtype=float), 1)
                                     for r in rows if np.size(r.get("ratio", []))], dtype=float))
            hi_all.append(np.array([np.percentile(np.asarray(r["ratio"], dtype=float), 99)
                                     for r in rows if np.size(r.get("ratio", []))], dtype=float))
        else:
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
    # Round ticks, anchored on this panel's own reference line, and never one
    # hard against either end of the axis. `MaxNLocator(4, prune="both")` was
    # here and did the pruning but not the rounding: over `_data_span`'s
    # data-derived range it chose a step of 0.015 for coverage and stepped
    # straight over the 0.95 the panel's reference line marks. `_pretty_ticks`
    # picks a round step AND anchors it on `ref_line`, so the reference is
    # always labelled; the edge pruning that `prune="both"` gave is kept by
    # dropping any tick within 4% of either end, which is what stopped a
    # right-edge label colliding with the neighbouring panel's left-edge one.
    ticks = _pretty_ticks(*ax.get_xlim(), ref=ref_line)
    x0, x1 = ax.get_xlim()
    edge = 0.04 * (x1 - x0)
    ticks = [t for t in ticks if x0 + edge < t < x1 - edge]
    if ticks:
        ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
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
                               ratio=ratio,
                               median=float(np.median(ratio)),
                                p25=float(np.percentile(ratio, 25)),
                                p75=float(np.percentile(ratio, 75)),
                                p05=float(np.percentile(ratio, 5)),
                                p95=float(np.percentile(ratio, 95))))
        else:
            nan = float("nan")
            rows_c.append(dict(label=c["label"], param=c["param"], group=c["group"],
                               ratio=np.array([]),
                               median=nan, p25=nan, p75=nan, p05=nan, p95=nan))
    return rows_a, rows_b, rows_c


_A_TITLE = "QIJ vs. Bootstrap: 95% CIs"


def fig_a(run_dir: str) -> plt.Figure:
    """Figure A -- accuracy (spec section "Figure A"). TWO panels side by
    side, the six fixed coordinates as rows in each, reading down in the
    spec table's order: MVT nu, MVT P_tail, FP a, FP scatter, IMF slope,
    IMF p.

    (a) coverage of both intervals at 0.95, MC-SE whiskers, nominal line.
    (b) width ratio QIJ/bootstrap at 0.95, MEDIAN with 5-95% whiskers,
    unity line. NO materiality band: the +/-10% one drawn here until
    20 September was passed into `_data_span`'s `include` list, so it
    forced the axis to span at least itself and could never render as
    anything but a full-panel grey wash -- it could only have shown as a
    band if some width ratio fell OUTSIDE +/-10%, i.e. only when the
    method was doing badly. Dropping it lets the axis close onto the data,
    which is what makes the coordinates distinguishable. Its one claim,
    that every coordinate is within 10% of the bootstrap's width, belongs
    in the caption.

    The variance panel, log(V_hat_tot/V_MC) against log(V_boot/V_MC), was
    REMOVED by the author on 19 September 2026 (figure spec, "Figure A"):
    it compared each method to a third quantity where these two compare
    the methods to each other on what a reader acts on, and the only
    dramatic thing in it -- the IMF p row -- was a property of that
    denominator rather than of either method. `_a_rows` still returns its
    rows (one pass builds all three); T1 keeps the ratio for all thirteen
    coordinates, written V_hat/V_hat_MC with both terms hatted.

    Row labels are drawn once, on panel (a), and omitted from (b) --
    repeating six labels twice does not fit in the panel height and adds
    nothing panel (a) did not already say.
    """
    _use_style()
    figsize = FIGSIZE_A
    # rows_a (the variance ratio) is still built -- one pass over the
    # coordinates makes all three -- but no longer plotted.
    _rows_a, rows_b, rows_c = _a_rows(run_dir)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)


    def key_b(rows, method):
        c = np.array([r[method]["p"] for r in rows])
        se = np.array([r[method]["se"] for r in rows])
        e = np.where(np.isfinite(se), 1.96 * se, 0.0)
        return c, e, e

    _row_panel(axes[0], rows_b, key_b, "a. Empirical Coverage",
               ref_line=_LEVEL, band=None, panel_lbl="a.", show_labels=True)

    def key_c(rows, method):
        # width ratio has only one series (QIJ/bootstrap is already a
        # comparison), plotted at the "qij" dodge slot; the "boot" slot is
        # empty so `_row_panel`'s shared two-marker loop still works.
        if method == "boot":
            nan = np.full(len(rows), np.nan)
            return nan, nan, nan
        # The IQR, not the 5-95% range: this panel draws a BAR spanning it,
        # so the two returned offsets are the bar's ends measured from the
        # median that sits on it.
        c = np.array([r["median"] for r in rows])
        lo = c - np.array([r["p25"] for r in rows])
        hi = np.array([r["p75"] for r in rows]) - c
        return c, lo, hi

    # NO materiality band. It was +/-10% around 1, and `_row_panel` passes a
    # band's edges into `_data_span`'s `include` list -- so the band forced the
    # axis to span at least itself and could never render as anything but a
    # full-panel grey wash. It could only have shown as a band if some width
    # ratio fell OUTSIDE +/-10%, i.e. only when the method was doing badly.
    # Dropping it also lets the axis collapse onto the data (0.948-1.087
    # rather than 0.90-1.10), which is what makes the differences between
    # coordinates legible instead of squashed into the middle third. Its one
    # claim -- that every coordinate is within 10% of the bootstrap's width --
    # belongs in the caption, where it is a sentence rather than a grey box.
    _row_panel(axes[1], rows_c, key_c, "b. Width: QIJ/Boot",
               ref_line=1.0, band=None, panel_lbl="b.", show_labels=False,
               bars_from=1.0)
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

    # (3) The legend sits in the figure's bottom-left MARGIN, under the last
    # row label, not inside a panel. Inset on panel (a) at `lower left` it
    # covered the `IMF x` row's own markers -- the corner it was placed in was
    # empty on an earlier run's numbers and is not empty now. The margin below
    # the y-tick labels is the one region of this figure that holds nothing at
    # any numbers: `subplots_adjust`'s left (0.155) and bottom (0.175) reserve
    # it for the labels, and the labels stop above it.
    handles = [Line2D([], [], color=METHOD[k]["color"], ls="none",
                       marker=METHOD[k]["marker"], label=METHOD[k]["label"])
               for k in ("qij", "boot")]
    # y = 0.049, not 0.004: measured, that puts the legend's lower edge on the
    # panel x-labels' own baseline (0.0656 in figure coords) instead of 0.045
    # below it, so it no longer hangs past the bottom of the figure's text.
    # Its top then reaches 0.163, still clear of the panels at 0.175.
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.004, 0.049),
               frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3,
               handletextpad=0.4, handlelength=1.0,
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
# markers are small and translucent and density reads as depth. NOT
# rasterized: the points are the figure's content and stay vectors, sharp at
# any zoom and at any print resolution. Nothing in this module rasterizes.
#
# `_B_POINT_SIZE` is matplotlib's `s`, an AREA in points squared, so the
# 4.0 -> 4.84 step is 10% on the marker's DIAMETER (1.1 squared), not 10% on
# the number. Alpha raised 0.60 -> 0.72 with it: the points are a little more
# opaque, while still letting the dense stretches along the diagonal read as
# depth rather than as one solid line.
_B_POINT_SIZE = 4.84
_B_POINT_ALPHA = 0.72

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

    # Purple, not QIJ's blue. The refined influence is an INPUT to QIJ, not
    # the method's answer: every other figure's blue marks an interval QIJ
    # produced, and colouring this the same would claim these points are one
    # of those. `OI["purple"]` is the palette's remaining categorical colour
    # and is used nowhere else now that Figure D's dataset colouring is gone.
    ax.scatter(psi_hat, psi, s=_B_POINT_SIZE, facecolors=OI["purple"],
               edgecolors="none", alpha=_B_POINT_ALPHA, zorder=2)
    ax.set_title(f"{panel_lbl} {title}")
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    fs = _ANNOTATION_FONTSIZE - 1.0
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
    figsize = FIGSIZE_B
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
    the MEAN over draws and the +/-1 SE band (the figure legend's own
    label, "mean +/- SE" -- not the median/interquartile band an earlier
    draft of the spec described and this docstring used to repeat). The
    dict keys stay `median`/`p25`/`p75` for historical reasons (an
    earlier median/IQR design this function no longer implements); they
    hold the mean and mean -/+ SE respectively, and are left unrenamed
    since renaming them would touch every plotting call that reads them
    for no benefit. Loop over draws and the b grid only, never over N.

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
    MEAN over draws of the QIJ interval's relative width error against
    the same scalar `w_true` the bootstrap curve uses, with +/-1 SE
    whiskers on y (the figure legend's own "mean +/- SE" -- not the
    median/interquartile whiskers this docstring used to describe; only
    x, `normalized_rows`, is still a median here).

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
    figsize = FIGSIZE_C
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

    fig.subplots_adjust(left=0.135, right=0.99, top=0.945, bottom=0.130,
                         hspace=0.42, wspace=0.34)
    # "Estimator calls (size N)", not "Full-data evaluations". The axis is a
    # COUNT OF EVALUATIONS OF THE ESTIMATOR, each on a dataset of size N: for
    # the bootstrap it is literally b, its number of size-N resamples; for QIJ
    # it is `normalized_rows`, the total rows handed to the estimator divided
    # by N, because QIJ's prototype stage evaluates on M_X rows rather than N
    # and the two have to be in one unit to share an axis. "Full-data" named
    # that normalisation rather than the thing being counted.
    # y pinned: the legend sits below this label, so the two have to be given
    # separate bands. Measured, the default put both in 0.008-0.062 and they
    # printed on top of one another.
    fig.supxlabel("Estimator Calls (size $N$)", y=0.040,
                  fontsize=plt.rcParams["axes.labelsize"],
                  fontweight="bold")
    # Normalized by the score of the SAMPLING DISTRIBUTION'S OWN 95% interval
    # (theta_hat's 2.5/97.5 quantiles over the draws, `_c_w_true`). That
    # interval contains the truth by construction, so it takes no penalty and
    # its score is exactly its width -- the divisor is a score like the
    # numerator, and the ratio is a plain normalisation. Hence 1.0, the dotted
    # line in every panel. Lower is better, and 1 is not a hard floor: a
    # deliberately narrow interval can score under it when its misses stay
    # small, so the caption must not claim otherwise.
    fig.supylabel("Normalized Interval Score",
                  fontsize=plt.rcParams["axes.labelsize"] * 0.92, fontweight="bold")

    # TWO legends, flanking the x-label on its own line rather than a band
    # beneath it. One band cost a whole line of height for two keys that sit
    # side by side anyway, and height is the scarce dimension at this size.
    # The x-label is centred and about a third of the figure wide, so the
    # margins either side of it are free at any numbers -- unlike the panel
    # corner the legend sat in before, which was empty only on an earlier
    # run's numbers and had the bootstrap curve running through it by this one.
    h_boot = Line2D([], [], color=METHOD["boot"]["color"], ls=METHOD["boot"]["ls"],
                    lw=plt.rcParams["lines.linewidth"] * 0.6, label="Boot (mean $\\pm$ SE)")
    h_qij = Line2D([], [], marker=METHOD["qij"]["marker"], color=METHOD["qij"]["color"],
                   ls="none", ms=plt.rcParams["lines.markersize"] * 0.9,
                   label="QIJ (mean $\\pm$ SE)")
    # Smaller than the body legend size and smaller marks with it: these two
    # sit on the same line as the x-label, and at equal weight they competed
    # with it instead of reading as annotations flanking it.
    _kw = dict(frameon=False, handletextpad=0.35, borderaxespad=0.0, borderpad=0.0,
               fontsize=plt.rcParams["legend.fontsize"] * 0.88)
    fig.legend(handles=[h_boot], loc="center right", bbox_to_anchor=(0.325, 0.048),
               handlelength=1.8, **_kw)
    fig.legend(handles=[h_qij], loc="center left", bbox_to_anchor=(0.675, 0.048),
               handlelength=1.0, **_kw)
    return fig


# ---------------------------------------------------------------------------
# Figure D -- cost (replaces F10 entirely, redesigned by the author, 19
# September 2026). Three unrelated inputs, one per panel: the timing run
# (a, wall time per evaluation), the FP cost-vs-N sweep (b and the FP half
# of c), and, optionally, the IMF cost-vs-N sweep (the IMF half of c). No
# panel here reads `qij_points.parquet` or `qij_prototypes.parquet` --
# unlike Figures B and C, cost is a property of the run, not of any one
# draw's refined influence.
# ---------------------------------------------------------------------------

_D_TIMING_B = 2000   # the bootstrap's fixed replicate count in every entry
                      # of the timing run (`config.yaml`'s own `B: 2000`),
                      # so "one evaluation" on the x-axis is exact, not a fit.

# Panels (b) and (c) tell the Fundamental Plane and the IMF apart by
# colour, not by method -- neither is `METHOD["qij"]`'s blue or
# `METHOD["boot"]`'s vermillion, which are reserved for QIJ/Boot on every
# figure in the paper, this one's own panel (a) included. Green and
# purple from the same Okabe-Ito set (`OI`), chosen for distance from
# both reserved colours and from each other and from the black truth/
# reference lines every panel here also draws.
_D_DATASET_COLOR = dict(fp=OI["green"], imf=OI["purple"])

# The four points panel (a) labels -- one per ESTIMATOR OBJECT, not one per
# dataset (author, corrected): an estimator evaluation is shared across a
# dataset's outputs (the Fundamental Plane's four coordinates -- a, b, c,
# scatter -- are ONE evaluation), so the unit panel (a) prices is the
# estimator, and the four that appear are exactly the four `_COORDS` above
# maps its six coordinates onto: mvt/nu, mvt/tail, fp/fp, imf/imf. Pareto is
# dropped entirely -- it appears in no other figure in the paper (`_COORDS`
# never names it), so a Figure D that still plotted it was the one place in
# the package an estimand outside the paper's own six coordinates showed up.
# Labels match the other figures' row labels exactly (`_COORDS`'s own
# `label` strings): S_99, not P_tail or tail.
_D_POINTS = [
    dict(dataset="mvt", estimator="nu", label=r"MVT $\nu$"),
    dict(dataset="mvt", estimator="tail", label=r"MVT $S_{99}$"),
    dict(dataset="fp", estimator="fp", label="FP"),
    dict(dataset="imf", estimator="imf", label="IMF"),
]

# The one-output timing entries the "one-output" QIJ line is averaged over
# (188 prototypes each, the spec's own figure). Pareto dropped along with
# the rest of it (above): the two MVT entries are both one-output estimators
# and both are now plotted points in their own right, so this line is their
# own average, not a quantity borrowed from a dataset the panel no longer
# shows.
_D_ONE_OUTPUT_REGIME = [("mvt", "nu"), ("mvt", "tail")]


def _d_timing_load(timing_dir: str, dataset: str, estimator: str) -> dict:
    """One (dataset, estimator) directory of the timing run, through the
    same `_product_dir`/`_outputs`/`_load_draws` every other figure in
    this file uses -- panel (a) reads no product the rest of the module
    does not already know how to read."""
    path = _product_dir(timing_dir, dataset, estimator)
    truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
    outputs = _outputs(truth)
    return _load_draws(path, outputs)


def _d_timing_medians(loaded: dict) -> dict:
    """The five medians panel (a) is built from, all taken PER DRAW before
    medians are taken, not from the ratio of two separately-taken medians
    (a slow or fast draw then contributes consistently to both terms of
    `overhead`, rather than an arbitrary pairing across draws):

      t_i         = wall_time_boot_i / B, the bootstrap's own per-draw
                    wall time divided by its exactly-known B=2000
                    evaluations -- exact, not fit, because B is fixed by
                    `config.yaml`.
      overhead_i  = wall_time_qij_i - normalized_rows_i * t_i, QIJ's own
                    wall time on that same draw minus what its work would
                    cost AT THAT DRAW'S OWN t_i -- "the quantizer and the
                    influence model" (spec), whatever is left once the
                    evaluations are charged at the bootstrap's unit price.

    NORMALIZED ROWS, not `evals_total`, and the difference is not small.
    `t_i` is the price of a FULL-DATA evaluation, because the bootstrap's
    B are all full-data. QIJ's are not: its stage-1 evaluations run on the
    M_X prototypes, not on N rows. Charging every one of QIJ's
    evaluations at the full-data rate bills work it never did -- on the
    IMF, 763 evaluations touching 434 full-data-equivalents of rows, so
    329 evaluations at 0.105 s invented, about 35 s -- and the residual
    went NEGATIVE, a median of about -4.3 s. `normalized_rows` is
    precisely the full-data-equivalent count (cost layer 1, plan
    section 8) and is what the bootstrap's unit price applies to.
    Corrected, every overhead is positive and the four one-output
    estimators fall in a 0.21-0.37 s band instead of scattering:

      fp/fp         evals 542   rows 238.5   overhead 1.341 s
      imf/imf             763        433.9           30.300
      mvt/nu              242         71.3            0.349
      mvt/tail            321        149.8            0.365
      pareto/shape        230         58.8            0.285
      pareto/tail         228         56.8            0.209

    The IMF's 30.3 s is a real cost the old accounting hid behind a
    pretended evaluation price: the quantizer and the influence model at
    N = 2000 over 415 prototypes, plus -- the caption says this -- the
    optimizer's own per-row cost differing between the prototypes and the
    full data, which is the one place the single-rate model still bends.
    The panel ILLUSTRATES the crossover; it does not measure it."""
    t_i = loaded["wall_time_boot"] / _D_TIMING_B
    overhead_i = loaded["wall_time_qij"] - loaded["normalized_rows"] * t_i
    return dict(
        t=float(np.median(t_i)),
        wt_boot=float(np.median(loaded["wall_time_boot"])),
        wt_qij=float(np.median(loaded["wall_time_qij"])),
        n_evals=float(np.median(loaded["normalized_rows"])),
        overhead=float(np.median(overhead_i)),
    )


def _d_regime(timing_dir: str, members: list) -> dict:
    """One (overhead, n_evals) pair for a QIJ cost LINE, averaged over
    `members` -- a list of (dataset, estimator) pairs, each reduced to its
    own median overhead and evaluation count by `_d_timing_medians` first,
    then averaged. The FP regime has one member; the one-output regime
    averages `_D_ONE_OUTPUT_REGIME`'s four, so that line does not depend
    on which single one-output estimator panel (a) happens to plot."""
    meds = [_d_timing_medians(_d_timing_load(timing_dir, ds, est)) for ds, est in members]
    return dict(overhead=float(np.mean([m["overhead"] for m in meds])),
                n_evals=float(np.mean([m["n_evals"] for m in meds])))


def _d_crossover(overhead: float, n_evals: float) -> float:
    """The per-evaluation cost t* at which the bootstrap's B*t equals this
    QIJ line's overhead + n_evals*t: B*t* = overhead + n_evals*t*, so
    t* = overhead / (B - n_evals). Both regimes measured here have
    n_evals well below B (255 and 542 against B=2000), so the denominator
    is always positive; a hypothetical regime with n_evals >= B (QIJ
    needing as many evaluations as the bootstrap's B) would have no
    crossover at all, which this does not special-case because nothing in
    this run reaches it."""
    return overhead / (_D_TIMING_B - n_evals)


def _log_span(values, include=(), pad: float = 0.15):
    """`_data_span`'s counterpart for a log axis -- the same rule (a range
    derived from what is actually drawn, padded only far enough to keep
    `include` in view, never hardcoded), applied in log10 space so the
    padding is a fraction of the log-range rather than of the linear range
    a log axis is about to compress. `None` when nothing positive and
    finite was passed, same convention as `_data_span`."""
    vals = np.array([v for v in values if v is not None and np.isfinite(v) and v > 0], dtype=float)
    if vals.size == 0:
        return None
    inc = [np.log10(v) for v in include if v is not None and np.isfinite(v) and v > 0]
    span = _data_span(np.log10(vals), np.log10(vals), include=inc, pad=pad)
    if span is None:
        return None
    return 10.0 ** span[0], 10.0 ** span[1]


def _plot_d_pay(ax, timing_dir: str, panel_lbl: str) -> None:
    """(a) When QIJ pays (spec, Figure D, corrected by the author 19
    September 2026 to plot the four ESTIMATOR OBJECTS the paper's six
    coordinates map to -- `_D_POINTS` -- rather than one estimator per
    dataset, which had put Pareto, an estimator no other figure ever
    shows, on this one). x the per-evaluation cost t (the bootstrap's own
    wall time / B, `_d_timing_medians`), y wall time per run, both log.
    The bootstrap's line, B*t; QIJ's line, overhead + n_evals*t -- but TWO
    of these, not one, at the two overhead regimes the spec itself gives
    ("about 0.25 s for a one-output estimator at 188 prototypes, about
    1.3 s for the Fundamental Plane at 371 prototypes and four outputs"),
    because they differ by about 5x and averaging them into a single line
    would misplace every one-output estimator's crossover by roughly that
    factor. Each line's own crossover (`_d_crossover`) is marked with a
    black x where it falls inside the plotted range. The two theoretical
    lines are tagged "1-output" and "FP model" in the panel itself -- "FP
    model" rather than the bare "FP" the spec used, because the FP POINT
    (one of the four estimators, `_D_POINTS`) is labelled "FP" too, and
    the two are different things: one is this run's median wall time for
    the FP estimator, the other is the overhead+rows*t line fitted to
    "an estimator that quantizes to 371 prototypes and reports four
    outputs" in general. The bare "FP" ran the line's tag straight through
    the point's own marker in the first render.

    Where the four points fall against the crossovers, this run's timing
    medians (`_d_timing_medians`, boot/QIJ wall-time ratio in parens,
    >1 favours QIJ):
      MVT S_99   t=1.81e-4 s  -- essentially AT the one-output crossover
                 (t*=1.89e-4 s), the two nearly coincide on the panel;
                 ratio 0.92, bootstrap fractionally cheaper.
      FP         t=2.07e-4 s  -- well below the FP line's own crossover
                 (t*=7.62e-4 s); ratio 0.30, QIJ costs 3.3x the bootstrap.
      MVT nu     t=5.27e-4 s  -- above the one-output crossover; ratio
                 2.72, QIJ cheaper.
      IMF        t=1.05e-1 s  -- far to the right of both crossovers;
                 ratio 2.77, QIJ cheaper (its own overhead, ~30s, is
                 mostly the optimizer's per-row cost on the prototypes,
                 not the influence model -- the docstring on
                 `_d_timing_medians` has the breakdown).
    MVT S_99 sitting almost exactly on the one-output crossover, at nearly
    the same (t, wall time) as the crossover's own x mark, is why that
    label carries a leader line into open data-space rather than a bare
    pixel offset from its own marker -- see `_LABEL_OFFSET`.
    """
    regime_one = _d_regime(timing_dir, _D_ONE_OUTPUT_REGIME)
    regime_fp = _d_regime(timing_dir, [("fp", "fp")])
    points = []
    for spec in _D_POINTS:
        loaded = _d_timing_load(timing_dir, spec["dataset"], spec["estimator"])
        points.append(dict(label=spec["label"], **_d_timing_medians(loaded)))

    x_span = _log_span([p["t"] for p in points], pad=0.18)
    x_lo, x_hi = x_span
    x_grid = np.geomspace(x_lo, x_hi, 200)

    ax.plot(x_grid, _D_TIMING_B * x_grid, color=METHOD["boot"]["color"],
            ls=METHOD["boot"]["ls"], lw=plt.rcParams["lines.linewidth"], zorder=2)

    fs = _ANNOTATION_FONTSIZE
    # The theoretical lines' own tags sit at 62% of the log range, not at
    # the right edge the module this replaces used: with the four points
    # now mvt/nu, mvt/tail, fp/fp, imf/imf (below), x_hi is set by the
    # IMF point (the largest t by two orders of magnitude), and a
    # right-edge tag landed within a few points of IMF's own QIJ marker
    # and its label. This far along the log range is clear of every
    # point and every point label at this run's numbers -- verified by
    # rendering, not assumed (55% put "FP model" close enough to the MVT
    # nu label's own corner to read as one run-on phrase).
    x_txt_frac = 0.62
    for regime, tag in ((regime_one, "1-output"), (regime_fp, "FP model")):
        y_grid = regime["overhead"] + regime["n_evals"] * x_grid
        ax.plot(x_grid, y_grid, color=METHOD["qij"]["color"], ls=METHOD["qij"]["ls"],
                lw=plt.rcParams["lines.linewidth"], zorder=2)
        x_txt = x_lo * (x_hi / x_lo) ** x_txt_frac
        y_txt = regime["overhead"] + regime["n_evals"] * x_txt
        ax.text(x_txt, y_txt, f" {tag}", color=METHOD["qij"]["color"], fontsize=fs,
                ha="left", va="bottom", clip_on=False)
        t_star = _d_crossover(regime["overhead"], regime["n_evals"])
        if x_lo <= t_star <= x_hi:
            ax.plot(t_star, _D_TIMING_B * t_star, marker="x", color=METHOD["truth"]["color"],
                    ms=5.5, mew=1.3, ls="none", zorder=4)

    # Per-point label offsets, tuned against this run's actual numbers
    # (`_plot_d_pay`'s own docstring). MVT S_99's marker pair (t=1.81e-4,
    # wall time 0.36-0.39s) sits almost exactly ON the one-output
    # crossover mark (t*=1.89e-4, 0.38s), and FP's own marker (t=2.07e-4)
    # is barely further along in t -- three features inside 15% of each
    # other in t, and MVT nu's own crossover (the FP line's, t*=7.6e-4)
    # falls close to MVT nu's own marker in turn. That crowding is a real
    # feature of the data (S_99 and, to a lesser extent, nu ARE close to
    # a crossover) and the markers stay put; only the LABELS move, in
    # DATA coordinates (`textcoords="data"`, not a pixel offset from each
    # point's own marker) so their target can be chosen by what is
    # actually open at this run's numbers: MVT S_99 into the "channel"
    # between the two QIJ lines' plateaus (`one(t)` below, `fp(t)` above,
    # evaluated by hand while tuning this), MVT nu ABOVE the FP line
    # entirely, clear of its own crossover mark and of the "FP model"/
    # "1-output" tags further out at `x_txt_frac`. FP and IMF keep simple
    # point-relative offsets -- their own neighbourhoods are clear on
    # this run.
    _LABEL_OFFSET = {
        r"MVT $S_{99}$": dict(xytext=(2.0e-3, 1.0), textcoords="data",
                              ha="left", va="center",
                              arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6,
                                               shrinkA=0, shrinkB=3)),
        "FP": dict(xytext=(0, 8), textcoords="offset points", ha="center", va="bottom"),
        r"MVT $\nu$": dict(xytext=(5.5e-4, 3.2), textcoords="data",
                          ha="left", va="bottom"),
        "IMF": dict(xytext=(-5, 6), textcoords="offset points", ha="right", va="bottom"),
    }
    for p in points:
        ax.plot(p["t"], p["wt_boot"], marker=METHOD["boot"]["marker"],
                color=METHOD["boot"]["color"], ls="none", zorder=3)
        ax.plot(p["t"], p["wt_qij"], marker=METHOD["qij"]["marker"],
                color=METHOD["qij"]["color"], ls="none", zorder=3)
        y_top = max(p["wt_boot"], p["wt_qij"])
        off = _LABEL_OFFSET[p["label"]]
        ax.annotate(p["label"], (p["t"], y_top), fontsize=fs, clip_on=False, **off)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    y_span = _log_span([p["wt_boot"] for p in points] + [p["wt_qij"] for p in points], pad=0.22)
    if y_span is not None:
        ax.set_ylim(*y_span)
    ax.set_xlabel(r"Time per evaluation $\mathbf{t}$ (s)", labelpad=2)
    ax.set_ylabel("Wall time per run (s)")
    ax.set_title(f"{panel_lbl} When QIJ pays")


def _d_ratio_n(run_dirs: dict) -> dict:
    """Median (with interquartile band) over draws of the WITHIN-DRAW
    QIJ/bootstrap wall-time ratio, against N -- panel (b)'s y, corrected
    by the author 19 September 2026 from absolute wall time (cost layer
    2, `tables.py`'s module docstring: "wall time only as the ratio
    QIJ/bootstrap within the same draw on the same worker, which cancels
    the machine and its load"). The cost sweeps that fill `run_dirs` ran
    45-100 workers to a node, so a draw's own wall time is contended by
    whatever else that worker was running and is not comparable to
    another draw's, let alone another run's -- an absolute second from
    this data is the thing this file's OWN cost discipline forbids
    plotting; only the timing run (one worker, one thread) earns panel
    (a)'s absolute seconds. What a contended run CAN support is the ratio
    `wall_time_qij[i] / wall_time_boot[i]` taken on the same draw i, same
    worker, same contention at that instant, which cancels in the
    quotient the way it does not cancel across draws. `_load_draws`
    already pairs the two per draw (both read from the same estimator
    directory's own products), so no join beyond that is needed here."""
    Ns = sorted(run_dirs)
    med, p25, p75 = [], [], []
    for N in Ns:
        path = run_dirs[N]
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        loaded = _load_draws(path, outputs)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = loaded["wall_time_qij"] / loaded["wall_time_boot"]
        s = _stat(ratio)
        med.append(s["median"]); p25.append(s["p25"]); p75.append(s["p75"])
    return dict(Ns=np.array(Ns, dtype=float), med=np.array(med),
                p25=np.array(p25), p75=np.array(p75))


def _plot_d_ratio_n(ax, fp_series: dict, imf_series, panel_lbl: str) -> None:
    """(b) Cost against N, corrected by the author 19 September 2026 to
    plot the within-draw QIJ/bootstrap wall-time RATIO rather than
    absolute wall time -- `_d_ratio_n`'s own docstring has the cost-
    discipline reason. One line per DATASET, not per method: the ratio
    already IS the QIJ-vs-bootstrap comparison, so there is no second
    method series left to draw here, and colour is free for the
    Fundamental Plane (from `run_dirs`, the FP cost-vs-N sweep) and the
    IMF (from `imf_run_dirs`, when the sweep is on the build), in
    `_D_DATASET_COLOR`'s green/purple rather than the reserved method
    colours -- named directly at each line's own right end, the same
    idiom panel (c) uses, so this panel needs no legend of its own.

    Why both lines belong here together (author, correcting the version
    that plotted only the Fundamental Plane): that was the one dataset
    where QIJ loses, so the panel by itself read as "QIJ is uniformly
    more expensive". It is not -- the IMF sweep runs QIJ at roughly a
    third of the bootstrap's wall time at every N it covers, the
    Fundamental Plane the other way around, and the horizontal line at
    1.0 is what makes both readable as a single "who pays, and when"
    picture rather than two disconnected facts."""
    fs = _ANNOTATION_FONTSIZE
    lo_all, hi_all = [], []
    for key, series, label in (("fp", fp_series, "FP"), ("imf", imf_series, "IMF")):
        if series is None:
            continue
        color = _D_DATASET_COLOR[key]
        Ns = series["Ns"]
        ax.fill_between(Ns, series["p25"], series["p75"], color=color, alpha=0.15, linewidth=0)
        ax.plot(Ns, series["med"], color=color, ls="-", marker="o")
        ax.text(Ns[-1], series["med"][-1], f" {label}", color=color, fontsize=fs,
                ha="left", va="center")
        lo_all.append(series["p25"]); hi_all.append(series["p75"])
    ax.axhline(1.0, color=METHOD["truth"]["color"], ls=":", lw=1.0, zorder=1)
    ax.set_xscale("log")
    # 1.0 is the reference the panel exists to show, so it is always kept
    # in view (`include=[1.0]`) even on a build where every N sits well
    # above or below it -- the same rule `_row_panel`'s unity/nominal
    # lines follow in Figures A and C.
    if lo_all:
        span = _data_span(np.concatenate(lo_all), np.concatenate(hi_all), include=[1.0])
        if span is not None:
            ax.set_ylim(*span)
    ax.set_xlabel(r"$\mathbf{N}$", labelpad=2)
    ax.set_ylabel("QIJ / Boot wall time")
    ax.set_title(f"{panel_lbl} Cost vs $N$")


def _d_fp_coverage_n(run_dirs: dict, level: float) -> dict:
    """Coverage at `level` against N for both methods, the Fundamental
    Plane's four outputs reduced to their mean with the min-to-max range
    across them (spec: "the mean over its four outputs with a band
    spanning the four") -- not a per-output MC-error band, which is a
    different quantity the old F10(c) plotted and this design drops."""
    Ns = sorted(run_dirs)
    qij_mean, qij_lo, qij_hi = [], [], []
    boot_mean, boot_lo, boot_hi = [], [], []
    for N in Ns:
        path = run_dirs[N]
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        loaded = _load_draws(path, outputs)
        cq, cb = [], []
        for j in range(len(outputs)):
            pq, pb = _coverage_pair(loaded, j, level)
            cq.append(pq["p"]); cb.append(pb["p"])
        cq, cb = np.array(cq), np.array(cb)
        qij_mean.append(np.nanmean(cq)); qij_lo.append(np.nanmin(cq)); qij_hi.append(np.nanmax(cq))
        boot_mean.append(np.nanmean(cb)); boot_lo.append(np.nanmin(cb)); boot_hi.append(np.nanmax(cb))
    return dict(
        Ns=np.array(Ns, dtype=float),
        qij=dict(mean=np.array(qij_mean), lo=np.array(qij_lo), hi=np.array(qij_hi)),
        boot=dict(mean=np.array(boot_mean), lo=np.array(boot_lo), hi=np.array(boot_hi)),
    )


def _d_imf_mstar_coverage_n(imf_run_dirs: dict, level: float, output: str = "Mstar") -> dict:
    """Coverage at `level` against N for both methods, IMF's M* alone
    (spec: "IMF M* as its own line ... the finite-sample effect vanishing
    as N grows"), with its ordinary Monte-Carlo SE (`_coverage_pair`), the
    same quantity Figure A's coverage panel uses."""
    Ns = sorted(imf_run_dirs)
    qij_p, qij_se, boot_p, boot_se = [], [], [], []
    for N in Ns:
        path = imf_run_dirs[N]
        truth = pd.read_parquet(os.path.join(path, "truth.parquet"))
        outputs = _outputs(truth)
        if output not in outputs:
            raise ValueError(f"{output!r} not among outputs {outputs} at N={N} under {path}")
        j = outputs.index(output)
        loaded = _load_draws(path, outputs)
        pq, pb = _coverage_pair(loaded, j, level)
        qij_p.append(pq["p"]); qij_se.append(pq["se"])
        boot_p.append(pb["p"]); boot_se.append(pb["se"])
    return dict(
        Ns=np.array(Ns, dtype=float),
        qij=dict(p=np.array(qij_p), se=np.array(qij_se)),
        boot=dict(p=np.array(boot_p), se=np.array(boot_se)),
    )


def _plot_d_accuracy(ax, fp_series: dict, imf_series, level: float, panel_lbl: str) -> None:
    """(c) Accuracy against N (spec, Figure D). Coverage at `level` for
    both methods, FP (mean over its four outputs with a min-max band
    across them) and, when supplied, IMF M* (its own ordinary Monte-Carlo
    band, `_coverage_pair`'s SE).

    Colour recoloured by the author 19 September 2026 to mark the
    DATASET, not the method (`_D_DATASET_COLOR`: FP green, IMF purple).
    Before, both series were `METHOD["qij"]["color"]`/`["boot"]["color"]`
    -- the same blue and vermillion every other panel uses -- so FP and
    IMF were the same two colours as each other, told apart only by an
    end label and a marker fill, and their coverage bands overlap through
    most of the N range each sweep covers. Method now rides on line style
    and marker shape alone (`METHOD["qij"]["ls"]`/`["marker"]`, still
    solid circle for QIJ and dashed square for Boot, just without their
    reserved colour in this one panel) -- the figure's single legend
    (`fig_d`, drawn on panel (a) where the reserved colours ARE used)
    is what teaches that shape/style pairing; here it is applied, not
    re-explained. Each dataset is still named directly at its own right
    end, the idiom style guide section 6/F4 uses for `d_eff`, rather than
    a second legend."""
    fs = _ANNOTATION_FONTSIZE
    Ns_fp = fp_series["Ns"]
    fp_color = _D_DATASET_COLOR["fp"]
    lo_all, hi_all = [], []
    for key in ("boot", "qij"):
        kw, s = METHOD[key], fp_series[key]
        ax.fill_between(Ns_fp, s["lo"], s["hi"], color=fp_color, alpha=0.15, linewidth=0)
        ax.plot(Ns_fp, s["mean"], color=fp_color, ls=kw["ls"], marker=kw["marker"])
        lo_all.append(s["lo"]); hi_all.append(s["hi"])
    ax.text(Ns_fp[-1], fp_series["qij"]["mean"][-1], " FP", color=fp_color,
            fontsize=fs, ha="left", va="center")

    if imf_series is not None:
        Ns_imf = imf_series["Ns"]
        imf_color = _D_DATASET_COLOR["imf"]
        for key in ("boot", "qij"):
            kw, s = METHOD[key], imf_series[key]
            band = np.where(np.isfinite(s["se"]), 1.96 * s["se"], 0.0)
            ax.fill_between(Ns_imf, s["p"] - band, s["p"] + band, color=imf_color,
                            alpha=0.12, linewidth=0)
            ax.plot(Ns_imf, s["p"], color=imf_color, ls=kw["ls"], marker=kw["marker"])
            lo_all.append(s["p"] - band); hi_all.append(s["p"] + band)
        ax.text(Ns_imf[-1], imf_series["qij"]["p"][-1], " IMF $M_*$", color=imf_color,
                fontsize=fs, ha="left", va="center")

    ax.axhline(level, color=METHOD["truth"]["color"], ls=":", lw=1.0)
    ax.set_xscale("log")
    span = _data_span(np.concatenate(lo_all), np.concatenate(hi_all), include=[level])
    if span is not None:
        ax.set_ylim(*span)
    ax.set_xlabel(r"$\mathbf{N}$", labelpad=2)
    ax.set_ylabel(f"Coverage at {level:.0%}")
    ax.set_title(f"{panel_lbl} Accuracy vs $N$")



# ---------------------------------------------------------------------------
# Figure D -- wall time per draw, and the QIJ/bootstrap ratio
# ---------------------------------------------------------------------------

# One row per ESTIMATOR OBJECT. A QIJ fit and a bootstrap each produce all of
# an estimator's outputs at once, so wall time is a property of the estimator,
# not of the estimand: the paper's six plotted coordinates map onto these four
# rows (FP contributes `a` and `s`, the IMF `m_c` and `x`). The output count is
# named in the label wherever a row covers more than one, the convention the
# superseded F7(b) used ("FP (4 outputs)"). Pareto is deliberately absent, as
# it is from Figures A, B and C -- Figure D introduces no row no other figure
# shows.
_D_ROWS = [
    dict(dataset="mvt", estimator="nu", label=r"MVT $\nu$"),
    dict(dataset="mvt", estimator="tail", label=r"MVT $S_{99}$"),
    dict(dataset="fp", estimator="fp", label="FP (all estimands)"),
    dict(dataset="imf", estimator="chabrier", label="IMF (all estimands)"),
]

# median, and the two whisker tiers: IQR drawn thick, 5-95% drawn thin.
_D_PCTL = (5.0, 25.0, 50.0, 75.0, 95.0)
_D_DODGE = 0.17
_D_STRIP_H = 0.115   # half-height of the per-draw strip's vertical jitter


def _d_wall(run_dir: str, dataset: str, estimator: str) -> dict:
    """Per-draw wall times for one estimator object, and their ratio.

    `wall_time_total` from `qij.parquet` is QIJ's whole fit -- quantizer,
    influence model, psi0, uncertainty, full-data stage and refinement --
    and `boot.h5`'s `wall_time` is the B-replicate loop. Neither includes
    building the interval itself (`qij_interval` 233us, `percentile_interval`
    784us at B=2000, both about 0.001% of their method's time, measured),
    nor the point estimate, which `study.py` computes outside both timers.
    The exclusion is therefore symmetric.

    `boot.h5` is ordered by its own `s`, which need not match `qij.parquet`'s
    row order under joblib's out-of-order completion, so it is sorted before
    the two are paired. The ratio is formed PER DRAW and summarised after --
    a median of ratios, not a ratio of medians -- because numerator and
    denominator share a draw and a process, so that pairing cancels machine
    noise the two marginals still carry.
    """
    path = _product_dir(run_dir, dataset, estimator)
    q = pd.read_parquet(os.path.join(path, "qij.parquet"))
    with h5py.File(os.path.join(path, "boot.h5"), "r") as f:
        bw = np.asarray(f["wall_time"][:], dtype=float)
        bs = np.asarray(f["s"][:])
    bw = bw[np.argsort(bs)][:len(q)]
    qs = np.asarray(q["wall_time_total"], dtype=float)
    ok = np.isfinite(qs) & np.isfinite(bw) & (bw > 0) & (qs > 0)
    return dict(qij=qs[ok], boot=bw[ok], ratio=qs[ok] / bw[ok], n=int(ok.sum()))


def _log_ticks(lo: float, hi: float, target: int = 4) -> list:
    """Round tick values spanning [lo, hi] on a log axis, at whatever
    resolution the span needs.

    A fixed 1-2-5 set is too coarse for a narrow row: the MVT panels span
    well under a decade and got one or two ticks from it. The mantissa sets
    below are tried coarsest first and the first to yield `target` ticks
    wins, so a decade-wide row still reads 1, 2, 5 while a half-decade one
    gets 0.4, 0.5, 0.7, 1.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= lo:
        return []
    for mant in ((1.0, 2.0, 5.0),
                 (1.0, 1.5, 2.0, 3.0, 5.0, 7.0),
                 (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)):
        out = []
        e = int(np.floor(np.log10(lo)))
        while 10.0 ** e <= hi:
            out += [m * 10.0 ** e for m in mant if lo <= m * 10.0 ** e <= hi]
            e += 1
        if len(out) >= target:
            return out
    return out


def _d_strip(ax, x, y: float, color: str, rng_seed: int = 0) -> None:
    """All 1000 per-draw values as a translucent strip beneath the summary
    glyph, jittered vertically so coincident values separate.

    This is the density the panels wanted and a violin could not give them:
    a KDE needs vertical room these panels do not have, while a strip renders
    at any height, fills the horizontal region where the data actually is,
    and shows skew and outliers without a smoothing choice. Deterministic
    jitter -- a fixed seed -- so the figure is reproducible. Drawn as
    vectors, not rasterized: the points are the figure's actual content and
    must stay sharp at print resolution.
    """
    rng = np.random.default_rng(rng_seed)
    jit = rng.uniform(-_D_STRIP_H, _D_STRIP_H, size=len(x))
    ax.plot(x, np.full(len(x), y) + jit, ls="none", marker="o", ms=1.1,
            color=color, alpha=0.13, mec="none", zorder=1.5)


def _d_glyph(ax, x, y: float, marker: str, color: str) -> None:
    """The summary mark: the median alone, in black, over its own strip.

    The IQR and 5-95% tiers this used to draw are gone -- with every draw
    already plotted as a point, a bar and whisker restate what the cloud
    shows and clutter the rows where the cloud is tight. Black keeps the
    median reading as an annotation ON the cloud rather than as one more
    member of it. `color` is accepted and unused: the series is identified
    by its cloud, and an outline in the series colour was tried and dropped
    for muddying the mark at this size.
    """
    ax.plot([float(np.median(x))], [y], marker=marker, color=OI["black"],
            ms=plt.rcParams["lines.markersize"] * 1.15, mec="none",
            ls="none", zorder=4)


def _d_facet(ax, st: dict, panel_lbl: str):
    """One estimator's facet, carrying TWO INDEPENDENT x scales.

    Three markers on two scales, with marker colour the cue for which scale
    each belongs to:

      * blue (QIJ) and orange (Boot) are ABSOLUTE SECONDS, read against the
        twin axis at the top, whose limits are local to this row.
      * green (Ratio) is QIJ/Boot, read against the shared axis at the
        bottom, which every facet has in common.

    The two scales are deliberately NOT a rescaling of one another -- the
    seconds axis is fitted to this row's own data while the ratio axis is
    shared -- so the green marker does not coincide with the blue one and
    carries information neither of the others does.

    Returns the twin so `fig_d` can set its scale and ticks once the shared
    ratio axis is final.
    """
    ratio = st["qij"] / st["boot"]          # per draw, the paired quantity
    # Green, matching the ratio series: the line marks ratio = 1 on the
    # PRIMARY axis, so it is the green marker it is there to be compared
    # against. In black it read as a neutral panel divider, and in panel (a)
    # it sits near the orange marker's seconds position by coincidence,
    # inviting exactly the wrong pairing.
    ax.axvline(1.0, color=OI["green"], ls=":", lw=1.1, zorder=1)
    _d_strip(ax, ratio, -_D_DODGE * 1.7, OI["green"], rng_seed=2)
    _d_glyph(ax, ratio, -_D_DODGE * 1.7, "s", OI["green"])

    sec = ax.twiny()
    _d_strip(sec, st["qij"], _D_DODGE * 1.7, METHOD["qij"]["color"], rng_seed=0)
    _d_strip(sec, st["boot"], 0.0, METHOD["boot"]["color"], rng_seed=1)
    _d_glyph(sec, st["qij"], _D_DODGE * 1.7, "o", METHOD["qij"]["color"])
    _d_glyph(sec, st["boot"], 0.0, "o", METHOD["boot"]["color"])

    for a in (ax, sec):
        a.set_ylim(-0.72, 1.05)
        a.set_yticks([])
        a.grid(False)
        # A closed box, and ticks turned inward, on BOTH axes: the twin draws
        # its own spines over the primary's, so enabling the top spine on the
        # primary alone leaves the top of the panel open.
        for side in ("top", "right", "bottom", "left"):
            a.spines[side].set_visible(True)
            a.spines[side].set_linewidth(0.8)
        a.tick_params(axis="x", direction="in", length=3.0)
    sec.tick_params(axis="x", labelsize=plt.rcParams["xtick.labelsize"] * 0.8,
                    pad=2.0, direction="in", length=3.0)

    # The panel label sits INSIDE the box: in a 2x2 the space left of the
    # right-hand column is a gutter between panels, and a label there reads
    # as belonging to neither. The y-range carries headroom above the top
    # marker for exactly this.
    ax.text(0.025, 0.93, f"{panel_lbl} {st['label']}", transform=ax.transAxes,
            ha="left", va="top", fontweight="bold",
            fontsize=plt.rcParams["axes.labelsize"] * 0.78, zorder=5)

    # The SECONDS range is per panel -- nobody compares MVT nu's 0.39 s to
    # the IMF's 14.4 s, and a shared one would leave every panel empty. The
    # RATIO range is not: it is dimensionless, so position means the same
    # thing in every panel, and sharing it is what lets a reader see at a
    # glance that the IMF sits far left and the Fundamental Plane far right.
    # Floating it per panel was tried and reverted: it put the unity line in
    # a different place in each panel, made (a) and (d) look alike at 0.37
    # and 0.23, and invited position comparisons that were no longer valid.
    # `fig_d` sets the shared one from the ranges returned here.
    s_lo = min(np.percentile(st["qij"], 1), np.percentile(st["boot"], 1))
    s_hi = max(np.percentile(st["qij"], 99), np.percentile(st["boot"], 99))
    r_lo = min(float(np.percentile(ratio, 1)), 1.0)
    r_hi = max(float(np.percentile(ratio, 99)), 1.0)
    return sec, s_lo, s_hi, r_lo, r_hi


def fig_d(run_dir: str, level: float = _LEVEL) -> plt.Figure:
    """Figure D -- wall time (redesigned by the author, 20 September 2026,
    replacing the 1x3 cost/accuracy/ratio-against-N design entirely).

    Answers one question: how long does QIJ take to run, against the
    standard bootstrap, for the estimators the paper plots.

    One column of facets, ONE PER ESTIMATOR OBJECT (`_D_ROWS`): a QIJ fit
    and a bootstrap each produce all of an estimator's outputs at once, so
    wall time belongs to the estimator, not the estimand -- FP's four
    outputs come from a single fit and the IMF's three from another, and
    the paper's six plotted coordinates collapse to these four rows.
    Pareto is absent, as it is from every other figure.

    Each facet carries BOTH quantities on one set of markers: a linear
    seconds axis below, scaled to that row alone, and a secondary axis
    above reading the identical positions as a QIJ/Boot ratio. That is
    what replaced a two-panel side-by-side layout, whose second panel
    duplicated the first panel's geometry to show a number the first panel
    already encoded as a gap -- and spent half the figure's width doing it.

    Bars were considered and rejected: a bar encodes distance from a
    baseline, and the useful baseline here is the other method, not zero.
    Violins and ridges likewise: measured over the 1000 draws every one of
    these distributions is unimodal with an IQR of 3-12% of its median, so
    a density has nothing to reveal.

    `run_dir` is the main study's product root and the figure's ONLY input:
    no cost-against-N sweep, no separate timing run. Both per-draw times
    come from products already there -- `qij.parquet`'s `wall_time_total`
    and `boot.h5`'s `wall_time` -- where a draw's QIJ and its bootstrap ran
    back to back in one process with one BLAS thread, so the ratio within a
    draw cancels the machine. The dedicated 20-draw timing run is NOT used:
    at 20 draws it disagreed with these 1000 by up to 0.35 on the cheap
    estimators, a sample-size effect rather than a contention one (checked:
    a fresh process's first draw is not anomalous).

    Two facts the caption must carry, neither visible in the figure:
      * the denominator is the bootstrap at B = 2000; at B = 1000 every
        ratio on the top axes doubles. The seconds axes are immune, which
        is why both are drawn.
      * both methods parallelise, and the BOOTSTRAP PARALLELISES MORE
        COMPLETELY -- its B replicates are fully independent, while QIJ's
        refinement is a sequential queue (4-103 evaluations deep here).
        These are single-core numbers, which measure total work.
    """
    _use_style()
    rows = _D_ROWS
    # NOT sharex: each panel's ratio range now floats to its own data (every
    # one still spanning 1), which is what stops three of the four panels
    # being mostly empty in order to reach the Fundamental Plane's 3.7. The
    # price is a set of ratio tick labels per panel rather than one per
    # column, and the green markers are no longer comparable BY POSITION
    # across panels -- they are read against each panel's own unity line.
    fig, axes2d = plt.subplots(2, 2, figsize=FIGSIZE_D,
                                constrained_layout=False)
    axes = axes2d.ravel()          # a, b across the top; c, d beneath

    stats = [dict(r, **_d_wall(run_dir, r["dataset"], r["estimator"])) for r in rows]
    facets = [_d_facet(ax, st, f"{chr(ord('a') + i)}.")
              for i, (ax, st) in enumerate(zip(axes, stats))]

    # One ratio scale for every panel, padded tightly. The old design's
    # emptiness came from its 0.70x/1.45x padding, not from sharing: at
    # 0.92x/1.09x the shared range is barely wider than the data it has to
    # cover, and the comparability is kept.
    r_lo = min(f[3] for f in facets)
    r_hi = max(f[4] for f in facets)
    r_ticks = _log_ticks(r_lo * 0.92, r_hi * 1.09)
    for i, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_xlim(r_lo * 0.92, r_hi * 1.09)
        ax.xaxis.set_major_locator(plt.FixedLocator(r_ticks))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        # Only the bottom row is labelled: the scale is identical in all four.
        ax.set_xticklabels([f"{t:g}" for t in r_ticks] if i >= 2 else [])

    for sec, s_lo, s_hi, _, _ in facets:
        sec.set_xscale("log")
        sec.set_xlim(s_lo * 0.92, s_hi * 1.09)
        ticks = _log_ticks(s_lo * 0.92, s_hi * 1.09)
        sec.xaxis.set_major_locator(plt.FixedLocator(ticks))
        sec.xaxis.set_minor_locator(plt.NullLocator())
        sec.set_xticklabels([f"{t:g}" for t in ticks])

    # Applied last and to every axis, primary and twin, major and minor: set
    # inside `_d_facet` alone it was silently undone for the bottom row,
    # whose labelled ticks rendered outside the box.
    for a in list(axes) + [f[0] for f in facets]:
        a.tick_params(axis="x", which="both", direction="in", length=3.0,
                      labelsize=plt.rcParams["xtick.labelsize"] * 0.8)

    fig.subplots_adjust(left=0.045, right=0.985, top=0.800, bottom=0.275,
                        hspace=0.72, wspace=0.11)
    # Both axis titles are placed with `fig.text` at coordinates derived from
    # the panels' OWN measured positions rather than with `supxlabel`/
    # `suptitle`, whose `y` was not honoured here and put the label inside the
    # panels. Note these coordinates are computed ONCE: they are only correct
    # while the axes stay where `subplots_adjust` put them, which is why
    # `figure.constrained_layout.use` must stay False (see its note in `RC`).
    top = max(a.get_position().y1 for a in axes)
    bot = min(a.get_position().y0 for a in axes)
    fig.text(0.5, top + 0.135, "Wall Time [seconds]", ha="center", va="center",
             fontsize=plt.rcParams["axes.labelsize"], fontweight="bold")
    fig.text(0.5, bot - 0.150, "Wall Time [Ratio]", ha="center", va="center",
             fontsize=plt.rcParams["axes.labelsize"], fontweight="bold")

    # "[s]" on the two seconds series and nothing on the ratio: the legend
    # is where a reader learns which of the two axes a colour belongs to,
    # and the bracketed unit says it without a sentence.
    # Legend markers stay in the SERIES colours even though every median is
    # drawn black: the colour belongs to the cloud, which is what identifies
    # a series and which axis it is read against. Boot is a circle, not the
    # square it is in Figure A -- the square is reserved here for the ratio,
    # the one series read against the other axis.
    handles = [
        Line2D([], [], color=METHOD["qij"]["color"], ls="none", marker="o", label="QIJ [s]"),
        Line2D([], [], color=METHOD["boot"]["color"], ls="none", marker="o", label="Boot [s]"),
        Line2D([], [], color=OI["green"], ls="none", marker="s", label="Ratio QIJ/Boot"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.008), columnspacing=1.6, handlelength=1.0,
               handletextpad=0.35, fontsize=plt.rcParams["legend.fontsize"])
    return fig
