"""
The study's tables (plan section 8): pure functions of the products in
`<run_dir>/<dataset>/<estimator>/` (plan section 3, pinned exactly in
the interface sheet section 5) -- `truth.parquet`, `qij.parquet` and
`boot.h5` -- and, for `coverage_grid`, `<run_dir>/config.yaml`. Nothing
else is read: not the estimator, not the dataset, not a re-run. No
interval is stored in any product, so every table here recomputes both
intervals itself, from `core.intervals.qij_interval` and
`core.intervals.percentile_interval`, joined on the draw index `s`.

Loops run over (dataset, estimator) directories, draws and coverage
levels only -- never over the N data points; S is at most a few
thousand, so a per-draw call to the interval functions is cheap and is
the direct, unambiguous way to recompute "an interval at any level
downstream without re-running anything" (`core/intervals.py`).

The second-order QIJ interval (plan §34-35) was built, validated
against §34's decision rule on the rev-7 products, and rejected: it
lowered coverage on every coordinate at every level (plan §36.1,
§36.2 ruling 6). It is removed from this module along with the
`qij2_intervals_closed_form` it called. `qij.parquet`'s stored bin
constituents (`bin_mass_<o>`, `bin_influence_<o>`, `bin_d2T_<o>`) stay
in the product regardless -- they are the between-bin and bias terms'
own inputs, not a second-order-only cost -- but nothing in this module
reads them any more; a run that carries them merges in and tables the
same as a run that does not, no special casing needed either way.

Comparison-set ruling (figure spec, "Comparison set"; plan §36.11, 19
September 2026): every column that compares the two methods -- `t1`'s
coverage of both, its width ratio, and `n_coverage`; `coverage_grid`'s
pooled coverage of both -- is computed on the set of draws where BOTH
methods' intervals are finite, never on each method's own separately
sized valid set. `_comparison_mask` is that test, written once, and is
the only place either table applies it.

Cost is normalized in three layers (plan section 8), kept apart by
construction rather than by a note to remember: (1) evaluation counts
and normalized rows, exact and machine-independent; (2) wall time only
as the ratio QIJ/bootstrap within the same draw on the same worker,
which cancels the machine and its load; (3) absolute seconds, only
from the timing run, one worker one thread. `cost_table` prefixes every
column `L1_`, `L2_` or `L3_` for exactly this reason: an `L3_*` column
can never be mistaken for a duration comparable to another run's, and
an `L2_*` column can never be mistaken for an absolute time.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from qij.core.intervals import percentile_interval, qij_interval

__all__ = ["t1", "coverage_grid", "cost_table"]

T1_LEVEL = 0.95


# ---------------------------------------------------------------------------
# Reading the products (truth.parquet, qij.parquet, boot.h5)
# ---------------------------------------------------------------------------

def _product_dirs(run_dir):
    """(dataset, estimator, path) for every `<run_dir>/<dataset>/
    <estimator>/` directory that has written `truth.parquet` (plan
    section 3's layout), discovered from the directories themselves --
    not from config.yaml, so a partially landed run still tables what
    exists."""
    run_dir = Path(run_dir)
    dirs = []
    for truth_path in sorted(run_dir.glob("*/*/truth.parquet")):
        estimator_dir = truth_path.parent
        dataset = estimator_dir.parent.name
        estimator = estimator_dir.name
        dirs.append((dataset, estimator, estimator_dir))
    return dirs


def _outputs(truth_df):
    """The estimator's output names, read off truth.parquet's own
    `theta_hat_<output>` columns -- never from estimators.py."""
    prefix = "theta_hat_"
    return [c[len(prefix):] for c in truth_df.columns if c.startswith(prefix)]


def _load_draws(estimator_dir, outputs):
    """Load and join truth.parquet, qij.parquet and boot.h5 for one
    (dataset, estimator) directory, aligned on `s` and on `outputs`'
    order. No interval is read here -- see `_compute_intervals`.

    `boot_n_failed` and `B` are boot.h5's per-draw failed-replicate
    count and its replicate count; `qij_n_failed` is qij.parquet's
    per-draw failed-evaluation count -- both shared across a vector
    estimator's outputs, as their products store them, and read here
    for T1's failure fractions (plan section 4's rare-support ruling).

    `qij.parquet` may also carry `bin_mass_<o>`/`bin_influence_<o>`/
    `bin_d2T_<o>` list columns (the between-bin and bias terms' own
    inputs, kept in the product regardless of what reads them, plan
    §36.2 ruling 6) -- nothing here loads them; `truth.merge(qij_df)`
    is indifferent to a source frame carrying extra columns, so a run
    that has them and a run that does not table identically, with no
    branch needed either way.
    """
    estimator_dir = Path(estimator_dir)
    truth = pd.read_parquet(estimator_dir / "truth.parquet")
    qij_df = pd.read_parquet(estimator_dir / "qij.parquet")
    df = truth.merge(qij_df, on="s", how="inner")

    with h5py.File(estimator_dir / "boot.h5", "r") as h5f:
        theta_bootstrap = h5f["theta"][...]
        s_bootstrap = h5f["s"][...]
        wall_time_bootstrap = h5f["wall_time"][...]
        n_failed_bootstrap = h5f["n_failed"][...]
        bootstrap_outputs = list(h5f.attrs["outputs"])

    bootstrap_row = pd.Series(np.arange(len(s_bootstrap)), index=s_bootstrap)
    order = bootstrap_row.loc[df["s"].to_numpy()].to_numpy()
    col_order = [bootstrap_outputs.index(o) for o in outputs]
    theta_bootstrap = theta_bootstrap[order][:, :, col_order]
    wall_time_bootstrap = wall_time_bootstrap[order]
    n_failed_bootstrap = n_failed_bootstrap[order]

    has_oracle = f"V_oracle_{outputs[0]}" in df.columns
    return {
        "s": df["s"].to_numpy(),
        "theta_true": df[[f"theta_true_{o}" for o in outputs]].to_numpy(),
        "theta_hat": df[[f"theta_hat_{o}" for o in outputs]].to_numpy(),
        "v_btw": df[[f"V_btw_{o}" for o in outputs]].to_numpy(),
        "v_tot_hat": df[[f"V_tot_hat_{o}" for o in outputs]].to_numpy(),
        "a_bca": df[[f"a_bca_{o}" for o in outputs]].to_numpy(),
        "v_oracle": (df[[f"V_oracle_{o}" for o in outputs]].to_numpy()
                     if has_oracle else None),
        "theta_bootstrap": theta_bootstrap,
        "wall_time_qij": df["wall_time_total"].to_numpy(),
        "wall_time_bootstrap": wall_time_bootstrap,
        "L": df[[f"L_{o}" for o in outputs]].to_numpy(),
        "evaluations": df["evals_total"].to_numpy(),
        "normalized_rows": df["normalized_rows"].to_numpy(),
        "boot_n_failed": n_failed_bootstrap,
        "B": int(theta_bootstrap.shape[1]),
        "qij_n_failed": df["n_failed"].to_numpy(),
        "outputs": outputs,
    }


def _compute_intervals(loaded, level):
    """The QIJ and bootstrap intervals at `level`, one row per draw,
    recomputed from the arrays `_load_draws` returned:
    `qij_interval(theta_hat, V_btw, level)` and
    `percentile_interval(theta_bootstrap, level)`. The QIJ interval is
    the normal interval on the measured between-bin variance, with no
    acceleration term and no support clip (see
    `core.intervals.qij_interval` for why each came out); `v_tot_hat`
    and `a_bca` are still loaded, and T1 still reports `V_tot_hat`
    against the oracle and Monte Carlo variances, but neither enters an
    interval. Loop over draws only (<= S), never over N."""
    theta_hat = loaded["theta_hat"]
    n, q = theta_hat.shape
    qij_lo_hi = np.full((n, q, 2), np.nan)
    bootstrap_lo_hi = np.full((n, q, 2), np.nan)
    for i in range(n):
        qij_lo_hi[i] = qij_interval(theta_hat[i], loaded["v_btw"][i], level)
        bootstrap_lo_hi[i] = percentile_interval(loaded["theta_bootstrap"][i], level)
    return qij_lo_hi, bootstrap_lo_hi


# ---------------------------------------------------------------------------
# Small numeric helpers shared by the tables below
# ---------------------------------------------------------------------------

def _safe_ratio(num, den):
    """num/den elementwise; NaN where den is zero or not finite."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
    bad = ~np.isfinite(den) | (den == 0)
    return np.where(bad, np.nan, ratio)


def _covered(theta_true, lo_hi):
    """1.0/0.0 per (draw, output): whether theta_true lies in [lo, hi];
    NaN where either limit is not finite (a failed evaluation), so it
    is excluded from coverage rather than counted as a miss."""
    lo, hi = lo_hi[..., 0], lo_hi[..., 1]
    ok = np.isfinite(lo) & np.isfinite(hi)
    covered = (theta_true >= lo) & (theta_true <= hi)
    return np.where(ok, covered.astype(float), np.nan)


def _coverage_se(indicator):
    """Empirical coverage (a proportion) and its Monte Carlo standard
    error sqrt(p(1-p)/n), over the finite entries of `indicator`."""
    valid = indicator[np.isfinite(indicator)]
    n = int(valid.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    p = float(valid.mean())
    se = float(np.sqrt(p * (1.0 - p) / n))
    return p, se, n


def _comparison_mask(qij_lo_hi, bootstrap_lo_hi):
    """(n, q) boolean: True for a (draw, output) where BOTH methods'
    intervals are finite -- the comparison-set ruling (figure spec,
    "Comparison set"; plan §36.11, 19 September 2026). A draw whose
    full-data fit failed has no QIJ interval, and is dropped from the
    bootstrap's side too, even though `percentile_interval` may still
    form and score one from its own converged replicates (an IMF box-
    rule draw, e.g. s = 936, is exactly this case); a draw where QIJ's
    stage-1 collapsed, or where the bootstrap has no converged replicate
    at all, is dropped the same way. This is the conjunction of the SAME
    finiteness test `_covered` already applies to each method on its
    own -- not a second convention -- so a (draw, output) this marks
    usable is exactly one where `_covered` returns a defined 0/1 for
    both methods, never NaN for either. Called once here and once more
    from `coverage_grid`; every column the ruling names (coverage of
    both methods, the width ratio, `n_coverage`) is built by masking
    with this, nowhere by re-deriving the test."""
    qij_ok = np.isfinite(qij_lo_hi[..., 0]) & np.isfinite(qij_lo_hi[..., 1])
    boot_ok = np.isfinite(bootstrap_lo_hi[..., 0]) & np.isfinite(bootstrap_lo_hi[..., 1])
    return qij_ok & boot_ok


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

def t1(run_dir):
    """
    Table T1 (plan section 8; `QIJ_figure_spec_rev51.md` "Table T1"),
    one row per (dataset, estimator, output):

      dataset, estimator, output,
      V_btw_V_tot_mc_{median,p05,p95}, V_btw_V_oracle_{median,p05,p95},
      Vhat_tot_V_tot_mc_{median,p05,p95}, Vhat_tot_V_oracle_{median,p05,p95},
      Vhat_tot_V_boot_{median,p05,p95},
      coverage_qij_0.95, coverage_qij_0.95_se,
      coverage_bootstrap_0.95, coverage_bootstrap_0.95_se, n_coverage,
      width_ratio_0.95_median, L_median, evaluations_median,
      normalized_rows_median, wall_time_ratio_qij_over_bootstrap_median,
      bootstrap_replicate_failure_fraction, qij_evaluation_failure_fraction,
      n_qij_failed, qij_failure_fraction,
      draw_failure_fraction, n_draws, n_draws_excluded.

    V_tot has two sources, both reported (plan section 8): the true
    sampling variance Var_s(theta_hat) over the draws in
    truth.parquet ("_tot_mc"), and, where an analytic influence exists,
    the per-draw oracle column V_oracle_<output> ("_oracle", NaN where
    absent). V_boot is the per-draw bootstrap variance,
    Var_b(theta_bootstrap) over boot.h5's replicate axis for that same
    draw. Coverage and width ratio are at the 0.95 level, recomputed
    from `qij_interval`/`percentile_interval`; coverage is pooled over
    draws within one (dataset, estimator, output) and carries its Monte
    Carlo standard error sqrt(p(1-p)/n_coverage). Wall time is reported
    only as the ratio QIJ/bootstrap within the same draw on the same
    worker (cost layer 2, plan section 8) -- never an absolute
    duration. L is per output (`L_<output>`): the refinement reaches a
    different bin count per coordinate, so it varies across an
    estimator's output rows. Evaluations and normalized rows are
    shared across a vector estimator's outputs, as qij.parquet stores
    them (one column, not one per output), so those two medians repeat
    across an estimator's output rows -- the product's own structure,
    not a table artefact.

    FAILURE ACCOUNTING (plan §36.1, §36.2 ruling 5 -- the rev-7
    coverage-exclusion audit and the ruling it produced). The rev-7
    products showed draws vanishing from coverage that no existing
    column counted: the IMF lost 31 of 1000 draws from coverage while
    `n_draws_excluded` (below) reported 17, and MVT's nu lost 3 while
    `n_draws_excluded` reported 0. The gap was two further, distinct
    causes -- a failed initial-bin evaluation, and a stage-1 collapse
    (the influence model's constant path, or a refinement that never
    splits) that leaves `a_bca` undefined (`0/0` in
    `core.outputs.acceleration`) even though `theta_hat` and
    `V_tot_hat` are both finite -- neither of which is a full-data box
    hit and neither of which touches `theta_hat`, so the old test
    (`theta_hat` NaN) never saw them.

    `n_qij_failed` and `qij_failure_fraction` close that gap directly,
    per output, by testing the thing coverage actually depends on
    rather than a proxy for it: a draw counts as QIJ-failed here
    exactly when `qij_interval`'s own `[lo, hi]` is non-finite for that
    output -- i.e. exactly the draws `cov_qij` (below) already marks
    NaN, whether it was `theta_hat` or `V_btw` that was NaN. Testing
    the interval, not a proxy such as `L == 1 and V_btw == 0`, counts
    every cause without a branch per cause and without double-counting
    a draw that is NaN by both routes at once (there is only one route
    once the interval itself is the test).

    One consequence of the 20 September interval change (the normal
    interval on `V_btw`; `core.intervals.qij_interval`) is worth
    stating, because it is a difference in what this column would count
    on an OLD product. A run written before the collapsed-stage-1 fix
    stored a collapsed draw as `L = 1`, `V_btw = V_win_hat = V_tot_hat
    = B_hat = 0.0`, `a_bca = NaN`, and the BCa adjustment turned that
    undefined acceleration into a non-finite interval, so the draw was
    counted failed. Under the new form the same row gives the finite
    degenerate interval `[theta_hat, theta_hat]`, which would be
    counted as a (certainly missing) draw instead. No such row exists
    in the products this tables: a run written after the fix stores
    `V_btw`/`V_tot_hat`/`a_bca` all NaN directly (plan §36.2 ruling 4),
    which is still NaN `[lo, hi]`, and on the main run the same three
    MVT nu draws are excluded before and after.

    `n_qij_failed` is built from the exact same finiteness mask
    `cov_qij` uses, so `n_draws - n_qij_failed` equals `n_qij`
    (`_coverage_se(cov_qij[:, j])`'s own valid count) by construction,
    for every row, on both product shapes -- but that count is QIJ's
    own, and is no longer what `n_coverage` reports (see below).

    SUPERSEDED BY THE COMPARISON-SET RULING (figure spec, "Comparison
    set"; plan §36.11, 19 September 2026). The paragraph above, and the
    `min(n_qij, n_bootstrap)` it replaced (plan §36.2 ruling 5), both
    took `n_coverage` to be QIJ-side only, resting on an audited
    empirical fact: across every directory checked (plan §36.1) the
    bootstrap side never lost a whole draw, `percentile_interval`
    needing every one of a draw's replicates to fail before it returns
    NaN. That fact does not cover the case the ruling was written for:
    a draw whose full-data fit itself failed has no `theta_hat` and
    therefore no QIJ interval, but `percentile_interval` still forms
    and this table still SCORED one from its converged replicates on
    that same draw -- on the IMF, `coverage_qij_0.95` stood on 969
    draws and `coverage_bootstrap_0.95` on up to 1000, two different
    populations averaged as though they were one (s = 936 is such a
    draw: no full-data fit, no QIJ interval, a bootstrap interval
    formed and scored anyway).

    `n_coverage` is now the size of the COMMON set: `_comparison_mask`
    applied to `qij_lo_hi`/`bootstrap_lo_hi`, True only where BOTH
    methods' intervals are finite for that (draw, output). `cov_qij`
    and `cov_bootstrap` are masked to that common set (`cov_qij_common`/
    `cov_bootstrap_common` below) before `coverage_qij_0.95`,
    `coverage_bootstrap_0.95` and `width_ratio_0.95_median` are computed
    from them, so those three columns and `n_coverage` are always over
    the identical set of draws, by construction, on both product
    shapes and for both methods.

    `n_qij_failed`/`qij_failure_fraction` below are UNCHANGED by this
    ruling and stay computed from the UNMASKED `cov_qij` -- they are
    QIJ's own failure count, a diagnostic of QIJ's construction alone,
    not the comparison, and the ruling does not ask that diagnostic to
    move. `n_coverage <= n_draws - n_qij_failed` in general now (with
    equality exactly when the bootstrap never independently loses a
    draw the common set would otherwise have kept): the old identity
    between them is no longer asserted, since `n_coverage` can now be
    smaller than QIJ's own valid count whenever the bootstrap fails a
    draw QIJ did not.

    The rare-support ruling (plan section 4) requires both methods'
    failure fractions to be visible rather than silently absorbed:
    `bootstrap_replicate_failure_fraction` is the share of that
    (dataset, estimator)'s bootstrap replicates that returned NaN
    (boot.h5's per-draw `n_failed` against its `B`, pooled over
    draws); `qij_evaluation_failure_fraction` is the share of QIJ
    evaluations that failed (qij.parquet's per-draw `n_failed` against
    `evals_total`, the evaluations that draw spent, pooled over
    draws) -- a noisy, pooled proxy for QIJ-side draw loss, not a count
    of it (plan §36.1: it is exactly 0.0 for MVT's nu despite 3 draws
    lost, since those 3 draws fail no evaluation at all, only the
    refinement's own decision to never split); `draw_failure_fraction`
    is, per output, the share of draws excluded entirely because the
    full-data fit failed -- `theta_hat` NaN in truth.parquet. These are
    distinct measurements at different stages: the replicate/evaluation
    fractions describe failures within draws that were run at all,
    `draw_failure_fraction`/`n_draws_excluded` count draws whose
    full-data fit never produced a usable `theta_hat`, and
    `n_qij_failed`/`qij_failure_fraction` count draws whose QIJ RESULT
    -- the interval coverage tests -- came back NaN regardless of cause.
    A failed draw's NaN `theta_hat` already falls out of `v_tot_mc`
    (`np.nanvar` above), so it is excluded from the truth product's
    variance without special-casing here.

    `n_draws_excluded` is `draw_failure_fraction`'s raw count and is
    UNCHANGED by this ruling -- it stays exactly what it measured
    before (plan §36.2 ruling 5's "second column"): the number of this
    (dataset, estimator, output)'s draws whose `theta_hat` is NaN in
    truth.parquet, i.e. the full-data box-rule count. It is kept
    alongside `n_qij_failed` specifically so both causes of lost
    coverage stay visible and distinguishable: `n_draws_excluded` is
    the full-data fit's own failure (theta_hat undefined, so nothing
    downstream of it is even attempted meaningfully), `n_qij_failed`
    is every draw QIJ's own construction could not turn into an
    interval, which is a strict superset whenever the two causes
    differ (plan §36.1: on the IMF, 17 of the 31 are box hits, the
    other 14 are QIJ-side; a box hit's NaN `theta_hat` also NaNs the
    interval, since `qij_interval` adds `+/- z sqrt(V_btw)` to
    `theta_hat` itself, so `n_qij_failed >= n_draws_excluded` always,
    never less).
    Per plan section 4, a fit resting on an active parameter-box bound
    is not a stationary point, so its influence is not defined there
    either; the estimator, never this table, is what applies the rule
    -- a bound hit already comes back as NaN `theta_hat` (and,
    following from it, NaN in every column this table derives from
    that draw), so testing `theta_hat` here only counts what the
    estimator already decided.

    It is named for what it MEASURES, not for what produces it. The
    test is "this draw has no usable full-data fit", and a fit at the
    box is one cause among several: a non-converged optimizer and a
    failed continuity solve return NaN by the same route. Since the
    IMF optimizer was given its exact gradient and Hessian those other
    causes measure 0.0% (the S=1000 run at N=2000), so in practice the
    count IS the box-rule count and the paper may report it as such --
    but that is an empirical fact about the current fitter, not a
    property of this column, and it should be re-checked rather than
    assumed if the count and the box-hit count ever disagree.

    A zero here on a run built BEFORE the box tolerance was corrected
    (commit afbf169) means only that the old tolerance, `eta*(1+|b|)`,
    was too tight to catch a fit parked 1e-4 from its bound -- not
    that no draw reached the box. The S=1000 run at 36ce72a reports 0
    while seventeen of its draws sit at p = 20 to within 2e-4.
    It is 0 for every closed-form and vector closed-form estimator
    (Pareto, MVT, FP), which never return NaN, and only the IMF rows
    (the one estimator with an active box, plan section 4) can be
    nonzero. Placed as T1's final column per the spec.
    """
    rows = []
    for dataset, estimator, estimator_dir in _product_dirs(run_dir):
        truth = pd.read_parquet(estimator_dir / "truth.parquet")
        outputs = _outputs(truth)
        loaded = _load_draws(estimator_dir, outputs)
        qij_lo_hi, bootstrap_lo_hi = _compute_intervals(loaded, T1_LEVEL)

        v_tot_mc = np.nanvar(loaded["theta_hat"], axis=0, ddof=1)          # (q,)
        v_boot = np.nanvar(loaded["theta_bootstrap"], axis=1, ddof=1)      # (n, q)

        r_btw_mc = _safe_ratio(loaded["v_btw"], v_tot_mc[None, :])
        r_tot_mc = _safe_ratio(loaded["v_tot_hat"], v_tot_mc[None, :])
        r_tot_boot = _safe_ratio(loaded["v_tot_hat"], v_boot)
        if loaded["v_oracle"] is not None:
            r_btw_oracle = _safe_ratio(loaded["v_btw"], loaded["v_oracle"])
            r_tot_oracle = _safe_ratio(loaded["v_tot_hat"], loaded["v_oracle"])
        else:
            r_btw_oracle = np.full_like(r_btw_mc, np.nan)
            r_tot_oracle = np.full_like(r_tot_mc, np.nan)

        cov_qij = _covered(loaded["theta_true"], qij_lo_hi)
        cov_bootstrap = _covered(loaded["theta_true"], bootstrap_lo_hi)

        # Comparison-set ruling (figure spec, "Comparison set"; plan
        # §36.11): a draw where either method has no usable interval is
        # dropped from the comparison for BOTH, so coverage and the
        # width ratio are never averaged over two differently sized sets
        # of draws. `common` masks `cov_qij`/`cov_bootstrap`/`width_ratio`
        # alike; `cov_qij`/`cov_bootstrap` themselves (unmasked) stay
        # around only for `n_qij_failed` below, QIJ's own diagnostic.
        common = _comparison_mask(qij_lo_hi, bootstrap_lo_hi)
        cov_qij_common = np.where(common, cov_qij, np.nan)
        cov_bootstrap_common = np.where(common, cov_bootstrap, np.nan)

        width_qij = qij_lo_hi[..., 1] - qij_lo_hi[..., 0]
        width_bootstrap = bootstrap_lo_hi[..., 1] - bootstrap_lo_hi[..., 0]
        width_ratio = np.where(common, _safe_ratio(width_qij, width_bootstrap), np.nan)

        wall_time_ratio = _safe_ratio(loaded["wall_time_qij"], loaded["wall_time_bootstrap"])

        n_draws = len(loaded["s"])
        # nansum, not sum: boot_n_failed/qij_n_failed/evaluations are
        # int-valued counts and stay finite even on a box-hit draw (the
        # estimator still ran and was still counted, module docstring),
        # but a plain sum is one NaN away from silently taking every
        # draw's failure fraction to NaN -- nansum costs nothing today
        # and keeps this column from being the one place in t1 that
        # isn't NaN-aware if that ever changes upstream.
        bootstrap_replicate_failure_fraction = float(_safe_ratio(
            np.nansum(loaded["boot_n_failed"]), float(loaded["B"] * n_draws)))
        qij_evaluation_failure_fraction = float(_safe_ratio(
            np.nansum(loaded["qij_n_failed"]), np.nansum(loaded["evaluations"])))

        for j, output in enumerate(outputs):
            # QIJ's own failure diagnostic (plan §36.2 ruling 5, UNCHANGED
            # by the comparison-set ruling): every draw whose QIJ interval
            # is non-finite for this output, whatever produced it -- the
            # same mask `cov_qij[:, j]`'s own NaNs already carry, computed
            # from the UNMASKED `cov_qij`, so `n_qij_failed` counts QIJ's
            # own construction alone, not the comparison with the bootstrap.
            _p_qij_own, _se_qij_own, n_qij = _coverage_se(cov_qij[:, j])
            n_qij_failed = n_draws - n_qij

            # Reported coverage/width columns: the common set only
            # (comparison-set ruling, docstring above). `cov_qij_common`
            # and `cov_bootstrap_common` are masked by the identical
            # `common[:, j]`, and both are NaN everywhere `common[:, j]`
            # is False and defined everywhere it is True, so their own
            # valid counts from `_coverage_se` already agree with each
            # other and with `n_coverage` below -- by construction, not
            # by separately reconciling them.
            p_qij, se_qij, n_coverage = _coverage_se(cov_qij_common[:, j])
            p_bootstrap, se_bootstrap, _n_coverage_boot = _coverage_se(cov_bootstrap_common[:, j])
            rows.append({
                "dataset": dataset, "estimator": estimator, "output": output,
                "V_btw_V_tot_mc_median": np.nanmedian(r_btw_mc[:, j]),
                "V_btw_V_tot_mc_p05": np.nanpercentile(r_btw_mc[:, j], 5),
                "V_btw_V_tot_mc_p95": np.nanpercentile(r_btw_mc[:, j], 95),
                "V_btw_V_oracle_median": np.nanmedian(r_btw_oracle[:, j]),
                "V_btw_V_oracle_p05": np.nanpercentile(r_btw_oracle[:, j], 5),
                "V_btw_V_oracle_p95": np.nanpercentile(r_btw_oracle[:, j], 95),
                "Vhat_tot_V_tot_mc_median": np.nanmedian(r_tot_mc[:, j]),
                "Vhat_tot_V_tot_mc_p05": np.nanpercentile(r_tot_mc[:, j], 5),
                "Vhat_tot_V_tot_mc_p95": np.nanpercentile(r_tot_mc[:, j], 95),
                "Vhat_tot_V_oracle_median": np.nanmedian(r_tot_oracle[:, j]),
                "Vhat_tot_V_oracle_p05": np.nanpercentile(r_tot_oracle[:, j], 5),
                "Vhat_tot_V_oracle_p95": np.nanpercentile(r_tot_oracle[:, j], 95),
                "Vhat_tot_V_boot_median": np.nanmedian(r_tot_boot[:, j]),
                "Vhat_tot_V_boot_p05": np.nanpercentile(r_tot_boot[:, j], 5),
                "Vhat_tot_V_boot_p95": np.nanpercentile(r_tot_boot[:, j], 95),
                "coverage_qij_0.95": p_qij,
                "coverage_qij_0.95_se": se_qij,
                "coverage_bootstrap_0.95": p_bootstrap,
                "coverage_bootstrap_0.95_se": se_bootstrap,
                # The common set's own size (see docstring): both
                # methods' coverage above are computed on exactly this
                # many draws, never a min() taken after the fact.
                "n_coverage": n_coverage,
                "width_ratio_0.95_median": np.nanmedian(width_ratio[:, j]),
                "L_median": np.nanmedian(loaded["L"][:, j]),
                "evaluations_median": np.nanmedian(loaded["evaluations"]),
                "normalized_rows_median": np.nanmedian(loaded["normalized_rows"]),
                "wall_time_ratio_qij_over_bootstrap_median": np.nanmedian(wall_time_ratio),
                "bootstrap_replicate_failure_fraction": bootstrap_replicate_failure_fraction,
                "qij_evaluation_failure_fraction": qij_evaluation_failure_fraction,
                "n_qij_failed": int(n_qij_failed),
                "qij_failure_fraction": float(n_qij_failed) / n_draws if n_draws else float("nan"),
                "draw_failure_fraction": float(np.mean(np.isnan(loaded["theta_hat"][:, j]))),
                "n_draws": n_draws,
                "n_draws_excluded": int(np.sum(np.isnan(loaded["theta_hat"][:, j]))),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coverage grid (F2(b))
# ---------------------------------------------------------------------------

def coverage_grid(run_dir):
    """
    The coverage grid for the calibration panel F2(b): empirical
    against nominal coverage over the config's levels (`levels` in
    `<run_dir>/config.yaml`, plan section 3 -- not hardcoded), for the
    QIJ and the bootstrap percentile interval, pooled across every
    (dataset, estimator, output, draw) coordinate. One row per level:
    `level, coverage_qij, coverage_qij_se, n_qij, coverage_bootstrap,
    coverage_bootstrap_se, n_bootstrap`, the standard errors
    sqrt(p(1-p)/n).

    Comparison-set ruling (figure spec, "Comparison set"; plan §36.11):
    applied PER (dataset, estimator, output, level) -- `_comparison_mask`
    on that coordinate's own `qij_lo_hi`/`bootstrap_lo_hi` at this level
    -- before the coordinate's draws are ravelled into the pool, not
    after pooling. Both methods are therefore masked to the identical
    set at every coordinate they are pooled from, so `n_qij` and
    `n_bootstrap` below come out equal: the pooled COMMON count, not two
    separately sized pools that happen to share a name.
    """
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    levels = config["levels"]

    loaded_by_dir = []
    for dataset, estimator, estimator_dir in _product_dirs(run_dir):
        truth = pd.read_parquet(estimator_dir / "truth.parquet")
        outputs = _outputs(truth)
        loaded_by_dir.append(_load_draws(estimator_dir, outputs))

    rows = []
    for level in levels:
        pooled_qij = []
        pooled_bootstrap = []
        for loaded in loaded_by_dir:
            qij_lo_hi, bootstrap_lo_hi = _compute_intervals(loaded, level)
            common = _comparison_mask(qij_lo_hi, bootstrap_lo_hi)
            cov_qij = np.where(common, _covered(loaded["theta_true"], qij_lo_hi), np.nan)
            cov_bootstrap = np.where(common, _covered(loaded["theta_true"], bootstrap_lo_hi), np.nan)
            pooled_qij.append(cov_qij.ravel())
            pooled_bootstrap.append(cov_bootstrap.ravel())
        p_qij, se_qij, n_qij = _coverage_se(np.concatenate(pooled_qij))
        p_bootstrap, se_bootstrap, n_bootstrap = _coverage_se(np.concatenate(pooled_bootstrap))
        rows.append({
            "level": level,
            "coverage_qij": p_qij, "coverage_qij_se": se_qij, "n_qij": n_qij,
            "coverage_bootstrap": p_bootstrap, "coverage_bootstrap_se": se_bootstrap,
            "n_bootstrap": n_bootstrap,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cost table
# ---------------------------------------------------------------------------

def cost_table(run_dir, timing_dir):
    """
    The cost table (plan section 8), one row per (dataset, estimator),
    with the three cost layers in separate, explicitly labelled
    columns so they cannot be mixed up:

      dataset, estimator,
      L1_evaluations_median, L1_normalized_rows_median,
      L1_bootstrap_replicates,
      L2_wall_time_ratio_qij_over_bootstrap_median,
      L3_wall_time_qij_timing_run_seconds,
      L3_wall_time_bootstrap_timing_run_seconds.

    L1 (evaluation counts, exact and machine-independent) is read from
    `run_dir`'s qij.parquet (evals_total, normalized_rows, medians over
    draws) and boot.h5 (its replicate count B). L2 (wall time) is only
    the ratio QIJ/bootstrap within the same draw on the same worker,
    median over `run_dir`'s draws -- never an absolute duration. L3
    (absolute seconds) is read only from `timing_dir` (the one-worker,
    one-thread timing run, plan section 7), medians of its own
    wall_time_total (QIJ) and boot.h5 wall_time (bootstrap); these are
    not compared to `run_dir`'s wall time, only reported alongside it.
    """
    rows = []
    for dataset, estimator, estimator_dir in _product_dirs(run_dir):
        truth = pd.read_parquet(estimator_dir / "truth.parquet")
        outputs = _outputs(truth)
        loaded = _load_draws(estimator_dir, outputs)

        with h5py.File(estimator_dir / "boot.h5", "r") as h5f:
            n_replicates = int(h5f["theta"].shape[1])

        wall_time_ratio = _safe_ratio(loaded["wall_time_qij"], loaded["wall_time_bootstrap"])

        timing_estimator_dir = Path(timing_dir) / dataset / estimator
        timing_truth = pd.read_parquet(timing_estimator_dir / "truth.parquet")
        timing_outputs = _outputs(timing_truth)
        timing_loaded = _load_draws(timing_estimator_dir, timing_outputs)

        rows.append({
            "dataset": dataset, "estimator": estimator,
            "L1_evaluations_median": np.nanmedian(loaded["evaluations"]),
            "L1_normalized_rows_median": np.nanmedian(loaded["normalized_rows"]),
            "L1_bootstrap_replicates": n_replicates,
            "L2_wall_time_ratio_qij_over_bootstrap_median": np.nanmedian(wall_time_ratio),
            "L3_wall_time_qij_timing_run_seconds": np.nanmedian(timing_loaded["wall_time_qij"]),
            "L3_wall_time_bootstrap_timing_run_seconds": np.nanmedian(timing_loaded["wall_time_bootstrap"]),
        })
    return pd.DataFrame(rows)
