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
    order. No interval is read here -- see `_compute_intervals`."""
    estimator_dir = Path(estimator_dir)
    truth = pd.read_parquet(estimator_dir / "truth.parquet")
    qij_df = pd.read_parquet(estimator_dir / "qij.parquet")
    df = truth.merge(qij_df, on="s", how="inner")

    with h5py.File(estimator_dir / "boot.h5", "r") as h5f:
        theta_bootstrap = h5f["theta"][...]
        s_bootstrap = h5f["s"][...]
        wall_time_bootstrap = h5f["wall_time"][...]
        bootstrap_outputs = list(h5f.attrs["outputs"])

    bootstrap_row = pd.Series(np.arange(len(s_bootstrap)), index=s_bootstrap)
    order = bootstrap_row.loc[df["s"].to_numpy()].to_numpy()
    col_order = [bootstrap_outputs.index(o) for o in outputs]
    theta_bootstrap = theta_bootstrap[order][:, :, col_order]
    wall_time_bootstrap = wall_time_bootstrap[order]

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
    }


def _compute_intervals(loaded, level):
    """The QIJ and bootstrap intervals at `level`, one row per draw,
    recomputed from the arrays `_load_draws` returned:
    `qij_interval(theta_hat, V_tot_hat, a_bca, level)` and
    `percentile_interval(theta_bootstrap, level)`. Loop over draws only
    (<= S), never over N."""
    theta_hat = loaded["theta_hat"]
    n, q = theta_hat.shape
    qij_lo_hi = np.full((n, q, 2), np.nan)
    bootstrap_lo_hi = np.full((n, q, 2), np.nan)
    for i in range(n):
        qij_lo_hi[i] = qij_interval(theta_hat[i], loaded["v_tot_hat"][i],
                                     loaded["a_bca"][i], level)
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
      n_draws.

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

        width_qij = qij_lo_hi[..., 1] - qij_lo_hi[..., 0]
        width_bootstrap = bootstrap_lo_hi[..., 1] - bootstrap_lo_hi[..., 0]
        width_ratio = _safe_ratio(width_qij, width_bootstrap)

        wall_time_ratio = _safe_ratio(loaded["wall_time_qij"], loaded["wall_time_bootstrap"])

        for j, output in enumerate(outputs):
            p_qij, se_qij, n_qij = _coverage_se(cov_qij[:, j])
            p_bootstrap, se_bootstrap, n_bootstrap = _coverage_se(cov_bootstrap[:, j])
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
                "n_coverage": min(n_qij, n_bootstrap),
                "width_ratio_0.95_median": np.nanmedian(width_ratio[:, j]),
                "L_median": np.nanmedian(loaded["L"][:, j]),
                "evaluations_median": np.nanmedian(loaded["evaluations"]),
                "normalized_rows_median": np.nanmedian(loaded["normalized_rows"]),
                "wall_time_ratio_qij_over_bootstrap_median": np.nanmedian(wall_time_ratio),
                "n_draws": int(len(loaded["s"])),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coverage grid (F2(b))
# ---------------------------------------------------------------------------

def coverage_grid(run_dir):
    """
    The coverage grid for the calibration panel F2(b): empirical
    against nominal coverage over the config's levels (`levels` in
    `<run_dir>/config.yaml`, plan section 3 -- not hardcoded), for both
    the QIJ and the bootstrap percentile interval, pooled across every
    (dataset, estimator, output, draw) coordinate. One row per level:
    `level, coverage_qij, coverage_qij_se, n_qij, coverage_bootstrap,
    coverage_bootstrap_se, n_bootstrap`, the standard errors
    sqrt(p(1-p)/n).
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
            pooled_qij.append(_covered(loaded["theta_true"], qij_lo_hi).ravel())
            pooled_bootstrap.append(_covered(loaded["theta_true"], bootstrap_lo_hi).ravel())
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
