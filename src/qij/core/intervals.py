"""
The two interval constructions (plan §3, §6). Both are pure functions
of their arguments: no interval is ever stored in a product, so a
study, table or figure recomputes the interval it needs at whatever
level is asked for, and the bootstrap's interval from any prefix of
its replicates.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def qij_interval(theta_hat: np.ndarray, V_btw: np.ndarray,
                  level: float) -> np.ndarray:
    """
    The QIJ interval h^QIJ_level (glossary): the normal interval on the
    measured between-bin variance,

        theta_hat +/- z_{(1+level)/2} * sqrt(V_btw),

    centred at `theta_hat`. Both `theta_hat` and `V_btw` are (q,).
    Returns (q, 2), [lo, hi].

    WHY V_btw AND NOTHING ELSE. The second quantizer's job is to bound
    the within-bin term below the declared tolerance epsilon, so V_btw
    is a lower bound on the variance that is tight to epsilon. That is
    the paper's justification for the second stage, and the interval
    rests on it directly.

    WHAT CAME OUT, and the ablation on these products that established
    each term was immaterial (it is stated here, not re-derived):
    V_win_hat is under 0.6% of the total variance on all eleven
    coordinates, so V_tot_hat = V_btw + V_win_hat moved the half-width
    by under 0.3%; the median |a_bca| is 0.0002-0.037, so the BCa
    acceleration adjustment (`z / (1 - a z)`, z0 = 0) moved either
    endpoint by a comparable fraction of one standard error; and the
    natural-parameter support clip changed NO endpoint on ANY draw. The
    clip mattered only for the superseded two-regime IMF estimator,
    whose near-singular draws produced blown-up intervals; the Chabrier
    estimator that replaced it has a Hessian condition number of at
    most 13, so nothing blows up and nothing clips. V_win_hat,
    V_tot_hat, B_hat and a_bca are still computed and still written to
    the products -- they are diagnostics now, not interval inputs, and
    the estimators' `supports` declarations stay as documentation of
    each estimator's domain.

    A negative `V_btw` (it cannot arise from `core.ivq.between_terms`,
    which sums squares, but the guard costs nothing) is floored at zero
    rather than producing a NaN half-width. NaN in `theta_hat` or
    `V_btw` -- a failed draw -- propagates to NaN in `lo`/`hi`, which is
    how every caller detects a draw with no QIJ interval.
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    V_btw = np.asarray(V_btw, dtype=float)

    z = float(norm.ppf(0.5 * (1.0 + level)))
    half = z * np.sqrt(np.maximum(V_btw, 0.0))

    lo = theta_hat - half
    hi = theta_hat + half

    return np.stack([lo, hi], axis=-1)


def percentile_interval(replicates: np.ndarray, level: float) -> np.ndarray:
    """
    The bootstrap's percentile interval h^boot_level (glossary): the
    level*100% interval from the empirical quantiles of `replicates`
    (b, q) over its replicate axis (axis 0), ignoring failed (NaN)
    replicates -- a resample that loses what identifies the fit is
    excluded from the interval, not propagated into it (plan section
    4, the rare-support ruling). `replicates` may be any prefix of the
    B replicates, which is what the cost-against-replicates frontier
    needs. A coordinate with no converged replicate in `replicates`
    returns NaN, correctly: there is nothing to form an interval from.
    Returns (q, 2), [lo, hi].
    """
    replicates = np.asarray(replicates, dtype=float)
    alpha = 1.0 - level
    lo = np.nanquantile(replicates, alpha / 2.0, axis=0)
    hi = np.nanquantile(replicates, 1.0 - alpha / 2.0, axis=0)
    return np.stack([lo, hi], axis=-1)
