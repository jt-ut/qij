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


def qij_interval(theta_hat: np.ndarray, variance: np.ndarray,
                  a_bca: np.ndarray, level: float) -> np.ndarray:
    """
    The QIJ interval h^QIJ_level (glossary): the level*100% BCa-form
    interval with variance `variance` (V_tot_hat), acceleration
    `a_bca`, z0 = 0, centred at `theta_hat`. Each argument is (q,).
    Returns (q, 2), [lo, hi].
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    variance = np.asarray(variance, dtype=float)
    a_bca = np.asarray(a_bca, dtype=float)

    alpha = 1.0 - level
    z_lo = float(norm.ppf(alpha / 2.0))
    z_hi = float(norm.ppf(1.0 - alpha / 2.0))

    se = np.sqrt(np.maximum(variance, 0.0))
    adj_lo = z_lo / (1.0 - a_bca * z_lo)
    adj_hi = z_hi / (1.0 - a_bca * z_hi)

    lo = theta_hat + adj_lo * se
    hi = theta_hat + adj_hi * se
    return np.stack([lo, hi], axis=-1)


def percentile_interval(replicates: np.ndarray, level: float) -> np.ndarray:
    """
    The bootstrap's percentile interval h^boot_level (glossary): the
    level*100% interval from the empirical quantiles of `replicates`
    (b, q) over its replicate axis (axis 0). `replicates` may be any
    prefix of the B replicates, which is what the cost-against-
    replicates frontier needs. Returns (q, 2), [lo, hi].
    """
    replicates = np.asarray(replicates, dtype=float)
    alpha = 1.0 - level
    lo = np.quantile(replicates, alpha / 2.0, axis=0)
    hi = np.quantile(replicates, 1.0 - alpha / 2.0, axis=0)
    return np.stack([lo, hi], axis=-1)
