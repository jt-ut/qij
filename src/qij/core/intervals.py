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


def qij2_interval(theta_hat: float, bin_mass: np.ndarray, bin_influence: np.ndarray,
                   bin_d2T: np.ndarray, N: int, level: float,
                   R: int = 10000, seed: int = 0) -> np.ndarray:
    """
    The QIJ interval to second order, h^QIJ2_level, for ONE coordinate:
    recomputed entirely from that coordinate's stored bin constituents
    (`bin_mass` p_k, `bin_influence` U_k, `bin_d2T` d2T_k -- `L` of
    each, `study.py`'s `_qij_row` docstring says why these are kept),
    spending NO further estimator evaluations, same as `qij_interval`
    above. R draws of the bin counts from Multinomial(N, p) give R
    samples of the mass perturbation Delta_p_k = n_k/N - p_k and, from
    those, R samples of the quadratic surrogate

        theta_hat + Delta_p . U + 0.5 * (Delta_p**2) . d2T

    whose (1-level)/2 and (1+level)/2 empirical quantiles are this
    interval's [lo, hi].

    This differs from `qij_interval` in SHAPE, not only width:
    `qij_interval` is a normal-theory interval (BCa-adjusted by
    `a_bca`) built from one number, `V_tot_hat`; this one is the
    empirical distribution of a quadratic functional of a multinomial
    draw, so it can be skewed wherever the surrogate itself is.

    The property that makes this checkable: under Multinomial(N, p),
    E[Delta_p_k] = 0 and E[Delta_p_k^2] = p_k(1-p_k)/N, so the
    surrogate's MEAN is theta_hat + B_hat by construction --
    E[0.5 * sum_k d2T_k * Delta_p_k^2] = 0.5 * sum_k d2T_k *
    p_k(1-p_k)/N is exactly the between-bin curvature the method
    already reports as B_hat, recovered here from the surrogate's
    Monte Carlo mean instead of from the refinement's own bookkeeping.

    `bin_mass` is renormalized to sum to exactly 1 before the
    multinomial draw (`rng.multinomial` requires an exact simplex
    point, and `p_k = n_k/N` as stored in `qij.parquet` can be off by
    floating-point roundoff after a parquet round trip); the
    renormalized mass is used for BOTH the draw and `Delta_p`, so
    E[Delta_p] = 0 holds to the same precision as the draw itself, not
    to the precision of the stored `bin_mass`.

    Returns [nan, nan] -- never raises -- when the inputs cannot
    support an interval: any non-finite value in `bin_mass`,
    `bin_influence` or `bin_d2T`; an empty array; or a mass vector
    that does not sum to a positive number. A failed draw's stored
    constituents propagate as NaN here, the same convention as the
    rest of the method (plan section 4).
    """
    theta_hat = float(theta_hat)
    bin_mass = np.asarray(bin_mass, dtype=float)
    bin_influence = np.asarray(bin_influence, dtype=float)
    bin_d2T = np.asarray(bin_d2T, dtype=float)

    nan_pair = np.array([np.nan, np.nan])
    if bin_mass.size == 0 or bin_influence.size == 0 or bin_d2T.size == 0:
        return nan_pair
    if not (np.all(np.isfinite(bin_mass)) and np.all(np.isfinite(bin_influence))
            and np.all(np.isfinite(bin_d2T))):
        return nan_pair
    mass_total = float(bin_mass.sum())
    if not (mass_total > 0.0):
        return nan_pair

    p = bin_mass / mass_total

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(int(N), p, size=int(R))
    delta_p = counts / float(N) - p
    surrogate = (theta_hat + delta_p @ bin_influence
                 + 0.5 * (delta_p ** 2) @ bin_d2T)

    alpha = 1.0 - level
    lo = float(np.quantile(surrogate, alpha / 2.0))
    hi = float(np.quantile(surrogate, 1.0 - alpha / 2.0))
    return np.array([lo, hi])
