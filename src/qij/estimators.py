"""The paper's six estimators (plan Sec 4).

Every estimator is a callable `T(X, w) -> ndarray(q)` carrying `name`,
`outputs` and `eta`, and optionally `T.influence(X, w) -> ndarray(N, q)`
(the analytic influence at the fitted parameters, recorded for the oracle
variance and the points product only -- the method never calls it). `w` is
non-negative and sums to `len(X)`.

Three conventions hold for every estimator below: it never raises (a
failed fit returns NaN, of shape `(q,)`, which the study counts); it holds
no randomness and no state between calls; it does not know which method
is calling it -- the weight vector is the only difference between a
bootstrap replicate and a QIJ perturbation.

Closed forms (`pareto_shape`, `pareto_tail`, `mvt_tail`, `weighted_mean`)
are exact algebraic evaluations, so `eta` is machine precision. The
solved scalar (`mvt_nu`) and the vector closed form (`fp`) declare the
precision of their own solve. `IMF` is the one piece of genuinely new
code: a generalized Schechter fit above a fixed split tau, joined in
value and slope to a gamma below tau, with the gamma's two parameters
determined from the Schechter's three by the audited continuity solve
ported from `vqboot.imf_fitter._solve_gamma_constraints`.

Ported from `vqboot/src/vqboot/qij/estimators.py`, `vqboot/src/vqboot/
imf_fitter.py` and `vqboot/src/vqboot/dgp/`: the mathematics only. Dropped:
the `Estimand` protocol/class machinery, `check_scale_invariance` /
`validate_psi` (assertion-based, forbidden here), the DGP wrapper classes,
the IMF's joint (alpha, Mstar, p, tau) profile-likelihood search and its
warm-start cache / pool start / second start / `x0_extra` / `skip_data_
start` (tau is fixed, never chosen from the data, and every evaluation
depends only on its own `(X, w)`), and the lognormal low-mass alternative
(the paper uses gamma only). `IMF.influence` is new here and supersedes
the old `IMFDGPEstimand.psi`: the per-observation score is the same
object, but the Jacobian `A` is analytic rather than a central finite
difference of the total score, which needs the second-order
implicit-function term written out. The old `IMFDGPEstimand.psi` returned
zeros from a failed inner solve in two places (the `_solve_constraints`
check and the FP-pattern `A` default); both become the NaN-on-failure
convention here.
"""

from __future__ import annotations

import pathlib
from math import exp, log, lgamma, sqrt, pi

import numpy as np
from scipy import optimize, special
from scipy.special import digamma
from scipy.stats import f as f_dist

EPS = float(np.finfo(np.float64).eps)


def estimator(outputs, eta, name=None):
    """Attach `name`, `outputs`, `eta` to a closed-form estimator function.

    `name` defaults to the function's own name; the product vocabulary
    (plan Sec 3 / interface sheet Amendment 2) requires it be given
    explicitly wherever the Python identifier and the product name differ
    (`pareto_tail`/`mvt_tail` -> 'tail', `weighted_mean` -> 'mean').
    """
    def decorate(fn):
        fn.name = name if name is not None else fn.__name__
        fn.outputs = tuple(outputs)
        fn.eta = float(eta)
        return fn
    return decorate


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """The weighted q-quantile of x by linear interpolation of the weighted
    CDF. w need not be normalized. O(N log N), no loop over N."""
    order = np.argsort(x)
    xs = x[order]
    cdf = np.cumsum(w[order]) / w.sum()
    return float(np.interp(q, cdf, xs))


# ======================================================================
# Weighted mean
# ======================================================================

@estimator(outputs=('theta',), eta=EPS, name='mean')
def weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """T = sum(w X) / sum(w)."""
    try:
        x = np.asarray(X, dtype=float).reshape(-1)
        theta = float(np.dot(w, x) / w.sum())
        if not np.isfinite(theta):
            return np.full(1, np.nan)
        return np.array([theta])
    except Exception:
        return np.full(1, np.nan)


def _weighted_mean_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """psi_i = x_i - theta; A = 1."""
    theta = weighted_mean(X, w)
    if not np.all(np.isfinite(theta)):
        return np.full((len(X), 1), np.nan)
    x = np.asarray(X, dtype=float).reshape(-1)
    return (x - theta[0])[:, None]


weighted_mean.influence = _weighted_mean_influence


# ======================================================================
# Pareto shape (Hill estimator) and Pareto tail probability
# ======================================================================

PARETO_X_MIN = 1.0
PARETO_ALPHA_TRUE = 2.0
# Fixed threshold: the 99th percentile of the true Pareto(alpha=2.0, x_min=1.0)
# law. Frozen at this value, never recomputed from a draw.
PARETO_TAIL_C = PARETO_X_MIN * 0.01 ** (-1.0 / PARETO_ALPHA_TRUE)


@estimator(outputs=('alpha',), eta=EPS, name='shape')
def pareto_shape(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Hill estimator: alpha = sum(w) / sum(w log(X / x_min))."""
    try:
        x = np.asarray(X, dtype=float).reshape(-1)
        log_ratio = np.log(x / PARETO_X_MIN)
        alpha = float(w.sum() / np.dot(w, log_ratio))
        if not np.isfinite(alpha):
            return np.full(1, np.nan)
        return np.array([alpha])
    except Exception:
        return np.full(1, np.nan)


def _pareto_shape_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """psi_i = 1/alpha - log(x_i/x_min); A = 1/alpha^2; IF = psi/A."""
    theta = pareto_shape(X, w)
    if not np.all(np.isfinite(theta)):
        return np.full((len(X), 1), np.nan)
    alpha = theta[0]
    x = np.asarray(X, dtype=float).reshape(-1)
    psi = 1.0 / alpha - np.log(x / PARETO_X_MIN)
    return (psi * alpha ** 2)[:, None]


pareto_shape.influence = _pareto_shape_influence


@estimator(outputs=('P_tail',), eta=EPS, name='tail')
def pareto_tail(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """T = sum(w 1{X > c}) / sum(w), c fixed at the true 99th percentile."""
    try:
        x = np.asarray(X, dtype=float).reshape(-1)
        indicator = (x > PARETO_TAIL_C).astype(float)
        theta = float(np.dot(w, indicator) / w.sum())
        return np.array([theta])
    except Exception:
        return np.full(1, np.nan)


def _pareto_tail_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """psi_i = 1{x_i > c} - theta; A = 1."""
    theta = pareto_tail(X, w)
    if not np.all(np.isfinite(theta)):
        return np.full((len(X), 1), np.nan)
    x = np.asarray(X, dtype=float).reshape(-1)
    indicator = (x > PARETO_TAIL_C).astype(float)
    return (indicator - theta[0])[:, None]


pareto_tail.influence = _pareto_tail_influence


# ======================================================================
# MVT nu (solved scalar) and MVT tail probability
# ======================================================================

MVT_D = 10
MVT_NU_TRUE = 5.0
# Fixed threshold: the 99th percentile of ||x|| under the true d=10, nu=5.0 law.
MVT_TAIL_C = float(np.sqrt(MVT_D * f_dist.ppf(0.99, MVT_D, MVT_NU_TRUE)))

_MVT_NU_LO, _MVT_NU_HI = 0.5, 200.0
_MVT_NU_MIN, _MVT_NU_MAX = 0.1, 1.0e6
_MVT_NU_EXT_FACTOR = 4.0


def _mvt_score_terms(r_sq: np.ndarray, nu: float, d: int):
    """(g, h) with psi_i = g + h_i, the per-observation profile score for nu."""
    half_nu = nu / 2.0
    half_nu_d = (nu + d) / 2.0
    g = (digamma(half_nu) - np.log(half_nu)
         - digamma(half_nu_d) + np.log(half_nu_d))
    h = np.log((nu + r_sq) / (nu + d)) - (nu + d) / (nu + r_sq) + 1.0
    return float(g), h


def _mvt_score_weighted(r_sq: np.ndarray, nu: float, w: np.ndarray, d: int) -> float:
    g, h = _mvt_score_terms(r_sq, nu, d)
    return g + float(np.dot(w, h)) / float(w.sum())


@estimator(outputs=('nu',), eta=1e-12, name='nu')
def mvt_nu(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """One-dimensional root-find of the weighted profile score in log(nu).

    Brackets [_MVT_NU_LO, _MVT_NU_HI] in log(nu), extending geometrically to
    [_MVT_NU_MIN, _MVT_NU_MAX] when the nominal bracket does not contain a
    root; a one-signed score over the whole admissible range returns the
    appropriate cap rather than failing.
    """
    try:
        d = X.shape[1]
        r_sq = np.sum(np.asarray(X, dtype=float) ** 2, axis=1)
        eta = mvt_nu.eta

        def score_log(log_nu):
            return _mvt_score_weighted(r_sq, float(np.exp(log_nu)), w, d)

        log_lo, log_hi = np.log(_MVT_NU_LO), np.log(_MVT_NU_HI)
        log_nu_min, log_nu_max = np.log(_MVT_NU_MIN), np.log(_MVT_NU_MAX)
        log_ext = np.log(_MVT_NU_EXT_FACTOR)

        s_lo, s_hi = score_log(log_lo), score_log(log_hi)
        while s_lo * s_hi > 0.0 and log_hi < log_nu_max:
            log_hi = min(log_hi + log_ext, log_nu_max)
            s_hi = score_log(log_hi)
        while s_lo * s_hi > 0.0 and log_lo > log_nu_min:
            log_lo = max(log_lo - log_ext, log_nu_min)
            s_lo = score_log(log_lo)

        if s_lo * s_hi > 0.0:
            nu_hat = _MVT_NU_MAX if s_hi < 0.0 else _MVT_NU_MIN
        else:
            log_nu_hat = optimize.brentq(score_log, log_lo, log_hi, xtol=eta)
            nu_hat = float(np.exp(log_nu_hat))

        if not np.isfinite(nu_hat):
            return np.full(1, np.nan)
        return np.array([nu_hat])
    except Exception:
        return np.full(1, np.nan)


def _mvt_nu_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """psi_i = g + h_i at nu_hat; A = -d/dnu E_w[psi] by central difference."""
    theta = mvt_nu(X, w)
    if not np.all(np.isfinite(theta)):
        return np.full((len(X), 1), np.nan)
    nu_hat = theta[0]
    d = X.shape[1]
    r_sq = np.sum(np.asarray(X, dtype=float) ** 2, axis=1)
    g, h = _mvt_score_terms(r_sq, nu_hat, d)
    psi = g + h
    delta = 1e-4 * nu_hat
    s_hi = _mvt_score_weighted(r_sq, nu_hat + delta, w, d)
    s_lo = _mvt_score_weighted(r_sq, nu_hat - delta, w, d)
    A = -(s_hi - s_lo) / (2.0 * delta)
    if not np.isfinite(A) or abs(A) < 1e-300:
        return np.full((len(X), 1), np.nan)
    return (psi / A)[:, None]


mvt_nu.influence = _mvt_nu_influence


@estimator(outputs=('P_tail',), eta=EPS, name='tail')
def mvt_tail(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """T = sum(w 1{||x|| > c}) / sum(w), c fixed at the true 99th percentile."""
    try:
        r = np.sqrt(np.sum(np.asarray(X, dtype=float) ** 2, axis=1))
        indicator = (r > MVT_TAIL_C).astype(float)
        theta = float(np.dot(w, indicator) / w.sum())
        return np.array([theta])
    except Exception:
        return np.full(1, np.nan)


def _mvt_tail_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    theta = mvt_tail(X, w)
    if not np.all(np.isfinite(theta)):
        return np.full((len(X), 1), np.nan)
    r = np.sqrt(np.sum(np.asarray(X, dtype=float) ** 2, axis=1))
    indicator = (r > MVT_TAIL_C).astype(float)
    return (indicator - theta[0])[:, None]


mvt_tail.influence = _mvt_tail_influence


# ======================================================================
# Fundamental Plane: weighted total least squares
# ======================================================================

_FP_DATA_PATH = pathlib.Path(__file__).resolve().parent / 'data' / 'fp_sdss.npz'
with np.load(_FP_DATA_PATH) as _fp_npz:
    _fp_pool_data = _fp_npz['fp_data']          # (N_pop, 3), raw log-space
# Population standardization, frozen once at import time -- never recomputed
# per draw (plan Sec 4).
FP_POP_MEAN = _fp_pool_data.mean(axis=0)
FP_POP_STD = _fp_pool_data.std(axis=0, ddof=0)
del _fp_pool_data


def _fp_standardize(X: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=float) - FP_POP_MEAN) / FP_POP_STD


def _fp_tls_fit(X_std: np.ndarray, w: np.ndarray) -> dict:
    """Weighted TLS plane via one eigendecomposition of the weighted
    covariance. w normalized to sum 1. O(N) plus one 3x3 eigendecomposition,
    never O(N) per candidate direction."""
    mu = X_std.T @ w
    centered = X_std - mu
    S = (centered * w[:, None]).T @ centered
    eigvals_asc, eigvecs_asc = np.linalg.eigh(S)
    idx = np.argsort(eigvals_asc)[::-1]
    eigvals = eigvals_asc[idx]
    eigvecs = eigvecs_asc[:, idx]
    if eigvecs[2, 2] < 0:
        eigvecs[:, 2] *= -1
    v3 = eigvecs[:, 2]
    n_sig, n_I, n_R = float(v3[0]), float(v3[1]), float(v3[2])
    d = float(v3 @ mu)
    a = -n_sig / n_R
    b = -n_I / n_R
    c = d / n_R
    scatter = float(np.sqrt(max(eigvals[2], 0.0)))
    return dict(a=a, b=b, c=c, scatter=scatter, mu=mu, eigvals=eigvals,
                eigvecs=eigvecs, v3=v3, d=d, lambda3=float(eigvals[2]))


@estimator(outputs=('a', 'b', 'c', 'scatter'), eta=1e-12, name='fp')
def fp(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    try:
        X_std = _fp_standardize(X)
        wn = w / w.sum()
        fit = _fp_tls_fit(X_std, wn)
        return np.array([fit['a'], fit['b'], fit['c'], fit['scatter']])
    except Exception:
        return np.full(4, np.nan)


def _fp_compute_if(X_std: np.ndarray, fit: dict) -> np.ndarray:
    """Closed-form implicit-function influence (N, 4), ported from
    vqboot.dgp.fundamental_plane.FundamentalPlaneDGP._compute_if."""
    n = len(X_std)
    mu, eigvals, eigvecs, v3 = fit['mu'], fit['eigvals'], fit['eigvecs'], fit['v3']
    d, lambda3, scatter = fit['d'], fit['lambda3'], fit['scatter']
    n_sig, n_I, n_R = float(v3[0]), float(v3[1]), float(v3[2])

    centered = X_std - mu
    r = centered @ eigvecs

    IF_v3 = np.zeros((n, 3))
    for m in range(2):
        denom = eigvals[2] - eigvals[m]
        coeff = r[:, m] * r[:, 2] / denom
        IF_v3 += coeff[:, None] * eigvecs[:, m]

    IF_d = r[:, 2] + IF_v3 @ mu
    IF_lam3 = r[:, 2] ** 2 - lambda3

    IF_a = -(1.0 / n_R) * IF_v3[:, 0] + (n_sig / n_R ** 2) * IF_v3[:, 2]
    IF_b = -(1.0 / n_R) * IF_v3[:, 1] + (n_I / n_R ** 2) * IF_v3[:, 2]
    IF_c = (1.0 / n_R) * IF_d - (d / n_R ** 2) * IF_v3[:, 2]
    IF_s = IF_lam3 / (2.0 * scatter) if scatter > 1e-12 else np.zeros(n)

    return np.column_stack([IF_a, IF_b, IF_c, IF_s])


def _fp_influence(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    try:
        X_std = _fp_standardize(X)
        wn = w / w.sum()
        fit = _fp_tls_fit(X_std, wn)
        return _fp_compute_if(X_std, fit)
    except Exception:
        return np.full((len(X), 4), np.nan)


fp.influence = _fp_influence


# ======================================================================
# IMF: gamma below tau, generalized Schechter above, joined by continuity
# ======================================================================

_EPS_MASS = np.finfo(np.float64).tiny
_DENSITY_FLOOR = 1e-12
_NLL_PENALTY = 1e10
_SQRT2PI = sqrt(2.0 * pi)


def _gamma_density_scalar(mass: float, shape: float, scale: float) -> float:
    if shape <= 0 or scale <= 0 or mass <= 0:
        return 0.0
    try:
        log_density = ((shape - 1.0) * log(mass) - mass / scale
                        - shape * log(scale) - lgamma(shape))
        return exp(log_density)
    except (ValueError, OverflowError):
        return 0.0


def _gamma_deriv_scalar(mass: float, shape: float, scale: float) -> float:
    f = _gamma_density_scalar(mass, shape, scale)
    if f <= 0 or mass <= 0:
        return 0.0
    return f * ((shape - 1.0) / mass - 1.0 / scale)


def gamma_density(masses: np.ndarray, shape: float, scale: float) -> np.ndarray:
    if shape <= 0 or scale <= 0:
        return np.zeros_like(masses, dtype=float)
    masses = np.maximum(masses, _EPS_MASS)
    return (masses ** (shape - 1) * np.exp(-masses / scale)
            / (scale ** shape * special.gamma(shape)))


def _upper_incomplete_gamma(s: float, x: float) -> float:
    """Upper incomplete gamma Gamma(s, x), x > 0, via the shift recurrence
    into scipy's gammaincc domain (s > 0)."""
    if not (np.isfinite(s) and np.isfinite(x) and x > 0):
        return np.nan
    n_shifts = max(0, int(np.floor(-s)) + 1) if s <= 0 else 0
    s_base = s + n_shifts
    val = special.gammaincc(s_base, x) * special.gamma(s_base)
    for i in range(n_shifts):
        s_curr = s_base - i - 1
        if s_curr == 0.0:
            val = special.exp1(x)
        else:
            val = (val - np.exp(s_curr * np.log(x) - x)) / s_curr
    return val


def _schechter_raw(masses: np.ndarray, alpha: float, Mstar: float, p: float) -> np.ndarray:
    ratio = masses / Mstar
    return (ratio ** alpha) * np.exp(-(ratio ** p)) / Mstar


def _schechter_norm(alpha: float, Mstar: float, p: float, tau: float) -> float:
    """Gamma((alpha+1)/p, (tau/Mstar)^p) / p, the analytic truncated norm."""
    if not (Mstar > 0 and p > 0 and tau > 0):
        return 1.0
    s = (alpha + 1.0) / p
    x = (tau / Mstar) ** p
    try:
        val = _upper_incomplete_gamma(s, x) / p
        if np.isfinite(val) and val > 0:
            return val
    except Exception:
        pass
    return 1.0


def schechter_density(masses: np.ndarray, alpha: float, Mstar: float,
                       p: float, tau: float) -> np.ndarray:
    masses = np.maximum(masses, _EPS_MASS)
    norm = _schechter_norm(alpha, Mstar, p, tau)
    return _schechter_raw(masses, alpha, Mstar, p) / norm


def schechter_deriv(mass: float, alpha: float, Mstar: float, p: float, tau: float) -> float:
    f = float(schechter_density(np.atleast_1d(mass), alpha, Mstar, p, tau)[0])
    ratio = mass / Mstar
    return f * (alpha / mass - p * ratio ** (p - 1) / Mstar)


def _solve_gamma_constraints(tau: float, target_f: float, target_fp: float,
                              shape_bounds: tuple) -> tuple:
    """The audited continuity solve, ported verbatim (up to the unused
    `masses_low` argument) from vqboot.imf_fitter._solve_gamma_constraints:
    a BRACKETED ROOT-FIND on the signed relative residual of the gamma
    density at tau, not a minimization of a squared residual, with the
    feasible shape floor in closed form. Returns (shape, scale, converged).
    """
    if abs(target_f) < _DENSITY_FLOOR:
        return 2.0, 0.5, False

    dratio = target_fp / target_f

    def _scale_from_shape(shape):
        denom = (shape - 1.0) / tau - dratio
        return 1.0 / denom if denom > 0 else None

    def _residual(shape):
        sc = _scale_from_shape(shape)
        if sc is None or sc != sc or sc <= 0:
            return None
        pred = _gamma_density_scalar(tau, shape, sc)
        if pred <= 0 or pred != pred:
            return None
        return pred / target_f - 1.0

    # scale(shape) > 0 only for shape > 1 + tau * dratio; nudge strictly
    # inside that feasible floor, and take the tighter of it and shape_bounds[0].
    feasible_lo = 1.0 + tau * dratio
    nudge = 1e-9 * max(abs(feasible_lo), 1.0)
    lo = max(shape_bounds[0], feasible_lo + nudge)
    hi = shape_bounds[1]
    if lo >= hi:
        return 2.0, 0.5, False

    grid = np.geomspace(lo, hi, 200)
    prev_shape, prev_val = None, None
    bracket = None
    for s in grid:
        val = _residual(float(s))
        if val is None:
            prev_shape, prev_val = None, None
            continue
        if val == 0.0:
            bracket = (float(s), float(s))
            break
        if prev_val is not None and (prev_val < 0) != (val < 0):
            bracket = (prev_shape, float(s))
            break
        prev_shape, prev_val = float(s), val

    if bracket is None:
        return 2.0, 0.5, False

    if bracket[0] == bracket[1]:
        shape_root = bracket[0]
    else:
        try:
            shape_root = optimize.brentq(_residual, bracket[0], bracket[1],
                                          xtol=1e-14, rtol=1e-14, maxiter=200)
        except Exception:
            return 2.0, 0.5, False

    sc = _scale_from_shape(shape_root)
    if sc is not None and sc > 0:
        return shape_root, sc, True
    return 2.0, 0.5, False


def _imf_continuity(tau: float, alpha: float, Mstar: float, p: float,
                     shape_bounds: tuple) -> tuple:
    """Given the Schechter's (alpha, Mstar, p) at the fixed split tau,
    recover the gamma's (shape, scale) from C0+C1 continuity at tau."""
    try:
        target_f = float(schechter_density(np.atleast_1d(tau), alpha, Mstar, p, tau)[0])
        target_fp = schechter_deriv(tau, alpha, Mstar, p, tau)
    except Exception:
        return np.nan, np.nan, False
    if not np.isfinite(target_f) or target_f <= 0:
        return np.nan, np.nan, False
    return _solve_gamma_constraints(tau, target_f, target_fp, shape_bounds)


def _imf_joint_density(masses: np.ndarray, shape: float, scale: float,
                        alpha: float, Mstar: float, p: float, tau: float) -> np.ndarray:
    """The C0-continuous piecewise density: gamma below tau, generalized
    Schechter (rescaled to match the gamma at tau) above."""
    lo_at_tau = float(gamma_density(np.atleast_1d(tau), shape, scale)[0])
    hi_at_tau = float(schechter_density(np.atleast_1d(tau), alpha, Mstar, p, tau)[0])
    c_rel = lo_at_tau / max(hi_at_tau, _DENSITY_FLOOR)

    low_mask = masses <= tau
    density = np.empty_like(masses, dtype=float)
    if low_mask.any():
        density[low_mask] = gamma_density(masses[low_mask], shape, scale)
    if (~low_mask).any():
        density[~low_mask] = c_rel * schechter_density(masses[~low_mask], alpha, Mstar, p, tau)
    return density


class IMF:
    """The two-regime IMF estimator at a fixed split tau (plan Sec 4).

    A generalized Schechter density above tau, free in (alpha, Mstar, p);
    a gamma density below tau, whose (shape, scale) are determined from
    the Schechter's three parameters by the audited continuity solve
    (`_solve_gamma_constraints`, a bracketed root-find, not a minimization).
    tau is fixed at construction, never chosen from a draw. Every call
    depends only on its own (X, w): one bounded weighted-NLL minimization
    from a single method-of-moments start -- no warm-start cache, no pool
    start, no second start, no analytic influence.
    """

    name = 'imf'
    outputs = ('slope', 'Mstar', 'p', 'gamma_shape', 'gamma_scale')
    eta = 1e-6   # sqrt of the objective (NLL) tolerance

    _SHAPE_BOUNDS = (1.1, 50.0)

    def __init__(self, tau: float,
                 bounds=((-10.0, 10.0), (1.0, 200.0), (0.1, 20.0))):
        self.tau = float(tau)
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)

    def _at_box(self, params: np.ndarray) -> bool:
        """True when any of the three free (alpha, Mstar, p) parameters
        lies within `eta` -- the estimator's own declared accuracy, 1e-6 --
        of either end of its own bound interval. A fit that runs to a
        bound is not a stationary point of the objective (the weighted
        score there is not zero), so it is not a root of anything and
        neither the fit nor its influence means anything there (plan
        Sec 4 ruling): the box stays, but the evaluation becomes a
        counted failure like any other, not a silent number."""
        for v, (lo, hi) in zip(params, self.bounds):
            if abs(v - lo) <= self.eta * (1.0 + abs(lo)):
                return True
            if abs(v - hi) <= self.eta * (1.0 + abs(hi)):
                return True
        return False

    def _moment_start(self, m_hi: np.ndarray, w_hi: np.ndarray,
                       masses: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Method-of-moments start, computed once from (X, w) (plan Sec 4):
        the high-mass slope from the weighted Hill estimator above tau; the
        cutoff mass from the weighted 99th percentile; the sharpness from
        the ratio of the weighted 99th to 90th percentile, or the model
        default 2.0 when that ratio is not usable. (The plan also names a
        gamma shape/scale moment start -- mean^2/variance, variance/mean,
        of the low-mass points -- but the gamma is not a free parameter of
        this objective: continuity determines it from (alpha, Mstar, p),
        so that moment pair has no x0 component to seed and is not
        computed here; see the coding-manager report.)

        Coding-manager correction to Sec 4 (interface sheet Amendment 5):
        the plan's sentence gives the slope start as 1/weighted_mean(log(m/tau)),
        but that is the Hill tail index of the SURVIVAL function, not the
        exponent of the Schechter DENSITY this objective is parameterized
        by. Above tau and below the cutoff the density goes as m^slope, so
        the survival function goes as m^(slope+1); inverting the Hill tail
        index xi = 1/weighted_mean(log(m/tau)) = -(slope+1) gives
        slope0 = -1/weighted_mean(log(m/tau)) - 1. The uninverted formula
        clips to the bound and carries no information from the data (e.g.
        +0.904 on the shipped IMF pool, against a fitted slope of -1.558);
        the corrected one lands at -1.904.
        """
        wh_sum = w_hi.sum()
        log_ratio = np.log(m_hi / self.tau)
        hill = float(np.dot(w_hi, log_ratio) / wh_sum)
        slope0 = -1.0 / hill - 1.0 if hill > 0 else 0.0

        Mstar0 = _weighted_quantile(masses, w, 0.99)

        q_hi = _weighted_quantile(masses, w, 0.99)
        q_lo = _weighted_quantile(masses, w, 0.90)
        ratio = q_hi / q_lo if q_lo > 0 else np.nan
        p0 = ratio if np.isfinite(ratio) and ratio > 0 else 2.0

        return np.array([slope0, Mstar0, p0])

    def __call__(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        try:
            masses = np.asarray(X, dtype=float).reshape(-1)
            tau = self.tau
            lo_mask = masses <= tau
            n_lo = int(lo_mask.sum())
            n_hi = int((~lo_mask).sum())
            if n_lo < 3 or n_hi < 3:
                return np.full(5, np.nan)

            m_hi, w_hi = masses[~lo_mask], w[~lo_mask]

            x0 = self._moment_start(m_hi, w_hi, masses, w)
            bounds_lo = np.array([b[0] for b in self.bounds])
            bounds_hi = np.array([b[1] for b in self.bounds])
            x0 = np.clip(x0, bounds_lo, bounds_hi)

            shape_bounds = self._SHAPE_BOUNDS

            def nll(params):
                alpha, Mstar, p = params
                if Mstar <= 0 or p <= 0:
                    return _NLL_PENALTY
                shape, scale, converged = _imf_continuity(tau, alpha, Mstar, p, shape_bounds)
                if not converged:
                    return _NLL_PENALTY
                density = _imf_joint_density(masses, shape, scale, alpha, Mstar, p, tau)
                density = np.maximum(density, _DENSITY_FLOOR)
                return -float(np.dot(w, np.log(density)))

            res = optimize.minimize(
                nll, x0, method='L-BFGS-B', bounds=self.bounds,
                options={'maxiter': 400, 'ftol': self.eta ** 2, 'gtol': 1e-8},
            )
            if not res.success:
                return np.full(5, np.nan)

            if self._at_box(res.x):
                return np.full(5, np.nan)

            alpha, Mstar, p = (float(v) for v in res.x)
            shape, scale, converged = _imf_continuity(tau, alpha, Mstar, p, shape_bounds)
            if not converged:
                return np.full(5, np.nan)
            return np.array([alpha, Mstar, p, shape, scale])
        except Exception:
            return np.full(5, np.nan)

    def influence(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """The analytic influence (N, 5) at the fitted parameters.

        `IF = A^-1 psi` with `psi` the per-observation score of the
        objective `__call__` minimizes and `A` the negative Hessian of the
        weighted-mean log density, the same M-estimator construction as
        `_pareto_shape_influence` and `_mvt_nu_influence`; the two derived
        outputs follow by the delta method, `IF_lam = J IF_theta`.
        NaN (N, 5) whenever the fit, the continuity solve or the solve
        against `A` fails, and also whenever the fit is at its box: a
        non-root evaluation (plan Sec 4 ruling) makes the influence there
        meaningless. O(N) arithmetic in one vectorized pass plus O(1)
        work, never a loop over the N masses.
        """
        try:
            theta = self(X, w)
            if not np.all(np.isfinite(theta)) or self._at_box(theta[:3]):
                return np.full((len(X), 5), np.nan)
            masses = np.asarray(X, dtype=float).reshape(-1)
            alpha, Mstar, p, shape, scale = (float(v) for v in theta)
            psi, A, J = _imf_score_hessian(
                masses, np.asarray(w, dtype=float), self.tau,
                alpha, Mstar, p, shape, scale)
            IF_free = np.linalg.solve(A, psi.T).T
            IF = np.hstack([IF_free, IF_free @ J.T])
            if not np.all(np.isfinite(IF)):
                return np.full((len(X), 5), np.nan)
            return IF
        except Exception:
            return np.full((len(X), 5), np.nan)


# Composite Gauss-Legendre rule for the incomplete gamma's order derivatives
# (`_gamma_order_moments`): panels x nodes, and the log-scale margin outside
# which the integrand is below exp(-_QUAD_MARGIN) of its peak.
_QUAD_Z, _QUAD_W = np.polynomial.legendre.leggauss(32)
_QUAD_PANELS = 8
_QUAD_MARGIN = 60.0


def _gamma_order_moments(a: float, x: float) -> tuple:
    """(G_a/G, G_aa/G, x G_x/G) for the upper incomplete gamma G = Gamma(a, x).

    The derivatives in the ORDER,

        G_a  = d Gamma(a,x)/da = int_x^inf t^(a-1) e^-t log(t)   dt
        G_aa =                   int_x^inf t^(a-1) e^-t log(t)^2 dt,

    have no elementary closed form: truncating at x breaks the Mellin
    identity that gives d Gamma(a)/da = Gamma(a) digamma(a); the order
    recurrence bottoms out on the exponential integral E_1 at a = 1 and
    has no base case at all for non-integer a; the known closed form is a
    Meijer G-function. `scipy.special` differentiates `gammaincc` in x
    only. **This one quantity is therefore evaluated by QUADRATURE ON ITS
    OWN DEFINING INTEGRAL** -- an exact evaluation of the true object, not
    a finite difference of the estimator, of `gammaincc`, or of anything
    else. Under t = exp(z),

        int_x^inf t^(a-1) e^-t log(t)^k dt = int_{log x}^inf e^phi(z) z^k dz,
        phi(z) = a z - e^z,

    so one set of nodes serves k = 0, 1, 2. phi is strictly concave, with
    its maximum at log(a) when a > 0 and log(a) > log(x) and at log(x)
    otherwise; the range is trimmed by bisection to where phi exceeds its
    maximum less `_QUAD_MARGIN` and covered by composite Gauss-Legendre.
    Needed at exactly one (a, x) per fit -- O(1), outside every
    per-observation and per-bin path. Everything returned is a ratio to G,
    formed as exp(phi - phi_max), so no intermediate overflows however
    large G is, and the x-scaled `x G_x/G` stays O(max(|a|, x)) even where
    x underflows toward zero.
    """
    z_min = log(x)
    z_peak = max(z_min, log(a)) if a > 0.0 else z_min
    phi_max = a * z_peak - exp(z_peak)
    cut = phi_max - _QUAD_MARGIN

    lo, hi = z_min, z_peak
    if a * z_min - exp(z_min) < cut:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if a * mid - exp(mid) < cut:
                lo = mid
            else:
                hi = mid
    z_lo = lo

    span = 1.0
    for _ in range(10):
        if a * (z_peak + span) - exp(z_peak + span) <= cut:
            break
        span *= 2.0
    lo, hi = z_peak, z_peak + span
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if a * mid - exp(mid) > cut:
            lo = mid
        else:
            hi = mid
    z_hi = hi

    edges = np.linspace(z_lo, z_hi, _QUAD_PANELS + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    z = mid[:, None] + half[:, None] * _QUAD_Z[None, :]
    g = np.exp(a * z - np.exp(z) - phi_max) * (half[:, None] * _QUAD_W[None, :])
    i0 = float(g.sum())
    i1 = float((g * z).sum())
    i2 = float((g * z * z).sum())
    return i1 / i0, i2 / i0, -exp(a * z_min - x - phi_max) / i0


def _imf_score_hessian(masses: np.ndarray, w: np.ndarray, tau: float,
                        alpha: float, Mstar: float, p: float,
                        shape: float, scale: float) -> tuple:
    """(psi (N, 3), A (3, 3), J (2, 3)) for the IMF at its fitted parameters.

    The objective `IMF.__call__` minimizes is, once the C0 continuity
    condition is substituted back (`c_rel` is 1 identically in theta, so
    the high-mass log density is exactly the truncated Schechter's),

        W loglik = sum_{m<=tau} w log gamma_pdf(m; lam(theta))
                 + sum_{m> tau} w [ log f_raw(m; theta) - log norm(theta) ],

    with lam = (shape, scale) and norm = Gamma((alpha+1)/p, (tau/Mstar)^p)/p.
    The normalization does NOT cancel: it cancels inside `c_rel * schechter`,
    but continuity pins gamma_pdf(tau) to f_raw(tau)/norm and puts it back.
    So the low-mass block reaches theta only through lam, the high-mass
    block not at all, and

        psi_lo = J' grad_lam log gamma_pdf(m)
        psi_hi = grad_theta [ log f_raw(m) - log norm ]
        A      = -(1/W) [ sum_lo w (J' H_lam J + sum_u grad_lam,u K_u)
                        + sum_hi w Hess_theta (log f_raw(m) - log norm) ]

    where J = d lam/d theta and K_u = d2 lam_u/d theta2 come from the
    implicit function theorem on the two continuity conditions. Both are
    additively separable, h_r = A_r(lam) - B_r(theta) -- the C1 condition
    because `norm` cancels from f_raw'(tau)/f_raw(tau), the C0 condition
    in its log form, whose zero set is the ratio form the bracketed solve
    drives to zero -- so every mixed second partial d2h/dlam dtheta
    vanishes and the second-order term collapses to

        A_lam K_{:,ab} = [ B_r,ab - (J' A_r,lamlam J)_ab ]_{r=1,2}.

    B_1 = f_raw'(tau)/f_raw(tau) and B_2 = log(f_raw(tau)/norm), the
    latter being the high-mass log density at m = tau, so it shares the
    high-mass gradient and Hessian code exactly.

    Derivatives of x = (tau/Mstar)^p are carried reduced by x and those of
    Gamma scaled by G, which keeps every intermediate O(1) where x reaches
    1e-70 and G_x/G reaches 1e+70. O(N) in one vectorized pass over the
    masses plus O(1) work; the quadrature of `_gamma_order_moments` runs
    once per call.
    """
    L = log(tau / Mstar)
    x = exp(p * L)
    order = (alpha + 1.0) / p
    r1, r2, cx = _gamma_order_moments(order, x)

    # x-reduced derivatives of x (X_c = dx/dtheta_c / x) and derivatives of
    # the incomplete gamma's order.
    X = np.array([0.0, -p / Mstar, L])
    XH = np.array([[0.0, 0.0, 0.0],
                   [0.0, p * (p + 1.0) / Mstar ** 2, -(1.0 + p * L) / Mstar],
                   [0.0, -(1.0 + p * L) / Mstar, L * L]])
    Ord = np.array([1.0 / p, 0.0, -order / p])
    OrdH = np.array([[0.0, 0.0, -1.0 / p ** 2],
                     [0.0, 0.0, 0.0],
                     [-1.0 / p ** 2, 0.0, 2.0 * order / p ** 2]])

    # log norm = log Gamma(order, x) - log p, its gradient and Hessian.
    s_oo = r2 - r1 * r1
    s_ox = cx * (log(x) - r1)
    s_xx = cx * (order - 1.0 - x) - cx * cx
    g_norm = r1 * Ord + cx * X
    g_norm[2] -= 1.0 / p
    H_norm = (s_oo * np.outer(Ord, Ord)
              + s_ox * (np.outer(Ord, X) + np.outer(X, Ord))
              + s_xx * np.outer(X, X)
              + r1 * OrdH + cx * XH)
    H_norm[2, 2] += 1.0 / p ** 2

    def hi_grad(m):
        """grad_theta [log f_raw(m) - log norm], (k, 3)."""
        ell = np.log(m / Mstar)
        u = np.exp(p * ell)
        return np.column_stack([ell - g_norm[0],
                                (p * u - alpha - 1.0) / Mstar - g_norm[1],
                                -u * ell - g_norm[2]])

    def hi_hess(m, wt):
        """sum_k wt_k Hess_theta [log f_raw(m_k) - log norm], (3, 3)."""
        ell = np.log(m / Mstar)
        u = np.exp(p * ell)
        s_w = float(wt.sum())
        s_u = float(np.dot(wt, u))
        s_ul = float(np.dot(wt, u * ell))
        s_ull = float(np.dot(wt, u * ell * ell))
        H = np.zeros((3, 3))
        H[0, 1] = H[1, 0] = -s_w / Mstar
        H[1, 1] = (s_w * (alpha + 1.0) - p * (p + 1.0) * s_u) / Mstar ** 2
        H[1, 2] = H[2, 1] = (s_u + p * s_ul) / Mstar
        H[2, 2] = -s_ull
        return H - s_w * H_norm

    tau_arr = np.full(1, tau)
    B2_g = hi_grad(tau_arr)[0]
    B2_H = hi_hess(tau_arr, np.ones(1))

    e_p = np.array([0.0, 0.0, 1.0])
    B1_g = np.array([1.0 / tau,
                     -(p / tau) * x * X[1],
                     -(x / tau) * (1.0 + p * L)])
    B1_H = -(1.0 / tau) * (p * x * XH
                           + x * (np.outer(e_p, X) + np.outer(X, e_p)))

    # The continuity conditions' lam-side: A_1 = (shape-1)/tau - 1/scale,
    # A_2 = log gamma_pdf(tau; shape, scale).
    trigamma = float(special.polygamma(1, shape))
    dg_shape = log(tau) - log(scale) - digamma(shape)
    dg_scale = tau / scale ** 2 - shape / scale
    A_lam = np.array([[1.0 / tau, 1.0 / scale ** 2],
                      [dg_shape, dg_scale]])
    A1_ll = np.array([[0.0, 0.0], [0.0, -2.0 / scale ** 3]])
    A2_ll = np.array([[-trigamma, -1.0 / scale],
                      [-1.0 / scale, -2.0 * tau / scale ** 3 + shape / scale ** 2]])

    J = np.linalg.solve(A_lam, np.vstack([B1_g, B2_g]))
    rhs = np.stack([B1_H - J.T @ A1_ll @ J, B2_H - J.T @ A2_ll @ J])
    K = np.linalg.solve(A_lam, rhs.reshape(2, 9)).reshape(2, 3, 3)

    low = masses <= tau
    m_lo, w_lo = masses[low], w[low]
    g_shape = np.log(m_lo) - log(scale) - digamma(shape)
    g_scale = m_lo / scale ** 2 - shape / scale

    psi = np.empty((masses.size, 3))
    psi[low] = np.column_stack([g_shape, g_scale]) @ J
    psi[~low] = hi_grad(masses[~low])

    s_w = float(w_lo.sum())
    s_m = float(np.dot(w_lo, m_lo))
    H_lam = np.array([[-s_w * trigamma, -s_w / scale],
                      [-s_w / scale,
                       -2.0 * s_m / scale ** 3 + s_w * shape / scale ** 2]])
    g_lam = np.array([float(np.dot(w_lo, g_shape)), float(np.dot(w_lo, g_scale))])
    H = (J.T @ H_lam @ J + g_lam[0] * K[0] + g_lam[1] * K[1]
         + hi_hess(masses[~low], w[~low]))
    return psi, -H / float(w.sum()), J
