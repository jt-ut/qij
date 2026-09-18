"""
The initial influence estimate (QIJ_method_outline.md step 4): a
Gaussian process with an affine mean and a Matern-3/2 kernel through
the pairs (w_j, I_j) in whitened coordinates, unweighted noise --

    I_proto(w_j) = h(w_j)^T beta + f(w_j) + e_j,   f ~ GP(0, s^2 k_ell)
    k_ell(r) = (1 + sqrt(3) r / ell) exp(-sqrt(3) r / ell)   (Matern 3/2,
               r the whitened distance)
    e_j ~ N(0, s^2 lam)                                      (unweighted
               noise: every prototype influence is an equally exact
               derivative)

h(x) = (1, x) in whitened coordinates (m = d_z + 1 basis functions),
constant-only (m = 1) when the design has too few points (<= d_z + 2).

**A failed prototype evaluation is a missing response, per coordinate
(ruled 18 September; plan §4).** `I_proto` (M_used, q) carries NaN at
(j, c) wherever the estimator's c-th output at prototype j could not be
evaluated -- left there by `xvq.prototype_influences`, not filled in and
not dropped, because a vector estimator can fail in a way that leaves
some outputs finite and others not for the same prototype. For each
coordinate c, this module regresses only on the prototypes where
I_proto[:, c] is finite: their whitened positions, their masses and
their responses. A Gaussian process is indifferent to which design
points it was given, so `psi0`, `uncertainty` and
`bin_posterior_variance` are unaffected in kind -- they still predict at
every row of Z -- only the design each coordinate's GP was fitted on
changes. See the `InfluenceModel` docstring for what this does to the
fields that used to be shared across coordinates. With every prototype
finite (the five estimators that never fail) this is identical to the
previous behaviour.

Width ell is searched ONCE, jointly, for every non-constant coordinate
that shares the SAME finite design (plan's "eigendecomposition per width
shared across a vector estimator's outputs", qij_package_plan.md §6):
non-constant coordinates are grouped by their finite prototype index set
-- one group holding everything when no prototype ever fails, which is
the ordinary case -- and within a group, the affine basis is projected
out with Q, an orthonormal basis of its complement (`numpy.linalg.qr`,
mode='complete'); for each candidate ell, one Matern-3/2 kernel K_ell
and one symmetric eigendecomposition of Q^T K_ell Q (Lambda, V) are
shared across every coordinate in the group -- this eigendecomposition
never repeats per coordinate within a group. From that shared
eigendecomposition, each coordinate in the group profiles its own
noise-to-signal ratio lam_c by a bounded scalar search over log lam
(s^2_c profiled out in closed form, the REML identity); the outer
objective minimized over log ell is the SUM of the group's
per-coordinate profiled restricted negative log marginal likelihoods.
The outer search evaluates 5 log-spaced widths on [ell_min, ell_max],
then refines with a second bounded search between the best grid point's
neighbours (`scipy.optimize.minimize_scalar(method='bounded')`
throughout). The chosen ell is common to every coordinate in the group;
lam_c, s^2_c and the final Cholesky remain per-coordinate (A depends on
lam_c). At q = 1, or whenever every coordinate shares one design, this
is identical to searching that one coordinate (or the whole group)
alone.

At the chosen (ell, lam_c), one Cholesky of A = K_ell + lam_c*I gives
beta, alpha, s^2 per coordinate; jitter only here, on the diagonal of A,
escalating by powers of ten from a trace-scaled base.

Point uncertainty is the posterior standard deviation sigma (R&W eq.
2.42, the affine-mean correction), computed in `uncertainty` from the
stored Cholesky factor and G = Hb^T A^-1 Hb -- no separate evaluation.

Constant-response path: a coordinate's finite design has fewer than 3
prototypes, or the mass-weighted spread of I_proto[:, c] over its own
finite design is negligible relative to |theta_Q[c]|, the estimator at
the quantized data. No width, no noise-to-signal ratio, no jitter, no
Cholesky; psi0/uncertainty return the constant / zero.

Cost classes (plan's "hot paths state their cost in the docstring"):
`fit_influence_model` costs O(M_X_used^3) (the shared eigendecompositions
and the per-coordinate Cholesky), never depending on N -- unchanged by
the per-coordinate design, since a coordinate's own finite subset is
never larger than M_X_used. `psi0` costs O(N * M_X_used) per coordinate
(one kernel evaluation and one matrix-vector product per point, against
that coordinate's own design). `uncertainty` costs O(N * M_X_used^2) per
coordinate: producing sigma at every data point requires, for every
point, solving against that coordinate's own M_c x M_c factor A -- this
is the per-point quadratic form the method needs at every x_i
(QIJ_method_outline.md step 4, "products at every data point: psi0(x_i)
and its uncertainty sigma_i"), and it is irreducibly O(N * M_X_used^2):
unlike the within-bin variance term below, sigma is a per-point, not a
per-bin, product. `bin_posterior_variance`'s first term (mean of sigma
squared over a bin) is O(N) total -- a plain average of the sigma
already computed by `uncertainty`, no new solve. Its second term (mean
of the posterior covariance over a bin) costs O(L * M_X_used^2) in its
two solves (a bin-summed kernel vector and a bin-summed basis-residual
vector, one solve each, per bin) -- never O(N * M_X_used^2), because no
per-point solve against A is repeated; the bin-summed vectors themselves
are formed in O(n_k * M_X_used) per bin, i.e. O(N * M_X_used) overall,
and the raw kernel-sum SS_k is formed by a chunked double sum over the
bin's own points (chunked at 2048 on both sides so no 2048 x 2048 block
is ever exceeded), an O(n_k^2) cost per bin that does not involve
M_X_used at all.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist

from .xvq import XVQ

_SQRT3 = math.sqrt(3.0)
_LOG_LAM_LO = math.log(1e-10)
_LOG_LAM_HI = math.log(1e2)
_N_WIDTH_GRID = 5
_UNCERTAINTY_BATCH_CAP = 4096
_BPV_CHUNK = 2048


@dataclass
class InfluenceModel:
    """The fitted initial influence estimate: one Gaussian process per
    estimand coordinate, sharing a whitening and, for the non-constant
    coordinates, a common kernel width ell across whichever other
    coordinates happen to share its design (QIJ_method_outline.md
    step 4).

    A failed prototype evaluation is a missing response (ruled 18
    September; module docstring): coordinate c is fitted only on the
    prototypes where I_proto[:, c] is finite, so the design -- its
    positions, its size and its basis -- is genuinely per coordinate,
    not one array shared by all q GPs. Every field below that used to
    be a single array indexed only by prototype (`centers`) or by a
    single shared scalar (`m`) is now a length-q list or array, one
    entry per coordinate, holding that coordinate's own design; `beta`
    and `alpha` were already per-coordinate columns and are now
    per-coordinate arrays of that coordinate's own (possibly smaller)
    length for the same reason. `chol_A`, `g_chol` and `ainv_hb` were
    already length-q lists and need no structural change -- their
    per-coordinate factors simply vary in size with the coordinate's
    own design now, exactly as they already varied between the
    constant and non-constant path. With every prototype finite, every
    coordinate's design is the same M_used prototypes and this
    reduces exactly to the shared, single-design model."""

    alpha: List[np.ndarray]       # per coordinate, (M_c,) GP weights on the kernel term; length-0 on the constant path
    beta: List[np.ndarray]        # per coordinate, (m_c,) affine-mean coefficients h(x)^T beta; length-0 on the constant path
    centers: List[np.ndarray]     # per coordinate, (M_c, d_z) whitened positions of THAT coordinate's finite prototypes
    whitening: Tuple[np.ndarray, np.ndarray]  # (mean, transform): raw Z -> whitened coordinates
    width: np.ndarray             # (q,) Matern-3/2 length scale ell_c; NaN on the constant path
    lam: np.ndarray               # (q,) profiled noise-to-signal ratio lam_c; NaN on the constant path
    s2: np.ndarray                # (q,) profiled GP signal variance s^2_c
    at_bound: np.ndarray          # (q, 2) bool: (ell_c, lam_c) within 1% in log of its search bound
    jitter: np.ndarray            # (q,) diagonal jitter added to A at the final solve; NaN on the constant path
    constant_path: np.ndarray     # (q,) bool: coordinate has no usable spread in I_proto over its own finite design
    const_value: np.ndarray       # (q,) the constant psi0 value returned on the constant path
    m: np.ndarray                 # (q,) int, basis size per coordinate: 1 (constant only) or d_z + 1 (affine); 0 on the constant path (unused there)
    n_width_evals: np.ndarray     # (q,) number of shared outer-objective evaluations used to pick ell, within that coordinate's group
    offset: np.ndarray            # (q,) mean of psi0 over Z (diagnostic; not subtracted from psi0)
    ml_wall_time: np.ndarray      # (q,) wall time of the marginal-likelihood fit, per coordinate
    median_sigma: np.ndarray      # (q,) median posterior sd (sigma) over Z
    p95_sigma: np.ndarray         # (q,) 95th percentile posterior sd (sigma) over Z
    chol_A: List[Optional[Tuple]]         # per coordinate, cho_factor of A = K_ell + lam_c*I (None on the constant path)
    g_chol: List[Optional[Tuple]]         # per coordinate, cho_factor of G = Hb^T A^-1 Hb (None on the constant path)
    ainv_hb: List[Optional[np.ndarray]]   # per coordinate, A^-1 Hb, (M_c, m_c) (None on the constant path)


def _whitening_from(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA whitening from the full 𝒳-VQ coordinates Z, used to map any
    new point into the same whitened space as `xvq.centers`: mean =
    Z.mean(0); covariance with ddof=0; eigh; transform =
    diag(1/sqrt(eigval)) @ eigvec.T. At d_z = 1 this reduces to plain
    standardization.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    mean = Z.mean(axis=0)
    Zc = Z - mean
    cov = (Zc.T @ Zc) / Z.shape[0]
    eigval, eigvec = np.linalg.eigh(cov)
    transform = np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
    return mean, transform


def _matern32(r: np.ndarray, ell: float) -> np.ndarray:
    """k_ell(r) = (1 + sqrt(3) r / ell) exp(-sqrt(3) r / ell)."""
    s = (_SQRT3 / ell) * r
    return (1.0 + s) * np.exp(-s)


def _basis(Zw: np.ndarray, m: int) -> np.ndarray:
    """h(x) = (1, x) in whitened coordinates (m = d_z + 1), or the
    constant basis (m = 1) when the design is too small for the affine
    basis."""
    N = Zw.shape[0]
    if m == 1:
        return np.ones((N, 1), dtype=float)
    return np.hstack([np.ones((N, 1), dtype=float), Zw])


def _length_scale_bounds(conn, D_full: np.ndarray) -> Tuple[float, float]:
    """
    ell_min = the median whitened distance between CONN-connected
    prototypes (upper triangle of conn, positive distances); fallback
    = the smallest positive inter-prototype distance if no CONN edges
    of positive length exist. ell_max = 10x the largest inter-prototype
    distance. `conn` and `D_full` are taken over whichever design is
    in play (the full prototype set, or a coordinate group's own finite
    subset, reindexed together).
    """
    M = D_full.shape[0]
    iu = np.triu_indices(M, k=1)
    all_d = D_full[iu]
    pos_all = all_d[all_d > 0.0]

    conn_coo = conn.tocoo()
    mask = conn_coo.row < conn_coo.col
    ii, jj = conn_coo.row[mask], conn_coo.col[mask]
    conn_d = D_full[ii, jj]
    conn_d = conn_d[conn_d > 0.0]

    if conn_d.size > 0:
        ell_min = float(np.median(conn_d))
    elif pos_all.size > 0:
        ell_min = float(pos_all.min())
    else:
        ell_min = 1.0  # degenerate: every prototype coincides

    ell_max = float(10.0 * all_d.max()) if all_d.size > 0 else 10.0 * ell_min
    if not (ell_max > ell_min):
        ell_max = ell_min * 10.0

    return ell_min, ell_max


def _cholesky_with_jitter(A: np.ndarray, K: np.ndarray, M_used: int):
    """
    One Cholesky of A, jitter only here: 0 -> 1e-10*trace(K)/M_used
    -> x10 -> x10 (at most three escalations). Raises RuntimeError if
    it still fails; the caller (`fit_influence_model`) lets this
    propagate -- the study records the draw as failed.

    Returns ((c, lower), jitter_used).
    """
    base = 1e-10 * float(np.trace(K)) / M_used
    eye = np.eye(M_used)
    for j in (0.0, base, base * 10.0, base * 100.0):
        try:
            chol = cho_factor(A + j * eye, lower=True)
            return chol, float(j)
        except LinAlgError:
            continue
    raise RuntimeError(
        "influence_model.fit_influence_model: Cholesky of A failed after jitter escalation"
    )


def _within_1pct_log(x: float, lo: float, hi: float) -> bool:
    """Within 1% in log of either bound."""
    if not (np.isfinite(x) and x > 0.0):
        return False
    near_lo = lo > 0.0 and abs(math.log(x) - math.log(lo)) < 0.01
    near_hi = hi > 0.0 and abs(math.log(x) - math.log(hi)) < 0.01
    return bool(near_lo or near_hi)


def fit_influence_model(
    Z: np.ndarray, xvq: XVQ, I_proto: np.ndarray, theta_Q: np.ndarray
) -> InfluenceModel:
    """
    The initial influence estimate for every estimand coordinate:
    Gaussian-process regression with an affine mean and a Matern-3/2
    kernel (QIJ_method_outline.md step 4). `xvq.p` (prototype masses)
    and `theta_Q` (the estimator evaluated at the quantized data,
    stage 1) are used only for the constant-response threshold; the
    kernel-regression noise itself is unweighted.

    A prototype whose evaluation failed for coordinate c is a missing
    response (module docstring): coordinate c's design is exactly the
    prototypes where `I_proto[:, c]` is finite -- `xvq.centers` and
    `xvq.p` restricted to that same subset (masses renormalized to sum
    to 1 over it, matching the mass-centering `xvq.prototype_influences`
    already applied per coordinate). Non-constant coordinates are
    grouped by their finite prototype index set so the shared width
    search (module docstring) still runs once per group, not once per
    coordinate; with no failures every coordinate is one group and this
    is exactly the previous computation.

    `xvq.centers` are prototype positions in Z's own NATIVE (unwhitened)
    coordinates -- the space `xvq.fit_xvq` clustered in, not the whitened
    space this function and `psi0`/`uncertainty` query in. This function
    whitens them with the SAME transform it derives from Z via
    `_whitening_from`, so a coordinate's design lands in exactly the
    space `Zw` (below, and in `psi0`/`uncertainty`) is queried in --
    without this, a query point and its nearest prototype would be
    compared in two different coordinate systems, and the kernel would
    see distances dominated by the raw-coordinate offset rather than the
    actual local geometry. `xvq.conn` (the CONN graph) supplies ell_min
    via the CONN-connected prototypes' median whitened distance,
    restricted to a group's own subset when that subset is not every
    prototype.

    Cost: O(M_X_used^3), independent of N (see module docstring).
    """
    p = np.asarray(xvq.p, dtype=float)
    I_proto = np.asarray(I_proto, dtype=float)
    theta_Q = np.asarray(theta_Q, dtype=float)
    M_X_used, q = I_proto.shape
    raw_centers = np.asarray(xvq.centers, dtype=float)
    d_z = raw_centers.shape[1]

    mean, transform = _whitening_from(Z)
    centers_full = (raw_centers - mean) @ transform.T

    finite = np.isfinite(I_proto)  # (M_X_used, q)

    constant_path = np.zeros(q, dtype=bool)
    const_value = np.zeros(q, dtype=float)
    offset = np.zeros(q, dtype=float)
    m_arr = np.zeros(q, dtype=int)

    centers: List[np.ndarray] = [np.empty((0, d_z), dtype=float)] * q
    alpha: List[np.ndarray] = [np.empty(0, dtype=float)] * q
    beta: List[np.ndarray] = [np.empty(0, dtype=float)] * q
    width = np.full(q, np.nan, dtype=float)
    lam = np.full(q, np.nan, dtype=float)
    s2 = np.zeros(q, dtype=float)
    at_bound = np.zeros((q, 2), dtype=bool)
    jitter = np.full(q, np.nan, dtype=float)
    n_width_evals = np.zeros(q, dtype=int)
    ml_wall_time = np.zeros(q, dtype=float)

    chol_A: List[Optional[tuple]] = [None] * q
    g_chol: List[Optional[tuple]] = [None] * q
    ainv_hb: List[Optional[np.ndarray]] = [None] * q

    idx_by_coord: List[np.ndarray] = []
    groups: Dict[tuple, List[int]] = {}
    for c in range(q):
        idx_c = np.where(finite[:, c])[0]
        idx_by_coord.append(idx_c)
        M_c = idx_c.size

        if M_c == 0:
            psi_bar = 0.0
        else:
            p_c = p[idx_c]
            mass_c = float(p_c.sum())
            psi_bar = float(np.sum(p_c * I_proto[idx_c, c]) / mass_c) if mass_c > 0.0 else 0.0

        coord_constant = M_c < 3
        if not coord_constant:
            p_c = p[idx_c]
            mass_c = float(p_c.sum())
            p_c_norm = p_c / mass_c
            psi_col = I_proto[idx_c, c]
            spread = float(np.sqrt(np.sum(p_c_norm * (psi_col - psi_bar) ** 2)))
            coord_constant = spread <= 1e-10 * max(abs(float(theta_Q[c])), 1e-300)

        if coord_constant:
            constant_path[c] = True
            const_value[c] = psi_bar
            offset[c] = psi_bar
        else:
            groups.setdefault(tuple(idx_c.tolist()), []).append(c)

    for idx_key, cols_g in groups.items():
        idx_g = np.asarray(idx_key, dtype=np.intp)
        M_g = idx_g.size
        centers_g = centers_full[idx_g]
        m_g = 1 if M_g <= d_z + 2 else d_z + 1
        Hb_g = _basis(centers_g, m_g)

        D_full_g = cdist(centers_g, centers_g)
        conn_g = xvq.conn[idx_g, :][:, idx_g]
        ell_min, ell_max = _length_scale_bounds(conn_g, D_full_g)
        grid_log_ell = np.linspace(math.log(ell_min), math.log(ell_max), _N_WIDTH_GRID)

        Qfull, _R = np.linalg.qr(Hb_g, mode='complete')
        Q = Qfull[:, m_g:]
        M_minus_m = M_g - m_g

        Qt_psi = {c: Q.T @ I_proto[idx_g, c] for c in cols_g}

        # ---- joint outer search over log ell, within this group ------
        # One kernel and one eigendecomposition of Q^T K_ell Q per
        # width, shared across every coordinate in this group; each
        # coordinate profiles its own noise-to-signal ratio lam_c from
        # that shared eigendecomposition, and the outer objective is
        # the SUM of the group's per-coordinate profiled restricted NLLs.
        t_shared0 = time.perf_counter()
        trace: List[dict] = []

        def outer_obj(log_ell: float, _trace=trace, _cols_g=cols_g,
                       _Qt_psi=Qt_psi, _Q=Q, _D_full=D_full_g,
                       _M_minus_m=M_minus_m) -> float:
            ell = math.exp(log_ell)
            K = _matern32(_D_full, ell)
            Mproj = _Q.T @ K @ _Q
            Lambda, V = np.linalg.eigh(Mproj)
            Lambda = np.maximum(Lambda, 0.0)

            per_c: Dict[int, dict] = {}
            total_nll = 0.0
            for c in _cols_g:
                z = V.T @ _Qt_psi[c]

                def inner(log_lam: float, _z=z, _Lambda=Lambda) -> float:
                    denom = _Lambda + math.exp(log_lam)
                    s2_val = np.sum(_z ** 2 / denom) / _M_minus_m
                    s2_val = max(s2_val, 1e-300)
                    return (_M_minus_m / 2.0) * math.log(s2_val) + 0.5 * np.sum(np.log(denom))

                ir = minimize_scalar(inner, bounds=(_LOG_LAM_LO, _LOG_LAM_HI), method='bounded')
                nll_c = float(ir.fun)
                per_c[c] = dict(log_lam=float(ir.x), nll=nll_c)
                total_nll += nll_c

            _trace.append(dict(log_ell=log_ell, ell=ell, K=K, per_c=per_c, nll=total_nll))
            return total_nll

        for le in grid_log_ell:
            outer_obj(float(le))

        grid_nlls = [t['nll'] for t in trace[:_N_WIDTH_GRID]]
        best_idx = int(np.argmin(grid_nlls))
        if best_idx == 0:
            lo, hi = grid_log_ell[0], grid_log_ell[1]
        elif best_idx == _N_WIDTH_GRID - 1:
            lo, hi = grid_log_ell[-2], grid_log_ell[-1]
        else:
            lo, hi = grid_log_ell[best_idx - 1], grid_log_ell[best_idx + 1]

        if hi > lo:
            minimize_scalar(outer_obj, bounds=(float(lo), float(hi)), method='bounded')

        best = min(trace, key=lambda t: t['nll'])
        n_shared_evals = len(trace)
        t_shared_total = time.perf_counter() - t_shared0
        t_shared_share = t_shared_total / len(cols_g)

        ell_c = best['ell']
        K_c = best['K']

        # ---- per-coordinate final solve (A depends on lam_c) ---------
        for c in cols_g:
            t0 = time.perf_counter()
            psi_c = I_proto[idx_g, c]
            lam_c = math.exp(best['per_c'][c]['log_lam'])

            A = K_c + lam_c * np.eye(M_g)
            chol, jit = _cholesky_with_jitter(A, K_c, M_g)

            AinvHb_c = cho_solve(chol, Hb_g)
            G = Hb_g.T @ AinvHb_c
            G_chol_c = cho_factor(G, lower=True)

            Ainv_psi = cho_solve(chol, psi_c)
            u_vec = Hb_g.T @ Ainv_psi
            beta_c = cho_solve(G_chol_c, u_vec)
            alpha_c = Ainv_psi - AinvHb_c @ beta_c

            denom_m = M_g - m_g
            if denom_m > 0:
                resid_quad = float(psi_c @ Ainv_psi - u_vec @ beta_c)
                s2_c = max(resid_quad, 0.0) / denom_m
            else:
                s2_c = 0.0

            centers[c] = centers_g
            m_arr[c] = m_g
            width[c] = ell_c
            lam[c] = lam_c
            s2[c] = s2_c
            jitter[c] = jit
            alpha[c] = alpha_c
            beta[c] = beta_c
            n_width_evals[c] = n_shared_evals
            at_bound[c, 0] = _within_1pct_log(ell_c, ell_min, ell_max)
            at_bound[c, 1] = _within_1pct_log(lam_c, 1e-10, 1e2)

            chol_A[c] = chol
            g_chol[c] = G_chol_c
            ainv_hb[c] = AinvHb_c

            ml_wall_time[c] = t_shared_share + (time.perf_counter() - t0)

    model = InfluenceModel(
        alpha=alpha,
        beta=beta,
        centers=centers,
        whitening=(mean, transform),
        width=width,
        lam=lam,
        s2=s2,
        at_bound=at_bound,
        jitter=jitter,
        constant_path=constant_path,
        const_value=const_value,
        m=m_arr,
        n_width_evals=n_width_evals,
        offset=offset,
        ml_wall_time=ml_wall_time,
        median_sigma=np.zeros(q, dtype=float),
        p95_sigma=np.zeros(q, dtype=float),
        chol_A=chol_A,
        g_chol=g_chol,
        ainv_hb=ainv_hb,
    )

    preds = psi0(model, Z)
    for c in range(q):
        if not constant_path[c]:
            model.offset[c] = float(preds[:, c].mean())

    sigma_full = uncertainty(model, Z)
    for c in range(q):
        if not constant_path[c]:
            model.median_sigma[c] = float(np.median(sigma_full[:, c]))
            model.p95_sigma[c] = float(np.percentile(sigma_full[:, c], 95))

    return model


def psi0(model: InfluenceModel, Z: np.ndarray) -> np.ndarray:
    """
    The initial influence estimate psi0(x) at Z (N, q): psi0(x) =
    h(x)^T beta + sum_j alpha_j k(x, w_j), the sum over coordinate c's
    OWN design `model.centers[c]` (module docstring: a failed prototype
    is missing only for the coordinates it failed at, so the design is
    per coordinate). Constant-path coordinates return `const_value[c]`
    at every row.

    Cost: O(N * M_X_used) per coordinate (one kernel row and one
    matrix-vector product per point, against that coordinate's own
    design); never O(N * M_X_used^2).
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    mean, transform = model.whitening
    Zw = (Z - mean) @ transform.T
    N = Zw.shape[0]
    q = len(model.centers)

    out = np.empty((N, q), dtype=float)
    for c in range(q):
        if model.constant_path[c]:
            out[:, c] = model.const_value[c]
        else:
            H = _basis(Zw, int(model.m[c]))
            D = cdist(Zw, model.centers[c])
            K = _matern32(D, float(model.width[c]))
            out[:, c] = H @ model.beta[c] + K @ model.alpha[c]
    return out


def uncertainty(model: InfluenceModel, Z: np.ndarray) -> np.ndarray:
    """
    Posterior standard deviation sigma at Z (N, q) (R&W eq. 2.42):
    sigma_i^2 = s^2 * [1 - k_i^T A^-1 k_i + r_i^T G^-1 r_i], r_i =
    h(x_i) - Hb^T A^-1 k_i, from the stored Cholesky factor and G =
    Hb^T A^-1 Hb, all coordinate c's own -- its own design, own A, own
    G (module docstring). Clipped at 0 before the square root.
    Constant-path coordinates return 0.

    Batched over rows of Z (`_UNCERTAINTY_BATCH_CAP` = 4096): at N =
    76997, M_X_used ~ 2297 (the largest sample size in the cost
    study), the unbatched (N, M) `K` / (M, N) `AinvKT` intermediates
    run to about 1.4 GB each, times 14 parallel workers -- per row
    chunk they are about 75 MB instead. `D` (raw whitened distance to
    a coordinate's own centers) is formed once per (chunk, coordinate)
    pair: it is no longer shared across coordinates, because a
    coordinate's design -- and so its centers -- can now differ from
    another coordinate's.

    Cost: O(N * M_X_used^2) per coordinate (a solve against that
    coordinate's own M_c x M_c factor A for every point's kernel
    vector). This is the one term in this module that is irreducibly
    O(N * M_X_used^2): sigma is a per-point product
    (QIJ_method_outline.md step 4), not a per-bin one, so there is no
    bin-summed reformulation available the way there is for
    `bin_posterior_variance`'s second term.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    mean, transform = model.whitening
    Zw = (Z - mean) @ transform.T
    N = Zw.shape[0]
    q = len(model.centers)

    out = np.zeros((N, q), dtype=float)
    batch = _UNCERTAINTY_BATCH_CAP
    for start in range(0, N, batch):
        sl = slice(start, start + batch)
        Zc = Zw[sl]

        for c in range(q):
            if model.constant_path[c]:
                continue
            Hc = _basis(Zc, int(model.m[c]))
            Dc = cdist(Zc, model.centers[c])  # (nb, M_c)
            Kc = _matern32(Dc, float(model.width[c]))  # (nb, M_c)
            chol = model.chol_A[c]
            AinvKcT = cho_solve(chol, Kc.T)  # (M_c, nb)
            term1 = np.einsum('ij,ji->i', Kc, AinvKcT)

            Rc = Hc - Kc @ model.ainv_hb[c]  # (nb, m_c)
            GinvRcT = cho_solve(model.g_chol[c], Rc.T)  # (m_c, nb)
            term2 = np.einsum('ij,ji->i', Rc, GinvRcT)

            sigma2 = model.s2[c] * (1.0 - term1 + term2)
            sigma2 = np.maximum(sigma2, 0.0)
            out[sl, c] = np.sqrt(sigma2)
    return out


def bin_posterior_variance(
    model: InfluenceModel,
    Z: np.ndarray,
    coordinate: int,
    groups: Sequence[np.ndarray],
    sigma_c: np.ndarray,
) -> np.ndarray:
    """
    The within-bin posterior variance v_k for coordinate `coordinate`
    at each group in `groups`: the variance of the initial influence
    estimate's own posterior error around its bin mean, v_k =
    mean(diag Sigma_k) - mean(Sigma_k), where Sigma_k is the posterior
    covariance of psi0 over the group's points (R&W eq. 2.42, the same
    three terms `uncertainty` forms on the diagonal, formed here off
    the diagonal too), all against coordinate `coordinate`'s OWN design
    (module docstring):

        Sigma_ij = s2_c * [k(x_i, x_j) - k_i^T A^-1 k_j + r_i^T G^-1 r_j]
        r_i = h(x_i) - Hb^T A^-1 k_i

    `mean(diag Sigma_k)` is the mean of sigma_i^2 over the group,
    taken directly from `sigma_c` (N,), the per-point posterior
    standard deviation the caller already computed once via
    `uncertainty(model, Z)` -- NOT re-derived here: re-deriving it
    would repeat the O(N * M_X_used^2) `cho_solve(chol, K.T)` product
    `uncertainty` already paid for. Squaring `sigma_c` recovers the
    same already-clipped-at-0 value `uncertainty` produces.

    `mean(Sigma_k)` is computed from bin-summed vectors, never by
    holding an n_k x n_k matrix: with s = sum_{i in k} k_i (length
    M_c, coordinate `coordinate`'s own design size) and R = sum_{i in
    k} r_i (length m_c),

        mean(Sigma_k) = s2_c / n_k^2 * [SS_k - s^T A^-1 s + R^T G^-1 R]
        SS_k = sum_{i,j in k} k(x_i, x_j)

    `s`, `R` and `SS_k` are accumulated over the group's rows in
    chunks of at most 2048; `SS_k`'s double sum chunks BOTH sides (a
    plain nested loop over chunk pairs, symmetry not exploited), so no
    block larger than 2048 x 2048 is ever materialized -- the naive
    one-sided chunking (chunk against the whole group) would still
    allocate a chunk_size x n_k block, unbounded in n_k. `A^-1 s` and
    `G^-1 R` (the only solves against A and G) are each one solve on a
    bin-summed vector, M_c^2 and m_c^2 per bin.

    Cost: O(L * M_X_used^2) in the solves (one A-solve and one
    G-solve per bin, never repeated per point); O(N * M_X_used) to
    form the bin-summed vectors s and R (one kernel row per point,
    summed); O(sum_k n_k^2) for the SS_k double sums, a cost bounded
    by chunking in memory but not in M_X_used or in L -- this term
    does not depend on M_X_used at all, since it is a sum of raw
    kernel values, not a solve. Never O(N * M_X_used^2): no per-point
    solve against A is formed here.

    Returns 0.0 for a group of size <= 1, and 0.0 for every group
    when `model.constant_path[coordinate]` is true. Clipped at 0 from
    below.
    """
    c = coordinate
    n_groups = len(groups)
    out = np.zeros(n_groups, dtype=float)
    if model.constant_path[c]:
        return out

    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    mean_w, transform = model.whitening
    Zw_full = (Z - mean_w) @ transform.T
    sigma_c = np.asarray(sigma_c, dtype=float)

    ell_c = float(model.width[c])
    s2_c = float(model.s2[c])
    chol = model.chol_A[c]
    ainv_hb_c = model.ainv_hb[c]
    g_chol_c = model.g_chol[c]
    m_c = int(model.m[c])
    centers_c = model.centers[c]
    M = centers_c.shape[0]

    for gi, idx in enumerate(groups):
        idx = np.asarray(idx)
        n_k = idx.size
        if n_k <= 1:
            continue

        Zw_k = Zw_full[idx]
        H_k = _basis(Zw_k, m_c)

        mean_diag = float(np.mean(sigma_c[idx] ** 2))

        s_vec = np.zeros(M, dtype=float)
        R_vec = np.zeros(m_c, dtype=float)
        SS_k = 0.0

        for start in range(0, n_k, _BPV_CHUNK):
            sl = slice(start, start + _BPV_CHUNK)
            Zc = Zw_k[sl]
            Hc = H_k[sl]

            Dc = cdist(Zc, centers_c)
            Kc = _matern32(Dc, ell_c)  # (nb, M)
            Rc = Hc - Kc @ ainv_hb_c  # (nb, m_c)

            s_vec += Kc.sum(axis=0)
            R_vec += Rc.sum(axis=0)

            for start2 in range(0, n_k, _BPV_CHUNK):
                sl2 = slice(start2, start2 + _BPV_CHUNK)
                Dcc = cdist(Zc, Zw_k[sl2])
                Kcc = _matern32(Dcc, ell_c)
                SS_k += float(Kcc.sum())

        Ainv_s = cho_solve(chol, s_vec)
        Ginv_R = cho_solve(g_chol_c, R_vec)
        mean_Sigma = (s2_c / (n_k ** 2)) * (SS_k - s_vec @ Ainv_s + R_vec @ Ginv_R)

        out[gi] = max(mean_diag - mean_Sigma, 0.0)

    return out
