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

**The noise-to-signal ratio's floor is declared, not fitted below the
declared level (plan §36.2(2), 19 September revision 8).** The prototype
influences REML fits are finite differences of relative accuracy `eta`,
so I_proto carries real evaluation noise -- noise sd sqrt(2)*eta*|T|/t_j
at prototype j, t_j = step_parameter(delta_f, p_j), delta_f =
forward_step(eta) -- and left unconstrained, REML happily interpolates
that noise as signal (driving lam_c to its absolute floor 1e-10 and
reporting the resulting wiggle as within-bin variance). Per coordinate
c, the declared noise level is n_c^2 = 2*eta^2*theta_Q,c^2 *
median_j(1/t_j^2), the median over j taken over coordinate c's OWN
finite design (module docstring above); lam_c is then searched by REML
exactly as before but on [max(lam_floor,c, 1e-10), 1e2], where
lam_floor,c solves lam*s_c^2(lam) = n_c^2 with s_c^2(lam) the profiled
closed form `inner` already computes. lam*s_c^2(lam) = (1/(M-m)) *
sum_i z_i^2 * lam/(Lambda_i + lam) is increasing in lam (from the
zero-eigenvalue directions' share of z at lam -> 0, if any, up to
mean(z^2) at lam -> infinity), so the root is unique and is found by one
`scipy.optimize.brentq` call on log lam, pinned to the domain's own
edges when n_c^2 falls outside the range lam*s_c^2(lam) reaches there
(`_lambda_floor` below) -- no iteration, no fallback search. Because
lam*s_c^2(lam) depends on ell (through Lambda and z = V^T Qt_psi at that
ell), the floor is recomputed once per candidate ell inside the shared
outer search, not once for the whole fit; the final per-coordinate solve
below simply reuses the already-floored lam_c the winning ell's `inner`
search already found. sigma_i, v_k and psi0_hat all follow from the same
posterior, unchanged in form. For the closed-form estimators (eta at
machine precision) n_c^2 is negligible and the floor is inactive --
lam_floor,c sits far below the absolute floor 1e-10 the search already
clips to, so those coordinates are unaffected to rounding.

Point uncertainty is the posterior standard deviation sigma (R&W eq.
2.42, the affine-mean correction), computed in `uncertainty` from the
stored Cholesky factor and G = Hb^T A^-1 Hb -- no separate evaluation.

Constant-response path: a coordinate's finite design has fewer than 3
prototypes, or the mass-weighted spread of I_proto[:, c] over its own
finite design is negligible relative to |theta_Q[c]|, the estimator at
the quantized data. No width, no noise-to-signal ratio, no jitter, no
Cholesky; psi0/uncertainty return the constant / zero.

**The per-point posterior terms are computed ONCE per draw, for every
output at once, through one eigendecomposition of K (20 September, the
wall-time refactor).** `psi0`, `uncertainty` and `bin_posterior_
variance` are all queries of the SAME fitted posterior at the SAME
points -- the N rows of Z -- and the package asks for every one of them
on every draw. They are therefore served from a single cached pass,
`_point_terms` below, keyed on the identity of the `Z` object they were
computed for; a second call with that same `Z` returns the cached
arrays rather than recomputing anything. (`_fit_gp_influence_model`
itself makes the first call, for its own `offset`/`median_sigma`/
`p95_sigma` diagnostics, so the pass `qij.py` then asks for is already
paid -- before this, psi0 and sigma were each computed twice per draw.)

Within that pass, the per-point quadratic form k_i^T A_c^-1 k_i -- the
one genuinely O(N * M^2) product in this module -- is paid ONCE for all
q outputs instead of once per output. A_c = K + (lam_c + jitter_c) I
differs between a vector estimator's outputs only through the scalar
lam_c + jitter_c, so with the symmetric eigendecomposition K = V Lambda
V^T (one `eigh` per coordinate group at the chosen width, stored on the
model as `k_eigval`/`k_eigvec`; Lambda clipped at 0, as the shared
outer search already clips its own),

    k_i^T A_c^-1 k_i = sum_m (V^T k_i)_m^2 / (Lambda_m + lam_c + jit_c)
    Hb^T A_c^-1 k_i  = (V^T Hb)^T (Lambda + lam_c + jit_c)^-1 (V^T k_i)

Both read the same P = K_n V (N x M), one BLAS product shared by every
output; each output then costs O(N * M) for its own weighted row sums
and O(N * M * m) for its affine-mean residual r_i. What used to be q
triangular-solve pairs against q different Cholesky factors of A_c, each
O(N * M^2), is one O(N * M^2) product plus O(q * N * M * m). This is a
different factorisation of the same quantity, not a different quantity:
sigma and v_k agree with the Cholesky form to rounding, not bit for
bit. `psi0` is untouched by it -- h(x)^T beta + k(x)^T alpha still reads
`alpha` and `beta` straight from the Cholesky solve
`fit_influence_model` performs, so the initial influence estimate, and
hence the 𝓘-VQ's initial bins, are bit-for-bit what they were.
`bin_posterior_variance` takes BOTH of its A^-1 terms -- the per-point
mean(diag Sigma_k) and the bin-summed s^T A^-1 s -- through that same
eigen form, deliberately: v_k is a difference of two nearly equal
quantities, and mixing two factorisations of A^-1 across that
subtraction would show up in the cancellation.

The kernel rows k(x_i, w_j) themselves are cached per coordinate group
(`_PointTerms.K`) when the (N, M) array fits `_KERNEL_CACHE_BYTES`, so
`bin_posterior_variance` never re-evaluates a kernel for a point that
was already visited -- in particular a child bin created by a
refinement split re-uses its parent's rows instead of re-forming them.
The cache is a memory-bounded convenience only: over the budget the
rows are re-formed in chunks exactly as before, and the numbers are
bit-identical either way (the same `cdist`/`_matern32` values, summed
in the same order), so nothing a run reports depends on whether the
cache was enabled.

Cost classes (plan's "hot paths state their cost in the docstring"):
`fit_influence_model` costs O(M_X_used^3) (the shared eigendecompositions
and the per-coordinate Cholesky), never depending on N -- unchanged by
the per-coordinate design, since a coordinate's own finite subset is
never larger than M_X_used. The one `eigh` of K per coordinate group at
the chosen width is part of that same O(M_X_used^3) class and is paid
once, not once per output. `psi0` costs O(N * M_X_used) per coordinate
(one kernel evaluation and one matrix-vector product per point, against
that coordinate's own design). `uncertainty` costs O(N * M_X_used^2)
ONCE PER COORDINATE GROUP (the shared P = K_n V product), plus O(N *
M_X_used * m) per coordinate -- it is no longer O(N * M_X_used^2) per
coordinate. `bin_posterior_variance`'s first term (mean of sigma
squared over a bin) is O(N) total -- a plain average of the sigma
already computed by the shared pass, no new solve. Its second term (mean
of the posterior covariance over a bin) costs O(L * M_X_used^2) in its
one M x M product (the bin-summed kernel vector against V) plus O(L *
m^2) for the affine-mean solve -- never O(N * M_X_used^2), because no
per-point solve against A is repeated; the bin-summed vectors themselves
are formed in O(n_k * M_X_used) per bin, i.e. O(N * M_X_used) overall,
from the cached kernel rows where those are available, and the raw
kernel-sum SS_k is formed by a chunked double sum over the bin's own
points (chunked at 2048 on both sides so no 2048 x 2048 block is ever
exceeded), an O(n_k^2) cost per bin that does not involve M_X_used at
all -- the one term here that no caching removes, since it is a sum of
raw kernel values between the bin's own points.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_triangular
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial.distance import cdist

from .differences import forward_step, step_parameter
from .xvq import XVQ

_SQRT3 = math.sqrt(3.0)
_LOG_LAM_LO = math.log(1e-10)  # the absolute floor: max(lambda_floor_c, 1e-10) never searches below this
_LOG_LAM_HI = math.log(1e2)
_N_WIDTH_GRID = 5
_UNCERTAINTY_BATCH_CAP = 4096
_BPV_CHUNK = 2048
# The widest range of c*(x - o) a block of `_matern32_self_sum_1d` may
# span: e^{+-200} is far inside double range, and the reordering it
# costs is bounded by about 100 roundings on a sum of positive terms.
_SELF_SUM_LOG_RANGE = 200.0
_FITC_D0_FLOOR = 1e-6  # see _fit_fitc_influence_model's docstring on D0

# The per-draw kernel-row cache's memory budget (module docstring, "The
# kernel rows k(x_i, w_j) themselves are cached"): the (N, M_g) kernel
# matrices of ALL coordinate groups together may occupy at most this
# many bytes, or none of them is cached and every consumer re-forms its
# rows in chunks exactly as it did before the cache existed. The budget
# is per draw, i.e. per parallel worker -- 14 workers at the study's
# usual width hold at most 14 times this. At the cost study's largest
# sample size (N = 76997, M_X_used ~ 2297) one group's matrix alone is
# 1.4 GB, so the cache stays OFF there and that run's memory profile is
# exactly what it was; at the paper's N = 2000 it is 6 MB and the cache
# is on.
_KERNEL_CACHE_BYTES = 256 * 1024 ** 2


@dataclass
class _PointTerms:
    """One cached pass of the posterior over the N rows of Z (module
    docstring, "The per-point posterior terms are computed ONCE per
    draw"). Held on the model as `_points` and keyed on the IDENTITY of
    the `Z` object it was computed for -- `qij.py` threads one `Z`
    array through `psi0`, `uncertainty` and every `run_refinement`
    call, so identity is the exact key, and holding a reference to `Z`
    here makes that identity unambiguous (the object cannot be freed
    and its id reused while the cache is alive).

    Z       the array this pass was computed for (the cache key, held
            by reference; never read).
    Zw      (N, d_z) Z in the model's whitened coordinates.
    psi0    (N, q) the initial influence estimate at every point.
    sigma   (N, q) the posterior standard deviation at every point.
    R       per coordinate, (N, m_c) the affine-mean residual r_i =
            h(x_i) - Hb^T A_c^-1 k_i, one row per point -- kept because
            `bin_posterior_variance`'s R = sum_{i in k} r_i is a plain
            row sum of it. None on the constant path, and None for
            every coordinate under the FITC model (which has its own
            `_fitc_bin_posterior_variance` and never reads this).
    K       per coordinate, (N, M_c) the kernel rows k(x_i, w_j)
            against that coordinate's own design -- the SAME array
            object for every coordinate sharing a design, so a group is
            stored once, not q times. None everywhere when the total
            would exceed `_KERNEL_CACHE_BYTES` (the caller then re-forms
            rows in chunks, to bit-identical values), and None on the
            constant path and under FITC."""

    Z: np.ndarray
    Zw: np.ndarray
    psi0: np.ndarray
    sigma: np.ndarray
    R: List[Optional[np.ndarray]]
    K: List[Optional[np.ndarray]]


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
    lam_floor: np.ndarray         # (q,) the declared-noise floor lam_floor,c at the final chosen ell (module docstring, plan sec 36.2(2)); NaN on the constant path
    s2: np.ndarray                # (q,) profiled GP signal variance s^2_c
    at_bound: np.ndarray          # (q, 2) bool: (ell_c, lam_c) within 1% in log of its search bound (lam_c's lower bound is the declared floor when it exceeds 1e-10)
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
    # The shared eigendecomposition K_ell = V Lambda V^T at the chosen
    # width (module docstring, "The per-point posterior terms are
    # computed ONCE per draw"): ONE `eigh` per coordinate GROUP, so
    # every coordinate in a group holds the SAME three array objects.
    # `k_eigval` is Lambda clipped at 0, `k_eigvec` is V, `hb_eig` is
    # V^T Hb (M_c, m_c) -- the affine-mean basis in the same eigenbasis,
    # so the mean correction never needs Hb itself at query time. None
    # on the constant path.
    k_eigval: List[Optional[np.ndarray]]  # per coordinate, (M_c,)
    k_eigvec: List[Optional[np.ndarray]]  # per coordinate, (M_c, M_c)
    hb_eig: List[Optional[np.ndarray]]    # per coordinate, (M_c, m_c)
    # The cached posterior pass over Z (`_PointTerms`); set by
    # `_point_terms` on first use, never by the constructor.
    _points: Optional[_PointTerms] = field(default=None, repr=False, compare=False)


@dataclass
class FITCInfluenceModel:
    """The FITC (sparse, inducing-point) alternative to `InfluenceModel`
    (module docstring's "The FITC model"): same whitening, affine mean,
    Matern-3/2 kernel, per-coordinate design and declared noise floor as
    the full-rank GP, but every M_c x M_c object is replaced by a rank-r
    structure, r << M_c, through the Woodbury identity -- nothing M_c x
    M_c is ever formed or factorised. Selected by `fit_influence_model
    (..., influence_model='fitc')`; the full-rank GP (`InfluenceModel`)
    remains the default and is computed by a completely separate,
    untouched code path (bit-identical to before this model existed).

    Unlike `InfluenceModel`, this dataclass carries no M_c-length
    per-coordinate arrays at all -- `psi0`/`uncertainty`/`bin_posterior_
    variance` need only each coordinate's r_c inducing points and a
    handful of r_c x r_c (or m_c x r_c, m_c x m_c) objects, so a query
    costs O(r_c) (mean) or O(r_c^2) (variance), never O(M_c) or
    O(M_c^2) -- the sparse method's whole point, at both fit time and
    query time.

    centers_ind        per coordinate, (r_c, d_z) whitened positions of
                        the r_c inducing points, a subset of that
                        coordinate's own finite design chosen by
                        deterministic farthest-point traversal
                        (`_farthest_point_inducing`); empty on the
                        constant path.
    inducing_idx        per coordinate, (r_c,) int, the chosen inducing
                        points' indices into that coordinate's own
                        finite design (`centers_full[idx_g]` order) --
                        recorded for audit, not read by prediction.
    rank                (q,) int, r_c actually used per coordinate (0 on
                        the constant path).
    alpha_m             per coordinate, (r_c,) the low-rank prediction
                        weights: alpha_m = B^-1 K_nm^T D^-1 (psi_c - Hb
                        beta), so psi0's kernel term is K(x, centers_ind)
                        @ alpha_m -- see the module report for the
                        derivation.
    W                   per coordinate, (r_c, r_c) = K_mm^-1 - B^-1, the
                        quadratic form `uncertainty`/`bin_posterior_
                        variance` use in place of k_i^T A^-1 k_j (module
                        report: k_i^T A_fitc^-1 k_j = k_mi^T W k_mj for
                        any two points i, j, exact under the FITC
                        generative model).
    E                   per coordinate, (m_c, r_c) = Hb^T A_fitc^-1 Phi
                        (Phi = K_nm K_mm^-1), so r_i = h(x_i) - E @ k_mi
                        replaces r_i = h(x_i) - Hb^T A^-1 k_i without
                        ever touching the M_c-length k_i.
    g_chol              per coordinate, cho_factor of G = Hb^T A_fitc^-1
                        Hb (m_c, m_c); None on the constant path.
    kmm_chol            per coordinate, cho_factor of K_mm (r_c, r_c),
                        jittered exactly as `_cholesky_with_jitter`
                        jitters the GP's A; used by `bin_posterior_
                        variance`'s SS_k (the bin-summed prior term) and
                        by the per-point diagonal correction it needs.
    beta, whitening, width, lam, lam_floor, s2, at_bound, jitter,
    constant_path, const_value, m, n_width_evals, offset, ml_wall_time,
    median_sigma, p95_sigma
                        exactly as `InfluenceModel` -- diagnostics
                        `study.py`'s `_qij_row` and the constant/affine
                        mean machinery read identically regardless of
                        which model was fitted. `jitter[c]` is the
                        larger of the jitter used on K_mm and on B (two
                        Cholesky factorizations happen per coordinate
                        under FITC, where the GP has one)."""

    beta: List[np.ndarray]
    whitening: Tuple[np.ndarray, np.ndarray]
    width: np.ndarray
    lam: np.ndarray
    lam_floor: np.ndarray
    s2: np.ndarray
    at_bound: np.ndarray
    jitter: np.ndarray
    constant_path: np.ndarray
    const_value: np.ndarray
    m: np.ndarray
    n_width_evals: np.ndarray
    offset: np.ndarray
    ml_wall_time: np.ndarray
    median_sigma: np.ndarray
    p95_sigma: np.ndarray
    centers_ind: List[np.ndarray]
    inducing_idx: List[np.ndarray]
    rank: np.ndarray
    alpha_m: List[np.ndarray]
    W: List[Optional[np.ndarray]]
    E: List[Optional[np.ndarray]]
    g_chol: List[Optional[Tuple]]
    kmm_chol: List[Optional[Tuple]]
    # The cached posterior pass over Z (`_PointTerms`), exactly as on
    # `InfluenceModel`: FITC shares `psi0`/`uncertainty`'s caching of
    # psi0 and sigma (they were each computed twice per draw before),
    # but not the kernel-row or residual caches -- `_fitc_bin_posterior_
    # variance` works from r_c-length objects and reads neither.
    _points: Optional[_PointTerms] = field(default=None, repr=False, compare=False)


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


def _matern32_self_sum_1d(x: np.ndarray, ell: float) -> float:
    """
    sum_{i,j} k_ell(|x_i - x_j|) over one bin's OWN points, for a
    one-dimensional design, in O(n log n) (the sort) rather than the
    O(n^2) of forming the bin's full pairwise kernel.

    With c = sqrt(3)/ell and the points sorted ascending, every pair
    contributes through r_ij = x_i - x_j for i > j, so

        sum_{i,j} (1 + c r_ij) e^{-c r_ij} = n + 2 sum_i (S_i + c T_i),
        S_i = sum_{j<i} e^{-c(x_i - x_j)},
        T_i = sum_{j<i} (x_i - x_j) e^{-c(x_i - x_j)}.

    Both are running sums. Against a local origin o, with u_i = x_i - o,

        S_i = e^{-c u_i} sum_{j<i} e^{c u_j},
        T_i = u_i S_i - e^{-c u_i} sum_{j<i} u_j e^{c u_j},

    so one exclusive cumulative sum of e^{c u_j} and one of
    u_j e^{c u_j} deliver every S_i and T_i at once.

    The origin has to move, because e^{c u} overflows once c u passes
    about 709. The points are therefore cut into consecutive blocks
    each spanning at most 200/c, which holds every exponential inside a
    block within e^{+-200}, and the two running sums are carried across
    a block boundary exactly: for a new origin o' = o + d, the carried
    sums become e^{-c d} times (the old sums over every point so far),
    the second one shifted by d. No pair is dropped and none is
    approximated -- the only difference from the pairwise sum is the
    order the terms are added in.

    `bin_posterior_variance` keeps its pairwise double sum for designs
    of two dimensions or more, where |x_i - x_j| is not a difference of
    coordinates and none of this applies.
    """
    xs = np.sort(np.asarray(x, dtype=float).ravel())
    n = xs.size
    if n <= 1:
        return float(n)

    c = _SQRT3 / ell
    block_span = _SELF_SUM_LOG_RANGE / c

    acc = 0.0
    s_carry = 0.0   # sum over earlier points of e^{c (x_j - o)}
    t_carry = 0.0   # sum over earlier points of (x_j - o) e^{c (x_j - o)}
    start = 0
    while start < n:
        end = int(np.searchsorted(xs, xs[start] + block_span, side='right'))
        u = xs[start:end] - xs[start]
        a = np.exp(-c * u)
        b = np.exp(c * u)
        ub = u * b
        # Exclusive cumulative sums, built by shifting an inclusive one
        # rather than subtracting the term back off: b increases across
        # a block, so `cumsum(b) - b` would cancel to nothing wherever a
        # gap makes b_i dominate everything before it.
        cb = np.empty(u.size, dtype=float)
        cb[0] = 0.0
        np.cumsum(b[:-1], out=cb[1:])
        cub = np.empty(u.size, dtype=float)
        cub[0] = 0.0
        np.cumsum(ub[:-1], out=cub[1:])

        s_tot = s_carry + cb
        S = a * s_tot
        T = u * S - a * (t_carry + cub)
        acc += float(np.add.reduce(S) + c * np.add.reduce(T))

        if end < n:
            s_all = s_carry + float(np.add.reduce(b))
            t_all = t_carry + float(np.add.reduce(ub))
            d = float(xs[end] - xs[start])
            phi = math.exp(-c * d)
            s_carry = phi * s_all
            t_carry = phi * (t_all - d * s_all)
        start = end

    return float(n) + 2.0 * acc


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


def _floor_equation(log_lam: float, z: np.ndarray, Lambda: np.ndarray, M_minus_m: int) -> float:
    """lam * s_c^2(lam), the same profiled closed-form quantity
    `inner`'s numerator computes (module docstring, declared noise
    floor), as a function of lam alone: sum(z^2 * lam / (Lambda + lam))
    / (M - m). Increasing in lam."""
    lam = math.exp(log_lam)
    return float(np.sum(z ** 2 * lam / (Lambda + lam)) / M_minus_m)


def _lambda_floor(z: np.ndarray, Lambda: np.ndarray, M_minus_m: int, n_c2: float) -> float:
    """The declared-noise floor lam_floor,c for one coordinate at one
    candidate ell (module docstring): the unique root of
    lam*s_c^2(lam) = n_c2 in log lam, found by one `brentq` call --
    no iteration, no fallback search -- because `_floor_equation` is
    increasing in lam. Pinned to the search domain's own edges,
    [1e-10, 1e2], rather than solved outside it, since a root outside
    that range does nothing more to `max(lam_floor_c, 1e-10)` than the
    nearer edge already does: 1e-10 when even the smallest lam already
    exceeds n_c2 (the declared level adds no restriction), 1e2 when
    even the largest lam cannot reach it (the declared level swamps
    the whole projected response and lam is pinned at the ceiling)."""
    lo, hi = _LOG_LAM_LO, _LOG_LAM_HI
    g_lo = _floor_equation(lo, z, Lambda, M_minus_m) - n_c2
    if g_lo >= 0.0:
        return math.exp(lo)
    g_hi = _floor_equation(hi, z, Lambda, M_minus_m) - n_c2
    if g_hi <= 0.0:
        return math.exp(hi)
    root = brentq(lambda x: _floor_equation(x, z, Lambda, M_minus_m) - n_c2, lo, hi)
    return math.exp(root)


def fit_influence_model(
    Z: np.ndarray, xvq: XVQ, I_proto: np.ndarray, theta_Q: np.ndarray, eta: float,
    influence_model: str = 'gp', fitc_rank: Optional[int] = None,
) -> Union[InfluenceModel, FITCInfluenceModel]:
    """
    Dispatcher (module docstring's "The switch"): `influence_model`
    selects the full-rank Gaussian process (`'gp'`, the default,
    `_fit_gp_influence_model` below -- completely unchanged by this
    dispatcher's existence) or the FITC sparse alternative (`'fitc'`,
    `_fit_fitc_influence_model`). `fitc_rank` is passed through to the
    FITC path only (ignored, and should be left None, for `'gp'`);
    `None` there means "compute the default rank per coordinate group"
    (`_fit_fitc_influence_model`'s own docstring). Any other
    `influence_model` value raises `ValueError` naming the two accepted
    strings.
    """
    if influence_model == 'gp':
        return _fit_gp_influence_model(Z, xvq, I_proto, theta_Q, eta)
    elif influence_model == 'fitc':
        return _fit_fitc_influence_model(Z, xvq, I_proto, theta_Q, eta, fitc_rank)
    else:
        raise ValueError(
            f"fit_influence_model: influence_model must be 'gp' or 'fitc', got {influence_model!r}"
        )


def _fit_gp_influence_model(
    Z: np.ndarray, xvq: XVQ, I_proto: np.ndarray, theta_Q: np.ndarray, eta: float
) -> InfluenceModel:
    """
    The initial influence estimate for every estimand coordinate:
    Gaussian-process regression with an affine mean and a Matern-3/2
    kernel (QIJ_method_outline.md step 4). `xvq.p` (prototype masses)
    and `theta_Q` (the estimator evaluated at the quantized data,
    stage 1) are used both for the constant-response threshold and,
    together with `eta` (the estimator's per-evaluation relative
    accuracy), for the declared noise floor on lam_c (module
    docstring, plan §36.2(2)); the kernel-regression noise itself is
    unweighted.

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

    # Declared noise floor (module docstring, plan §36.2(2)): delta_f is
    # the SAME forward step `xvq.prototype_influences` used to measure
    # I_proto, so t_j below reproduces that function's own t_j exactly,
    # not a second way of deriving it.
    delta_f = forward_step(eta)

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
    lam_floor = np.full(q, np.nan, dtype=float)
    s2 = np.zeros(q, dtype=float)
    at_bound = np.zeros((q, 2), dtype=bool)
    jitter = np.full(q, np.nan, dtype=float)
    n_width_evals = np.zeros(q, dtype=int)
    ml_wall_time = np.zeros(q, dtype=float)

    chol_A: List[Optional[tuple]] = [None] * q
    g_chol: List[Optional[tuple]] = [None] * q
    ainv_hb: List[Optional[np.ndarray]] = [None] * q
    k_eigval: List[Optional[np.ndarray]] = [None] * q
    k_eigvec: List[Optional[np.ndarray]] = [None] * q
    hb_eig: List[Optional[np.ndarray]] = [None] * q

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

        # ---- the outer search's projected kernel, without Q^T K Q ----
        # `outer_obj` below needs the spectrum of Q^T K_ell Q and the
        # projected response z = V^T Q^T psi_c at every candidate ell.
        # Forming Q^T K Q costs two M_g x M_g matmuls (~4 M_g^3 flops),
        # more than a quarter of the whole evaluation. The same spectrum
        # comes out of an M_g x M_g matrix built in O(m_g M_g^2):
        #
        #     A_ell = (I - P) K_ell (I - P) + tau P,   P = W W^T
        #
        # with W = Qfull[:, :m_g] the orthonormal basis spanning the
        # affine mean space, so I - P = Q Q^T exactly, whatever Hb_g's
        # rank. A_ell is block diagonal in the (range P, range Q) split:
        # its spectrum is eig(Q^T K Q) together with tau repeated m_g
        # times, and its range-Q eigenvectors are Q times those of
        # Q^T K Q. The shift tau is what makes that split unambiguous
        # instead of asking `eigh` to tell m_g structural zeros apart
        # from the kernel's OWN near-zero eigenvalues, which at this
        # conditioning it cannot: k(0) = 1 for the Matern-3/2 kernel, so
        # trace(K_ell) = M_g and every eigenvalue of Q^T K Q lies in
        # [0, M_g]; tau = 2 M_g puts the m_g shifted eigenvalues above
        # all of them with a gap of at least M_g, half the norm of
        # A_ell. `eigh` returns them last, so the FIRST M_minus_m pairs
        # are exactly the ones the profiling needs. The response is
        # projected once here rather than per ell: with psi_proj_c =
        # Q Q^T psi_c, V_A^T psi_proj_c is V^T Q^T psi_c, the same z the
        # two-matmul form produced.
        W_g = Qfull[:, :m_g]
        tau_g = 2.0 * float(M_g)
        psi_proj = {c: Q @ Qt_psi[c] for c in cols_g}

        # ---- declared noise floor, per coordinate in this group ------
        # n_c^2 = 2*eta^2*theta_Q,c^2 * median_j(1/t_j^2) (module
        # docstring), t_j = step_parameter(delta_f, p_j) over coordinate
        # c's OWN finite design idx_g (the group's design, shared by
        # every coordinate in cols_g) -- independent of ell, so computed
        # once per group, not once per candidate ell.
        p_g = p[idx_g]
        t_g = np.array([step_parameter(delta_f, float(pj)) for pj in p_g])
        inv_t2_median_g = float(np.median(1.0 / t_g ** 2))
        n_c2_g = {c: 2.0 * eta ** 2 * float(theta_Q[c]) ** 2 * inv_t2_median_g for c in cols_g}

        # ---- joint outer search over log ell, within this group ------
        # One kernel and one eigendecomposition of Q^T K_ell Q per
        # width, shared across every coordinate in this group; each
        # coordinate profiles its own noise-to-signal ratio lam_c from
        # that shared eigendecomposition, and the outer objective is
        # the SUM of the group's per-coordinate profiled restricted NLLs.
        t_shared0 = time.perf_counter()
        trace: List[dict] = []

        def outer_obj(log_ell: float, _trace=trace, _cols_g=cols_g,
                       _psi_proj=psi_proj, _W=W_g, _tau=tau_g, _D_full=D_full_g,
                       _M_minus_m=M_minus_m, _n_c2=n_c2_g) -> float:
            ell = math.exp(log_ell)
            K = _matern32(_D_full, ell)
            # A = K - W (W^T K) - (W^T K)^T W^T + W (W^T K W + tau I) W^T,
            # the shifted projected kernel of the comment above.
            C = _W.T @ K                      # (m_g, M_g)
            E = C @ _W                        # (m_g, m_g)
            E[np.diag_indices_from(E)] += _tau
            WC = _W @ C                       # (M_g, M_g)
            A = K - WC - WC.T + _W @ (E @ _W.T)
            Lambda, V = np.linalg.eigh(A)
            Lambda = np.maximum(Lambda[:_M_minus_m], 0.0)
            V = V[:, :_M_minus_m]

            per_c: Dict[int, dict] = {}
            total_nll = 0.0
            for c in _cols_g:
                z = V.T @ _psi_proj[c]

                def inner(log_lam: float, _z=z, _Lambda=Lambda) -> float:
                    denom = _Lambda + math.exp(log_lam)
                    s2_val = np.sum(_z ** 2 / denom) / _M_minus_m
                    s2_val = max(s2_val, 1e-300)
                    return (_M_minus_m / 2.0) * math.log(s2_val) + 0.5 * np.sum(np.log(denom))

                # Declared noise floor (module docstring): the search
                # for lam_c is bounded below by lam_floor,c at THIS
                # candidate ell (Lambda, z both depend on ell), not by
                # the absolute floor alone.
                lam_floor_c = _lambda_floor(z, Lambda, _M_minus_m, _n_c2[c])
                lo_bound = max(math.log(lam_floor_c), _LOG_LAM_LO)
                if lo_bound >= _LOG_LAM_HI:
                    # The declared floor already pins lam_c at the
                    # ceiling: no interval left to search.
                    log_lam_star = _LOG_LAM_HI
                    nll_c = inner(log_lam_star)
                else:
                    ir = minimize_scalar(inner, bounds=(lo_bound, _LOG_LAM_HI), method='bounded')
                    log_lam_star = float(ir.x)
                    nll_c = float(ir.fun)
                per_c[c] = dict(log_lam=log_lam_star, nll=nll_c, lam_floor=lam_floor_c)
                total_nll += nll_c

            _trace.append(dict(log_ell=log_ell, ell=ell, K=K, per_c=per_c, nll=total_nll))
            return total_nll

        for le in grid_log_ell:
            outer_obj(float(le))

        grid_nlls = [t['nll'] for t in trace[:_N_WIDTH_GRID]]
        best_idx = int(np.argmin(grid_nlls))
        # Revision 10 (plan section 36.17(2), 20 September 2026): the
        # bounded refinement is SKIPPED when the best grid point is the
        # upper endpoint, and kept everywhere else. With the affine mean
        # projected out the Matern-3/2 expansion loses its constant and
        # its quadratic term exactly (Q^T 1 = 0 and Q^T X = 0), so the
        # leading survivor is sqrt(3) r^3 / ell^3: past about ell_max
        # the family collapses to ONE fixed kernel, the r^3 polyharmonic
        # spline with an affine null space, rescaled by s^2 / ell^3.
        # lambda_max(Q^T K Q) ell^3 is constant to three digits over
        # four decades there, so ell is not identified and the
        # refinement can only climb the shallow log-determinant tilt
        # left where the ell^-3 collapse meets the lambda floor -- 26 of
        # 31 outer evaluations on fp did exactly that, moving V_btw by
        # at most 0.08% and usually not at all. At an INTERIOR minimum
        # the refinement does real work and stays: on Pareto seed 0 it
        # moves ell from 14.47 to 5.99 and buys 42 nats. The LOWER
        # endpoint keeps it too -- that bound has never been hit.
        if best_idx == _N_WIDTH_GRID - 1:
            lo = hi = None
        elif best_idx == 0:
            lo, hi = grid_log_ell[0], grid_log_ell[1]
        else:
            lo, hi = grid_log_ell[best_idx - 1], grid_log_ell[best_idx + 1]

        if lo is not None and hi > lo:
            minimize_scalar(outer_obj, bounds=(float(lo), float(hi)), method='bounded')

        best = min(trace, key=lambda t: t['nll'])
        n_shared_evals = len(trace)
        t_shared_total = time.perf_counter() - t_shared0
        t_shared_share = t_shared_total / len(cols_g)

        ell_c = best['ell']
        K_c = best['K']

        # ---- the group's shared eigendecomposition of K --------------
        # ONE `eigh` of the chosen width's kernel, shared by every
        # coordinate in the group and by every later query of the
        # posterior (module docstring): A_c = K_c + (lam_c + jit_c) I
        # differs between the group's coordinates only through that
        # scalar, so V and Lambda serve all of them. Lambda is clipped
        # at 0 exactly as the outer search clips its own projected
        # spectrum -- lam_c is at least 1e-10, so Lambda + lam_c + jit_c
        # is strictly positive and the reciprocal below is safe.
        # O(M_g^3), the same cost class as the per-coordinate Cholesky
        # that follows, and paid once per group rather than per output.
        Lambda_K, V_K = np.linalg.eigh(K_c)
        Lambda_K = np.maximum(Lambda_K, 0.0)
        HbV_g = V_K.T @ Hb_g

        # ---- per-coordinate final solve (A depends on lam_c) ---------
        for c in cols_g:
            t0 = time.perf_counter()
            psi_c = I_proto[idx_g, c]
            lam_c = math.exp(best['per_c'][c]['log_lam'])
            lam_floor_final = best['per_c'][c]['lam_floor']

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
            lam_floor[c] = lam_floor_final
            s2[c] = s2_c
            jitter[c] = jit
            alpha[c] = alpha_c
            beta[c] = beta_c
            n_width_evals[c] = n_shared_evals
            at_bound[c, 0] = _within_1pct_log(ell_c, ell_min, ell_max)
            # lam_c's own search bound is [max(lam_floor_final, 1e-10),
            # 1e2] (module docstring), not the fixed [1e-10, 1e2]: the
            # declared floor, when active, moves the lower edge.
            at_bound[c, 1] = _within_1pct_log(lam_c, max(lam_floor_final, 1e-10), 1e2)

            chol_A[c] = chol
            g_chol[c] = G_chol_c
            ainv_hb[c] = AinvHb_c
            k_eigval[c] = Lambda_K
            k_eigvec[c] = V_K
            hb_eig[c] = HbV_g

            ml_wall_time[c] = t_shared_share + (time.perf_counter() - t0)

    model = InfluenceModel(
        alpha=alpha,
        beta=beta,
        centers=centers,
        whitening=(mean, transform),
        width=width,
        lam=lam,
        lam_floor=lam_floor,
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
        k_eigval=k_eigval,
        k_eigvec=k_eigvec,
        hb_eig=hb_eig,
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


def _coordinate_groups(model: InfluenceModel) -> List[List[int]]:
    """The non-constant coordinates grouped by shared design, in
    coordinate order. `_fit_gp_influence_model` assigns every
    coordinate in a group the SAME `centers` array object (and the same
    `k_eigvec`), so object identity is the group key -- no comparison of
    index sets is needed or wanted here. With no failed prototype
    evaluation this is one group holding every coordinate, which is the
    ordinary case."""
    groups: Dict[int, List[int]] = {}
    for c in range(len(model.centers)):
        if model.constant_path[c]:
            continue
        groups.setdefault(id(model.centers[c]), []).append(c)
    return list(groups.values())


def _point_terms(model, Z: np.ndarray) -> _PointTerms:
    """
    The one pass of the posterior over the N rows of Z (module
    docstring, "The per-point posterior terms are computed ONCE per
    draw"), cached on the model and keyed on the identity of `Z`.
    Returns psi0, sigma, the per-point affine-mean residuals r_i and --
    within the memory budget -- the kernel rows themselves, all of which
    `psi0`, `uncertainty` and `bin_posterior_variance` read instead of
    recomputing.

    psi0 and sigma are produced together because they share the one
    expensive object, the kernel rows k(x_i, w_j): every caller in this
    package asks for both on every draw, so forming them in one pass
    halves the kernel evaluations and drops one of the two whitening
    products. A caller who genuinely wants only psi0 pays for sigma too.

    The FITC model keeps its own `_fitc_psi0`/`_fitc_uncertainty`
    entirely -- only the caching of their results is shared, since those
    were each computed twice per draw as well.
    """
    cached = model._points
    if cached is not None and cached.Z is Z:
        return cached

    if isinstance(model, FITCInfluenceModel):
        q = len(model.rank)
        terms = _PointTerms(
            Z=Z, Zw=np.empty((0, 0), dtype=float),
            psi0=_fitc_psi0(model, Z), sigma=_fitc_uncertainty(model, Z),
            R=[None] * q, K=[None] * q,
        )
        model._points = terms
        return terms

    Za = np.asarray(Z, dtype=float)
    if Za.ndim == 1:
        Za = Za.reshape(-1, 1)
    mean, transform = model.whitening
    Zw = (Za - mean) @ transform.T
    N = Zw.shape[0]
    q = len(model.centers)

    psi0_out = np.empty((N, q), dtype=float)
    sigma_out = np.zeros((N, q), dtype=float)
    R_out: List[Optional[np.ndarray]] = [None] * q
    K_out: List[Optional[np.ndarray]] = [None] * q

    for c in range(q):
        if model.constant_path[c]:
            psi0_out[:, c] = model.const_value[c]

    groups = _coordinate_groups(model)

    # The kernel-row cache is all-or-nothing across the groups (module
    # docstring): one budget, so a draw either keeps every group's rows
    # or re-forms every group's rows, and no group's numbers depend on
    # how another group's memory came out.
    cache_bytes = sum(N * model.centers[g[0]].shape[0] for g in groups) * 8
    cache_rows = cache_bytes <= _KERNEL_CACHE_BYTES

    batch = _UNCERTAINTY_BATCH_CAP
    for cols in groups:
        c0 = cols[0]
        centers_g = model.centers[c0]
        M_g = centers_g.shape[0]
        ell_g = float(model.width[c0])
        m_g = int(model.m[c0])
        V_K = model.k_eigvec[c0]
        Lambda_K = model.k_eigval[c0]
        HbV_g = model.hb_eig[c0]

        # Per-coordinate constants of the eigen form: w_c = 1 /
        # (Lambda + lam_c + jit_c) is A_c^-1's spectrum, and B_c =
        # (V^T Hb) scaled by it turns the affine-mean correction into
        # one (nb, M) x (M, m) product against the shared P.
        w_by_c = {}
        B_by_c = {}
        for c in cols:
            w_c = 1.0 / (Lambda_K + float(model.lam[c]) + float(model.jitter[c]))
            w_by_c[c] = w_c
            B_by_c[c] = HbV_g * w_c[:, None]
            R_out[c] = np.empty((N, m_g), dtype=float)

        K_g = np.empty((N, M_g), dtype=float) if cache_rows else None
        for c in cols:
            K_out[c] = K_g

        for start in range(0, N, batch):
            sl = slice(start, start + batch)
            Zc = Zw[sl]
            Hc = _basis(Zc, m_g)
            Kc = _matern32(cdist(Zc, centers_g), ell_g)  # (nb, M_g)

            # psi0 is unchanged arithmetic: the same kernel rows, the
            # same alpha and beta from the Cholesky solve, the same
            # 4096-row batching. Bit-for-bit what it was.
            for c in cols:
                psi0_out[sl, c] = Hc @ model.beta[c] + Kc @ model.alpha[c]

            if K_g is not None:
                K_g[sl] = Kc

            Pc = Kc @ V_K  # (nb, M_g); ONE product for every output
            for c in cols:
                Rc = Hc - Pc @ B_by_c[c]  # (nb, m_g)
                R_out[c][sl] = Rc

            # Pc^2 goes into Kc's buffer: Kc has done its work above and
            # the two have identical shape, so the pass holds two
            # (nb, M_g) arrays at a time, the same peak the per-output
            # `cho_solve(chol, Kc.T)` it replaces used to hold.
            np.multiply(Pc, Pc, out=Kc)
            for c in cols:
                term1 = Kc @ w_by_c[c]
                Rc = R_out[c][sl]
                GinvRcT = cho_solve(model.g_chol[c], Rc.T)  # (m_c, nb)
                term2 = np.einsum('ij,ji->i', Rc, GinvRcT)
                sigma2 = model.s2[c] * (1.0 - term1 + term2)
                sigma2 = np.maximum(sigma2, 0.0)
                sigma_out[sl, c] = np.sqrt(sigma2)

    terms = _PointTerms(Z=Z, Zw=Zw, psi0=psi0_out, sigma=sigma_out,
                        R=R_out, K=K_out)
    model._points = terms
    return terms


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

    Computed by `_point_terms`, in the same pass as `uncertainty` and
    cached with it: the second call with the same `Z` costs nothing.
    That pass batches over rows of Z (`_UNCERTAINTY_BATCH_CAP` = 4096),
    for the same reason it always did -- unbatched, the (N, M_X_used)
    distance matrix and the kernel formed from it are each about 1.4 GB
    at the cost study's largest sample size (N = 76997, M_X_used ~
    2300). The arithmetic per row is identical either way, and
    identical to what it was before the pass was shared.

    THE RETURNED ARRAY IS THE CACHE, not a copy: callers read it, never
    write into it. Every caller in this package does (`qij.py` hands it
    to `QIJResult` and slices columns out of it; `refine.py` subtracts
    from it into a new array).

    `model` may be an `InfluenceModel` (full-rank GP) or a
    `FITCInfluenceModel` (sparse, `_fitc_psi0`) -- delegated inside
    `_point_terms` before any GP-specific code runs, so the GP path is
    untouched by FITC's existence.
    """
    return _point_terms(model, Z).psi0


def uncertainty(model: InfluenceModel, Z: np.ndarray) -> np.ndarray:
    """
    Posterior standard deviation sigma at Z (N, q) (R&W eq. 2.42):
    sigma_i^2 = s^2 * [1 - k_i^T A^-1 k_i + r_i^T G^-1 r_i], r_i =
    h(x_i) - Hb^T A^-1 k_i, all coordinate c's own -- its own design,
    own A, own G (module docstring). Clipped at 0 before the square
    root. Constant-path coordinates return 0.

    Both A^-1 terms come from the group's shared eigendecomposition
    K = V Lambda V^T (module docstring, "The per-point posterior terms
    are computed ONCE per draw") rather than from a per-output Cholesky
    of A_c: A_c = K + (lam_c + jit_c) I differs between a vector
    estimator's outputs only through that scalar, so P = K_n V is
    formed ONCE for every output and each output takes its own weighted
    row sums from it. Same quantity, different factorisation -- equal
    to the Cholesky form to rounding, not bit for bit.

    Cost: O(N * M_X_used^2) once per coordinate GROUP (the P product),
    plus O(N * M_X_used * m) per coordinate. Producing sigma at every
    data point is still a per-point product (QIJ_method_outline.md step
    4), not a per-bin one -- there is no bin-summed reformulation of it
    the way there is for `bin_posterior_variance`'s second term -- but
    the N * M^2 part of it is now paid once for the whole vector
    estimator instead of once per output.

    Computed and cached by `_point_terms`, in the same pass as `psi0`;
    the returned array is the cache, read-only by convention, exactly
    as `psi0`'s is.

    `model` may be an `InfluenceModel` (full-rank GP) or a
    `FITCInfluenceModel` (sparse, `_fitc_uncertainty`) -- delegated
    inside `_point_terms` before any GP-specific code runs.
    """
    return _point_terms(model, Z).sigma


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
    would repeat the O(N * M_X_used^2) product `uncertainty` already
    paid for. Squaring `sigma_c` recovers the same already-clipped-at-0
    value `uncertainty` produces.

    `mean(Sigma_k)` is computed from bin-summed vectors, never by
    holding an n_k x n_k matrix: with s = sum_{i in k} k_i (length
    M_c, coordinate `coordinate`'s own design size) and R = sum_{i in
    k} r_i (length m_c),

        mean(Sigma_k) = s2_c / n_k^2 * [SS_k - s^T A^-1 s + R^T G^-1 R]
        SS_k = sum_{i,j in k} k(x_i, x_j)

    `R` is a plain row sum of the per-point residuals `_point_terms`
    already computed and cached (module docstring) -- a refinement
    split's child bin never re-derives r_i for its own points. `s` is
    accumulated over the group's rows in chunks of at most 2048, from
    the cached kernel rows when the cache is on and by re-forming them
    when it is not (bit-identical either way). `SS_k` is the one term
    here that no cache removes: it is a sum of raw kernel values
    between the bin's OWN points, not a product against the design.
    Where the design is one-dimensional it is not summed pairwise at
    all -- `_matern32_self_sum_1d` gets it from running sums over the
    bin's sorted points, O(n_k log n_k) rather than O(n_k^2). In two
    dimensions or more the pairwise double sum stands, chunked on BOTH
    sides (a plain nested loop over chunk pairs, symmetry not
    exploited), so no block larger than 2048 x 2048 is ever
    materialized -- the naive one-sided chunking (chunk against the
    whole group) would still allocate a chunk_size x n_k block,
    unbounded in n_k.

    `s^T A^-1 s` is taken through the group's shared eigendecomposition
    K = V Lambda V^T, as sum_m (V^T s)_m^2 / (Lambda_m + lam_c +
    jit_c), not through a Cholesky solve of A_c -- deliberately, and
    not only for speed: v_k is `mean_diag - mean_Sigma`, a difference
    of two nearly equal quantities, and `mean_diag` comes from
    `sigma_c`, which `uncertainty` built from that same eigen form.
    Using one factorisation of A_c^-1 on both sides of the subtraction
    keeps the cancellation clean. `G^-1 R` is still one Cholesky solve
    on a bin-summed m_c-vector.

    Cost: O(L * M_X_used^2) in the one M x M product per bin (V against
    the bin-summed kernel vector), never repeated per point, plus O(L *
    m_c^2) for the G-solves; O(N * M_X_used) to form the bin-summed
    vector s (one cached kernel row per point, summed); and for the
    SS_k terms O(sum_k n_k log n_k) in one dimension, O(sum_k n_k^2)
    above it, a cost bounded by chunking in memory but not in M_X_used
    or in L -- this term does not depend on M_X_used at all, since it
    is a sum of raw kernel values, not a solve. Never
    O(N * M_X_used^2): no per-point solve against A is formed here.

    Returns 0.0 for a group of size <= 1, and 0.0 for every group
    when `model.constant_path[coordinate]` is true. Clipped at 0 from
    below.

    `model` may be an `InfluenceModel` (full-rank GP, below) or a
    `FITCInfluenceModel` (sparse, `_fitc_bin_posterior_variance`) --
    delegated on `isinstance` before any GP-specific code runs.
    """
    if isinstance(model, FITCInfluenceModel):
        return _fitc_bin_posterior_variance(model, Z, coordinate, groups, sigma_c)
    c = coordinate
    n_groups = len(groups)
    out = np.zeros(n_groups, dtype=float)
    if model.constant_path[c]:
        return out

    terms = _point_terms(model, Z)
    Zw_full = terms.Zw
    R_full = terms.R[c]
    K_full = terms.K[c]
    sigma_c = np.asarray(sigma_c, dtype=float)

    ell_c = float(model.width[c])
    s2_c = float(model.s2[c])
    g_chol_c = model.g_chol[c]
    centers_c = model.centers[c]
    M = centers_c.shape[0]
    V_K = model.k_eigvec[c]
    w_c = 1.0 / (model.k_eigval[c] + float(model.lam[c]) + float(model.jitter[c]))
    # A one-dimensional design takes SS_k from the sorted running sums
    # of `_matern32_self_sum_1d` instead of the pairwise double sum --
    # the O(sum_k n_k^2) term, and the only one here that no cache
    # removes. In two dimensions or more the pairwise path stands.
    one_dim = Zw_full.shape[1] == 1

    for gi, idx in enumerate(groups):
        idx = np.asarray(idx)
        n_k = idx.size
        if n_k <= 1:
            continue

        Zw_k = Zw_full[idx]

        mean_diag = float(np.mean(sigma_c[idx] ** 2))

        s_vec = np.zeros(M, dtype=float)
        SS_k = _matern32_self_sum_1d(Zw_k[:, 0], ell_c) if one_dim else 0.0

        for start in range(0, n_k, _BPV_CHUNK):
            sl = slice(start, start + _BPV_CHUNK)
            Zc = Zw_k[sl]

            if K_full is not None:
                Kc = K_full[idx[sl]]  # (nb, M), the rows already formed
            else:
                Kc = _matern32(cdist(Zc, centers_c), ell_c)  # (nb, M)

            s_vec += Kc.sum(axis=0)

            if one_dim:
                continue
            for start2 in range(0, n_k, _BPV_CHUNK):
                sl2 = slice(start2, start2 + _BPV_CHUNK)
                Dcc = cdist(Zc, Zw_k[sl2])
                Kcc = _matern32(Dcc, ell_c)
                SS_k += float(Kcc.sum())

        R_vec = R_full[idx].sum(axis=0)
        Vt_s = s_vec @ V_K
        quad_A = float(np.sum(Vt_s ** 2 * w_c))
        Ginv_R = cho_solve(g_chol_c, R_vec)
        mean_Sigma = (s2_c / (n_k ** 2)) * (SS_k - quad_A + R_vec @ Ginv_R)

        out[gi] = max(mean_diag - mean_Sigma, 0.0)

    return out


# ─────────────────────────────────────────────────────────────────────
# FITC (sparse, inducing-point) alternative (module docstring's "The
# FITC model"): a rank-r replacement for the M_c x M_c kernel, r <<
# M_c, selected by `fit_influence_model(..., influence_model='fitc')`.
# Every solve below goes through the Woodbury identity, so nothing
# M_c x M_c is ever formed or factorised -- only r x r (K_mm, B) and
# m_c x m_c (G) objects, plus O(M_c * r) intermediates at fit time.
#
# Notation (matches the report handed back with this change): centers_g
# is the group's M_g finite prototypes; Z_ind (r, d_z) the inducing
# points, a subset of centers_g; K_mm = k(Z_ind, Z_ind) (r, r); K_nm =
# k(centers_g, Z_ind) (M_g, r); D0 = diag(K_nn) - diag(Q_nn), Q_nn =
# K_nm K_mm^-1 K_mn (M_g,), independent of lam_c -- diag(K_nn) is
# exactly 1 everywhere for the Matern-3/2 kernel (k(0) = 1), so this is
# an EXACT quantity, not itself an approximation. D(lam) = D0 + lam
# (M_g,); B(lam) = K_mm + K_nm^T diag(1/D) K_nm (r, r) is the one
# object factorised per lambda trial (`_cholesky_with_jitter`'s own
# jitter policy, base scaled by K_mm's own trace). The determinant
# identity used throughout (Sylvester's lemma on A_fitc = diag(D) +
# K_nm K_mm^-1 K_mn):
#
#     log det(A_fitc) = sum(log D) + log det(B) - log det(K_mm)
#
# lets the REML criterion be evaluated from D, K_mm and B alone --
# never from an (M_g - m_g)-dimensional eigendecomposition the way the
# GP's shared `Q^T K Q` eigh is. A parallel identity (the FITC
# generative model's u-marginal, derived in the report) gives the
# predictive covariance between any two points i, j -- training,
# query, or both -- entirely through r x r and m_c x r objects:
#
#     Cov(f_i, f_j) = delta_ij * (k_ii - Q_ii) + k_mi^T K_mm^-1 k_mj
#     Cov(f_i, f_j | y) = delta_ij * (k_ii - Q_ii) + k_mi^T W k_mj
#         W := K_mm^-1 - B^-1          (r, r), PSD (B >= K_mm)
#
# so `k_i^T A^-1 k_j` (the GP's per-pair correction term, an M_c-length
# inner product) becomes `k_mi^T W k_mj`, an r-length one, for ANY pair
# of points -- diagonal (i = j, `uncertainty`) or off-diagonal (i != j,
# `bin_posterior_variance`'s double sum) alike. The affine-mean
# correction is handled the same way: E := Hb_g^T A_fitc^-1 Phi (m_c,
# r), Phi := K_nm K_mm^-1, so r_i = h(x_i) - E @ k_mi replaces r_i =
# h(x_i) - Hb^T A^-1 k_i without ever touching an M_c-length k_i.
# ─────────────────────────────────────────────────────────────────────


def _farthest_point_inducing(centers: np.ndarray, r: int) -> np.ndarray:
    """r inducing points chosen from `centers` (M, d_z) by DETERMINISTIC
    farthest-point traversal (module docstring's "Inducing points"): the
    first point is the one nearest the design's centroid; each further
    point maximizes the minimum squared distance to the points already
    chosen. No RNG, no seed, no tie-breaking rule needed beyond
    `argmin`/`argmax`'s own leftmost-index convention on a fixed array,
    which is itself deterministic. r is clipped to M (a group smaller
    than the requested rank simply uses every one of its own points,
    reducing FITC to an exact, if pointless, reparametrization of the
    same design).

    Cost: O(r * M) (one running min-distance array, updated once per
    chosen point), done once per coordinate group, not per candidate
    ell -- inducing points are a geometric choice, not a function of
    the kernel width.
    """
    M = centers.shape[0]
    r = min(r, M)
    centroid = centers.mean(axis=0)
    d_to_centroid = np.sum((centers - centroid) ** 2, axis=1)
    first = int(np.argmin(d_to_centroid))
    chosen = [first]
    min_dist2 = np.sum((centers - centers[first]) ** 2, axis=1)
    min_dist2[first] = -np.inf
    for _ in range(1, r):
        nxt = int(np.argmax(min_dist2))
        chosen.append(nxt)
        d2 = np.sum((centers - centers[nxt]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, d2)
        min_dist2[nxt] = -np.inf
    return np.array(chosen, dtype=np.intp)


def _fitc_profile(
    log_lam: float, psi_c: np.ndarray, K_nm: np.ndarray, K_mm: np.ndarray,
    cholKmm: Tuple, D0: np.ndarray, Hb_g: np.ndarray, M_minus_m: int, r: int,
) -> dict:
    """The FITC analogue of the GP's `inner`: the exact profiled REML
    criterion at one candidate lam_c, given the ell-fixed, group-shared
    (and, for K_nm/K_mm/cholKmm/D0, coordinate-shared) pieces -- see the
    section docstring above for the determinant identity this uses to
    avoid ever forming an M_g x M_g matrix or an (M_g - m_g)-dimensional
    eigendecomposition. `psi_c` and `Hb_g` are solved against A_fitc(lam)
    together, in one Woodbury-applied batch.

    UNLIKE the GP's `inner` (O(M_g - m_g) per lambda trial, reusing one
    eigendecomposition shared across every trial and every coordinate
    in the group), this function is O(M_g * r^2) per trial: D0 varies
    per training point, so D(lam)^-1 = 1/(D0 + lam) is not a uniform
    shift of a fixed eigenbasis the way `Lambda + lam` is in the GP,
    and no lambda-independent, M_g-independent reduction of B(lam)
    exists in general (see the report handed back with this change).
    This is still far cheaper than the O(M_g^3) eigh it replaces --
    the reduction from M_g^3 to M_g*r^2 is FITC's actual saving here,
    not a reduction to O(1) per lambda trial -- and it is exact, not a
    coarser search: every quantity below is the same REML criterion
    the GP profiles, evaluated through a different, equivalent route.
    """
    lam = math.exp(log_lam)
    D = D0 + lam
    Dinv = 1.0 / D
    Kw = K_nm * Dinv[:, None]
    C = K_nm.T @ Kw
    B = K_mm + C
    # Jitter base is B's OWN trace here, not K_mm's (the GP's
    # `_cholesky_with_jitter(A, K, ...)` calls all use the PRE-noise
    # kernel as the jitter reference, since there A = K + lam*I never
    # leaves K's own O(1) scale, lam bounded to [1e-10, 1e2]). B has no
    # such bound: every inducing point coincides EXACTLY with one of
    # its own training points (D0 = 0 there, exactly, by construction --
    # diag(Q_nn) = diag(K_nn) = 1 at an inducing point itself), so at
    # lam near the absolute floor 1e-10, 1/D reaches ~1e10 at that row
    # and C = K_nm^T diag(1/D) K_nm can be many orders of magnitude
    # larger than K_mm. A jitter scaled to K_mm's O(1) trace is then far
    # too small to cover the roundoff in forming that huge C, so the
    # jitter reference here tracks B's own scale instead.
    cholB, jitB = _cholesky_with_jitter(B, B, r)

    RHS = np.column_stack([psi_c, Hb_g])
    Dinv_rhs = RHS * Dinv[:, None]
    tmp = K_nm.T @ Dinv_rhs
    corr = cho_solve(cholB, tmp)
    Ainv_rhs = Dinv_rhs - (K_nm @ corr) * Dinv[:, None]
    Ainv_psi = Ainv_rhs[:, 0]
    Ainv_Hb = Ainv_rhs[:, 1:]

    u_vec = Hb_g.T @ Ainv_psi
    G = Hb_g.T @ Ainv_Hb
    g_chol = cho_factor(G, lower=True)
    beta = cho_solve(g_chol, u_vec)

    resid_quad = float(psi_c @ Ainv_psi - u_vec @ beta)
    s2_hat = max(resid_quad, 0.0) / M_minus_m if M_minus_m > 0 else 0.0

    logdetA = (
        float(np.sum(np.log(D)))
        + 2.0 * float(np.sum(np.log(np.diag(cholB[0]))))
        - 2.0 * float(np.sum(np.log(np.diag(cholKmm[0]))))
    )
    logdetG = 2.0 * float(np.sum(np.log(np.diag(g_chol[0]))))
    s2_for_log = max(s2_hat, 1e-300)
    nll = (M_minus_m / 2.0) * math.log(s2_for_log) + 0.5 * logdetA + 0.5 * logdetG

    return dict(
        nll=nll, s2=s2_hat, beta=beta, Ainv_Hb=Ainv_Hb, cholB=cholB, jitB=jitB,
        D=D,
    )


def _fitc_lambda_floor(profile_fn, n_c2: float) -> float:
    """The declared-noise floor lam_floor,c for one coordinate under
    FITC, at one candidate ell: the unique root in log lam of
    lam*s2_hat(lam) = n_c2 (module docstring, plan §36.2(2)) -- the same
    monotonicity argument `_lambda_floor` relies on, generic to any
    A(lam) = A0 + lam*I with A0 PSD (it does not depend on A0's
    structure, only that lam enters as a uniform additive shift), here
    applied to `_fitc_profile`'s s2_hat instead of the GP's closed-form
    eigenvalue sum. One `brentq` call, pinned to the search domain's own
    edges exactly as `_lambda_floor` is."""
    lo, hi = _LOG_LAM_LO, _LOG_LAM_HI

    def g(log_lam: float) -> float:
        return math.exp(log_lam) * profile_fn(log_lam)['s2'] - n_c2

    g_lo = g(lo)
    if g_lo >= 0.0:
        return math.exp(lo)
    g_hi = g(hi)
    if g_hi <= 0.0:
        return math.exp(hi)
    root = brentq(g, lo, hi)
    return math.exp(root)


def _fitc_default_rank(M_c: int) -> int:
    """min(M_c, max(32, ceil(M_c / 6))) (module docstring's "The
    switch"), the default FITC rank when `fitc_rank` is not given,
    computed from that coordinate's (or, for a shared design, that
    coordinate group's) own finite design size."""
    return min(M_c, max(32, math.ceil(M_c / 6)))


def _fit_fitc_influence_model(
    Z: np.ndarray, xvq: XVQ, I_proto: np.ndarray, theta_Q: np.ndarray, eta: float,
    fitc_rank: Optional[int],
) -> FITCInfluenceModel:
    """
    The FITC sparse alternative to `_fit_gp_influence_model`: identical
    design construction (whitening, per-coordinate finite design,
    grouping by shared finite index set, the affine/constant basis
    choice, the declared noise floor) -- this preamble is a deliberate
    copy of `_fit_gp_influence_model`'s, not a shared helper, so a
    refactor of one can never silently change the other -- with the
    kernel algebra alone replaced: inducing points chosen once per
    group by `_farthest_point_inducing`, the shared outer search over
    log ell profiling each coordinate's lam_c through `_fitc_profile`
    (the r x r analogue of the GP's shared eigendecomposition of
    Q^T K_ell Q), and a final per-coordinate solve that stores only
    r_c- and m_c-scale objects (`FITCInfluenceModel`'s docstring) --
    never an M_c-length weight vector or an M_c x M_c factor.

    `fitc_rank`: None means "use `_fitc_default_rank(M_g)` for every
    group" (module docstring); given, it is used as every group's rank,
    clipped to that group's own M_g (a group cannot have more inducing
    points than training points).

    Cost: O(M_X_used * r^2 + r^3) per shared outer-objective evaluation
    (forming B(lam) and its Cholesky), replacing the GP's O(M_X_used^3)
    eigh -- see `_fitc_profile`'s docstring for why the inner lambda
    search itself is not reducible to an M-independent cost the way the
    GP's is. `psi0`/`uncertainty`/`bin_posterior_variance` are O(r) /
    O(r^2) / O(r^2) per point thereafter, never O(M_X_used) or
    O(M_X_used^2).
    """
    p = np.asarray(xvq.p, dtype=float)
    I_proto = np.asarray(I_proto, dtype=float)
    theta_Q = np.asarray(theta_Q, dtype=float)
    M_X_used, q = I_proto.shape
    raw_centers = np.asarray(xvq.centers, dtype=float)
    d_z = raw_centers.shape[1]

    delta_f = forward_step(eta)

    mean, transform = _whitening_from(Z)
    centers_full = (raw_centers - mean) @ transform.T

    finite = np.isfinite(I_proto)

    constant_path = np.zeros(q, dtype=bool)
    const_value = np.zeros(q, dtype=float)
    offset = np.zeros(q, dtype=float)
    m_arr = np.zeros(q, dtype=int)

    beta: List[np.ndarray] = [np.empty(0, dtype=float)] * q
    width = np.full(q, np.nan, dtype=float)
    lam = np.full(q, np.nan, dtype=float)
    lam_floor = np.full(q, np.nan, dtype=float)
    s2 = np.zeros(q, dtype=float)
    at_bound = np.zeros((q, 2), dtype=bool)
    jitter = np.full(q, np.nan, dtype=float)
    n_width_evals = np.zeros(q, dtype=int)
    ml_wall_time = np.zeros(q, dtype=float)
    rank = np.zeros(q, dtype=int)

    centers_ind: List[np.ndarray] = [np.empty((0, d_z), dtype=float)] * q
    inducing_idx: List[np.ndarray] = [np.empty(0, dtype=np.intp)] * q
    alpha_m: List[np.ndarray] = [np.empty(0, dtype=float)] * q
    W: List[Optional[np.ndarray]] = [None] * q
    E: List[Optional[np.ndarray]] = [None] * q
    g_chol: List[Optional[Tuple]] = [None] * q
    kmm_chol: List[Optional[Tuple]] = [None] * q

    groups: Dict[tuple, List[int]] = {}
    for c in range(q):
        idx_c = np.where(finite[:, c])[0]
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

        M_minus_m = M_g - m_g

        r_default = _fitc_default_rank(M_g)
        r_g = min(int(fitc_rank), M_g) if fitc_rank is not None else r_default
        r_g = max(r_g, 1)
        ind_local = _farthest_point_inducing(centers_g, r_g)
        centers_ind_g = centers_g[ind_local]

        p_g = p[idx_g]
        t_g = np.array([step_parameter(delta_f, float(pj)) for pj in p_g])
        inv_t2_median_g = float(np.median(1.0 / t_g ** 2))
        n_c2_g = {c: 2.0 * eta ** 2 * float(theta_Q[c]) ** 2 * inv_t2_median_g for c in cols_g}

        psi_by_c = {c: I_proto[idx_g, c] for c in cols_g}

        # ---- joint outer search over log ell, within this group ------
        # One K_mm/K_nm pair per width, shared across every coordinate
        # in this group -- the FITC analogue of the GP's shared kernel
        # and eigendecomposition, computed once per candidate ell.
        t_shared0 = time.perf_counter()
        trace: List[dict] = []

        def outer_obj(log_ell: float, _trace=trace, _cols_g=cols_g, _psi=psi_by_c,
                      _centers_g=centers_g, _centers_ind=centers_ind_g, _Hb_g=Hb_g,
                      _M_minus_m=M_minus_m, _n_c2=n_c2_g, _r=r_g) -> float:
            ell = math.exp(log_ell)
            K_mm = _matern32(cdist(_centers_ind, _centers_ind), ell)
            K_nm = _matern32(cdist(_centers_g, _centers_ind), ell)
            cholKmm, jit_mm = _cholesky_with_jitter(K_mm, K_mm, _r)
            V = solve_triangular(cholKmm[0], K_nm.T, lower=True)  # (r, M_g)
            diag_Qnn = np.sum(V ** 2, axis=0)
            # D0 is EXACTLY 0 at every inducing point's own index (an
            # inducing point's K_nm row exactly reproduces the matching
            # K_mm row, so diag_Qnn = 1 there, to machine precision) --
            # by construction, on every FITC fit, not a data-dependent
            # edge case. Left unfloored, D = D0 + lam can sit at lam's
            # own absolute floor (1e-10), so 1/D reaches ~1e10 at those
            # rows and swamps the Woodbury solve's floating-point
            # precision (observed: G = Hb^T A_fitc^-1 Hb losing positive
            # definiteness numerically, though it is PD in exact
            # arithmetic whenever Hb has full column rank). Floored at
            # `_FITC_D0_FLOOR` = 1e-6 -- eight orders of magnitude below
            # the kernel's own unit scale, negligible against every lam
            # candidate except the handful nearest the absolute floor,
            # where it keeps 1/D under ~1e6 instead of ~1e10. The same
            # kind of deterministic, scale-appropriate numerical
            # safeguard `_cholesky_with_jitter` already applies
            # throughout this module, not a modeling choice.
            D0 = np.maximum(1.0 - diag_Qnn, _FITC_D0_FLOOR)

            per_c: Dict[int, dict] = {}
            total_nll = 0.0
            for c in _cols_g:
                psi_c = _psi[c]

                def profile(log_lam: float, _psi_c=psi_c, _K_nm=K_nm, _K_mm=K_mm,
                            _cholKmm=cholKmm, _D0=D0, _Hb=_Hb_g, _Mm=_M_minus_m,
                            _rr=_r) -> dict:
                    return _fitc_profile(log_lam, _psi_c, _K_nm, _K_mm, _cholKmm,
                                         _D0, _Hb, _Mm, _rr)

                lam_floor_c = _fitc_lambda_floor(profile, _n_c2[c])
                lo_bound = max(math.log(lam_floor_c), _LOG_LAM_LO)
                if lo_bound >= _LOG_LAM_HI:
                    log_lam_star = _LOG_LAM_HI
                    prof = profile(log_lam_star)
                else:
                    ir = minimize_scalar(lambda lg: profile(lg)['nll'],
                                          bounds=(lo_bound, _LOG_LAM_HI), method='bounded')
                    log_lam_star = float(ir.x)
                    prof = profile(log_lam_star)
                per_c[c] = dict(log_lam=log_lam_star, nll=prof['nll'], lam_floor=lam_floor_c)
                total_nll += prof['nll']

            _trace.append(dict(log_ell=log_ell, ell=ell, K_mm=K_mm, K_nm=K_nm,
                               cholKmm=cholKmm, jit_mm=jit_mm, D0=D0,
                               per_c=per_c, nll=total_nll))
            return total_nll

        for le in grid_log_ell:
            outer_obj(float(le))

        grid_nlls = [t['nll'] for t in trace[:_N_WIDTH_GRID]]
        best_idx = int(np.argmin(grid_nlls))
        # The upper-endpoint skip of revision 10, for the same reason it
        # is taken on the dense path above (plan section 36.17(2)): past
        # about ell_max the projected Matern-3/2 family collapses to the
        # r^3 polyharmonic spline and ell stops being identified, so the
        # refinement there buys nothing. Kept in step with the dense path
        # deliberately -- this influence model is built and committed but
        # not adopted, and two different width-search rules in one file
        # is exactly the trap that would be found the hard way if it ever
        # were switched on.
        if best_idx == _N_WIDTH_GRID - 1:
            lo = hi = None
        elif best_idx == 0:
            lo, hi = grid_log_ell[0], grid_log_ell[1]
        else:
            lo, hi = grid_log_ell[best_idx - 1], grid_log_ell[best_idx + 1]

        if lo is not None and hi > lo:
            minimize_scalar(outer_obj, bounds=(float(lo), float(hi)), method='bounded')

        best = min(trace, key=lambda t: t['nll'])
        n_shared_evals = len(trace)
        t_shared_total = time.perf_counter() - t_shared0
        t_shared_share = t_shared_total / len(cols_g)

        ell_c = best['ell']
        K_mm = best['K_mm']
        K_nm = best['K_nm']
        cholKmm = best['cholKmm']
        jit_mm = best['jit_mm']
        D0 = best['D0']

        for c in cols_g:
            t0 = time.perf_counter()
            psi_c = I_proto[idx_g, c]
            lam_c = math.exp(best['per_c'][c]['log_lam'])
            lam_floor_final = best['per_c'][c]['lam_floor']

            prof = _fitc_profile(math.log(lam_c), psi_c, K_nm, K_mm, cholKmm, D0,
                                 Hb_g, M_minus_m, r_g)
            beta_c = prof['beta']
            s2_c = prof['s2']
            cholB = prof['cholB']
            D = prof['D']

            resid = psi_c - Hb_g @ beta_c
            w = resid / D
            rhs_m = K_nm.T @ w
            alpha_m_c = cho_solve(cholB, rhs_m)

            eye_r = np.eye(r_g)
            Kmm_inv = cho_solve(cholKmm, eye_r)
            Binv = cho_solve(cholB, eye_r)
            W_c = Kmm_inv - Binv

            tmp = prof['Ainv_Hb'].T @ K_nm       # (m_g, r)
            E_c = cho_solve(cholKmm, tmp.T).T    # (m_g, r)

            centers_ind[c] = centers_ind_g
            inducing_idx[c] = ind_local
            rank[c] = r_g
            m_arr[c] = m_g
            width[c] = ell_c
            lam[c] = lam_c
            lam_floor[c] = lam_floor_final
            s2[c] = s2_c
            jitter[c] = max(jit_mm, prof['jitB'])
            beta[c] = beta_c
            alpha_m[c] = alpha_m_c
            W[c] = W_c
            E[c] = E_c
            n_width_evals[c] = n_shared_evals
            at_bound[c, 0] = _within_1pct_log(ell_c, ell_min, ell_max)
            at_bound[c, 1] = _within_1pct_log(lam_c, max(lam_floor_final, 1e-10), 1e2)

            # G's Cholesky is recomputed fresh from `prof` (Hb_g^T
            # A_fitc^-1 Hb_g at THIS coordinate's own final lam_c),
            # mirroring the GP's per-coordinate `g_chol` -- `_fitc_
            # profile` already forms it via `cho_factor(G, ...)`
            # internally, but does not return it, so it is rebuilt
            # here from the same Ainv_Hb it returns (one m_g x m_g
            # factorisation, negligible cost).
            G_c = Hb_g.T @ prof['Ainv_Hb']
            g_chol[c] = cho_factor(G_c, lower=True)
            kmm_chol[c] = cholKmm

            ml_wall_time[c] = t_shared_share + (time.perf_counter() - t0)

    model = FITCInfluenceModel(
        beta=beta,
        whitening=(mean, transform),
        width=width,
        lam=lam,
        lam_floor=lam_floor,
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
        centers_ind=centers_ind,
        inducing_idx=inducing_idx,
        rank=rank,
        alpha_m=alpha_m,
        W=W,
        E=E,
        g_chol=g_chol,
        kmm_chol=kmm_chol,
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


def _fitc_psi0(model: FITCInfluenceModel, Z: np.ndarray) -> np.ndarray:
    """FITC's `psi0`: psi0(x) = h(x)^T beta + K(x, centers_ind) @
    alpha_m, alpha_m the (r_c,) low-rank prediction weights (section
    docstring). Same batching, same constant-path handling, same
    public contract as `psi0` -- called only from there.

    Cost: O(N * r) per coordinate -- never O(N * M_X_used)."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    mean, transform = model.whitening
    Zw = (Z - mean) @ transform.T
    N = Zw.shape[0]
    q = len(model.centers_ind)

    out = np.empty((N, q), dtype=float)
    batch = _UNCERTAINTY_BATCH_CAP
    for start in range(0, N, batch):
        sl = slice(start, start + batch)
        Zc = Zw[sl]
        for c in range(q):
            if model.constant_path[c]:
                out[sl, c] = model.const_value[c]
            else:
                Hc = _basis(Zc, int(model.m[c]))
                Dc = cdist(Zc, model.centers_ind[c])
                Kmc = _matern32(Dc, float(model.width[c]))  # (nb, r_c)
                out[sl, c] = Hc @ model.beta[c] + Kmc @ model.alpha_m[c]
    return out


def _fitc_uncertainty(model: FITCInfluenceModel, Z: np.ndarray) -> np.ndarray:
    """FITC's `uncertainty`: sigma_i^2 = s2_c * [1 - k_mi^T W k_mi +
    r_i^T G^-1 r_i], r_i = h(x_i) - E @ k_mi (section docstring's
    identity, exact under the FITC generative model, with k(x_i,x_i) =
    1 exactly for the Matern-3/2 kernel -- no Nystrom approximation on
    the diagonal). Same batching, same constant-path handling, same
    public contract as `uncertainty`.

    Cost: O(N * r^2) per coordinate -- never O(N * M_X_used^2)."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    mean, transform = model.whitening
    Zw = (Z - mean) @ transform.T
    N = Zw.shape[0]
    q = len(model.centers_ind)

    out = np.zeros((N, q), dtype=float)
    batch = _UNCERTAINTY_BATCH_CAP
    for start in range(0, N, batch):
        sl = slice(start, start + batch)
        Zc = Zw[sl]

        for c in range(q):
            if model.constant_path[c]:
                continue
            Hc = _basis(Zc, int(model.m[c]))
            Dc = cdist(Zc, model.centers_ind[c])
            Kmc = _matern32(Dc, float(model.width[c]))  # (nb, r_c)

            term1 = np.einsum('ij,jk,ik->i', Kmc, model.W[c], Kmc)

            Rc = Hc - Kmc @ model.E[c].T  # (nb, m_c)
            GinvRcT = cho_solve(model.g_chol[c], Rc.T)
            term2 = np.einsum('ij,ji->i', Rc, GinvRcT)

            sigma2 = model.s2[c] * (1.0 - term1 + term2)
            sigma2 = np.maximum(sigma2, 0.0)
            out[sl, c] = np.sqrt(sigma2)
    return out


def _fitc_bin_posterior_variance(
    model: FITCInfluenceModel,
    Z: np.ndarray,
    coordinate: int,
    groups: Sequence[np.ndarray],
    sigma_c: np.ndarray,
) -> np.ndarray:
    """FITC's `bin_posterior_variance`: the same v_k = mean(diag Sigma_k)
    - mean(Sigma_k) decomposition, `mean(diag Sigma_k)` read from the
    caller's `sigma_c` exactly as the GP path does, and `mean(Sigma_k)`
    built from BIN-SUMMED vectors -- never an n_k x n_k block, and,
    under FITC, never even an n_k x M_X_used block (module docstring:
    "the double sum of the prior term becomes a quadratic form in the
    bin's summed cross-kernel rows"). With s_m = sum_{i in k} k_mi
    (length r_c, the bin's summed cross-kernel to the inducing points)
    and R = sum_{i in k} r_i (length m_c, r_i as in `_fitc_uncertainty`):

        mean(Sigma_k) = s2_c / n_k^2 * [SS_k - s_m^T W s_m + R^T G^-1 R]
        SS_k = s_m^T K_mm^-1 s_m + sum_{i in k} (1 - ||L_mm^-1 k_mi||^2)

    SS_k's second term is each point's own diagonal correction (D0_i,
    section docstring), summed in O(n_k * r_c) per bin via one
    triangular solve per chunk -- cheaper than the GP's chunked O(n_k^2)
    double sum, not merely never-larger.

    Cost: O(L * r_c^2) in the solves (one K_mm-solve and one G-solve
    per bin); O(N * r_c) to form s_m, R and the diagonal-correction sum.
    Never O(N * M_X_used) or O(sum_k n_k^2): no quantity here scales
    with M_X_used or with n_k^2."""
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
    W_c = model.W[c]
    E_c = model.E[c]
    g_chol_c = model.g_chol[c]
    kmm_chol_c = model.kmm_chol[c]
    m_c = int(model.m[c])
    centers_ind_c = model.centers_ind[c]
    r_c = centers_ind_c.shape[0]

    for gi, idx in enumerate(groups):
        idx = np.asarray(idx)
        n_k = idx.size
        if n_k <= 1:
            continue

        Zw_k = Zw_full[idx]
        H_k = _basis(Zw_k, m_c)

        mean_diag = float(np.mean(sigma_c[idx] ** 2))

        s_m = np.zeros(r_c, dtype=float)
        H_sum = np.zeros(m_c, dtype=float)
        diag_corr_sum = 0.0

        for start in range(0, n_k, _BPV_CHUNK):
            sl = slice(start, start + _BPV_CHUNK)
            Zc = Zw_k[sl]
            Hc = H_k[sl]

            Dc = cdist(Zc, centers_ind_c)
            Kmc = _matern32(Dc, ell_c)  # (nb, r_c)

            s_m += Kmc.sum(axis=0)
            H_sum += Hc.sum(axis=0)

            V = solve_triangular(kmm_chol_c[0], Kmc.T, lower=True)  # (r_c, nb)
            diag_corr_sum += float(np.sum(1.0 - np.sum(V ** 2, axis=0)))

        R_vec = H_sum - E_c @ s_m

        Kmm_inv_sm = cho_solve(kmm_chol_c, s_m)
        SS_k = float(s_m @ Kmm_inv_sm) + diag_corr_sum

        term2_sum = float(s_m @ (W_c @ s_m))
        Ginv_R = cho_solve(g_chol_c, R_vec)
        term3_sum = float(R_vec @ Ginv_R)

        mean_Sigma = (s2_c / (n_k ** 2)) * (SS_k - term2_sum + term3_sum)
        out[gi] = max(mean_diag - mean_Sigma, 0.0)

    return out
