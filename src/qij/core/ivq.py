"""
The 𝓘-VQ: bins of the initial influence estimate psi0_c, one
quantizer per estimand coordinate, and each bin's three-point-stencil
measurement on the full data (the count rule and 1-D k-means of the
method outline; the one difference rule of `core.differences`).

`build_bins` runs 1-D k-means on the raw psi0_c values at the initial
count `M_init = min(ceil(sqrt(2.7/eps)), n_distinct)`. `bin_differences`
then measures the resulting bins on the full data with
`core.differences.difference` (central where feasible, one-sided
otherwise -- the natural rule, not forced), giving each bin's
finite-differenced influence U_k and second difference d2T_k.
`between_terms` reduces those to the between-bin variance V_btw and
the between-bin curvature term B_hat.

`kmeans_1d` (Lloyd's iteration on a sorted 1-D array) lives here, not
in `refine.py`, because both this module's `build_bins` and
`refine.py`'s within-bin level split use the same one-dimensional
quantizer: one implementation of the rule, called from both places.

A failed evaluation of an initial bin (ruled 18 September) makes the
whole draw's QIJ result for this coordinate a write-off: the
measurement of the initial bins is what the between-bin term and
everything after it rests on, so there is nothing to salvage.
`bin_differences` detects this the same way `core.counter.Counter`
does (any NaN in a bin's finite-differenced `U`/`d2T`), stops at the
bin that failed -- no further bins are measured, nothing retried --
and returns a `BinSet` with `failed=True` and `U`/`d2T` all NaN.
`refine.run_refinement` turns that into a NaN `CoordinateResult` whose
diagnostics carry what stage 2 established before the failure (the
bin count `M_used`; the evaluations actually spent are `Counter`'s own
count, not reported here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .differences import central_step, difference, perturbed_weights

__all__ = ["BinSet", "kmeans_1d", "within_share", "build_bins", "bin_differences", "between_terms"]


@dataclass
class BinSet:
    """
    One estimand coordinate's 𝓘-VQ: the initial bins over psi0_c and
    their full-data measurement.

    labels        (N,) int, bin index per data point, contiguous
                  0..M_used-1.
    n             (M_used,) int, bin sizes, all > 0.
    U             (M_used, q) finite-differenced bin influences,
                  centered by the bin-mass-weighted residual
                  (`bin_differences`); shape (M_used, 0) before
                  `bin_differences` has run.
    d2T           (M_used, q) finite-differenced bin second
                  differences, uncentered; consumed only by
                  `between_terms`'s B_hat. Kept alongside U in the same
                  object because `between_terms` receives only a
                  `BinSet` and must produce both V_btw and B_hat from
                  it.
    centering_residual  (q,) the bin-mass-weighted residual
                  sum_k p_k U_k subtracted out of U before it was
                  stored; `refine.run_refinement` subtracts this same
                  residual from every forward-differenced split it
                  measures afterwards, so a split's U stays on the
                  same centered scale as the initial bins' U. Zero
                  (shape (0, q) sized to M_used == 1) when M_used == 1.
    M_init        the initial bin count from the count rule.
    M_used        bins actually used (<= M_init; empty bins are
                  dropped by `kmeans_1d`).
    within_share  achieved share of Var(psi0_c) left inside the bins
                  (0 when M_used == 1).
    failed        True when a bin's full-data evaluation returned any
                  NaN (ruled 18 September: a failed evaluation of an
                  initial bin, stage 2). `U`/`d2T`/`centering_residual`
                  are then all NaN and MUST NOT be read by
                  `between_terms`; `refine.run_refinement` checks this
                  flag first and returns a NaN `CoordinateResult`
                  instead. False from `build_bins` (which never
                  evaluates T) and from the single-bin shortcut of
                  `bin_differences` (which also never evaluates T).
    """

    labels: np.ndarray
    n: np.ndarray
    U: np.ndarray
    d2T: np.ndarray
    centering_residual: np.ndarray
    M_init: int
    M_used: int
    within_share: float
    failed: bool = False

    @property
    def p(self) -> np.ndarray:
        """Bin masses n / N."""
        return self.n / self.labels.size


def kmeans_1d(values: np.ndarray, M: int):
    """
    Lloyd's iteration on the sorted 1-D array `values`, with M target
    prototypes.

    Initialization: the M quantiles at levels (k + 1/2)/M of the
    distinct sorted values u = np.unique(values); 1 <= M <= u.size is
    required, which guarantees the initial prototypes are distinct and
    strictly increasing.

    Assignment: interior boundaries are the midpoints between
    consecutive prototypes; label = np.searchsorted(mids, v,
    side='left') (a value exactly on a boundary goes to the lower
    prototype).

    Update: each prototype becomes the mean of its bin. A prototype
    whose bin is empty after an assignment step is dropped immediately
    and the remaining prototypes are relabelled contiguously, so no
    mean is ever undefined.

    Convergence: iterate assignment+update until the assignment (post
    drop-and-relabel) is unchanged from the previous iteration, or 100
    iterations.

    Cost: O(M) per iteration for the update (a loop over prototypes,
    never over the N points -- assignment and the per-prototype mean
    are vectorized numpy operations over the N-length array), at most
    100 iterations.

    Returns (labels, prototypes): labels (N,) int, contiguous
    0..M_used-1, ordered by increasing prototype value; prototypes
    (M_used,), strictly increasing.
    """
    v = np.asarray(values, dtype=float).ravel()
    u = np.unique(v)
    M = int(M)

    prototypes = np.quantile(u, (np.arange(M) + 0.5) / M)

    prev_labels = None
    labels = None
    for _ in range(100):
        if prototypes.size == 1:
            labels = np.zeros(v.size, dtype=int)
        else:
            mids = (prototypes[:-1] + prototypes[1:]) / 2.0
            labels = np.searchsorted(mids, v, side="left")

        present = np.unique(labels)
        if present.size < prototypes.size:
            remap = np.full(prototypes.size, -1, dtype=int)
            remap[present] = np.arange(present.size)
            labels = remap[labels]
            prototypes = prototypes[present]

        converged = (
            prev_labels is not None
            and prev_labels.shape == labels.shape
            and np.array_equal(prev_labels, labels)
        )
        if converged:
            break
        prev_labels = labels

        prototypes = np.array(
            [v[labels == k].mean() for k in range(prototypes.size)]
        )

    labels = labels.astype(int)
    return labels, prototypes


def within_share(values: np.ndarray, labels: np.ndarray) -> float:
    """
    The share of the plain variance of `values` left inside the bins
    defined by `labels`:

        sum_k sum_{i in k} (v_i - mean_k)^2 / sum_i (v_i - mean)^2

    0.0 if there is one bin or the total variance is 0. Cost: O(M)
    Python-level iterations over the bin labels present, each doing a
    vectorized reduction over its own members.
    """
    v = np.asarray(values, dtype=float).ravel()
    labels = np.asarray(labels)

    total_var = float(np.sum((v - v.mean()) ** 2))
    unique_labels = np.unique(labels)
    if unique_labels.size <= 1 or total_var == 0.0:
        return 0.0

    within = 0.0
    for k in unique_labels:
        vk = v[labels == k]
        within += float(np.sum((vk - vk.mean()) ** 2))
    return within / total_var


def build_bins(values: np.ndarray, eps: float) -> BinSet:
    """
    Build the 𝓘-VQ for one estimand coordinate, from the raw
    initial-influence-estimate values psi0_c(x_i) at all N data
    points, at the initial count M_init = min(ceil(sqrt(2.7/eps)),
    n_distinct).

    No minimum bin size, no merging: empty bins are dropped during
    `kmeans_1d`'s iteration and the count actually used, M_used <=
    M_init, is what the returned `BinSet` reports. `refine.run_
    refinement` may grow the bin count further where the
    finite-differenced gains justify it -- that growth is not
    represented in this `BinSet`, which always describes the INITIAL
    bins only.

    U and d2T come back with shape (M_used, 0): this function knows
    nothing about q (the estimator has not been evaluated yet);
    `bin_differences` fills them in.

    Cost: O(N) vectorized (see `kmeans_1d`); no Python loop over the N
    data points.
    """
    v = np.asarray(values, dtype=float).ravel()
    N = v.size
    D = int(np.unique(v).size)

    m_ref = math.ceil(math.sqrt(2.7 / eps))
    M_init = int(min(m_ref, D))

    if M_init <= 1:
        labels = np.zeros(N, dtype=int)
        n = np.array([N], dtype=int)
    else:
        labels, _prototypes = kmeans_1d(v, M_init)
        n = np.bincount(labels)

    share = within_share(v, labels)
    L = int(n.size)

    return BinSet(
        labels=labels,
        n=n,
        U=np.zeros((L, 0)),
        d2T=np.zeros((L, 0)),
        centering_residual=np.zeros(0),
        M_init=M_init,
        M_used=L,
        within_share=share,
        failed=False,
    )


def bin_differences(X: np.ndarray, counter, theta_hat: np.ndarray, binset: BinSet, eta: float) -> BinSet:
    """
    Measure one estimand coordinate's 𝓘-VQ on the full data: for each
    bin k, difference the estimator along the bin's mass around the
    shared base value theta_hat (evaluated once by the caller and
    never re-evaluated here), using `core.differences.difference`
    (central where feasible, one-sided otherwise -- selected
    automatically, never forced).

    Single-bin shortcut: when `binset.M_used == 1`, perturbing the
    whole data's mass changes no weight, so U = 0 and d2T = 0 with no
    evaluations spent (T is never called, so this shortcut can never
    fail).

    Otherwise, for each of the M_used bins k, in order:
        evaluate(t) = counter(X, perturbed_weights(ones(N), labels == k, t))
        U_k, d2T_k, _ = difference(theta_hat, p[k], step, evaluate)
    If U_k or d2T_k carries any NaN -- one of `evaluate`'s two calls
    returned a failed fit -- this is a failed evaluation of an initial
    bin (ruled 18 September): there is nothing to salvage, so the loop
    stops at bin k (no further bins are measured, nothing retried) and
    the returned `BinSet` has `failed=True`, `U`/`d2T`/
    `centering_residual` all NaN; `refine.run_refinement` reads
    `failed` and returns a NaN `CoordinateResult` instead of calling
    `between_terms` on this data.

    Otherwise (`failed=False`), U is centered by subtracting the
    bin-mass-weighted residual centering_residual = sum_k p_k U_k,
    which is kept on the returned `BinSet` (see its docstring:
    `refine.run_refinement` reuses it to keep later, single-evaluation
    splits on the same centered scale); d2T is left uncentered.

    Cost: at most O(M_used) evaluations of `counter` (two per bin),
    never a Python loop over the N data points -- each evaluation
    itself processes all N rows inside `counter`/T, which is T's cost,
    not a loop of ours.

    Returns a new `BinSet` (same labels/n/M_init/M_used/within_share;
    U, d2T, centering_residual and failed filled in).
    """
    N = binset.labels.size
    theta_hat = np.asarray(theta_hat, dtype=float)
    q = theta_hat.size
    M_used = binset.M_used

    if M_used == 1:
        return replace(binset, U=np.zeros((1, q)), d2T=np.zeros((1, q)),
                        centering_residual=np.zeros(q), failed=False)

    step = central_step(eta)
    p = binset.p
    U = np.empty((M_used, q))
    d2T = np.empty((M_used, q))

    for k in range(M_used):
        mask = binset.labels == k

        def evaluate(t: float, _mask=mask) -> np.ndarray:
            omega = perturbed_weights(np.ones(N), _mask, t)
            return counter(X, omega)

        U_k, d2T_k, _one_sided = difference(theta_hat, float(p[k]), step, evaluate)
        if np.any(np.isnan(U_k)) or np.any(np.isnan(d2T_k)):
            return replace(
                binset, U=np.full((M_used, q), np.nan), d2T=np.full((M_used, q), np.nan),
                centering_residual=np.full(q, np.nan), failed=True,
            )
        U[k] = U_k
        d2T[k] = d2T_k

    centering_residual = p @ U
    U = U - centering_residual

    return replace(binset, U=U, d2T=d2T, centering_residual=centering_residual, failed=False)


def between_terms(binset: BinSet, coordinate: int):
    """
    The between-bin finite-differenced scale and curvature term for
    one estimand coordinate:

        V_btw = (1/N) * sum_k p_k * U[k, coordinate]^2
        B_hat = sum_k p_k * d2T[k, coordinate] / (2N)

    B_hat is the between-bin curvature term: an estimate of part of
    the second-order bias, not the full second-order bias. Off-
    coordinate columns of U and d2T are never mixed in: only column
    `coordinate` is used here, even though `binset` carries all q
    columns because every evaluation returns all coordinates.

    Callers must not call this on a `BinSet` with `failed=True`
    (`bin_differences`'s stage-2 evaluation failure, ruled 18
    September): `refine.run_refinement` checks `binset.failed` first
    and returns a NaN `CoordinateResult` without calling this function.

    Returns (V_btw, B_hat), both floats.
    """
    N = binset.labels.size
    p = binset.p
    U_c = binset.U[:, coordinate]
    d2T_c = binset.d2T[:, coordinate]

    V_btw = float(np.sum(p * U_c ** 2)) / N
    B_hat = float(np.sum(p * d2T_c) / (2.0 * N))

    return V_btw, B_hat
