"""
The refinement loop: per estimand coordinate, starting from the
𝓘-VQ's initial bins (`ivq.build_bins`, measured on the full data by
`ivq.bin_differences` with the natural central/one-sided rule),
adaptively split the bin with the largest expected gain, spending one
forward evaluation per split on the smaller child -- at the relative
step delta_f (`differences.forward_step`), i.e. the weight parameter
t_small = delta_f * p_small / (1 - p_small), so the child's mass rises
by the fraction delta_f of itself and U_small = [T(omega_small) -
T_hat] / t_small stays the derivative with respect to t -- and
deriving the larger child by mass balance, until the
largest expected gain drops below a tolerance `eps * V_btw / L` or a
cost guard (at most `1 + M_X_used` refinement evaluations) binds. The
between term V_btw only ever rises through this; the within term
V_win_hat is a predicted estimate from the FINAL bin set alone, not
accumulated incrementally.

Revision 7 (plan §33): every bin with more than one point proposes
BOTH a level split and an adjacency split, each with its own expected
gain in variance units -- not a single selected candidate. The level
split (`_try_level_split`) cuts on psi0_hat, the k-means-2 partition
of the bin's own initial influence estimates; the CADJ adjacency
split (`_try_adjacency_split`) instead separates the bin's points by
which side of the boundary each point's SECOND-nearest first-stage
prototype falls on -- a point with I_proto[bmu2[i], c] >
I_proto[bmu[i], c] (the mass-centered prototype influence of the
coordinate being refined) goes to one side, "the rest" to the other.
Both candidates are tried independently and unconditionally; a bin
can queue the level candidate, the adjacency candidate, both, or
neither (bin closed), depending only on whether each split is
well-defined (`_try_level_split`/`_try_adjacency_split` return None
on a degenerate split -- fewer than 2 distinct psi0 values, k-means
collapsing to one side, or one CADJ side empty). Neither candidate
falls back to the other: an adjacency split that degenerates does not
become a duplicate level candidate, and vice versa. The refinement
loop's queue then takes the single largest expected gain over EVERY
CANDIDATE OF EVERY OPEN BIN, not the largest per-bin gain.

This replaces a prior rule (`Var_k(psi0) < mean_k sigma^2` selected
one of the two splits per bin, sigma from `influence_model`'s
posterior uncertainty) that measurement showed does not work: on the
MVT tail probability estimand the adjacency split count sat at 16
regardless of whether N was 1000 or 2000, while the prototype count
grew 133 -> 188 and the level split count grew 18 -> 37, and
V_btw/oracle fell from 1.0000 to 0.9131. The level split cuts on
psi0_hat, which is nearly constant across a bin straddling a curved
decision boundary (so a level split there buys almost nothing), and
the adjacency split is exactly the one that resolves a straddling
bin -- but its old sigma-based expected gain could not compete for
the queue as sigma fell with growing N, so the split that was needed
kept losing the auction to splits that were not. Running both
candidates unconditionally on every bin removes that competition
failure: the queue now compares actual expected between-variance
gains, not a proxy that happens to shrink with N for the wrong
reason. `sigma_c` (`influence_model`'s per-point posterior sd)
therefore no longer enters any proposal; it is read only by the
final `bin_posterior_variance` call, for v_k in V_win_hat below.

Two failure cases, both ruled 18 September (an estimator never raises;
a failed fit returns NaN, plan §4). Nothing is retried in either case.

* Stage 2's initial-bin measurement (`ivq.bin_differences`) fails: the
  between-bin term and everything after it rests on that measurement,
  so there is nothing to salvage. `run_refinement` checks
  `bins0.failed` right after `bin_differences` and returns a
  `CoordinateResult` whose per-output quantities (`V_btw`,
  `V_win_hat`, `V_tot_hat`, `B_hat`, `a_bca`, `field`) are NaN, with
  the diagnostics that stage 2 genuinely established before the
  failure (`L`/`M_used` = the bin count actually built,
  `bins0.labels`) kept rather than discarded; `n_level_splits`,
  `n_adjacency_splits` and `n_refine_evals` are 0 because the
  refinement loop never starts.
* A refinement split's own evaluation (the forward-differenced
  `T_small`) fails: that split is cancelled -- the parent stays a bin
  and is CLOSED, so the loop does not try it again -- and the rest of
  the refinement proceeds normally; one failed split does not spoil
  the draw. The failure itself is counted by `core.counter.Counter`
  (whose `.failed` increments on that same call), not re-counted here.

THE SETTLED FORMULA (final; not open to reinterpretation):

    V_win_hat = rho^2 * (1/N) * sum_k p_k * gamma_k * [Var_k(psi0_hat) + v_k]

over bins holding more than one point. `v_k` is the within-bin
posterior variance from `influence_model.bin_posterior_variance`.
`gamma_k` is the realized-over-expected gain of the split that CREATED
bin k, clipped to [0, 1]; both children of a split take that split's
ratio; `gamma_k = 1` for a bin never split, and `gamma_k = 1` whenever
the split's expected gain is not strictly positive or either quantity
(realized, expected) is non-finite -- no usable information, so the
term is not down-weighted. `gamma_k` scales the WHOLE bracket: a low
gain ratio is measured evidence that the bin holds no variation the
model successfully predicts, and the model's own uncertainty `v_k` is
part of the prediction being discounted; bins never split keep
gamma_k = 1 and have small `v_k` anyway. V_win_hat is computed for
every multi-point final bin in one call to `bin_posterior_variance` so
the kernel work for v_k is done once.

Dropped from the reference implementation this is ported from: the
`FinalBinSet` dataclass (the final bin arrays are local bookkeeping
only -- nothing downstream reads them, so they are never assembled
into a returned object); the 'measured' vs 'mass-balance-derived' flag
and `n_derived_bins` it fed; `guard_bound`, `max_remaining_gain`,
`rho2`, `field_constant`, `M_init`, `within_share` on the RESULT
(surplus diagnostics not in the plan's product columns -- the
underlying centering_residual is still used operationally, via
`ivq.BinSet.centering_residual`, to keep a refinement split's
forward-differenced U on the same centered scale as the initial bins'
U; it is simply not reported); the per-stage wall-time dict (timing is
the caller's job, not this function's); forcing the initial-bin
measurement to the one-sided rule (the natural central/one-sided
selection of `differences.difference` is used instead -- the old
forcing existed only to match a separate first-stage-only comparison
report that this package does not have).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .differences import forward_step, perturbed_weights, step_parameter
from .influence_model import bin_posterior_variance
from .ivq import BinSet, between_terms, bin_differences, build_bins, kmeans_1d
from .outputs import acceleration

__all__ = ["CoordinateResult", "run_refinement"]

# Fixed candidate order for the refinement queue's tie-break (plan
# §33): when two candidates of the SAME bin have exactly equal
# expected gain, level goes first. Bins are already distinguished by
# `id` before this key is ever consulted.
_CANDIDATE_ORDER = {"level": 0, "adjacency": 1}


@dataclass
class CoordinateResult:
    """
    One estimand coordinate's complete refinement result.

    coordinate, name    the coordinate's index and display name.
    V_btw, V_win_hat, V_tot_hat, B_hat, a_bca
                        the finite-differenced between term, the
                        predicted within term, their sum, the
                        between-bin curvature term, and the
                        acceleration (`outputs.acceleration`) of the
                        influence field below.
    field               (N,) psi_hat_i = U_{k(i)}[c] + rho*(psi0_i -
                        ubar_{k(i)}), the influence field, for the
                        acceleration and for `outputs`/`intervals`
                        downstream.
    labels              (N,) int, the final 𝓘-VQ bin index per data
                        point (contiguous 0..L-1) -- the partition
                        refinement actually produced, as opposed to
                        `field`, which is the influence estimate on
                        that partition, not the assignment. `qij.
                        check`'s weighted-mean identity reads this to
                        form p_k and Var_k(psi) on the real bins.
    L                   final bin count.
    n_level_splits, n_adjacency_splits
                        counts of each split kind actually taken.
    rho                 sqrt(V_btw / ((1/N) sum_k p_k ubar_k^2)) at the
                        FINAL bin set (NaN if that denominator is 0 or
                        negative).
    gain_ratio          sum of realized split gains over sum of
                        expected split gains (NaN if no split was
                        taken).
    bin_mass, bin_influence, bin_d2T
                        (L,) float each, ordered like `labels` (index k
                        <-> labels == k) -- the final bin set's
                        constituents for a downstream second-order
                        interval, which reconstructs a quadratic
                        surrogate of the estimator in the bin masses
                        and needs no further estimator evaluations.
                        `bin_mass` is p_k = n_k / N. `bin_influence` is
                        this coordinate's own U_k (U_arr[:,
                        coordinate]) -- centered by
                        `centering_residual`, exactly as `field` is.
                        `bin_d2T` is the second difference each final
                        bin carries: an initial bin (one refinement
                        never split) keeps its own bins0.d2T[k,
                        coordinate]; a bin CREATED by refinement
                        inherits its PARENT's value unchanged --
                        curvature is inherited, not re-measured, since
                        refinement spends no evaluation that would
                        remeasure it. `bin_d2T` is uncentered, the same
                        convention as `ivq.BinSet.d2T`.
    M_X                 the first-stage prototype count actually used
                        (M_X_used, an input).
    M_used              the INITIAL 𝓘-VQ's bin count actually used
                        (`ivq.BinSet.M_used`, before refinement grows
                        it).
    n_refine_evals      forward evaluations spent by the refinement
                        loop, including a cancelled split's evaluation
                        (ruled 18 September: a failed split evaluation
                        is still an evaluation spent, it is simply not
                        applied -- the parent stays a bin, closed).
                        0 when stage 2's own initial-bin measurement
                        failed (`ivq.BinSet.failed`), since the
                        refinement loop never starts.
    failed              True only when THIS coordinate's own
                        initial-bin measurement failed (set by
                        `_failed_result`; False for a normal or
                        degenerate/constant-path coordinate). `qij.py`
                        reads this across all q coordinates: if any one
                        is True, the whole draw's initial-bin
                        measurement is compromised (plan §4), so every
                        coordinate's variance quantities (V_btw,
                        V_win_hat, V_tot_hat, B_hat, a_bca) are voided
                        to NaN there -- not just this one's.
    """

    coordinate: int
    name: str
    V_btw: float
    V_win_hat: float
    V_tot_hat: float
    B_hat: float
    a_bca: float
    field: np.ndarray
    labels: np.ndarray
    L: int
    n_level_splits: int
    n_adjacency_splits: int
    rho: float
    gain_ratio: float
    bin_mass: np.ndarray
    bin_influence: np.ndarray
    bin_d2T: np.ndarray
    M_X: int
    M_used: int
    n_refine_evals: int
    failed: bool = False


def _variance(values: np.ndarray) -> float:
    """Population variance (ddof=0); 0.0 for an empty array."""
    if values.size == 0:
        return 0.0
    return float(np.mean((values - values.mean()) ** 2))


def _split_gamma(delta_realized: float, g_expected: float) -> float:
    """gamma_k for the two children of a split: clip(delta_realized /
    g_expected, 0.0, 1.0) when g_expected is strictly positive and
    both quantities are finite; 1.0 otherwise (no usable information,
    so the within term is not down-weighted -- the settled ruling)."""
    if g_expected > 0.0 and math.isfinite(g_expected) and math.isfinite(delta_realized):
        return min(1.0, max(0.0, delta_realized / g_expected))
    return 1.0


def _try_level_split(idx: np.ndarray, psi_leaf: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Level split: `ivq.kmeans_1d(psi_leaf, 2)`; None if the bin has
    fewer than 2 distinct psi0 values, or k-means collapses to one
    non-empty side."""
    distinct = np.unique(psi_leaf)
    if distinct.size < 2:
        return None
    labels2, protos2 = kmeans_1d(psi_leaf, 2)
    if protos2.size < 2:
        return None
    idx_a = idx[labels2 == 0]
    idx_b = idx[labels2 == 1]
    if idx_a.size == 0 or idx_b.size == 0:
        return None
    return idx_a, idx_b


def _try_adjacency_split(
    idx: np.ndarray, I_proto_c: np.ndarray, bmu: np.ndarray, bmu2: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    The CADJ adjacency split: the bin's points with
    I_proto[bmu2[i], c] > I_proto[bmu[i], c] (b = BMU, b2 = second BMU,
    both already resolved to live prototypes; the mass-centered
    prototype influence of the coordinate being refined) against the
    rest; ties (equal, or b2 == b) go to "the rest". None if one side
    is empty -- the bin then simply has no adjacency candidate (plan
    §33, revision 7: it does NOT fall back to the level split; see
    `propose`).

    Harmless by construction if a prototype's own influence could not be
    evaluated (the companion 18 September ruling for stage 1): `I_proto_c`
    then stays NaN at that prototype's index, so a point whose bmu/bmu2
    lands on it has `diff` NaN here, which `diff > 0.0` silently resolves
    to False (NumPy's NaN comparison) -- that point falls to "the rest"
    with no ordering rationale. This only decides which SIDE of a split a
    point lands on; each child's influence is then obtained by an actual
    evaluation of `T`, and the variance decomposition is computed from
    those measurements, not from `diff`. So an arbitrary side for a few
    points can only make a split slightly worse at its job (and a
    degenerate split simply yields no adjacency candidate for that bin)
    -- it cannot make any reported number wrong.
    """
    diff = I_proto_c[bmu2[idx]] - I_proto_c[bmu[idx]]
    mask = diff > 0.0
    if not mask.any() or mask.all():
        return None
    return idx[mask], idx[~mask]


def _degenerate_result(coordinate: int, name: str, N: int, bins0: BinSet, M_X_used: int) -> CoordinateResult:
    """One bin, no evaluations, V_btw = V_win_hat = B_hat = 0, field 0
    (the coordinate's constant path, or an initial quantizer with
    M_used <= 1). This function returns before `bin_differences` is
    ever called, so `bins0.U`/`bins0.d2T` are still the (M_used, 0)
    placeholders `build_bins` leaves them at -- there is no per-
    coordinate influence or curvature to read. `bin_influence` and
    `bin_d2T` therefore report 0.0 for the single synthetic bin, the
    same "no evaluations happened" convention as `field`/`V_btw`/
    `V_win_hat`/`B_hat`; `bin_mass` is exactly 1.0 (the one bin holds
    every point, by construction of `labels` below)."""
    field = np.zeros(N, dtype=float)
    labels = np.zeros(N, dtype=int)
    bin_mass = np.array([1.0], dtype=float)
    bin_influence = np.zeros(1, dtype=float)
    bin_d2T = np.zeros(1, dtype=float)
    return CoordinateResult(
        coordinate=coordinate, name=name,
        V_btw=0.0, V_win_hat=0.0, V_tot_hat=0.0, B_hat=0.0,
        a_bca=acceleration(field), field=field, labels=labels,
        L=1, n_level_splits=0, n_adjacency_splits=0,
        rho=float('nan'), gain_ratio=float('nan'),
        bin_mass=bin_mass, bin_influence=bin_influence, bin_d2T=bin_d2T,
        M_X=M_X_used, M_used=bins0.M_used, n_refine_evals=0,
    )


def _failed_result(coordinate: int, name: str, N: int, bins0: BinSet, M_X_used: int) -> CoordinateResult:
    """Case 1 (ruled 18 September): `bins0.failed` -- stage 2's
    initial-bin measurement hit a failed evaluation. There is nothing
    to salvage, so every per-output quantity is NaN; the diagnostics
    carry what stage 2 genuinely established before the failure: the
    bin count actually built (`bins0.M_used`, reported as both `L` and
    `M_used`) and its partition (`bins0.labels`, the bins that were
    built even though their measurement failed). No refinement is
    attempted, so its counts are 0 and `n_refine_evals` is 0; the
    evaluations `counter` already spent are `Counter`'s own count, not
    reported on this result. Nothing is retried. `failed=True` marks
    this coordinate for `qij.py`, which voids every coordinate's
    variance quantities on the draw when any one of them failed here
    (plan §4: a failed initial-bin evaluation NaNs the whole draw).

    `bin_mass`/`bin_influence`/`bin_d2T` split the same way as the
    rest of this result: `bin_mass` is a partition fact that stage 2
    genuinely established before the failure (bins0.n / N, kept, like
    `labels`/`L`/`M_used`), while `bin_influence`/`bin_d2T` are outputs
    of the differencing that FAILED, so they are read straight off
    `bins0.U`/`bins0.d2T` at coordinate `coordinate` -- already
    NaN-filled, (M_used, q) arrays, by `ivq.bin_differences` on this
    same failure."""
    field = np.full(N, np.nan, dtype=float)
    bin_mass = bins0.n.astype(float) / N
    bin_influence = np.asarray(bins0.U[:, coordinate], dtype=float)
    bin_d2T = np.asarray(bins0.d2T[:, coordinate], dtype=float)
    return CoordinateResult(
        coordinate=coordinate, name=name,
        V_btw=float('nan'), V_win_hat=float('nan'), V_tot_hat=float('nan'),
        B_hat=float('nan'), a_bca=acceleration(field), field=field, labels=bins0.labels,
        L=bins0.M_used, n_level_splits=0, n_adjacency_splits=0,
        rho=float('nan'), gain_ratio=float('nan'),
        bin_mass=bin_mass, bin_influence=bin_influence, bin_d2T=bin_d2T,
        M_X=M_X_used, M_used=bins0.M_used, n_refine_evals=0,
        failed=True,
    )


def run_refinement(
    X: np.ndarray,
    counter,
    theta_hat: np.ndarray,
    coordinate: int,
    name: str,
    psi0_c: np.ndarray,
    m_c: float,
    sigma_c: np.ndarray,
    I_proto_c: np.ndarray,
    bmu: np.ndarray,
    bmu2: np.ndarray,
    eta: float,
    eps: float,
    M_X_used: int,
    constant_path: bool,
    Z: np.ndarray,
    model,
) -> CoordinateResult:
    """
    Refinement for one estimand coordinate. `counter` is the run's
    single `Counter`; every evaluation, initial-bin and refinement
    alike, goes through it. `theta_hat` is the shared full-data base
    value (all q coordinates); `psi0_c`/`sigma_c` are this
    coordinate's own initial-influence-estimate and per-point
    posterior-sd arrays (N,); `m_c` is the data-mean offset.
    `I_proto_c` (M_X_used,) is this coordinate's own mass-centered
    prototype influence at the first-stage prototypes; `bmu`/`bmu2`
    (N,) are the per-point first- and (resolved) second-BMU indices,
    shared across every coordinate -- the CADJ adjacency split reads
    `I_proto_c[bmu2[i]] - I_proto_c[bmu[i]]`. `M_X_used` is the
    first-stage prototype count actually used (the cost guard). `Z`
    (N, d_z) is the same first-stage-quantizer coordinates `model` was
    fitted on, unwhitened; `model` is the run's `InfluenceModel` --
    both are passed only to compute the final V_win_hat's within-bin
    posterior variance v_k via `influence_model.bin_posterior_variance`,
    in one call over every multi-point final bin.

    Cost: the refinement loop runs O(L) iterations, L the final bin
    count (bounded by `1 + M_X_used`, the cost guard); each iteration
    does O(1) bin bookkeeping (index arrays already materialized by
    vectorized numpy operations) plus exactly one evaluation of
    `counter`. `compute_rho2`/`propose` are O(current bin count) per
    call, so the whole loop is O(L^2) in bin bookkeeping, never
    O(N*M^2) and never a Python loop over the N data points -- the
    only per-point work is the vectorized boolean/integer indexing
    that produces each split's two index arrays.
    """
    N = len(X)
    q = theta_hat.shape[0]

    bins0 = build_bins(psi0_c, eps)

    if constant_path or bins0.M_used <= 1:
        return _degenerate_result(coordinate, name, N, bins0, M_X_used)

    bins0 = bin_differences(X, counter, theta_hat, bins0, eta)
    if bins0.failed:
        return _failed_result(coordinate, name, N, bins0, M_X_used)

    V_btw0, B_hat0 = between_terms(bins0, coordinate)

    psi_centered = psi0_c - m_c
    delta_f = forward_step(eta)
    evals_cap = 1 + M_X_used

    # ---- initial bins, as refinement leaves --------------------------
    leaves: Dict[int, dict] = {}
    next_id = 0
    for k in range(bins0.M_used):
        idx = np.where(bins0.labels == k)[0]
        leaves[next_id] = dict(
            id=next_id, indices=idx, n=int(idx.size),
            U=np.asarray(bins0.U[k], dtype=float).copy(),
            # Curvature is measured once, at the initial bins, and
            # never re-measured by refinement (plan §33 change 2): an
            # initial leaf seeds its own d2T here; a bin created by a
            # later split inherits its parent's value unchanged (see
            # the split branch below).
            d2T=float(bins0.d2T[k, coordinate]),
            open=True, candidates=[], gamma=1.0,
        )
        next_id += 1

    V_btw = float(V_btw0)
    n_refine_evals = 0
    n_level_splits = 0
    n_adjacency_splits = 0
    sum_measured_delta = 0.0
    sum_expected_g = 0.0

    def leaf_ubar(leaf: dict) -> float:
        return float(psi_centered[leaf['indices']].mean())

    def sum_p_ubar2() -> float:
        total = 0.0
        for leaf in leaves.values():
            total += (leaf['n'] / N) * leaf_ubar(leaf) ** 2
        return total

    def compute_rho2() -> float:
        denom = sum_p_ubar2() / N
        return (V_btw / denom) if denom != 0.0 else float('nan')

    def propose(leaf: dict, rho2_current: float) -> None:
        """Compute the bin's candidate splits and their expected gains
        (plan §33, revision 7): BOTH a level and an adjacency
        candidate, independently -- not a selection between them.
        `rho2_current` is the rho^2 in force when the bin was created
        (frozen into each candidate's gain at proposal time, exactly
        as the old single-candidate gain was).

        LEVEL candidate (`_try_level_split`, unchanged): the
        between-children variance of psi0_hat over the bin,
        mass-weighted --

            g_level = rho2 * (p_a*ubar_a^2 + p_b*ubar_b^2
                               - p_k*ubar_k^2) / N

        ADJACENCY candidate (`_try_adjacency_split`, unchanged split
        rule; new gain): a mean OVER THE BIN'S POINTS of the SQUARE of
        half the absolute prototype-influence difference between each
        point's second- and first-nearest prototype -- a mean of
        squares, paralleling the retired mean_sigma2_k it replaces,
        NOT the square of a mean --

            g_adj = rho2 * p_k * mean_{i in k}[(0.5 * abs(
                        I_proto_c[bmu2[i]] - I_proto_c[bmu[i]]))^2] / N

        computed with a NaN-ignoring mean so a prototype whose own
        influence could not be evaluated (diff NaN at that point) does
        not poison the bin's gain. In practice this branch never sees
        an all-NaN bin: `_try_adjacency_split` already needs at least
        one point with diff > 0.0 (hence non-NaN, since NaN > 0.0 is
        False) to return a split at all, so an all-NaN bin fails there
        first and never reaches the gain calculation -- it is closed
        with no adjacency candidate, which is the same outcome the
        NaN-ignoring mean would give it directly. The guard below is
        kept anyway, so the rule ("all-NaN bin -> gain 0.0, no
        candidate") is stated where it is used and the calculation
        cannot raise `np.nanmean`'s empty-slice warning if that
        invariant is ever broken.

        Neither candidate falls back to the other on degeneracy --
        each is simply omitted when its `_try_*_split` returns None.
        A bin with neither candidate is closed (`leaf['open'] =
        False`)."""
        idx = leaf['indices']
        n_k = leaf['n']
        candidates = []

        if n_k > 1:
            psi_leaf = psi0_c[idx]
            p_k = n_k / N
            ubar_k = leaf_ubar(leaf)
            rho2_local = rho2_current if np.isfinite(rho2_current) else 0.0

            level_split = _try_level_split(idx, psi_leaf)
            if level_split is not None:
                idx_a, idx_b = level_split
                ubar_a = float(psi_centered[idx_a].mean())
                ubar_b = float(psi_centered[idx_b].mean())
                p_a = idx_a.size / N
                p_b = idx_b.size / N
                g_level = (rho2_local * (p_a * ubar_a ** 2 + p_b * ubar_b ** 2 - p_k * ubar_k ** 2)) / N
                candidates.append(('level', g_level, idx_a, idx_b))

            adj_split = _try_adjacency_split(idx, I_proto_c, bmu, bmu2)
            if adj_split is not None:
                diff = I_proto_c[bmu2[idx]] - I_proto_c[bmu[idx]]
                sq = (0.5 * np.abs(diff)) ** 2
                if np.all(np.isnan(sq)):
                    # Unreachable given adj_split is not None (see
                    # docstring); defensive only.
                    pass
                else:
                    mean_sq = float(np.nanmean(sq))
                    g_adj = (rho2_local * p_k * mean_sq) / N
                    idx_a2, idx_b2 = adj_split
                    candidates.append(('adjacency', g_adj, idx_a2, idx_b2))

        leaf['candidates'] = candidates
        leaf['open'] = len(candidates) > 0

    rho2 = compute_rho2()
    for leaf in leaves.values():
        propose(leaf, rho2)

    # ---- refinement loop ----------------------------------------------
    while True:
        open_leaves = [l for l in leaves.values() if l['open']]
        if not open_leaves:
            break
        tau = eps * V_btw / len(leaves)

        # The queue ranges over every (bin, candidate) pair of every
        # open bin (plan §33: two candidates per bin, not one), and
        # takes the single largest expected gain among all of them --
        # not the largest per-bin gain. Tie-break, fully deterministic:
        # descending gain, then ascending bin id (the old per-bin
        # tie-break), then a fixed candidate order -- level before
        # adjacency (`_CANDIDATE_ORDER`) -- for the rare case of an
        # exact tie between one bin's own two candidates.
        pairs = [(leaf, cand) for leaf in open_leaves for cand in leaf['candidates']]
        best_leaf, best_candidate = min(
            pairs,
            key=lambda pair: (-pair[1][1], pair[0]['id'], _CANDIDATE_ORDER[pair[1][0]]),
        )
        kind, g_expected, idx_a, idx_b = best_candidate
        if not (g_expected >= tau):
            break
        if n_refine_evals >= evals_cap:
            break

        tau_at_selection = tau
        if idx_a.size <= idx_b.size:
            idx_small, idx_large = idx_a, idx_b
        else:
            idx_small, idx_large = idx_b, idx_a

        p_small = idx_small.size / N
        p_large = idx_large.size / N
        t_small = step_parameter(delta_f, p_small)

        mask_small = np.zeros(N, dtype=bool)
        mask_small[idx_small] = True
        omega = perturbed_weights(np.ones(N), mask_small, t_small)
        T_small = np.asarray(counter(X, omega), dtype=float)
        n_refine_evals += 1

        if np.any(np.isnan(T_small)):
            # Case 2 (ruled 18 September): this split's own evaluation
            # failed. Cancel it -- the parent stays a bin and is
            # CLOSED, so the loop does not try it again. With two
            # candidates per bin (plan §33) this still closes the
            # WHOLE BIN, not just the candidate that failed: there is
            # only one evaluation (on the winning candidate's small
            # side), so the other candidate is never separately
            # tested and there is nothing left to propose from this
            # bin. `counter` already counted the failure
            # (`Counter.failed`). Nothing is retried, and the rest of
            # the refinement proceeds normally -- one failed split
            # does not spoil the draw.
            best_leaf['open'] = False
            continue

        U_small = (T_small - theta_hat) / t_small - bins0.centering_residual
        p_parent = best_leaf['n'] / N
        U_parent = best_leaf['U']
        U_large = (p_parent * U_parent - p_small * U_small) / p_large

        Delta = (
            p_small * U_small[coordinate] ** 2
            + p_large * U_large[coordinate] ** 2
            - p_parent * U_parent[coordinate] ** 2
        ) / N

        V_btw += Delta
        sum_measured_delta += Delta
        sum_expected_g += g_expected
        if kind == 'level':
            n_level_splits += 1
        else:
            n_adjacency_splits += 1

        gamma_children = _split_gamma(Delta, g_expected)

        # Curvature is inherited from the parent, not re-measured
        # (plan §33 change 2): both children of this split carry the
        # SAME d2T the parent bin carried.
        parent_d2T = best_leaf['d2T']

        del leaves[best_leaf['id']]

        leaf_small = dict(
            id=next_id, indices=idx_small, n=int(idx_small.size),
            U=U_small, d2T=parent_d2T, open=True, candidates=[], gamma=gamma_children,
        )
        leaf_large = dict(
            id=next_id + 1, indices=idx_large, n=int(idx_large.size),
            U=U_large, d2T=parent_d2T, open=True, candidates=[], gamma=gamma_children,
        )
        next_id += 2
        leaves[leaf_small['id']] = leaf_small
        leaves[leaf_large['id']] = leaf_large

        if Delta < tau_at_selection:
            leaf_small['open'] = False
            leaf_large['open'] = False
        else:
            rho2 = compute_rho2()
            propose(leaf_small, rho2)
            propose(leaf_large, rho2)

    # ---- final diagnostics and outputs --------------------------------
    final_rho2 = compute_rho2()
    rho = math.sqrt(final_rho2) if (np.isfinite(final_rho2) and final_rho2 >= 0.0) else float('nan')

    multi_ids = [lid for lid, leaf in leaves.items() if leaf['n'] > 1]
    v_by_id: Dict[int, float] = {}
    if multi_ids:
        groups = [leaves[lid]['indices'] for lid in multi_ids]
        v_vals = bin_posterior_variance(model, Z, coordinate, groups, sigma_c)
        v_by_id = dict(zip(multi_ids, v_vals))

    V_win_hat = 0.0
    for leaf in leaves.values():
        if leaf['n'] > 1:
            idx = leaf['indices']
            psi_leaf = psi0_c[idx]
            var_k = _variance(psi_leaf)
            v_k = v_by_id[leaf['id']]
            V_win_hat += (leaf['n'] / N) * leaf['gamma'] * (var_k + v_k)
    V_win_hat = (V_win_hat * final_rho2) / N if np.isfinite(final_rho2) else float('nan')
    V_tot_hat = V_btw + V_win_hat

    gain_ratio = (sum_measured_delta / sum_expected_g) if sum_expected_g != 0.0 else float('nan')

    ordered_ids = sorted(leaves.keys())
    L = len(ordered_ids)
    labels_final = np.empty(N, dtype=int)
    U_arr = np.empty((L, q), dtype=float)
    ubar_arr = np.empty(L, dtype=float)
    for new_id, old_id in enumerate(ordered_ids):
        leaf = leaves[old_id]
        labels_final[leaf['indices']] = new_id
        U_arr[new_id] = leaf['U']
        ubar_arr[new_id] = leaf_ubar(leaf)

    # The interval's constituents (plan §33 change 2): the final bin
    # set's mass, this coordinate's own influence, and the second
    # difference each bin carries, ordered like `labels_final` so a
    # downstream second-order interval can reconstruct a quadratic
    # surrogate of the estimator in the bin masses with no further
    # estimator evaluations.
    bin_mass = np.array([leaves[old_id]['n'] / N for old_id in ordered_ids], dtype=float)
    bin_d2T = np.array([leaves[old_id]['d2T'] for old_id in ordered_ids], dtype=float)
    bin_influence = U_arr[:, coordinate].copy()

    if np.isfinite(rho):
        field = U_arr[labels_final, coordinate] + rho * (psi_centered - ubar_arr[labels_final])
    else:
        field = np.full(N, np.nan, dtype=float)

    return CoordinateResult(
        coordinate=coordinate, name=name,
        V_btw=float(V_btw), V_win_hat=float(V_win_hat), V_tot_hat=float(V_tot_hat),
        B_hat=float(B_hat0), a_bca=acceleration(field), field=field, labels=labels_final,
        L=L, n_level_splits=n_level_splits, n_adjacency_splits=n_adjacency_splits,
        rho=float(rho), gain_ratio=float(gain_ratio),
        bin_mass=bin_mass, bin_influence=bin_influence, bin_d2T=bin_d2T,
        M_X=M_X_used, M_used=bins0.M_used, n_refine_evals=n_refine_evals,
    )
