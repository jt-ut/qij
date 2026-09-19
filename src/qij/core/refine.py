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

Plan §36.2(1), 19 September (revision 8, ruling 1): ONE proposal per
bin with more than one point -- revision 7 (plan §33) gave every bin
TWO candidate splits, level and adjacency, and queued over all
candidates of all open bins; that is withdrawn (§36.1: on the MVT
tail probability it bought nothing the sigma-priced single-candidate
rule at revision 6 did not already buy, and it cost a second column
of bookkeeping this file no longer carries). This reverts to revision
6's single-candidate shape, with ONE substitution: `v_k`, the
within-bin posterior variance from `influence_model.
bin_posterior_variance`, wherever revision 6 read mean_k sigma_i^2
(the mean, over the bin, of `sigma_c`'s per-point posterior variance)
directly from `sigma_c`.

    Var_k(psi0_hat) > v_k:  LEVEL split (`_try_level_split`,
                            two-means on psi0_hat), gain the
                            between-children variance of psi0_hat
                            over the bin, mass-weighted --
                                g = rho^2 * (p_a*ubar_a^2 +
                                    p_b*ubar_b^2 - p_k*ubar_k^2) / N
    otherwise:              ADJACENCY split (`_try_adjacency_split`,
                            the bin's points whose SECOND-nearest
                            first-stage prototype carries the higher
                            prototype influence against the rest),
                            falling back to the level split if one
                            side is empty (and closing the bin, g =
                            0, if neither is well-defined) --
                                g = rho^2 * p_k * v_k / N

v_k replaces mean_k sigma_i^2 in BOTH roles it held in revision 6 --
the selection test above AND the adjacency gain -- because v_k is the
within-bin posterior variance the model actually PREDICTS (the
bin-common part of the posterior already removed; see `THE SETTLED
FORMULA` below and `bin_posterior_variance`'s own docstring), hence
the gain a split can actually recover, where mean_k sigma_i^2
includes the bin-common uncertainty no split recovers. Under the
declared-noise-floor ruling (plan §36.2(2), a parallel change to
`influence_model.py`) sigma^2 grows roughly 300x on the IMF, which
would put a sigma-priced rule at the stopping threshold on every IMF
bin; v_k does not inherit that inflation, since the bin-common part
it strips out is exactly where that inflation lives.

The CADJ graph decides HOW an adjacency split divides a bin: points
with I_proto[bmu2[i], c] > I_proto[bmu[i], c] (the mass-centered
prototype influence of the coordinate being refined) against the
rest, ties to "the rest".

THE ONE HARD PART this substitution creates: `v_k` used to be needed
exactly ONCE per coordinate, at the very end, for the final multi-
point bins only (one `bin_posterior_variance` call over the whole
final bin set). Now it is needed at PROPOSAL time -- for every bin
when it is first proposed (the initial bins, and both children of
every accepted split) -- because the selection test above reads it
before a split is even chosen. A call to `bin_posterior_variance` per
BIN (rather than per proposal ROUND) would multiply its O(n_k *
M_X_used^2)-ish solve by the number of proposals instead of the
number of bins and be far slower than the refinement loop it prices.
This file batches it instead (`_batch_v`, near the top of
`run_refinement`): ONE call over ALL of the initial bins when they
are first proposed, and ONE call over BOTH children whenever a split
is accepted (covering both their own split decision and, unchanged,
their eventual contribution to V_win_hat below, should they end up as
final bins) -- never a call per single bin. A leaf's v_k depends only
on its own fixed point indices, so it is computed exactly once, when
the leaf is created, and simply read again wherever it is needed
later; nothing is ever recomputed. The number of `bin_posterior_
variance` calls a draw spends on one coordinate is therefore 1 (the
initial bins) plus the number of splits the queue actually accepts --
bounded the same way `n_refine_evals` already is, by `evals_cap = 1 +
M_X_used` -- not by the number of bins proposed in a round, which a
naive per-bin call would have been.

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

A third case, ruled 19 September (plan §36.2(4), revision 8): stage 1
itself collapses for this output -- the coordinate's constant path
(`influence_model`'s fallback when the first-stage prototype
influences have no usable spread) or an initial quantizer with
`M_used <= 1` (psi0_c itself has at most one distinct value; `build_
bins`). Before this ruling `_degenerate_result` reported V_btw =
V_win_hat = B_hat = 0 and did not mark the coordinate failed -- a
ZERO-VARIANCE result asserting a fact (this coordinate has no
influence anywhere) that stage 1's collapse never established; it was
only ever not measured. §36.1 found 13 IMF draws and 3 MVT nu draws
ending this way, silently excluded from coverage without appearing in
any failure column. Ruling 4: a collapsed stage 1 is a FAILED draw,
never a zero-variance one -- `_degenerate_result` now NaNs the same
variance quantities `_failed_result` does and sets `failed=True`, so
`qij.py`'s existing whole-draw NaN propagation (plan §4) treats it
exactly like a failed initial-bin evaluation. The genuine diagnostics
(`L`, `M_used`, `labels`, the split counts, `n_refine_evals`) are
untouched -- they are facts about what stage 2 would have started
from, not about a measurement that happened and returned zero.

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
gamma_k = 1 and have small `v_k` anyway. V_win_hat's final gather
below does NOT issue a fresh `bin_posterior_variance` call over the
final bin set -- every final bin already has its `v_k` cached from
the proposal round that created it (see "THE ONE HARD PART" above),
and that value is exactly the group's v_k regardless of when it was
computed, so the gather simply reads `leaf['v']` back.

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
from typing import Dict, List, Optional, Tuple

import numpy as np

from .differences import forward_step, perturbed_weights, step_parameter
from .influence_model import bin_posterior_variance
from .ivq import BinSet, between_terms, bin_differences, build_bins, kmeans_1d
from .outputs import acceleration

__all__ = ["CoordinateResult", "run_refinement"]


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
                        constituents for the between-bin and bias
                        terms' inputs (plan §36.2(6): the second-order
                        interval built from them is removed from the
                        package, but these columns stay -- they cost
                        no evaluations). `bin_mass` is p_k = n_k / N.
                        `bin_influence` is this coordinate's own U_k
                        (U_arr[:, coordinate]) -- centered by
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
    failed              True when THIS coordinate's own initial-bin
                        measurement failed (`_failed_result`) OR stage
                        1 collapsed for this output -- the
                        coordinate's constant path, or an initial
                        quantizer with M_used <= 1 (`_degenerate_
                        result`; plan §36.2(4), 19 September: a
                        collapsed stage 1 is a FAILED draw, never a
                        zero-variance result). `qij.py` reads this
                        across all q coordinates: if any one is True,
                        the whole draw's initial-bin measurement is
                        compromised (plan §4), so every coordinate's
                        variance quantities (V_btw, V_win_hat,
                        V_tot_hat, B_hat, a_bca) are voided to NaN
                        there -- not just this one's.
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
    is empty (the caller falls back to the level split).

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
    degenerate split already falls back to the level split) -- it cannot
    make any reported number wrong.
    """
    diff = I_proto_c[bmu2[idx]] - I_proto_c[bmu[idx]]
    mask = diff > 0.0
    if not mask.any() or mask.all():
        return None
    return idx[mask], idx[~mask]


def _degenerate_result(coordinate: int, name: str, N: int, bins0: BinSet, M_X_used: int) -> CoordinateResult:
    """One synthetic bin holding every point -- stage 1 collapsed for
    this output: the coordinate's constant path (`influence_model`'s
    fallback when the first-stage prototype influences have no usable
    spread) or an initial quantizer collapsing to M_used <= 1 (psi0_c
    itself has at most one distinct value; `build_bins`). This
    function returns before `bin_differences` is ever called, so
    `bins0.U`/`bins0.d2T` are still the (M_used, 0) placeholders
    `build_bins` leaves them at -- there is no per-coordinate
    influence or curvature to read, measured or otherwise.

    Plan §36.2(4), 19 September (ruling 4): a collapsed stage 1 is a
    FAILED draw, never a zero-variance result -- there was no split
    to measure and nothing for the between-bin term to be computed
    FROM, so reporting V_btw = 0 would assert a fact (this coordinate
    has no influence anywhere) that stage 1's collapse never
    established; it was only ever not measured. The variance
    quantities (`V_btw`, `V_win_hat`, `V_tot_hat`, `B_hat`, `a_bca`)
    are therefore NaN and `failed=True`, exactly like `_failed_result`
    below, so `qij.py`'s existing whole-draw NaN propagation (plan §4)
    treats this coordinate the same way it treats a failed initial-bin
    evaluation -- neither is silently excluded from coverage without
    appearing in a failure column.

    The genuine diagnostics stand, untouched: `L`/`M_used` = 1 (one
    bin, by construction of `labels` below), `labels` all-zero (every
    point in that one bin), `n_level_splits`/`n_adjacency_splits`/
    `n_refine_evals` = 0 (no refinement is attempted -- there is
    nothing to split). `bin_mass` is exactly 1.0, the same partition
    fact `labels` states -- it genuinely happened. `bin_influence` and
    `bin_d2T` stay 0.0 for the one synthetic bin: there is no
    per-coordinate influence or curvature to read even in principle
    here (see above), so 0.0 remains the "no evaluations happened"
    convention this file already uses elsewhere for that distinct
    situation -- not the same thing as the zero-VARIANCE result this
    ruling forbids, which is `V_btw`/`V_win_hat`/`B_hat` reporting a
    measured fact that was never measured. `field` is NaN (not 0), the
    same convention `_failed_result` uses, since it feeds `a_bca` and
    any downstream per-point reporting that this ruling also voids."""
    field = np.full(N, np.nan, dtype=float)
    labels = np.zeros(N, dtype=int)
    bin_mass = np.array([1.0], dtype=float)
    bin_influence = np.zeros(1, dtype=float)
    bin_d2T = np.zeros(1, dtype=float)
    return CoordinateResult(
        coordinate=coordinate, name=name,
        V_btw=float('nan'), V_win_hat=float('nan'), V_tot_hat=float('nan'), B_hat=float('nan'),
        a_bca=acceleration(field), field=field, labels=labels,
        L=1, n_level_splits=0, n_adjacency_splits=0,
        rho=float('nan'), gain_ratio=float('nan'),
        bin_mass=bin_mass, bin_influence=bin_influence, bin_d2T=bin_d2T,
        M_X=M_X_used, M_used=bins0.M_used, n_refine_evals=0,
        failed=True,
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
    both are passed to compute v_k, the within-bin posterior variance,
    via `influence_model.bin_posterior_variance` -- now needed both at
    PROPOSAL time (plan §36.2(1); see `_batch_v` below) and, reusing
    the same cached values, for the final V_win_hat's within-bin term.

    Cost: the refinement loop runs O(L) iterations, L the final bin
    count (bounded by `1 + M_X_used`, the cost guard); each iteration
    does O(1) bin bookkeeping (index arrays already materialized by
    vectorized numpy operations), exactly one evaluation of `counter`,
    and -- only when the split is accepted -- one `_batch_v` call
    pricing v_k for the split's two new children in a single
    `bin_posterior_variance` call (never one call per bin). So the
    whole loop spends at most `1 + (1 + M_X_used)` `bin_posterior_
    variance` calls in total (the initial-bins call plus one per
    accepted split), never one per bin PROPOSED. `compute_rho2`/
    `propose` are O(current bin count) per call, so the bin-bookkeeping
    part of the loop is O(L^2), never O(N*M^2) and never a Python loop
    over the N data points -- the only per-point work is the
    vectorized boolean/integer indexing that produces each split's two
    index arrays.
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
            # never re-measured by refinement: an initial leaf seeds
            # its own d2T here; a bin created by a later split
            # inherits its parent's value unchanged (see the split
            # branch below).
            d2T=float(bins0.d2T[k, coordinate]),
            open=True, split=None, g=0.0, gamma=1.0,
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

    def batch_v(leaf_list: List[dict]) -> None:
        """Set `leaf['v']` -- the within-bin posterior variance v_k --
        on every leaf in `leaf_list` that holds more than one point,
        in ONE call to `bin_posterior_variance` over all of them (plan
        §36.2(1)'s hard part; see the module docstring's "THE ONE HARD
        PART"). Leaves with n_k <= 1 are skipped: `propose` closes
        them before ever reading `leaf['v']`, and `bin_posterior_
        variance` would only return 0.0 for them anyway, so there is
        no reason to pay for the call. A leaf's v_k is a function only
        of its own (fixed, never-mutated) point indices, so it is
        computed here exactly once per leaf, the moment the leaf is
        created -- both `propose`'s split-selection test and the final
        V_win_hat gather below read the SAME cached value, never a
        recomputation."""
        qualifying = [leaf for leaf in leaf_list if leaf['n'] > 1]
        if not qualifying:
            return
        groups = [leaf['indices'] for leaf in qualifying]
        v_vals = bin_posterior_variance(model, Z, coordinate, groups, sigma_c)
        for leaf, v in zip(qualifying, v_vals):
            leaf['v'] = float(v)

    def propose(leaf: dict, rho2_current: float) -> None:
        """Compute the bin's proposed split and expected gain g (plan
        §36.2(1), revision 8: reverts to revision 6's single-candidate
        selection between a level and an adjacency split, substituting
        `leaf['v']` -- v_k, already computed by `batch_v` before this
        is ever called -- everywhere revision 6 read mean_k sigma_i^2
        from `sigma_c` directly; see the module docstring for why.
        `rho2_current` is the rho^2 in force when the bin was created
        (frozen into the gain at proposal time)."""
        idx = leaf['indices']
        n_k = leaf['n']
        if n_k <= 1:
            leaf['open'] = False
            leaf['split'] = None
            leaf['g'] = 0.0
            return

        psi_leaf = psi0_c[idx]
        var_k = _variance(psi_leaf)
        v_k = leaf['v']
        p_k = n_k / N
        ubar_k = leaf_ubar(leaf)
        rho2_local = rho2_current if np.isfinite(rho2_current) else 0.0

        if var_k > v_k:
            split = _try_level_split(idx, psi_leaf)
            if split is None:
                leaf['open'] = False
                leaf['split'] = None
                leaf['g'] = 0.0
                return
            idx_a, idx_b = split
            ubar_a = float(psi_centered[idx_a].mean())
            ubar_b = float(psi_centered[idx_b].mean())
            p_a = idx_a.size / N
            p_b = idx_b.size / N
            g = (rho2_local * (p_a * ubar_a ** 2 + p_b * ubar_b ** 2 - p_k * ubar_k ** 2)) / N
            leaf['split'] = ('level', idx_a, idx_b)
            leaf['g'] = g
            leaf['open'] = True
        else:
            split = _try_adjacency_split(idx, I_proto_c, bmu, bmu2)
            kind = 'adjacency'
            if split is None:
                split = _try_level_split(idx, psi_leaf)
                kind = 'level'
            if split is None:
                leaf['open'] = False
                leaf['split'] = None
                leaf['g'] = 0.0
                return
            idx_a, idx_b = split
            g = (rho2_local * p_k * v_k) / N
            leaf['split'] = (kind, idx_a, idx_b)
            leaf['g'] = g
            leaf['open'] = True

    rho2 = compute_rho2()
    batch_v(list(leaves.values()))
    for leaf in leaves.values():
        propose(leaf, rho2)

    # ---- refinement loop ----------------------------------------------
    while True:
        open_leaves = [l for l in leaves.values() if l['open']]
        if not open_leaves:
            break
        tau = eps * V_btw / len(leaves)
        best = min(open_leaves, key=lambda l: (-l['g'], l['id']))
        if not (best['g'] >= tau):
            break
        if n_refine_evals >= evals_cap:
            break

        tau_at_selection = tau
        kind, idx_a, idx_b = best['split']
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
            # CLOSED, so the loop does not try it again; `counter`
            # already counted the failure (`Counter.failed`). Nothing
            # is retried, and the rest of the refinement proceeds
            # normally -- one failed split does not spoil the draw.
            # The parent's own v_k (cached when it was proposed) is
            # untouched and still correct if it ends up a final bin.
            best['open'] = False
            continue

        U_small = (T_small - theta_hat) / t_small - bins0.centering_residual
        p_parent = best['n'] / N
        U_parent = best['U']
        U_large = (p_parent * U_parent - p_small * U_small) / p_large

        Delta = (
            p_small * U_small[coordinate] ** 2
            + p_large * U_large[coordinate] ** 2
            - p_parent * U_parent[coordinate] ** 2
        ) / N

        V_btw += Delta
        sum_measured_delta += Delta
        sum_expected_g += best['g']
        if kind == 'level':
            n_level_splits += 1
        else:
            n_adjacency_splits += 1

        gamma_children = _split_gamma(Delta, best['g'])

        # Curvature is inherited from the parent, not re-measured:
        # both children of this split carry the SAME d2T the parent
        # bin carried.
        parent_d2T = best['d2T']

        del leaves[best['id']]

        leaf_small = dict(
            id=next_id, indices=idx_small, n=int(idx_small.size),
            U=U_small, d2T=parent_d2T, open=True, split=None, g=0.0, gamma=gamma_children,
        )
        leaf_large = dict(
            id=next_id + 1, indices=idx_large, n=int(idx_large.size),
            U=U_large, d2T=parent_d2T, open=True, split=None, g=0.0, gamma=gamma_children,
        )
        next_id += 2
        leaves[leaf_small['id']] = leaf_small
        leaves[leaf_large['id']] = leaf_large

        # Both children need v_k regardless of what happens next: a
        # child closed immediately below (Delta < tau_at_selection) is
        # still a FINAL bin if it is never split again, and the
        # V_win_hat gather at the end reuses this same cached value
        # rather than recomputing it. One call prices both.
        batch_v([leaf_small, leaf_large])

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

    V_win_hat = 0.0
    for leaf in leaves.values():
        if leaf['n'] > 1:
            idx = leaf['indices']
            psi_leaf = psi0_c[idx]
            var_k = _variance(psi_leaf)
            v_k = leaf['v']
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

    # The interval's constituents: the final bin set's mass, this
    # coordinate's own influence, and the second difference each bin
    # carries, ordered like `labels_final` (plan §36.2(6): the
    # second-order interval built from these is removed, but the
    # columns themselves stay -- they cost no evaluations).
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
