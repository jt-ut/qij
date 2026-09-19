"""
The 𝒳-VQ: quantize the data, then measure the influence at each
prototype by forward differences (method outline, Stage 1, steps 1-3).

`cost_rule_M` is the cost balance between the two stages that sets the
requested (and bounded) prototype count. `fit_xvq` fits the codebook and
returns its structure -- prototypes, receptive fields, masses, first and
second best-matching units, the CADJ adjacency -- with no estimator
evaluations. `prototype_influences` is steps 2-3 together: evaluate T
once on the prototypes (weights M_used * p) for the base value theta_Q,
then one forward difference per prototype against it. `run_xvq` is
stage 1 in full (interface sheet Amendment 1): it fits the codebook on
Z, maps the prototypes to T's native coordinates with `inverse`, and
computes the prototype influences -- the only place in `core/` that
needs to know about the 𝒳-VQ's whitening transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from vqlp import VQFitter

from .differences import forward_step, perturbed_weights, step_parameter

_KAPPA_REF = 2.7

# Row batch for the second-BMU repair, matching `influence_model`'s cap:
# it bounds the (rows, M_used) distance block at tens of megabytes rather
# than gigabytes at the cost study's largest sample size.
_BMU2_BATCH_CAP = 4096


@dataclass
class XVQ:
    centers: np.ndarray      # (M_used, d_z) whitened prototype positions
    labels: np.ndarray       # (N,) receptive-field index per point
    p: np.ndarray            # (M_used,) receptive-field masses, summing to 1
    bmu: np.ndarray          # (N,) best-matching prototype
    bmu2: np.ndarray         # (N,) second best-matching prototype, resolved
    conn: object             # (M_used, M_used) CADJ adjacency, sparse
    M_requested: int
    M_used: int


def cost_rule_M(N: int, q: int, eps: float) -> int:
    """
    The requested first-stage prototype count, from the cost balance
    between the two stages (method outline step 1):

        M_ref = ceil(sqrt(kappa_ref / eps)),  kappa_ref = 2.7
        M_X   = ceil(sqrt((1 + 2*q*M_ref) * N / 2))

    floored at 20 (a regression needs points) and capped at N // 2
    (beyond which stage 1 degrades toward a subsample jackknife) -- the
    𝒳-VQ is given the same row budget as the finite differences of
    stage 2.
    """
    M_ref = math.ceil(math.sqrt(_KAPPA_REF / eps))
    M_X = math.ceil(math.sqrt((1.0 + 2.0 * q * M_ref) * N / 2.0))
    return min(max(M_X, 20), N // 2)


def fit_xvq(Z: np.ndarray, M: int, seed: int) -> XVQ:
    """
    The first-stage quantizer (method outline step 1): k-means on Z via
    `vqlp.VQFitter`, with empty receptive fields dropped (M_used <= M),
    the per-point first and second best-matching units, and the directed
    CADJ adjacency, all reindexed into the live prototype space.

    O(M) evaluations of nothing -- this is the quantizer only; no
    estimator evaluation happens here.
    """
    Z = np.asarray(Z, dtype=float)
    N = Z.shape[0]

    fitter = VQFitter(M=M, p=2, max_bmu=2, random_state=seed, verbose=False)
    fitter.fit(Z)
    fitter.recall(Z)
    rec = fitter.recaller

    RFSize = rec.RFSize
    live_idx = np.where(RFSize > 0)[0]
    M_used = live_idx.size
    old2new = np.full(RFSize.shape[0], -1, dtype=np.intp)
    old2new[live_idx] = np.arange(M_used)

    BMU = rec.BMU
    bmu = old2new[BMU[:, 0]]
    bmu2 = old2new[BMU[:, 1]]

    p = RFSize[live_idx] / N
    centers = fitter.W[live_idx]
    conn = rec.CADJ.tocsr()[np.ix_(live_idx, live_idx)].tocsr()

    bmu2 = _resolve_bmu2(Z, centers, bmu, bmu2)

    return XVQ(
        centers=centers, labels=bmu, p=p, bmu=bmu, bmu2=bmu2,
        conn=conn, M_requested=M, M_used=M_used,
    )


def _resolve_bmu2(
    Z: np.ndarray,
    centers: np.ndarray,
    bmu: np.ndarray,
    bmu2: np.ndarray,
) -> np.ndarray:
    """
    Repair the second-BMU column: `bmu2` is -1 wherever a point's second
    best-matching unit pointed to a prototype dropped as empty (dropping
    empty receptive fields does not repair `bmu2` for the survivors).
    Each such point gets the nearest LIVE prototype other than its own
    `bmu`, by Euclidean distance in Z.

    Batched over rows: the number of points needing repair is not always
    small -- when the quantizer drops many empty receptive fields it can
    approach N -- and the distance block is then (N, M_used), about 1.4 GB
    at the cost study's largest sample size. The batch bounds it at a few
    tens of megabytes whatever fraction is missing, for the same reason
    `influence_model.psi0` and `uncertainty` are batched.
    """
    bmu2 = np.array(bmu2, dtype=int, copy=True)
    missing = bmu2 < 0
    if not missing.any():
        return bmu2

    Zm = Z[missing]
    bmu_m = bmu[missing]
    centers_sq = np.sum(centers ** 2, axis=1)[None, :]
    nearest = np.empty(Zm.shape[0], dtype=int)
    for start in range(0, Zm.shape[0], _BMU2_BATCH_CAP):
        sl = slice(start, start + _BMU2_BATCH_CAP)
        Zb = Zm[sl]
        D2 = np.sum(Zb ** 2, axis=1)[:, None] + centers_sq - 2.0 * Zb @ centers.T
        D2[np.arange(D2.shape[0]), bmu_m[sl]] = np.inf
        nearest[sl] = np.argmin(D2, axis=1)
    bmu2[missing] = nearest
    return bmu2


def prototype_influences(
    W_X: np.ndarray,
    counter,
    p: np.ndarray,
    eta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prototype influences (method outline steps 2-3): evaluate T once on
    the prototypes W_X (M_used, d), in T's native coordinates, with
    weights M_used * p, for the base value theta_Q; then for each
    prototype j, raise its mass by the forward step delta_f =
    2*sqrt(eta) OF ITSELF -- the weight parameter t_j =
    step_parameter(delta_f, p_j) along e_j - p, lowering the rest in
    proportion (`differences.perturbed_weights`) -- and difference the
    re-evaluation against theta_Q with respect to t_j, which is the
    forward difference divided by t_j. Mass-centred so
    sum_j p_j I_proto[j] = 0 (the constant-path test's scale reference
    is theta_Q, not this mean, which is ~0 by construction).

    1 + M_used evaluations of `counter` on M_used rows: O(M_used),
    never O(N).
    """
    M_used = len(p)
    omega0 = M_used * p
    delta_f = forward_step(eta)

    theta_Q = np.asarray(counter(W_X, omega0), dtype=float)
    q = theta_Q.shape[0]

    I_proto = np.empty((M_used, q), dtype=float)
    for j in range(M_used):
        member = np.zeros(M_used, dtype=bool)
        member[j] = True
        t_j = step_parameter(delta_f, float(p[j]))
        omega = perturbed_weights(omega0, member, t_j)
        I_proto[j] = (counter(W_X, omega) - theta_Q) / t_j

    # A failed evaluation at a prototype is a missing response (ruled 18
    # September): centre over the finite prototypes only, mass-weighted
    # and renormalized, per coordinate (a vector estimator can fail in a
    # way that leaves some outputs finite and others not). A prototype
    # left NaN by its own evaluation stays NaN here -- not filled in, not
    # dropped -- since I_proto is indexed by prototype and the influence
    # model reads that alignment. Identical to the unconditional mean
    # when every prototype is finite.
    finite = np.isfinite(I_proto)
    mass = np.sum(np.where(finite, p[:, None], 0.0), axis=0)
    psi_bar = np.sum(np.where(finite, p[:, None] * I_proto, 0.0), axis=0) / mass
    I_proto -= psi_bar[None, :]
    return theta_Q, I_proto


def run_xvq(
    Z: np.ndarray,
    inverse: Callable[[np.ndarray], np.ndarray],
    counter,
    eta: float,
    M: int,
    seed: int,
) -> Tuple[XVQ, np.ndarray, np.ndarray]:
    """
    Stage 1 in full (method outline steps 1-3, interface sheet
    Amendment 1): fit the codebook on Z (`fit_xvq`), map its prototype
    centers to T's native coordinates with `inverse` (the identity when
    `qij.py` was given no `vq_transform`), and compute the prototype
    influences (`prototype_influences`). This is the only function in
    `core/` that calls `inverse`, so nothing outside this module needs
    to know about the 𝒳-VQ's whitening.
    """
    xvq = fit_xvq(Z, M, seed)
    W_X = inverse(xvq.centers)
    theta_Q, I_proto = prototype_influences(W_X, counter, xvq.p, eta)
    return xvq, theta_Q, I_proto
