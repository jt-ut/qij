"""
The one weight constructor and the one difference rule (method outline
steps 3 and 7), used by both stage 1 (the 𝒳-VQ's prototype influences,
`xvq.prototype_influences`) and stage 2 (the 𝓘-VQ's bin differences).

The difference step is RELATIVE to the mass it moves (glossary,
"difference step"): a bin or prototype of mass p is stepped to mass
p*(1 + delta), so delta is a fraction of that set's own mass, the
natural unit for a derivative along its direction. `step_parameter`
turns a relative step delta into the weight parameter t the one weight
constructor takes; U stays the derivative with respect to t.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def forward_step(eta: float) -> float:
    """
    The forward-difference step delta_f = 2*sqrt(eta) (method outline
    step 3), a fraction of the mass being moved. Used wherever only an
    upward step is taken, which is always feasible: stage 1's prototype
    influences and stage 2's single-evaluation child measurement.
    """
    return float(2.0 * np.sqrt(eta))


def central_step(eta: float) -> float:
    """
    The central/one-sided difference step delta = (3*eta)^(1/3) (method
    outline step 7), a fraction of the mass being moved. Used by stage
    2's bin differences.
    """
    return float(np.cbrt(3.0 * eta))


def step_parameter(delta: float, p: float) -> float:
    """
    The weight parameter t of a relative step delta on a member set of
    mass p (method outline steps 3 and 7):

        t = delta * p / (1 - p)

    The one weight constructor below moves the member set's mass from p
    to (1 - t)*p + t; setting that equal to p*(1 + delta) gives this t.
    Every member weight is then multiplied by exactly (1 + delta) and
    every other weight by (1 - t), so the step is the same fraction of
    its own mass whatever that mass is.
    """
    return float(delta * p / (1.0 - p))


def perturbed_weights(
    omega0: np.ndarray,
    member_mask: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    The one weight constructor: weights at signed weight parameter t
    along e_K - p, for member set K = {i : member_mask[i]}, base weights
    omega0 summing to R = omega0.size.

        p = sum_{i in K} omega0_i / R
        omega_i(t) = (1 - t) * omega0_i + t * omega0_i * 1{i in K} / p

    Every omega_i(t) sums to R for every t. With t from
    `step_parameter(delta, p)` this multiplies each member weight by
    (1 + delta) and K's mass becomes p*(1 + delta). Called directly by
    stage 1 (omega0 = M_used * p, K = a single prototype) and by stage
    2's bin differences (omega0 = ones(N), K = a bin's member mask).
    """
    omega0 = np.asarray(omega0, dtype=float)
    member_mask = np.asarray(member_mask, dtype=bool)
    R = omega0.size
    p = float(omega0[member_mask].sum()) / R
    return (1.0 - t) * omega0 + t * omega0 * member_mask.astype(float) / p


def difference(
    T_base: np.ndarray,
    p: float,
    delta: float,
    evaluate: Callable[[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    The one difference rule (method outline step 7), used by stage 2's
    bin differences. The relative steps delta and 2*delta on a member
    set of mass p are the weight parameters t = step_parameter(delta, p)
    and 2*t (t is linear in delta), and U is the derivative with respect
    to t, so the stencils are the usual ones in t -- equivalently the
    stencils in delta multiplied by (1 - p)/p = delta/t, once for U and
    twice for the second difference.

    Central when the downward step keeps every weight non-negative: at
    -delta each member weight is multiplied by (1 - delta) and every
    other by 1 + delta*p/(1 - p) > 0, so that is exactly delta <= 1:

        U   = [T(+t) - T(-t)] / (2*t)
        d2T = [T(+t) - 2*T_base + T(-t)] / t^2

    otherwise one-sided second-order:

        U   = [-3*T_base + 4*T(+t) - T(+2t)] / (2*t)
        d2T = [T_base - 2*T(+t) + T(+2t)] / t^2

    Exactly two calls to `evaluate`, which takes the weight parameter,
    always in the order the rule requires (central: +t then -t;
    one-sided: +t then +2t); exceptions from `evaluate` propagate
    uncaught.

    Returns (U, d2T, one_sided), each of T_base's shape except
    `one_sided`, a plain bool.
    """
    T_base = np.asarray(T_base, dtype=float)
    t = step_parameter(delta, p)
    one_sided = not (delta <= 1.0)

    if not one_sided:
        T_plus = np.asarray(evaluate(+t), dtype=float)
        T_minus = np.asarray(evaluate(-t), dtype=float)
        U = (T_plus - T_minus) / (2.0 * t)
        d2T = (T_plus - 2.0 * T_base + T_minus) / (t ** 2)
    else:
        T_h = np.asarray(evaluate(+t), dtype=float)
        T_2h = np.asarray(evaluate(+2.0 * t), dtype=float)
        U = (-3.0 * T_base + 4.0 * T_h - T_2h) / (2.0 * t)
        d2T = (T_base - 2.0 * T_h + T_2h) / (t ** 2)

    return U, d2T, one_sided
