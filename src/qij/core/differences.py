"""
The one weight constructor and the one difference rule (method outline
steps 3 and 7), used by both stage 1 (the 𝒳-VQ's prototype influences,
`xvq.prototype_influences`) and stage 2 (the 𝓘-VQ's bin differences).
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def forward_step(eta: float) -> float:
    """
    The forward-difference step delta_f = 2*sqrt(eta) (method outline
    step 3). Used wherever only an upward step is taken, which is always
    feasible: stage 1's prototype influences and stage 2's single-
    evaluation child measurement.
    """
    return float(2.0 * np.sqrt(eta))


def central_step(eta: float) -> float:
    """
    The central/one-sided difference step delta = (3*eta)^(1/3) (method
    outline step 7). Used by stage 2's bin differences.
    """
    return float(np.cbrt(3.0 * eta))


def perturbed_weights(
    omega0: np.ndarray,
    member_mask: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    The one weight constructor: weights at signed step t along e_K - p,
    for member set K = {i : member_mask[i]}, base weights omega0 summing
    to R = omega0.size.

        p = sum_{i in K} omega0_i / R
        omega_i(t) = (1 - t) * omega0_i + t * omega0_i * 1{i in K} / p

    Every omega_i(t) sums to R for every t. Called directly by stage 1
    (omega0 = M_used * p, K = a single prototype) and by stage 2's bin
    differences (omega0 = ones(N), K = a bin's member mask).
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
    bin differences: central when the downward step keeps every weight
    non-negative, i.e. p >= delta / (1 + delta):

        U = [T(+delta) - T(-delta)] / (2*delta)
        d2T = [T(+delta) - 2*T_base + T(-delta)] / delta^2

    otherwise one-sided second-order:

        U = [-3*T_base + 4*T(+delta) - T(+2*delta)] / (2*delta)
        d2T = [T_base - 2*T(+delta) + T(+2*delta)] / delta^2

    Exactly two calls to `evaluate`, always in the order the rule
    requires (central: +delta then -delta; one-sided: +delta then
    +2*delta); exceptions from `evaluate` propagate uncaught.

    Returns (U, d2T, one_sided), each of T_base's shape except
    `one_sided`, a plain bool.
    """
    T_base = np.asarray(T_base, dtype=float)
    one_sided = not (p >= delta / (1.0 + delta))

    if not one_sided:
        T_plus = np.asarray(evaluate(+delta), dtype=float)
        T_minus = np.asarray(evaluate(-delta), dtype=float)
        U = (T_plus - T_minus) / (2.0 * delta)
        d2T = (T_plus - 2.0 * T_base + T_minus) / (delta ** 2)
    else:
        T_h = np.asarray(evaluate(+delta), dtype=float)
        T_2h = np.asarray(evaluate(+2.0 * delta), dtype=float)
        U = (-3.0 * T_base + 4.0 * T_h - T_2h) / (2.0 * delta)
        d2T = (T_base - 2.0 * T_h + T_2h) / (delta ** 2)

    return U, d2T, one_sided
