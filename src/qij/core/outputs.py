"""
The BCa acceleration (glossary: a_bca; plan §3, §6): the third-moment
constant of the refined influence estimate, used by `intervals.qij_interval`.
"""

from __future__ import annotations

import numpy as np


def acceleration(field: np.ndarray) -> float:
    """
    a_bca from the third moment of the refined influence estimate
    psi_hat(x_i) (glossary: "the per-point estimate whose third
    moment gives the acceleration"). `field` is (N,).

        a_bca = mean(field^3) / (6 * sqrt(N) * mean(field^2)^1.5)

    Equivalent to Efron's sum form sum(field^3) / (6 * sum(field^2)^1.5).
    O(N), one pass over the field; no loop over N.
    """
    field = np.asarray(field, dtype=float)
    n = field.shape[0]
    m2 = np.mean(field ** 2)
    m3 = np.mean(field ** 3)
    return float(m3 / (6.0 * np.sqrt(n) * m2 ** 1.5))
