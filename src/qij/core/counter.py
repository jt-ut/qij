"""
`Counter`: wraps the estimator so the method sees only T(X, w), never
T.influence (plan §6, interface sheet §4). Counts evaluations, rows and
evaluations that returned any NaN, and exposes T.outputs, T.name and
T.eta as attributes of the same name -- nothing else of T.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class Counter:
    """
    Counting wrapper around one estimator T. `T` and `N` (the number of
    rows in the full data) are recorded at construction; every call to
    `T` is routed through `__call__`, which counts the evaluation, its
    rows and, if the result carries any NaN, the failure. There is no
    `__getattr__` forwarding, so nothing of `T` beyond `outputs`, `name`
    and `eta` is reachable through a `Counter`.
    """

    def __init__(self, T, N: int) -> None:
        self._T = T
        self.N = N
        self.outputs = T.outputs
        self.name = T.name
        self.eta = T.eta

        self.evaluations = 0
        self.rows = 0
        self.failed = 0

    def __call__(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        result = np.asarray(self._T(X, w), dtype=float)
        self.evaluations += 1
        self.rows += len(X)
        if np.any(np.isnan(result)):
            self.failed += 1
        return result

    def snapshot(self) -> Tuple[int, int]:
        """Return (evaluations, rows) so far, for per-stage differencing."""
        return (self.evaluations, self.rows)
