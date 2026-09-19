"""`Bootstrap`: the comparator method (plan §1).

`Bootstrap(B=2000, seed=0).fit(X, T)` draws `B` multinomial resamples of
`X` (weights non-negative, summing to `len(X)`, plan §4), evaluates `T` on
each, and returns a `BootstrapResult` carrying `.replicates`.
"""
from __future__ import annotations

import time

import numpy as np

from .qij import _wrap
from .result import BootstrapResult


class Bootstrap:
    """The percentile bootstrap (plan §1)."""

    def __init__(self, B: int = 2000, seed: int = 0) -> None:
        self.B = B
        self.seed = seed

    def fit(self, X: np.ndarray, T) -> BootstrapResult:
        X = np.asarray(X)
        N = len(X)
        T = _wrap(T)
        q = len(T.outputs)

        rng = np.random.default_rng(self.seed)
        # One replicate's weights at a time. Drawing all B at once costs
        # B*N floats -- 1.2 GB at the cost study's largest sample size with
        # B = 2000 -- to hold resamples that are used one after another and
        # never again. Drawing inside the loop holds one (N,) vector, and
        # the draw is negligible beside the estimator evaluation it feeds.
        probs = np.full(N, 1.0 / N)

        replicates = np.empty((self.B, q), dtype=float)
        t0 = time.perf_counter()
        for b in range(self.B):
            replicates[b] = T(X, rng.multinomial(N, probs).astype(float))
        wall_time = time.perf_counter() - t0

        n_failed = int(np.any(np.isnan(replicates), axis=1).sum())

        return BootstrapResult(
            outputs=T.outputs, replicates=replicates,
            n_failed=n_failed, wall_time=wall_time,
        )
