"""`QIJ`: the influence-aligned variance method (plan §1, §6).

`QIJ(eps=0.01, eta=None, seed=0, vq_transform=None).fit(X, T, influence=None)`
runs stage 1 (the 𝒳-VQ, the prototype influences, the initial influence
estimate) and stage 2 (the one shared full-data evaluation, then
per-output gain-driven refinement) and returns a `QIJResult`. `eta=None`
takes the per-evaluation relative accuracy from `T.eta`; `seed` drives the
𝒳-VQ only (plan §7). A plain callable `T` is wrapped with
`outputs=('theta',)` and machine-precision `eta` (plan §4); an estimator
already carrying `name`, `outputs`, `eta` is used as given. `influence`,
when given, is recorded as the oracle variance and per-point oracle
values and is never reached by the method: the method sees `T` only
through `Counter` (plan §1; interface sheet §4 `core/counter.py`).
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional

import numpy as np

from .core.counter import Counter
from .core.influence_model import fit_influence_model
from .core.influence_model import psi0 as _psi0
from .core.influence_model import uncertainty as _uncertainty
from .core.refine import run_refinement
from .core.xvq import cost_rule_M, run_xvq
from .result import QIJResult


def _wrap(T):
    """A plain callable gains `outputs=('theta',)` and machine-precision
    `eta` (plan §4); a `T` already carrying `outputs` is used as given."""
    if hasattr(T, 'outputs'):
        return T
    T.outputs = ('theta',)
    T.eta = float(np.finfo(float).eps)
    T.name = getattr(T, '__name__', 'theta')
    return T


class QIJ:
    """The QIJ method (plan §1, §6)."""

    def __init__(self, eps: float = 0.01, eta: Optional[float] = None,
                 seed: int = 0, vq_transform=None,
                 influence_model: str = 'gp', fitc_rank: Optional[int] = None) -> None:
        self.eps = eps
        self.eta = eta
        self.seed = seed
        self.vq_transform = vq_transform
        # The influence-model switch (FITC smoke-test addition): 'gp'
        # (default, bit-identical to before this option existed) or
        # 'fitc' (sparse, inducing-point); `fitc_rank` is the FITC
        # inducing-point count, None meaning "compute the default per
        # coordinate group" (`core.influence_model._fitc_default_rank`).
        # Both are threaded straight through to `core.influence_model.
        # fit_influence_model`, which validates `influence_model` itself.
        self.influence_model = influence_model
        self.fitc_rank = fitc_rank

    def fit(self, X: np.ndarray, T, influence=None) -> QIJResult:
        """Run the method on one draw. Order (plan §6): the 𝒳-VQ and
        prototype influences and the initial influence estimate (one
        `core.xvq.run_xvq` call) -> the one shared full-data base
        evaluation θ̂ -> per output, gain-driven refinement
        (`core.refine.run_refinement`)."""
        t_start = time.perf_counter()
        X = np.asarray(X)
        N = len(X)
        T = _wrap(T)
        eta = self.eta if self.eta is not None else T.eta
        outputs = T.outputs
        q = len(outputs)

        counter = Counter(T, N)

        # `vq_transform` returns (Z, inverse): Z the whitened coordinates,
        # `inverse` the map back to the coordinate space T expects, fitted
        # to this draw (amendment 4 -- the MVT standardization depends on
        # this draw's own mean and sd, so the inverse cannot be a static
        # attribute on the callable). Identity when `vq_transform` is None.
        Z, inverse = self.vq_transform(X) if self.vq_transform is not None else (X, lambda A: A)
        Z = np.asarray(Z, dtype=float)
        # The quantizer needs a two-dimensional Z; a one-dimensional
        # dataset's estimator (Pareto, the IMF) still wants rows back in
        # its own native (N,) shape, so the column promotion is undone on
        # the way out rather than pushed into `core` or the estimator.
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
            inverse = lambda A, _inv=inverse: _inv(A.reshape(-1))

        M_requested = cost_rule_M(N, q, self.eps)

        t0 = time.perf_counter()
        xvq, theta_Q, I_proto = run_xvq(Z, inverse, counter, eta, M_requested, self.seed)
        # W_X: the prototypes' positions in T's own native coordinates --
        # exactly what `core.xvq.run_xvq` already mapped `xvq.centers`
        # through `inverse` to get, to evaluate the prototype influences
        # (`core/xvq.py`'s `prototype_influences`). `run_xvq` does not
        # return that intermediate, so it is recovered here by the same
        # pure, deterministic `inverse` call on the same `xvq.centers` --
        # not a second way to derive a quantity, since `inverse` performs
        # no estimation. Needed on the result for `qij_prototypes.parquet`
        # (plan's F9 measured-points product).
        W_X = np.asarray(inverse(xvq.centers), dtype=float)
        model = fit_influence_model(Z, xvq, I_proto, theta_Q, eta,
                                     influence_model=self.influence_model,
                                     fitc_rank=self.fitc_rank)
        psi0_all = _psi0(model, Z)
        sigma_all = _uncertainty(model, Z)
        wall_time_prototype = time.perf_counter() - t0
        ev1, rows1 = counter.snapshot()

        t0 = time.perf_counter()
        theta_hat = np.asarray(counter(X, np.ones(N)), dtype=float)
        wall_time_full_data = time.perf_counter() - t0

        t0 = time.perf_counter()
        coordinates = [
            run_refinement(
                X, counter, theta_hat, c, name,
                psi0_all[:, c], float(model.offset[c]), sigma_all[:, c],
                I_proto[:, c], xvq.bmu, xvq.bmu2,
                eta, self.eps, xvq.M_used, bool(model.constant_path[c]),
                Z, model,
            )
            for c, name in enumerate(outputs)
        ]
        # A failed initial-bin evaluation NaNs the whole draw, not just
        # the coordinate it happened on (plan §4): the between-bin term
        # and everything after it rests on that measurement, and every
        # output shares the same stage-2 initial bins in spirit (each
        # coordinate's own build), so one coordinate's failure means the
        # draw as a whole is not a usable QIJ draw. Diagnostics that
        # genuinely happened (bin counts, evaluations, wall time, the
        # per-coordinate `L`/`M_used`/`labels`/split counts) are left
        # alone -- only the variance quantities (V_btw, V_win_hat,
        # V_tot_hat, B_hat, a_bca) are voided to NaN, on every
        # coordinate, not just the one whose own bins failed.
        if any(cr.failed for cr in coordinates):
            coordinates = [
                replace(
                    cr, V_btw=float('nan'), V_win_hat=float('nan'),
                    V_tot_hat=float('nan'), B_hat=float('nan'), a_bca=float('nan'),
                )
                for cr in coordinates
            ]
        wall_time_refinement = time.perf_counter() - t0
        ev3, rows3 = counter.snapshot()

        refine_evals = sum(cr.n_refine_evals for cr in coordinates)
        evals_by_stage = {
            'prototype': ev1,
            'full_data': (ev3 - ev1) - refine_evals,
            'refinement': refine_evals,
            'total': ev3,
        }
        rows_by_stage = {
            'prototype': rows1,
            'full_data': (rows3 - rows1) - refine_evals * N,
            'refinement': refine_evals * N,
            'total': rows3,
        }
        wall_time_total = time.perf_counter() - t_start
        wall_time_by_stage = {
            'prototype': wall_time_prototype,
            'full_data': wall_time_full_data,
            'refinement': wall_time_refinement,
            'total': wall_time_total,
        }

        oracle_variance = None
        psi_oracle = None
        if influence is not None:
            psi_oracle = np.asarray(influence(X, np.ones(N)), dtype=float)
            # The variance of theta_hat, not of the influence: an estimator
            # with influence psi has asymptotic variance mean(psi^2) / N.
            # Without the 1/N this is off by exactly the sample size, which
            # is how the validation run found it -- every oracle ratio came
            # out at 0.001 on the nose at N = 1000.
            oracle_variance = np.mean(psi_oracle ** 2, axis=0) / N

        return QIJResult(
            theta_hat=theta_hat, outputs=outputs, name=T.name, N=N,
            coordinates=coordinates, xvq=xvq, model=model, theta_Q=np.asarray(theta_Q, dtype=float),
            W_X=W_X, I_proto=np.asarray(I_proto, dtype=float),
            psi0=psi0_all, sigma=sigma_all,
            evaluations=ev3, rows=rows3, n_failed=counter.failed,
            evals_by_stage=evals_by_stage, rows_by_stage=rows_by_stage,
            wall_time_by_stage=wall_time_by_stage,
            oracle_variance=oracle_variance, psi_oracle=psi_oracle,
        )
