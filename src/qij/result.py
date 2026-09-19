"""Result objects (plan §1): `.interval(level)`, `.variance`, `.summary()`,
and `.replicates` for the bootstrap. No interval is stored on either
result: `.interval` calls `core.intervals` fresh every time, so the study
can ask for any level after the fact (plan §3 "qij.parquet": "The interval
is not stored").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .core import intervals
from .core.influence_model import InfluenceModel
from .core.refine import CoordinateResult
from .core.xvq import XVQ


@dataclass
class QIJResult:
    """The QIJ method's result for one draw (plan §1, §3 "qij.parquet").

    `.interval`, `.variance`, `.summary()` are the pinned user surface
    (interface sheet §1). The remaining fields are what `study.py` needs
    to write the qij, qij_partition, qij_points and qij_prototypes
    products (plan §3) without a second computation path of its own:
    `coordinates` (the per-output diagnostics from `core.refine`,
    including each output's influence `field` and its final 𝓘-VQ bin
    `labels`, per point -- `labels` is also what `qij.check`,
    amendment 6, evaluates the scale identity on), `xvq` and `model`
    (the fitted quantizer and influence model; `xvq.p` is also the
    receptive-field masses `qij_prototypes` writes as `p`), `theta_Q`
    (the stage-1 value, for the θ_Q/θ̂ discrepancy), `W_X` (the
    prototypes' positions in T's own native coordinates, `core.xvq.
    run_xvq`'s `inverse(xvq.centers)`) and `I_proto` (the measured,
    mass-centred prototype influences `core.xvq.prototype_influences`
    returns) together for `qij_prototypes`, `psi0`/`sigma` (per point,
    for `qij_points`), and `oracle_variance`/`psi_oracle` (only when
    `influence` was given to `.fit`, plan §1).
    """
    theta_hat: np.ndarray            # (q,)
    outputs: Tuple[str, ...]
    name: str
    N: int
    coordinates: List[CoordinateResult]
    xvq: XVQ
    model: InfluenceModel
    theta_Q: np.ndarray               # (q,) the stage-1 quantized-data value
    W_X: np.ndarray                   # (M_used,) or (M_used, d) prototype positions in T's native coordinates
    I_proto: np.ndarray               # (M_used, q) mass-centred measured prototype influence
    psi0: np.ndarray                  # (N, q)
    sigma: np.ndarray                 # (N, q)
    evaluations: int
    rows: int
    n_failed: int
    evals_by_stage: Dict[str, int]
    rows_by_stage: Dict[str, int]
    wall_time_by_stage: Dict[str, float]
    oracle_variance: Optional[np.ndarray] = None   # (q,)
    psi_oracle: Optional[np.ndarray] = None         # (N, q)

    @property
    def variance(self) -> np.ndarray:
        """V̂_tot per output (q,)."""
        return np.array([cr.V_tot_hat for cr in self.coordinates])

    @property
    def _a_bca(self) -> np.ndarray:
        return np.array([cr.a_bca for cr in self.coordinates])

    def interval(self, level: float) -> np.ndarray:
        """(q, 2) lower, upper; a pure function of θ̂, V̂_tot and â_BCa
        (interface sheet §4 `core/intervals.py`), computed fresh at any
        level."""
        return intervals.qij_interval(self.theta_hat, self.variance, self._a_bca, level)

    def summary(self) -> pd.DataFrame:
        """One row per output: `output, V_btw, V_win_hat, V_tot_hat, B_hat,
        a_bca, L, evaluations, normalized_rows, wall_time` (interface sheet
        §1). `evaluations`, `normalized_rows` and `wall_time` are the whole
        fit's totals, shared across outputs since the prototype and
        full-data stages are shared."""
        normalized_rows = self.rows_by_stage['total'] / self.N
        wall_time = self.wall_time_by_stage['total']
        rows = [
            dict(
                output=name,
                V_btw=cr.V_btw, V_win_hat=cr.V_win_hat, V_tot_hat=cr.V_tot_hat,
                B_hat=cr.B_hat, a_bca=cr.a_bca, L=cr.L,
                evaluations=self.evaluations, normalized_rows=normalized_rows,
                wall_time=wall_time,
            )
            for name, cr in zip(self.outputs, self.coordinates)
        ]
        return pd.DataFrame(rows, columns=[
            'output', 'V_btw', 'V_win_hat', 'V_tot_hat', 'B_hat', 'a_bca',
            'L', 'evaluations', 'normalized_rows', 'wall_time',
        ])


@dataclass
class BootstrapResult:
    """The bootstrap's result (plan §1): `.replicates` is the (B, q) array
    of resampled estimates."""
    outputs: Tuple[str, ...]
    replicates: np.ndarray            # (B, q)
    n_failed: int
    wall_time: float

    @property
    def variance(self) -> np.ndarray:
        """Var(θ*) per output (q,), over the non-failed replicates."""
        return np.nanvar(self.replicates, axis=0, ddof=1)

    def interval(self, level: float) -> np.ndarray:
        """(q, 2) lower, upper: the percentile interval of `.replicates`."""
        return intervals.percentile_interval(self.replicates, level)

    def summary(self) -> pd.DataFrame:
        """One row per output: `output, variance, B, n_failed, wall_time`
        (interface sheet §1)."""
        B = self.replicates.shape[0]
        rows = [
            dict(output=name, variance=v, B=B, n_failed=self.n_failed,
                 wall_time=self.wall_time)
            for name, v in zip(self.outputs, self.variance)
        ]
        return pd.DataFrame(rows, columns=[
            'output', 'variance', 'B', 'n_failed', 'wall_time',
        ])
