"""`python -m qij.check`: the one check in the package (plan §10, amendment 6).

The weighted-mean scale identity (old plan §5.3):
`S_btw + S_win == (1/N) sum_i psi_i^2`, where `psi_i = x_i - weighted_mean(x)`
is the weighted mean's exact analytic influence, evaluated on the
pipeline's own final 𝓘-VQ bins (`CoordinateResult.labels`, amendment 6):
`S_btw = sum_k p_k * psi_bar_k^2`, `S_win = sum_k p_k * Var_k(psi)`. This is
an exact ANOVA decomposition of the same `psi` values, so it holds to
machine precision for ANY partition or the partition arithmetic is wrong
-- it is the only verification code in the package, and it can fail.
"""
from __future__ import annotations

import numpy as np

from .qij import QIJ


def _weighted_mean(X, w):
    return np.array([np.average(X[:, 0], weights=w)])


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 1))
    N = X.shape[0]

    res = QIJ().fit(X, _weighted_mean)
    psi = X[:, 0] - res.theta_hat[0]

    cr = res.coordinates[0]
    L = cr.L
    counts = np.bincount(cr.labels, minlength=L).astype(float)
    p = counts / N
    psi_bar = np.bincount(cr.labels, weights=psi, minlength=L) / counts
    psi_sq_bar = np.bincount(cr.labels, weights=psi ** 2, minlength=L) / counts
    psi_var = psi_sq_bar - psi_bar ** 2

    S_btw = float(np.sum(p * psi_bar ** 2))
    S_win = float(np.sum(p * psi_var))
    S_tot = float(np.mean(psi ** 2))

    rel_err = abs((S_btw + S_win) - S_tot) / abs(S_tot)
    status = 'PASS' if rel_err <= 1e-12 else 'FAIL'
    print(f'S_btw + S_win == (1/N) sum psi^2: rel_err={rel_err:.3e}  {status}')


if __name__ == '__main__':
    main()
