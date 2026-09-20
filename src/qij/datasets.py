"""The paper's four datasets (plan Sec 5): pareto, mvt, fp, imf; plus truth.

Parametric datasets (`pareto`, `mvt`) sample the law directly, so their
draws are literally i.i.d. from the named distribution. Population datasets
(`fp`, `imf`) sample WITH replacement from the file shipped under
`qij/data/`, which is what makes the target theta_true exactly the
estimator applied to the empirical distribution of the shipped file, with
no finite-population correction.

The dataset parameters below are ported from vqboot's `dgp/pareto.py`,
`dgp/mvt.py` and `dgp/imf.py` (and vqboot's `configs/*.yaml`, which pinned
their numeric values for the paper): Pareto(alpha=2.0, x_min=1.0), a d=10,
nu=5.0 multivariate t, and the IMF's fixed breakpoint tau=0.176609 with its
(alpha, Mstar, p) optimization box. Only the parameters are ported, not the
old DGP class machinery.
"""

import functools
import pathlib
from typing import Callable, Tuple

import h5py
import numpy as np

_DATA_DIR = pathlib.Path(__file__).parent / 'data'

# ---- Pareto(alpha, x_min) --------------------------------------------
PARETO_ALPHA = 2.0
PARETO_X_MIN = 1.0

# ---- multivariate t, d dimensions, nu degrees of freedom --------------
MVT_D = 10
MVT_NU = 5.0

# ---- IMF: fixed breakpoint tau from the full-pool 4D DE fit, and the
# (alpha, Mstar, p) box that bounds the per-draw optimization. Never
# chosen from a draw.
IMF_TAU = 0.176609
IMF_BOUNDS = ((-10.0, -0.01), (1.0, 200.0), (0.1, 20.0))

# ---- Chabrier IMF (estimators.Chabrier): a candidate replacement for
# `IMF`, not wired into `study._DATASET_ESTIMATORS`. m_b is Chabrier's
# own break; m_min is the shipped stars.h5 pool's own minimum mass
# (`_imf_pool().min()`), fixed here exactly the way `IMF_TAU` is fixed
# from a full-pool fit -- never a draw's own minimum, which would make
# the normalization depend on which points a resample happened to keep.
# `CHABRIER_BOUNDS` is the (m_c, sigma, x) optimizer box, parallel to
# `IMF_BOUNDS`.
CHABRIER_M_B = 1.0
CHABRIER_M_MIN = 0.0017582750879228115
CHABRIER_BOUNDS = ((0.01, 5.0), (0.02, 5.0), (0.05, 10.0))

_PARAMETRIC_TRUTH = {
    ('pareto', 'shape'): np.array([PARETO_ALPHA]),
    ('pareto', 'tail'): np.array([0.01]),   # P(X > F^-1(0.99)) = 0.01 by construction
    ('mvt', 'nu'): np.array([MVT_NU]),
    ('mvt', 'tail'): np.array([0.01]),      # P(||x|| > F^-1(0.99)) = 0.01 by construction
}


def pareto(N: int, seed: int) -> np.ndarray:
    """Draw N points from Pareto(alpha=2.0, x_min=1.0). Returns (N,)."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(0.0, 1.0, size=N)
    return PARETO_X_MIN * (1.0 - U) ** (-1.0 / PARETO_ALPHA)


def mvt(N: int, seed: int) -> np.ndarray:
    """Draw N points from a d=10, nu=5.0 multivariate t. Returns (N, 10)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(N, MVT_D))
    V = rng.chisquare(MVT_NU, size=(N, 1))
    return Z / np.sqrt(V / MVT_NU)


@functools.lru_cache(maxsize=1)
def _fp_pool() -> np.ndarray:
    data = np.load(_DATA_DIR / 'fp_sdss.npz')
    return data['fp_data']   # (76997, 3): [log_sigma, log_I_e, log_R_half]


def fp(N: int, seed: int) -> np.ndarray:
    """Draw N galaxies WITH replacement from the 76,997-galaxy SDSS pool."""
    rng = np.random.default_rng(seed)
    pool = _fp_pool()
    idx = rng.choice(len(pool), size=N, replace=True)
    return pool[idx]


@functools.lru_cache(maxsize=1)
def _imf_pool() -> np.ndarray:
    with h5py.File(_DATA_DIR / 'stars.h5', 'r') as f:
        return f['data/BH_Mass'][()]   # (19857,)


def imf(N: int, seed: int) -> np.ndarray:
    """Draw N stellar masses WITH replacement from the 19,857-star pool."""
    rng = np.random.default_rng(seed)
    pool = _imf_pool()
    idx = rng.choice(len(pool), size=N, replace=True)
    return pool[idx]


def mvt_vq_transform(X: np.ndarray) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """The MVT's standardization: per-coordinate mean/sd, mapping X to the
    whitened 𝒳-VQ coordinates. Returns (Z, inverse); `inverse` closes over
    this call's mean and std and maps rows in Z's coordinates back to X's
    (whitened prototype positions back to the estimator's own coordinates).
    Not applied by `mvt` itself: the study passes this callable to
    `QIJ(vq_transform=mvt_vq_transform)` for the MVT draws only (plan Sec 5,
    last sentence)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Z = (X - mean) / std

    def inverse(W: np.ndarray) -> np.ndarray:
        return W * std + mean

    return Z, inverse


@functools.lru_cache(maxsize=None)
def truth(dataset: str, estimator: str) -> np.ndarray:
    """theta_true (q,): closed form for pareto/mvt; for fp/imf, the
    estimator applied to the whole shipped file with unit weights. Memoized
    on (dataset, estimator), so the population estimator fits at most once
    regardless of how many draws are taken."""
    key = (dataset, estimator)
    if key in _PARAMETRIC_TRUTH:
        return _PARAMETRIC_TRUTH[key]
    from . import estimators as _est
    if key == ('fp', 'fp'):
        pool = _fp_pool()
        return _est.fp(pool, np.ones(len(pool)))
    if key == ('imf', 'imf'):
        pool = _imf_pool()
        T = _est.IMF(tau=IMF_TAU, bounds=IMF_BOUNDS)
        return T(pool, np.ones(len(pool)))
    if key == ('imf', 'chabrier'):
        pool = _imf_pool()
        T = _est.Chabrier(m_min=CHABRIER_M_MIN, m_b=CHABRIER_M_B,
                           bounds=CHABRIER_BOUNDS)
        return T(pool, np.ones(len(pool)))
