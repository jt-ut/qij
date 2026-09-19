"""The study loop (plan Sec 3, Sec 6, Sec 7): `run_study(config, out_dir,
workers)` writes the truth, boot and qij products -- and, wherever an
estimator carries an analytic influence, qij_partition, qij_points and
qij_prototypes -- for every dataset, sample size, estimator and draw
named by a config.

One loop, one sentence (plan Sec 6): for each dataset and draw s,
`X = dataset(N, master + s)`; for each of that dataset's estimators,
`theta_hat = T(X, ones)`, `Bootstrap(B, seed).fit(X, T)`,
`QIJ(eps, eta, seed).fit(X, T, influence=psi)`; append one row to
`truth`, B rows to `boot`, one row to `qij` (and the partition, points
and prototypes rows wherever `T.influence` exists). `s` is the join key
everywhere: draw s of a dataset is the same X_s for every estimator and
both methods -- `qij_points` and `qij_prototypes` each carry `s` too, so
either joins back to that draw's `qij.parquet` row. Adding a method later
means adding one more product write inside `_run_draw`'s per-estimator
body -- `boot` and `qij` never share a code path beyond that shared `X`.

Config schema (interface sheet Amendment 6; `configs/*.yaml` is the
datasets agent's file, on disk as of that amendment):

    master_seed: <int>       # draw s uses seed = master_seed + s
    eps: <float>              # QIJ(eps=...)
    S: <int>                  # draws per dataset
    B: <int>                  # bootstrap replicates
    datasets:                 # a LIST of dicts, not a mapping
      - name: <dataset name>  # one of pareto, mvt, fp, imf (datasets.py)
        N: [<int>, ...]       # always a list, even one element
        estimators: [<T.name>, ...]   # e.g. [shape, tail] for pareto

There is no `partition` or `points` config key (Amendment 6: one fewer
configuration layer, plan Sec 12): whether an estimator's draws get a
`qij_partition` row and which draw is `qij_points`'s (and
`qij_prototypes`'s) designated one are both decided from `T.influence`
alone, per the module constants `PARTITION_S` and `POINTS_DRAW` below,
not from the config.

`vq_transform` is never a config key (plan Sec 5): the MVT dataset alone
gets `datasets.mvt_vq_transform` passed to `QIJ`, every other dataset
gets none, decided here from the dataset name, not from the config.

Product paths (interface sheet Sec 5, amended): `<out_dir>/<dataset>/
<T.name>/{truth.parquet, boot.h5, qij.parquet, qij_partition.parquet,
qij_points.parquet, qij_prototypes.parquet}` when a dataset's config
carries one N; when it carries more than one (only `cost_vs_n.yaml`), an
`N<size>` directory is inserted before the product files.

Resume: a draw s is considered done for a dataset once its `s` appears in
every one of that dataset's estimators' `truth.parquet` -- written last,
after `boot.h5` and `qij.parquet`, so a draw interrupted partway is
retried in full next time (`boot.h5`'s fixed-index slot write and
`qij.parquet`'s dedup-on-flush make that retry idempotent).
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import datasets as _datasets
from . import estimators as _est
from .bootstrap import Bootstrap
from .core.xvq import fit_xvq
from .qij import QIJ

# ---- study-level constants (plan Sec 3, Sec 7) ------------------------

PARTITION_S = 200            # qij_partition: the first 200 draws
POINTS_DRAW = 0               # qij_points: the one designated draw
PARTITION_M_GRID = (4, 6, 8, 12, 16, 24, 32, 48, 64)
_FLUSH_EVERY = 20

# (dataset, T.name) -> callable (interface sheet Amendment 2). IMF is
# built once, from the same (tau, bounds) `datasets.truth` uses, so the
# study's theta_hat and datasets.truth's theta_true agree on the model.
_DATASET_ESTIMATORS = {
    'pareto': {'shape': _est.pareto_shape, 'tail': _est.pareto_tail},
    'mvt': {'nu': _est.mvt_nu, 'tail': _est.mvt_tail},
    'fp': {'fp': _est.fp},
    'imf': {'imf': _est.IMF(tau=_datasets.IMF_TAU, bounds=_datasets.IMF_BOUNDS)},
}

# The MVT's standardization, passed to `QIJ` for the MVT draws only
# (plan Sec 5, interface sheet Amendment 4); every other dataset passes
# nothing.
_VQ_TRANSFORM = {'mvt': _datasets.mvt_vq_transform}


def _pin_threads() -> None:
    """Both methods for a draw run in one process with one BLAS thread
    (plan Sec 6), so the QIJ/bootstrap wall-time ratio within a draw is
    meaningful; `scripts/run.py` also sets these before numpy is
    imported, which this call alone cannot do for the main process."""
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'


# ─────────────────────────────────────────────────────────────────────
# One draw, every estimator, both methods (joblib worker; module-level
# so loky can pickle it)
# ─────────────────────────────────────────────────────────────────────

def _run_draw(dataset_fn, N, s, master_seed, estimator_items, vq_transform,
              eps, B):
    """
    One draw s of one dataset: `X = dataset_fn(N, seed)` once, then for
    every (name, T, theta_true) in `estimator_items`, `Bootstrap` and
    `QIJ` both fit `X`. Returns `(s, seed, {name: row-dict})`; a worker
    never touches disk -- the caller does every write, which is what
    makes both `boot.h5`'s fixed-index slots and `truth.parquet`'s
    resume marker safe under joblib's out-of-order completion.

    The partition, points and prototypes rows are written wherever
    `T.influence` exists, not by a config flag (Amendment 6):
    `qij_res.psi_oracle` is None exactly when no `influence` was passed
    to `QIJ.fit`, i.e. `T` has no `.influence`.
    """
    seed = master_seed + s
    X = dataset_fn(N, seed)
    ones = np.ones(len(X))

    per_est = {}
    for name, T, theta_true in estimator_items:
        theta_hat = np.asarray(T(X, ones), dtype=float)
        boot_res = Bootstrap(B=B, seed=seed).fit(X, T)
        psi_fn = getattr(T, 'influence', None)
        qij_res = QIJ(eps=eps, seed=seed, vq_transform=vq_transform).fit(
            X, T, influence=psi_fn)

        truth_row = {'s': s, 'seed': seed}
        for j, o in enumerate(T.outputs):
            truth_row[f'theta_true_{o}'] = float(theta_true[j])
            truth_row[f'theta_hat_{o}'] = float(theta_hat[j])
        if qij_res.oracle_variance is not None:
            for j, o in enumerate(T.outputs):
                truth_row[f'V_oracle_{o}'] = float(qij_res.oracle_variance[j])

        partition_rows = None
        if qij_res.psi_oracle is not None and s < PARTITION_S:
            Z = vq_transform(X)[0] if vq_transform is not None else X
            Z = np.asarray(Z, dtype=float)
            if Z.ndim == 1:
                # `fit_xvq` needs two dimensions; `qij.py` does this same
                # promotion at its own `core` boundary. No inverse to
                # compose here -- `_partition_rows` only partitions, it
                # never evaluates the estimator.
                Z = Z.reshape(-1, 1)
            partition_rows = _partition_rows(
                s, T.outputs, qij_res.psi_oracle, qij_res.psi0, Z,
                PARTITION_M_GRID, seed)

        points_df = None
        prototypes_df = None
        if qij_res.psi_oracle is not None and s == POINTS_DRAW:
            points_df = _points_frame(
                s, T.outputs, qij_res.psi0, qij_res.psi_oracle, qij_res.sigma)
            prototypes_df = _prototypes_frame(
                s, T.outputs, qij_res.xvq, qij_res.W_X, qij_res.I_proto)

        per_est[name] = dict(
            truth_row=truth_row,
            qij_row=_qij_row(s, T.outputs, qij_res),
            boot_theta=boot_res.replicates,
            boot_n_failed=boot_res.n_failed,
            boot_wall_time=boot_res.wall_time,
            partition_rows=partition_rows,
            points_df=points_df,
            prototypes_df=prototypes_df,
        )
    return s, seed, per_est


def _qij_row(s: int, outputs, res) -> dict:
    """One `qij.parquet` row (interface sheet Sec 5, Amendment 3): `s`
    plus the shared, unsuffixed diagnostics (`M_X` from the shared
    X-VQ, the per-stage evaluation/row/wall-time dicts from `Counter`,
    `normalized_rows`, `n_failed`), then, per output, the suffixed
    `CoordinateResult` fields and the per-coordinate influence-model
    fields (`ell`=`model.width`, `lam`=`model.lam`, the two
    `at_bound` flags). Every value is read off `res`; nothing here
    re-derives any part of the method.

    Three more per-output columns, `bin_mass_<o>`, `bin_influence_<o>`
    and `bin_d2T_<o>`, each a plain Python list of length `L_<o>`
    (pyarrow stores these as list<double>): `CoordinateResult`'s
    `bin_mass` (p_k), `bin_influence` (that coordinate's centered
    bin influence U_k) and `bin_d2T` (that bin's uncentered second
    difference), i.e. the constituents of a quadratic surrogate of the
    estimator in the bin masses. They cost kilobytes per draw and let
    `core.intervals.qij2_interval` recompute a second-order interval
    downstream, at any level, with NO further estimator evaluations --
    the same spirit as this module's other products, where no interval
    is ever stored (`core/intervals.py`'s module docstring)."""
    row = {'s': s, 'M_X': int(res.xvq.M_used), 'n_failed': int(res.n_failed)}
    for stage in ('prototype', 'full_data', 'refinement', 'total'):
        row[f'evals_{stage}'] = int(res.evals_by_stage[stage])
        row[f'rows_{stage}'] = int(res.rows_by_stage[stage])
        row[f'wall_time_{stage}'] = float(res.wall_time_by_stage[stage])
    row['normalized_rows'] = float(res.rows_by_stage['total']) / res.N

    model = res.model
    for j, o in enumerate(outputs):
        c = res.coordinates[j]
        row[f'V_btw_{o}'] = float(c.V_btw)
        row[f'V_win_hat_{o}'] = float(c.V_win_hat)
        row[f'V_tot_hat_{o}'] = float(c.V_tot_hat)
        row[f'B_hat_{o}'] = float(c.B_hat)
        row[f'a_bca_{o}'] = float(c.a_bca)
        row[f'L_{o}'] = int(c.L)
        row[f'n_level_splits_{o}'] = int(c.n_level_splits)
        row[f'n_adjacency_splits_{o}'] = int(c.n_adjacency_splits)
        row[f'rho_{o}'] = float(c.rho)
        row[f'gain_ratio_{o}'] = float(c.gain_ratio)
        row[f'n_refine_evals_{o}'] = int(c.n_refine_evals)
        row[f'ell_{o}'] = float(model.width[j])
        row[f'lam_{o}'] = float(model.lam[j])
        row[f'ell_bound_{o}'] = bool(model.at_bound[j, 0])
        row[f'lam_bound_{o}'] = bool(model.at_bound[j, 1])
        row[f'bin_mass_{o}'] = np.asarray(c.bin_mass, dtype=float).tolist()
        row[f'bin_influence_{o}'] = np.asarray(c.bin_influence, dtype=float).tolist()
        row[f'bin_d2T_{o}'] = np.asarray(c.bin_d2T, dtype=float).tolist()
    return row


def _quantile_labels(values: np.ndarray, M: int) -> np.ndarray:
    """M equal-count bin labels for a 1-D array, by rank -- O(N log N)
    via one argsort, never a Python loop over the N points."""
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return (ranks * M) // len(values)


def _btw_over_tot(value: np.ndarray, labels: np.ndarray, p: np.ndarray) -> float:
    """V_btw/V_tot of `value` under the partition (`labels`, bin masses
    `p`): bin sums via `np.bincount`, O(N + len(p)), never a Python loop
    over points or bins beyond that one vectorized call."""
    N = value.shape[0]
    sums = np.bincount(labels, weights=value, minlength=len(p))
    counts = np.bincount(labels, minlength=len(p)).astype(float)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    v_btw = float(np.sum(p * means ** 2)) / N
    v_tot = float(np.mean(value ** 2)) / N
    return v_btw / v_tot if v_tot > 0 else float('nan')


def _partition_rows(s, outputs, psi_oracle, psi0_hat, Z, M_grid, seed):
    """`qij_partition.parquet` rows for draw s (plan Sec 3): at each M,
    the true influence's V_btw/V_tot ratio under three M-bin
    partitions -- the X-VQ receptive fields (`core.xvq.fit_xvq` refit
    at this M; the one place `study.py` calls a `core` partitioning
    primitive directly, since the product needs a whole M grid and a
    single `QIJ.fit` gives only its own operating M), an M-quantile
    split of psi0_hat (the model's own estimate), and an M-quantile
    split of the true psi itself (the bound any M-bin partition of
    this coordinate could achieve)."""
    rows = []
    for M in M_grid:
        xvq_obj = fit_xvq(Z, M, seed)
        row = {'s': s, 'M': M}
        for j, o in enumerate(outputs):
            value = psi_oracle[:, j]
            row[f'xvq_{o}'] = _btw_over_tot(value, xvq_obj.labels, xvq_obj.p)

            labels_psi0 = _quantile_labels(psi0_hat[:, j], M)
            p_psi0 = np.bincount(labels_psi0, minlength=M).astype(float) / len(value)
            row[f'ivq_psi0_{o}'] = _btw_over_tot(value, labels_psi0, p_psi0)

            labels_true = _quantile_labels(value, M)
            p_true = np.bincount(labels_true, minlength=M).astype(float) / len(value)
            row[f'ivq_true_{o}'] = _btw_over_tot(value, labels_true, p_true)
        rows.append(row)
    return rows


def _points_frame(s, outputs, psi0_hat, psi_oracle, sigma) -> pd.DataFrame:
    """`qij_points.parquet` rows for the designated draw: `s` (the draw
    index, so a join back to that draw's `qij.parquet` row is possible),
    `i`, plus per output `psi0`, `psi` and `sigma` -- built as one dict
    of column arrays, never a Python loop over the N points."""
    N = psi0_hat.shape[0]
    data = {'s': np.full(N, s), 'i': np.arange(N)}
    for j, o in enumerate(outputs):
        data[f'psi0_{o}'] = psi0_hat[:, j]
        data[f'psi_{o}'] = psi_oracle[:, j]
        data[f'sigma_{o}'] = sigma[:, j]
    return pd.DataFrame(data)


def _prototypes_frame(s, outputs, xvq, W_X, I_proto) -> pd.DataFrame:
    """`qij_prototypes.parquet` rows for the designated draw (plan Sec 3
    "What to add" 1): `s`, `j` (the prototype index), `p` (its
    receptive-field mass, `xvq.p`, summing to 1 over the file), `w_0,
    w_1, ...` (its position in T's OWN native coordinates, `res.W_X` --
    not the whitened quantizer coordinates), and per output `I_<output>`
    (the measured, mass-centred prototype influence, `res.I_proto`, as
    `core.xvq.prototype_influences` returns it -- NaN where a
    prototype's evaluation failed, left as NaN, not dropped or filled,
    so the row stays aligned with `j`). Built as one dict of column
    arrays, never a Python loop over the M_used prototypes."""
    M_used = W_X.shape[0]
    W_X2 = np.asarray(W_X, dtype=float).reshape(M_used, -1)
    data = {'s': np.full(M_used, s), 'j': np.arange(M_used), 'p': xvq.p}
    for d in range(W_X2.shape[1]):
        data[f'w_{d}'] = W_X2[:, d]
    for j, o in enumerate(outputs):
        data[f'I_{o}'] = I_proto[:, j]
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────
# Resumable I/O -- parquet products
# ─────────────────────────────────────────────────────────────────────

def _truth_done(path: str) -> set:
    """The set of draws already in `truth.parquet`, the canonical
    per-(dataset, estimator) resume marker (module docstring)."""
    if not os.path.exists(path):
        return set()
    return set(int(v) for v in pd.read_parquet(path, columns=['s'])['s'])


def _append_parquet(path: str, rows: list, key) -> None:
    """Append `rows` to the parquet at `path`, deduplicating on `key`
    (keep last) so a retried draw's earlier partial row is replaced,
    not doubled."""
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), new_df], ignore_index=True)
    else:
        df = new_df
    subset = key if isinstance(key, list) else [key]
    df = df.drop_duplicates(subset=subset, keep='last')
    df = df.sort_values(subset).reset_index(drop=True)
    df.to_parquet(path, index=False)


def _write_designated_frame(path: str, df: pd.DataFrame) -> None:
    """Write a designated-draw product (`qij_points.parquet`,
    `qij_prototypes.parquet`) outright: unlike `_append_parquet`, there
    is exactly one draw's rows to write, so no dedup-on-flush is
    needed."""
    df.to_parquet(path, index=False)


def _flush(buffers: dict, est_dirs: dict) -> None:
    for name, buf in buffers.items():
        d = est_dirs[name]
        if buf['truth']:
            _append_parquet(os.path.join(d, 'truth.parquet'), buf['truth'], 's')
            buf['truth'] = []
        if buf['qij']:
            _append_parquet(os.path.join(d, 'qij.parquet'), buf['qij'], 's')
            buf['qij'] = []
        if buf['partition']:
            _append_parquet(os.path.join(d, 'qij_partition.parquet'),
                             buf['partition'], ['s', 'M'])
            buf['partition'] = []


# ─────────────────────────────────────────────────────────────────────
# Resumable I/O -- boot.h5 (interface sheet Sec 5)
# ─────────────────────────────────────────────────────────────────────

def _open_boot_h5(path: str, S: int, B: int, q: int, outputs) -> h5py.File:
    """Open `boot.h5` for incremental, per-draw writes: fixed-shape
    datasets sized to the full `S` so a partially complete file can be
    resumed and extended (interface sheet Sec 5). An unwritten draw's
    `s` slot stays -1, the sentinel a resumed run would use to tell it
    apart from a written one, were `boot.h5` ever consulted for resume
    on its own (it currently is not: `truth.parquet` is the resume
    marker, module docstring)."""
    if os.path.exists(path):
        return h5py.File(path, 'a')
    f = h5py.File(path, 'w')
    f.create_dataset('theta', shape=(S, B, q), dtype='f8',
                      fillvalue=np.nan, chunks=(1, B, q))
    f.create_dataset('s', shape=(S,), dtype='i8', data=np.full(S, -1, dtype=np.int64))
    f.create_dataset('seed', shape=(S,), dtype='i8', data=np.zeros(S, dtype=np.int64))
    f.create_dataset('wall_time', shape=(S,), dtype='f8', data=np.full(S, np.nan))
    f.create_dataset('n_failed', shape=(S,), dtype='i8', data=np.zeros(S, dtype=np.int64))
    f.attrs['outputs'] = list(outputs)
    return f


def _write_boot_slot(f: h5py.File, s: int, seed: int, res: dict) -> None:
    f['theta'][s] = res['boot_theta']
    f['s'][s] = s
    f['seed'][s] = seed
    f['wall_time'][s] = res['boot_wall_time']
    f['n_failed'][s] = res['boot_n_failed']


# ─────────────────────────────────────────────────────────────────────
# Per-dataset, per-sample-size loop
# ─────────────────────────────────────────────────────────────────────

def _run_dataset(dataset_name, dataset_fn, estimators, N, S, B, master_seed,
                  eps, vq_transform, est_dirs, workers) -> None:
    for d in est_dirs.values():
        os.makedirs(d, exist_ok=True)

    theta_true = {name: _datasets.truth(dataset_name, T.name)
                  for name, T in estimators.items()}

    done = None
    for name, d in est_dirs.items():
        s_here = _truth_done(os.path.join(d, 'truth.parquet'))
        done = s_here if done is None else (done & s_here)
    pending = [s for s in range(S) if s not in done]
    if not pending:
        return

    boot_files = {
        name: _open_boot_h5(os.path.join(d, 'boot.h5'), S, B,
                             len(estimators[name].outputs), estimators[name].outputs)
        for name, d in est_dirs.items()
    }
    buffers = {name: {'truth': [], 'qij': [], 'partition': []} for name in estimators}
    estimator_items = [(name, estimators[name], theta_true[name]) for name in estimators]

    parallel = Parallel(n_jobs=workers, backend='loky', return_as='generator_unordered')
    gen = parallel(
        delayed(_run_draw)(
            dataset_fn, N, s, master_seed, estimator_items, vq_transform, eps, B,
        )
        for s in pending
    )

    n_done = 0
    for s, seed, per_est in gen:
        for name, res in per_est.items():
            d = est_dirs[name]
            _write_boot_slot(boot_files[name], s, seed, res)
            buffers[name]['truth'].append(res['truth_row'])
            buffers[name]['qij'].append(res['qij_row'])
            if res['partition_rows']:
                buffers[name]['partition'].extend(res['partition_rows'])
            if res['points_df'] is not None:
                _write_designated_frame(os.path.join(d, 'qij_points.parquet'), res['points_df'])
            if res['prototypes_df'] is not None:
                _write_designated_frame(os.path.join(d, 'qij_prototypes.parquet'), res['prototypes_df'])
        n_done += 1
        if n_done % _FLUSH_EVERY == 0:
            _flush(buffers, est_dirs)

    _flush(buffers, est_dirs)
    for f in boot_files.values():
        f.flush()
        f.close()


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def run_study(config: dict, out_dir: str, workers: int) -> None:
    """One loop over datasets, sample sizes and draws (plan Sec 6, module
    docstring's config schema). Draws run in parallel across `workers`
    joblib processes; within a draw, every estimator's truth, boot and
    qij rows are computed in that one process with one BLAS thread."""
    _pin_threads()
    master_seed = int(config['master_seed'])
    eps = float(config.get('eps', 0.01))
    S = int(config['S'])
    B = int(config['B'])
    os.makedirs(out_dir, exist_ok=True)

    for dcfg in config['datasets']:
        dataset_name = dcfg['name']
        dataset_fn = getattr(_datasets, dataset_name)
        vq_transform = _VQ_TRANSFORM.get(dataset_name)
        estimators = {name: _DATASET_ESTIMATORS[dataset_name][name]
                      for name in dcfg['estimators']}

        N_list = dcfg['N']
        multi_N = len(N_list) > 1
        for N in N_list:
            N = int(N)
            est_dirs = {}
            for name in estimators:
                d = os.path.join(out_dir, dataset_name, name)
                if multi_N:
                    d = os.path.join(d, f'N{N}')
                est_dirs[name] = d
            _run_dataset(
                dataset_name, dataset_fn, estimators, N, S, B, master_seed,
                eps, vq_transform, est_dirs, workers,
            )
