# qij

Variance for estimators without a closed-form influence function.

```python
import numpy as np
from qij import QIJ, Bootstrap

def T(X, w):                       # any function of the data and non-negative
    return np.array([np.average(X[:, 0], weights=w)])   # weights summing to len(X)

res = QIJ().fit(X, T)
res.interval(0.95)                 # (q, 2) lower and upper limits
res.variance                       # (q,)  the variance estimate
res.summary()                      # per output: the terms, the bin count, the cost

ref = Bootstrap(B=2000).fit(X, T)  # the comparator, same three methods
```

An estimator is a plain callable. Vector estimators return more than one number.
A study may pass `QIJ().fit(X, T, influence=psi)` where `psi(X, w)` is the
analytic influence; the method never uses it, and the result then also carries
the oracle variance for comparison.

## Install

`vqlp` is the vector quantizer and is not on PyPI, so install it first.

```sh
pip install git+https://github.com/jt-ut/vqlp.git
pip install git+https://github.com/jt-ut/qij.git      # or: pip install -e . from a checkout
```

Python 3.9. The two population files the paper's datasets draw from —
the SDSS Fundamental Plane catalogue and the STARFORGE stellar masses —
ship inside the package, because they define the estimand: change the file
and you change what is being measured.

## Reproducing the paper

Configurations and outputs are **the user's**, not the package's, and both are
given at invocation. Nothing in the package holds a path to either.

```sh
python scripts/run.py        <project>/configs/main.yaml <project>/runs/main --workers 100
python scripts/make_tables.py  <project>/runs/main <project>/runs/timing
python scripts/make_figures.py <project>/runs/main --cost-vs-n <project>/runs/cost_vs_n
```

The project folder alongside this package holds `configs/`, `runs/` and a
`runbook.sh` that carries every command with its worker count and the reasoning
behind it. On a cluster the runs go to scratch and the finished folder comes
back.

A run is **resumable**: draws already written are skipped, so a job that hits a
wall clock can be requeued and will continue.

### Parallelism

The one axis is the draw. Each worker runs one draw's QIJ and its bootstrap in
the same process with **one BLAS thread**, and that is deliberate: the only
defensible timing is the ratio of the two methods within a draw on the same
worker, which cancels the machine and its load. Giving the quantizer threads,
or spreading the bootstrap replicates, would give the two arms different
parallel efficiency and that ratio would stop meaning anything.

It is also sufficient — the main study is 1000 independent draws — so do not
raise `OMP_NUM_THREADS` above 1.

## Verification

```sh
python -m qij.check
```

runs the weighted-mean scale identity through the full pipeline and prints the
relative error. It is the only check in the package, by design: there are no
tests, and "done" is a run, not an assertion.

## Layout

```
src/qij/
  qij.py  bootstrap.py  result.py     the three-line surface
  core/                               the method: xvq, influence_model, ivq,
                                      refine, differences, outputs, intervals
  estimators.py  datasets.py          the paper's six estimators, four datasets
  study.py                            the loop that writes the products
  tables.py  figures.py               downstream, reading only those products
```

Every table and figure is a function of the study's products and reads nothing
else. The products are written at the finest granularity any of them needs, so
an analysis thought of later can usually be done without re-running anything.
