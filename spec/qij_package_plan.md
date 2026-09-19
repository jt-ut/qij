*Live copy in the `qij` package since 19 September 2026; the `vqboot/spec/` copy is frozen.*

# The `qij` package: plan for the coding manager (18 September 2026, ruled)

One standalone package. The user-facing part is three lines. Results are stored per dataset, per estimator, per estimation method, at the finest granularity any figure needs; tables and figures are computed downstream from those products and read nothing else. No registries, manifests, hashes, checksums or version stamps: reproduction is the same config, the same seed, the same package. Terms and symbols per `QIJ_glossary.md`; the method per `QIJ_method_outline.md`.

## 1. What a user types
```python
import numpy as np
from qij import QIJ, Bootstrap

def T(X, w):                      # any function of the data and non-negative weights summing to len(X)
    return np.array([np.average(X[:, 0], weights=w)])

res = QIJ().fit(X, T)             # defaults eps=0.01, eta=machine precision; QIJ(eps=..., eta=..., seed=...)
res.interval(0.95)                # (q, 2) lower and upper limits
res.variance                      # (q,)  V̂_tot
res.summary()                     # small DataFrame: V_btw, V̂_win, V̂_tot, B̂, â_BCa, L, evaluations, normalized rows, wall time, per output

ref = Bootstrap(B=2000).fit(X, T) # the comparator: .interval(q), .variance, .summary(), .replicates (B, q)
```
Two classes with `fit`, one result object with `interval`, `variance`, `summary`. An estimator is a plain callable. Vector estimators return more than one number. Studies may pass `QIJ().fit(X, T, influence=psi)` with `psi(X, w)` the analytic influence at the fitted parameters; the method never uses it, and the result then also carries the oracle variance and per-point values for figures.

## 2. Layout
```
qij/                        the package repo: code and the two shipped population files; nothing else
  pyproject.toml            deps numpy, scipy, pandas, pyyaml, joblib, h5py, matplotlib, vqlp
  src/qij/
    __init__.py             QIJ, Bootstrap
    qij.py  bootstrap.py  result.py
    core/                   xvq.py influence_model.py ivq.py differences.py refine.py outputs.py intervals.py counter.py
    estimators.py           the paper's six estimators (§4)
    datasets.py             the paper's four datasets (§5)
    data/                   the SDSS Fundamental Plane catalogue and the STARFORGE stellar masses (they define θ_true for the two population datasets, so they are part of what the package is)
    study.py                the loop that writes the products (§6)
    tables.py  figures.py   downstream (§8)
  scripts/                  run.py  make_tables.py  make_figures.py

QIJ_WSOM2026/               the project folder, the user's, outside the repo
  configs/                  main.yaml  cost_vs_n.yaml  smoke.yaml  timing.yaml
  runs/                     the products (§3)
```
A study configuration is the user's, not the package's (ruled 18 September): a user changes S from 10 to 1000 by editing a file in their project folder, never the installed package. Both the configuration and the output directory are given at invocation and nothing in the package holds a path to either: `python scripts/run.py <project>/configs/main.yaml <project>/runs/main --workers 14`. On TACC the project folder sits on scratch; the finished `runs/` is synced back for the paper. The only package-relative path is `src/qij/data/`, deliberately.

## 3. The products (per dataset, per estimator, per method)
```
qij_runs/main/
  config.yaml                          a copy of what was run
  pareto/shape/   truth.parquet  boot.h5  qij.parquet  qij_partition.parquet  qij_points.parquet
  pareto/tail/    ...
  mvt/nu/  mvt/tail/  fp/fp/  imf/imf/
  tables/  figures/                    written downstream
```
The **draw index s** joins everything: draw s of a dataset is `X_s = dataset(N, seed = master + s)`, shared by every estimator and both methods; every row carries `s` and the seed.

- **`truth`**, the true sampling distribution by Monte Carlo. One row per draw: `s`, `seed`, `theta_true_<output>` (the same on every row), `theta_hat_<output>` (the estimator on X_s). The θ̂ columns over draws are the sampling distribution: their variance is the true variance, their quantiles the reference interval, and θ_true against another method's limits is coverage. Where an analytic influence exists, `V_oracle_<output>` per draw.
- **`boot`**, one row per draw and replicate: `s`, `b`, `theta_<output>` for the replicate; per draw the wall time and the count of failed (NaN) replicates. The full S × B × q array (about 8 million values for the FP at production size, 60 MB). The interval after the first b replicates is a prefix quantile computed downstream at any b and level; this is what the frontier needs.
- **`qij`**, one row per draw: `s`; per output `V_btw`, `V_win_hat`, `V_tot_hat`, `B_hat`, `a_bca`; the diagnostics `M_X`, `L`, `n_level_splits`, `n_adjacency_splits`, `rho`, `gain_ratio`, `ell`, `lam`, `ell_bound`, `lam_bound`, evaluations and rows by stage, `normalized_rows`, wall time by stage, failed-evaluation count. The interval is not stored: it is a function of θ̂, V̂_tot and â_BCa at a level, computed downstream like the bootstrap's.
- **`qij_partition`**, the first 200 draws when an analytic influence exists: `s`, `M`, and V_btw/V_tot for the 𝒳-VQ receptive fields, the 𝓘-VQ bins from ψ̂₀, and the bins from the true ψ, at M ∈ {4, 6, 8, 12, 16, 24, 32, 48, 64}.
- **`qij_points`**, one designated draw: `i`, `psi0_<output>`, `psi_<output>`, `sigma_<output>`.

Per-output quantities are columns named by the estimator's output names; a four-output estimator is one row per draw with four times the columns, never four rows. Adding a method later means adding one product beside `qij` and `boot`, touching nothing else.

## 4. Estimators (`estimators.py`)
One interface: a callable `T(X, w) -> ndarray(q)` with attributes `name`, `outputs` and `eta`, and optionally `T.influence(X, w) -> ndarray(N, q)`. A user's function is wrapped with `outputs=('theta',)` and `eta` at machine precision if not given. Three levels of complexity behind the one interface:
- **Closed forms** (Pareto shape, Pareto tail, MVT tail, weighted mean): a function with the attributes attached by a decorator, `@estimator(outputs=('alpha',), eta=EPS)`; the influence a one-line companion.
- **Vector closed form** (FP): weighted total least squares, one eigendecomposition, four outputs, `eta = 1e-12`; the influence is the existing implicit-function calculation. The population standardization is a module constant.
- **Solved scalar** (MVT ν): a one-dimensional root-find of the weighted profile score in log ν; the tolerance is `eta`.
- **The IMF**, a class constructed once from the dataset's config: `imf = IMF(tau=..., bounds=...)`, outputs `('slope', 'Mstar', 'p', 'gamma_shape', 'gamma_scale')`, `eta = 1e-6` (the square root of the objective tolerance). Inside: the audited continuity solve (bracketed root-find), the constrained objective, and **one start computed by the method of moments from (X, w) itself**: low-mass gamma shape and scale from the weighted mean and variance of the masses below the fixed split τ (shape = mean²/variance, scale = variance/mean); the high-mass slope from the weighted Hill estimator above τ, one over the weighted mean of log(m/τ); the cutoff mass from the weighted 99th percentile; the sharpness from the ratio of two upper quantiles or the model default. The split is the fixed τ, never chosen from the data. No warm-start cache, no pool start, no second start: every evaluation depends only on its own (X, w), so a bootstrap replicate and a QIJ perturbation are computed identically and deterministically. **Analytic influence (derived 18 September):** the weighted likelihood is closed form, so the influence follows from the analytic score and Hessian through the second-order implicit function theorem across the two continuity constraints; the one non-elementary piece, the derivative of the upper incomplete gamma with respect to its order, is evaluated by quadrature on its defining integral, once per fit. Verified by the O(ε) scaling of the influence's prediction of a one-weight perturbation over three decades.

**Optimizer (ruled 18 September, after the fit was repaired).** The fit uses the analytic gradient and Hessian that the influence derivation already provides (the per-observation score is the objective's gradient; the Hessian is A times the weight sum), in a trust-region method. With finite-difference gradients the search inherited the continuity solver's tolerance as noise and wandered to a bound on up to a quarter of replicates; with exact gradient and Hessian the failure rate is zero in 1,800 replicates on six draws. Bound hits are still returned as NaN and counted, for both methods under the one rule, and the box stays where the physics put it; with the repaired fit they do not occur.

**The cutoff sharpness p is bimodal under resampling, and that is the finding.** With every fit converging, the bootstrap variance for p is 70–80 times the oracle's on the same draws while the other four outputs agree with the oracle to within about a quarter (which also confirms the analytic influence by an independent route). The replicates do not fail; they converge to a second mode. A resample that drops the rare massive stars loses what identifies the cutoff, so a sharper cutoff at a higher M* becomes the maximum; QIJ's evaluations never drop a point and stay at the primary mode. The fraction of replicates in the second mode varies from 0.4% to 29% between draws, so a single draw cannot tell a user whether theirs is affected. p is kept as an output, its box is not narrowed, and the paper reports this as the rare-support case: an identifiability failure of resampling, not a numerical one, which no better optimizer can answer.

**What the paper measures it against.** The factor of eighty is against the oracle linearization at the primary mode. The truth product decides: the Monte-Carlo sampling distribution of p̂ over the S draws is the reference, and if some draws' full-data fits themselves land in the second mode, that distribution is bimodal and the oracle understates it; coverage of both methods' intervals against θ_true is then the honest comparison, together with the fraction of draws whose full-data fit is in the second mode. For the IMF the study therefore records, per draw, the fraction of replicates in the second mode (p above a fixed cut of 5, the antimode) and, in the truth product, whether the draw's own fit is there. Failure-fraction columns are dropped from T1 where every entry is zero.

Three conventions for all levels: an estimator never raises (a failed fit returns NaN and the study counts NaN rows per method, which is how a failure rate on rare-support data is recorded); an estimator holds no randomness; an estimator does not know which method is calling it, the weight vector being the only difference between a replicate and a perturbation.

**What the method does with a failed evaluation (ruled 18 September).** A failed evaluation is a NaN result; the counter counts it. In stage 1, a prototype whose evaluation failed is a missing response: it is left out of the mass-centering (ψ̄ over the finite prototypes only, weighted by their masses) and out of the influence model's design points; the model still predicts at every data point; the paper says "fitted on the prototypes whose influence could be evaluated". In stage 2, a failed evaluation of an initial bin makes the draw's QIJ result NaN for every output, counted as a failed QIJ draw, the counterpart of a failed bootstrap replicate; a failed evaluation during refinement cancels that split (the parent stays a bin, closed) and is counted. Nothing is retried.

## 5. Datasets (`datasets.py`)
Plain functions returning a draw: `pareto(N, seed)`, `mvt(N, seed)`, `fp(N, seed)`, `imf(N, seed)`. Parametric datasets sample the law; population datasets sample with replacement from the file shipped with the package. `truth(dataset, estimator)` returns θ_true: closed form for parametric datasets; the estimator on the whole file for population ones, cached in memory. Pareto and MVT each carry two estimators, FP and IMF one each with several outputs. `vq_transform` (standardization for the MVT) is a keyword the study passes; a user sees it only as `QIJ(vq_transform=...)` if their data need it.

## 6. The method (`core/`) and the loop (`study.py`)
`core/` is `QIJ_method_outline.md` with the glossary's final formulas (γ_k, v_k), and three implementation rules: per-bin quantities from bin-summed kernel vectors, never per point (the 8-hour estimate at N = 77 000 was a per-point computation of a quantity the method uses only per bin); the influence model's eigendecomposition per width shared across a vector estimator's outputs; deterministic for a given (X, T, eta, eps, seed), the seed entering only the 𝒳-VQ. The counter wraps `T` so the analytic influence is unreachable from the method.

`run_study(config, out_dir, workers)` is one loop: for each dataset and draw s, `X = dataset(N, master + s)`; for each estimator, θ̂ = T(X, 1), `Bootstrap(B, seed).fit(X, T)`, `QIJ(eps, eta, seed).fit(X, T, influence=psi)`; append one row to `truth`, B rows to `boot`, one row to `qij` (and the partition and points products where configured). Both methods on a draw run in the same process on one worker with one BLAS thread. Draws run in parallel with joblib; rows already present are skipped, so a run resumes.

## 7. Sizes and seeds
- Main study: N = 2000 for every dataset (the author's decision, 19 September: one sample size for all four; the IMF needs 2000 for identifiability, so all use it), S = 1000, B = 2000 for every dataset including the IMF (about 100 workers on TACC; time is not a constraint), eps = 0.01, one master seed; levels for downstream intervals {0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99}.
- Cost study: FP at N ∈ {1000, 3000, 10 000, 30 000, 76 997}, S = 100, B = 2000.
- Smoke: main with S = 3, B = 50. Timing: main with S = 20, one worker, one thread.
- Seeds: `seed = master + s`, used for the data draw, the 𝒳-VQ and the bootstrap generator alike; recorded in every row. Nothing hashed.

## 8. Downstream (`tables.py`, `figures.py`)
Every table and figure is a function of the products in §3 and reads nothing else. T1 per (dataset, estimator, output): medians and 5th/95th of V_btw/V_tot, V̂_tot/V_tot and V̂_tot/V_boot (V_tot from `truth`'s θ̂ variance and, where present, the oracle), coverage at 0.95 of both intervals with Monte-Carlo SE, width ratio, L, evaluations, normalized rows, wall time. The coverage grid for F2(b). The cost table. Figures F2, F3, F7, F9, F10 per `QIJ_figure_spec_rev51.md` and the 18 September rulings (one interval; F2(a) two markers against the oracle; F3 linear axis and the fixed labels; F7 with the FP once; F9 four scatter panels and the Pareto curve; F10 four FP outputs with per-output bands).

**Cost, normalized in three layers, used in this order.** (1) Evaluation counts and normalized rows: exact, machine-independent, the paper's cost claim. (2) Wall time only as the ratio QIJ / bootstrap within the same draw on the same worker, which cancels the machine and its load. (3) Absolute seconds only from the timing run (one worker, one thread), labelled as such. Never a wall time compared across runs.

## 9. Not carried over
The analytic-route package (`methods/`, `engine.py`, `variance_correction.py`, `study.py`, `influence.py`, `variance.py`, the old plotting), the gate scripts and `_common.py`, the joint-metric and joint-𝓘-VQ modules, the second IMF estimator and its warm-start cache, `runs_full/`. Analytic influence functions survive only as the companion functions in `estimators.py`.

## 10. Verification and size
No tests, no assertions in production code, one check: `python -m qij.check` runs the weighted-mean scale identity through `QIJ().fit`. The smoke config is the reproducibility path. The line count is reported, not enforced (expected about 4 500 for the whole package; the method itself is about 2 500); bloat is caught by §12, not by a cap.

## 11. Order of work
1. `core/`, the two classes and the result object; `qij.check` passes; the three-line example runs on random data.
2. `estimators.py` and `datasets.py`, the IMF as one class with the moment start and the audited solver.
3. `study.py` and `scripts/run.py`; smoke runs end to end on the four datasets from the project folder's config and writes the products of §3 there.
4. `tables.py`, `figures.py`; F2, F3, F7, F9 reproduced from a smoke run.
5. Install on TACC; run `main`, `cost_vs_n` and `timing`; their tables and figures replace everything in the paper.

## 12. Rules for coding agents

**Tests are structurally impossible, not merely forbidden.**
- Every task names, by path, the files the agent may create or edit, taken from the layout in §2. Anything outside that list is rejected at review without being read. There is no file in which a test could live.
- "Done" is a run, not a check: `python scripts/run.py configs/smoke.yaml /tmp/smoke` writes the products of §3 and `python scripts/make_figures.py /tmp/smoke` renders. The smoke run is the only permitted way to find out whether code works.
- `qij.check` is the only verification code in the package. The reviewer's first question on any diff is whether it adds a second, under any name.

**Bloat is caught by its shape, not by a line count.** These are rejected on sight: a wrapper around a wrapper; a `utils`, `common` or `helpers` module; a configuration layer between the YAML and the function that reads it; a defensive branch for a case the plan says cannot occur; an option, flag or parameter the plan does not name; a second code path for a case the first already covers; a fallback the plan does not name. A module over about 300 lines is read for these before merge; a fix that grows a file is never blocked on size.

**Efficiency and simplicity are one rule.**
- No Python loop over the N data points. Loops run over prototypes, bins and draws only. The hot paths state their cost in the docstring ("O(L·M²), never O(N·M²)") so a reviewer can see a violation.
- The plan is the specification. A behaviour, parameter or fallback that the plan does not name is out; if the plan cannot be implemented as written, the agent stops and quotes the sentence.
- One way to do each thing: if two functions could compute the same quantity, one is deleted.
- Functions take arrays and return arrays; state lives in the result object only.
- Names and docstrings use the glossary's words and nothing else.

**The reviewer's checklist**, applied by the coding manager to every diff before acceptance: (1) files touched are on the task's list; (2) no test or check-like file; (3) no loop over N; (4) no option, flag, branch or fallback the plan does not name; (5) no dead code; (6) names and docstrings in the glossary's words; (7) the smoke run passes. A diff failing any item is returned, not fixed by the reviewer.
