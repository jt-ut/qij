# QIJ method specification, revision 8 (19 September 2026)

**Purpose.** This is the document a reviewer checks the `qij` package against. It states the method as ruled, every formula with its conventions, every edge-case rule, the invariants that hold on the products, and which module implements which step. Where the code and this document disagree, that is a finding; where this document is silent, the code has no licence to add behaviour. Terms and symbols follow `QIJ_glossary.md` in this folder; subscripts i for data points, j for 𝒳-VQ prototypes, k for 𝓘-VQ bins, c for estimator outputs. The dated history of how the method was reached (`QIJ_revision_plan.md`, §1–§36) is not needed to review the code.

**Reading order for a reviewer.** §1 (contract), §2–§4 (the method), §5 (rules for failures and degenerate cases), §6 (products), §7 (invariants a reviewer can compute from the products without running anything), §8 (code map), §9 (what the code must not contain).

---

## 1. Contract

**Input.** Data X with N rows; an estimator T(X, ω) returning q outputs for any weights ω ≥ 0 with Σω = N, carrying three attributes: `name`, `outputs` (a tuple of q names) and `eta`, the declared relative accuracy of one evaluation. A plain callable is wrapped with `outputs = ('theta',)` and `eta` at machine precision. Two user numbers: ε, the share of the variance the user accepts leaving inside the bins (default 0.01), and a seed that drives the 𝒳-VQ only. Optionally a `vq_transform(X) -> (Z, inverse)` giving the quantizer's coordinates Z and the map back to T's coordinates (identity when absent; a one-dimensional Z is promoted to a column). Optionally an analytic influence `influence(X, ω) -> (N, q)`, recorded as the oracle and never read by the method.

**The method sees T only through the counter** (`core/counter.py`): every evaluation is counted with its row count, and any evaluation returning a NaN is counted as failed. Nothing else of T is reachable. T never raises; a failed fit returns NaN. T holds no randomness and does not know which method calls it.

**Output** (`QIJResult`), per output c: V_btw, V̂_win, V̂_tot = V_btw + V̂_win, B̂, â_BCa, and the interval h^{QIJ}_q at any level, computed on demand from θ̂, V̂_tot and â_BCa and never stored. Diagnostics per output: L, the numbers of level and adjacency splits, ρ, the realized-to-expected gain ratio, ℓ, λ, whether either sits on its search bound, the refinement evaluation count, the final bins' constituents (p_k, U_k, Δ²T_k), each point's bin label and refined influence. Shared: M_𝒳, θ_Q (the stage-1 value; never reported as an estimate), evaluations and rows by stage, wall time by stage, the failed-evaluation count.

**Cost.** Stage 1: 1 + M_𝒳 evaluations on M_𝒳 rows. Stage 2: 1 evaluation on N rows, then per output 2·M_𝓘,c evaluations on N rows for the initial bins and at most 1 + M_𝒳 refinement evaluations on N rows. Nothing is ever retried. Nothing in the method is computed by resampling.

---

## 2. Stage 1: the 𝒳-VQ and the initial influence estimate

**2.1 The prototype count.** M_ref = ⌈√(2.7/ε)⌉ (17 at the default). M_𝒳 = ⌈√((1 + 2 q M_ref) N / 2)⌉, floored at 20 and capped at ⌊N/2⌋. Empty receptive fields are dropped; M_𝒳 then denotes the count used.

**2.2 The quantizer.** k-means (`vqlp.VQFitter`, Euclidean, two best-matching units, `random_state = seed`) on Z. Products: prototypes w_j in Z's coordinates, receptive-field masses p_j = n_j/N, each point's nearest prototype b(i) and second-nearest b₂(i) (resolved against the live prototypes when the fitter's second unit was dropped), and the directed CADJ adjacency restricted to live prototypes. No estimator evaluation.

**2.3 The prototype influences.** Prototypes are mapped to T's coordinates by `inverse`. θ_Q = T(W, ω⁰) with ω⁰_j = M_𝒳 p_j: one evaluation on M_𝒳 rows. For each j, with δ_f = 2√η and t_j = δ_f p_j/(1 − p_j), the weights ω(t_j) = (1 − t_j) ω⁰ + t_j ω⁰ 1{j}/p_j raise prototype j's mass to p_j(1 + δ_f) and lower every other weight in proportion; I_j = [T(W, ω(t_j)) − θ_Q]/t_j. M_𝒳 evaluations on M_𝒳 rows. Mass-centring, per output, over the prototypes whose evaluation is finite: I_j ← I_j − Σ_j p_j I_j / Σ_j p_j (sums over finite j). A prototype whose evaluation failed keeps NaN in its row; it is neither filled nor dropped.

**2.4 The one weight constructor and the one step rule** (`core/differences.py`), shared by both stages. For a member set K of mass p in base weights ω⁰ (summing to the row count R), ω_i(t) = (1 − t) ω⁰_i + t ω⁰_i 1{i ∈ K}/p; Σω(t) = R for every t. A relative step δ on K is the weight parameter t = δ p/(1 − p), under which every member weight is multiplied by (1 + δ) and every other by (1 − t). Steps: δ_f = 2√η for a single forward difference (stage 1, refinement); δ = (3η)^{1/3} for the stencil of the initial bins.

**2.5 The initial influence estimate.** Z is whitened by its own mean and covariance (the transform derived from Z once and stored). Per output c, a Gaussian process on that output's finite prototypes: I_j = h(w_j)ᵀβ + f(w_j) + e_j, h(z) = (1, z) in whitened coordinates (constant-only when the finite design has at most d_z + 2 points), f ~ GP(0, s² k_ℓ) with the Matérn-3/2 kernel k_ℓ(r) = (1 + √3 r/ℓ) exp(−√3 r/ℓ) on whitened distance, e_j ~ N(0, s² λ) homoscedastic.

- **Width ℓ** is shared by all outputs that share the same finite design (all of them when no prototype failed), searched on [ℓ_min, ℓ_max] with ℓ_min the median whitened distance between CADJ-connected prototypes and ℓ_max ten times the largest inter-prototype distance: five log-spaced widths, then one bounded scalar search between the best grid point's neighbours; the objective is the sum over the group's outputs of the profiled restricted negative log marginal likelihood (REML: the affine basis projected out by an orthonormal complement Q, one eigendecomposition of QᵀK_ℓQ per width shared across the group).
- **Noise-to-signal ratio λ_c**, per output, at each candidate width: s_c² is profiled in closed form, s_c²(λ) = (1/(M − m)) Σ_i z_i²/(Λ_i + λ), and λ_c minimizes the profiled REML objective by a bounded search of log λ on [log max(λ_floor,c, 10⁻¹⁰), log 10²].
- **The declared noise floor (revision 8).** The prototype influences are differences of evaluations of accuracy η, so their noise standard deviation at prototype j is √2·η·|θ_Q,c|/t_j. The declared homoscedastic level is n_c² = 2 η² θ_Q,c² · median_j(1/t_j²) over the output's finite design. λ_floor,c is the unique root of λ·s_c²(λ) = n_c² (the left side, (1/(M − m)) Σ_i z_i² λ/(Λ_i + λ), is increasing in λ), found by one bracketed root find on log λ and pinned to the nearer edge of [10⁻¹⁰, 10²] when the equation has no root inside it. The floor is recomputed at every candidate width. For an estimator with η at machine precision the floor lies below 10⁻¹⁰ and the search is unchanged.
- At the chosen (ℓ, λ_c): one Cholesky of A = K_ℓ + λ_c I (jitter only here, escalating 0 → 10⁻¹⁰ tr(K)/M → ×10 → ×10; a fourth failure raises and the draw is recorded as failed), giving β_c, α_c = A⁻¹(I − Hβ_c), s_c².
- **Predictions at every data point:** ψ̂₀(x_i) = h(z_i)ᵀβ_c + k_iᵀα_c; σ_i² = s_c² [1 − k_iᵀA⁻¹k_i + r_iᵀG⁻¹r_i] with r_i = h(z_i) − HᵀA⁻¹k_i and G = HᵀA⁻¹H, clipped at 0. The **within-bin posterior variance** of a set K of points is v_K = mean(diag Σ_K) − mean(Σ_K), Σ_K the posterior covariance of f + hᵀβ over K, computed from bin-summed vectors (the double sums over K accumulated in row chunks), never by forming the |K| × |K| block.
- **Constant path.** An output whose finite design has fewer than 3 prototypes, or whose mass-weighted spread of I_j is at most 10⁻¹⁰ |θ_Q,c|, has no model: it is a collapsed stage 1 and the output fails under §5.3.
- Reported: ℓ, λ_c, and whether each sits within 1% in log of its search bound (for λ_c the lower bound is the floored one).

---

## 3. Stage 2: the 𝓘-VQ, the finite differences and gain-driven refinement

**3.1 The base value.** θ̂ = T(X, 1): one evaluation on N rows, shared by every output.

**3.2 The initial bins**, per output. M_𝓘 = min(M_ref, number of distinct ψ̂₀ values). One-dimensional k-means on the ψ̂₀(x_i): initial prototypes at the (k + ½)/M_𝓘 quantiles of the distinct values; assignment by midpoints (a value on a boundary goes to the lower prototype); an empty bin is dropped at once and the rest relabelled; iterate to a fixed assignment or 100 iterations. Bins are indexed in increasing order of their mean. M_𝓘,c denotes the count used. If M_𝓘,c ≤ 1 the output is a collapsed stage 1 (§5.3).

**3.3 The bin stencil.** For each initial bin k, mass p_k, weight parameter t_k = δ p_k/(1 − p_k), δ = (3η)^{1/3}: the central stencil
  U_k = [T(ω(+t_k)) − T(ω(−t_k))]/(2 t_k),   Δ²T_k = [T(ω(+t_k)) − 2θ̂ + T(ω(−t_k))]/t_k²,
two evaluations on N rows per bin, always in the order +t then −t. The downward step is feasible whenever δ ≤ 1 (every member weight is multiplied by 1 − δ ≥ 0), which holds for every η ≤ 1/3; the one-sided second-order stencil at +t, +2t is the rule's stated fallback and does not fire. *(The older method outline said one-sided; the central stencil is what is built and what this specification rules.)* U_k carries all q outputs; only column c is used for output c. Centring: r_c = Σ_k p_k U_k,c is subtracted from every U_k,c and stored as the centring residual; Δ²T is not centred. A bin's measurement is complete before the next bin's begins; a NaN in any U_k or Δ²T_k stops the loop (§5.2).

**3.4 Between-bin term and bias term** over the initial bins: V_btw = (1/N) Σ_k p_k U_k², B̂ = (1/2N) Σ_k p_k Δ²T_k.

**3.5 The FD-to-prediction scale.** ψ̃₀(x_i) = ψ̂₀(x_i) − mean_i ψ̂₀; ψ̄₀,k = mean of ψ̃₀ over bin k; ρ² = V_btw / ((1/N) Σ_k p_k ψ̄₀,k²), recomputed over the current bins after every accepted split.

**3.6 Refinement** (revision 8; one proposal per bin). Bins are leaves; each leaf carries its mass, its U (all q columns), its Δ²T, its γ (1 for an initial bin) and an open flag. Cost guard: at most 1 + M_𝒳 refinement evaluations. τ = ε V_btw / (current number of leaves), recomputed each iteration.

- **Proposal** for an open leaf with more than one point, using v_k (§2.5) for that leaf:
  - level split if Var_k(ψ̂₀) > v_k: two-means (the §3.2 quantizer at M = 2) on the leaf's ψ̂₀ values; children a, b; expected gain ĝ = ρ² (p_a ψ̄₀,a² + p_b ψ̄₀,b² − p_k ψ̄₀,k²)/N with ψ̄₀ the means of ψ̃₀;
  - otherwise an adjacency split: side a = points with I_{b₂(i)} > I_{b(i)} (this output's prototype influences), side b the rest; if either side is empty, the level split instead; expected gain ĝ = ρ² p_k v_k / N.
  A leaf with one point, or with no valid proposal, is closed.
- **Queue.** Take the open leaf with the largest ĝ (ties: lower leaf id). Stop when ĝ < τ or the cost guard is reached.
- **Measurement.** The smaller child s (ties: side a) is stepped by δ_f: t_s = δ_f p_s/(1 − p_s), one evaluation on N rows; U_s = [T(ω(t_s)) − θ̂]/t_s − r (the stored centring residual, all q columns); the larger child by mass balance U_l = (p_k U_k − p_s U_s)/p_l. Realized gain g = (p_s U_s,c² + p_l U_l,c² − p_k U_k,c²)/N; V_btw += g. Both children inherit the parent's Δ²T. γ for both children = clip(g/ĝ, 0, 1) when ĝ > 0 and both are finite, else 1.
- **Closing.** If g < τ (the τ at selection) both children are closed; otherwise ρ² is recomputed and both children receive proposals. A failed evaluation cancels the split: the parent stays, closed, its evaluation counted (§5.2).
- Products: L leaves; n_level, n_adjacency; the sequence sums Σg and Σĝ, reported as the gain ratio Σg/Σĝ (NaN if Σĝ = 0).

---

## 4. Outputs

- **V_btw** = (1/N) Σ_k p_k U_k,c² over the L final bins (equal to the running value).
- **V̂_win** = ρ² (1/N) Σ_{k: n_k > 1} p_k γ_k [Var_k(ψ̂₀) + v_k], with Var_k the population variance of ψ̂₀ over the bin and v_k the within-bin posterior variance of the final bin.
- **V̂_tot** = V_btw + V̂_win.
- **B̂** = (1/2N) Σ_k p_k Δ²T_k over the final bins (identical to the initial-bin value, since children inherit Δ²T and their masses sum to the parent's). Reported, not applied.
- **Refined influence estimate** ψ̂(x_i) = U_{k(i),c} + ρ (ψ̃₀(x_i) − ψ̄₀,k(i)).
- **Acceleration** â_BCa = mean(ψ̂³) / (6 √N mean(ψ̂²)^{3/2}).
- **The QIJ interval** at level q: z = Φ⁻¹((1 ± q)/2), adjusted z' = z/(1 − â z), h = θ̂ + z' √max(V̂_tot, 0). Median-bias constant zero. The interval is a pure function of (θ̂, V̂_tot, â_BCa, q) and is recomputed wherever needed.
- **The bootstrap comparator** (`bootstrap.py`): B multinomial weight vectors (N, 1/N) from `default_rng(seed)`, one at a time; replicates T(X, ω_b); the percentile interval from the finite replicates at any level and any prefix b ≤ B; a replicate with any NaN counts as failed.
- Nothing else is an output. In particular no second-order interval, no sampler, no interval computed from resampled quantities of any kind exists in the package.

---

## 5. Failure and degenerate-case rules

**5.1 Stage 1, a failed prototype evaluation** (NaN in T at prototype j) is a missing response for the outputs that are NaN: left out of the mass-centring and out of the influence model's design for those outputs; the model still predicts at every data point. A failed base evaluation θ_Q is a collapsed stage 1 (§5.3).

**5.2 Stage 2.** A NaN in any initial bin's stencil: that output's measurement stops at that bin, nothing further is measured, and the output fails; the failure voids the whole draw (§5.4). A NaN in a refinement evaluation: the split is cancelled, the parent bin stays and is closed, the evaluation and the failure are counted, the refinement continues.

**5.3 Collapsed stage 1 (revision 8).** An output on the constant path (§2.5) or with M_𝓘,c ≤ 1 is a **failed** output: V_btw, V̂_win, V̂_tot, B̂, â_BCa NaN, refined influence NaN, L = 1 with all points in one bin, no refinement. A zero-variance result is never produced.

**5.4 The draw.** If any output failed under §5.2 or §5.3, every output's five variance quantities are set to NaN; the diagnostics that genuinely happened (counts, wall times, L, labels) stand. The draw counts as a failed QIJ draw for every output.

**5.5 Bounded estimators (revision 8).** Any estimator with a box or bracket returns NaN when a fitted parameter rests within 10⁻⁴ of the box width of either edge, in the coordinates the search walks in (log ν for the multivariate-t degrees of freedom on its bracket [0.1, 10⁶]; the IMF's three bounded parameters directly). The box is never narrowed. This holds for every caller, so a bootstrap replicate and a QIJ evaluation fail under the same test.

**5.6 Nothing is retried, cached, warm-started or given a second start.** Every evaluation depends on its own (X, ω) only.

---

## 6. Products (the study, `study.py`; per dataset, per estimator, one directory)

Draw s uses seed = master_seed + s for the data, the 𝒳-VQ and the bootstrap generator. All per-output columns are named `<field>_<output>`; a q-output estimator is one row per draw.

- **`truth.parquet`**, one row per draw: `s`, `seed`, `theta_true_<c>`, `theta_hat_<c>`, `V_oracle_<c>` where an analytic influence exists (V_oracle = mean(ψ²)/N over the draw).
- **`boot.h5`**: `theta` (S, B, q), `n_failed` (S,), `seed` (S,), `wall_time` (S,), `s` (S,).
- **`qij.parquet`**, one row per draw: `s`, `M_X`, `n_failed` (failed evaluations), `evals_*` / `rows_*` / `wall_time_*` for `prototype`, `full_data`, `refinement`, `total`, `normalized_rows` = rows_total/N; per output `V_btw`, `V_win_hat`, `V_tot_hat`, `B_hat`, `a_bca`, `L`, `n_level_splits`, `n_adjacency_splits`, `rho`, `gain_ratio`, `n_refine_evals`, `ell`, `lam`, `ell_bound`, `lam_bound`, and the final bins' constituents `bin_mass`, `bin_influence`, `bin_d2T` (length-L lists). No interval is stored.
- **`qij_prototypes.parquet`**, the designated draw: `s`, `j`, `p`, `w_0..w_{d−1}` (T's coordinates), `I_<c>` (mass-centred).
- **`qij_points.parquet`**, the designated draw: `s`, `i`, `bmu`, per output `psi0`, `psi` (the oracle influence, where it exists), `sigma`, `psi_hat`, `bin_label` (the final bin).
- **`qij_partition.parquet`**, where an oracle exists: `s`, `M`, and V_btw/V_tot for the 𝒳-VQ receptive fields, the 𝓘-VQ bins from ψ̂₀, and the bins from the true ψ at M ∈ {4, 6, 8, 12, 16, 24, 32, 48, 64}.
- **Tables** (`tables.py`): per (dataset, estimator, output) the medians and 5–95% of V_btw/V_tot, V̂_tot/V_tot (V_tot the Monte-Carlo variance of θ̂ over the truth product's finite draws) and of the same against V_oracle and V_boot; coverage at 0.95 of h^{QIJ} and h^{boot} with standard errors; the width ratio; L; evaluations; normalized rows; the within-draw QIJ/bootstrap wall-time ratio; the bootstrap's replicate-failure fraction; and (revision 8) two failure counts per output — draws whose full-data fit failed the box rule, and draws whose QIJ result is NaN for any reason — with n_coverage = n_draws − (the union of the two) reconciled per row. A draw excluded from coverage always appears in one of the two counts.

---

## 7. Invariants a reviewer can check on the products, without running anything

For every draw with a finite result (exact arithmetic identities hold to 10⁻¹² relative unless stated):

1. Σ_k bin_mass = 1; len(bin_mass) = L; bin_mass·N are integers.
2. Σ_k bin_mass_k · bin_influence_k = 0 (the centring survives every split by mass balance).
3. (1/N) Σ_k bin_mass_k · bin_influence_k² = V_btw.
4. (1/2N) Σ_k bin_mass_k · bin_d2T_k = B_hat.
5. V_tot_hat = V_btw + V_win_hat, V_win_hat ≥ 0, V_btw ≥ 0.
6. n_level_splits + n_adjacency_splits = L − M_𝓘,c ≤ n_refine_evals ≤ 1 + M_X (M_𝓘,c ≤ 17 at the default; cancelled splits spend an evaluation without adding a bin).
7. evals_prototype = 1 + M_X and rows_prototype = (1 + M_X)·M_X; evals_full_data = 1 + 2 Σ_c M_𝓘,c; evals_refinement = Σ_c n_refine_evals; rows_full_data = N·evals_full_data; rows_refinement = N·evals_refinement; normalized_rows = rows_total/N.
8. A draw with NaN in any of V_btw, V_win_hat, V_tot_hat, B_hat, a_bca for one output has NaN in all five for every output. No row has L = 1 together with V_btw = 0 (a collapsed stage 1 is NaN, revision 8).
9. lam ≥ 10⁻¹⁰; for an estimator with η at machine precision, lam at 10⁻¹⁰ is the search floor and lam_bound is true whenever lam is within 1% in log of it.
10. On the designated draw: Σ_j p_j = 1 and Σ_j p_j I_j = 0 over finite j (prototypes); the mean of psi_hat over each final bin equals bin_influence_k; psi_hat − bin_influence_{bin_label} = ρ·(psi0 − mean(psi0) − ψ̄₀,k) pointwise; ρ² = V_btw / ((1/N) Σ_k p_k ψ̄₀,k²) with ψ̄₀,k the bin means of centred psi0.
11. The interval at 0.95 recomputed from (theta_hat, V_tot_hat, a_bca) by §4 reproduces the table's coverage exactly.
12. The one check in the package (`python -m qij.check`): for the weighted mean, whose influence is x_i − θ̂ exactly, on the pipeline's own final bins, Σ_k p_k ψ̄_k² + Σ_k p_k Var_k(ψ) = (1/N) Σ_i ψ_i² to machine precision. This is the only verification code in the package.

Statistical expectations on the study (not identities; for the reviewer's orientation only): on estimators with a smooth influence V_btw/V_oracle is 0.98–1.00 and the coverage of h^{QIJ} at 0.95 is within two standard errors of nominal; on a needle-like influence (a small tail probability in many dimensions) V_btw sits below the oracle and the interval under-covers, which the paper reports as the method's limit.

---

## 8. Code map

| step | module / function |
|---|---|
| §1 contract, wrapping, stage order, whole-draw failure | `qij.py: QIJ.fit`, `_wrap`; `result.py: QIJResult` |
| counter | `core/counter.py: Counter` |
| §2.1 prototype count | `core/xvq.py: cost_rule_M` |
| §2.2 quantizer, BMUs, CADJ | `core/xvq.py: fit_xvq`, `_resolve_bmu2` |
| §2.3 prototype influences, mass-centring | `core/xvq.py: prototype_influences`, `run_xvq` |
| §2.4 weights and steps | `core/differences.py: forward_step`, `central_step`, `step_parameter`, `perturbed_weights`, `difference` |
| §2.5 influence model, floor, predictions, σ, v_k, constant path | `core/influence_model.py: fit_influence_model`, `_lambda_floor`, `psi0`, `uncertainty`, `bin_posterior_variance` |
| §3.2 initial bins | `core/ivq.py: build_bins`, `kmeans_1d` |
| §3.3–3.4 stencil, centring, V_btw, B̂ | `core/ivq.py: bin_differences`, `between_terms` |
| §3.5–3.6 ρ, proposals, queue, measurement, γ, closing; §4 V̂_win, ψ̂, constituents | `core/refine.py: run_refinement`, `_split_gamma`, `_try_level_split`, `_try_adjacency_split` |
| §4 acceleration | `core/outputs.py: acceleration` |
| §4 intervals | `core/intervals.py: qij_interval`, `percentile_interval` |
| §4 bootstrap | `bootstrap.py: Bootstrap` |
| §5.5 box rule | `estimators.py: _at_bound`, `IMF._at_box`, `mvt_nu` |
| §6 products | `study.py`; downstream `tables.py`, `figures.py` |
| §7.12 the one check | `check.py` |

---

## 9. What the code must not contain

Any of these is a finding, whatever it is named: a second verification routine, test or assertion; a Monte-Carlo or resampling step inside the method (the bootstrap comparator excepted); an interval other than the two in §4; a retry, cache, warm start or second start in an estimator; an option, flag, branch or fallback not named in this document; a second code path for a case the first covers; a Python loop over the N data points (loops run over prototypes, bins and draws only; hot paths state their cost in the docstring); a quantity computed in two places; a name outside the glossary.
