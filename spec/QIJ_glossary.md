*Live copy in the `qij` package since 19 September 2026; the `vqboot/spec/` copy is frozen.*

# QIJ glossary and notation — the single source for the plan, the code, the figures and the paper

**17 September 2026.** Fixed by the authors after the revision-5.1 results. Every document, identifier, record key, table header and figure label uses these words and symbols. Where a symbol is given, prose uses the name and equations the symbol. Subscripts: i for data points, j for 𝒳-VQ prototypes, k for 𝓘-VQ bins.

## Terms

| term | symbol | meaning |
|---|---|---|
| data space | 𝒳 | the space the data live in, in the quantizer's coordinates (whitened) |
| influence space | 𝓘 | the axis of influence values |
| 𝒳-VQ | | the vector quantizer of the data (k-means, centroids); stage 1 |
| 𝓘-VQ | | the one-dimensional vector quantizer of the initial influence estimate; stage 2, on which the finite differences are taken |
| prototype | w_j | a prototype of the 𝒳-VQ |
| receptive field | RF_j | the data points nearest prototype j; its mass p_j = n_j/N |
| connectivity graph | CADJ, CONN | the 𝒳-VQ's adjacency from second-nearest prototypes (directed, symmetric) |
| bin | | a cell of the 𝓘-VQ: the data points whose initial influence estimate lies in one interval; mass p_k = n_k/N |
| prototype influence | I_j | the finite difference of T along prototype j's mass, T evaluated on the prototypes as a weighted dataset; mass-centred |
| initial influence estimate | ψ̂₀(x) | the Gaussian-process model through the I_j, evaluated at any point |
| its uncertainty | σ_i | the posterior standard deviation of ψ̂₀ at x_i |
| bin influence | U_k | the finite difference of T along bin k's mass on the full data: the average influence over the bin |
| second difference | Δ²T_k | the second finite difference of T along bin k's mass, divided by δ² |
| refined influence estimate | ψ̂(x_i) | U_{k(i)} + ρ(ψ̂₀(x_i) − ψ̄₀,k): the per-point estimate whose third moment gives the acceleration |
| FD-to-prediction scale | ρ | √(V_btw / Σ_k p_k ψ̄₀,k²): the ratio of finite-differenced to predicted between-bin spread |
| between-bin term | V_btw | (1/N) Σ_k p_k U_k²: the part of the variance the bin influences account for; finite-differenced; a lower bound on V_tot |
| within-bin term | V_win | (1/N) Σ_k p_k Var_k(ψ): the part inside the bins; not observable from bin influences |
| predicted within-bin term | V̂_win | ρ² (1/N) Σ_k p_k γ_k [Var_k(ψ̂₀) + v_k] over bins with more than one point; v_k the within-bin posterior variance of ψ̂₀ (mean of the bin's posterior covariance diagonal minus its grand mean); γ_k = min(1, g/ĝ) of the split that created the bin, 1 if never split (scales the whole bracket, revision 6) |
| variance estimate | V̂_tot | V_btw + V̂_win |
| bias term | B̂ | (1/2N) Σ_k p_k Δ²T_k over the initial bins; reported, not applied |
| acceleration | â_BCa | the BCa acceleration constant from the third moment of ψ̂ |
| QIJ interval | h^{QIJ}_q | the q·100% BCa-form interval with variance V̂_tot, acceleration â_BCa, z₀ = 0, centred at T̂ |
| bootstrap interval | h^{boot}_q | the percentile interval from B replicates |
| between-bin interval | h^{btw}_q | study only: the normal interval from V_btw alone |
| declared accuracy | η | the relative accuracy of one estimator evaluation, declared by the user |
| tolerance | ε | the share of the variance the user accepts leaving inside the bins (default 0.01) |
| difference step | δ | the relative change of the mass being moved: (3η)^{1/3} for the three-point stencil, δ_f = 2√η for a single forward step; for a bin or prototype of mass p the weight parameter is t = δ·p/(1 − p), so its mass becomes p(1 + δ) |
| prototype count | M_𝒳 | ⌈√((1 + 2q M_ref) N / 2)⌉, floor 20, cap N/2 |
| initial bin count | M_𝓘 | ⌈√(2.7/ε)⌉ = 17 at the default |
| reference bin count | M_ref | the same number, as it enters the prototype-count rule |
| final bin count | L | the number of bins after refinement |
| expected gain | ĝ_k | the predicted increase in V_btw from splitting bin k |
| gain | g_k | the finite-differenced increase in V_btw the split produced |
| gain-driven refinement | | the loop that splits the bin with the largest expected gain and stops when gains fall below the share τ = ε V_btw / L |
| level split | | a split of a bin by ψ̂₀ (two-means on the bin's values) |
| adjacency split | | a split of a bin by data-space adjacency: points whose second-nearest prototype has a higher prototype influence than their nearest, against the rest |
| estimator evaluation | T(X, ω) | one run of the estimator on the data with weights ω ≥ 0, Σω = N |
| normalized rows | | Σ over evaluations of (rows in the evaluation)/N: full-data-evaluation equivalents |

## Words that are retired

survey, coreset, pilot (the estimate is "initial", the count is M_𝒳), prototype stage, full-data stage, measurement stage, calibrated / calibration, measured (except as "finite-differenced" where the contrast with "predicted" is needed), model-based (say "predicted"), accel / accelerated (say "the QIJ interval"; the acceleration is named once where BCa is introduced), headline, curve / learned influence / shape / learned curve (say "initial influence estimate"), influence cell / cell / group / leaf (say "bin"; "receptive field" for the 𝒳-VQ), uncertainty split / sector split (say "adjacency split"), e_shape, κ, M_pilot, M_1, M_c, S_btw / S_win / S_tot (the variance-scale V is used throughout; Var = V, no /N step), a_c / a² (say ρ), c_k (say Δ²T_k), bias_btw (say B̂), h as a step.

## Find-and-replace for the code (identifiers, record keys, table headers, figure labels)

| old | new |
|---|---|
| `prototype_stage`, `PrototypeStageResult` | `x_vq`, `XVQResult` |
| `M_pilot`, `M_pilot_requested`, `M_used` | `M_X`, `M_X_requested`, `M_X_used` |
| `psi_Q`, `coreset_influence` | `I_proto` |
| `shape`, `ShapeModel`, `shape_ml` | `influence_model`, `InfluenceModel`, `influence_model_ml` |
| `predict` (values), `u` / `posterior_sd` | `psi0`, `sigma` |
| `cells`, `CellSet`, `cell_weights`, `second_stage_quantizers` | `bins`, `BinSet`, `bin_weights`, `i_vq` |
| `M_c`, `M_target`, `M_ref` | `M_I`, `M_I_initial`, `M_ref` |
| `full_data_stage`, `FullDataResult`, `full_data_differences` | `bin_differences`, `BinDifferenceResult`, `bin_differences` |
| `U`, `c` (second difference), `r_c` | `U`, `d2T`, `centering_residual` |
| `refine`, `n_leaves`, `n_level_splits`, `n_sector_splits`, `n_refine_evals` | `refine`, `L`, `n_level_splits`, `n_adjacency_splits`, `n_refine_evals` |
| `a2`, `scale` | `rho2`, `rho` |
| `gain_ratio` | `gain_ratio` |
| `S_btw`, `S_win_hat`, `S_tot_hat`, `S_survey`, `S_survey_scaled` | `V_btw`, `V_win_hat`, `V_tot_hat` (drop the survey fields) |
| `bias_btw` | `B_hat` |
| `a_hat` | `a_bca` |
| `measured` (interval) | `h_btw` (study tables only) |
| `accel` (interval) | `h_qij` |
| `boot_pct` | `h_boot` |
| `eta`, `eps_rf` | `eta`, `eps` |
| `h`, `h_f` (steps) | `delta`, `delta_f` |
| `run_wave1`, `Wave1Result` | `run_qij`, `QIJResult` |
| `agreement_a`, `survey_agreement`, `beyond_hull_share`, `cond_A`, `e_shape`, `kappa` | removed |
| figure/table labels "measured (headline)", "calibrated (accel)", "bootstrap pct." | "QIJ", "bootstrap" (and "between-bin only" where h_btw is shown in a study table) |

Docstrings and comments follow the term table. Module `differences.py` keeps its name. The one check is unchanged.
