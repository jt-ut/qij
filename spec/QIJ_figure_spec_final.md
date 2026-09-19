*Live copy in the `qij` package since 19 September 2026; the `vqboot/spec/` copy is frozen.*

# Final figure specification for the paper (19 September 2026) — one coordinate set, one grid

Supersedes `QIJ_figure_spec_rev51.md`. Every figure shows the same six coordinates in the same positions, a 2 × 3 grid, rows by dataset:

| position | coordinate | axis for the influence figure |
|---|---|---|
| (1,1) | MVT ν | radius ‖x‖ in whitened coordinates |
| (1,2) | MVT P_tail | radius ‖x‖ in whitened coordinates |
| (1,3) | FP a | log σ (data coordinate 0) |
| (2,1) | FP scatter | log σ (data coordinate 0) |
| (2,2) | IMF slope | log stellar mass |
| (2,3) | IMF p | log stellar mass |

Pareto (both), FP b and c, IMF M*, γ shape and γ scale appear in the table only. Labels per `QIJ_glossary.md`; style per `QIJ_figure_style.md`; LNCS text width. The interval plotted is a parameter of the figure code, `h_qij` by default and `h_qij2` if the §34 decision rule selects the second-order interval; the caption names which. Build and test against a smoke run of the current build; switch the run path when the main run lands.

## Figure A — accuracy (replaces F2)
Three panels side by side, six rows each in the grid order above, reading down.
- (a) Variance against the truth: log(V̂_tot / V_MC) for QIJ and log(V_boot / V_MC) for the bootstrap, V_MC the Monte-Carlo variance of θ̂ over the draws in the truth product (box-rule draws excluded, per §4 of the package plan); two markers per row with 95% whiskers over draws; shaded ±0.05 band. Note for the IMF p row: V_MC is inflated by interior fits far out on the ridge; the caption says the row's coverage and width are the informative numbers.
- (b) Coverage at 0.95 of the QIJ interval and the bootstrap percentile interval, with Monte-Carlo standard-error whiskers, nominal line.
- (c) Width ratio QIJ / bootstrap at 0.95, median with 5–95% whiskers, shaded ±10% band, unity line.

## Figure B — the refined influence against the truth (replaces F3 and F9)
2 × 3 panels. In each: one point per prototype of the 𝒳-VQ of the designated draw; x = the mean over the prototype's receptive field of the panel's data axis; y = influence; two series: the true influence averaged over the receptive field (hollow marker) and the refined influence estimate ψ̂ averaged over the same receptive field (filled marker), from `qij_points` (`psi`, `psi_hat`, nearest-prototype index). Marker size proportional to receptive-field mass, optional. Printed in the corner: the rank correlation between the two series and ρ. y axes on each panel's own scale. The caption says what each panel shows: the smoothed step on P_tail, the scale on ν, the curvature on the scatter, the tail on the IMF.

## Figure C — cost (replaces F7)
2 × 3 panels. In each: the bootstrap's relative width error of its 95% interval after the first b replicates, against b on a log axis, mean over draws with the interquartile band, from the stored replicates (prefix quantiles); the QIJ marker at the coordinate's median normalized rows with its interquartile whiskers, its width error against the converged bootstrap; in the corner, the two wall times per run from the timing run (one worker, one thread), "QIJ x s / bootstrap y s". The Fundamental Plane's timing is one run for four outputs; both FP panels print the same pair and the caption says so.

## Table T1
All thirteen coordinates, rows in dataset order, columns: V_btw/V_tot, V̂_tot/V_tot, V̂_tot/V_boot, coverage of h^{QIJ} (and h^{QIJ2}) and of h^{boot} at 0.95 with SE, width ratio, L, evaluations, normalized rows, QIJ / bootstrap wall-time ratio (within-draw median); medians with 5–95% where a ratio. Draws excluded by the box rule are counted in a final column for the IMF rows.

## Figure D — cost against sample size (F10, unchanged in design)
(a) Fundamental Plane: normalized rows and wall time per run against N for QIJ and the bootstrap; (b) coverage of both intervals against N, four thin lines for the four outputs with per-output Monte-Carlo bands, and the width ratio. If the IMF cost sweep is rerun with the final build, a second pair of panels for the IMF: coverage of both intervals against N (slope, M*, p) and the wall-time ratio against N, the sweet-spot panel; plus the fraction of bootstrap replicates reaching the constraint against N with QIJ's zero.
