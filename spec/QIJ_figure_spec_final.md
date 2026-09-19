*Live copy in the `qij` package since 19 September 2026; the `vqboot/spec/` copy is frozen.*

# Final figure specification for the paper (19 September 2026) — one coordinate set, one grid

Supersedes `QIJ_figure_spec_rev51.md`. Every figure shows the same six coordinates in the same positions, a 2 × 3 grid, rows by dataset:

| position | coordinate | axis for the influence figure |
|---|---|---|
| (1,1) | MVT ν | radius ‖x‖ in whitened coordinates |
| (1,2) | MVT S_99 | radius ‖x‖ in whitened coordinates |
| (1,3) | FP a | log σ (data coordinate 0) |
| (2,1) | FP s | log σ (data coordinate 0) |
| (2,2) | IMF α | log stellar mass |
| (2,3) | IMF p | log stellar mass |

Pareto (both), FP b and c, IMF M*, γ shape and γ scale appear in the table only.

**Label conventions (author, 19 September 2026):** the bootstrap is "Boot" in every figure legend and axis label (the paper defines the abbreviation in §3); both tail probabilities are $S_{99}$, the survival function at the population's 0.99 quantile; the Fundamental Plane scatter is $s$ (not σ, which is the velocity dispersion); the IMF slope is α and its sharpness p. Table T1 uses the same symbols in its row names. Labels per `QIJ_glossary.md`; style per `QIJ_figure_style.md`.

**Row labels are symbols, not words** (19 September): `MVT ν`, `MVT S_99`, `FP a`, `FP s`, `IMF α`, `IMF p`.

S_99 is the SURVIVAL function at the 99th percentile — the probability of exceeding a threshold fixed at the true 99th percentile of the reference distribution, so its true value is 0.01. It is not P_tail (vague), not P_99 (reads as a probability indexed by a percentile), and emphatically not Q_99, which would name the threshold itself, a radius of about 5.6, rather than the probability of exceeding it. The estimator is the weighted fraction of points above that threshold, which is the empirical survival function there. The same construction and the same notation apply to Pareto's tail row in T1. s is the Fundamental Plane's scatter — the root of the smallest eigenvalue of the weighted covariance, whose eigenvector gives (a, b, c), i.e. the RMS distance of the galaxies from the fitted plane, measured perpendicular to it. It is not written σ: a bare σ is the velocity dispersion that is Figure B's own x-axis on those same panels.

**One size, one build.** Every figure is built at the size it is placed at — LNCS text width 4.80in — so `\includegraphics` takes it 1:1 and its type matches the body text. There is no second "draft" rendering: a larger figure scaled down by LaTeX puts its labels at a fraction of their intended size, differently for every figure. Each figure writes a `.pdf` (for the paper) and a `.png` at 300 dpi (for reading and for circulating), the same figure at the same size.

**Fonts** come from one dict, `RC_LNCS` in `figures.py`: titles 9pt (both `axes.titlesize` and `figure.titlesize`), axis labels 8.5, ticks and legend 7.5. They are fixed point sizes, which is only correct because of the 1:1 rule above.

**Quantitative ranges are derived from the data** (`_data_span`), never written into the code: from the extremes actually drawn, widened only far enough to keep a reference line or tolerance band in view. A hardcoded range goes stale the moment the numbers change.

**The second-order interval is gone** (revision plan §36.2(6)); the figures plot `h_qij` and there is no interval parameter. Build and test against a smoke run of the current build; switch the run path when the main run lands.

## Figure A — the intervals (replaces F2)

Title, exactly: **"QIJ vs. Bootstrap: 95% CIs"**. Two panels side by side, six rows each in the grid order above, reading down; two markers per row (QIJ, Boot). Figure size (4.80, 3.20) — taller than the style guide's original 2.20, which was set for one-line labels and no title and left six rows about 1.2in of panel.

- (a) Coverage at 0.95 of the QIJ interval and the bootstrap percentile interval, with Monte-Carlo standard-error whiskers, nominal line. Row labels appear here only; the legend is an inset in this panel.
- (b) Width ratio QIJ / Boot at 0.95, median with 5–95% whiskers, shaded ±10% band, unity line.

**Panel (a), the variance against the truth, was removed on 19 September 2026 by the author.** It compared each method to a THIRD quantity, V̂_MC, where the two surviving panels compare the methods to each other on what a reader acts on; and it was the only panel where anything dramatic happened, while what was dramatic in it — the IMF p row — is a property of that denominator, not of either method. Table T1 keeps the variance ratio for all thirteen coordinates, written V̂/V̂_MC with BOTH terms hatted: both estimate V(θ̂) and differ only in how, the numerator by QIJ or the bootstrap, the denominator by Monte-Carlo sampling. Writing the denominator bare would assert it is the target rather than a second estimate of it.

Each panel's x-label carries its own tag and is bold, the math included; nothing is drawn above the axes, where a tag sat detached from its panel. The two labels are one line each and level with one another: `(a) Empirical Coverage` and `(b) Width: QIJ/Boot`.

## Figure B — the refined influence against the truth (replaces F3 and F9)

**Pairwise form (author, 19 September 2026; replaces the data-space form).** 2 × 3 panels in the grid order above. In each: one point per DATA POINT of the designated draw (not per receptive field: within-field error is what must be visible), x = the refined influence estimate ψ̂(x_i), y = the true influence ψ(x_i), from `qij_points` (`psi_hat`, `psi`); the diagonal y = x as the black dotted reference; small markers with transparency; printed in the corner: the rank correlation r_s between the two. Each panel on its own scale, equal on x and y. Agreement is the diagonal; a wrong scale is a tilt; on S₉₉ the tail points inside mixed bins appear at true value 0.99 with ψ̂ near the bin's mean, and their bin-mates at true value −0.01 pulled to the same ψ̂. No data axis: the encoding does not depend on the data's dimension, which is why the Fundamental Plane panels are readable in this form and not in the data-space one. Labels per the conventions above (S₉₉, s, α, p).

## Figure C — cost (replaces F7)
2 × 3 panels. In each: the bootstrap's relative width error of its 95% interval after the first b replicates, against b on a log axis, mean over draws with the interquartile band, from the stored replicates (prefix quantiles); the QIJ marker at the coordinate's median normalized rows with its interquartile whiskers, its width error against the converged bootstrap; in the corner, the two wall times per run from the timing run (one worker, one thread), "QIJ x s / Boot y s". The Fundamental Plane's timing is one run for four outputs; both FP panels print the same pair and the caption says so.

## Table T1
All thirteen coordinates, rows in dataset order, columns: V_btw/V_tot, V̂_tot/V_tot, V̂_tot/V_boot, coverage of h^{QIJ} and of h^{boot} at 0.95 with SE, width ratio, L, evaluations, normalized rows, QIJ / bootstrap wall-time ratio (within-draw median); medians with 5–95% where a ratio. Draws excluded by the box rule are counted in a final column for the IMF rows.

## Figure D — cost against sample size (F10, unchanged in design)
(a) Fundamental Plane: normalized rows and wall time per run against N for QIJ and the bootstrap; (b) coverage of both intervals against N, four thin lines for the four outputs with per-output Monte-Carlo bands, and the width ratio. If the IMF cost sweep is rerun with the final build, a second pair of panels for the IMF: coverage of both intervals against N (slope, M*, p) and the wall-time ratio against N, the sweet-spot panel; plus the fraction of bootstrap replicates reaching the constraint against N with QIJ's zero.
