# Revision plan for the QIJ paper (WSOM+ 2026)

**Written 16 September 2026; superseded in part on 17 September.** The §2 and §3 drafts in `qij_wsom_rev.tex` describe revision 3.2 (a calibration stage, a scale factor, two intervals) and will be rewritten, not patched, to the method as built: revision 5.1, described in `spec/QIJ_method_outline.md` with the terms of `spec/QIJ_glossary.md`. The paper reports one interval, h^{QIJ}_q; the between-bin term appears as a variance share, not as a second interval. The results section is written from the revision-5.1 tables and the reformatted figures once the terminology pass has regenerated them under the new labels. Page count: 13 with markup, 12 without, with the old §4 still in place. The new §4 replaces about 2.5 pages of old text with three figures and one table, so the plain count will land at 12–13 and cuts are needed (see Page budget). The plan below says what each section becomes, which numbers are settled and which wait on the rerun, and the writing rules for the pass.

Sources: `spec/QIJ_revision_plan.md` (revision 3.4; §0 framing, §2 disclosures, §3 method, §8 manuscript list, §15 wave-1 results, §17 production results), `spec/QIJ_writeup_notes.md` (glossary; the first-stage count derivation; Q&A points), `runs_full/aligned/gates/*.md` (numbers).

## Files in this folder

| file | role |
|---|---|
| `qij_wsom_rev.tex` | the revision; edit this one |
| `qij_wsom_original.tex` | the submitted text, untouched, for diffing |
| `llncs.cls`, `splncs04.bst`, `refs.bib` | class, bibliography style, references (copied; `refs.bib` will gain entries, see below) |
| `f2.pdf`, `f3.pdf`, `f7.pdf`, `f9.pdf` | the submitted figures; F2, F3, F7 are replaced by the study's new versions, F9 is kept |

Build: `pdflatex qij_wsom_rev && bibtex qij_wsom_rev && pdflatex qij_wsom_rev && pdflatex qij_wsom_rev` (or `latexmk -pdf qij_wsom_rev`).

## The `\rev{}` command

Defined in the preamble after `booktabs`: `\rev{text}` prints bold blue while `\revtrue` (the default). Switch to `\revfalse` for the final build and the markup prints plainly; no other change is needed. Use `\rev{}` around every sentence or passage that is new or rewritten, at sentence granularity, so unchanged sentences stay black and the reviewer's reading load is visible. Math inside `\rev{}` works (`\bfseries` does not affect math; blue does). For a whole rewritten paragraph, wrap the paragraph. Do not wrap section headings; headings that change are noted in the margin comment instead.

## What the revision answers

The reviewer's question: how does QIJ work when the influence function has no closed form? The submitted paper's results came from analytic influence functions (its "Level 1" path), and its derivative-free route measured only the between term, which on the multivariate t and Fundamental Plane held 20–46% of the variance. The revision presents a derivative-free method that measures 98–99% of the variance directly, at 6–27 times less computation than the bootstrap, and uses the analytic influence only as an oracle to validate it.

## The method, as the paper will describe it (from plan §0, §3; glossary in the writeup notes)

Two quantizers, two declared inputs (η, the estimator's relative accuracy; ε_rf, the target share of influence variance left inside the second-stage receptive fields, default 0.01).

1. **Prototype stage.** A vector quantizer on the data with M_pilot prototypes (centroids). The estimator is evaluated on the weighted prototypes and differenced along each prototype's mass: the *coreset influences*, contamination derivatives at the quantized distribution. A smooth curve, kernel regression with an affine mean and prototype centers, width and strength by maximum likelihood, gives the *learned influence* at every data point. Cost: 1 + 2M_pilot evaluations on M_pilot rows.
2. **Full-data stage.** For each estimand coordinate, a one-dimensional quantizer of the learned influence values with M prototypes from M = ⌈√(κ/ε_rf)⌉ (Panter–Dite); each prototype's receptive field is an *influence cell*. The estimator is differenced along each cell's mass on the full data: 1 + 2M evaluations on N rows, one base shared across coordinates. This gives the measured between term S_btw, the second-order bias term, and the receptive-field averages that calibrate the curve.
3. **Report in tiers.** Measured (S_btw, the lower bound, and the bias term); predicted (the curve's total after one scale factor from the measurements); model-based (the calibrated field's remainder, covariance and skewness); the interval.
4. **The first-stage count** by the cost balance M_pilot = ⌈√((1 + 2qM_ref)N/2)⌉, M_ref = ⌈√(2.7/ε_rf)⌉: the survey gets the same evaluation budget as the measurement. Derivation and the two limits to state are in the writeup notes ("The first-stage count").

Terminology in the paper: prototype, Voronoi cell, receptive field, mass; first-stage and second-stage quantizer; coreset influence; learned influence (influence curve); influence cell; measured receptive-field average; estimator evaluation. Not: pilot (except as the symbol M_pilot if kept; consider renaming to M_1 in the paper), group, block, band, component, refit, fit (for an evaluation).

## Section-by-section plan

**Title.** Keep. Running title keep.

**Abstract.** Rewrite in full. Drop "no resampling anywhere" as a slogan (still true, but no longer the point) and "scales with M rather than B". New content: the bootstrap's cost; the two-stage construction in two sentences; what is measured versus modelled; the numbers (98–99% of the variance measured directly on six smooth estimands; interval coverage equal to the bootstrap's; 6–27× fewer normalized evaluations); the three datasets. Pending: the exact width figures and whether the calibrated total appears in the abstract (after the rerun).

**§1 Introduction.** Keep paragraphs 1–2 (VQ background; the bootstrap; why prototypes cannot replace resampling) with light edits. Rewrite paragraph 3 (the "two parts" of the variance) to state the revision's central observation: averages over receptive fields do not determine spreads inside them, so the partition must be aligned with the influence, and a cheap survey on the quantizer can learn that alignment. Rewrite paragraph 4 (the decomposition as a correctness check) since F3 changes meaning. Related work: keep the jackknife/IJ/BLB paragraph; add coresets (Feldman & Langberg 2011; Bachem, Lucic & Krause 2017), influence functions in ML (Koh & Liang 2017; Giordano et al. 2019), Efron's empirical influence, Zador / Graf & Luschgy, Panter & Dite 1951, Rasmussen & Williams 2006, DiCiccio & Efron 1996 (ABC). Add these entries to `refs.bib` (verify each DOI before adding; do not invent volume or page numbers).

**§2 Method.** Restructure:
- 2.1 Setup: keep eq. (if), eq. (Uj) and the multinomial argument; keep eq. (Vcell) and the identity (eq. identity), fixing the scale so both are on one scale (plan §1: S-scale, Var = S/N). Cut the "computable analytically ... or by a delete-one-RF secant" paragraph and its commented-out predecessors; replace with one paragraph: the between term is measurable by two evaluations per receptive field around a shared base, with the difference rule and step from η (plan §3.6, §2), and the within term is not measurable from averages.
- 2.2 (new) Influence-aligned receptive fields: why data-space receptive fields leave 54–80% of the variance inside them on smooth multivariate estimands, and why a one-dimensional quantizer of the influence itself makes the between term nearly the whole variance (M^{-2} rate for the oracle; Panter–Dite count). One figure candidate: F3 replacement.
- 2.3 (new) Learning the influence on a coreset: the prototype stage; coreset influences are contamination derivatives at the quantized distribution, not the full-data influence; ordering and level structure are what the second stage uses, values enter only through the count and the predicted tier after a scale factor; the affine-mean kernel regression; the first-stage count and its derivation in short form with the two limits.
- 2.4 (new) Calibration and the three tiers: the scale factor; the constrained field (state the rank-robust solve and the calibration noise in one sentence each); what is measured, what is predicted, what is model-based; the bias term as a between-receptive-field curvature term, not the full second-order bias; the interval (acceleration-corrected, z0 = 0, centred at the estimate).
- 2.5 Scope and disclosure (was "Estimator Limitations"): plan §2 (i)–(iv): weighted evaluation required; non-differentiable-in-weights estimators out of scope; discontinuous influence (tail probabilities) in scope for measurement with the loss reported; the prototype-stage value is not an estimate of the target; η is the user's declaration.
- Fig. F9 (RF influences on a shared partition) keep, recaption in the new vocabulary; possibly show coreset influences at the prototypes instead of analytic ones (the plan lists F9 as "free from the prototype stage").

**§3 Experimental data.** Keep the three datasets. Update the protocol paragraph: S = 1000 draws, N = 1000, bootstrap B = 2000 paired by seed; QIJ with the declared inputs (η per estimator: machine precision for Pareto and tail probabilities, 1e-12 for FP and MVT ν; ε_rf = 0.01); first-stage counts 133 (one-output estimators) and 262 (FP); second-stage counts 14–19 as found. Remove the M grids (they belonged to the old design). State which estimands: eight coordinates; the Pareto extreme quantile excluded and why.

**§4 Results.** Rebuild around the plan's T1/F2/F3/F7:
- 4.1 (was "Examining the variance decomposition"): replaced by "How much of the variance the receptive fields hold": F3 replacement, R² against M for data-space receptive fields (Zador slope −2/d), learned-influence receptive fields, and the oracle one-dimensional quantizer (slope −2). The wave-1 numbers: learned receptive fields 97.5–99% at the rule's counts; ideal 98.1–99.0%. Text notes the loss falls as 1/M² (FP) or 1/M (Pareto) with the first-stage count, and that dimension does not predict it.
- 4.2 Validation against the bootstrap: F2 replacement (log(S_tot_hat/S_boot) with S_btw as hollow markers; coverage; width ratio; endpoints). Settled numbers: measured-interval coverage 0.937–0.950 against bootstrap 0.931–0.957 (SE 0.007) on six smooth coordinates; S_btw/S_boot 0.94–0.99; S_btw/S_tot 0.980–0.989. Pending after the rerun: the calibrated total and its coverage and width ratio per coordinate; whether it is shown for all coordinates or FP only.
- 4.3 Bias and skewness (new short subsection): bias term matches the exact Pareto bias (0.00196 vs 0.00200) and the MVT Monte-Carlo bias; centring on it changes coverage by ≤ 0.001, so the interval stays centred at the estimate; acceleration matches the oracle and improves the tail probabilities' upper miss (Pareto 0.061 → 0.036, MVT 0.092 → 0.068).
- 4.4 The computation–accuracy frontier: F7 replacement; the method at its normalized rows with the cost model and its two error directions in the caption; wall time inset. Settled: Pareto 267 prototype evaluations (35.5 N-rows) + 31 full-data; FP and MVT from v8.md after the accessor fix. The old text's "M+1 = 51 evaluations" is gone.
- Tail probabilities: reported in one paragraph, not gated: 88% and 56% of the variance measured; coverage 0.936 and 0.898 against the bootstrap's 0.964; the loss is the discontinuity the curve smooths, disclosed in §2.5.
- T1 (table): per coordinate S_btw/S_tot, S_tot_hat/S_tot, S_tot_hat/S_boot, coverage, width ratio, counts, rows, wall time. Fits in LNCS width with `booktabs`.

**§5 Conclusions.** Rewrite. The three-sentence summary of what changed: alignment, measurement, cost; what is measured versus modelled; the limits (a priori count rule cannot guarantee accuracy on an unseen estimator; tail probabilities; iid). Future work: sharing quantizers across coordinates; larger N; the calibrated field where the curve and the measurements disagree (if the rerun leaves it failing on Pareto and MVT, say so here and in 4.2).

**E5.** The Level-1 (analytic) rerun with z0 = 0 is a reference only; it appears in F2 as the oracle if at all.

## Numbers settled now versus pending

Settled (wave 1 at 200 draws, production at 1000 draws): everything in the measured tier, the bias and acceleration findings, the tail probabilities, the coverage of the measured interval, the first-stage count rule and its grid evidence, the Pareto cost.

Pending the rerun (calibration with λ_cal): S_tot_hat/S_tot, S_win_hat/S_win, coverage and width ratio of the calibrated interval, agreement (b), per coordinate; the FP and MVT cost rows (accessor fix); the headline sentence (measured bound alone, or measured bound plus calibrated total). The figures F2, F3, F7 and table T1 come from the coordinator's round 3.

## Writing rules for the pass

From the authors:
1. Plain, simple, human-readable tone. One idea per sentence. Say what was done and what was found.
2. Think like a scientist: no vague phrases; no claim without a number or a section that supports it; say "we did not test" where true; report the failures (tail probabilities, the calibrated field where it fails) as findings.
3. Pronouns: name the thing. "It" refers to the nearest noun and nothing else; where the antecedent is more than one noun back, repeat the noun. The same for "this" and "these" as standalone subjects: "this construction", not "this".
4. Wrap every new or rewritten sentence in `\rev{}`.

Signs of machine writing to avoid (from the Wikipedia page the authors cited, condensed to what applies to a methods paper):
- Words to leave out: additionally, crucial, delve, emphasize/emphasizing, enhance, highlight/highlighting, intricate, interplay, key (as an adjective), landscape, leverage, pivotal, robust (unless in its statistical sense), showcase, underscore, testament, valuable, vibrant, seamless, holistic, framework (unless it is one), notably, importantly.
- Do not replace "is" with "serves as", "stands as", "represents", "functions as", "marks".
- No "not only X but also Y", no "not X but Y" as a rhetorical frame, no "rather than X, Y" as a sentence opener, no "it is worth noting", no "in this work we" as a refrain.
- No sentence-final participle clauses that add a claimed significance ("..., highlighting the method's flexibility").
- No em dashes as a habit; use a comma, a full stop, or parentheses.
- No lists of three for rhythm. No headings that are titles for nothing. No bold in prose.
- No vague attribution ("it is well known", "practitioners often"). Cite or drop.
- Straight quotes. No summary sentence that restates the paragraph.
- Keep the submitted paper's voice where its sentences survive; the revision should read as one author.

Domain rules:
- Keep the S-scale versus S/N-scale straight; the submitted eq. (Vcell) carries 1/N while eq. (identity) does not (plan §1).
- Say "estimator evaluation", never "fit" or "refit" for a run of the estimator.
- Cost claims in three forms only (evaluation counts by size, normalized rows with the cost model stated, wall time) and the caption names the cost model's two error directions.
- Never say "only the ordering is used" (plan §14 item 1); say what the level structure sets and what the values set.
- The bias term is a between-receptive-field curvature term; the interval is not centred on it.
- The prototype-stage estimate is not an estimate of the target; say so once in §2.5.

## Status of the pass (22:40, 16 Sept)

- Done: §2.1 setup (retained equations, S-scale), §2.2 decomposition with the averages-do-not-determine-spreads argument, §2.3 the difference rule, §2.4 aligned receptive fields with the prototype stage and the count rule (eq. M1), §2.5 calibration and the reported quantities, §2.6 scope; §3 protocol and dataset notes. New macros `\Sbtw \Swin \Stot \Mone \psiQ \psit \psih`. Six references added to `refs.bib` (panter1951, rasmussen2006, feldman2011, koh2017, giordano2019, diciccio1996): pages and volumes from memory, verify before submission; koh2017, giordano2019 and diciccio1996 are not yet cited.
- Waiting: the regenerated T1 (the copy on disk predates the rerun) and figures F2, F3, F7 from the coordinator; the authors' reading of the MVT calibrated-field result (plan §18) for the abstract and §4.2.
- Then: §4 (four subsections per the plan), §1 paragraphs 3–4 and related work, abstract, §5; recaption F9 or drop it for space.

## Page budget

WSOM+ rule (from the website): papers should be within 10 pages including references, and must not exceed 12, in Springer LNCS format. The submitted paper is 10 pages. Target for the revision: 11 pages, hard stop at 12. The extra page pays for the new §2.2–2.4 and §4.3, and for T1. Cuts that keep the count down, in order of preference (the plain count is already 12 before the new §4): the secant paragraph in §2.1 and all commented-out blocks (already marked); the old F3 discussion (replaced by §4.1 at similar length); the introduction's paragraph 3 and 4 rewrite should be no longer than the text it replaces; F2 and F7 keep their current sizes; F9 may drop to 0.85 linewidth. The `\rev{}` markup does not change the page count (bold blue at the same size).

## Order of work when the rerun is in

1. Round-3 outputs (T1, F2, F3, F7) into this folder; recaption F9.
2. §2 and §3 first (settled), then §4 with the tables, then abstract and §5 last.
3. One compile per section; keep the page count in view.
4. A final read with `\revfalse` for flow, then `\revtrue` for the submission with markup if the venue wants it, or `\revfalse` if not.
