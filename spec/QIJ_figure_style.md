*Live copy in the `qij` package since 19 September 2026; the `vqboot/spec/` copy is frozen.*

# QIJ — Figure style guide

Companion to `QIJ_figure_plan.md`. Palette, encoding rules, typography, and a drop-in matplotlib module.

---

## 1. What owns colour

**Method owns colour, globally.** It is the comparison the paper is built on, it appears in most figures, and a reader should be able to glance at any panel and know which line is QIJ without consulting a legend.

Two tiers:

| tier | dimension | rule |
|---|---|---|
| **1 — globally reserved** | `method` | fixed hues, never reused for anything else, anywhere |
| **2 — figure-local** | variance component, DGP, `vq_input` | consistent where they recur, but may reuse tier-2 hues across figures since they never co-occur |

**DGP is encoded by facet, not colour**, wherever possible — F1 (rows), F2 (panels), F3 (panels). The single exception is F4, where all DGPs share axes precisely to show the spread of `d_eff`; there DGP takes tier-2 colour.

**Never let one hue mean two things in the same figure.**

---

## 2. Palette: Okabe–Ito

Colourblind-safe (deuteranopia, protanopia, tritanopia), print-legible, and a recognisably deliberate choice rather than a default.

```
black            #000000
orange           #E69F00
sky blue         #56B4E9
bluish green     #009E73
yellow           #F0E442
blue             #0072B2
vermillion       #D55E00
reddish purple   #CC79A7
```

### Tier 1 — methods (reserved)

**Exactly two methods appear in any published figure.**

| entity | hue | hex | rationale |
|---|---|---|---|
| **QIJ** | blue | `#0072B2` | the protagonist; strongest, most saturated |
| **Standard bootstrap** | vermillion | `#D55E00` | the incumbent; maximum contrast against blue |
| **Truth / reference** | black | `#000000` | dotted lines, reference values, tolerance-band centres |

**Never plotted.** The compressed VQ bootstrap is retained in the codebase for error localization during debugging (spec §8.1), but it has no published role: its `z0` job is void (`z0 = 0` is sufficient) and its non-smooth-fallback job was the quantile, which is degenerate. Likewise the legacy ablations — StratBoot-Q, MN Boot, VQ-RF, VQ-CADJ — require explaining the VQ-bootstrap lineage, which belongs in the setup prose, not a legend.

Assign them **no reserved hue**. If one ever needs to appear in a diagnostic plot for internal use, use `#999999` gray and do not add it to the paper palette.

### Tier 2a — variance components (F3)

| entity | hex | note |
|---|---|---|
| `V_total` | `#000000` | black: the invariant, the line that must be flat |
| `V_cell` | `#009E73` | bluish green |
| `V_within` | `#CC79A7` | reddish purple — chosen over orange to stay clear of vermillion |

### Tier 2b — DGPs (F4 only)

| DGP | hex |
|---|---|
| Pareto | `#E69F00` |
| Fundamental plane | `#009E73` |
| Multivariate t | `#CC79A7` |
| IMF | `#56B4E9` | (sky blue is free — no method reserves it)

### Tier 2c — quantizer input (F8)

| arm | hex | note |
|---|---|---|
| radial (informed) | `#0072B2` | |
| ambient (naive) | `#999999` | gray: the uninformed baseline |

### Continuous scales

`M` and nominal level → **viridis**. Never a rainbow map.

---

## 3. Non-colour encodings — so figures survive grayscale

Every method distinction carries **two** cues. A referee printing in black and white must still be able to read the paper.

| entity | colour | linestyle | marker |
|---|---|---|---|
| QIJ | blue | **solid** | circle `o` |
| Standard bootstrap | vermillion | **dashed** | square `s` |
| Truth / reference | black | **dotted** | — |

**True sampling distribution** (F0, and matching the convention from the original VQ-bootstrap panels): gray fill `#BBBBBB` at `alpha=0.55` with a thin black outline.

**Failure states** get no new hue. Non-covering intervals in F0b keep their method colour but take a heavier linewidth plus a black endpoint marker.

**Semantic bands:**

| band | style |
|---|---|
| materiality (±0.05, F1) | `#DDDDDD` fill, no edge |
| MC-error floor (F5) | gray hatch `///`, `alpha=0.3` |
| MC-error band on coverage (F2) | `#DDDDDD` fill |

---

## 4. Typography

DejaVu Sans throughout — matplotlib's default, so nothing to install and nothing to embed.

| element | size | weight |
|---|---|---|
| figure suptitle | 14 | **bold** |
| axes title | 12 | **bold** |
| axis label | 11 | **bold** |
| legend title | 10 | **bold** |
| legend text | 9.5 | regular |
| tick label | 9.5 | regular |
| annotation | 9 | regular |
| panel label `(a)` | 12 | **bold**, upper-left, offset outside the axes |

Matplotlib has no rcParam for legend *title* weight — set it per legend via the helper in §5.

---

## 5. Drop-in module

```python
# qij/plotting/style.py
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Okabe-Ito ------------------------------------------------------------
OI = dict(black="#000000", orange="#E69F00", skyblue="#56B4E9",
          green="#009E73", yellow="#F0E442", blue="#0072B2",
          vermillion="#D55E00", purple="#CC79A7")

# ---- Tier 1: reserved, never reused --------------------------------------
METHOD = {
    "qij":        dict(color=OI["blue"],       ls="-",   marker="o", label="QIJ"),
    "boot":       dict(color=OI["vermillion"], ls="--",  marker="s", label="Bootstrap"),
    "truth":      dict(color=OI["black"],      ls=":",   marker=None, label="Truth"),
    # Diagnostic-only, never published: compressed VQ bootstrap, legacy ablations.
    # If needed internally, use "#999999" -- do not add to the paper palette.
    "diagnostic": dict(color="#999999",        ls="-.",  marker=None, label=None),
}

# ---- Tier 2 ---------------------------------------------------------------
COMPONENT = {"V_total": OI["black"], "V_cell": OI["green"], "V_within": OI["purple"]}
DGP       = {"pareto": OI["orange"], "fp": OI["green"],
             "mvt": OI["purple"],    "imf": OI["skyblue"]}
VQ_INPUT  = {"radial": OI["blue"], "ambient": "#999999"}

TRUE_DIST = dict(color="#BBBBBB", alpha=0.55, edgecolor="black", linewidth=0.6)
BAND      = dict(materiality="#DDDDDD", mc_floor="#CCCCCC")

RC = {
    "font.family":        "DejaVu Sans",
    "figure.titlesize":   14,
    "figure.titleweight": "bold",
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.labelweight":   "bold",
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    9.5,
    "ytick.labelsize":    9.5,
    "legend.fontsize":    9.5,
    "legend.title_fontsize": 10,
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#CCCCCC",
    "lines.linewidth":    1.8,
    "lines.markersize":   5,
    "figure.dpi":         120,
    "savefig.dpi":        300,
    "savefig.bbox":       None,
    "figure.constrained_layout.use": False,   # see the note below -- load-bearing
}

def use_qij_style():
    mpl.rcParams.update(RC)

def legend(ax, title=None, **kw):
    """Legend with a bold title (no rcParam exists for this)."""
    leg = ax.legend(title=title, **kw)
    if title:
        leg.get_title().set_fontweight("bold")
    return leg

def panel_label(ax, text, dx=-0.12, dy=1.04):
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")
```

---

## 6. Per-figure application

| fig | colour encodes | facet / panel | notes |
|---|---|---|---|
| **F0** | method (2 arms) | draw (2×2) | gray fill = true sampling dist; QIJ dashed *here only*, to sit over the solid bootstrap KDE — state it in the caption |
| **F0b** | method | — | non-covering: heavier line + black endpoint marker, no new hue |
| **F1** | method | rows = `(dgp, estimand)` | `#DDDDDD` materiality band behind everything; zero line black dotted |
| **F2** | method | panel per DGP | `y = x` black dotted; MC band `#DDDDDD` |
| **F3** | **component** | panel per DGP | bootstrap `V` as a vermillion horizontal band; log-log; scientific tick labels |
| **F4** | **DGP** | single panel | linestyle distinguishes `vq_input`; annotate fitted `d_eff` at each line's right end |
| **F5** | method (2 arms) | panel per estimand | truth black dotted; MC floor as gray hatch on the bootstrap violin only |
| **F6** | method (2 arms) | grouped bars or dots | ratio panel: 1.0 black dotted, ±10% shaded as the suspicion zone; no broken-control arm |
| **F7** | method | panel per DGP (lead IMF) | log x; bootstrap points sized by `B`; QIJ points sized by `M` |
| **F8** | **`vq_input`** | single panel | log-log; annotate both fitted slopes |
| **F9** | estimand | panel per estimand | markers sized by `n_j`; x = cell position |

---

## 7. Two conventions worth fixing now

**Direction of "good".** In every figure where a reference exists — 0 in F1, the diagonal in F2, 1.0 in F6, flatness in F3 — draw it **black dotted** and put the tolerance band behind it in `#DDDDDD`. A reader then learns one visual idiom and applies it everywhere.

**F0's linestyle inversion.** Elsewhere QIJ is solid and the bootstrap dashed. In F0 the bootstrap KDE is the broad underlying shape and QIJ's implied density overlays it, so dashed-on-top reads better. This is the one exception; say so in the caption rather than silently breaking the convention.

---

## 8. LNCS output specification

Target: Springer *Lecture Notes in Computer Science* (`llncs.cls`), **single column**, text block ~**122 mm x 193 mm** (4.80 x 7.60 in), **12 pages including references**.

**Verify the width rather than trusting the number** — put `\the\textwidth` in the document and read it in TeX points (1 pt = 1/72.27 in). Verified 19 Sept 2026 with the revision's `llncs.cls`, `runningheads`: text width 347.12 pt = 4.80 in, text height 549.14 pt = 7.60 in.

### The rule that prevents mismatched fonts

**Design at final size; never scale in LaTeX.** Building at 8 in and placing with `\includegraphics[width=\textwidth]` shrinks every font by 0.6x, and inconsistently between figures built at different sizes.

Set `figsize` to the exact final inches, set fonts to the exact final points, save with `bbox_inches=None`, place with a bare `\includegraphics{fig}` — no width option, 1:1.

### Sizes

| use | `figsize` (in) | applies to |
|---|---|---|
| full width, 1x3 panels | `(4.80, 1.90)` | F0, merged validation figure |
| full width, 1x2 panels | `(4.80, 2.20)` | |
| full width, single panel | `(4.80, 2.80)` | F4, F7 |
| full width, 2x2 panels | `(4.80, 3.60)` | F3, one panel per DGP |
| half width | `(2.32, 2.00)` | 4 mm gutter |
| full width, 2x3 panels | `(4.80, 3.00)` | the final six-coordinate figures B (influence) and C (cost), 19 Sept 2026: three panels across at about 1.4 in each after gutters, two rows of near-square panels; figure A (accuracy): two panels (coverage, width ratio), six rows, two markers per row; its size and layout are set by the author directly with the coder (19 Sept 2026), not by this guide |

### Fonts at 4.8 in

The §4 sizes assume a larger canvas and are oversized here. LNCS body is 10 pt, captions 9 pt; figure text sits just below the caption.

| element | pt | weight |
|---|---|---|
| axes title (panel, e.g. "Pareto") | 9 | **bold** |
| axis label | 8.5 | **bold** |
| legend title | 8 | **bold** |
| legend text, tick labels | 7.5 | regular |
| annotation | 7 | regular |
| panel label `(a)` | 9 | **bold** |

**Drop the figure `suptitle`.** The caption carries it, and a title duplicates the caption while costing ~0.25 in of vertical space per figure. Bold axis and legend titles stay.

### PDF settings

There is **one** rcParams dict, the `RC` above, applied by `use_qij_style()`.
An earlier draft of this guide printed a second, `RC_LNCS`, layered over it
with `**RC`. That two-dict form is gone from `figures.py` (20 September) and
must not come back: one size, one build, one style. The paper-size values it
carried are already in `RC`, alongside:

```python
    "pdf.fonttype":       42,      # TrueType, embedded. Default 3 is often rejected
    "ps.fonttype":        42,
    "savefig.transparent": False,
```

`pdf.fonttype = 42` is the setting that matters; Type 3 fonts fail many publisher checks.

### `figure.constrained_layout.use` MUST be False

This is not a style preference, and the value printed here was `True` until
20 September, which was a live bug. Every figure lays itself out by hand with
`subplots_adjust` and passes `constrained_layout=False` to `plt.subplots`,
which does leave the engine as `None` **at construction** -- but that opt-out
does not survive a save. `Figure.savefig` wraps the save in
`figure._cm_set(layout_engine='none')`, and on exit that restores the captured
value by calling `set_layout_engine(layout=None)`, whose contract for `None`
is not "no engine" but "take it from rcParams". With this True, the first
`savefig` silently installs a live `ConstrainedLayoutEngine`; the **second**
save on the same figure object then executes it and discards every
`subplots_adjust`.

Measured on matplotlib 3.9.4: Figure D's axes went from top 0.800 / bottom
0.275 to top 0.936 / bottom 0.072 between the first and second save. Since the
figures are written as a PDF and a PNG from one object, whichever was written
SECOND got the broken geometry -- order-dependent, not format-dependent, which
is what made it confusing to diagnose. `savefig.bbox` is likewise `None` and
not `"tight"`: cropping was investigated as the cause and ruled out, but the
two must stay consistent for widths to be exactly equal across figures.

### Page budget: consolidate nine figures into five

Twelve pages less ~1.5 for references leaves ~10.5. At ~0.4 page each including caption, nine figures consume ~3.6 pages and leave under 7 pages of text (~3300 words) for a method with this much to explain.

| fig | panels | `figsize` | contains |
|---|---|---|---|
| **1** | 1x3 | `(4.80, 1.90)` | F0 — three draws, one estimand |
| **2** | 1x3 | `(4.80, 2.20)` | (a) paired ratio [F1] (b) calibration [F2] (c) width parity [F6] |
| **3** | 2x2 | `(4.80, 3.60)` | F3 — decomposition invariance, one panel per DGP |
| **4** | 1x2 | `(4.80, 2.20)` | (a) `d_eff` across DGPs [F4] (b) MVT arms [F8] |
| **5** | 1x2 | `(4.80, 2.20)` | F7 — cost frontier, IMF lead |

~2.2 pages of figures. **F5 (variance-estimate stability) and F9 (influence field) move to supplementary** — the first is a second-order claim, the second is interpretability, and neither is load-bearing.

Merging F1/F2/F6 into a single validation figure is also better rhetoric: correctness is one point, not three.
