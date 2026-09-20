#!/usr/bin/env python3.9
"""Render the paper's figures (Figure A/B/C/D, `QIJ_figure_spec_final.md`)
from the products of `qij.study` runs. Reads nothing but those products --
no estimator is re-run, no dataset is redrawn.

    python scripts/make_figures.py <run_dir> [options]

`<run_dir>` (a main or smoke run) supplies Figure A, Figure B and Figure C
directly -- Figure C's 19 September redesign measures both methods
against the truth in evaluation-cost units, so it no longer needs a
separate timing run (the reason `--timing-dir` below is Figure D's flag
alone). Figure D needs the cost-vs-N study's per-N products
(`--cost-vs-n`), and, optionally, an IMF cost-vs-N sweep
(`--cost-vs-n-imf`) for its accuracy panel's IMF M* line -- `_cost_dirs`
finds the `N<size>` directories under each by globbing for
`truth.parquet`, the same layout `study.py`'s multi-N runs write, one
(dataset, estimator) pair per root. Figure D's 19 September redesign also
needs the timing run (`--timing-dir`, 20 single-worker single-thread
draws, `<root>/<dataset>/<estimator>/`) for its new "when QIJ pays" panel
-- the one panel in this whole CLI that reads wall time as its subject
rather than as a byproduct, so it is the one panel that needs this flag.

`--out-dir` overrides where the `figures/` directory is written (default:
alongside each source run, matching the old CLI's behaviour). This exists
so a smoke rendering used only to check that the code runs can be pointed
at a scratch directory instead of a project's `runs/` tree -- the smoke
run is not a result and its figures should not land next to a real run's.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

from qij import figures


def _save(fig, out_dir: str, name: str) -> None:
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    # Both formats, every time: the PDF is what the paper includes at 1:1,
    # the PNG is what gets looked at and passed around. They are the same
    # figure at the same size, so a comment on one applies to the other.
    path = os.path.join(figures_dir, f"{name}.pdf")
    fig.savefig(path)
    fig.savefig(os.path.join(figures_dir, f"{name}.png"), dpi=300)
    print(f"saved {path}")


def _cost_dirs(root: str) -> dict:
    """{N: product_dir} for a cost-vs-N study's one (dataset, estimator)
    pair under `root` -- `study.py` writes a multi-N run's products under
    `<root>/<dataset>/<estimator>/N<size>/`, so the `N<size>` directories
    are found by globbing for `truth.parquet` three levels down, not by
    assuming `root`'s immediate children are the N directories or that
    the pair is any particular dataset or estimator."""
    out = {}
    pattern = os.path.join(root, "*", "*", "N*", "truth.parquet")
    for truth_path in sorted(glob.glob(pattern)):
        n_dir = os.path.dirname(truth_path)
        m = re.fullmatch(r"N(\d+)", os.path.basename(n_dir))
        if m is None:
            continue
        out[int(m.group(1))] = n_dir
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="products directory from qij.study (e.g. a main or smoke run)")
    parser.add_argument("--cost-vs-n", type=str, default=None,
                        help="products directory from a cost_vs_n.yaml study run (Fundamental Plane), for Figure D")
    parser.add_argument("--cost-vs-n-imf", type=str, default=None,
                        help="products directory from an IMF cost_vs_n run, for Figure D's "
                             "accuracy panel's optional IMF M* line")
    parser.add_argument("--timing-dir", type=str, default=None,
                        help="products directory from the timing study (20 single-worker, "
                             "single-thread draws per dataset/estimator) -- Figure D's own "
                             "panel (a) alone; Figure C's 19 September redesign dropped this "
                             "package's other use of a timing run, so nothing else reads it")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="write figures/ here instead of alongside each source run "
                             "(use a scratch directory for a smoke rendering)")
    args = parser.parse_args()

    fig_a = figures.fig_a(args.run_dir)
    _save(fig_a, args.out_dir or args.run_dir, "fig_a")

    fig_b = figures.fig_b(args.run_dir)
    _save(fig_b, args.out_dir or args.run_dir, "fig_b")

    fig_c = figures.fig_c(args.run_dir)
    _save(fig_c, args.out_dir or args.run_dir, "fig_c")

    if args.cost_vs_n:
        if not args.timing_dir:
            parser.error("--timing-dir is required together with --cost-vs-n: Figure D's "
                         "'when QIJ pays' panel reads it and nothing else in this figure does")
        cost_dirs = _cost_dirs(args.cost_vs_n)
        imf_cost_dirs = _cost_dirs(args.cost_vs_n_imf) if args.cost_vs_n_imf else None
        fig_d = figures.fig_d(cost_dirs, args.timing_dir, imf_run_dirs=imf_cost_dirs)
        _save(fig_d, args.out_dir or args.cost_vs_n, "fig_d")


if __name__ == "__main__":
    main()
