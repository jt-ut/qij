#!/usr/bin/env python3.9
"""Render the paper's figures (Figure A/B/C/D, `QIJ_figure_spec_final.md`)
from the products of `qij.study` runs. Reads nothing but those products --
no estimator is re-run, no dataset is redrawn.

    python scripts/make_figures.py <run_dir> [options]

`<run_dir>` (a main or smoke run) supplies Figure A and Figure B directly.
Figure C additionally needs the separate timing run (`--timing-dir`, one
worker, one thread) for its corner wall-time annotation. Figure D needs
the cost-vs-N study's per-N products (`--cost-vs-n`), and, optionally, an
IMF cost-vs-N sweep (`--cost-vs-n-imf`) to add its sweet-spot row --
`_cost_dirs` finds the `N<size>` directories under each by globbing for
`truth.parquet`, the same layout `study.py`'s multi-N runs write, one
(dataset, estimator) pair per root.

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


def _save(fig, out_dir: str, name: str, lncs: bool) -> None:
    suffix = "_lncs" if lncs else ""
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"{name}{suffix}.pdf")
    fig.savefig(path)
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
    parser.add_argument("--lncs", action="store_true", help="render at LNCS final size instead of the draft size")
    parser.add_argument("--timing-dir", type=str, default=None,
                        help="the one-worker, one-thread timing run, for Figure C's corner wall times")
    parser.add_argument("--cost-vs-n", type=str, default=None,
                        help="products directory from a cost_vs_n.yaml study run (Fundamental Plane), for Figure D")
    parser.add_argument("--cost-vs-n-imf", type=str, default=None,
                        help="products directory from an IMF cost_vs_n run, for Figure D's optional IMF sweep row")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="write figures/ here instead of alongside each source run "
                             "(use a scratch directory for a smoke rendering)")
    args = parser.parse_args()

    fig_a = figures.fig_a(args.run_dir, lncs=args.lncs)
    _save(fig_a, args.out_dir or args.run_dir, "fig_a", args.lncs)

    fig_b = figures.fig_b(args.run_dir, lncs=args.lncs)
    _save(fig_b, args.out_dir or args.run_dir, "fig_b", args.lncs)

    if args.timing_dir:
        fig_c = figures.fig_c(args.run_dir, args.timing_dir, lncs=args.lncs)
        _save(fig_c, args.out_dir or args.run_dir, "fig_c", args.lncs)

    if args.cost_vs_n:
        cost_dirs = _cost_dirs(args.cost_vs_n)
        imf_cost_dirs = _cost_dirs(args.cost_vs_n_imf) if args.cost_vs_n_imf else None
        fig_d = figures.fig_d(cost_dirs, imf_run_dirs=imf_cost_dirs, lncs=args.lncs)
        _save(fig_d, args.out_dir or args.cost_vs_n, "fig_d", args.lncs)


if __name__ == "__main__":
    main()
