#!/usr/bin/env python3.9
"""Render the paper's figures (F2, F3, F7, F9, F10) from the products of
a `qij.study` run (plan section 3). Reads nothing but those products --
no estimator is re-run, no dataset is redrawn.

    python scripts/make_figures.py <run_dir> [--lncs]
    python scripts/make_figures.py <run_dir> --cost-vs-n <cost_dir> [--lncs]

F2, F3, F7 and F9 are rendered from <run_dir> (a main or smoke run).
F10 needs the cost-vs-N study's per-N products; --cost-vs-n names the
directory a `cost_vs_n.yaml` run wrote (`study.py`'s own multi-N
layout, module docstring there): `<cost_dir>/<dataset>/<estimator>/
N<size>/{truth.parquet, qij.parquet, boot.h5, ...}`, one (dataset,
estimator) pair with several `N<size>` subdirectories. Discovered by
globbing for `truth.parquet` under `N*` directories, the same way
every other figure discovers its products -- no dataset or estimator
name is assumed, so this works for a cost study on any estimator.
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
    """{N: product_dir} for the cost-vs-N study's one (dataset,
    estimator) pair under `root` -- `study.py` writes a multi-N run's
    products under `<root>/<dataset>/<estimator>/N<size>/`, so the
    `N<size>` directories are found by globbing for `truth.parquet`
    three levels down, not by assuming `root`'s immediate children are
    the N directories or that the pair is any particular dataset or
    estimator (see the module docstring)."""
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
    parser.add_argument("--lncs", action="store_true", help="render at LNCS final size instead of the default")
    parser.add_argument("--cost-vs-n", type=str, default=None,
                        help="products directory from a cost_vs_n.yaml study run, for F10")
    args = parser.parse_args()

    for name, make in (("f2", figures.fig2), ("f3", figures.fig3),
                       ("f7", figures.fig7), ("f9", figures.fig9)):
        fig = make(args.run_dir, lncs=args.lncs)
        _save(fig, args.run_dir, name, args.lncs)

    if args.cost_vs_n:
        cost_dirs = _cost_dirs(args.cost_vs_n)
        fig = figures.fig10(cost_dirs, lncs=args.lncs)
        _save(fig, args.cost_vs_n, "f10", args.lncs)


if __name__ == "__main__":
    main()
