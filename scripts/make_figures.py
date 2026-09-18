#!/usr/bin/env python3.9
"""Render the paper's figures (F2, F3, F7, F9, F10) from the products of
a `qij.study` run (plan section 3). Reads nothing but those products --
no estimator is re-run, no dataset is redrawn.

    python scripts/make_figures.py <run_dir> [--lncs]
    python scripts/make_figures.py <run_dir> --cost-vs-n <cost_dir> [--lncs]

F2, F3, F7 and F9 are rendered from <run_dir> (a main or smoke run).
F10 needs the cost-vs-N study's per-N runs; --cost-vs-n names a
directory holding one subdirectory per N, named by the integer N (e.g.
`N1000`, `N3000`, ...), each itself a normal run_dir containing
fp/fp/{truth,qij}.parquet and boot.h5 -- no product records N (plan
section 5's schema has no N column), so this directory-naming
convention is this script's own, not a pinned interface.
"""

from __future__ import annotations

import argparse
import os

from qij import figures


def _save(fig, out_dir: str, name: str, lncs: bool) -> None:
    suffix = "_lncs" if lncs else ""
    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"{name}{suffix}.pdf")
    fig.savefig(path)
    print(f"saved {path}")


def _cost_dirs(root: str) -> dict:
    """{N: run_dir} for every N-named subdirectory of `root` (see the
    module docstring's --cost-vs-n convention)."""
    out = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            n = int(name.lstrip("N"))
        except ValueError:
            continue
        out[n] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="products directory from qij.study (e.g. a main or smoke run)")
    parser.add_argument("--lncs", action="store_true", help="render at LNCS final size instead of the default")
    parser.add_argument("--cost-vs-n", type=str, default=None,
                        help="directory holding one N-named subdirectory per cost_vs_n run, for F10")
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
