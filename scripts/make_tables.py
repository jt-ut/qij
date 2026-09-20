"""
Write the study's tables (plan section 8) from a run's products,
into `<run_dir>/tables/`.

    python3.9 scripts/make_tables.py /path/to/qij_runs/main

`run_dir` is the study directory to table, and the only input: the
products of plan section 3 under `<run_dir>/<dataset>/<estimator>/`,
plus `<run_dir>/config.yaml` for the coverage grid's levels.

This used to take a second argument, the one-worker timing run, as the
cost table's only source of absolute seconds. The author retired that
run on 20 September: every wall time now comes from the main run's own
per-draw fields, which is what Figure D reads too. `timing_dir` is
accepted and ignored so an older invocation does not error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qij.tables import cost_table, coverage_grid, t1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str,
                         help="the study directory to table, e.g. qij_runs/main")
    parser.add_argument("timing_dir", type=str, nargs="?", default=None,
                         help="ignored (retired 20 September); accepted so an "
                              "older invocation does not error")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    t1(run_dir).to_csv(out_dir / "t1.csv", index=False)
    coverage_grid(run_dir).to_csv(out_dir / "coverage_grid.csv", index=False)
    cost_table(run_dir).to_csv(out_dir / "cost_table.csv", index=False)

    print(f"wrote tables to {out_dir}")


if __name__ == "__main__":
    main()
