"""
Write the study's tables (plan section 8) from a run's products,
into `<run_dir>/tables/`.

    python3.9 scripts/make_tables.py /path/to/qij_runs/main /path/to/qij_runs/timing

`run_dir` is the study directory to table (the products of plan
section 3 under `<run_dir>/<dataset>/<estimator>/`, plus
`<run_dir>/config.yaml` for the coverage grid's levels). `timing_dir`
is the one-worker, one-thread timing run (plan section 7), the only
source for the cost table's absolute-seconds layer (layer 3, plan
section 8) -- required, not optional, because the cost table has
nowhere else to read it from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qij.tables import cost_table, coverage_grid, t1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str,
                         help="the study directory to table, e.g. qij_runs/main")
    parser.add_argument("timing_dir", type=str,
                         help="the timing study directory (plan section 7), "
                              "for the cost table's layer-3 seconds")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    timing_dir = Path(args.timing_dir)
    out_dir = run_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    t1(run_dir).to_csv(out_dir / "t1.csv", index=False)
    coverage_grid(run_dir).to_csv(out_dir / "coverage_grid.csv", index=False)
    cost_table(run_dir, timing_dir).to_csv(out_dir / "cost_table.csv", index=False)

    print(f"wrote tables to {out_dir}")


if __name__ == "__main__":
    main()
