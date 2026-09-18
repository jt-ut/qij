#!/usr/bin/env python3.9
"""Run the QIJ study (plan Sec 6):

    python scripts/run.py configs/main.yaml /path/to/qij_runs/main --workers 14

Sets one BLAS thread before numpy is imported anywhere -- plan Sec 6:
both methods for a draw run in one process on one worker with one BLAS
thread, so the wall-time ratio within a draw is meaningful -- then copies
the config verbatim to `<out_dir>/config.yaml` (plan Sec 3) and calls
`qij.study.run_study`.
"""
import os

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import shutil

import yaml

from qij.study import run_study


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', help='path to a study config YAML')
    p.add_argument('out_dir', help='output directory for the products (plan Sec 3)')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help='draws run in parallel across this many workers '
                         '(default: cpu_count - 1)')
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(args.out_dir, 'config.yaml'))

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_study(config, args.out_dir, args.workers)


if __name__ == '__main__':
    main()
