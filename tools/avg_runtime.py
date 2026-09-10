#!/usr/bin/env python3
"""Compute consistent average runtime from summary.csv.

Usage:
  python3 tools/avg_runtime.py experiments/May31_1930_CASF-2016_vina_cuda_8192auto/summary.csv
  python3 tools/avg_runtime.py experiments/*/summary.csv

Output: avg (s/tgt), total (s), n, wall-time estimate.
Always uses sum(OK runtime) / count(OK) = s/tgt.
"""

import csv
import sys
from pathlib import Path


def avg_runtime(summary_path):
    runtimes = []
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            if row['status'] == 'OK' and row.get('runtime_s'):
                try:
                    runtimes.append(float(row['runtime_s']))
                except ValueError:
                    pass

    if not runtimes:
        return None, 0, 0

    total = sum(runtimes)
    n = len(runtimes)
    avg = total / n
    return avg, total, n


def main():
    if len(sys.argv) < 2:
        print("Usage: avg_runtime.py <summary.csv> [...]", file=sys.stderr)
        sys.exit(1)

    width_name = max(len(p) for p in sys.argv[1:]) + 2
    print(f"{'Experiment':<{width_name}} {'s/tgt':>7}  {'total_s':>8}  {'n':>4}  {'wall_min':>8}")
    print("-" * (width_name + 36))

    for path in sys.argv[1:]:
        avg, total, n = avg_runtime(path)
        if avg is None:
            print(f"{path:<{width_name}}  {'N/A':>7}")
        else:
            wall = total / 60
            name = Path(path).parent.name if path.endswith('/summary.csv') else path
            print(f"{name:<{width_name}} {avg:7.2f}  {total:8.0f}  {n:4d}  {wall:8.1f}")


if __name__ == "__main__":
    main()
