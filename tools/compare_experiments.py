#!/usr/bin/env python3
"""Compare two docking experiments (same CASF dataset) on docking-power + speed.

Mirrors the statistics printed by run_experiment.py's run_rmsd_analysis:
  - Success rate, Top-1 RMSD <=2.0A (all-atom), <=1.0A (all-atom)
  - Median / Mean RMSD (all-atom, symmetry-corrected via spyrmsd)
  - Avg / Median runtime per target (from summary.csv, OK rows)

This is the canonical docking-power comparison used in the thesis. Always feed it
two experiment dirs produced by run_experiment.py (they contain rmsd_results.csv
+ summary.csv).

Usage:
  python3 tools/compare_experiments.py \
      experiments/Jul03_2008_CASF-2016_qvina2_cuda \
      experiments/Jul15_1454_CASF-2016_qvina2_cuda_m1_fusion

Output: side-by-side table (baseline vs new) + delta %, and a PASS/FAIL on
"RMSD<=2A% must not regress" (tolerance configurable via --tol, default 2pp).
"""
import csv
import os
import sys
import statistics
from pathlib import Path


def load_rmsd(exp_dir):
    p = Path(exp_dir) / "rmsd_results.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    vals = [float(r["rmsd_all_atom"]) for r in rows if r["status"] == "OK" and r["rmsd_all_atom"]]
    return vals


def load_runtime(exp_dir):
    p = Path(exp_dir) / "summary.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    rt = [float(r["runtime_s"]) for r in rows if r["status"] == "OK" and r.get("runtime_s")]
    return rt


def stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return {
        "n": n,
        "le2": sum(1 for x in s if x <= 2.0),
        "le1": sum(1 for x in s if x <= 1.0),
        "median": s[n // 2],
        "mean": sum(s) / n,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    base_dir, new_dir = sys.argv[1], sys.argv[2]
    tol = 2.0  # percentage-point tolerance for regression check
    if len(sys.argv) > 3:
        tol = float(sys.argv[3])

    rb = load_rmsd(base_dir)
    rn = load_rmsd(new_dir)
    tb = load_runtime(base_dir)
    tn = load_runtime(new_dir)

    if rb is None or rn is None:
        print("ERROR: missing rmsd_results.csv in one of the experiment dirs")
        sys.exit(1)

    sb, sn = stats(rb), stats(rn)
    ab, an = (statistics.mean(tb), statistics.median(tb)) if tb else (None, None)
    an_, mn_ = (statistics.mean(tn), statistics.median(tn)) if tn else (None, None)

    def pct(a, b):
        return 100 * (b - a) / a if a else 0.0

    print(f"{'metric':<20}{'baseline':>14}{'new':>14}{'delta':>12}")
    print("-" * 60)
    print(f"{'targets (OK)':<20}{sb['n']:>14}{sn['n']:>14}")
    print(f"{'Top-1 <=2.0A':<20}{sb['le2']:>10}/{sb['n']:<3}{sn['le2']:>10}/{sn['n']:<3}"
          f"{(100*sn['le2']/sn['n']-100*sb['le2']/sb['n']):>+10.1f}pp")
    print(f"{'Top-1 <=1.0A':<20}{sb['le1']:>10}/{sb['n']:<3}{sn['le1']:>10}/{sn['n']:<3}"
          f"{(100*sn['le1']/sn['n']-100*sb['le1']/sb['n']):>+10.1f}pp")
    print(f"{'Median RMSD':<20}{sb['median']:>14.3f}{sn['median']:>14.3f}{pct(sb['median'],sn['median']):>+10.1f}%")
    print(f"{'Mean RMSD':<20}{sb['mean']:>14.3f}{sn['mean']:>14.3f}{pct(sb['mean'],sn['mean']):>+10.1f}%")
    if ab:
        print(f"{'Avg time (s)':<20}{ab:>14.2f}{an_:>14.2f}{pct(ab,an_):>+10.1f}%")
        print(f"{'Med time (s)':<20}{mn_:>14.2f}{statistics.median(tn):>14.2f}"
              f"{pct(mn_,statistics.median(tn)):>+10.1f}%")

    # regression check on docking power
    dp_base = 100 * sb['le2'] / sb['n']
    dp_new = 100 * sn['le2'] / sn['n']
    verdict = "PASS (no regression)" if dp_new >= dp_base - tol else "FAIL (regression!)"
    print("-" * 60)
    print(f"Docking-power check (tol={tol}pp): {verdict}")


if __name__ == "__main__":
    main()
