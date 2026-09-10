#!/usr/bin/env python3
"""
CASF-2016 Forward Screening Power evaluation (Python 3 rewrite).

Usage:
  python3 screening_power_py3.py \\
    -t TargetInfo.dat \\
    -s scores_dir \\
    -p negative \\
    -o output_prefix

Inputs:
  -t: TargetInfo.dat (57 targets + known binders)
  -s: Directory containing {target}_score.dat files
  -p: 'negative' (Vina, lower=better) or 'positive'
  -o: Output prefix (default: screening_result)

Output:
  {prefix}_summary.txt  — EF at 1%, 5%, 10% for each target + mean
  {prefix}_detail.txt   — full ranked list per target
"""

import argparse
import os
import sys
from collections import defaultdict

def parse_target_info(path):
    """Parse TargetInfo.dat → {target: [binder1, binder2, ...]}"""
    targets = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            targets[parts[0]] = parts[1:]  # all binder PDB IDs
    return targets


def parse_score_file(path):
    """Parse score file → {ligand_id: score}"""
    scores = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    scores[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return scores


def compute_ef(scores, binders, reverse=False):
    """Compute Enrichment Factor at 1%, 5%, 10%.
    
    Args:
        scores: {ligand_id: score}
        binders: list of known binder IDs
        reverse: if True, higher score = better (e.g. X-Score)
                 if False, lower score = better (e.g. Vina)
    
    Returns:
        (ef_1, ef_5, ef_10, n_total, n_binders)
    """
    if not scores:
        return (0.0, 0.0, 0.0, 0, 0)

    n_total = len(scores)
    n_binders = sum(1 for b in binders if b in scores)

    if n_binders == 0 or n_total == 0:
        return (0.0, 0.0, 0.0, n_total, n_binders)

    # Sort by score (ascending = best first for Vina)
    sorted_ligs = sorted(scores.items(), key=lambda x: x[1], reverse=reverse)

    cutoffs = {
        '1%': max(1, round(n_total * 0.01)),
        '5%': max(1, round(n_total * 0.05)),
        '10%': max(1, round(n_total * 0.10)),
    }

    efs = {}
    for label, cutoff in cutoffs.items():
        top_ligs = [lig for lig, _ in sorted_ligs[:cutoff]]
        n_found = sum(1 for lig in top_ligs if lig in binders)
        expected = n_binders * cutoff / n_total
        if expected > 0:
            efs[label] = n_found / expected
        else:
            efs[label] = 0.0

    return (efs['1%'], efs['5%'], efs['10%'], n_total, n_binders)


def main():
    parser = argparse.ArgumentParser(description='CASF-2016 Screening Power (Python 3)')
    parser.add_argument('-t', '--target', required=True, help='TargetInfo.dat path')
    parser.add_argument('-s', '--scores', required=True, help='Scores directory')
    parser.add_argument('-p', '--prefer', default='negative', choices=['negative', 'positive'],
                        help='Scoring preference: negative (Vina) or positive')
    parser.add_argument('-o', '--output', default='screening_result', help='Output prefix')
    args = parser.parse_args()

    reverse = (args.prefer == 'positive')

    # Parse targets
    targets = parse_target_info(args.target)
    print(f"Loaded {len(targets)} screening targets")

    # Process each target
    results = []
    detail_lines = []

    for target in sorted(targets):
        binders = set(targets[target])
        score_file = os.path.join(args.scores, f"{target}_score.dat")

        if not os.path.exists(score_file):
            print(f"  WARNING: {score_file} not found, skipping")
            continue

        scores = parse_score_file(score_file)
        ef1, ef5, ef10, n_total, n_binders = compute_ef(scores, binders, reverse=reverse)

        results.append({
            'target': target,
            'n_total': n_total,
            'n_binders': n_binders,
            'ef_1': ef1,
            'ef_5': ef5,
            'ef_10': ef10,
        })

        # Detail: ranked list
        sorted_ligs = sorted(scores.items(), key=lambda x: x[1], reverse=reverse)
        detail_lines.append(f"\n=== {target} (n={n_total}, binders={n_binders}) ===")
        for rank, (lig, score) in enumerate(sorted_ligs, 1):
            marker = " *" if lig in binders else ""
            detail_lines.append(f"  {rank:4d}. {lig:10s} {score:8.3f}{marker}")

    # Summary stats
    if not results:
        print("ERROR: No results computed")
        sys.exit(1)

    ef1_vals = [r['ef_1'] for r in results]
    ef5_vals = [r['ef_5'] for r in results]
    ef10_vals = [r['ef_10'] for r in results]

    mean_ef1 = sum(ef1_vals) / len(ef1_vals)
    mean_ef5 = sum(ef5_vals) / len(ef5_vals)
    mean_ef10 = sum(ef10_vals) / len(ef10_vals)

    median_ef1 = sorted(ef1_vals)[len(ef1_vals) // 2]
    median_ef5 = sorted(ef5_vals)[len(ef5_vals) // 2]
    median_ef10 = sorted(ef10_vals)[len(ef10_vals) // 2]

    # Write summary
    summary_file = f"{args.output}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"CASF-2016 Forward Screening Power ({args.prefer})\n")
        f.write(f"{'='*70}\n")
        f.write(f"{'Target':<8} {'N_total':>7} {'N_bind':>7} {'EF_1%':>8} {'EF_5%':>8} {'EF_10%':>8}\n")
        f.write(f"{'-'*70}\n")

        for r in results:
            f.write(f"{r['target']:<8} {r['n_total']:>7} {r['n_binders']:>7} "
                    f"{r['ef_1']:>8.2f} {r['ef_5']:>8.2f} {r['ef_10']:>8.2f}\n")

        f.write(f"{'-'*70}\n")
        f.write(f"{'Mean':<8} {'':>7} {'':>7} "
                f"{mean_ef1:>8.2f} {mean_ef5:>8.2f} {mean_ef10:>8.2f}\n")
        f.write(f"{'Median':<8} {'':>7} {'':>7} "
                f"{median_ef1:>8.2f} {median_ef5:>8.2f} {median_ef10:>8.2f}\n")
        f.write(f"{'='*70}\n")

    # Write detail
    detail_file = f"{args.output}_detail.txt"
    with open(detail_file, 'w') as f:
        f.write(f"CASF-2016 Forward Screening Power — Detail\n")
        f.write(f"(* = known binder)\n")
        f.write('\n'.join(detail_lines))

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Screening Power Summary ({len(results)} targets)")
    print(f"{'='*70}")
    print(f"  EF @ 1%:  mean={mean_ef1:.2f}  median={median_ef1:.2f}")
    print(f"  EF @ 5%:  mean={mean_ef5:.2f}  median={median_ef5:.2f}")
    print(f"  EF @ 10%: mean={mean_ef10:.2f}  median={median_ef10:.2f}")
    print(f"{'='*70}")
    print(f"  Output: {summary_file}")
    print(f"          {detail_file}")


if __name__ == '__main__':
    main()
