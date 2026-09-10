#!/usr/bin/env python3
"""CASF-2016 Scoring Power Evaluation (Python 3, no sklearn)"""
import os, sys, re, csv, math
import numpy as np
import pandas as pd
from scipy import stats

CORESET = "/home/ubuntu/grad/dataset/CASF-2016/power_scoring/CoreSet.dat"

# Experiments to evaluate: (label, summary_csv_path)
EXPERIMENTS = [
    ("Vina-CPU",       "/home/ubuntu/grad/experiments/May29_0028_CASF-2016_Vina-CPU_auto/summary.csv"),
    ("Vina-GPU2",      "/home/ubuntu/grad/experiments/May31_2050_CASF-2016_vina_gpu2_8192auto/summary.csv"),
    ("QVina-W-CUDA",   "/home/ubuntu/grad/experiments/May31_1228_CASF-2016_qvina_w_cuda/summary.csv"),
    ("Vina-CUDA",      "/home/ubuntu/grad/experiments/May31_1930_CASF-2016_vina_cuda_8192auto/summary.csv"),
    ("QuickVina2-CUDA","/home/ubuntu/grad/experiments/May31_2017_CASF-2016_qvina2_cuda_8192auto/summary.csv"),
    ("QuickVina2-GPU2","/home/ubuntu/grad/experiments/May31_2232_CASF-2016_qvina2_gpu2/summary.csv"),
    ("Uni-Dock detail","/home/ubuntu/grad/experiments/May31_1818_CASF-2016_unidock_A_detail/summary.csv"),
]

def parse_coreset(path):
    """Parse CoreSet.dat -> dict {pdb_code: logKa}"""
    coreset = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                code = parts[0]
                logKa = float(parts[3])
                coreset[code] = logKa
    return coreset

def parse_summary(path):
    """Parse summary.csv -> dict {pdb_code: affinity}"""
    scores = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'OK':
                target = row['target']
                affinity = float(row['affinity_kcal_mol'])
                scores[target] = affinity
    return scores

def scoring_power(coreset, scores, label):
    """Compute Pearson R, SD, regression for scoring power."""
    # Vina scores are negative (more negative = stronger binding)
    # So we negate them to make positive (higher = stronger)
    matched_codes = []
    matched_logka = []
    matched_score = []
    
    for code, logka in coreset.items():
        if code in scores:
            matched_codes.append(code)
            matched_logka.append(logka)
            matched_score.append(-scores[code])  # negate for positive correlation
    
    n = len(matched_codes)
    if n < 3:
        return None
    
    logka = np.array(matched_logka)
    score = np.array(matched_score)
    
    # Pearson correlation
    r, p_value = stats.pearsonr(logka, score)
    
    # Linear regression: logKa = a * Score + b
    slope, intercept, r_val, p_val, std_err = stats.linregress(score, logka)
    
    # SD = sqrt(SSE / (n-1))
    predicted = slope * score + intercept
    sse = np.sum((logka - predicted) ** 2)
    sd = math.sqrt(sse / (n - 1))
    
    return {
        'label': label,
        'N': n,
        'R': r,
        'R2': r**2,
        'SD': sd,
        'slope': slope,
        'intercept': intercept,
        'p': p_value,
    }

def main():
    print("=" * 70)
    print("CASF-2016 Scoring Power Evaluation")
    print("=" * 70)
    
    coreset = parse_coreset(CORESET)
    print(f"\nCoreSet: {len(coreset)} targets with logKa values")
    
    results = []
    for label, path in EXPERIMENTS:
        if not os.path.exists(path):
            print(f"  [SKIP] {label}: {path} not found")
            continue
        scores = parse_summary(path)
        res = scoring_power(coreset, scores, label)
        if res:
            results.append(res)
            print(f"  [OK]   {label}: N={res['N']}, R={res['R']:.3f}, SD={res['SD']:.3f}")
        else:
            print(f"  [FAIL] {label}: insufficient data")
    
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'Method':<22} {'N':>4} {'R':>8} {'R²':>8} {'SD':>8} {'p-value':>12}")
    print("-" * 70)
    
    # Sort by R descending
    results.sort(key=lambda x: x['R'], reverse=True)
    for r in results:
        print(f"{r['label']:<22} {r['N']:>4} {r['R']:>8.4f} {r['R2']:>8.4f} {r['SD']:>8.3f} {r['p']:>12.2e}")
    
    print("-" * 70)
    
    # Also print CASF-2016 benchmark values for reference
    print("\nReference (CASF-2016 paper benchmarks):")
    print(f"  AutoDock Vina:  R = 0.566 (285 targets)")
    print(f"  X-Score HM:    R = 0.644 (285 targets)")
    print(f"  GlideScore-SP: R = 0.576 (285 targets)")

    # Score distribution analysis
    print("\n" + "=" * 70)
    print("Score Statistics per Method")
    print("=" * 70)
    print(f"{'Method':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 56)
    for label, path in EXPERIMENTS:
        if not os.path.exists(path):
            continue
        scores = parse_summary(path)
        vals = list(scores.values())
        if vals:
            print(f"{label:<22} {np.mean(vals):>8.2f} {np.std(vals):>8.2f} {np.min(vals):>8.2f} {np.max(vals):>8.2f}")

if __name__ == '__main__':
    main()
