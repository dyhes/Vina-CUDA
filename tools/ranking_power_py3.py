#!/usr/bin/env python3
"""CASF-2016 Ranking Power Evaluation (Python 3, no sklearn)"""
import os, sys, csv, math
import numpy as np
import pandas as pd
from scipy import stats

CORESET = "/home/ubuntu/grad/dataset/CASF-2016/power_ranking/CoreSet.dat"

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
    """Parse CoreSet.dat -> list of dicts with code, logKa, target_group"""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 6:
                rows.append({
                    'code': parts[0],
                    'logKa': float(parts[3]),
                    'target': int(parts[5])
                })
    return rows

def parse_summary(path):
    """Parse summary.csv -> dict {pdb_code: affinity}"""
    scores = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'OK':
                scores[row['target']] = float(row['affinity_kcal_mol'])
    return scores

def predictive_index(logka, score):
    """
    Predictive Index: weighted concordance for top-5 ranking.
    For each pair (i,j) with i < j (sorted by logKa ascending):
      w_ij = |logKa_i - logKa_j|
      if score_i < score_j (correct order): +w_ij
      if score_i > score_j (wrong order):   -w_ij
      if score_i == score_j:                  0
    PI = sum(w_ij * concordance) / sum(w_ij)
    """
    n = len(logka)
    # Sort indices by logKa ascending (weakest to strongest binder)
    order = np.argsort(logka)
    W = []
    WC = []
    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            i = order[i_idx]
            j = order[j_idx]
            w = abs(logka[i] - logka[j])
            W.append(w)
            if score[i] < score[j]:
                WC.append(w)    # correct: higher logKa has higher (negated) score
            elif score[i] > score[j]:
                WC.append(-w)   # wrong order
            else:
                WC.append(0)
    total_w = sum(W)
    if total_w == 0:
        return 0.0
    return sum(WC) / total_w

def ranking_power(coreset, scores, label):
    """Compute Spearman, Kendall, PI for each target group."""
    # Merge
    merged = []
    for row in coreset:
        if row['code'] in scores:
            merged.append({
                'code': row['code'],
                'logKa': row['logKa'],
                'target': row['target'],
                'score': -scores[row['code']]  # negate: higher = stronger
            })
    
    df = pd.DataFrame(merged)
    
    # Group by target
    groups = df.groupby('target')
    
    spearman_list = []
    kendall_list = []
    pi_list = []
    valid_groups = 0
    
    for tgt_id, group_df in groups:
        if len(group_df) != 5:
            continue
        valid_groups += 1
        
        logka = group_df['logKa'].values
        score = group_df['score'].values
        
        # Spearman
        sp, _ = stats.spearmanr(logka, score)
        spearman_list.append(sp)
        
        # Kendall
        kt, _ = stats.kendalltau(logka, score)
        kendall_list.append(kt)
        
        # PI
        pi = predictive_index(logka, score)
        pi_list.append(pi)
    
    n = valid_groups
    if n == 0:
        return None
    
    return {
        'label': label,
        'N': n,
        'SP': np.mean(spearman_list),
        'SP_std': np.std(spearman_list),
        'tau': np.mean(kendall_list),
        'tau_std': np.std(kendall_list),
        'PI': np.mean(pi_list),
        'PI_std': np.std(pi_list),
        'SP_list': spearman_list,
        'tau_list': kendall_list,
        'PI_list': pi_list,
    }

def main():
    print("=" * 75)
    print("CASF-2016 Ranking Power Evaluation")
    print("=" * 75)
    
    coreset = parse_coreset(CORESET)
    n_groups = len(set(r['target'] for r in coreset))
    print(f"\nCoreSet: {len(coreset)} entries, {n_groups} target groups (5 ligands each)")
    
    results = []
    for label, path in EXPERIMENTS:
        if not os.path.exists(path):
            print(f"  [SKIP] {label}: {path} not found")
            continue
        scores = parse_summary(path)
        res = ranking_power(coreset, scores, label)
        if res:
            results.append(res)
            print(f"  [OK]   {label}: N={res['N']}, SP={res['SP']:.3f}, tau={res['tau']:.3f}, PI={res['PI']:.3f}")
        else:
            print(f"  [FAIL] {label}: no valid 5-member groups")
    
    print("\n" + "=" * 75)
    print("Summary Table")
    print("=" * 75)
    print(f"{'Method':<22} {'N':>3} {'SP':>8} {'SP_std':>8} {'tau':>8} {'tau_std':>8} {'PI':>8} {'PI_std':>8}")
    print("-" * 75)
    
    results.sort(key=lambda x: x['SP'], reverse=True)
    for r in results:
        print(f"{r['label']:<22} {r['N']:>3} {r['SP']:>8.3f} {r['SP_std']:>8.3f} {r['tau']:>8.3f} {r['tau_std']:>8.3f} {r['PI']:>8.3f} {r['PI_std']:>8.3f}")
    
    print("-" * 75)
    print("\nReference (CASF-2016 paper benchmarks):")
    print(f"  AutoDock Vina:  SP = 0.441, tau = 0.313, PI = 0.497")
    print(f"  X-Score HM:    SP = 0.577, tau = 0.436, PI = 0.629")
    print(f"  GlideScore-SP: SP = 0.547, tau = 0.409, PI = 0.601")
    
    # Per-target breakdown: show best and worst targets
    if results:
        best = results[0]
        sp_vals = best['SP_list']
        print(f"\n\nPer-target Spearman distribution ({best['label']}):")
        print(f"  Mean: {np.mean(sp_vals):.3f}")
        print(f"  Median: {np.median(sp_vals):.3f}")
        print(f"  Min: {np.min(sp_vals):.3f}")
        print(f"  Max: {np.max(sp_vals):.3f}")
        print(f"  > 0.5: {sum(1 for v in sp_vals if v > 0.5)}/{len(sp_vals)}")
        print(f"  > 0.0: {sum(1 for v in sp_vals if v > 0.0)}/{len(sp_vals)}")
        print(f"  < -0.5: {sum(1 for v in sp_vals if v < -0.5)}/{len(sp_vals)}")

if __name__ == '__main__':
    main()
