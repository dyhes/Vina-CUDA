#!/usr/bin/env python3
"""CASF-2016 Scoring + Ranking Power (9 methods including Uni-Dock fast/balance)"""
import os, sys, csv, math
import numpy as np
from scipy import stats

CORESET = "/home/ubuntu/grad/dataset/CASF-2016/power_scoring/CoreSet.dat"

EXPERIMENTS = [
    ("Vina-CPU",       "/home/ubuntu/grad/experiments/May29_0028_CASF-2016_Vina-CPU_auto/summary.csv"),
    ("Vina-GPU2",      "/home/ubuntu/grad/experiments/May31_2050_CASF-2016_vina_gpu2_8192auto/summary.csv"),
    ("UniDock detail", "/home/ubuntu/grad/experiments/May31_1818_CASF-2016_unidock_A_detail/summary.csv"),
    ("UniDock balance","/home/ubuntu/grad/experiments/May31_1758_CASF-2016_unidock_A_balance/summary.csv"),
    ("UniDock fast",   "/home/ubuntu/grad/experiments/May31_1202_CASF-2016_unidock_A_fast/summary.csv"),
    ("QuickVina2-CUDA","/home/ubuntu/grad/experiments/May31_2017_CASF-2016_qvina2_cuda_8192auto/summary.csv"),
    ("QVina-W-CUDA",   "/home/ubuntu/grad/experiments/May31_1228_CASF-2016_qvina_w_cuda/summary.csv"),
    ("Vina-CUDA",      "/home/ubuntu/grad/experiments/May31_1930_CASF-2016_vina_cuda_8192auto/summary.csv"),
    ("QuickVina2-GPU2","/home/ubuntu/grad/experiments/May31_2232_CASF-2016_qvina2_gpu2/summary.csv"),
]

def parse_coreset(path):
    coreset = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 6:
                coreset[parts[0]] = {"logKa": float(parts[3]), "target": int(parts[5])}
    return coreset

def parse_summary(path):
    scores = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "OK":
                scores[row["target"]] = float(row["affinity_kcal_mol"])
    return scores

def predictive_index(logka, score):
    n = len(logka)
    order = np.argsort(logka)
    W, WC = [], []
    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            i, j = order[i_idx], order[j_idx]
            w = abs(logka[i] - logka[j])
            W.append(w)
            if score[i] < score[j]:
                WC.append(w)
            elif score[i] > score[j]:
                WC.append(-w)
            else:
                WC.append(0)
    total_w = sum(W)
    return sum(WC) / total_w if total_w > 0 else 0.0

# ============================================================
# SCORING POWER
# ============================================================
print("=" * 75)
print("CASF-2016 Scoring Power (9 methods)")
print("=" * 75)

coreset = parse_coreset(CORESET)
all_scores = {}

scoring_results = []
for label, path in EXPERIMENTS:
    scores = parse_summary(path)
    all_scores[label] = scores
    matched = []
    for code, info in coreset.items():
        if code in scores:
            matched.append((info["logKa"], -scores[code]))
    n = len(matched)
    if n < 3:
        continue
    logka = np.array([m[0] for m in matched])
    score = np.array([m[1] for m in matched])
    r, p = stats.pearsonr(logka, score)
    slope, intercept, _, _, _ = stats.linregress(score, logka)
    predicted = slope * score + intercept
    sd = math.sqrt(np.sum((logka - predicted)**2) / (n - 1))
    scoring_results.append((label, n, r, r**2, sd, p))

print("{:<22} {:>4} {:>8} {:>8} {:>8}".format("Method", "N", "R", "R2", "SD"))
print("-" * 55)
scoring_results.sort(key=lambda x: x[2], reverse=True)
for label, n, r, r2, sd, p in scoring_results:
    print("{:<22} {:>4} {:>8.4f} {:>8.4f} {:>8.3f}".format(label, n, r, r2, sd))
print("-" * 55)
print("  CASF-2016 Vina: R = 0.566")

# ============================================================
# RANKING POWER
# ============================================================
print()
print("=" * 75)
print("CASF-2016 Ranking Power (9 methods)")
print("=" * 75)

rank_results = []
for label, path in EXPERIMENTS:
    scores = parse_summary(path)
    # Build merged df
    rows = []
    for code, info in coreset.items():
        if code in scores:
            rows.append({"code": code, "logKa": info["logKa"],
                         "target": info["target"], "score": -scores[code]})
    import pandas as pd
    df = pd.DataFrame(rows)
    groups = df.groupby("target")

    sp_list, kt_list, pi_list = [], [], []
    for tgt_id, gdf in groups:
        if len(gdf) != 5:
            continue
        logka = gdf["logKa"].values
        score = gdf["score"].values
        sp, _ = stats.spearmanr(logka, score)
        kt, _ = stats.kendalltau(logka, score)
        pi = predictive_index(logka, score)
        sp_list.append(sp)
        kt_list.append(kt)
        pi_list.append(pi)

    n = len(sp_list)
    if n > 0:
        rank_results.append((label, n, np.mean(sp_list), np.mean(kt_list), np.mean(pi_list)))

print("{:<22} {:>3} {:>8} {:>8} {:>8}".format("Method", "N", "SP", "tau", "PI"))
print("-" * 55)
rank_results.sort(key=lambda x: x[2], reverse=True)
for label, n, sp, kt, pi in rank_results:
    print("{:<22} {:>3} {:>8.3f} {:>8.3f} {:>8.3f}".format(label, n, sp, kt, pi))
print("-" * 55)
print("  CASF-2016 Vina: SP=0.441, tau=0.313, PI=0.497")
