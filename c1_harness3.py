#!/usr/bin/env python3
"""C-1 concurrency harness v3: capture per-target stderr, tunable POOL, find GPU
saturation point. No sudo. Writes <tid>.err next to each failure for diagnosis."""
import os, subprocess, sys, csv
from pathlib import Path

ROOT = Path("/home/ubuntu/grad")
BIN = ROOT / "variants/QuickVina2-CUDA-m1_fusion_p1_c/QuickVina2-GPU-2-1-CUDA"
CORET = ROOT / "dataset/CASF-2016-PDBQT/coreset"
EXP = ROOT / "experiments/Jul18_C1_concurrent_casf_p1_c"
OUT = EXP / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BOX = 23
THREAD = 8192

def center_of(lig):
    try:
        with open(lig) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    p = line.split()
                    return float(p[6]), float(p[7]), float(p[8])
    except Exception:
        return None
    return None

targets = sorted(p for p in os.listdir(CORET) if (CORET/p).is_dir())
print(f"targets={len(targets)} pool={POOL}", flush=True)

procs = {}
results = {}

def wait_one():
    pid, rc = os.wait()
    for k, v in list(procs.items()):
        p = v[0] if isinstance(v, tuple) else v
        if p.pid == pid:
            del procs[k]
            return k
    return None

for tid in targets:
    d = CORET/tid
    prot = d/f"{tid}_protein.pdbqt"
    lig = d/f"{tid}_ligand.pdbqt"
    if not (prot.exists() and lig.exists()):
        results[tid] = "SKIP"; continue
    c = center_of(lig)
    if c is None:
        results[tid] = "FAIL_center"; continue
    cx, cy, cz = c
    outf = OUT/f"{tid}_out.pdbqt"
    errf = OUT/f"{tid}.err"
    cmd = [str(BIN), "--receptor", str(prot), "--ligand", str(lig),
           "--thread", str(THREAD),
           "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", str(BOX), "--size_y", str(BOX), "--size_z", str(BOX),
           "--out", str(outf)]
    while len(procs) >= POOL:
        wait_one()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(errf, "w"))
    procs[tid] = (p, errf)

while procs:
    wait_one()

ok = 0
for tid in targets:
    outf = OUT/f"{tid}_out.pdbqt"
    if outf.exists() and outf.stat().st_size > 0:
        results[tid] = "OK"; ok += 1
    elif results.get(tid) in (None,):
        results[tid] = "FAIL"
# clean empty err files for OK
for tid in targets:
    errf = OUT/f"{tid}.err"
    if errf.exists() and errf.stat().st_size == 0:
        errf.unlink()

print(f"DONE_OK={ok}/{len(targets)}", flush=True)
with open(EXP/f"c1_summary_pool{POOL}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["target", "status"])
    for tid in targets:
        w.writerow([tid, results.get(tid, "FAIL")])
print("written", flush=True)
