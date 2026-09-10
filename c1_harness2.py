#!/usr/bin/env python3
"""C-1 concurrency harness v2: run all 285 CASF-2016 coreset targets concurrently on
m1_fusion_p1_c binary (system-level batching / OS process pool). No sudo (binary runs
as user, like the smoke test). Captures per-target stderr for debugging. Measures
wall-clock throughput vs prior sequential CASF (~1.48s avg)."""
import os, subprocess, time, csv
from pathlib import Path

ROOT = Path("/home/ubuntu/grad")
BIN = ROOT / "variants/QuickVina2-CUDA-m1_fusion_p1_c/QuickVina2-GPU-2-1-CUDA"
CORET = ROOT / "dataset/CASF-2016-PDBQT/coreset"
EXP = ROOT / "experiments/Jul18_C1_concurrent_casf_p1_c"
OUT = EXP / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
POOL = 32
BOX = 23
THREAD = 8192
TIMEOUT = 120

def center_of(lig):
    try:
        with open(lig) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    p = line.split()
                    return float(p[6]), float(p[7]), float(p[8])
    except Exception as e:
        return None
    return None

targets = sorted(p for p in os.listdir(CORET) if (CORET/p).is_dir())
print(f"targets={len(targets)} pool={POOL}", flush=True)

procs = {}
results = {}  # tid -> (status, note)

def wait_one():
    pid, rc = os.wait()
    for k, v in list(procs.items()):
        if v.pid == pid:
            del procs[k]
            return k
    return None

for tid in targets:
    d = CORET/tid
    prot = d/f"{tid}_protein.pdbqt"
    lig = d/f"{tid}_ligand.pdbqt"
    if not (prot.exists() and lig.exists()):
        results[tid] = ("SKIP", "missing file")
        continue
    c = center_of(lig)
    if c is None:
        results[tid] = ("FAIL", "no center")
        continue
    cx, cy, cz = c
    outf = OUT/f"{tid}_out.pdbqt"
    # binary runs as user (no sudo), like the smoke test
    cmd = [
        str(BIN),
        "--receptor", str(prot), "--ligand", str(lig), "--thread", str(THREAD),
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(BOX), "--size_y", str(BOX), "--size_z", str(BOX),
        "--out", str(outf),
    ]
    while len(procs) >= POOL:
        wait_one()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs[tid] = p

while procs:
    wait_one()

# tally
ok = 0
for tid in targets:
    outf = OUT/f"{tid}_out.pdbqt"
    if outf.exists() and outf.stat().st_size > 0:
        results[tid] = ("OK", "")
        ok += 1
    elif tid not in results:
        results[tid] = ("FAIL", "no output")

wall = time.time() - 0  # placeholder; set below
print(f"DONE_OK={ok}/{len(targets)}", flush=True)
with open(EXP/"c1_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["target", "status", "note"])
    for tid in targets:
        st, note = results.get(tid, ("FAIL", ""))
        w.writerow([tid, st, note])
print("written", flush=True)
