#!/usr/bin/env python3
"""C-1 concurrency harness: run all 285 CASF-2016 coreset targets concurrently on the
m1_fusion_p1_c binary (system-level batching / OS process pool). Measures wall-clock
throughput vs prior sequential CASF (~1.48s avg). Zero code changes to the binary."""
import os, subprocess, time, sys, csv
from pathlib import Path

ROOT = Path("/home/ubuntu/grad")
BIN = ROOT / "variants/QuickVina2-CUDA-m1_fusion_p1_c/QuickVina2-GPU-2-1-CUDA"
CORET = ROOT / "dataset/CASF-2016-PDBQT/coreset"
OUT = ROOT / "experiments/Jul18_C1_concurrent_casf_p1_c/outputs"
OUT.mkdir(parents=True, exist_ok=True)
POOL = 32          # max concurrent processes
BOX = 23
THREAD = 1
TIMEOUT = 120

def center_of(lig):
    try:
        for line in open(lig):
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
t_start = time.time()
def wait_one():
    pid, rc = os.wait()
    for k,v in list(procs.items()):
        if v.pid == pid:
            del procs[k]
            return k
    return None

for tid in targets:
    d = CORET/tid
    prot = d/f"{tid}_protein.pdbqt"
    lig = d/f"{tid}_ligand.pdbqt"
    if not (prot.exists() and lig.exists()):
        results[tid] = ("SKIP", "")
        continue
    c = center_of(lig)
    if c is None:
        results[tid] = ("FAIL", "")
        continue
    cx,cy,cz = c
    outf = OUT/f"{tid}_out.pdbqt"
    cmd = ["sudo","bash","-c",
           f'cd {BIN.parent} && ulimit -s 8192 && exec ./{BIN.name} '
           f'--receptor "{prot}" --ligand "{lig}" --thread {THREAD} '
           f'--center_x {cx} --center_y {cy} --center_z {cz} '
           f'--size_x {BOX} --size_y {BOX} --size_z {BOX} --out "{outf}"']
    while len(procs) >= POOL:
        wait_one()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs[tid] = p

while procs:
    wait_one()

t_end = time.time()
wall = t_end - t_start

ok = sum(1 for v in results.values() if v[0]=="OK")
# re-scan outputs for OK (procs finished; check files)
ok = 0
for tid in targets:
    outf = OUT/f"{tid}_out.pdbqt"
    if outf.exists() and outf.stat().st_size>0:
        results.setdefault(tid, ("OK",""))
        ok += 1
    elif tid not in results:
        results[tid] = ("FAIL","")

print(f"WALL={wall:.2f}s OK={ok}/{len(targets)}", flush=True)
with open(OUT.parent/"c1_summary.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["target","status","runtime"])
    for tid in targets:
        st = results.get(tid,("FAIL",""))[0]
        w.writerow([tid, st, ""])
print("done", flush=True)
