#!/usr/bin/env python3
"""
Multi-ligand batch runner for QuickVina2-CUDA.

Strategy: Instead of spawning one subprocess per ligand (285 × ~380ms overhead),
spawn N parallel subprocesses, each handling a subset of ligands.
Each process gets its own CUDA context but shares the GPU.

This is the system-level parallelism approach (verified 2.2× in earlier experiments).
"""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

KBASE = Path(os.path.expanduser("~")) / "grad"
DATASET_DIR = KBASE / "dataset/CASF-2016-PDBQT/coreset"
BINARY = KBASE / "milestones/v5/QuickVina2-GPU-2-1-CUDA"
OUTPUT_DIR = KBASE / "experiments"
BOX_SIZE = 22.5
THREAD = 8192

def get_center_from_pdbqt(pdbqt_path):
    xs, ys, zs = [], [], []
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                try:
                    x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                    xs.append(x); ys.append(y); zs.append(z)
                except (IndexError, ValueError):
                    continue
    if not xs:
        return None
    return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)

def parse_affinity(pdbqt_path):
    if not os.path.exists(pdbqt_path):
        return None
    with open(pdbqt_path) as f:
        for line in f:
            if "VINA RESULT" in line or "REMARK VINA" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == "affinity" and i + 1 < len(parts):
                        try:
                            return float(parts[i + 1])
                        except ValueError:
                            pass
    return None

def parse_gpu_time(output):
    k1 = k2 = ""
    for line in output.split("\n"):
        if "GPU_KERNEL1_TIME=" in line:
            try: k1 = f"{float(line.split('=')[1]):.3f}"
            except: pass
        elif "GPU_KERNEL2_TIME=" in line:
            try: k2 = f"{float(line.split('=')[1]):.3f}"
            except: pass
    return k1, k2

def dock_one(tid, workdir):
    """Dock a single target. Returns (tid, status, affinity, runtime, k1, k2)."""
    protein = DATASET_DIR / tid / f"{tid}_protein.pdbqt"
    ligand = DATASET_DIR / tid / f"{tid}_ligand.pdbqt"
    output_file = workdir / f"{tid}_out.pdbqt"

    if not protein.exists() or not ligand.exists():
        return (tid, "SKIP", "", "", "", "")

    center = get_center_from_pdbqt(str(ligand))
    if center is None:
        return (tid, "FAIL", "", "", "", "")

    cx, cy, cz = center
    cmd = [
        "sudo", "bash", "-c",
        f'cd "{workdir}" && ulimit -s 8192 && exec "{BINARY}" '
        f'--receptor "{protein}" --ligand "{ligand}" '
        f'--thread {THREAD} '
        f'--center_x {cx} --center_y {cy} --center_z {cz} '
        f'--size_x {BOX_SIZE} --size_y {BOX_SIZE} --size_z {BOX_SIZE} '
        f'--out "{output_file}"'
    ]

    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        rt = time.time() - t0
        k1, k2 = parse_gpu_time(r.stderr + r.stdout)

        if r.returncode == 0 and output_file.exists():
            aff = parse_affinity(str(output_file))
            return (tid, "OK", f"{aff:.1f}" if aff else "", f"{rt:.2f}", k1, k2)
        else:
            return (tid, "FAIL", "", f"{rt:.2f}", k1, k2)
    except subprocess.TimeoutExpired:
        return (tid, "TIMEOUT", "", "300", "", "")

def main():
    parser = argparse.ArgumentParser(description="Batch CASF runner with process parallelism")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output directory name")
    args = parser.parse_args()

    timestamp = time.strftime("%b%d_%H%M")
    exp_name = args.output or f"Jul{timestamp}_CASF-2016_v5_pool{args.workers}"
    exp_dir = OUTPUT_DIR / exp_name
    os.makedirs(str(exp_dir), exist_ok=True)
    outputs_dir = exp_dir / "outputs"
    os.makedirs(str(outputs_dir), exist_ok=True)

    targets = sorted(os.listdir(str(DATASET_DIR)))
    print(f"CASF-2016: {len(targets)} targets, {args.workers} parallel workers")
    print(f"Output: {exp_dir}")

    results = []
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(dock_one, tid, outputs_dir): tid for tid in targets}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            tid = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append((tid, "ERROR", "", "", "", ""))

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t_start
                ok = sum(1 for r in results if r[1] == "OK")
                print(f"  [{i+1}/{len(targets)}] {elapsed:.0f}s | OK={ok}")

    wall = time.time() - t_start

    # Write summary
    summary_path = exp_dir / "summary.csv"
    with open(summary_path, "w") as f:
        f.write("target,status,affinity_kcal_mol,runtime_s,gpu_kernel1_s,gpu_kernel2_s\n")
        for tid, status, aff, rt, k1, k2 in sorted(results):
            f.write(f"{tid},{status},{aff},{rt},{k1},{k2}\n")

    ok = sum(1 for r in results if r[1] == "OK")
    fail = len(results) - ok
    print(f"\nDone: {ok}/{len(targets)} OK, {fail} FAIL, {wall/60:.1f} min")
    print(f"Wall: {wall:.1f}s | Per-ligand: {wall/len(targets):.3f}s")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()
