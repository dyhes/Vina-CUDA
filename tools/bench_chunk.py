#!/usr/bin/env python3
"""Run batch screening with configurable chunk size."""
import os, sys, time, subprocess, shutil
from pathlib import Path

KBASE = Path(os.path.expanduser("~")) / "grad"
DATASET_DIR = KBASE / "dataset/CASF-2016-PDBQT/coreset"
BINARY = KBASE / "variants/QuickVina2-CUDA-batch_chunk/QuickVina2-GPU-2-1-CUDA"
BOX_SIZE = 22.5
THREAD = 8192

def get_center_from_pdbqt(path):
    xs, ys, zs = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                try:
                    xs.append(float(parts[5])); ys.append(float(parts[6])); zs.append(float(parts[7]))
                except: pass
    if not xs: return None
    return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)

def main():
    chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    target = "1a30"
    protein = DATASET_DIR / target / f"{target}_protein.pdbqt"
    center_lig = DATASET_DIR / target / f"{target}_ligand.pdbqt"
    cx, cy, cz = get_center_from_pdbqt(str(center_lig))

    lig_dir = Path("/tmp/batch_screen_285")
    out_dir = Path(f"/tmp/batch_screen_285_out_chunk{chunk_size}")

    cmd = [
        "sudo", "bash", "-c",
        f'export CHUNK_SIZE={chunk_size} && ulimit -s 8192 && exec "{BINARY}" '
        f'--receptor "{protein}" '
        f'--ligand_directory "{lig_dir}" '
        f'--output_directory "{out_dir}" '
        f'--thread {THREAD} '
        f'--center_x {cx} --center_y {cy} --center_z {cz} '
        f'--size_x {BOX_SIZE} --size_y {BOX_SIZE} --size_z {BOX_SIZE}'
    ]

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0

    outputs = list(out_dir.glob("*_out.pdbqt")) if out_dir.exists() else []

    k2 = ""
    for line in (r.stdout + r.stderr).split("\n"):
        if "GPU_KERNEL2_TIME=" in line:
            k2 = line.split("=")[1].strip()

    print(f"chunk_size={chunk_size}: {len(outputs)}/285 outputs, wall={wall:.1f}s ({wall/285:.3f}s/lig), k2={k2}s, rc={r.returncode}")

if __name__ == "__main__":
    main()
