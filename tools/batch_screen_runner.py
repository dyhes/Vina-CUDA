#!/usr/bin/env python3
"""
Batch screening runner: 1 protein × 285 ligands in a single process.
Uses --ligand_directory mode (all ligands share one center/box).
Measures wall time for batch kernel vs serial per-ligand calls.
"""
import os, sys, time, subprocess, shutil
from pathlib import Path

KBASE = Path(os.path.expanduser("~")) / "grad"
DATASET_DIR = KBASE / "dataset/CASF-2016-PDBQT/coreset"
BINARY = sys.argv[1] if len(sys.argv) > 1 else KBASE / "variants/QuickVina2-CUDA-batch_p2/QuickVina2-GPU-2-1-CUDA"
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
    # Use 1a30 as the target protein
    target = "1a30"
    protein = DATASET_DIR / target / f"{target}_protein.pdbqt"
    center_lig = DATASET_DIR / target / f"{target}_ligand.pdbqt"
    center = get_center_from_pdbqt(str(center_lig))
    cx, cy, cz = center

    # Create ligand directory with ALL 285 coreset ligands
    lig_dir = Path("/tmp/batch_screen_285")
    if lig_dir.exists():
        shutil.rmtree(str(lig_dir))
    lig_dir.mkdir(parents=True)

    count = 0
    for tid in sorted(os.listdir(str(DATASET_DIR))):
        lig = DATASET_DIR / tid / f"{tid}_ligand.pdbqt"
        if lig.exists():
            shutil.copy(str(lig), str(lig_dir / f"{tid}_ligand.pdbqt"))
            count += 1

    print(f"Protein: {target}, Ligands: {count}")
    print(f"Center: ({cx:.1f}, {cy:.1f}, {cz:.1f}), Box: {BOX_SIZE}")

    out_dir = Path("/tmp/batch_screen_285_out")
    if out_dir.exists():
        shutil.rmtree(str(out_dir))

    cmd = [
        "sudo", "bash", "-c",
        f'ulimit -s 8192 && exec "{BINARY}" '
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

    # Count outputs
    outputs = list(out_dir.glob("*_out.pdbqt")) if out_dir.exists() else []

    # Parse GPU time
    k1 = k2 = ""
    for line in (r.stdout + r.stderr).split("\n"):
        if "GPU_KERNEL1_TIME=" in line: k1 = line.split("=")[1].strip()
        elif "GPU_KERNEL2_TIME=" in line: k2 = line.split("=")[1].strip()

    print(f"\nResults: {len(outputs)}/{count} outputs")
    print(f"Wall: {wall:.1f}s ({wall/count:.3f}s/ligand)")
    print(f"GPU kernel1: {k1}s, kernel2: {k2}s")
    print(f"Return code: {r.returncode}")
    if r.returncode != 0:
        print(f"STDERR (last 5 lines): {r.stderr[-500:]}")

if __name__ == "__main__":
    main()
