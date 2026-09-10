#!/usr/bin/env python3
"""
Unified docking experiment runner — one script for all variants.

Usage:
  # Redocking (default, 1:1)
  python3 run_experiment.py --variant unidock-A    --dataset CASF-2016 --search-mode fast
  python3 run_experiment.py --variant unidock-B    --dataset CASF-2016 --search-mode fast --chunk 64
  python3 run_experiment.py --variant vina-cpu     --dataset CASF-2016
  python3 run_experiment.py --variant qvina-w-cuda --dataset CASF-2016

  # Screening Power (57 targets × 285 ligands)
  python3 run_experiment.py --variant unidock-A    --dataset CASF-2016 --mode screening --search-mode fast
  python3 run_experiment.py --variant vina-cpu     --dataset CASF-2016 --mode screening
  python3 run_experiment.py --variant vina-gpu2    --dataset CASF-2016 --mode screening

Variants:
  unidock-A      Uni-Dock per-target GPU batch (aligns with Vina-CPU/QVina-W-CUDA)
  unidock-B      Uni-Dock paired_batch (bulk 1:1 redocking, with chunk support)
  vina-cpu       AutoDock Vina CPU
  qvina-w-cuda   QuickVina-W-CUDA GPU
  vina-cuda      Vina-CUDA GPU
  qvina2-cuda    QuickVina2-CUDA GPU
  vina-gpu2      Vina-GPU2 GPU
  qvina2-gpu2    QuickVina2-GPU2 GPU

Modes:
  redocking      1:1 redocking (default) — each target docks its own ligand
  screening      Screening Power — 57 targets × 285 coreset ligands

Output (redocking):
  experiments/<TIMESTAMP>_<DATASET>_<VARIANT>[_<MODE>]/
  ├── configs/          per-target config files
  ├── outputs/          docked PDBQT files
  ├── summary.csv       target,status,affinity_kcal_mol,runtime_s
  ├── rmsd_results.csv  target,status,rmsd_all_atom,n_atoms
  └── unidock.log       (Uni-Dock variants only)

Output (screening):
  experiments/<TIMESTAMP>_<DATASET>_<VARIANT>_screening/
  ├── scores/           {target}_score.dat files (#code_ligand_num score)
  ├── outputs/{target}/ docked PDBQT files per target
  ├── summary.csv       target,status,num_docked,runtime_s
  └── unidock.log       (Uni-Dock variants only)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── AICloud defaults (overridable via CLI) ──────────────────────────────
ROOT = Path(os.environ.get("GRAD_ROOT", "/home/ubuntu/grad"))
VARIANTS_DIR = ROOT / "variants"
MILESTONES_DIR = ROOT / "milestones"
DATASET_DIR = ROOT / "dataset"
EXPERIMENTS_DIR = ROOT / "experiments"
TOOLS_DIR = ROOT / "tools"

BINARIES = {
    "unidock":       Path("/usr/local/bin/unidock"),
    "vina-cpu":      VARIANTS_DIR / "AutoDock-Vina/build/linux/release/vina",
    "qvina-w-cuda":  VARIANTS_DIR / "QuickVina-W-CUDA/Vina-GPU-2-1-W-CUDA",
    "vina-cuda":     VARIANTS_DIR / "Vina-CUDA/Vina-GPU-2-1-CUDA",
    "qvina2-cuda":   VARIANTS_DIR / "QuickVina2-CUDA-m0/QuickVina2-GPU-2-1-CUDA",
    "m0":            VARIANTS_DIR / "QuickVina2-CUDA-m0/QuickVina2-GPU-2-1-CUDA",
    "qvina2-cuda-m1-fusion": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion/QuickVina2-GPU-2-1-CUDA",
    "m1_fusion_p1_tc":    VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_tc/QuickVina2-GPU-2-1-CUDA",
    "m1_fusion_p1_b":     VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_b/QuickVina2-GPU-2-1-CUDA",
    "m1_fusion_p1_c":     VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_c/QuickVina2-GPU-2-1-CUDA",
    "m1_fusion_p1_prefetch": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_prefetch/QuickVina2-GPU-2-1-CUDA",
    "m1_fusion_p1_gamma": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_gamma/QuickVina2-GPU-2-1-CUDA",
    "gamma_b1_g0a": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a/QuickVina2-GPU-2-1-CUDA",
    "gamma_b1_g0a_sg": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_sharedgrid/QuickVina2-GPU-2-1-CUDA",
    "gamma_b1_g0a_gs": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate/QuickVina2-GPU-2-1-CUDA",
    "gamma_b1_g0a_b1": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate_b1/QuickVina2-GPU-2-1-CUDA",
    "v1_lsmax6": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_lsmax6/QuickVina2-GPU-2-1-CUDA",
    "v1_lsmax8": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_lsmax8/QuickVina2-GPU-2-1-CUDA",
    "v1_lsmax8_a2": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_lsmax8_bfgs_early/QuickVina2-GPU-2-1-CUDA",
    "v2_d1b": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_v2_d1b/QuickVina2-GPU-2-1-CUDA",
    "v2_d1c": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_v2_d1c/QuickVina2-GPU-2-1-CUDA",
    "v2_d1d": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_v2_d1d/QuickVina2-GPU-2-1-CUDA",
    "v2_d1e": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_v2_d1e/QuickVina2-GPU-2-1-CUDA",
    "v2_pscan": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_v2_parallel_scan/QuickVina2-GPU-2-1-CUDA",
    "v3_lbfgs8": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_v3_lbfgs8/QuickVina2-GPU-2-1-CUDA",
    "v3_qg": VARIANTS_DIR / "tmp_qg/QuickVina2-GPU-2-1-CUDA",
    "vdeserted1": MILESTONES_DIR / "vdeserted1/QuickVina2-GPU-2-1-CUDA",
    "vdeserted3": MILESTONES_DIR / "vdeserted3/QuickVina2-GPU-2-1-CUDA",
    "vdeserted4": VARIANTS_DIR / "QuickVina2-CUDA-v6_trilinear_ilp/QuickVina2-GPU-2-1-CUDA",
    "v3": MILESTONES_DIR / "v3/QuickVina2-GPU-2-1-CUDA",
    "v4": MILESTONES_DIR / "v4/QuickVina2-GPU-2-1-CUDA",
    "v4_5": VARIANTS_DIR / "QuickVina2-CUDA-v45/QuickVina2-GPU-2-1-CUDA",
    "v4_7": VARIANTS_DIR / "QuickVina2-CUDA-v45/QuickVina2-GPU-2-1-CUDA",
    "v4_8": VARIANTS_DIR / "QuickVina2-CUDA-v45/QuickVina2-GPU-2-1-CUDA",
    "v4_8b": VARIANTS_DIR / "QuickVina2-CUDA-v45/QuickVina2-GPU-2-1-CUDA",
    "v4_resync": VARIANTS_DIR / "v4_resync/QuickVina2-GPU-2-1-CUDA",
    "v4_mixed_fp16": VARIANTS_DIR / "v4_mixed_fp16/QuickVina2-GPU-2-1-CUDA",
    "v5_memcpy_single_m": MILESTONES_DIR / "v5/QuickVina2-GPU-2-1-CUDA",
    "v6_batch_chunk": MILESTONES_DIR / "v6/QuickVina2-GPU-2-1-CUDA",
    "fp16_grid_v5": VARIANTS_DIR / "QuickVina2-CUDA-fp16_grid_v5/QuickVina2-GPU-2-1-CUDA",
    "fp16_sr_grid_v5": VARIANTS_DIR / "QuickVina2-CUDA-fp16_sr_grid_v5/QuickVina2-GPU-2-1-CUDA",
    "warp_spec": VARIANTS_DIR / "QuickVina2-CUDA-warp_spec/QuickVina2-GPU-2-1-CUDA",
    "v5_sigtest": VARIANTS_DIR / "QuickVina2-CUDA-v5_sigtest/QuickVina2-GPU-2-1-CUDA",
    "v0":    MILESTONES_DIR / "v0/QuickVina2-GPU-2-1-CUDA",
    "v1":    MILESTONES_DIR / "v1/QuickVina2-GPU-2-1-CUDA",
    "v2_adals": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_v2_adals/QuickVina2-GPU-2-1-CUDA",
    "v1_a2_only": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_bfgs_early/QuickVina2-GPU-2-1-CUDA",
    "v1_ls_dyn3": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_ls_dyn3/QuickVina2-GPU-2-1-CUDA",
    "gamma_b1_g0a_pf": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate_prefetch/QuickVina2-GPU-2-1-CUDA",
    "gamma_b1_g0a_b3cb": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate_b3cb/QuickVina2-GPU-2-1-CUDA",
    "m1_fusion_p1_c_batch": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_c_batch/QuickVina2-GPU-2-1-CUDA",
    "vina-gpu2":     VARIANTS_DIR / "Vina-GPU2/AutoDock-Vina-GPU-2-1",
    "vina-up":       VARIANTS_DIR / "Vina-UP/build/linux/release/vina",
    "qvina2-gpu2":   VARIANTS_DIR / "QuickVina2-GPU2/QuickVina2-GPU",
}

CENTER_SCRIPT = TOOLS_DIR / "pdbqt_center.py"

# ── Dataset registry ────────────────────────────────────────────────────
DATASETS = {
    "CASF-2016": {
        "dir":             DATASET_DIR / "CASF-2016-PDBQT/coreset",
        "ref_type":        "ligand_pdbqt",
        "ligand_suffix":   "_ligand.pdbqt",
        "protein_suffix":  "_protein.pdbqt",
    },
    "AutoDockGPU": {
        "dir":             DATASET_DIR / "AutoDockGPU",
        "ref_type":        "flex-xray",
        "ligand_suffix":   "/flex-xray.pdbqt",
        "protein_suffix":  "/protein.pdbqt",
    },
}

# ── Screening Power constants ──────────────────────────────────────────
SCREENING_TARGET_INFO = DATASET_DIR / "CASF-2016/power_screening/TargetInfo.dat"


# ══════════════════════════════════════════════════════════════════════════
#  Core helpers
# ══════════════════════════════════════════════════════════════════════════

def get_center_from_pdbqt(pdbqt_path):
    """Extract geometric center from a PDBQT file using fixed-width columns."""
    cmd = ["python3", str(CENTER_SCRIPT), str(pdbqt_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return None
    parts = r.stdout.strip().split()
    if len(parts) >= 3:
        return tuple(float(x) for x in parts[:3])
    return None


def parse_affinity_from_pdbqt(pdbqt_path):
    """Parse Vina score from REMARK VINA RESULT line."""
    if not os.path.exists(pdbqt_path):
        return None
    with open(pdbqt_path) as f:
        for line in f:
            m = re.match(r"REMARK VINA RESULT:\s+([-\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


def parse_gpu_time(stderr_text):
    """Parse GPU kernel/memcpy timings from program output.
    k1/k2 always present (inherited). HtoD/DtoH/All present when the binary
    is built with gpu_timer (in-binary GPU-timeline ledger)."""
    out = {"k1": "", "k2": "", "htod": "", "dtoh": "", "memset": "", "all": "",
           "e2e_total": "", "e2e_warm": "", "e2e_dock": ""}
    for line in stderr_text.splitlines():
        if line.startswith("GPU_KERNEL1_TIME="):
            out["k1"] = line.split("=")[1].strip()
        elif line.startswith("GPU_KERNEL2_TIME="):
            out["k2"] = line.split("=")[1].strip()
        elif line.startswith("GPU_HtoD_TIME="):
            out["htod"] = line.split("=")[1].strip()
        elif line.startswith("GPU_DtoH_TIME="):
            out["dtoh"] = line.split("=")[1].strip()
        elif line.startswith("GPU_Memset_TIME="):
            out["memset"] = line.split("=")[1].strip()
        elif line.startswith("GPU_ALL_TIME="):
            out["all"] = line.split("=")[1].strip()
        elif line.startswith("E2E_TOTAL_S="):
            out["e2e_total"] = line.split("=")[1].strip()
        elif line.startswith("E2E_WARM_S="):
            out["e2e_warm"] = line.split("=")[1].strip()
        elif line.startswith("E2E_DOCK_S="):
            out["e2e_dock"] = line.split("=")[1].strip()
    return out


def write_summary(summary_path, results):
    """Write summary.csv: base 6 cols + gpu ledger cols (gpu_htod_s, gpu_dtoh_s, gpu_memset_s, gpu_all_s)."""
    with open(summary_path, "w") as f:
        f.write("target,status,affinity_kcal_mol,runtime_s,gpu_kernel1_s,gpu_kernel2_s,gpu_htod_s,gpu_dtoh_s,gpu_memset_s,gpu_all_s,e2e_total_s,e2e_warm_s,e2e_dock_s\n")
        for tid, info in sorted(results.items()):
            gk1 = info.get("gpu_kernel1", "")
            gk2 = info.get("gpu_kernel2", "")
            ghd = info.get("gpu_htod", "")
            gdh = info.get("gpu_dtoh", "")
            gms = info.get("gpu_memset", "")
            gall = info.get("gpu_all", "")
            et = info.get("e2e_total", ""); ew = info.get("e2e_warm", ""); ed = info.get("e2e_dock", "")
            f.write(f"{tid},{info.get('status','')},{info.get('affinity','')},{info.get('runtime','')},{gk1},{gk2},{ghd},{gdh},{gms},{gall},{et},{ew},{ed}\n")


def run_rmsd_analysis(exp_dir, dataset_info):
    """Compute Top-1 RMSD via spyrmsd (symmetry-corrected Kabsch)."""
    data_dir = dataset_info["dir"]
    ref_type = dataset_info["ref_type"]
    outputs_dir = os.path.join(exp_dir, "outputs")

    try:
        from spyrmsd import rmsd as srmsd, io as sio
    except ImportError:
        print("ERROR: spyrmsd not installed. pip install spyrmsd && apt install python3-openbabel",
              file=sys.stderr)
        return

    results = {}
    for outfile in sorted(Path(outputs_dir).glob("*_out.pdbqt")):
        basename = outfile.name
        if basename.endswith("_ligand_out.pdbqt"):
            tid = basename.replace("_ligand_out.pdbqt", "")
        elif basename.endswith("_out.pdbqt"):
            tid = basename.replace("_out.pdbqt", "")
        else:
            continue

        if ref_type == "ligand_pdbqt":
            ref = data_dir / tid / f"{tid}_ligand.pdbqt"
        else:
            ref = data_dir / tid / "flex-xray.pdbqt"

        score = parse_affinity_from_pdbqt(str(outfile))

        if not os.path.exists(str(ref)):
            results[tid] = {"score": score, "rmsd": None, "rmsd_heavy": None,
                            "n_atoms": 0, "status": "NO_REF"}
            continue

        try:
            ref_mols = sio.loadallmols(str(ref))
            out_mols = sio.loadallmols(str(outfile))
        except Exception:
            results[tid] = {"score": score, "rmsd": None, "rmsd_heavy": None,
                            "n_atoms": 0, "status": "LOAD_ERR"}
            continue

        if not ref_mols or not out_mols:
            results[tid] = {"score": score, "rmsd": None, "rmsd_heavy": None,
                            "n_atoms": 0, "status": "EMPTY"}
            continue

        top1_mol = out_mols[0]
        n_atoms = top1_mol.natoms

        try:
            r_all = srmsd.rmsdwrapper(ref_mols[0], top1_mol, symmetry=True, strip=False)[0]
            r_heavy = srmsd.rmsdwrapper(ref_mols[0], top1_mol, symmetry=True, strip=True)[0]
            results[tid] = {"score": score, "rmsd": r_all, "rmsd_heavy": r_heavy,
                            "n_atoms": n_atoms, "status": "OK"}
        except Exception:
            results[tid] = {"score": score, "rmsd": None, "rmsd_heavy": None,
                            "n_atoms": n_atoms, "status": "RMSD_ERR"}

    valid = {k: v for k, v in results.items() if v["rmsd"] is not None}
    rmsds = sorted([v["rmsd"] for v in valid.values()])
    n, n_total = len(rmsds), len(results)

    if n == 0:
        print("RMSD: no valid results")
        return

    leq_2 = sum(1 for r in rmsds if r <= 2.0)
    leq_1 = sum(1 for r in rmsds if r <= 1.0)
    mean_val = sum(rmsds) / n
    median_val = rmsds[n // 2]

    print("=" * 50)
    print(f"RMSD Results (spyrmsd, symmetry-corrected Top-1)")
    print("=" * 50)
    print(f"  Success:     {n}/{n_total} ({100*n/max(n_total,1):.1f}%)")
    print(f"  Top-1 ≤2.0Å: {leq_2}/{n} ({100*leq_2/n:.1f}%)")
    print(f"  Top-1 ≤1.0Å: {leq_1}/{n} ({100*leq_1/n:.1f}%)")
    print(f"  Median RMSD: {median_val:.3f}Å")
    print(f"  Mean RMSD:   {mean_val:.3f}Å")

    rmsd_csv = os.path.join(exp_dir, "rmsd_results.csv")
    with open(rmsd_csv, "w") as f:
        f.write("target,status,rmsd_all_atom,rmsd_heavy_atom,n_atoms\n")
        for tid in sorted(results):
            v = results[tid]
            r_all_str = str(round(v["rmsd"], 4)) if v["rmsd"] is not None else ""
            r_hvy_str = str(round(v["rmsd_heavy"], 4)) if v.get("rmsd_heavy") is not None else ""
            n_str = str(v["n_atoms"]) if v.get("n_atoms", 0) > 0 else ""
            f.write(f"{tid},{v['status']},{r_all_str},{r_hvy_str},{n_str}\n")


# ══════════════════════════════════════════════════════════════════════════
#  Screening Power helpers
# ══════════════════════════════════════════════════════════════════════════

def parse_target_info(path):
    """Parse TargetInfo.dat → {target_pdb: [binder_pdb, ...]}"""
    targets = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            targets[parts[0]] = parts[1:]
    return targets


def get_all_coreset_ligands(data_dir):
    """Return {pdb_id: ligand_path} for all coreset ligands."""
    ligands = {}
    for tid in sorted(os.listdir(str(data_dir))):
        lig = data_dir / tid / f"{tid}_ligand.pdbqt"
        if lig.exists():
            ligands[tid] = str(lig)
    return ligands


def write_score_file(path, scores):
    """Write CASF-2016 score file: #code_ligand_num score"""
    with open(path, 'w') as f:
        f.write("#code_ligand_num score\n")
        for lig_id, score in sorted(scores.items()):
            f.write(f"{lig_id} {score:.3f}\n")


def write_screening_summary(summary_path, results):
    """Write screening summary.csv: target,status,num_docked,runtime_s"""
    with open(summary_path, "w") as f:
        f.write("target,status,num_docked,runtime_s\n")
        for tid in sorted(results):
            v = results[tid]
            f.write(f"{tid},{v['status']},{v['num_docked']},{v['runtime']}\n")


def run_screening_evaluation(exp_dir):
    """Evaluate screening power using CASF-2016 evaluation script."""
    scores_dir = exp_dir / "scores"
    if not scores_dir.exists() or not list(scores_dir.glob("*.dat")):
        print("No score files found in scores/")
        return

    eval_script = TOOLS_DIR / "screening_power_py3.py"
    if not eval_script.exists():
        print(f"NOTE: {eval_script} not found yet. Screening evaluation deferred.")
        print(f"  Score files ready in: {scores_dir}")
        print(f"  Target info: {SCREENING_TARGET_INFO}")
        print(f"  Create screening_power_py3.py to compute Enrichment Factor (EF).")
        return

    # Combine all score files into one for evaluation
    combined_score = exp_dir / "combined_scores.dat"
    score_files = sorted(scores_dir.glob("*_score.dat"))

    with open(combined_score, 'w') as out:
        out.write("#code_ligand_num score target\n")
        for sf in score_files:
            target = sf.stem.replace("_score", "")
            with open(sf) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        out.write(f"{parts[0]} {parts[1]} {target}\n")

    print(f"Combined {len(score_files)} score files → {combined_score}")
    print(f"Run screening evaluation:")
    print(f"  python3 {eval_script} -t {SCREENING_TARGET_INFO} -s {combined_score} -p negative")


# ══════════════════════════════════════════════════════════════════════════
#  Variant runners (redocking)
# ══════════════════════════════════════════════════════════════════════════

def run_unidock_a(dataset_info, exp_dir, search_mode, box_size, num_modes, scoring):
    """Uni-Dock A: per-target GPU batch. One unidock call per target."""
    data_dir = dataset_info["dir"]
    lig_sfx = dataset_info["ligand_suffix"]
    prot_sfx = dataset_info["protein_suffix"]
    outputs_dir = exp_dir / "outputs"
    summary_path = exp_dir / "summary.csv"
    log_path = exp_dir / "unidock.log"

    targets = sorted(os.listdir(str(data_dir)))
    results = {}
    ok = fail = 0

    t_start = time.time()
    with open(log_path, "w") as log:
        for i, tid in enumerate(targets):
            protein = data_dir / tid / f"{tid}{prot_sfx.split('/')[-1]}" if "/" in prot_sfx else data_dir / tid / f"{tid}{prot_sfx}"
            ligand = data_dir / tid / f"{tid}{lig_sfx.split('/')[-1]}" if "/" in lig_sfx else data_dir / tid / f"{tid}{lig_sfx}"

            if not protein.exists() or not ligand.exists():
                results[tid] = {"status": "SKIP", "affinity": "", "runtime": ""}
                continue

            center = get_center_from_pdbqt(str(ligand))
            if center is None:
                results[tid] = {"status": "FAIL", "affinity": "", "runtime": ""}
                fail += 1
                continue

            cx, cy, cz = center
            output_file = outputs_dir / f"{tid}_ligand_out.pdbqt"

            t0 = time.time()
            cmd = [
                str(BINARIES["unidock"]),
                "--receptor", str(protein),
                "--gpu_batch", str(ligand),
                "--scoring", scoring,
                "--search_mode", search_mode,
                "--num_modes", str(num_modes),
                "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                "--size_x", str(box_size), "--size_y", str(box_size), "--size_z", str(box_size),
                "--dir", str(outputs_dir),
                "--verbosity", "0",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            rt = time.time() - t0

            if r.returncode != 0:
                log.write(f"[STDERR {tid}] {r.stderr[:500]}\n")
            log.write(f"[{tid}] exit={r.returncode} rt={rt:.2f}s\n")
            log.flush()

            if r.returncode == 0 and output_file.exists():
                aff = parse_affinity_from_pdbqt(str(output_file))
                results[tid] = {"status": "OK", "affinity": str(aff) if aff else "", "runtime": f"{rt:.2f}"}
                ok += 1
            else:
                results[tid] = {"status": "FAIL", "affinity": "", "runtime": f"{rt:.2f}"}
                fail += 1

            elapsed = time.time() - t_start
            if (i + 1) % 50 == 0:
                remaining = len(targets) - (i + 1)
                avg = elapsed / (i + 1)
                eta = avg * remaining / 60
                print(f"  [{i+1}/{len(targets)}] {elapsed:.0f}s | OK={ok} FAIL={fail} | ETA ~{eta:.0f}min")

    write_summary(summary_path, results)
    wall = time.time() - t_start
    print(f"\nDone: {ok}/{len(targets)} OK, {fail} FAIL, {wall/60:.1f} min")
    return results


def run_unidock_b(dataset_info, exp_dir, search_mode, box_size, num_modes, scoring, chunk_size):
    """Uni-Dock B: paired_batch with chunked execution."""
    data_dir = dataset_info["dir"]
    lig_sfx = dataset_info["ligand_suffix"]
    prot_sfx = dataset_info["protein_suffix"]
    outputs_dir = exp_dir / "outputs"
    configs_dir = exp_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    summary_path = exp_dir / "summary.csv"
    log_path = exp_dir / "unidock.log"

    targets = []
    pairs = {}
    for tid in sorted(os.listdir(str(data_dir))):
        if "/" in lig_sfx:
            ligand = data_dir / tid / lig_sfx.split("/")[-1]
            protein = data_dir / tid / prot_sfx.split("/")[-1]
        else:
            ligand = data_dir / tid / f"{tid}{lig_sfx}"
            protein = data_dir / tid / f"{tid}{prot_sfx}"

        if not ligand.exists() or not protein.exists():
            continue

        center = get_center_from_pdbqt(str(ligand))
        if center is None:
            continue

        cx, cy, cz = center
        config = {
            "center_x": round(cx, 3),
            "center_y": round(cy, 3),
            "center_z": round(cz, 3),
            "size_x": box_size,
            "size_y": box_size,
            "size_z": box_size,
        }
        config_path = configs_dir / f"{tid}.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        pairs[tid] = {
            "protein": str(protein),
            "ligand": str(ligand),
            "ligand_config": str(config_path),
        }
        targets.append(tid)

    print(f"Generated {len(targets)} pairs")

    target_list = sorted(targets)
    chunks = [target_list[i:i + chunk_size] for i in range(0, len(target_list), chunk_size)]
    print(f"Splitting into {len(chunks)} chunks (chunk_size={chunk_size})")

    all_results = {}
    t_start = time.time()

    with open(log_path, "w") as log:
        for ci, chunk in enumerate(chunks):
            chunk_json = {tid: pairs[tid] for tid in chunk}
            chunk_path = configs_dir / f"chunk_{ci:02d}.json"
            with open(chunk_path, "w") as f:
                json.dump(chunk_json, f, indent=2)

            batch_size = min(len(chunk), chunk_size)
            print(f"  Chunk {ci+1}/{len(chunks)}: {len(chunk)} pairs, batch_size={batch_size}")

            cmd = [
                str(BINARIES["unidock"]),
                "--ligand_index", str(chunk_path),
                "--paired_batch_size", str(batch_size),
                "--scoring", scoring,
                "--search_mode", search_mode,
                "--num_modes", str(num_modes),
                "--size_x", str(box_size), "--size_y", str(box_size), "--size_z", str(box_size),
                "--dir", str(outputs_dir),
                "--verbosity", "1",
            ]

            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            rt = time.time() - t0

            log.write(f"[CHUNK {ci}] exit={r.returncode} rt={rt:.1f}s\n")
            log.write(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
            if r.stderr.strip():
                log.write(f"\n[STDERR]\n{r.stderr[-2000:]}")
            log.flush()

            avg_rt = rt / len(chunk)
            if r.returncode != 0:
                print(f"    FAILED (exit={r.returncode}) — check {log_path}")
                for tid in chunk:
                    all_results[tid] = {"status": "FAIL", "affinity": "", "runtime": f"{avg_rt:.2f}"}
                continue

            for tid in chunk:
                if dataset_info["ref_type"] == "ligand_pdbqt":
                    outfile = outputs_dir / f"{tid}_ligand_out.pdbqt"
                else:
                    outfile = outputs_dir / f"flex-xray_out.pdbqt"

                if outfile.exists():
                    aff = parse_affinity_from_pdbqt(str(outfile))
                    all_results[tid] = {"status": "OK", "affinity": str(aff) if aff else "", "runtime": f"{avg_rt:.2f}"}
                else:
                    all_results[tid] = {"status": "NO_OUTPUT", "affinity": "", "runtime": f"{avg_rt:.2f}"}

    write_summary(summary_path, all_results)
    wall = time.time() - t_start
    ok = sum(1 for v in all_results.values() if v["status"] == "OK")
    print(f"\nDone: {ok}/{len(all_results)} OK, {wall/60:.1f} min")
    return all_results


def run_vina_cpu(dataset_info, exp_dir, exhaustiveness, box_size, timeout, max_evals=0, global_steps=0):
    """AutoDock Vina CPU — per-target."""
    data_dir = dataset_info["dir"]
    lig_sfx = dataset_info["ligand_suffix"]
    prot_sfx = dataset_info["protein_suffix"]
    outputs_dir = exp_dir / "outputs"
    configs_dir = exp_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    summary_path = exp_dir / "summary.csv"

    binary = BINARIES["vina-cpu"]
    targets = sorted(os.listdir(str(data_dir)))
    results = {}
    ok = fail = 0

    t_start = time.time()
    for i, tid in enumerate(targets):
        if "/" in lig_sfx:
            protein = data_dir / tid / lig_sfx.split("/")[-1].replace("flex-xray", "protein")
            ligand = data_dir / tid / lig_sfx.split("/")[-1]
        else:
            protein = data_dir / tid / f"{tid}{prot_sfx}"
            ligand = data_dir / tid / f"{tid}{lig_sfx}"

        if not protein.exists() or not ligand.exists():
            results[tid] = {"status": "SKIP", "affinity": "", "runtime": ""}
            continue

        center = get_center_from_pdbqt(str(ligand))
        if center is None:
            results[tid] = {"status": "FAIL", "affinity": "", "runtime": ""}
            fail += 1
            continue

        cx, cy, cz = center
        output_file = outputs_dir / f"{tid}_out.pdbqt"
        config_file = configs_dir / f"{tid}_config.txt"

        with open(config_file, "w") as f:
            f.write(f"receptor = {protein}\n")
            f.write(f"ligand = {ligand}\n")
            f.write(f"center_x = {cx}\ncenter_y = {cy}\ncenter_z = {cz}\n")
            f.write(f"size_x = {box_size}\nsize_y = {box_size}\nsize_z = {box_size}\n")
            f.write(f"out = {output_file}\n")
            f.write(f"exhaustiveness = {exhaustiveness}\n")
            if max_evals > 0:
                f.write(f"max_evals = {max_evals}\n")
            if global_steps > 0:
                f.write(f"global_steps = {global_steps}\n")

        t0 = time.time()
        try:
            r = subprocess.run(
                [str(binary), "--config", str(config_file), "--cpu", "0"],
                capture_output=True, text=True, timeout=timeout,
            )
            rt = time.time() - t0

            if r.returncode == 0 and output_file.exists():
                aff = parse_affinity_from_pdbqt(str(output_file))
                results[tid] = {"status": "OK", "affinity": str(aff) if aff else "", "runtime": f"{rt:.2f}"}
                ok += 1
            else:
                results[tid] = {"status": "FAIL", "affinity": "", "runtime": f"{rt:.2f}"}
                fail += 1
        except subprocess.TimeoutExpired:
            results[tid] = {"status": "TIMEOUT", "affinity": "", "runtime": str(timeout)}
            fail += 1

        elapsed = time.time() - t_start
        eta_str = ""
        if ok + fail > 0:
            remaining = len(targets) - (ok + fail)
            avg = elapsed / (ok + fail)
            eta = avg * remaining / 60
            eta_str = f"ETA ~{eta:.0f}min"
        if (i + 1) % 50 == 0 or (i + 1) == len(targets):
            print(f"  [{i+1}/{len(targets)}] {elapsed:.0f}s | OK={ok} FAIL={fail} | {eta_str}")

    write_summary(summary_path, results)
    wall = time.time() - t_start
    print(f"\nDone: {ok}/{len(targets)} OK, {fail} FAIL, {wall/60:.1f} min")
    return results


def run_qvina_w_cuda(dataset_info, exp_dir, thread, box_size, timeout, search_depth=0):
    """QuickVina-W-CUDA — per-target GPU."""
    data_dir = dataset_info["dir"]
    lig_sfx = dataset_info["ligand_suffix"]
    prot_sfx = dataset_info["protein_suffix"]
    outputs_dir = exp_dir / "outputs"
    configs_dir = exp_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    summary_path = exp_dir / "summary.csv"

    binary = BINARIES["qvina-w-cuda"]
    targets = sorted(os.listdir(str(data_dir)))
    results = {}
    ok = fail = 0

    t_start = time.time()
    for i, tid in enumerate(targets):
        if "/" in lig_sfx:
            protein = data_dir / tid / lig_sfx.split("/")[-1].replace("flex-xray", "protein")
            ligand = data_dir / tid / lig_sfx.split("/")[-1]
        else:
            protein = data_dir / tid / f"{tid}{prot_sfx}"
            ligand = data_dir / tid / f"{tid}{lig_sfx}"

        if not protein.exists() or not ligand.exists():
            results[tid] = {"status": "SKIP", "affinity": "", "runtime": ""}
            continue

        center = get_center_from_pdbqt(str(ligand))
        if center is None:
            results[tid] = {"status": "FAIL", "affinity": "", "runtime": ""}
            fail += 1
            continue

        cx, cy, cz = center
        output_file = outputs_dir / f"{tid}_out.pdbqt"

        t0 = time.time()
        try:
            search_depth_part = f' --search_depth {search_depth}' if search_depth > 0 else ''
            cmd = [
                "sudo", "bash", "-c",
                f'ulimit -s 8192 && exec "{binary}" '
                f'--receptor "{protein}" --ligand "{ligand}" '
                f'--thread {thread}{search_depth_part} '
                f'--center_x {cx} --center_y {cy} --center_z {cz} '
                f'--size_x {box_size} --size_y {box_size} --size_z {box_size} '
                f'--out "{output_file}"',
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            rt = time.time() - t0

            if r.returncode == 0 and output_file.exists():
                aff = parse_affinity_from_pdbqt(str(output_file))
                results[tid] = {"status": "OK", "affinity": str(aff) if aff else "", "runtime": f"{rt:.2f}"}
                ok += 1
            else:
                results[tid] = {"status": "FAIL", "affinity": "", "runtime": f"{rt:.2f}"}
                fail += 1
        except subprocess.TimeoutExpired:
            results[tid] = {"status": "TIMEOUT", "affinity": "", "runtime": str(timeout)}
            fail += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(targets) - (i + 1)) / 60
            print(f"  [{i+1}/{len(targets)}] {elapsed:.0f}s | OK={ok} FAIL={fail} | ETA ~{eta:.0f}min")

    write_summary(summary_path, results)
    wall = time.time() - t_start
    print(f"\nDone: {ok}/{len(targets)} OK, {fail} FAIL, {wall/60:.1f} min")
    return results


def run_cuda_vina(dataset_info, exp_dir, thread, box_size, timeout, search_depth, binary_key, opencl_binary_path=None, workdir=None):
    """Vina-CUDA / QuickVina2-CUDA / Vina-GPU2 — per-target GPU, same CLI as QVina-W-CUDA."""
    data_dir = dataset_info["dir"]
    lig_sfx = dataset_info["ligand_suffix"]
    prot_sfx = dataset_info["protein_suffix"]
    outputs_dir = exp_dir / "outputs"
    configs_dir = exp_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    summary_path = exp_dir / "summary.csv"

    binary = BINARIES[binary_key]
    targets = sorted(os.listdir(str(data_dir)))
    results = {}
    ok = fail = 0

    t_start = time.time()
    for i, tid in enumerate(targets):
        if "/" in lig_sfx:
            protein = data_dir / tid / lig_sfx.split("/")[-1].replace("flex-xray", "protein")
            ligand = data_dir / tid / lig_sfx.split("/")[-1]
        else:
            protein = data_dir / tid / f"{tid}{prot_sfx}"
            ligand = data_dir / tid / f"{tid}{lig_sfx}"

        if not protein.exists() or not ligand.exists():
            results[tid] = {"status": "SKIP", "affinity": "", "runtime": ""}
            continue

        center = get_center_from_pdbqt(str(ligand))
        if center is None:
            results[tid] = {"status": "FAIL", "affinity": "", "runtime": ""}
            fail += 1
            continue

        cx, cy, cz = center
        output_file = outputs_dir / f"{tid}_out.pdbqt"

        t0 = time.time()
        try:
            search_depth_part = f" --search_depth {search_depth}" if search_depth > 0 else ""
            opencl_part = f' --opencl_binary_path "{opencl_binary_path}"' if opencl_binary_path else ""
            cmd_prefix = f'cd "{workdir}" && ulimit -s 8192 && exec "{binary}"' if workdir else f'ulimit -s 8192 && exec "{binary}"'
            cmd = [
                "sudo", "bash", "-c",
                f'{cmd_prefix} '
                f'--receptor "{protein}" --ligand "{ligand}" '
                f"--thread {thread}{search_depth_part}{opencl_part} "
                f"--center_x {cx} --center_y {cy} --center_z {cz} "
                f"--size_x {box_size} --size_y {box_size} --size_z {box_size} "
                f'--out "{output_file}"',
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=workdir)
            rt = time.time() - t0
            gt = parse_gpu_time(r.stderr + r.stdout)

            base = {"gpu_kernel1": gt["k1"], "gpu_kernel2": gt["k2"],
                    "gpu_htod": gt["htod"], "gpu_dtoh": gt["dtoh"],
                    "gpu_memset": gt["memset"], "gpu_all": gt["all"],
                    "e2e_total": gt["e2e_total"], "e2e_warm": gt["e2e_warm"],
                    "e2e_dock": gt["e2e_dock"]}
            if r.returncode == 0 and output_file.exists():
                aff = parse_affinity_from_pdbqt(str(output_file))
                results[tid] = {"status": "OK", "affinity": str(aff) if aff else "", "runtime": f"{rt:.2f}", **base}
                ok += 1
            else:
                results[tid] = {"status": "FAIL", "affinity": "", "runtime": f"{rt:.2f}", **base}
                fail += 1
        except subprocess.TimeoutExpired:
            results[tid] = {"status": "TIMEOUT", "affinity": "", "runtime": str(timeout),
                            "gpu_kernel1": "", "gpu_kernel2": "", "gpu_htod": "", "gpu_dtoh": "",
                            "gpu_memset": "", "gpu_all": ""}
            fail += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(targets) - (i + 1)) / 60
            print(f"  [{i+1}/{len(targets)}] {elapsed:.0f}s | OK={ok} FAIL={fail} | ETA ~{eta:.0f}min")

    write_summary(summary_path, results)
    wall = time.time() - t_start
    print(f"\nDone: {ok}/{len(targets)} OK, {fail} FAIL, {wall/60:.1f} min")

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Screening Power runners
# ══════════════════════════════════════════════════════════════════════════

def _resolve_protein_path(data_dir, target, prot_sfx):
    if "/" in prot_sfx:
        return data_dir / target / prot_sfx.split("/")[-1]
    return data_dir / target / f"{target}{prot_sfx}"


def run_screening_unidock(dataset_info, exp_dir, search_mode, box_size, num_modes, scoring):
    """Uni-Dock screening: one receptor × all coreset ligands (GPU batch).
    57 targets × 285 ligands. One unidock call per target (285 ligands batched)."""
    data_dir = dataset_info["dir"]
    prot_sfx = dataset_info["protein_suffix"]
    target_info = parse_target_info(str(SCREENING_TARGET_INFO))
    all_ligands = get_all_coreset_ligands(data_dir)
    n_ligands = len(all_ligands)

    scores_dir = exp_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    outputs_dir = exp_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    log_path = exp_dir / "unidock.log"

    print(f"Screening targets: {len(target_info)}  |  Ligand pool: {n_ligands}")

    results = {}
    t_start = time.time()

    with open(log_path, "w") as log:
        for i, (target, binders) in enumerate(sorted(target_info.items())):
            # --- resume: skip targets with complete score files ---
            score_file = scores_dir / f"{target}_score.dat"
            if score_file.exists():
                n_lines = sum(1 for line in open(score_file) if not line.startswith('#'))
                if n_lines >= n_ligands:
                    results[target] = {"status": "SKIP_RESUME", "num_docked": n_lines, "runtime": "0"}
                    print(f"  [{i+1}/{len(target_info)}] {target}: SKIP (already {n_lines} scores)")
                    continue
            # --- end resume ---

            protein = _resolve_protein_path(data_dir, target, prot_sfx)
            center_lig = data_dir / target / f"{target}_ligand.pdbqt"

            if not protein.exists() or not center_lig.exists():
                results[target] = {"status": "SKIP", "num_docked": 0, "runtime": ""}
                continue

            center = get_center_from_pdbqt(str(center_lig))
            if center is None:
                results[target] = {"status": "FAIL_CENTER", "num_docked": 0, "runtime": ""}
                continue

            cx, cy, cz = center
            target_out = outputs_dir / target
            target_out.mkdir(exist_ok=True)

            # One receptor + all 285 ligands in a single GPU batch
            ligand_paths = [p for _, p in sorted(all_ligands.items())]

            cmd = [
                str(BINARIES["unidock"]),
                "--receptor", str(protein),
                "--gpu_batch"] + ligand_paths + [
                "--scoring", scoring,
                "--search_mode", search_mode,
                "--num_modes", str(num_modes),
                "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                "--size_x", str(box_size), "--size_y", str(box_size), "--size_z", str(box_size),
                "--dir", str(target_out),
                "--verbosity", "0",
            ]

            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            rt = time.time() - t0

            log.write(f"[{target}] exit={r.returncode} rt={rt:.1f}s\n")
            if r.stderr.strip():
                log.write(f"  stderr: {r.stderr[:500]}\n")
            log.flush()

            # Extract scores from output files
            scores = {}
            for lig_id, lig_path in sorted(all_ligands.items()):
                basename = os.path.splitext(os.path.basename(lig_path))[0]
                outfile = target_out / f"{basename}_out.pdbqt"
                if outfile.exists():
                    aff = parse_affinity_from_pdbqt(str(outfile))
                    if aff is not None:
                        scores[lig_id] = aff

            # Write score file
            write_score_file(str(scores_dir / f"{target}_score.dat"), scores)

            n_docked = len(scores)
            status = "OK" if n_docked >= n_ligands * 0.7 else ("PARTIAL" if n_docked > 0 else "FAIL")
            results[target] = {"status": status, "num_docked": n_docked, "runtime": f"{rt:.2f}"}

            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(target_info) - i - 1) / 60
            print(f"  [{i+1}/{len(target_info)}] {target}: {n_docked}/{n_ligands} docked, "
                  f"{rt:.1f}s | ETA ~{eta:.0f}min")

    write_screening_summary(exp_dir / "summary.csv", results)
    wall = time.time() - t_start
    ok = sum(1 for v in results.values() if v["status"] == "OK")
    total_docked = sum(v["num_docked"] for v in results.values())
    print(f"\nDone: {ok}/{len(results)} targets OK, {total_docked} total dockings, {wall/60:.1f} min")

    run_screening_evaluation(exp_dir)
    return results


def run_screening_unidock_sequential(dataset_info, exp_dir, search_mode, box_size, num_modes, scoring):
    """Uni-Dock screening (non-batch): one receptor × 285 ligands, sequential --ligand calls.
    57 targets × 285 ligands = 16,245 individual docking calls."""
    data_dir = dataset_info["dir"]
    prot_sfx = dataset_info["protein_suffix"]
    target_info = parse_target_info(str(SCREENING_TARGET_INFO))
    all_ligands = get_all_coreset_ligands(data_dir)
    n_ligands = len(all_ligands)

    scores_dir = exp_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    outputs_dir = exp_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    log_path = exp_dir / "unidock.log"

    print(f"Screening targets: {len(target_info)}  |  Ligand pool: {n_ligands}")
    print(f"Total docking jobs: {len(target_info)} × {n_ligands} = {len(target_info) * n_ligands}")

    results = {}
    t_start = time.time()

    with open(log_path, "w") as log:
        for i, (target, binders) in enumerate(sorted(target_info.items())):
            # --- resume: skip targets with complete score files ---
            score_file = scores_dir / f"{target}_score.dat"
            if score_file.exists():
                n_lines = sum(1 for line in open(score_file) if not line.startswith('#'))
                if n_lines >= n_ligands:
                    results[target] = {"status": "SKIP_RESUME", "num_docked": n_lines, "runtime": "0"}
                    print(f"  [{i+1}/{len(target_info)}] {target}: SKIP (already {n_lines} scores)")
                    continue
            # --- end resume ---

            protein = _resolve_protein_path(data_dir, target, prot_sfx)
            center_lig = data_dir / target / f"{target}_ligand.pdbqt"

            if not protein.exists() or not center_lig.exists():
                results[target] = {"status": "SKIP", "num_docked": 0, "runtime": ""}
                continue

            center = get_center_from_pdbqt(str(center_lig))
            if center is None:
                results[target] = {"status": "FAIL_CENTER", "num_docked": 0, "runtime": ""}
                continue

            cx, cy, cz = center
            target_out = outputs_dir / target
            target_out.mkdir(exist_ok=True)

            scores = {}
            t_target = time.time()

            for lig_id, lig_path in sorted(all_ligands.items()):
                output_file = target_out / f"{lig_id}_ligand_out.pdbqt"

                cmd = [
                    str(BINARIES["unidock"]),
                    "--receptor", str(protein),
                    "--gpu_batch", str(lig_path),
                    "--scoring", scoring,
                    "--search_mode", search_mode,
                    "--num_modes", str(num_modes),
                    "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                    "--size_x", str(box_size), "--size_y", str(box_size), "--size_z", str(box_size),
                    "--dir", str(target_out),
                    "--verbosity", "0",
                ]

                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    basename = os.path.splitext(os.path.basename(str(lig_path)))[0]
                    gpu_out = target_out / f"{basename}_out.pdbqt"
                    if r.returncode == 0 and gpu_out.exists():
                        aff = parse_affinity_from_pdbqt(str(gpu_out))
                        if aff is not None:
                            scores[lig_id] = aff
                    else:
                        log.write(f"[{target}/{lig_id}] exit={r.returncode}\n")
                except subprocess.TimeoutExpired:
                    log.write(f"[{target}/{lig_id}] TIMEOUT\n")

            rt_target = time.time() - t_target
            log.write(f"[{target}] {len(scores)}/{n_ligands} docked, {rt_target:.1f}s\n")
            log.flush()

            write_score_file(str(scores_dir / f"{target}_score.dat"), scores)

            n_docked = len(scores)
            status = "OK" if n_docked >= n_ligands * 0.7 else ("PARTIAL" if n_docked > 0 else "FAIL")
            results[target] = {"status": status, "num_docked": n_docked, "runtime": f"{rt_target:.2f}"}

            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(target_info) - i - 1) / 60
            print(f"  [{i+1}/{len(target_info)}] {target}: {n_docked}/{n_ligands} docked, "
                  f"{rt_target:.0f}s | ETA ~{eta:.0f}min")

    write_screening_summary(exp_dir / "summary.csv", results)
    wall = time.time() - t_start
    ok = sum(1 for v in results.values() if v["status"] == "OK")
    total_docked = sum(v["num_docked"] for v in results.values())
    print(f"\nDone: {ok}/{len(results)} targets OK, {total_docked} total dockings, {wall/60:.1f} min")

    run_screening_evaluation(exp_dir)
    return results


def run_screening_vina_cpu(dataset_info, exp_dir, exhaustiveness, box_size, timeout):
    """Vina-CPU screening: one receptor × all coreset ligands (sequential)."""
    data_dir = dataset_info["dir"]
    prot_sfx = dataset_info["protein_suffix"]
    target_info = parse_target_info(str(SCREENING_TARGET_INFO))
    all_ligands = get_all_coreset_ligands(data_dir)
    n_ligands = len(all_ligands)

    binary = BINARIES["vina-cpu"]
    scores_dir = exp_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    outputs_dir = exp_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    configs_dir = exp_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    print(f"Screening targets: {len(target_info)}  |  Ligand pool: {n_ligands}")
    print(f"Total docking jobs: {len(target_info) * n_ligands}")

    results = {}
    t_start = time.time()

    for i, (target, binders) in enumerate(sorted(target_info.items())):
        # --- resume: skip targets with complete score files ---
        score_file = scores_dir / f"{target}_score.dat"
        if score_file.exists():
            n_lines = sum(1 for line in open(score_file) if not line.startswith('#'))
            if n_lines >= n_ligands:
                results[target] = {"status": "SKIP_RESUME", "num_docked": n_lines, "runtime": "0"}
                print(f"  [{i+1}/{len(target_info)}] {target}: SKIP (already {n_lines} scores)")
                continue
        # --- end resume ---

        protein = _resolve_protein_path(data_dir, target, prot_sfx)
        center_lig = data_dir / target / f"{target}_ligand.pdbqt"

        if not protein.exists() or not center_lig.exists():
            results[target] = {"status": "SKIP", "num_docked": 0, "runtime": ""}
            continue

        center = get_center_from_pdbqt(str(center_lig))
        if center is None:
            results[target] = {"status": "FAIL_CENTER", "num_docked": 0, "runtime": ""}
            continue

        cx, cy, cz = center
        target_out = outputs_dir / target
        target_out.mkdir(exist_ok=True)

        scores = {}
        target_rt = 0

        for lig_id, lig_path in sorted(all_ligands.items()):
            output_file = target_out / f"{lig_id}_out.pdbqt"
            config_file = configs_dir / f"{target}_{lig_id}_config.txt"

            with open(config_file, "w") as f:
                f.write(f"receptor = {protein}\n")
                f.write(f"ligand = {lig_path}\n")
                f.write(f"center_x = {cx}\ncenter_y = {cy}\ncenter_z = {cz}\n")
                f.write(f"size_x = {box_size}\nsize_y = {box_size}\nsize_z = {box_size}\n")
                f.write(f"out = {output_file}\n")
                f.write(f"exhaustiveness = {exhaustiveness}\n")

            t0 = time.time()
            try:
                r = subprocess.run(
                    [str(binary), "--config", str(config_file), "--cpu", "0"],
                    capture_output=True, text=True, timeout=timeout,
                )
                rt = time.time() - t0
                target_rt += rt

                if r.returncode == 0 and output_file.exists():
                    aff = parse_affinity_from_pdbqt(str(output_file))
                    if aff is not None:
                        scores[lig_id] = aff
            except subprocess.TimeoutExpired:
                pass

        write_score_file(str(scores_dir / f"{target}_score.dat"), scores)

        n_docked = len(scores)
        status = "OK" if n_docked >= n_ligands * 0.7 else ("PARTIAL" if n_docked > 0 else "FAIL")
        results[target] = {"status": status, "num_docked": n_docked, "runtime": f"{target_rt:.2f}"}

        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(target_info) - i - 1) / 60
        print(f"  [{i+1}/{len(target_info)}] {target}: {n_docked}/{n_ligands} docked, "
              f"{target_rt:.0f}s | ETA ~{eta:.0f}min")

    write_screening_summary(exp_dir / "summary.csv", results)
    wall = time.time() - t_start
    ok = sum(1 for v in results.values() if v["status"] == "OK")
    total_docked = sum(v["num_docked"] for v in results.values())
    print(f"\nDone: {ok}/{len(results)} targets OK, {total_docked} total dockings, {wall/60:.1f} min")

    run_screening_evaluation(exp_dir)
    return results


def run_screening_cuda_vina(dataset_info, exp_dir, thread, box_size, timeout, search_depth,
                            binary_key, opencl_binary_path=None, workdir=None):
    """GPU variant screening: one receptor × all coreset ligands (sequential GPU)."""
    data_dir = dataset_info["dir"]
    prot_sfx = dataset_info["protein_suffix"]
    target_info = parse_target_info(str(SCREENING_TARGET_INFO))
    all_ligands = get_all_coreset_ligands(data_dir)
    n_ligands = len(all_ligands)

    binary = BINARIES[binary_key]
    scores_dir = exp_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    outputs_dir = exp_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    print(f"Screening targets: {len(target_info)}  |  Ligand pool: {n_ligands}")
    print(f"Total docking jobs: {len(target_info) * n_ligands}")

    results = {}
    t_start = time.time()

    for i, (target, binders) in enumerate(sorted(target_info.items())):
        # --- resume: skip targets with complete score files ---
        score_file = scores_dir / f"{target}_score.dat"
        if score_file.exists():
            n_lines = sum(1 for line in open(score_file) if not line.startswith('#'))
            if n_lines >= n_ligands:
                results[target] = {"status": "SKIP_RESUME", "num_docked": n_lines, "runtime": "0"}
                print(f"  [{i+1}/{len(target_info)}] {target}: SKIP (already {n_lines} scores)")
                continue
        # --- end resume ---

        protein = _resolve_protein_path(data_dir, target, prot_sfx)
        center_lig = data_dir / target / f"{target}_ligand.pdbqt"

        if not protein.exists() or not center_lig.exists():
            results[target] = {"status": "SKIP", "num_docked": 0, "runtime": ""}
            continue

        center = get_center_from_pdbqt(str(center_lig))
        if center is None:
            results[target] = {"status": "FAIL_CENTER", "num_docked": 0, "runtime": ""}
            continue

        cx, cy, cz = center
        target_out = outputs_dir / target
        target_out.mkdir(exist_ok=True)

        scores = {}
        target_rt = 0

        for lig_id, lig_path in sorted(all_ligands.items()):
            output_file = target_out / f"{lig_id}_out.pdbqt"

            t0 = time.time()
            try:
                sd_part = f" --search_depth {search_depth}" if search_depth > 0 else ""
                ocl_part = f' --opencl_binary_path "{opencl_binary_path}"' if opencl_binary_path else ""
                wd_part = f'cd "{workdir}" && ' if workdir else ""
                cmd = [
                    "sudo", "bash", "-c",
                    f'{wd_part}ulimit -s 8192 && exec "{binary}" '
                    f'--receptor "{protein}" --ligand "{lig_path}" '
                    f"--thread {thread}{sd_part}{ocl_part} "
                    f"--center_x {cx} --center_y {cy} --center_z {cz} "
                    f"--size_x {box_size} --size_y {box_size} --size_z {box_size} "
                    f'--out "{output_file}"',
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=workdir)
                rt = time.time() - t0
                target_rt += rt

                if r.returncode == 0 and output_file.exists():
                    aff = parse_affinity_from_pdbqt(str(output_file))
                    if aff is not None:
                        scores[lig_id] = aff
            except subprocess.TimeoutExpired:
                pass

        write_score_file(str(scores_dir / f"{target}_score.dat"), scores)

        n_docked = len(scores)
        status = "OK" if n_docked >= n_ligands * 0.7 else ("PARTIAL" if n_docked > 0 else "FAIL")
        results[target] = {"status": status, "num_docked": n_docked, "runtime": f"{target_rt:.2f}"}

        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(target_info) - i - 1) / 60
        print(f"  [{i+1}/{len(target_info)}] {target}: {n_docked}/{n_ligands} docked, "
              f"{target_rt:.0f}s | ETA ~{eta:.0f}min")

    write_screening_summary(exp_dir / "summary.csv", results)
    wall = time.time() - t_start
    ok = sum(1 for v in results.values() if v["status"] == "OK")
    total_docked = sum(v["num_docked"] for v in results.values())
    print(f"\nDone: {ok}/{len(results)} targets OK, {total_docked} total dockings, {wall/60:.1f} min")

    run_screening_evaluation(exp_dir)
    return results


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Unified docking experiment runner")
    p.add_argument("--variant", required=True,
                   choices=["unidock-A", "unidock-B", "vina-cpu", "qvina-w-cuda",
                            "vina-cuda", "qvina2-cuda", "qvina2-cuda-m1-fusion",
                            "m1_fusion_p1_tc",
                            "m1_fusion_p1_b",
                            "m1_fusion_p1_c",
                            "m1_fusion_p1_c_batch",
                    "m1_fusion_p1_warpspec",
                    "m1_fusion_p1_gamma",
                    "gamma_b1_g0a",
                    "gamma_b1_g0a_sg",
                    "gamma_b1_g0a_gs",
                    "gamma_b1_g0a_b1",
                    "v1_lsmax6",
                    "v1_lsmax8",
                    "v1_lsmax8_a2",
                    "v2_d1b",
                    "v2_d1c",
                    "v2_d1d",
                    "v2_d1e",
                    "v2_pscan",
                    "v3_lbfgs8",
                    "v3_qg",
                    "vdeserted1",
                    "v2_adals",
                    "v2_adals",
                    "vdeserted1",
                    "v1_a2_only",
                    "v1_ls_dyn3",
                    "gamma_b1_g0a_pf",
                    "gamma_b1_g0a_b3cb",
                    "m1_fusion_p1_prefetch",
                            "vina-gpu2", "qvina2-gpu2", "vina-up",
                            "vdeserted3",
                            "vdeserted4",
                            "v3",
                            "v4",
                            "v4_5",
                            "v4_7",
                            "v4_8",
                            "v4_8b",
                            "v4_resync",
                            "v4_mixed_fp16",
                            "v5_memcpy_single_m", "fp16_grid_v5", "fp16_sr_grid_v5", "warp_spec",
                            "v5_sigtest",
                            "v6_batch_chunk",
                            "v0",
                            "v1"])
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--mode", default="redocking", choices=["redocking", "screening"],
                   help="Experiment mode: redocking (1:1) or screening (N×M) (default: redocking)")
    p.add_argument("--search-mode", default="fast",
                   choices=["fast", "balance", "detail"],
                   help="Uni-Dock search mode (default: fast)")
    p.add_argument("--scoring", default="vina",
                   choices=["vina", "vinardo", "ad4"],
                   help="Uni-Dock scoring function (default: vina)")
    p.add_argument("--chunk", type=int, default=64,
                   help="Uni-Dock B chunk size (default: 64)")
    p.add_argument("--num-modes", type=int, default=9,
                   help="Number of output poses (default: 9)")
    p.add_argument("--box-size", type=float, default=22.5,
                   help="Docking box edge length in Å (default: 22.5)")
    p.add_argument("--exhaustiveness", type=int, default=8,
                   help="Vina-CPU exhaustiveness (default: 8)")
    p.add_argument("--thread", type=int, default=8192,
                   help="GPU variant thread count (default: 8192)")
    p.add_argument("--search-depth", type=int, default=0,
                   help="GPU variant search depth (0=heuristic, default: 0)")
    p.add_argument("--max-evals", type=int, default=0,
                   help="Vina-CPU max_evals per MC run (0=auto heuristic, default: 0)")
    p.add_argument("--global-steps", type=int, default=0,
                   help="Vina-CPU MC steps per run (overrides heuristic when > 0)")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-target timeout in seconds (default: 600)")
    p.add_argument("--tag", type=str, default="",
                   help="Append tag to experiment directory name")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Override output directory")
    p.add_argument("--no-batch", action="store_true",
                   help="Uni-Dock screening: use sequential --ligand instead of --gpu_batch")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview without executing")
    args = p.parse_args()

    dataset_info = DATASETS[args.dataset]

    # Build experiment name
    timestamp = time.strftime("%b%d_%H%M")
    variant_label = args.variant.replace("-", "_")
    mode_suffix = f"_{args.search_mode}" if args.variant.startswith("unidock") else ""
    screening_suffix = "_screening" if args.mode == "screening" else ""
    exp_name = f"{timestamp}_{args.dataset}_{variant_label}{mode_suffix}{screening_suffix}"

    if args.tag:
        exp_name += f"_{args.tag}"
    if args.out_dir:
        exp_dir = Path(args.out_dir)
    else:
        exp_dir = EXPERIMENTS_DIR / exp_name

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "outputs").mkdir(exist_ok=True)

    # Header
    print("=" * 60)
    print(f"  Variant:  {args.variant}")
    print(f"  Dataset:  {args.dataset}  ({dataset_info['ref_type']})")
    print(f"  Mode:     {args.mode}")
    if args.mode == "screening":
        ti = parse_target_info(str(SCREENING_TARGET_INFO))
        print(f"  Targets:  {len(ti)} screening targets")
        print(f"  Ligands:  coreset pool (all ligands)")
    if args.variant.startswith("unidock"):
        print(f"  Search:   {args.search_mode}  |  num_modes={args.num_modes}  |  scoring={args.scoring}")
    if args.variant == "unidock-B":
        print(f"  Chunk:    {args.chunk}")
    if args.variant == "vina-cpu":
        gs_str = f"  |  global_steps={args.global_steps}" if args.global_steps > 0 else ""
        print(f"  Exhaust:  {args.exhaustiveness}  |  max_evals={args.max_evals}{gs_str}  |  timeout={args.timeout}s")
    if args.variant in ("qvina-w-cuda", "vina-cuda", "qvina2-cuda", "qvina2-cuda-m1-fusion",
                    "m1_fusion_p1_tc",
                    "m1_fusion_p1_b",
                    "m1_fusion_p1_c",
                    "m1_fusion_p1_c_batch",
                    "m1_fusion_p1_warpspec",
                    "m1_fusion_p1_gamma",
                    "gamma_b1_g0a",
                    "gamma_b1_g0a_sg",
                    "gamma_b1_g0a_gs",
                    "gamma_b1_g0a_b1",
                    "v1_lsmax6",
                    "v1_lsmax8",
                    "v1_lsmax8_a2",
                    "v2_d1b",
                    "v2_d1c",
                    "v2_d1d",
                    "v2_d1e",
                    "v2_pscan",
                    "v3_lbfgs8",
                    "v1_a2_only",
                    "v3_qg",
                    "v1_ls_dyn3",
                    "gamma_b1_g0a_pf",
                    "gamma_b1_g0a_b3cb",
                    "m1_fusion_p1_prefetch",
                        "vina-gpu2", "qvina2-gpu2"):
        sd_str = f"  |  search_depth={args.search_depth}" if args.search_depth > 0 else " (heuristic)"
        print(f"  Thread:   {args.thread}{sd_str}  |  timeout={args.timeout}s")
    print(f"  Box:      {args.box_size}³ Å")
    print(f"  Output:   {exp_dir}")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN] No docking executed.")
        return

    # ── Routing ──
    if args.mode == "screening":
        # Screening mode
        if args.variant == "unidock-A":
            if args.no_batch:
                run_screening_unidock_sequential(dataset_info, exp_dir, args.search_mode,
                                                 args.box_size, args.num_modes, args.scoring)
            else:
                run_screening_unidock(dataset_info, exp_dir, args.search_mode,
                                      args.box_size, args.num_modes, args.scoring)
        elif args.variant == "unidock-B":
            print("ERROR: unidock-B (paired_batch) not supported for screening. Use unidock-A.")
            sys.exit(1)
        elif args.variant == "vina-cpu":
            run_screening_vina_cpu(dataset_info, exp_dir, args.exhaustiveness,
                                   args.box_size, args.timeout)
        elif args.variant == "qvina-w-cuda":
            run_screening_cuda_vina(dataset_info, exp_dir, args.thread,
                                     args.box_size, args.timeout, args.search_depth, "qvina-w-cuda")
        elif args.variant == "vina-cuda":
            run_screening_cuda_vina(dataset_info, exp_dir, args.thread,
                                     args.box_size, args.timeout, args.search_depth, "vina-cuda")
        elif args.variant == "qvina2-cuda":
            run_screening_cuda_vina(dataset_info, exp_dir, args.thread,
                                     args.box_size, args.timeout, args.search_depth, "qvina2-cuda")
        elif args.variant == "vina-gpu2":
            run_screening_cuda_vina(dataset_info, exp_dir, args.thread,
                                     args.box_size, args.timeout, args.search_depth, "vina-gpu2",
                                     opencl_binary_path=str(VARIANTS_DIR / "Vina-GPU2"),
                                     workdir=str(VARIANTS_DIR / "Vina-GPU2"))
        elif args.variant == "qvina2-gpu2":
            run_screening_cuda_vina(dataset_info, exp_dir, args.thread,
                                     args.box_size, args.timeout, args.search_depth, "qvina2-gpu2",
                                     workdir=str(VARIANTS_DIR / "QuickVina2-GPU2"))
        else:
            # 通用 fallback: 任何注册在 BINARIES 里、非 unidock/vina-cpu 的 CUDA vina 类变体
            # 自动走 run_cuda_vina(binary_key=args.variant)。以后新增 variant 只需在 BINARIES 注册。
            if args.variant in BINARIES and args.variant not in ("unidock", "vina-cpu"):
                run_screening_cuda_vina(dataset_info, exp_dir, args.thread,
                                         args.box_size, args.timeout, args.search_depth, args.variant)
            else:
                print(f"ERROR: unsupported variant for screening: {args.variant}")
                sys.exit(1)
    else:
        # Redocking mode (existing behavior)
        if args.variant == "unidock-A":
            run_unidock_a(dataset_info, exp_dir, args.search_mode,
                          args.box_size, args.num_modes, args.scoring)
        elif args.variant == "unidock-B":
            run_unidock_b(dataset_info, exp_dir, args.search_mode,
                          args.box_size, args.num_modes, args.scoring, args.chunk)
        elif args.variant == "vina-cpu":
            run_vina_cpu(dataset_info, exp_dir, args.exhaustiveness,
                         args.box_size, args.timeout, args.max_evals, args.global_steps)
        elif args.variant == "qvina-w-cuda":
            run_qvina_w_cuda(dataset_info, exp_dir, args.thread,
                             args.box_size, args.timeout, args.search_depth)
        elif args.variant == "vina-cuda":
            run_cuda_vina(dataset_info, exp_dir, args.thread,
                          args.box_size, args.timeout, args.search_depth, "vina-cuda")
        elif args.variant == "qvina2-cuda":
            run_cuda_vina(dataset_info, exp_dir, args.thread,
                          args.box_size, args.timeout, args.search_depth, "qvina2-cuda")
        elif args.variant == "vina-gpu2":
            run_cuda_vina(dataset_info, exp_dir, args.thread,
                          args.box_size, args.timeout, args.search_depth, "vina-gpu2",
                          opencl_binary_path=str(VARIANTS_DIR / "Vina-GPU2"),
                          workdir=str(VARIANTS_DIR / "Vina-GPU2"))
        elif args.variant == "qvina2-gpu2":
            run_cuda_vina(dataset_info, exp_dir, args.thread,
                          args.box_size, args.timeout, args.search_depth, "qvina2-gpu2",
                          workdir=str(VARIANTS_DIR / "QuickVina2-GPU2"))
        else:
            # 通用 fallback: 任何注册在 BINARIES 里、非 unidock/vina-cpu 的 CUDA vina 类变体
            # 自动走 run_cuda_vina(binary_key=args.variant)。以后新增 variant 只需在 BINARIES 注册。
            if args.variant in BINARIES and args.variant not in ("unidock", "vina-cpu"):
                run_cuda_vina(dataset_info, exp_dir, args.thread,
                              args.box_size, args.timeout, args.search_depth, args.variant)
            else:
                print(f"ERROR: unsupported variant for redocking: {args.variant}")
                sys.exit(1)

        # RMSD analysis (redocking only)
        print("\n--- RMSD Analysis ---")
        run_rmsd_analysis(str(exp_dir), dataset_info)

    print(f"\nOutput: {exp_dir}")
    if args.mode == "screening":
        print(f"  scores/*_score.dat")
        print(f"  summary.csv")
    else:
        print(f"  summary.csv")
        print(f"  rmsd_results.csv")
        print(f"  outputs/*_out.pdbqt")

    return exp_dir


if __name__ == "__main__":
    main()
