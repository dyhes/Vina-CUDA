#!/usr/bin/env python3
"""Compute Top-1 RMSD via spyrmsd (symmetry-corrected Kabsch).

Usage:
    python3 rmsd_analysis.py <experiment_dir>

Requires: spyrmsd, python3-openbabel (on AICloud: apt install python3-openbabel)

Output:
    rmsd_results.csv — target,status,rmsd_all_atom,rmsd_heavy_atom,n_atoms

Reference:
    Meli & Biggin, J. Cheminf. 12, 49 (2020). https://doi.org/10.1186/s13321-020-00455-2
"""

import sys
import csv
from pathlib import Path

from spyrmsd import rmsd as srmsd, io as sio


def find_dataset_dir(exp_dir):
    """Walk up from exp_dir to find the grad/dataset directory."""
    p = exp_dir.resolve()
    while p != p.parent:
        for ds_name in ["dataset/CASF-2016-PDBQT/coreset", "dataset/AutoDockGPU"]:
            candidate = p / ds_name
            if candidate.is_dir():
                return candidate
        p = p.parent
    return None


def resolve_ref(dataset_dir, target, configs_dir):
    """Try multiple reference lookup strategies."""
    # Strategy 1: CASF-2016 coreset/{target}/{target}_ligand.pdbqt
    ref = dataset_dir / target / f"{target}_ligand.pdbqt"
    if ref.exists():
        return ref
    # Strategy 2: AutoDockGPU {target}/flex-xray.pdbqt
    ref = dataset_dir / target / "flex-xray.pdbqt"
    if ref.exists():
        return ref
    # Strategy 3: read from config file
    for suffix in ("_config.txt", ".txt"):
        config_path = configs_dir / f"{target}{suffix}"
        if config_path.exists():
            with open(config_path) as cf:
                for line in cf:
                    stripped = line.strip()
                    if stripped.startswith("ligand"):
                        return Path(stripped.split(" = ", 1)[-1])
    return None


def extract_target_id(filename_stem):
    if filename_stem.endswith("_ligand_out"):
        return filename_stem.replace("_ligand_out", "")
    elif filename_stem.endswith("_out"):
        return filename_stem.replace("_out", "")
    return filename_stem


def main():
    exp_dir = Path(sys.argv[1])
    configs_dir = exp_dir / "configs"
    outputs_dir = exp_dir / "outputs"
    dataset_dir = find_dataset_dir(exp_dir)

    if dataset_dir is None:
        print("ERROR: Cannot find dataset directory.", file=sys.stderr)
        sys.exit(1)

    targets = [(extract_target_id(f.stem), f) for f in sorted(outputs_dir.glob("*.pdbqt"))]
    targets = [(t, p) for t, p in targets if t]

    if not targets:
        print("ERROR: No output PDBQT files found.", file=sys.stderr)
        sys.exit(1)

    results = []
    n_valid = 0
    for target, output_path in targets:
        ref_path = resolve_ref(dataset_dir, target, configs_dir)
        if ref_path is None or not ref_path.exists():
            results.append((target, "NO_REF", "", "", ""))
            continue

        try:
            ref_mols = sio.loadallmols(str(ref_path))
            out_mols = sio.loadallmols(str(output_path))
        except Exception as e:
            results.append((target, "LOAD_ERR", "", "", ""))
            print(f"  {target}: load error — {e}", file=sys.stderr)
            continue

        if not ref_mols or not out_mols:
            results.append((target, "EMPTY", "", "", ""))
            continue

        top1_mol = out_mols[0]
        n_atoms = top1_mol.natoms

        try:
            r_all = srmsd.rmsdwrapper(ref_mols[0], top1_mol, symmetry=True, strip=False)[0]
            r_heavy = srmsd.rmsdwrapper(ref_mols[0], top1_mol, symmetry=True, strip=True)[0]
        except Exception as e:
            results.append((target, "RMSD_ERR", "", "", ""))
            print(f"  {target}: rmsd error — {e}", file=sys.stderr)
            continue

        results.append((target, "OK", round(r_all, 4), round(r_heavy, 4), n_atoms))
        n_valid += 1

    out_path = exp_dir / "rmsd_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "status", "rmsd_all_atom", "rmsd_heavy_atom", "n_atoms"])
        writer.writerows(results)

    print(f"spyrmsd: {n_valid} valid / {len(targets)} total → {out_path}")


if __name__ == "__main__":
    main()