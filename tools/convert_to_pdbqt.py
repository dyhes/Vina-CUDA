#!/usr/bin/env python3

"""Convert receptor and ligand files into PDBQT format.

This script uses:
- ADFR `prepare_receptor` for receptor conversion
- Meeko `mk_prepare_ligand.py` for ligand conversion

Examples:
  python3 tools/convert_to_pdbqt.py \
    --receptor dataset/CASF-2016/CASF-2016/coreset/1a30/1a30_protein.pdb \
    --ligand-dir dataset/CASF-2016/CASF-2016/coreset/1a30 \
    --output-dir dataset/CASF-2016/CASF-2016/coreset/1a30_pdbqt
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ADFR_BIN = Path("tools/adfr/ADFRsuite-1.0/bin")
DEFAULT_MEEKO = Path.home() / ".local/bin/mk_prepare_ligand.py"


def run_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print("Command failed:", " ".join(cmd), file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        raise RuntimeError("Command execution failed")


def resolve_executable(path_or_name: str) -> str:
    candidate = Path(path_or_name)
    if candidate.exists():
        return str(candidate.resolve())

    resolved = shutil.which(path_or_name)
    if resolved:
        return resolved

    raise FileNotFoundError(f"Executable not found: {path_or_name}")


def convert_receptor(receptor: Path, output_dir: Path, prepare_receptor_cmd: str) -> Path:
    out_file = output_dir / f"{receptor.stem}.pdbqt"
    cmd = [prepare_receptor_cmd, "-r", str(receptor), "-o", str(out_file)]
    print("[receptor]", " ".join(cmd))
    run_command(cmd)
    return out_file


def should_convert_ligand(path: Path, allowed_exts: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in allowed_exts


def convert_ligand(ligand: Path, output_dir: Path, meeko_cmd: str) -> Path:
    out_file = output_dir / f"{ligand.stem}.pdbqt"
    cmd = [meeko_cmd, "-i", str(ligand), "-o", str(out_file)]
    print("[ligand]", " ".join(cmd))
    run_command(cmd)
    return out_file


def collect_ligand_inputs(single_ligand: Path | None, ligand_dir: Path | None, recursive: bool) -> list[Path]:
    allowed_exts = {".sdf", ".mol2", ".pdb", ".pdbqt"}

    ligands: list[Path] = []
    if single_ligand:
        if not single_ligand.exists():
            raise FileNotFoundError(f"Ligand file not found: {single_ligand}")
        ligands.append(single_ligand)

    if ligand_dir:
        if not ligand_dir.exists():
            raise FileNotFoundError(f"Ligand directory not found: {ligand_dir}")
        globber = ligand_dir.rglob("*") if recursive else ligand_dir.glob("*")
        ligands.extend([p for p in globber if should_convert_ligand(p, allowed_exts)])

    # Unique and stable ordering
    unique_sorted = sorted({p.resolve() for p in ligands})
    return unique_sorted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert receptor/ligands to PDBQT with ADFR + Meeko")
    parser.add_argument("--receptor", type=Path, help="Input receptor structure (e.g. .pdb/.mol2/.pdbqt)")
    parser.add_argument("--ligand", type=Path, help="Single ligand file to convert")
    parser.add_argument("--ligand-dir", type=Path, help="Directory containing ligands to batch convert")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for .pdbqt files")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan ligand directory")
    parser.add_argument(
        "--adfr-bin-dir",
        type=Path,
        default=DEFAULT_ADFR_BIN,
        help="Directory containing ADFR binaries (default: tools/adfr/ADFRsuite-1.0/bin)",
    )
    parser.add_argument(
        "--meeko-cmd",
        type=str,
        default=str(DEFAULT_MEEKO),
        help="Path or command name for mk_prepare_ligand.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.receptor and not args.ligand and not args.ligand_dir:
        print("Nothing to do. Provide at least one of --receptor, --ligand, --ligand-dir.", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adfr_bin = args.adfr_bin_dir.resolve()
    prepare_receptor_cmd = resolve_executable(str(adfr_bin / "prepare_receptor"))
    meeko_cmd = resolve_executable(args.meeko_cmd)

    receptor_out: Path | None = None
    if args.receptor:
        receptor_in = args.receptor.resolve()
        if not receptor_in.exists():
            raise FileNotFoundError(f"Receptor file not found: {receptor_in}")
        receptor_out = convert_receptor(receptor_in, output_dir, prepare_receptor_cmd)

    ligand_inputs = collect_ligand_inputs(args.ligand, args.ligand_dir, args.recursive)
    ligand_outs: list[Path] = []
    for lig in ligand_inputs:
        ligand_outs.append(convert_ligand(lig, output_dir, meeko_cmd))

    print("\nConversion completed.")
    if receptor_out:
        print(f"- receptor: {receptor_out}")
    print(f"- ligands converted: {len(ligand_outs)}")
    if ligand_outs:
        preview = ligand_outs[:5]
        print("- sample outputs:")
        for out in preview:
            print(f"  - {out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise