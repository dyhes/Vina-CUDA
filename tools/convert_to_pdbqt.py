#!/usr/bin/env python3

"""Convert receptor and ligand files into PDBQT format.

This script uses:
- ADFR `prepare_receptor` for receptor conversion, with post-cleaning
- OpenBabel `obabel` for ligand conversion (default), or Meeko via --ligand-tool

Examples:
  # Single target
  python3 tools/convert_to_pdbqt.py \\
    --receptor dataset/CASF-2016/coreset/1a30/1a30_protein.pdb \\
    --ligand dataset/CASF-2016/coreset/1a30/1a30_ligand.mol2 \\
    --output-dir dataset/CASF-2016-PDBQT/coreset/1a30

  # Batch: re-convert missing targets
  for id in $(comm -23 /tmp/all.txt /tmp/done.txt); do
    python3 tools/convert_to_pdbqt.py \\
      --receptor dataset/CASF-2016/coreset/$id/${id}_protein.pdb \\
      --ligand dataset/CASF-2016/coreset/$id/${id}_ligand.mol2 \\
      --output-dir dataset/CASF-2016-PDBQT/coreset/$id
  done
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ADFR_BIN = Path("tools/adfr/ADFRsuite-1.0/bin")

# Non-standard tags that ADFR prepare_receptor may produce but Vina-GPU
# parse_pdbqt_rigid() does not recognize.  These tree-structure markers
# (ROOT/BRANCH/ENDBRANCH/ENDROOT) belong in ligand PDBQT, not protein PDBQT.
_RECEPTOR_STRIP_TAGS = {"ROOT", "BRANCH", "ENDBRANCH", "ENDROOT", "TORSDOF"}

# obabel is shipped inside ADFRsuite and its ADS_ROOT is already correct.
_DEFAULT_OBABEL = DEFAULT_ADFR_BIN / "obabel"


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


def clean_protein_pdbqt(path: Path) -> int:
    """Strip ROOT/BRANCH/ENDBRANCH/ENDROOT/TORSDOF lines from a receptor PDBQT.

    ADFR ``prepare_receptor`` occasionally emits tree-structure tags from its
    internal ligand-processing pipeline, especially for large proteins.  Vina-GPU
    ``parse_pdbqt_rigid()`` only accepts REMARK / ATOM / HETATM / TER / WARNING.

    Returns the number of lines stripped.
    """
    keep_lines: list[str] = []
    stripped = 0
    with open(path) as fh:
        for line in fh:
            tag = line.split(maxsplit=1)[0] if line.strip() else ""
            if tag in _RECEPTOR_STRIP_TAGS:
                stripped += 1
            else:
                keep_lines.append(line)

    if stripped:
        with open(path, "w") as fh:
            fh.writelines(keep_lines)

    return stripped


def convert_receptor(receptor: Path, output_dir: Path, prepare_receptor_cmd: str) -> Path:
    out_file = output_dir / f"{receptor.stem}.pdbqt"
    cmd = [prepare_receptor_cmd, "-r", str(receptor), "-o", str(out_file)]
    print("[receptor]", " ".join(cmd))
    run_command(cmd)

    stripped = clean_protein_pdbqt(out_file)
    if stripped:
        print(f"[receptor] stripped {stripped} non-standard tag(s) from {out_file.name}")

    return out_file


def should_convert_ligand(path: Path, allowed_exts: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in allowed_exts


def convert_ligand_obabel(ligand: Path, output_dir: Path, obabel_cmd: str) -> Path:
    """Convert a ligand to PDBQT using OpenBabel.

    obabel auto-detects input format from extension; "-opdbqt" produces
    standard AutoDock atom types compatible with Vina-GPU.
    """
    out_file = output_dir / f"{ligand.stem}.pdbqt"
    cmd = [obabel_cmd, str(ligand), "-opdbqt", "-O", str(out_file)]
    print("[ligand]", " ".join(cmd))
    run_command(cmd)
    return out_file


def convert_ligand_meeko(ligand: Path, output_dir: Path, meeko_cmd: str) -> Path:
    """Convert a ligand to PDBQT using Meeko."""
    out_file = output_dir / f"{ligand.stem}.pdbqt"
    cmd = [meeko_cmd, "-i", str(ligand), "-o", str(out_file)]
    print("[ligand]", " ".join(cmd))
    run_command(cmd)
    return out_file


def collect_ligand_inputs(
    single_ligand: Path | None, ligand_dir: Path | None, recursive: bool
) -> list[Path]:
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
    parser = argparse.ArgumentParser(
        description="Convert receptor/ligands to PDBQT (ADFR + obabel/Meeko)"
    )
    parser.add_argument(
        "--receptor", type=Path,
        help="Input receptor structure (e.g. .pdb/.mol2/.pdbqt)",
    )
    parser.add_argument(
        "--ligand", type=Path,
        help="Single ligand file to convert",
    )
    parser.add_argument(
        "--ligand-dir", type=Path,
        help="Directory containing ligands to batch convert",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for .pdbqt files",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively scan ligand directory",
    )
    parser.add_argument(
        "--adfr-bin-dir", type=Path, default=DEFAULT_ADFR_BIN,
        help=f"Directory containing ADFR binaries (default: {DEFAULT_ADFR_BIN})",
    )
    parser.add_argument(
        "--ligand-tool", type=str, choices=["obabel", "meeko"], default="obabel",
        help="Ligand conversion tool: obabel (bundled, default) or meeko",
    )
    parser.add_argument(
        "--obabel-cmd", type=str, default=str(_DEFAULT_OBABEL),
        help=f"Path to obabel (default: {_DEFAULT_OBABEL})",
    )
    parser.add_argument(
        "--meeko-cmd", type=str, default=str(Path.home() / ".local/bin/mk_prepare_ligand.py"),
        help="Path to mk_prepare_ligand.py (only used when --ligand-tool=meeko)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.receptor and not args.ligand and not args.ligand_dir:
        print(
            "Nothing to do. Provide at least one of --receptor, --ligand, --ligand-dir.",
            file=sys.stderr,
        )
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    receptor_out: Path | None = None
    if args.receptor:
        adfr_bin = args.adfr_bin_dir.resolve()
        prepare_receptor_cmd = resolve_executable(str(adfr_bin / "prepare_receptor"))
        receptor_in = args.receptor.resolve()
        if not receptor_in.exists():
            raise FileNotFoundError(f"Receptor file not found: {receptor_in}")
        receptor_out = convert_receptor(receptor_in, output_dir, prepare_receptor_cmd)

    ligand_inputs = collect_ligand_inputs(args.ligand, args.ligand_dir, args.recursive)
    ligand_outs: list[Path] = []

    if args.ligand_tool == "meeko":
        meeko_cmd = resolve_executable(args.meeko_cmd)
        for lig in ligand_inputs:
            ligand_outs.append(convert_ligand_meeko(lig, output_dir, meeko_cmd))
    else:
        obabel_cmd = resolve_executable(args.obabel_cmd)
        for lig in ligand_inputs:
            ligand_outs.append(convert_ligand_obabel(lig, output_dir, obabel_cmd))

    print("\nConversion completed.")
    if receptor_out:
        print(f"- receptor: {receptor_out}")
    print(f"- ligands converted ({args.ligand_tool}): {len(ligand_outs)}")
    if ligand_outs:
        for out in ligand_outs[:5]:
            print(f"  - {out}")
        if len(ligand_outs) > 5:
            print(f"  ... and {len(ligand_outs) - 5} more")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
