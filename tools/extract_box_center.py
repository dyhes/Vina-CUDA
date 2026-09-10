#!/usr/bin/env python3
"""Extract geometric center from a mol2 or PDB file, print center_x center_y center_z."""

import sys
from pathlib import Path

def mol2_center(path: Path) -> tuple[float, float, float]:
    xs, ys, zs = [], [], []
    in_atoms = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if in_atoms and line.startswith("@<TRIPOS>"):
                break
            if in_atoms and line:
                parts = line.split()
                if len(parts) >= 4:
                    xs.append(float(parts[2]))
                    ys.append(float(parts[3]))
                    zs.append(float(parts[4]))
    if not xs:
        raise ValueError(f"No atoms found in {path}")
    return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)

def main():
    path = Path(sys.argv[1])
    cx, cy, cz = mol2_center(path)
    # Vina expects 3 decimal places
    print(f"{cx:.3f} {cy:.3f} {cz:.3f}")

if __name__ == "__main__":
    main()
