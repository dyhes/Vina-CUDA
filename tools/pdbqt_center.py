#!/usr/bin/env python3
"""Print geometric center of a PDBQT file: cx cy cz"""
import sys

def pdbqt_center(path):
    xs, ys, zs = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
    if not xs:
        raise ValueError("No atoms found")
    return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)

def main():
    cx, cy, cz = pdbqt_center(sys.argv[1])
    print(f"{cx:.3f} {cy:.3f} {cz:.3f}")

if __name__ == "__main__":
    main()
