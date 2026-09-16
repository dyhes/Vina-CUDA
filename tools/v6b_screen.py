#!/usr/bin/env python3
# v6b screening driver: 57 CASF-2016 screening targets x 285 coreset ligands.
# Per target: ONE process with --ligand_directory (v6b batch path: N async
# launches + single tail sync, entry-gap amortized). Scores from the FIRST
# "REMARK VINA RESULT" of each <lig>_out.pdbqt. Chunked batch (default 96),
# per-target score file + failure isolation with chunk-size fallback.
# Usage: python3 v6b_screen.py <binary> <expdir> [chunk] [seed-dir-tag]
import os, re, sys, time, shutil
from pathlib import Path

BIN = Path(sys.argv[1])
EXP = Path(sys.argv[2])
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 96
GRAD = Path("/home/ubuntu/grad")
DATA = GRAD / "dataset/CASF-2016-PDBQT/coreset"
TINFO = GRAD / "dataset/CASF-2016/power_screening/TargetInfo.dat"

sys.path.insert(0, str(GRAD / "tools"))
from pdbqt_center import pdbqt_center

SCORES = EXP / "scores"; SCORES.mkdir(parents=True, exist_ok=True)
OUTPUTS = EXP / "outputs"; OUTPUTS.mkdir(exist_ok=True)
LOG = EXP / "v6b_screen.log"

def parse_targets():
    targets = {}
    for line in open(TINFO):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        codes = line.split()
        targets[codes[0]] = codes[1:]
    return targets

def ligand_affinity(p):
    try:
        for line in open(p):
            if line.startswith("REMARK VINA RESULT:"):
                return float(line.split()[3])
    except OSError:
        pass
    return None

def run_one(protein, workdir, outdir, cx, cy, cz, tag):
    cmd = ("ulimit -s 8192 && exec %s --receptor '%s' --ligand_directory '%s' --thread 8192 "
           "--center_x %.3f --center_y %.3f --center_z %.3f "
           "--size_x 20 --size_y 20 --size_z 20 --output_directory '%s'"
           % (BIN, protein, workdir, cx, cy, cz, outdir))
    logf = EXP / ("log_%s.txt" % tag)
    r = os.system("timeout 7200 bash -c \"%s\" > %s 2>&1" % (cmd, logf))
    return os.waitstatus_to_exitcode(r) if os.WIFEXITED(r) else -999

def main():
    targets = parse_targets()
    # ligand pool: coreset dirs with <code>_ligand.pdbqt
    pool = {}
    for d in sorted(os.listdir(DATA)):
        lp = DATA / d / ("%s_ligand.pdbqt" % d)
        if lp.exists():
            pool[d] = lp
    n_pool = len(pool)
    print("targets=%d pool=%d chunk=%d" % (len(targets), n_pool, CHUNK), flush=True)

    t_start = time.time()
    done = 0
    for ti, (target, binders) in enumerate(sorted(targets.items())):
        score_file = SCORES / ("%s_score.dat" % target)
        if score_file.exists():
            n_have = sum(1 for l in open(score_file) if not l.startswith("#"))
            if n_have >= n_pool:
                done += 1
                print("[%d/%d] %s SKIP (resume %d)" % (ti+1, len(targets), target, n_have), flush=True)
                continue
        prot = DATA / target / ("%s_protein.pdbqt" % target)
        clig = pool.get(target)
        if not prot.exists() or clig is None:
            print("[%d/%d] %s SKIP files" % (ti+1, len(targets), target), flush=True)
            continue
        cx, cy, cz = pdbqt_center(str(clig))
        t_out = OUTPUTS / target; t_out.mkdir(exist_ok=True)
        t0 = time.time()

        codes = sorted(pool.keys())
        affs = {}
        cs = 0
        for cs in range(0, n_pool, CHUNK):
            chunk = codes[cs:cs+CHUNK]
            wdir = EXP / ("work_%s_%d" % (target, cs))
            if wdir.exists(): shutil.rmtree(wdir)
            wdir.mkdir()
            for k, c in enumerate(chunk):
                shutil.copy(pool[c], wdir / ("lig%05d_%s.pdbqt" % (k, c)))
            odir = wdir / "out"; odir.mkdir()
            tag = "%s_%d" % (target, cs)
            rc = run_one(prot, wdir, odir, cx, cy, cz, tag)
            if rc != 0 and len(chunk) > 12:
                # failure isolation: halve the chunk (exception-prone ligand wedges its block)
                mid = len(chunk)//2
                for half in (chunk[:mid], chunk[mid:]):
                    w2 = EXP / ("work_%s_h%d_%d" % (target, cs, len(half)))
                    if w2.exists(): shutil.rmtree(w2)
                    w2.mkdir()
                    for k, c in enumerate(half):
                        shutil.copy(pool[c], w2 / ("lig%05d_%s.pdbqt" % (k, c)))
                    o2 = w2 / "out"; o2.mkdir()
                    run_one(prot, w2, o2, cx, cy, cz, "%s_h%d_%d" % (target, cs, len(half)))
                    for f in o2.glob("*.pdbqt"):
                        shutil.move(str(f), t_out / f.name)
                    shutil.rmtree(w2)
            else:
                for f in odir.glob("*.pdbqt"):
                    shutil.move(str(f), t_out / f.name)
                shutil.rmtree(wdir, ignore_errors=True)

        # collect scores — outputs carry the chunk-local index prefix: lig%05d_<code>_out.pdbqt
        for c in codes:
            hits = list(t_out.glob("*_%s_out.pdbqt" % c))
            a = ligand_affinity(hits[0]) if hits else None
            if a is not None:
                affs[c] = a
        with open(score_file, "w") as f:
            f.write("# %s ligand_num score\n" % target)
            for c in codes:
                if c in affs:
                    f.write("%s %s\n" % (c, affs[c]))
        rt = time.time() - t0
        nd = len(affs)
        status = "OK" if nd >= n_pool*0.7 else ("PARTIAL" if nd > 0 else "FAIL")
        print("[%d/%d] %s %s %d/%d %.1fs" % (ti+1, len(targets), target, status, nd, n_pool, rt), flush=True)
        with open(LOG, "a") as f:
            f.write("%s %s %d/%d %.1fs\n" % (target, status, nd, n_pool, rt))
    wall = time.time() - t_start
    print("DONE %d targets in %.1f min" % (len(targets), wall/60), flush=True)

main()
