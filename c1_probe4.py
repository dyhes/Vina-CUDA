#!/usr/bin/env python3
"""C-1 probe: 12 targets, pool=4, capture stderr to diagnose FAIL cause."""
import os, subprocess, csv
from pathlib import Path
ROOT = Path("/home/ubuntu/grad")
BIN = ROOT/"variants/QuickVina2-CUDA-m1_fusion_p1_c/QuickVina2-GPU-2-1-CUDA"
CORET = ROOT/"dataset/CASF-2016-PDBQT/coreset"
OUT = ROOT/"experiments/Jul18_C1_concurrent_casf_p1_c/probe4"
OUT.mkdir(parents=True, exist_ok=True)
POOL=4; BOX=23; THREAD=8192
def center_of(lig):
    for line in open(lig):
        if line.startswith(("ATOM","HETATM")):
            p=line.split(); return float(p[6]),float(p[7]),float(p[8])
    return None
targets=sorted(p for p in os.listdir(CORET) if (CORET/p).is_dir())[:12]
procs={}
def wait_one():
    pid,_=os.wait()
    for k,v in list(procs.items()):
        if v[0].pid==pid: del procs[k]; return k
for tid in targets:
    d=CORET/tid; prot=d/f"{tid}_protein.pdbqt"; lig=d/f"{tid}_ligand.pdbqt"
    c=center_of(lig); cx,cy,cz=c
    outf=OUT/f"{tid}_out.pdbqt"; errf=OUT/f"{tid}.err"
    cmd=[str(BIN),"--receptor",str(prot),"--ligand",str(lig),"--thread",str(THREAD),
         "--center_x",str(cx),"--center_y",str(cy),"--center_z",str(cz),
         "--size_x",str(BOX),"--size_y",str(BOX),"--size_z",str(BOX),"--out",str(outf)]
    while len(procs)>=POOL: wait_one()
    procs[tid]=(subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=open(errf,"w")),errf)
while procs: wait_one()
ok=0
for tid in targets:
    outf=OUT/f"{tid}_out.pdbqt"
    if outf.exists() and outf.stat().st_size>0: ok+=1
    else:
        errf=OUT/f"{tid}.err"
        print(f"FAIL {tid}: {(errf.read_text()[:200] if errf.exists() else 'no err')}")
print(f"PROBE4 OK={ok}/{len(targets)}")
