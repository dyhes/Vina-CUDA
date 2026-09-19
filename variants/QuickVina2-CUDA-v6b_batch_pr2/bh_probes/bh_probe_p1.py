#!/usr/bin/env python3
# BH-PROBE-P1: macro-gated uniformization of interesting() branch (bisect aid)
import sys
p = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3/inc/cuda/visited.cuh"
src = open(p).read()
anchor = "\tint len = visited->index;"
probe = ("\tint len = visited->index;\n"
         "#ifdef BH_PROBE_INTERESTING\n"
         "\treturn true; // [BH-PROBE-P1] uniformize branch: bisect divergence-fed deadlock\n"
         "#endif")
if "BH_PROBE_INTERESTING" in src:
    print("ALREADY-PATCHED"); sys.exit(1)
if src.count(anchor) != 1:
    print("ANCHOR-NOT-UNIQUE:", src.count(anchor)); sys.exit(1)
open(p, "w").write(src.replace(anchor, probe))
print("PATCHED-P1")
