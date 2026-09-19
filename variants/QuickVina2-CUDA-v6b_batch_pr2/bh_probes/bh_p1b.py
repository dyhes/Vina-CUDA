#!/usr/bin/env python3
# DECISIVE: P1 (interesting()->true) + P8 softbar. If stuck-lanes==0, the
# interesting()-gated RILC entry branch is the warp-split source.
p = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3/inc/cuda/visited.cuh"
s = open(p).read()
anchor = "\tint len = visited->index;"
probe = ("\tint len = visited->index;\n"
         "#ifdef BH_PROBE_INTERESTING\n"
         "\treturn true; // [BH-PROBE-P1b] uniformize branch under softbar forensics\n"
         "#endif")
if "BH_PROBE_INTERESTING" in s:
    print("ALREADY"); import sys; sys.exit(1)
assert s.count(anchor) == 1
open(p, "w").write(s.replace(anchor, probe))
print("P1b-APPLIED")
