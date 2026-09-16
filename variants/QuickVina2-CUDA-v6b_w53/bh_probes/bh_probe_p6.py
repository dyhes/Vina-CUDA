#!/usr/bin/env python3
# BH-PROBE-P6: per-(block,domain) gates — the P5 global-per-domain gate jams
# under 1024-block contention (cnt==4 overshoot); with per-block gates a
# timeout means a TRUE divergence.
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"

p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()

old = "__device__ BhGate d_bh_gates[8];"
new = "__device__ BhGate d_bh_gates[8 * 1024]; // [block * 8 + domain]"
assert old in s
s = s.replace(old, new)

old = """\t\tvolatile unsigned int *fl = &d_bh_gates[dom].flip;
\t\tunsigned int f = *fl;
\t\tunsigned int *myc = f ? &d_bh_gates[dom].c1 : &d_bh_gates[dom].c0;"""
new = """\t\tconst unsigned gb = (blockIdx.x + blockIdx.y * gridDim.x) * 8 + dom;
\t\tvolatile unsigned int *fl = &d_bh_gates[gb].flip;
\t\tunsigned int f = *fl;
\t\tunsigned int *myc = f ? &d_bh_gates[gb].c1 : &d_bh_gates[gb].c0;"""
assert old in s
s = s.replace(old, new)

old = """\t\t\t\tatomicExch(&d_bh_gates[dom].c0, 0U);
\t\t\t\tatomicExch(&d_bh_gates[dom].c1, 0U);"""
new = """\t\t\t\tatomicExch(&d_bh_gates[gb].c0, 0U);
\t\t\t\tatomicExch(&d_bh_gates[gb].c1, 0U);"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("BH_SHARED-P6")
