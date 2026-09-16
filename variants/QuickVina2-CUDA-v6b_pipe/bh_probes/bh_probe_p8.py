#!/usr/bin/env python3
# BH-PROBE-P8: race-free softbar (no counter resets; monotonic flip; modulo-4
# arrival via atomicAdd old-value). Removes the reset-vs-increment race that
# contaminated P5-P7 stuck data.
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()

old = """\t\tvolatile unsigned int *fl = &d_bh_gates[gb].flip;
\t\tunsigned int f = *fl;
\t\tunsigned int *myc = f ? &d_bh_gates[gb].c1 : &d_bh_gates[gb].c0;
\t\tif (lane == 0) atomicAdd(myc, 1U);
\t\t__syncwarp();
\t\tunsigned int cnt = *myc;
\t\tif (cnt == 4U) { // last-arriving warp: reset + flip (single actor)
\t\t\tif (lane == 0) {
\t\t\t\tatomicExch(&d_bh_gates[gb].c0, 0U);
\t\t\t\tatomicExch(&d_bh_gates[gb].c1, 0U);
\t\t\t\tatomicExch((unsigned int*)fl, f ^ 1U);
\t\t\t}
\t\t\t__syncwarp();
\t\t\treturn;
\t\t}
\t\tunsigned long long t0 = clock64();
\t\twhile (*fl == f) {
\t\t\tif (d_bh_poison & (1u << dom)) break;
\t\t\tif (clock64() - t0 > 4000000000ULL) { // ~1.6s: declare stuck
\t\t\t\t((volatile unsigned long long*)d_bh_stuck)[gtid] = (unsigned long long)site; // sticky
\t\t\t\tif (lane == 0 && atomicExch(&d_bh_snap_done[gb], 1) == 0) {
\t\t\t\t\tconst unsigned base = (blockIdx.x + blockIdx.y * gridDim.x) * 1024 + dom * 128;
\t\t\t\t\tfor (int t = 0; t < 128; t++)
\t\t\t\t\t\td_bh_snap[gb * 128 + t] = ((volatile unsigned long long*)d_bh_last)[base + t];
\t\t\t\t}
\t\t\t\tif (lane == 0) atomicOr((unsigned int*)&d_bh_poison, 1u << dom);
\t\t\t\tbreak;
\t\t\t}
\t\t}
\t\t__syncwarp();"""
new = """\t\tvolatile unsigned int *fl = &d_bh_gates[gb].flip;
\t\tunsigned int f = *fl;
\t\tunsigned int oldc = 0;
\t\tif (lane == 0) oldc = atomicAdd(&d_bh_gates[gb].c0, 1U); // never reset
\t\toldc = __shfl_sync(0xFFFFFFFF, oldc, 0); // broadcast lane0's old value
\t\tif ((oldc & 3U) == 3U) { // 4th arriver of this round: release (monotonic)
\t\t\tif (lane == 0) { __threadfence(); atomicAdd((unsigned int*)fl, 1U); }
\t\t\t__syncwarp();
\t\t\treturn;
\t\t}
\t\tunsigned long long t0 = clock64();
\t\twhile (*fl == f) {
\t\t\tif (d_bh_poison & (1u << dom)) break;
\t\t\tif (clock64() - t0 > 4000000000ULL) { // ~1.6s: declare stuck
\t\t\t\t((volatile unsigned long long*)d_bh_stuck)[gtid] = (unsigned long long)site; // sticky
\t\t\t\tif (lane == 0 && atomicExch(&d_bh_snap_done[gb], 1) == 0) {
\t\t\t\t\tconst unsigned base = (blockIdx.x + blockIdx.y * gridDim.x) * 1024 + dom * 128;
\t\t\t\t\tfor (int t = 0; t < 128; t++)
\t\t\t\t\t\td_bh_snap[gb * 128 + t] = ((volatile unsigned long long*)d_bh_last)[base + t];
\t\t\t\t}
\t\t\t\tif (lane == 0) atomicOr((unsigned int*)&d_bh_poison, 1u << dom);
\t\t\t\tbreak;
\t\t\t}
\t\t}
\t\t__syncwarp();"""
assert old in s, "P8 gate anchor"
s = s.replace(old, new)
open(p, "w").write(s)
print("BH_SHARED-P8 (race-free gate)")
