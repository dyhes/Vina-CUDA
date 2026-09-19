#!/usr/bin/env python3
# v3 fix: BH_NBLOCKS was a P12-era macro (not in clean tree). Use a fixed
# upper bound for the drain bitset + escape counter.
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()
old = "__device__ unsigned int d_bh_drain[BH_NBLOCKS];      // per-block drained-domain bitset (escape)"
new = "__device__ unsigned int d_bh_drain[1024];            // per-block drained-domain bitset (escape; GRID_DIM1*2<=1024)"
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)

p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
old = "\t  cudaMemsetAsync(dp, 0, sizeof(d_bh_drain)); cudaMemsetAsync(ep, 0, 8); }"
assert s.count(old) == 1
s = s.replace(old, "\t  cudaMemsetAsync(dp, 0, 1024*4); cudaMemsetAsync(ep, 0, 8); }")
open(p, "w").write(s)
print("v3a-OK")
