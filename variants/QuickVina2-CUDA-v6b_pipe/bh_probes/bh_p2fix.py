#!/usr/bin/env python3
# P2-fix: remove explicit __device__ defs (header externs already demote to per-TU defs)
p = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3/inc/cuda/kernel2.cu"
s = open(p).read()
block = """#ifdef BH_SOFTBAR
__device__ BhGate d_bh_gates[8];
__device__ volatile unsigned int d_bh_poison;
__device__ unsigned long long d_bh_last[32768];
#endif
"""
assert block in s, "defs block not found"
open(p, "w").write(s.replace(block, ""))
print("DEFS-REMOVED")
