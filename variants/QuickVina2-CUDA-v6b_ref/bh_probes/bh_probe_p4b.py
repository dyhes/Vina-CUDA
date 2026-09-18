#!/usr/bin/env python3
# BH-PROBE-P4b: drop extern (single-TU defs, q4_skip_row_g precedent),
# remove per-barrier printf spam, check symbol-address errors, full ledger dump
import sys
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"

# ---- bh_shared.h ----
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()
old = """struct BhGate { unsigned int c0; unsigned int c1; unsigned int flip; };
extern __device__ BhGate d_bh_gates[8];
extern __device__ volatile unsigned int d_bh_poison;
extern __device__ unsigned long long d_bh_last[1048576]; // per-lane: block*1024+tid (1024 blocks)"""
new = """struct BhGate { unsigned int c0; unsigned int c1; unsigned int flip; };
// single-TU definitions (no extern): kernel2.cu is the only includer; extern
// __device__ demotes and breaks the HOST-side symbol binding (P2-P4 lesson).
__device__ BhGate d_bh_gates[8];
__device__ volatile unsigned int d_bh_poison;
__device__ unsigned long long d_bh_last[1048576]; // per-lane: block*1024+tid (1024 blocks)"""
assert old in s, "extern block anchor"
s = s.replace(old, new)

old = """\t\t((volatile unsigned long long*)d_bh_last)[gtid] = (unsigned long long)site;
\t\tif (gtid == 0) printf("BHDEVADDR %p\\n", (void*)d_bh_last);"""
new = """\t\t((volatile unsigned long long*)d_bh_last)[gtid] = (unsigned long long)site;"""
assert old in s, "printf anchor"
s = s.replace(old, new)
open(p, "w").write(s)
print("BH_SHARED-P4b")

# ---- kernel2.cu ----
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
old = """\t{ void *gp, *pp, *lp;
\t  cudaGetSymbolAddress(&gp, d_bh_gates); cudaGetSymbolAddress(&pp, d_bh_poison);
\t  cudaGetSymbolAddress(&lp, d_bh_last);"""
new = """\t{ void *gp = 0, *pp = 0, *lp = 0; cudaError_t e1, e2, e3;
\t  e1 = cudaGetSymbolAddress(&gp, d_bh_gates); e2 = cudaGetSymbolAddress(&pp, d_bh_poison);
\t  e3 = cudaGetSymbolAddress(&lp, d_bh_last);
\t  if (e1 || e2 || e3) fprintf(stderr, "BHSYMERR %d %d %d\\n", e1, e2, e3);
\t  if (!lp) fprintf(stderr, "BHSYMNULL\\n");"""
assert old in s, "symbol block anchor"
s = s.replace(old, new)

old = "\t  fwrite(hlast, 8, 65536, f); fclose(f);"
new = "\t  fwrite(hlast, 8, 1048576, f); fclose(f);"
assert old in s, "fwrite anchor"
s = s.replace(old, new)
open(p, "w").write(s)
print("KERNEL2-P4b")
