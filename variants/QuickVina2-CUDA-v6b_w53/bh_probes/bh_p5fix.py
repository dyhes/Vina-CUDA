#!/usr/bin/env python3
# P5-fix: remove duplicate e2/e3 declaration (P4c already declared them)
p = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3/inc/cuda/kernel2.cu"
s = open(p).read()
old = """\t  void *sp = 0; cudaError_t e2, e3, e4;
\t  e2 = cudaGetSymbolAddress(&pp, d_bh_poison); e3 = cudaGetSymbolAddress(&lp, d_bh_last);"""
new = """\t  void *sp = 0; cudaError_t e4;
\t  e2 = cudaGetSymbolAddress(&pp, d_bh_poison); e3 = cudaGetSymbolAddress(&lp, d_bh_last);"""
assert old in s, "decl anchor"
s = s.replace(old, new)
open(p, "w").write(s)
print("P5FIX-OK")
