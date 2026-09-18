#!/usr/bin/env python3
# BH-PROBE-P4c: kernel2.cu readout — add symbol-error checks + fix fwrite size
p = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3/inc/cuda/kernel2.cu"
s = open(p).read()

old = """\t{ unsigned int poison = 0; void *pp, *lp;
\t  cudaGetSymbolAddress(&pp, d_bh_poison); cudaGetSymbolAddress(&lp, d_bh_last);"""
new = """\t{ unsigned int poison = 0; void *pp = 0, *lp = 0; cudaError_t e2, e3;
\t  e2 = cudaGetSymbolAddress(&pp, d_bh_poison); e3 = cudaGetSymbolAddress(&lp, d_bh_last);
\t  if (e2 || e3) fprintf(stderr, "BHSYMERR %d %d\\n", e2, e3);
\t  if (!lp) fprintf(stderr, "BHSYMNULL\\n");"""
assert old in s, "readout symbol anchor"
s = s.replace(old, new)

old = "\t  fwrite(hlast, 8, 65536, f); fclose(f);"
new = "\t  fwrite(hlast, 8, 1048576, f); fclose(f);"
assert old in s, "fwrite anchor"
s = s.replace(old, new)
open(p, "w").write(s)
print("KERNEL2-P4c-OK")
