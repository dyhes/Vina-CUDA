#!/usr/bin/env python3
# BH-PROBE-P4: ledger sized for 1024 blocks, drop memsets (module zero-init),
# magic + dual-side symbol-address probe
import sys
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"

# ---- bh_shared.h ----
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()
old = "extern __device__ unsigned long long d_bh_last[65536]; // per-lane: block*1024+tid"
new = "extern __device__ unsigned long long d_bh_last[1048576]; // per-lane: block*1024+tid (1024 blocks)"
assert old in s, "ledger decl anchor"
s = s.replace(old, new)

old = """\t\tconst unsigned gtid = (blockIdx.x + blockIdx.y * gridDim.x) * 1024
\t\t                    + (threadIdx.x + blockDim.x * threadIdx.y);
\t\t((volatile unsigned long long*)d_bh_last)[gtid] = (unsigned long long)site;"""
new = """\t\tconst unsigned gtid = (blockIdx.x + blockIdx.y * gridDim.x) * 1024
\t\t                    + (threadIdx.x + blockDim.x * threadIdx.y);
\t\t((volatile unsigned long long*)d_bh_last)[gtid] = (unsigned long long)site;
\t\tif (gtid == 0) printf("BHDEVADDR %p\\n", (void*)d_bh_last);"""
assert old in s, "ledger body anchor"
s = s.replace(old, new)
open(p, "w").write(s)
print("BH_SHARED-P4")

# ---- kernel2.cu ----
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
old_zero = """#ifdef BH_SOFTBAR
\t{ void *gp, *pp, *lp;
\t  cudaGetSymbolAddress(&gp, d_bh_gates); cudaGetSymbolAddress(&pp, d_bh_poison);
\t  cudaGetSymbolAddress(&lp, d_bh_last);
\t  cudaMemsetAsync(gp, 0, sizeof(d_bh_gates)); cudaMemsetAsync(pp, 0, 4);
\t  cudaMemsetAsync(lp, 0, 65536*8); }
#endif
"""
assert old_zero in s, "zero block anchor"
s = s.replace(old_zero, "#ifdef BH_SOFTBAR\n#endif\n")  # drop memsets: single launch/process, module .bss zero-init

old_ro = """\t  static unsigned long long hlast[65536];
\t  cudaMemcpy(hlast, lp, 65536*8, cudaMemcpyDeviceToHost);"""
new_ro = """\t  fprintf(stderr, "BHHOSTADDR %p\\n", lp);
\t  static unsigned long long hlast[1048576];
\t  cudaMemcpy(hlast, lp, 1048576*8, cudaMemcpyDeviceToHost);"""
assert old_ro in s, "readout anchor"
s = s.replace(old_ro, new_ro)
open(p, "w").write(s)
print("KERNEL2-P4")
