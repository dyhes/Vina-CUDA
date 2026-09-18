#!/usr/bin/env python3
# BH-PROBE-P9: per-BLOCK poison (a jam in block7/dom6 must not release
# block3/dom6) — removes the cross-block observation cascade that polluted
# P5-P8 snapshots.
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()

old = "__device__ volatile unsigned int d_bh_poison;"
new = "__device__ unsigned int d_bh_poisonb[1024]; // per-block bitset of jammed domains"
assert old in s
s = s.replace(old, new)

old = "\t\tif (d_bh_poison & (1u << dom)) { __syncwarp(); return; } // poisoned (this domain): drain fast"
new = "\t\tif (d_bh_poisonb[bid] & (1u << dom)) { __syncwarp(); return; } // poisoned (this block+domain): drain fast"
assert old in s
s = s.replace(old, new)

old = """\t\t\tif (d_bh_poison & (1u << dom)) break;"""
new = """\t\t\tif (d_bh_poisonb[bid] & (1u << dom)) break;"""
assert old in s
s = s.replace(old, new)

old = "\t\t\t\tif (lane == 0) atomicOr((unsigned int*)&d_bh_poison, 1u << dom);"
new = "\t\t\t\tif (lane == 0) atomicOr(&d_bh_poisonb[bid], 1u << dom);"
assert old in s
s = s.replace(old, new)

# bid needs defining before first use (top of softbar block)
old = """\t\tconst int dom = conf_id & 7;
\t\tconst int lane = threadIdx.x & 31;"""
new = """\t\tconst int dom = conf_id & 7;
\t\tconst unsigned bid = blockIdx.x + blockIdx.y * gridDim.x;
\t\tconst int lane = threadIdx.x & 31;"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("BH_SHARED-P9")

# host: read the array
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
old = """\t{ unsigned int poison = 0; void *pp = 0, *lp = 0; cudaError_t e2, e3;
\t  e2 = cudaGetSymbolAddress(&pp, d_bh_poison); e3 = cudaGetSymbolAddress(&lp, d_bh_last);"""
new = """\t{ unsigned int hpoison[1024] = {0}; void *pp = 0, *lp = 0; cudaError_t e2, e3;
\t  e2 = cudaGetSymbolAddress(&pp, d_bh_poisonb); e3 = cudaGetSymbolAddress(&lp, d_bh_last);"""
assert old in s
s = s.replace(old, new)
old = """\t  cudaMemcpy(&poison, pp, 4, cudaMemcpyDeviceToHost);
\t  fprintf(stderr, "BH-GATE poison=0x%x\\n", poison);
\t  if (poison) {
\t      for (int dom = 0; dom < 8; dom++) if (poison & (1u << dom))
\t          fprintf(stderr, "BH-GATE stuck-domain %d\\n", dom);
\t  }"""
new = """\t  cudaMemcpy(hpoison, pp, sizeof(hpoison), cudaMemcpyDeviceToHost);
\t  int nstuckblocks = 0;
\t  for (int i = 0; i < 1024; i++) if (hpoison[i]) nstuckblocks++;
\t  fprintf(stderr, "BH-GATE jammed-blocks=%d\\n", nstuckblocks);"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("KERNEL2-P9")
