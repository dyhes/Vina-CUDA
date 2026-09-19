#!/usr/bin/env python3
# BH-PROBE-P3: per-LANE last-site ledger (volatile) + binary dump
import sys
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()

# per-lane ledger: 1024 threads x 64 blocks
old_arr = "extern __device__ unsigned long long d_bh_last[32768];"
new_arr = "extern __device__ unsigned long long d_bh_last[65536]; // per-lane: block*1024+tid"
assert old_arr in s
s = s.replace(old_arr, new_arr)

old_body = """\t\tconst int dom = conf_id & 7;
\t\tconst int lane = threadIdx.x & 31;
\t\tconst int gwid = (blockIdx.x + blockIdx.y * gridDim.x) * 32
\t\t               + ((threadIdx.x + blockDim.x * threadIdx.y) >> 5);
\t\tif (lane == 0) d_bh_last[gwid] = (unsigned long long)site;
\t\t__syncwarp();"""
new_body = """\t\tconst int dom = conf_id & 7;
\t\tconst int lane = threadIdx.x & 31;
\t\tconst unsigned gtid = (blockIdx.x + blockIdx.y * gridDim.x) * 1024
\t\t                    + (threadIdx.x + blockDim.x * threadIdx.y);
\t\t((volatile unsigned long long*)d_bh_last)[gtid] = (unsigned long long)site;
\t\t__syncwarp();"""
assert old_body in s, "body anchor fail"
s = s.replace(old_body, new_body)
open(p, "w").write(s)
print("BH_SHARED-P3")

# host readout: binary dump instead of site histogram
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
old_zero = "cudaMemsetAsync(lp, 0, sizeof(d_bh_last)); }"
assert old_zero in s
s = s.replace(old_zero, "cudaMemsetAsync(lp, 0, 65536*8); }")

old_ro = """\t  static unsigned long long hlast[32768];
\t  cudaMemcpy(hlast, lp, sizeof(hlast), cudaMemcpyDeviceToHost);
\t  fprintf(stderr, "BH-GATE poison=0x%x\\n", poison);
\t  if (poison) {
\t      for (int dom = 0; dom < 8; dom++) if (poison & (1u << dom))
\t          fprintf(stderr, "BH-GATE stuck-domain %d\\n", dom);
\t  }
\t  int nsites = 0; unsigned long long sites[64]; int cnts[64];
\t  for (unsigned i = 0; i < 32768; i++) {
\t      unsigned long long v = hlast[i]; if (!v) continue;
\t      int k = 0; for (; k < nsites; k++) if (sites[k] == v) break;
\t      if (k == nsites && nsites < 64) { sites[nsites] = v; cnts[nsites] = 0; nsites++; }
\t      if (k < 64) cnts[k]++;
\t  }
\t  for (int k = 0; k < nsites; k++)
\t      fprintf(stderr, "BH-GATE last-site %llu warps %d\\n", sites[k], cnts[k]);"""
new_ro = """\t  static unsigned long long hlast[65536];
\t  cudaMemcpy(hlast, lp, 65536*8, cudaMemcpyDeviceToHost);
\t  fprintf(stderr, "BH-GATE poison=0x%x\\n", poison);
\t  if (poison) {
\t      for (int dom = 0; dom < 8; dom++) if (poison & (1u << dom))
\t          fprintf(stderr, "BH-GATE stuck-domain %d\\n", dom);
\t  }
\t  FILE *f = fopen("/tmp/bh_ledger.bin", "wb");
\t  fwrite(hlast, 8, 65536, f); fclose(f);
\t  fprintf(stderr, "BH-GATE ledger dumped\\n");"""
assert old_ro in s, "readout anchor fail"
s = s.replace(old_ro, new_ro)
open(p, "w").write(s)
print("KERNEL2-P3")
