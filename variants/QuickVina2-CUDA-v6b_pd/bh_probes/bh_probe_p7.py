#!/usr/bin/env python3
# BH-PROBE-P7: freeze domain snapshot (128 lanes' last-sites) at FIRST timeout
# of each (block, domain) -> captures the split pairs directly.
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"

p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()
old = """__device__ unsigned long long d_bh_stuck[1048576]; // sticky: site where this lane TIMED OUT waiting"""
new = """__device__ unsigned long long d_bh_stuck[1048576]; // sticky: site where this lane TIMED OUT waiting
__device__ unsigned long long d_bh_snap[8 * 1024 * 128]; // [gb*128 + t] frozen at first timeout
__device__ int d_bh_snap_done[8 * 1024]; // 0 = not snapshotted"""
assert old in s
s = s.replace(old, new)

old = """\t\t\tif (clock64() - t0 > 4000000000ULL) { // ~1.6s: declare stuck
\t\t\t\t((volatile unsigned long long*)d_bh_stuck)[gtid] = (unsigned long long)site; // sticky
\t\t\t\tif (lane == 0) atomicOr((unsigned int*)&d_bh_poison, 1u << dom);
\t\t\t\tbreak;
\t\t\t}"""
new = """\t\t\tif (clock64() - t0 > 4000000000ULL) { // ~1.6s: declare stuck
\t\t\t\t((volatile unsigned long long*)d_bh_stuck)[gtid] = (unsigned long long)site; // sticky
\t\t\t\tif (lane == 0 && atomicExch(&d_bh_snap_done[gb], 1) == 0) {
\t\t\t\t\tconst unsigned base = (blockIdx.x + blockIdx.y * gridDim.x) * 1024 + dom * 128;
\t\t\t\t\tfor (int t = 0; t < 128; t++)
\t\t\t\t\t\td_bh_snap[gb * 128 + t] = ((volatile unsigned long long*)d_bh_last)[base + t];
\t\t\t\t}
\t\t\t\tif (lane == 0) atomicOr((unsigned int*)&d_bh_poison, 1u << dom);
\t\t\t\tbreak;
\t\t\t}"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("BH_SHARED-P7")

p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
old = """\t  void *sp = 0; cudaError_t e4;"""
new = """\t  void *sp = 0, *snp = 0; cudaError_t e4, e5;
\t  e5 = cudaGetSymbolAddress(&snp, d_bh_snap);
\t  if (e5) fprintf(stderr, "BHSNAPERR %d\\n", e5);"""
assert old in s
s = s.replace(old, new)

old = """\t  f = fopen("/tmp/bh_stuck.bin", "wb");
\t  fwrite(hstuck, 8, 1048576, f); fclose(f);"""
new = """\t  f = fopen("/tmp/bh_stuck.bin", "wb");
\t  fwrite(hstuck, 8, 1048576, f); fclose(f);
\t  static unsigned long long hsnap[8 * 1024 * 128];
\t  cudaMemcpy(hsnap, snp, 8 * 1024 * 128 * 8, cudaMemcpyDeviceToHost);
\t  f = fopen("/tmp/bh_snap.bin", "wb");
\t  fwrite(hsnap, 8, 8 * 1024 * 128, f); fclose(f);"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("KERNEL2-P7")
