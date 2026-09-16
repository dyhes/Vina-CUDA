#!/usr/bin/env python3
# BH-PROBE-P5: sticky stuck-site ledger at timeout + per-domain poison
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"

# ---- bh_shared.h ----
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()

old = "__device__ unsigned long long d_bh_last[1048576]; // per-lane: block*1024+tid (1024 blocks)"
new = """__device__ unsigned long long d_bh_last[1048576]; // per-lane: block*1024+tid (1024 blocks)
__device__ unsigned long long d_bh_stuck[1048576]; // sticky: site where this lane TIMED OUT waiting"""
assert old in s
s = s.replace(old, new)

old = """\t\tif (lane == 0) atomicAdd(myc, 1U);
\t\t__syncwarp();"""
assert old in s
# (unchanged, anchor only)

old = """\t\tif (d_bh_poison) { __syncwarp(); return; } // poisoned: drain fast"""
new = """\t\tif (d_bh_poison & (1u << dom)) { __syncwarp(); return; } // poisoned (this domain): drain fast"""
assert old in s
s = s.replace(old, new)

old = """\t\tunsigned long long t0 = clock64();
\t\twhile (*fl == f) {
\t\t\tif (d_bh_poison) break;
\t\t\tif (clock64() - t0 > 4000000000ULL) { // ~1.6s: declare stuck
\t\t\t\tif (lane == 0) atomicOr((unsigned int*)&d_bh_poison, 1u << dom);
\t\t\t\tbreak;
\t\t\t}
\t\t}
\t\t__syncwarp();"""
new = """\t\tunsigned long long t0 = clock64();
\t\twhile (*fl == f) {
\t\t\tif (d_bh_poison & (1u << dom)) break;
\t\t\tif (clock64() - t0 > 4000000000ULL) { // ~1.6s: declare stuck
\t\t\t\t((volatile unsigned long long*)d_bh_stuck)[gtid] = (unsigned long long)site; // sticky
\t\t\t\tif (lane == 0) atomicOr((unsigned int*)&d_bh_poison, 1u << dom);
\t\t\t\tbreak;
\t\t\t}
\t\t}
\t\t__syncwarp();"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("BH_SHARED-P5")

# ---- kernel2.cu: also dump stuck ledger ----
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
old = """\t  e2 = cudaGetSymbolAddress(&pp, d_bh_poison); e3 = cudaGetSymbolAddress(&lp, d_bh_last);
\t  if (e2 || e3) fprintf(stderr, "BHSYMERR %d %d\\n", e2, e3);
\t  if (!lp) fprintf(stderr, "BHSYMNULL\\n");"""
new = """\t  void *sp = 0; cudaError_t e2, e3, e4;
\t  e2 = cudaGetSymbolAddress(&pp, d_bh_poison); e3 = cudaGetSymbolAddress(&lp, d_bh_last);
\t  e4 = cudaGetSymbolAddress(&sp, d_bh_stuck);
\t  if (e2 || e3 || e4) fprintf(stderr, "BHSYMERR %d %d %d\\n", e2, e3, e4);
\t  if (!lp || !sp) fprintf(stderr, "BHSYMNULL\\n");"""
assert old in s
s = s.replace(old, new)

old = """\t  FILE *f = fopen("/tmp/bh_ledger.bin", "wb");
\t  fwrite(hlast, 8, 1048576, f); fclose(f);
\t  fprintf(stderr, "BH-GATE ledger dumped\\n");"""
new = """\t  FILE *f = fopen("/tmp/bh_ledger.bin", "wb");
\t  fwrite(hlast, 8, 1048576, f); fclose(f);
\t  static unsigned long long hstuck[1048576];
\t  cudaMemcpy(hstuck, sp, 1048576*8, cudaMemcpyDeviceToHost);
\t  f = fopen("/tmp/bh_stuck.bin", "wb");
\t  fwrite(hstuck, 8, 1048576, f); fclose(f);
\t  unsigned nstuck = 0;
\t  for (unsigned i = 0; i < 1048576; i++) if (hstuck[i]) nstuck++;
\t  fprintf(stderr, "BH-GATE ledgers dumped stuck-lanes=%u\\n", nstuck);"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("KERNEL2-P5")
