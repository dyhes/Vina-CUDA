#!/usr/bin/env python3
# BH-PROBE-P2: instrumented software domain barrier (site-tagged, poison escape)
# site id = file_id*100000 + __LINE__  (1=kernel2.cu, 2=quasi_newton.cuh, 3=mutate via macro in bh_shared.h)
import sys
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"

def rd(p): return open(p).read()
def wr(p, s): open(p, "w").write(s)

# ---------- 1. revert P1 ----------
p = Q + "/inc/cuda/visited.cuh"
s = rd(p)
p1 = ("\tint len = visited->index;\n"
      "#ifdef BH_PROBE_INTERESTING\n"
      "\treturn true; // [BH-PROBE-P1] uniformize branch: bisect divergence-fed deadlock\n"
      "#endif")
if p1 in s:
    s = s.replace(p1, "\tint len = visited->index;")
    wr(p, s); print("P1-REVERTED")
else:
    print("P1-NOT-PRESENT (ok)")

# ---------- 2. bh_shared.h: two-arg conf_barrier + softbar ----------
p = Q + "/inc/cuda/bh_shared.h"
s = rd(p)
old = """__device__ __forceinline__ void conf_barrier(int conf_id) {
\t__syncwarp(); // [Q7-BH] reconverge warp (thread0 gates diverge it)
\t__asm__ volatile("bar.sync %0, %1;" :: "r"(conf_id + 1), "n"(BH_TPB));
}
__device__ __forceinline__ void conf_barrier_m(int conf_id) {
\tconf_barrier(conf_id);
}"""
new = """#ifdef BH_SOFTBAR
// [BH-PROBE-P2] instrumented software domain barrier: count-based gate in
// global memory, per-warp last-site ledger, poison-timeout escape. The
// software barrier cannot itself deadlock: stuck warps escape after ~1.6s
// and the kernel completes, letting the host dump d_bh_last.
struct BhGate { unsigned int c0; unsigned int c1; unsigned int flip; };
extern __device__ BhGate d_bh_gates[8];
extern __device__ volatile unsigned int d_bh_poison;
extern __device__ unsigned long long d_bh_last[32768];
#endif
#define conf_barrier_m(conf_id) conf_barrier((conf_id), 300000 + __LINE__)
__device__ __forceinline__ void conf_barrier(int conf_id, int site = 0) {
\t__syncwarp(); // [Q7-BH] reconverge warp (thread0 gates diverge it)
#ifndef BH_SOFTBAR
\t(void)site;
\t__asm__ volatile("bar.sync %0, %1;" :: "r"(conf_id + 1), "n"(BH_TPB));
#else
\t{
\t\tconst int dom = conf_id & 7;
\t\tconst int lane = threadIdx.x & 31;
\t\tconst int gwid = (blockIdx.x + blockIdx.y * gridDim.x) * 32
\t\t               + ((threadIdx.x + blockDim.x * threadIdx.y) >> 5);
\t\tif (lane == 0) d_bh_last[gwid] = (unsigned long long)site;
\t\t__syncwarp();
\t\tif (d_bh_poison) { __syncwarp(); return; } // poisoned: drain fast
\t\tvolatile unsigned int *fl = &d_bh_gates[dom].flip;
\t\tunsigned int f = *fl;
\t\tunsigned int *myc = f ? &d_bh_gates[dom].c1 : &d_bh_gates[dom].c0;
\t\tif (lane == 0) atomicAdd(myc, 1U);
\t\t__syncwarp();
\t\tunsigned int cnt = *myc;
\t\tif (cnt == 4U) { // last-arriving warp: reset + flip (single actor)
\t\t\tif (lane == 0) {
\t\t\t\tatomicExch(&d_bh_gates[dom].c0, 0U);
\t\t\t\tatomicExch(&d_bh_gates[dom].c1, 0U);
\t\t\t\tatomicExch((unsigned int*)fl, f ^ 1U);
\t\t\t}
\t\t\t__syncwarp();
\t\t\treturn;
\t\t}
\t\tunsigned long long t0 = clock64();
\t\twhile (*fl == f) {
\t\t\tif (d_bh_poison) break;
\t\t\tif (clock64() - t0 > 4000000000ULL) { // ~1.6s: declare stuck
\t\t\t\tif (lane == 0) atomicOr((unsigned int*)&d_bh_poison, 1u << dom);
\t\t\t\tbreak;
\t\t\t}
\t\t}
\t\t__syncwarp();
\t}
#endif
}"""
if old not in s:
    print("BH_ANCHOR-FAIL"); sys.exit(1)
wr(p, s.replace(old, new)); print("BH_SHARED-PATCHED")

# ---------- 3. kernel2.cu: definitions + site-tag macro + host zero/readout ----------
p = Q + "/inc/cuda/kernel2.cu"
s = rd(p)
def_anchor = "// [Q7-BH] per-conf state now in dynamic shared (BhConfState bh[8])"
defs = """// [Q7-BH] per-conf state now in dynamic shared (BhConfState bh[8])
#ifdef BH_SOFTBAR
__device__ BhGate d_bh_gates[8];
__device__ volatile unsigned int d_bh_poison;
__device__ unsigned long long d_bh_last[32768];
#endif"""
assert s.count(def_anchor) == 1
s = s.replace(def_anchor, defs)

oldm = "#define conf_barrier_k(conf_id) conf_barrier(conf_id)"
newm = "#define conf_barrier_k(conf_id) conf_barrier((conf_id), 100000 + __LINE__)"
assert oldm in s, "kernel2 macro anchor fail"
s = s.replace(oldm, newm)

launch_anchor = "kernel2_update<<<grid_size, block_size, bh_dyn_smem>>>"
zero_code = """#ifdef BH_SOFTBAR
\t{ void *gp, *pp, *lp;
\t  cudaGetSymbolAddress(&gp, d_bh_gates); cudaGetSymbolAddress(&pp, d_bh_poison);
\t  cudaGetSymbolAddress(&lp, d_bh_last);
\t  cudaMemsetAsync(gp, 0, sizeof(d_bh_gates)); cudaMemsetAsync(pp, 0, 4);
\t  cudaMemsetAsync(lp, 0, sizeof(d_bh_last)); }
#endif
\t""" + launch_anchor
assert s.count(launch_anchor) == 1
s = s.replace(launch_anchor, zero_code)

sync_anchor = """checkCUDA(cudaDeviceSynchronize());
\t{ struct timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t);"""
readout = """checkCUDA(cudaDeviceSynchronize());
#ifdef BH_SOFTBAR
\t{ unsigned int poison = 0; void *pp, *lp;
\t  cudaGetSymbolAddress(&pp, d_bh_poison); cudaGetSymbolAddress(&lp, d_bh_last);
\t  cudaMemcpy(&poison, pp, 4, cudaMemcpyDeviceToHost);
\t  static unsigned long long hlast[32768];
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
\t      fprintf(stderr, "BH-GATE last-site %llu warps %d\\n", sites[k], cnts[k]);
\t}
#endif
\t{ struct timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t);"""
assert s.count(sync_anchor) == 1, "sync anchor fail: %d" % s.count(sync_anchor)
s = s.replace(sync_anchor, readout)
wr(p, s); print("KERNEL2-PATCHED")

# ---------- 4. quasi_newton.cuh: site-tag all conf_barrier(conf_id) calls ----------
p = Q + "/inc/cuda/quasi_newton.cuh"
s = rd(p)
n = s.count("conf_barrier(conf_id)")
s = s.replace("conf_barrier(conf_id)", "conf_barrier((conf_id), 200000 + __LINE__)")
wr(p, s); print("QUASI-PATCHED sites:", n)
