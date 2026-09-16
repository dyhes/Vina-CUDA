#!/usr/bin/env python3
# BH-FIX-SOFTGATE v2: per-lane shared gate (P12b protocol) — v2's warp-granular
# gate hung 3/3: entry __syncwarp re-created the P14 wedge point, and
# warp-granular counting assumes warp coherence (false). This version matches
# P14b/P12b exactly (per-lane arrival, all-lane spin, no prologue syncwarp)
# with gates in SHARED memory + a loud timeout escape (d_bh_escapes counter,
# CASF gate requires 0).
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()

old = """__device__ __forceinline__ void conf_barrier(int conf_id, int site) {
\t// Software domain barrier. Hardware bar.sync(id, count) counts arrivals,
\t// not sites: skew between sibling warps lets an arrival at one site
\t// release a waiter at another (bh_idtest.cu), amplifying into a wedged
\t// count. Here a site releases only when its OWN 128-lane count completes.
\t// Warp-granular arrival: lane0 adds 32 per warp; 4 warps -> 128/round.
\tconst int lane = threadIdx.x & 31;
\t__syncwarp(); // warp-local reconvergence (harmless: uniform decision regime)
\t__threadfence_block(); // release: our shared writes precede arrival
\tif (lane == 0) {
\t\tunsigned int *g = bh_gate_ptr(conf_id, site);
\t\tunsigned int oldc = atomicAdd(g, 32u);
\t\tunsigned int target = (oldc & ~127u) + 128u;
\t\twhile (*(volatile unsigned int *)g < target) { __nanosleep(20); }
\t}
\t__syncwarp(); // whole warp waits for lane0's release
\t__threadfence_block(); // acquire: all arrivals' shared writes visible
}"""
new = """__device__ unsigned int d_bh_drain[BH_NBLOCKS];      // per-block drained-domain bitset (escape)
__device__ unsigned long long d_bh_escapes;          // loud wedge counter (host gate: must be 0)
__device__ __forceinline__ void conf_barrier(int conf_id, int site) {
\t// Software domain barrier, per-lane protocol (P14b/P12b-verified): a site
\t// releases only when its OWN 128 arrivals complete — bar.sync(id,count)'s
\t// cross-site release cannot occur by construction. No prologue
\t// __syncwarp: a diverged warp must reach the gate lane-by-lane (P14:
\t// warp-wedge died there). Escape: a >24ms wait drains the domain
\t// (liveness) and bumps the LOUD counter — CASF gate requires 0.
\tconst int dom = conf_id & 7;
\tconst unsigned bid = blockIdx.x + blockIdx.y * gridDim.x;
\tconst unsigned drain = d_bh_drain[bid];
\tif (drain & (1u << dom)) return; // domain already escaped: drain
\tunsigned int *g = bh_gate_ptr(conf_id, site);
\t__threadfence_block(); // release: our shared writes precede arrival
\tunsigned int oldc = atomicAdd(g, 1u); // EVERY lane counts itself
\tunsigned int target = (oldc & ~127u) + 128u;
\tif (oldc + 1u != target) {
\t\tunsigned long long t0 = clock64();
\t\twhile (*(volatile unsigned int *)g < target) {
\t\t\tif (d_bh_drain[bid] & (1u << dom)) break;
\t\t\tif (clock64() - t0 > 60000000ULL) { // ~24ms: wedge
\t\t\t\tatomicOr(&d_bh_drain[bid], 1u << dom);
\t\t\t\tatomicAdd(&d_bh_escapes, 1ULL);
\t\t\t\tbreak;
\t\t\t}
\t\t\t__nanosleep(40);
\t\t}
\t}
\t__threadfence_block(); // acquire: all arrivals' shared writes visible
}"""
assert s.count(old) == 1, "v2 gate anchor"
s = s.replace(old, new)
open(p, "w").write(s)
print("BH_SHARED v3 (per-lane shared gate + escape)")

# kernel2.cu: zero drain/escape before each launch + host readout after sync
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
anchor = "\tcudaError_t bh_err = cudaFuncSetAttribute((const void*)kernel2_update,"
assert s.count(anchor) == 1
s = s.replace(anchor, """\t{ void *dp = 0, *ep = 0;
\t  cudaGetSymbolAddress(&dp, d_bh_drain); cudaGetSymbolAddress(&ep, d_bh_escapes);
\t  cudaMemsetAsync(dp, 0, sizeof(d_bh_drain)); cudaMemsetAsync(ep, 0, 8); }
""" + anchor)
anchor2 = "\tcheckCUDA(cudaDeviceSynchronize());\n\t{ struct timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t);"
assert s.count(anchor2) == 1
s = s.replace(anchor2, """\tcheckCUDA(cudaDeviceSynchronize());
\t{ unsigned long long esc = 0; void *ep = 0;
\t  cudaGetSymbolAddress(&ep, d_bh_escapes);
\t  cudaMemcpy(&esc, ep, 8, cudaMemcpyDeviceToHost);
\t  fprintf(stderr, "BHSG escapes=%llu\\n", esc); }
\t{ struct timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t);""")
open(p, "w").write(s)
print("KERNEL2 zeroing + escape readout")
