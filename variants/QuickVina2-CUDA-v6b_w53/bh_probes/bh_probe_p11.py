#!/usr/bin/env python3
# BH-PROBE-P11: per-(block,domain,slot) monotonic arrival counters.
# INVARIANT: a coherent domain visits each barrier site exactly 4x per round.
# count % 4 != 0 at end  <=>  the divergence site itself. No timing inference.
import re
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"

# ---------- bh_shared.h: replace softbar block wholesale ----------
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()
start = s.index("#ifdef BH_SOFTBAR")
end = s.index("#endif", s.index("__forceinline__ void conf_barrier"))
end = s.index("\n#endif", end)  # file-level endif
tail = "\n#endif"
new_block = """#ifdef BH_SOFTBAR
// [BH-PROBE-P11] per-(block,domain,slot) monotonic arrival counters.
// A coherent 4-warp domain visits each site 4x per round: count%4==0.
// Timeout only ESCAPES the wait (poison drain) so the kernel finishes and
// the host can dump all counters. Slots: file1=1000+i(kernel2),
// file2=2000+i(quasi), file3=3000+i(mutate).
__device__ unsigned int d_bh_gate[1024 * 8 * 64];      // arrival counters
__device__ unsigned int d_bh_poisonb[1024];            // per-block jam mask
__device__ unsigned long long d_bh_last[1048576];      // per-lane last site
__device__ unsigned long long d_bh_stuck[1048576];     // per-lane timeout site
#endif
#define conf_barrier_m(conf_id) conf_barrier((conf_id), 3000)
__device__ __forceinline__ void conf_barrier(int conf_id, int site) {
\t__syncwarp(); // [Q7-BH] reconverge warp (thread0 gates diverge it)
#ifndef BH_SOFTBAR
\t(void)site;
\t__asm__ volatile("bar.sync %0, %1;" :: "r"(conf_id + 1), "n"(BH_TPB));
#else
\t{
\t\tconst int dom = conf_id & 7;
\t\tconst unsigned bid = blockIdx.x + blockIdx.y * gridDim.x;
\t\tconst int lane = threadIdx.x & 31;
\t\tconst unsigned gtid = bid * 1024 + (threadIdx.x + blockDim.x * threadIdx.y);
\t\tconst unsigned slot = (unsigned)site & 63u;
\t\tconst unsigned gidx = (bid * 8 + dom) * 64 + slot;
\t\t((volatile unsigned long long*)d_bh_last)[gtid] = (unsigned long long)site;
\t\tunsigned int oldc = 0;
\t\tif (lane == 0) oldc = atomicAdd(&d_bh_gate[gidx], 1U);
\t\toldc = __shfl_sync(0xFFFFFFFF, oldc, 0);
\t\tconst unsigned target = (oldc & ~3u) + 4u; // next multiple of 4
\t\tif (oldc + 1u != target) { // not the 4th arriver: wait
\t\t\tunsigned long long t0 = clock64();
\t\t\twhile (d_bh_gate[gidx] < target) {
\t\t\t\tif (d_bh_poisonb[bid] & (1u << dom)) break; // jam drain
\t\t\t\tif (clock64() - t0 > 100000000ULL) { // ~0.04s: timeout, record + drain
\t\t\t\t\t((volatile unsigned long long*)d_bh_stuck)[gtid] = (unsigned long long)site;
\t\t\t\t\tatomicOr(&d_bh_poisonb[bid], 1u << dom);
\t\t\t\t\tbreak;
\t\t\t\t}
\t\t\t}
\t\t\t__syncwarp();
\t\t}
\t}
#endif
"""
s = s[:start] + new_block + tail
open(p, "w").write(s)
print("BH_SHARED-P11-REWRITTEN")

# ---------- renumber quasi_newton.cuh sites: 2000+i in file order ----------
p = Q + "/inc/cuda/quasi_newton.cuh"
s = open(p).read()
cnt = [0]
def rep(m):
    cnt[0] += 1
    return f"conf_barrier((conf_id), {2000 + cnt[0] - 1})"
s2 = re.sub(r"conf_barrier\(\(conf_id\), \d+\)", rep, s)
open(p, "w").write(s2)
print("QUASI slots:", cnt[0])

# ---------- kernel2.cu macro slot 1000 ----------
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
s = s.replace("#define conf_barrier_k(conf_id) conf_barrier((conf_id), 100000 + __LINE__)",
              "#define conf_barrier_k(conf_id) conf_barrier((conf_id), 1000)")
open(p, "w").write(s)
print("KERNEL2 macro slot 1000")

# ---------- mutate_conf.cuh: 3 explicit slots ----------
p = Q + "/inc/cuda/mutate_conf.cuh"
s = open(p).read()
n = [0]
def rep3(m):
    n[0] += 1
    return f"conf_barrier_m2((conf_id), {3000 + n[0]})"
s = re.sub(r"conf_barrier_m\(conf_id\)", rep3, s)
# conf_barrier_m2 -> conf_barrier direct (macro took __LINE__; now explicit)
s = s.replace("conf_barrier_m2(", "conf_barrier(")
open(p, "w").write(s)
print("MUTATE slots:", n[0])
