#!/usr/bin/env python3
# v4 fix v3: banner-to-EOF replacement (v3 file has no conf_barrier_m).
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()
start = s.index("// [Q7-BH-FIX] gate pointers:")
new_tail = """// [Q7-BH-FIX v4] hardware domain gate + step-boundary block sync. The
// per-lane software gate (v3) exposed a REAL protocol imbalance: sibling
// warps legitimately run different LS/BFGS trip counts (452k escapes/3 runs
// on 1pxn), so arrival-count-per-site is not an invariant of this algorithm.
// Hardware bar.sync(id,count) survives that (cross-site release absorbs it)
// but can wedge when skew accumulates across MC steps; the block-wide
// __syncthreads() at every MC-step boundary (kernel2.cu) resets skew each
// step — the MC loop itself is conf-uniform so all domains arrive.
#define BH_SLOTS 64 // site numbering kept for documentation/instrumentation
__device__ __forceinline__ void conf_barrier(int conf_id, int site) {
\t(void)site;
\t__syncwarp(); // [Q7-BH] warp-local reconvergence (thread0 gates diverge it)
\t__asm__ volatile("bar.sync %0, %1;" :: "r"(conf_id + 1), "n"(BH_TPB) : "memory");
}
__device__ __forceinline__ void conf_barrier_m(int conf_id) {
\tconf_barrier(conf_id, 0);
}
#endif
"""
s = s[:start] + new_tail
open(p, "w").write(s)
print("BH_SHARED v4 OK (banner-to-EOF)")

# kernel2.cu: v4 edits (idempotent)
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
changed = False
old = """\t{ void *dp = 0, *ep = 0;
\t  cudaGetSymbolAddress(&dp, d_bh_drain); cudaGetSymbolAddress(&ep, d_bh_escapes);
\t  cudaMemsetAsync(dp, 0, 1024*4); cudaMemsetAsync(ep, 0, 8); }
"""
if old in s:
    s = s.replace(old, ""); changed = True
old2 = """\tcheckCUDA(cudaDeviceSynchronize());
\t{ unsigned long long esc = 0; void *ep = 0;
\t  cudaGetSymbolAddress(&ep, d_bh_escapes);
\t  cudaMemcpy(&esc, ep, 8, cudaMemcpyDeviceToHost);
\t  fprintf(stderr, "BHSG escapes=%llu\\n", esc); }
"""
if old2 in s:
    s = s.replace(old2, "\tcheckCUDA(cudaDeviceSynchronize());\n"); changed = True
old3 = "\tfor (int step = 0; step < search_depth; step++)\n\t{"
new3 = """\tfor (int step = 0; step < search_depth; step++)
\t{
\t\t__syncthreads(); // [Q7-BH-FIX] block-wide skew killer: resets any
\t\t// domain-barrier skew accumulated in the previous MC step. The MC
\t\t// loop is conf-uniform, so all 8 domains arrive here every step."""
if "[Q7-BH-FIX] block-wide skew killer" not in s:
    assert s.count(old3) == 1
    s = s.replace(old3, new3); changed = True
open(p, "w").write(s)
print("KERNEL2 v4 OK, changed:", changed)
