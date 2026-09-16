#!/usr/bin/env python3
# BH-FIX-SOFTGATE: replace hardware bar.sync with a shared-memory software
# domain barrier (per-(conf,site) monotonic counters). Two-phase: verify all
# anchors, then apply. Run with --apply to write.
import sys
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"
APPLY = "--apply" in sys.argv
def rd(p): return open(p).read()

bh = rd(Q + "/inc/cuda/bh_shared.h")
qn = rd(Q + "/inc/cuda/quasi_newton.cuh")
mu = rd(Q + "/inc/cuda/mutate_conf.cuh")
k2 = rd(Q + "/inc/cuda/kernel2.cu")

fails = []
def chk(c, m):
    if not c: fails.append(m)

# ---- tree clean ----
chk("bh_gate_ptr" not in bh, "already patched")
# ---- anchors: bh_shared.h ----
old_fn = """__device__ __forceinline__ void conf_barrier(int conf_id) {
\t__syncwarp(); // [Q7-BH] reconverge warp (thread0 gates diverge it)
\t__asm__ volatile("bar.sync %0, %1;" :: "r"(conf_id + 1), "n"(BH_TPB));
}
__device__ __forceinline__ void conf_barrier_m(int conf_id) {
\tconf_barrier(conf_id);
}"""
chk(bh.count(old_fn) == 1, "bh conf_barrier fn anchor")
chk("extern __shared__ BhConfState bh[];" in bh, "bh arena decl")
# ---- anchors: call sites ----
chk(qn.count("conf_barrier(conf_id)") == 23, f"quasi sites {qn.count('conf_barrier(conf_id)')}")
chk(mu.count("conf_barrier_m(conf_id)") == 3, f"mutate sites {mu.count('conf_barrier_m(conf_id)')}")
chk(k2.count("#define conf_barrier_k(conf_id) conf_barrier(conf_id)") == 1, "k2 macro def")
chk(k2.count("conf_barrier_k(conf_id);") == 1, "k2 call sites")  # def line excluded (has no ';')
# ---- anchors: kernel2 zero block + launch size ----
zero_blk = """\tif (local_tid == 0) {
\t\tfor (int w = 0; w < (MAX_NUM_OF_LIG_PAIRS + 63) / 64; w++) q4_skip_row[w] = 0ULL;
\t\tq4_epoch_i = 0;
\t}"""
chk(k2.count(zero_blk) == 1, "k2 zero block")
chk("bh_dyn_smem = BH_NCONF * sizeof(BhConfState);" in k2, "k2 dyn size")
chk("cudaFuncAttributeMaxDynamicSharedMemorySize,\n\t\t(int)(BH_NCONF * sizeof(BhConfState)));" in k2, "k2 attr size")
# ---- anchors: volatile consumption (P16 sites) ----
chk(qn.count("\t\tmy_e = warp_e[0];") == 2, "my_e sites")
chk(qn.count("\treturn tmp[0];") == 1, "scalar ret")
chk(mu.count("\treturn (warp_cnt[0] > 0) ? sqrtf(warp_acc[0] / (flo)warp_cnt[0]) : 0.0f;") == 1, "gyration ret")

if fails:
    print("PHASE-A FAILURES:")
    for f in fails: print("  -", f)
    sys.exit(1)
print("PHASE-A OK (all anchors verified)")
if not APPLY:
    print("(dry-run)")
    sys.exit(0)

# ================= PHASE B =================
# ---- bh_shared.h: new barrier ----
new_fn = """// [Q7-BH-FIX] gate pointers: per-(conf,slot) monotonic u32 counters live in
// dynamic shared right after the bh[NCONF] arena (launched with
// BH_GATE_BYTES extra). One gate per (conf, site): the cross-site release
// race of bar.sync(id,count) cannot occur by construction.
#define BH_SLOTS 64
#define BH_GATE_BYTES (BH_NCONF * BH_SLOTS * 4)
__device__ __forceinline__ unsigned int *bh_gate_ptr(int conf_id, int site) {
\treturn reinterpret_cast<unsigned int *>(
\t\t\t   reinterpret_cast<char *>(&bh[0]) + BH_NCONF * sizeof(BhConfState)) +
\t\t   (unsigned)((conf_id & 7) * BH_SLOTS + ((unsigned)site & (BH_SLOTS - 1)));
}
__device__ __forceinline__ void conf_barrier(int conf_id, int site) {
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
bh = bh.replace(old_fn, new_fn)
open(Q + "/inc/cuda/bh_shared.h", "w").write(bh)
print("BH_SHARED rewritten (soft gate)")

# ---- quasi_newton.cuh: line-based slot renumber + volatile reads ----
lines = qn.split("\n")
n = 0
for i, line in enumerate(lines):
    if "conf_barrier(conf_id)" in line:
        n += 1
        lines[i] = line.replace("conf_barrier(conf_id)", "conf_barrier((conf_id), %d)" % (16 + n - 1))
assert n == 23
qn = "\n".join(lines)
qn = qn.replace("\t\tmy_e = warp_e[0];", "\t\tmy_e = *(volatile float *)&warp_e[0];")
qn = qn.replace("\treturn tmp[0];", "\treturn *(volatile flo *)&tmp[0];")
open(Q + "/inc/cuda/quasi_newton.cuh", "w").write(qn)
print("QUASI renumbered (slots 16-38) + volatile reads")

# ---- mutate_conf.cuh: 3 sites + gyration volatile ----
lines = mu.split("\n")
n = 0
for i, line in enumerate(lines):
    if "conf_barrier_m(conf_id)" in line:
        n += 1
        lines[i] = line.replace("conf_barrier_m(conf_id)", "conf_barrier((conf_id), %d)" % (56 + n - 1))
assert n == 3
mu = "\n".join(lines)
mu = mu.replace("\treturn (warp_cnt[0] > 0) ? sqrtf(warp_acc[0] / (flo)warp_cnt[0]) : 0.0f;",
                "\treturn (warp_cnt[0] > 0) ? sqrtf(*(volatile float *)&warp_acc[0] / (flo)*(volatile int *)&warp_cnt[0]) : 0.0f;")
open(Q + "/inc/cuda/mutate_conf.cuh", "w").write(mu)
print("MUTATE renumbered (slots 56-58) + gyration volatile")

# ---- kernel2.cu: macro def removal, call site, zeroing, sizes ----
k2 = k2.replace("#define conf_barrier_k(conf_id) conf_barrier(conf_id)\n", "")
k2 = k2.replace("conf_barrier_k(conf_id);", "conf_barrier((conf_id), 40);")
k2 = k2.replace(zero_blk, """\tif (local_tid == 0) {
\t\tfor (int w = 0; w < (MAX_NUM_OF_LIG_PAIRS + 63) / 64; w++) q4_skip_row[w] = 0ULL;
\t\tq4_epoch_i = 0;
\t\tunsigned int *gbase = bh_gate_ptr(conf_id, 0); // zero THIS conf's 64 gates
\t\tfor (int i = 0; i < BH_SLOTS; i++) gbase[i] = 0u;
\t}""")
k2 = k2.replace("bh_dyn_smem = BH_NCONF * sizeof(BhConfState);",
                "bh_dyn_smem = BH_NCONF * sizeof(BhConfState) + BH_GATE_BYTES;")
k2 = k2.replace("cudaFuncAttributeMaxDynamicSharedMemorySize,\n\t\t(int)(BH_NCONF * sizeof(BhConfState)));",
                "cudaFuncAttributeMaxDynamicSharedMemorySize,\n\t\t(int)(BH_NCONF * sizeof(BhConfState) + BH_GATE_BYTES));")
open(Q + "/inc/cuda/kernel2.cu", "w").write(k2)
print("KERNEL2 patched (slot 40, gate zeroing, dyn size)")
print("PHASE-B COMPLETE")
