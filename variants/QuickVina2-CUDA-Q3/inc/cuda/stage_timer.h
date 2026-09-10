// stage_timer.h — v2 (Q4-2): robust in-kernel stage timing.
// v1 defects (validated 2026-09-10): per-TU symbol copies (d_stage defined in
// .cu + non-extern in .h), stage_timer.cu missing from Makefile, thread0-only
// spans, no cross-check. v2: definition lives HERE (single inclusion domain:
// only kernel2.cu includes this header — enforced below); per-block shared-mem
// accumulation flushed once at kernel end; built-in cross-check print.
#ifndef STAGE_TIMER_H
#define STAGE_TIMER_H
#include <cstdio>

struct StageStats {
    unsigned long long mutate, eval_init, ls_total, ls_evals, twop, evals_n;
    unsigned long long grid, intra, pot;
    unsigned long long n_ep, n_iter, n_ls;
    unsigned long long ls_total_chk;  // v2 cross-check: sum of per-block flushes
};

__device__ StageStats d_stage;
__device__ unsigned long long d_k2_wall;   // kernel2 wall (block0 measures)

__device__ __forceinline__ unsigned long long st_clock_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void st_add(unsigned long long* addr, unsigned long long ns) {
    if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(addr, ns);
}
#define STAGE_T0 unsigned long long st_t0 = st_clock_ns()
#define STAGE_ADD(field) st_add(&d_stage.field, st_clock_ns() - st_t0)
#define STAGE_ADD2(t0var, field) st_add(&d_stage.field, st_clock_ns() - t0var)
#endif
