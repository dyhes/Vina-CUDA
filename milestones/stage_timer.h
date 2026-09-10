// stage_timer.h — fine-grained in-kernel stage timing for kernel2/BFGS.
// Uses %globaltimer (nanosecond, global across SMs) read by thread0 of each block.
// Accumulates per-stage nanoseconds into global counters (one set, atomicAdd from block0 only
// would bias; instead EVERY block adds, then host divides by num trajectories).
// Stages: MUTATE, EVAL_INIT(1st E), LS_TOTAL(line search incl inner E), TWOP(two-loop/updates),
//         and inside E: GRID (intermol grid gather), INTRA(pair terms), POT(force->change)
#ifndef STAGE_TIMER_H
#define STAGE_TIMER_H
#include <cstdio>

struct StageStats {
    unsigned long long mutate, eval_init, ls_total, ls_evals, twop, evals_n;
    unsigned long long grid, intra, pot;
    unsigned long long n_ep;   // bfgs episodes
};
extern __device__ StageStats d_stage;   // one global set, atomicAdd per block

// read globaltimer (ns)
__device__ __forceinline__ unsigned long long st_clock_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void st_add(unsigned long long* addr, unsigned long long ns) {
    if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(addr, ns);
}
#define STAGE_T0 unsigned long long st_t0 = st_clock_ns();
#define STAGE_ADD(field) st_add(&d_stage.field, st_clock_ns() - st_t0)
#endif
