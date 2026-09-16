// gpu_timer.h — in-binary GPU-timeline accounting (no profiler needed).
// Categories: K1, K2, HtoD, DtoH, Memset. Each cudaMemcpy/Memset is wrapped
// by an event pair on the same stream as the copy; cudaEventElapsedTime
// measures pure device-side duration of that op. Settled once at exit.
#ifndef GPU_TIMER_H
#define GPU_TIMER_H
#include <cstdio>
#include <cuda_runtime.h>

struct GpuTimers {
    cudaEvent_t start_ev, stop_ev; // settlement window: first record -> last record
    bool started = false;
    float k1 = 0.f, k2 = 0.f;          // from existing kernel wrappers (ms, accumulated)
    float htod = 0.f, dtoh = 0.f, memst = 0.f; // memcpy/memset device time (ms, accumulated)
    int n_htod = 0, n_dtoh = 0, n_memst = 0, n_pairs = 0;
};
extern GpuTimers g_timer; // defined in gpu_timer.cu (nvcc TU)

// Wrap one async op on stream s: record ev before, caller issues op, then record after.
__attribute__((unused)) static inline void gt_pre(cudaStream_t s){ cudaEventRecord(g_timer.start_ev, s); }
__attribute__((unused)) static inline void gt_post(cudaStream_t s, int cat){
    cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, s);
    float ms = 0.f; cudaEventSynchronize(e); cudaEventElapsedTime(&ms, g_timer.start_ev, e);
    cudaEventDestroy(e);
    if (cat == 0) { g_timer.htod += ms; g_timer.n_htod++; }
    else if (cat == 1) { g_timer.dtoh += ms; g_timer.n_dtoh++; }
    else { g_timer.memst += ms; g_timer.n_memst++; }
    g_timer.n_pairs++;
}
// Settlement: accumulate k1/k2 reported by kernel wrappers via macros below.
__attribute__((unused)) static inline void gt_add_kernel(int which, float ms){
    if (which == 1) g_timer.k1 += ms; else g_timer.k2 += ms;
}
__attribute__((unused)) static inline void gt_report(){
    if (!g_timer.n_pairs && g_timer.k1 == 0.f && g_timer.k2 == 0.f) return;
    printf("GPU_HtoD_TIME=%.3f\n", g_timer.htod / 1000.f);
    printf("GPU_DtoH_TIME=%.3f\n", g_timer.dtoh / 1000.f);
    printf("GPU_Memset_TIME=%.3f\n", g_timer.memst / 1000.f);
    float gpu_all = g_timer.k1 + g_timer.k2 + g_timer.htod + g_timer.dtoh + g_timer.memst;
    printf("GPU_ALL_TIME=%.3f\n", gpu_all / 1000.f);
    printf("GPU_N_HtoD=%d GPU_N_DtoH=%d GPU_N_Memset=%d\n", g_timer.n_htod, g_timer.n_dtoh, g_timer.n_memst);
}
#define GT_EVENT_PRE(s) gt_pre(s)
#define GT_EVENT_POST(s, cat) gt_post(s, cat)
#endif
