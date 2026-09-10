#include "gpu_timer.h"
#include "gt_block.h"

GpuTimers g_timer;
GtBlock g_blk;

void gt_block_start() {
    if (!g_blk.open) {
        cudaEventCreate(&g_blk.open_ev);
        cudaEventRecord(g_blk.open_ev, 0);
        g_blk.open = true;
    }
}

void gt_block_end(int cat) {
    if (!g_blk.open) return;
    cudaEvent_t e; cudaEventCreate(&e);
    cudaEventRecord(e, 0);
    float ms = 0.f;
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, g_blk.open_ev, e);
    cudaEventDestroy(e);
    cudaEventDestroy(g_blk.open_ev);
    g_blk.open = false;
    if (cat == 0) { g_timer.htod += ms; g_timer.n_htod++; }
    else if (cat == 1) { g_timer.dtoh += ms; g_timer.n_dtoh++; }
    else { g_timer.memst += ms; g_timer.n_memst++; }
}
