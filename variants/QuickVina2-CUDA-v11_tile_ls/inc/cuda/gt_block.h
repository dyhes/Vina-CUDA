// block-level helpers for gpu_timer (one event pair per contiguous copy block)
#ifndef GT_BLOCK
#define GT_BLOCK
#include <cuda_runtime.h>
struct GtBlock {
    cudaEvent_t open_ev;
    bool open;
};
extern GtBlock g_blk;
void gt_block_start();
void gt_block_end(int cat);
#define GT_BLOCK_START() gt_block_start()
#define GT_BLOCK_END(cat) gt_block_end(cat)
#endif
