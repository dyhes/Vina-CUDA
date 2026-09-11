// Q4-7C: global per-trajectory pair skip-bitmap (design per
// q4_7c_pairs_prefilter_2026-09-10.md, "global bitmap" variant).
//
// Per-block (=per-trajectory) bitmap in GLOBAL memory, zeroed once by host
// cudaMemset before kernel_monte; access via L2 atomics (fast, no shared
// budget usage, no init-flags needed). Rotating 1/8 re-probe window catches
// pairs entering the cutoff within <=8 evals.
//
// Layout: bitmap_words = (MAX_NUM_OF_LIG_PAIRS+63)/64 = 32 u64 per block
//         buffer = blocks(8192) * 32 * 8B = 2MB
//
// NOTE: __device__ variables cannot be extern across TUs without -rdc in
// CUDA 11.8; the DEFINITIONS live in kernel2.cu and this header must ONLY
// be included by kernel2.cu (enforced below via a TU marker from kernel2.h).
#ifndef Q4_PAIRS_BITMAP_CUH
#define Q4_PAIRS_BITMAP_CUH
#include <cuda_runtime.h>
#include <cstdio>

#define Q4_PAIRS_WORDS ((MAX_NUM_OF_LIG_PAIRS + 63) / 64)
#define Q4_PAIRS_BLOCKS 8192

#ifdef Q4_BITMAP_DEFS
__device__ unsigned long long* q4_pair_skip_buf = 0;
__device__ unsigned long long q4_eval_epoch_g[Q4_PAIRS_BLOCKS];
#else
extern __device__ unsigned long long* q4_pair_skip_buf;
extern __device__ unsigned long long q4_eval_epoch_g[Q4_PAIRS_BLOCKS];
#endif
// NOTE: nvcc 11.8 no-rdc demotes extern __device__ to per-TU storage!
// => set the pointer FROM DEVICE code via a one-time init kernel in the same
//    TU as the definition (kernel2.cu), and have quasi_newton read the buffer
//    through q4_skip_row/q4_epoch PARAMS ONLY (passed from kernel2_update).

__device__ __forceinline__ unsigned long long* q4_skip_row(int block) {
	return q4_pair_skip_buf + (size_t)block * Q4_PAIRS_WORDS;
}
__device__ __forceinline__ unsigned long long q4_epoch(int block) {
	return q4_eval_epoch_g[block];
}
#endif
