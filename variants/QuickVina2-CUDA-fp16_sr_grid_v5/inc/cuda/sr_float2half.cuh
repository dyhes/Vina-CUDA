/* Stochastic rounding FP32→FP16 for CUDA
 * Uses xorshift32 PRNG seeded per-thread, no global state.
 * Quantization noise is zero-mean → BFGS trajectory less systematically biased.
 *
 * Usage: __half sr_float2half(float x, uint32_t *state)
 *        const flo f000 = __half2float(sr_float2half(*base_ptr, &rng_state));
 */

#ifndef SR_FLOAT2HALF_H
#define SR_FLOAT2HALF_H

#include <cuda_fp16.h>

__device__ __forceinline__ uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Stochastic rounding: round to nearest FP16 with probability proportional to distance.
 * For x between two FP16 representable values h_lo and h_hi:
 *   P(round to h_hi) = (x - h_lo) / (h_hi - h_lo)
 *   P(round to h_lo) = (h_hi - x) / (h_hi - h_lo)
 *
 * In expectation, E[round(x)] = x → zero-mean quantization noise.
 */
__device__ __forceinline__ __half sr_float2half(float x, uint32_t *state) {
    // Fast path: if x is exactly representable in FP16, just convert
    __half h = __float2half(x);
    float round_tripped = __half2float(h);
    if (x == round_tripped) return h;

    // Get the two nearest FP16 values
    __half h_up = __float2half_ru(x);  // round up
    __half h_dn = __float2half_rd(x);  // round down
    float f_up = __half2float(h_up);
    float f_dn = __half2float(h_dn);

    float range = f_up - f_dn;
    if (range == 0.0f) return h;

    // Probability of rounding up
    float p_up = (x - f_dn) / range;

    // Random decision
    float r = (float)(xorshift32(state) >> 8) / (float)(0x00FFFFFF);
    return (r < p_up) ? h_up : h_dn;
}

#endif
