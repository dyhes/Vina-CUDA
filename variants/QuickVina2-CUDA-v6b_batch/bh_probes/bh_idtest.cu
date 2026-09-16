// bh_idtest.cu: does bar.sync(id,count) release across DIFFERENT arrival sites?
// warp0: long spin, then bar.sync(1, 64) at SITE-A; writes mA=1.
// warp1: immediate bar.sync(1, 64) at SITE-B; writes mB=1; then waits for mA.
// Cross-site release => both markers set quickly (warp1 released by warp0's
// SITE-A arrival). Same-site matching => warp1 hangs at SITE-B.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(int *out, unsigned long long *times) {
	const int lane = threadIdx.x & 31;
	unsigned long long t0 = clock64();
	if (threadIdx.x < 32) { // warp0: long spin then SITE-A barrier
		if (lane == 0) { unsigned long long s = clock64(); while (clock64() - s < 2000000ULL) {} }
		__syncwarp();
		asm volatile("bar.sync 1, 64;"); // SITE-A
		if (lane == 0) { out[0] = 1; times[0] = clock64() - t0; }
	} else {                // warp1: SITE-B barrier immediately
		asm volatile("bar.sync 1, 64;"); // SITE-B
		if (lane == 0) { out[1] = 1; times[1] = clock64() - t0; }
		// spin until warp0's marker (proves both passed)
		if (lane == 0) { while (out[0] == 0 && clock64() - t0 < 3000000000ULL) {} times[2] = clock64() - t0; }
	}
}
int main() {
	int *d; unsigned long long *dt;
	cudaMalloc(&d, 8); cudaMalloc(&dt, 32); cudaMemset(d, 0, 8);
	k<<<1, 64>>>(d, dt);
	cudaError_t e = cudaDeviceSynchronize();
	int h[2]; unsigned long long ht[3];
	cudaMemcpy(h, d, 8, cudaMemcpyDeviceToHost);
	cudaMemcpy(ht, dt, 24, cudaMemcpyDeviceToHost);
	printf("rc=%s mA=%d mB=%d t_warp0=%.1fms t_warp1=%.1fms t_join=%.1fms\n",
	       cudaGetErrorString(e), h[0], h[1],
	       ht[0]/1e6, ht[1]/1e6, ht[2]/1e6);
	printf(h[0] && h[1] ? "VERDICT: CROSS-SITE RELEASE CONFIRMED\n"
	                    : "VERDICT: sites matched independently (or hang)\n");
	return 0;
}
