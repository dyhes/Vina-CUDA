#!/usr/bin/env python3
# BH-PROBE-P10: snapshot trigger by ANY stuck lane (not just lane0); shorter
# timeout (0.2s) to limit cross-domain poison windows; snap_done kept as
# triggerer gid+1 for host verification.
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"
p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()

old = "\t\t\tif (clock64() - t0 > 4000000000ULL) { // ~1.6s: declare stuck"
new = "\t\t\tif (clock64() - t0 > 300000000ULL) { // ~0.12s: declare stuck"
assert old in s, "t0 anchor"
s = s.replace(old, new)

old = """\t\t\t\t((volatile unsigned long long*)d_bh_stuck)[gtid] = (unsigned long long)site; // sticky
\t\t\t\tif (lane == 0 && atomicExch(&d_bh_snap_done[gb], 1) == 0) {
\t\t\t\t\tconst unsigned base = (blockIdx.x + blockIdx.y * gridDim.x) * 1024 + dom * 128;
\t\t\t\t\tfor (int t = 0; t < 128; t++)
\t\t\t\t\t\td_bh_snap[gb * 128 + t] = ((volatile unsigned long long*)d_bh_last)[base + t];
\t\t\t\t}"""
new = """\t\t\t\t((volatile unsigned long long*)d_bh_stuck)[gtid] = (unsigned long long)site; // sticky
\t\t\t\tint prev = atomicExch(&d_bh_snap_done[gb], gtid + 1); // any lane may snap
\t\t\t\tif (prev == 0) {
\t\t\t\t\tconst unsigned base = (blockIdx.x + blockIdx.y * gridDim.x) * 1024 + dom * 128;
\t\t\t\t\tfor (int t = 0; t < 128; t++)
\t\t\t\t\t\td_bh_snap[gb * 128 + t] = ((volatile unsigned long long*)d_bh_last)[base + t];
\t\t\t\t}"""
assert old in s, "snap anchor"
s = s.replace(old, new)
open(p, "w").write(s)
print("BH_SHARED-P10")

# host: also dump snap_done
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
old = """\t  static unsigned long long hsnap[8 * 1024 * 128];
\t  cudaMemcpy(hsnap, snp, 8 * 1024 * 128 * 8, cudaMemcpyDeviceToHost);
\t  f = fopen("/tmp/bh_snap.bin", "wb");
\t  fwrite(hsnap, 8, 8 * 1024 * 128, f); fclose(f);"""
new = """\t  static unsigned long long hsnap[8 * 1024 * 128];
\t  cudaMemcpy(hsnap, snp, 8 * 1024 * 128 * 8, cudaMemcpyDeviceToHost);
\t  f = fopen("/tmp/bh_snap.bin", "wb");
\t  fwrite(hsnap, 8, 8 * 1024 * 128, f); fclose(f);
\t  void *sdp = 0; cudaError_t e6 = cudaGetSymbolAddress(&sdp, d_bh_snap_done);
\t  if (e6) fprintf(stderr, "BHSDERR %d\\n", e6);
\t  static int hsnapdone[8 * 1024];
\t  cudaMemcpy(hsnapdone, sdp, sizeof(hsnapdone), cudaMemcpyDeviceToHost);
\t  int nsnap = 0;
\t  for (int i = 0; i < 8 * 1024; i++) if (hsnapdone[i]) nsnap++;
\t  fprintf(stderr, "BH-GATE snapshots-taken=%d\\n", nsnap);"""
assert old in s, "host snap anchor"
s = s.replace(old, new)
open(p, "w").write(s)
print("KERNEL2-P10")
