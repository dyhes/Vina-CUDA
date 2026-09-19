#!/usr/bin/env python3
# P11 part2: host readout -> dump d_bh_gate counters + stuck + last
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()

# replace the whole readout block
start = s.index("#ifdef BH_SOFTBAR\n\t{ unsigned int hpoison[1024]")
end_marker = "\t}\n#endif\n\t{ struct timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t);"
end = s.index(end_marker)
new_block = """#ifdef BH_SOFTBAR
\t{ unsigned int hpoison[1024] = {0}; void *pp = 0, *lp = 0, *sp = 0, *gp = 0;
\t  cudaError_t e2, e3, e4, e5;
\t  e2 = cudaGetSymbolAddress(&pp, d_bh_poisonb); e3 = cudaGetSymbolAddress(&lp, d_bh_last);
\t  e4 = cudaGetSymbolAddress(&sp, d_bh_stuck);  e5 = cudaGetSymbolAddress(&gp, d_bh_gate);
\t  if (e2 || e3 || e4 || e5) fprintf(stderr, "BHSYMERR %d %d %d %d\\n", e2, e3, e4, e5);
\t  cudaMemcpy(hpoison, pp, sizeof(hpoison), cudaMemcpyDeviceToHost);
\t  int njam = 0; unsigned jammask = 0;
\t  for (int i = 0; i < 1024; i++) if (hpoison[i]) { njam++; jammask |= hpoison[i]; }
\t  fprintf(stderr, "BH-GATE jammed-blocks=%d dommask=0x%x\\n", njam, jammask);
\t  static unsigned long long hlast[1048576];
\t  cudaMemcpy(hlast, lp, 1048576*8, cudaMemcpyDeviceToHost);
\t  FILE *f = fopen("/tmp/bh_ledger.bin", "wb");
\t  fwrite(hlast, 8, 1048576, f); fclose(f);
\t  static unsigned long long hstuck[1048576];
\t  cudaMemcpy(hstuck, sp, 1048576*8, cudaMemcpyDeviceToHost);
\t  f = fopen("/tmp/bh_stuck.bin", "wb");
\t  fwrite(hstuck, 8, 1048576, f); fclose(f);
\t  static unsigned int hgate[1024 * 8 * 64];
\t  cudaMemcpy(hgate, gp, sizeof(hgate), cudaMemcpyDeviceToHost);
\t  f = fopen("/tmp/bh_gate.bin", "wb");
\t  fwrite(hgate, 4, 1024 * 8 * 64, f); fclose(f);
\t  unsigned nstuck = 0;
\t  for (unsigned i = 0; i < 1048576; i++) if (hstuck[i]) nstuck++;
\t  fprintf(stderr, "BH-GATE ledgers dumped stuck-lanes=%u\\n", nstuck);
"""
s = s[:start] + new_block + s[end:]
open(p, "w").write(s)
print("KERNEL2-P11b-OK")

# record slot->line map for quasi_newton.cuh (call order = slot order)
import re, subprocess
r = subprocess.run(["ssh", "ai_cloud",
    f"cd {Q} && awk '/conf_barrier\\(\\(conf_id\\), 200[0-9][0-9]\\)/{{print NR}}' inc/cuda/quasi_newton.cuh"],
    capture_output=True, text=True, timeout=60)
lines = [int(x) for x in r.stdout.split()]
open("/tmp/bh_quasi_slot_lines.txt", "w").write("\n".join(f"{i} {ln}" for i, ln in enumerate(lines)))
print("QUASI slot->line:", list(enumerate(lines)))
