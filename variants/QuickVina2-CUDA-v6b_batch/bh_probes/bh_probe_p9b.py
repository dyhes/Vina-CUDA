#!/usr/bin/env python3
# P9 part2 v2: anchor respects the BHHOSTADDR line between memcpy and print
p = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3/inc/cuda/kernel2.cu"
s = open(p).read()

old = "\t{ unsigned int poison = 0; void *pp = 0, *lp = 0; cudaError_t e2, e3;"
new = "\t{ unsigned int hpoison[1024] = {0}; void *pp = 0, *lp = 0; cudaError_t e2, e3;"
assert old in s, "a1"
s = s.replace(old, new)

old = "\t  e2 = cudaGetSymbolAddress(&pp, d_bh_poison); e3 = cudaGetSymbolAddress(&lp, d_bh_last);"
new = "\t  e2 = cudaGetSymbolAddress(&pp, d_bh_poisonb); e3 = cudaGetSymbolAddress(&lp, d_bh_last);"
assert old in s, "a2"
s = s.replace(old, new)

old = """\t  cudaMemcpy(&poison, pp, 4, cudaMemcpyDeviceToHost);
\t  fprintf(stderr, "BHHOSTADDR %p\\n", lp);
\t  static unsigned long long hlast[1048576];
\t  cudaMemcpy(hlast, lp, 1048576*8, cudaMemcpyDeviceToHost);
\t  fprintf(stderr, "BH-GATE poison=0x%x\\n", poison);
\t  if (poison) {
\t      for (int dom = 0; dom < 8; dom++) if (poison & (1u << dom))
\t          fprintf(stderr, "BH-GATE stuck-domain %d\\n", dom);
\t  }"""
new = """\t  cudaMemcpy(hpoison, pp, sizeof(hpoison), cudaMemcpyDeviceToHost);
\t  fprintf(stderr, "BHHOSTADDR %p\\n", lp);
\t  static unsigned long long hlast[1048576];
\t  cudaMemcpy(hlast, lp, 1048576*8, cudaMemcpyDeviceToHost);
\t  int njam = 0; unsigned jammask = 0;
\t  for (int i = 0; i < 1024; i++) if (hpoison[i]) { njam++; jammask |= hpoison[i]; }
\t  fprintf(stderr, "BH-GATE jammed-blocks=%d dommask=0x%x\\n", njam, jammask);"""
assert old in s, "a3"
s = s.replace(old, new)
open(p, "w").write(s)
print("KERNEL2-P9b2-OK")
