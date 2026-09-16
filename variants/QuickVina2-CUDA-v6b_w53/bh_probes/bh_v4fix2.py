#!/usr/bin/env python3
# v4 fix v4: remove v3 leftovers in kernel2.cu (gate zeroing + BH_GATE_BYTES)
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"
p = Q + "/inc/cuda/kernel2.cu"
s = open(p).read()
import re
# remove the whole v3 zeroing block (uses bh_gate_ptr)
old = """\tif (local_tid == 0) {
\t\tfor (int w = 0; w < (MAX_NUM_OF_LIG_PAIRS + 63) / 64; w++) q4_skip_row[w] = 0ULL;
\t\tq4_epoch_i = 0;
\t\tunsigned int *gbase = bh_gate_ptr(conf_id, 0); // zero THIS conf's 64 gates
\t\tfor (int i = 0; i < BH_SLOTS; i++) gbase[i] = 0u;
\t}"""
assert s.count(old) == 1, "zero block"
s = s.replace(old, """\tif (local_tid == 0) {
\t\tfor (int w = 0; w < (MAX_NUM_OF_LIG_PAIRS + 63) / 64; w++) q4_skip_row[w] = 0ULL;
\t\tq4_epoch_i = 0;
\t}""")
# dyn sizes back to arena-only
s = s.replace("bh_dyn_smem = BH_NCONF * sizeof(BhConfState) + BH_GATE_BYTES;",
              "bh_dyn_smem = BH_NCONF * sizeof(BhConfState);")
s = s.replace("cudaFuncAttributeMaxDynamicSharedMemorySize,\n\t\t(int)(BH_NCONF * sizeof(BhConfState) + BH_GATE_BYTES));",
              "cudaFuncAttributeMaxDynamicSharedMemorySize,\n\t\t(int)(BH_NCONF * sizeof(BhConfState)));")
open(p, "w").write(s)
print("KERNEL2 v4 clean (no v3 leftovers)")
