#!/usr/bin/env python3
# v6b shape A/B full screening: run BOTH shapes over all 57 targets with the
# same driver (chunk 96), into separate experiment dirs, then compare EF + k2.
# Uses the resume flag: each run dir independent.
import subprocess, sys, time, os

BINS = {
    "W6cS": "/home/ubuntu/grad/variants/QuickVina2-CUDA-v6b_batch/QuickVina2-GPU-2-1-CUDA",
    "Q53S": "/home/ubuntu/grad/variants/QuickVina2-CUDA-v6b_w53/QuickVina2-GPU-2-1-CUDA",
}
order = sys.argv[1:] if len(sys.argv) > 1 else ["Q53S", "W6cS"]  # W6cS already has full data; run Q53S first
for tag in order:
    expdir = "/home/ubuntu/grad/experiments/Sep16_screen_%s" % tag
    if os.path.exists(expdir + "/v6b_screen.log"):
        n = sum(1 for l in open(expdir + "/v6b_screen.log") if "OK 285" in l)
        if n >= 57:
            print(tag, "already complete (57 OK), skip")
            continue
    print("=== running", tag, flush=True)
    t0 = time.time()
    subprocess.run(["python3", "/tmp/v6b_screen.py", BINS[tag], expdir, "96"])
    print(tag, "wall %.1f min" % ((time.time()-t0)/60), flush=True)
