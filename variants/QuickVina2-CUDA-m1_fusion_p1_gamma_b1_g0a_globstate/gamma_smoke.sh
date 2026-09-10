#!/bin/bash
cd ~/grad/variants/QuickVina2-CUDA-m1_fusion_p1_gamma
export PATH=/usr/local/cuda-11.8/bin:$PATH
D=input_file_example
rm -f /tmp/gamma_1u1b.log /tmp/gamma_1u1b_out.pdbqt
# config uses thread=8192; override to 256 for a fast-but-valid smoke
timeout 150 ./QuickVina2-GPU-2-1-CUDA \
  --config $D/1u1b_config.txt --thread 1000 > /tmp/gamma_1u1b.log 2>&1
echo "EXIT=$?"
grep -A2 "mode |" /tmp/gamma_1u1b.log | head -4
echo "ERROR_count=$(grep -c ERROR /tmp/gamma_1u1b.log)"
tail -3 /tmp/gamma_1u1b.log
ps aux | grep -c "[Q]uickVina2"
