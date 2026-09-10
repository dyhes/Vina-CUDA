#!/bin/bash
cd ~/grad/variants/QuickVina2-CUDA-m1_fusion
export PATH=/usr/local/cuda-11.8/bin:$PATH
D=input_file_example
rm -f /tmp/v1_1u1b.log
timeout 150 ./QuickVina2-GPU-2-1-CUDA \
  --config $D/1u1b_config.txt --thread 1000 > /tmp/v1_1u1b.log 2>&1
echo "EXIT=$?"
grep -A2 "mode |" /tmp/v1_1u1b.log | head -4
echo "ERROR_count=$(grep -c ERROR /tmp/v1_1u1b.log)"
tail -2 /tmp/v1_1u1b.log
