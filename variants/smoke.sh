#!/bin/bash
# usage: bash smoke.sh <variant_dir> <label>
DIR="$1"; LABEL="$2"
cd ~/grad/variants/$DIR
export PATH=/usr/local/cuda-11.8/bin:$PATH
D=input_file_example
rm -f /tmp/${LABEL}_1u1b.log
timeout 200 ./QuickVina2-GPU-2-1-CUDA \
  --config $D/1u1b_config.txt --thread 8192 > /tmp/${LABEL}_1u1b.log 2>&1
echo "[$LABEL] EXIT=$?"
grep -A3 "mode |" /tmp/${LABEL}_1u1b.log | head -5
echo "[$LABEL] ERROR_count=$(grep -c ERROR /tmp/${LABEL}_1u1b.log)"
tail -2 /tmp/${LABEL}_1u1b.log
