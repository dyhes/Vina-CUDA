#!/bin/bash
# thread-count sweep for the pipe build (clean env), 96-ligand 1e66 batch.
set -u
export PATH=/usr/local/cuda-11.8/bin:$PATH
BIN=~/grad/variants/QuickVina2-CUDA-v6b_pipe/QuickVina2-GPU-2-1-CUDA
PROT=$(find ~/grad/dataset/CASF-2016-PDBQT/coreset/1e66 -name "*protein*.pdbqt" | head -1)
for TH in 1 2 4 8; do
  export OMP_NUM_THREADS=$TH
  export OMP_STACKSIZE=64M
  OD=/tmp/v6b_pipe_smoke/thr$TH
  rm -rf $OD; mkdir -p $OD
  T0=$(date +%s.%N)
  bash -c "ulimit -s 8192 && exec $BIN --receptor '$PROT' --ligand_directory '/tmp/v6b_pipe_smoke/ligs' --thread 8192 \
    --center_x 6.559 --center_y 74.413 --center_z 61.033 --size_x 20 --size_y 20 --size_z 20 \
    --output_directory '$OD'" > /tmp/v6b_pipe_smoke/thr$TH.log 2>&1
  RC=$?
  T1=$(date +%s.%N)
  echo "threads=$TH rc=$RC outs=$(ls $OD | wc -l)/96 wall=$(python3 -c "print('%.1f' % ($T1-$T0))")s"
  grep -E "PD_REFINE_S|E2E_TOTAL_S" /tmp/v6b_pipe_smoke/thr$TH.log | head -2
done
