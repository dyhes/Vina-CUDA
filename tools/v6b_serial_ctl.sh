#!/bin/bash
# Serial (pre-batch) baseline: Q3 binary (b725964e, same kernel code as v6b)
# using --ligand_directory = per-ligand kernel_monte + internal sync each.
# 3 probe targets, 285 ligands each, one process per target (prep shared).
set -u
export PATH=/usr/local/cuda-11.8/bin:$PATH
BIN=~/grad/variants/QuickVina2-CUDA-Q3/QuickVina2-GPU-2-1-CUDA
md5sum $BIN | cut -c1-8
D=~/grad/dataset/CASF-2016-PDBQT/coreset
O=/tmp/v6b_serial_ctl; mkdir -p $O
for T in 1e66 2vw5 3ag9; do
  PROT=$(find $D/$T -name "*protein*.pdbqt" | head -1)
  CLIG=$D/$T/${T}_ligand.pdbqt
  C=$(python3 -c "
import sys; sys.path.insert(0,'/home/ubuntu/grad/tools')
from pdbqt_center import pdbqt_center
print(' '.join('%.3f'%v for v in pdbqt_center('$CLIG')))")
  read cx cy cz <<< "$C"
  BD=$O/ligs_$T; rm -rf $BD; mkdir -p $BD
  i=0
  for L in $(ls $D | sort); do
    LG=$D/$L/${L}_ligand.pdbqt
    [ -f "$LG" ] || continue
    cp "$LG" $BD/${L}.pdbqt; i=$((i+1))
  done
  OD=$O/out_$T; rm -rf $OD; mkdir -p $OD
  T0=$(date +%s.%N)
  bash -c "ulimit -s 8192 && exec $BIN --receptor '$PROT' --ligand_directory '$BD' --thread 8192 \
    --center_x $cx --center_y $cy --center_z $cz --size_x 20 --size_y 20 --size_z 20 \
    --output_directory '$OD'" > $O/${T}.log 2>&1
  RC=$?
  T1=$(date +%s.%N)
  K2S=$(grep "GPU_KERNEL2_TIME" $O/${T}.log | awk '{s+=$NF} END {printf "%.2f", s}')
  NK2=$(grep -c "GPU_KERNEL2_TIME" $O/${T}.log)
  NOUT=$(ls $OD | wc -l)
  WALL=$(python3 -c "print('%.1f' % ($T1-$T0))")
  echo "SERIAL $T rc=$RC outs=$NOUT/$i k2launches=$NK2 k2sum=${K2S}s wall=${WALL}s"
done
