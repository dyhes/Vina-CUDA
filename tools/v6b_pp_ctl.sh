#!/bin/bash
# Per-process screening mode (the OLD screening shape): one process PER LIGAND,
# same current binary (Q3 b725964e kernel), same box. 1 probe target (1e66).
set -u
export PATH=/usr/local/cuda-11.8/bin:$PATH
BIN=~/grad/variants/QuickVina2-CUDA-Q3/QuickVina2-GPU-2-1-CUDA
D=~/grad/dataset/CASF-2016-PDBQT/coreset
T=1e66
PROT=$(find $D/$T -name "*protein*.pdbqt" | head -1)
CLIG=$D/$T/${T}_ligand.pdbqt
C=$(python3 -c "
import sys; sys.path.insert(0,'/home/ubuntu/grad/tools')
from pdbqt_center import pdbqt_center
print(' '.join('%.3f'%v for v in pdbqt_center('$CLIG')))")
read cx cy cz <<< "$C"
OD=/tmp/v6b_pp_out; rm -rf $OD; mkdir -p $OD
T0=$(date +%s.%N)
n=0
for L in $(ls $D | sort); do
  LG=$D/$L/${L}_ligand.pdbqt
  [ -f "$LG" ] || continue
  bash -c "ulimit -s 8192 && exec $BIN --receptor '$PROT' --ligand '$LG' --thread 8192 \
    --center_x $cx --center_y $cy --center_z $cz --size_x 20 --size_y 20 --size_z 20 \
    --out $OD/${L}_out.pdbqt" > /dev/null 2>&1 &
  n=$((n+1))
  # cap concurrency at 1 (GPU-serial discipline): wait if too many children
  while [ "$(jobs -r | wc -l)" -ge 1 ]; do wait -n; done
done
wait
T1=$(date +%s.%N)
NOUT=$(ls $OD | wc -l)
WALL=$(python3 -c "print('%.1f' % ($T1-$T0))")
echo "PERPROCESS $T outs=$NOUT/285 wall=${WALL}s (per-ligand mean: $(python3 -c "print('%.2f' % (($T1-$T0)/$n))"))s"
