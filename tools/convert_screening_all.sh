#!/bin/bash
DECOY_DIR=dataset/CASF-2016/decoys_screening
OUT_DIR=dataset/CASF-2016-PDBQT/decoys_screening

cd ~/grad
for target_dir in "$DECOY_DIR"/*/; do
    target=$(basename "$target_dir")
    echo "=== $target ==="
    python3 tools/convert_to_pdbqt.py \
        --ligand-dir "$DECOY_DIR/$target" \
        --output-dir "$OUT_DIR/$target" \
        --obabel-cmd /usr/bin/obabel 2>&1 | tail -2
done
