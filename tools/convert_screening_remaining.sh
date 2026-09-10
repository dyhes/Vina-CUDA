#!/bin/bash
DECOY_DIR=dataset/CASF-2016/decoys_screening
OUT_DIR=dataset/CASF-2016-PDBQT/decoys_screening

cd ~/grad
for target_dir in "$DECOY_DIR"/*/; do
    target=$(basename "$target_dir")
    out_sub="$OUT_DIR/$target"
    # skip if already done
    if [ -d "$out_sub" ] && [ $(ls "$out_sub"/*.pdbqt 2>/dev/null | wc -l) -ge 280 ]; then
        echo "SKIP $target (already done)"
        continue
    fi
    echo "=== $target ==="
    python3 tools/convert_to_pdbqt.py \
        --ligand-dir "$DECOY_DIR/$target" \
        --output-dir "$OUT_DIR/$target" \
        --obabel-cmd /usr/bin/obabel > /dev/null 2>&1
    count=$(ls "$out_sub"/*.pdbqt 2>/dev/null | wc -l)
    echo "  -> $count pdbqt"
done
echo "ALL DONE"
