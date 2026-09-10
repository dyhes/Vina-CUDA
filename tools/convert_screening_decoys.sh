#!/bin/bash
# Batch convert screening decoys mol2 -> pdbqt
# 57 targets × 285 decoys = 16,245 files

DECOY_DIR=/home/ubuntu/grad/dataset/CASF-2016/decoys_screening
OUT_DIR=/home/ubuntu/grad/dataset/CASF-2016-PDBQT/decoys_screening
OBABEL=/usr/bin/obabel

mkdir -p "$OUT_DIR"

total=0
converted=0
failed=0

for target_dir in "$DECOY_DIR"/*/; do
    target=$(basename "$target_dir")
    out_sub="$OUT_DIR/$target"
    mkdir -p "$out_sub"
    
    for mol2 in "$target_dir"*.mol2; do
        [ -f "$mol2" ] || continue
        base=$(basename "$mol2" .mol2)
        pdbqt="$out_sub/${base}.pdbqt"
        total=$((total + 1))
        
        # Skip if already converted
        if [ -f "$pdbqt" ]; then
            converted=$((converted + 1))
            continue
        fi
        
        $OBABEL "$mol2" -opdbqt -O "$pdbqt" 2>/dev/null
        if [ $? -eq 0 ] && [ -f "$pdbqt" ]; then
            converted=$((converted + 1))
        else
            failed=$((failed + 1))
            echo "FAIL: $mol2"
        fi
    done
    echo "[$target] done"
done

echo ""
echo "Total: $total, Converted: $converted, Failed: $failed"
