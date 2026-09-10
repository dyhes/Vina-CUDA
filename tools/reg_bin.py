f = "run_experiment.py"
s = open(f).read()
anchor = '    "m1_fusion_p1_warpspec": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_warpspec/QuickVina2-GPU-2-1-CUDA",\n'
add = anchor + '    "m1_fusion_p1_gamma": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_gamma/QuickVina2-GPU-2-1-CUDA",\n'
assert anchor in s, "anchor not found"
s = s.replace(anchor, add, 1)
open(f, "w").write(s)
print("added m1_fusion_p1_gamma to BINARIES")
