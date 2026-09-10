f = "run_experiment.py"
s = open(f).read()
anchor1 = '"m1_fusion_p1_c":     VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_c/QuickVina2-GPU-2-1-CUDA",'
add1 = anchor1 + '\n    "m1_fusion_p1_prefetch": VARIANTS_DIR / "QuickVina2-CUDA-m1_fusion_p1_prefetch/QuickVina2-GPU-2-1-CUDA",'
assert anchor1 in s
s = s.replace(anchor1, add1, 1)
anchor2 = '"m1_fusion_p1_c",\n                            "m1_fusion_p1_c_batch",\n                            "vina-gpu2", "qvina2-gpu2"]'
add2 = '"m1_fusion_p1_c",\n                            "m1_fusion_p1_c_batch",\n                            "m1_fusion_p1_prefetch",\n                            "vina-gpu2", "qvina2-gpu2"]'
assert anchor2 in s
s = s.replace(anchor2, add2, 1)
anchor3 = '                    "m1_fusion_p1_c",\n                    "m1_fusion_p1_c_batch",\n                        "vina-gpu2", "qvina2-gpu2"]:'
add3 = '                    "m1_fusion_p1_c",\n                    "m1_fusion_p1_c_batch",\n                    "m1_fusion_p1_prefetch",\n                        "vina-gpu2", "qvina2-gpu2"]:'
assert anchor3 in s
s = s.replace(anchor3, add3, 1)
open(f, "w").write(s)
print("OK: m1_fusion_p1_prefetch registered")
