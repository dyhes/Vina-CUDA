# Q6-IR 架构约束解决方案: 跳过 CPU refine — 2026-09-12

## 核心发现

**CPU `refine_structure` 可以安全跳过。** 三锚点验证:
- 1pxn: -8.9(Q5: -9.0/-9.1), 1c5z: -5.9(Q5: -5.9), 3ag9: -5.2(Q5: -5.6/-5.0)
- 所有 affinity 差异 ≤0.4 kcal = MC 噪声带内
- **GPU BFGS 已经充分收敛, CPU refine 是冗余的二次优化**

这意味着: **不需要将 CPU refine 移到 GPU** — 直接跳过即可。

## 对 FP16-state 路线的影响

E1-retry 失败的根因(m_cl 桥接 host FP32 refine 和 device FP16 kernel)
被 **-DNO_REFINEMENT 完全消除**:
- 没有 host 端 refine → 没有 host-device 精度冲突
- m_cl 可以安全地在 device 端用 __half
- **Q6-IR 双 kernel 方案的架构约束被解除**

## 修订后的 Q6-IR 架构

```
GPU kernel_A (FP16, shared ~7100B, 13 块, 26 warps):
  N_A 步 BFGS, m_coords/minus_forces 用 __half
  
GPU kernel_B (FP32, shared 12684B, 7 块, 14 warps):
  1 步 FP32 精确重置
  
→ DtoH 最终结果 (FP32 output_type_cl)
→ CPU: 仅 PDBQT 格式化输出(无 BFGS)
```

**不再需要 host 端 BFGS** → m_cl 的 m_coords/minus_forces 可以
安全地在 device 端用 __half → shared 减半 → occupancy 增益可达成。

## 执行计划

1. Q6-IR kernel_A: FP16 m_coords/forces, 7100B shared, 13 blocks
2. Q6-IR kernel_B: FP32 reset, 12684B shared, 7 blocks
3. MC 循环拆分: A×(N_A-1) → B×1 → A×(N_A-1) → B×1 → ...
4. 门禁: 冒烟 + 2×CASF
预期: k2 -15~25%(大配体 -25~35%), rt -5~10%

## 但这也改变了搜索语义

- **-DNO_REFINEMENT**: 跳过 CPU refine = 跳过精确评分的二次校正。
  三锚点显示 affinity 在噪声带内, 但**全量 CASF 需要验证** —
  某些配体可能在 CPU refine 中获得了显著改善
- 建议: 先跑 2×CASF (NO_REFINEMENT), 对比 Top-1 和 aff 分布,
  确认无误后再纳入主线
