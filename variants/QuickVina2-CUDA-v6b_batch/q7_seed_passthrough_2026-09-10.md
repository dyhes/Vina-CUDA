# --seed 透传与确定性验证结果 — 2026-09-10

## 一、--seed 透传已落地

- 二进制本就支持 `--seed <int>`（main.cpp L693，0=auto_seed）
- runner 两条路径（run_qvina_w_cuda / run_cuda_vina）已加 `--seed` 参数
  透传 + argparse（`--seed`，默认 0=auto）
- 实测：seed=42 两次运行 "Using random seed: 42" 均生效 ✓

## 二、同 seed 位对比：**不逐位一致**

```
affinity（-9.0 / -8.9）与 rmsd 排序一致，但坐标末位不同
（ATOM 坐标差 ~0.001-0.01Å，mode2 rmsd 0.444 vs 0.346）
```

### 根因（源码定位）：`minus_forces` 跨线程非原子累加

`eval_interacting_pairs_deriv_update_optimal` 中，同一原子的
`minus_forces->coords[i]` 被所有含该原子的 pair 所在**不同线程**以
`-=` / `+=` 非原子写更新——线程调度顺序不确定 → 浮点加法不结合 →
末位抖动 → LS 阈值判定可能翻转 → 轨迹分叉。

这是 v1 遗留的**既有竞态**（Q4-5/Q4-6 未触碰该路径），也解释了此前
家族 Top-1 极差 13-16 的一部分来源。

## 三、对 Q4-7C（pairs 预筛）的影响

- C 的"同 seed 逐位对比"验证标准**不可达**（基线本身不确定）
- 降级为可执行标准：
  1. 同 seed **affinity 分布统计一致**（mean|Δ| < 0.02，逐 target）
  2. 六档 MODE1 落同 seed 多跑的重复区间内
- 且 C 的收益判定不受影响（k2 时间与正确性分布）

## 四、附带产出（可选后续）

minus_forces 竞态的根治 = 分色累加（pair 按 atom 着色分桶，桶内串行）
或 atomicAdd_float（sm_70+ 原生支持，性能代价实测）——**独立立项
Q4-8（确定性/可复现性）**，对论文可复现性声明有价值，非性能项。

## 状态

- --seed 透传：✅ 已落地（binary + runner）
- 逐位确定性：❌ 既有竞态（非本轮引入），立 Q4-8 候选
- Q4-7C 实施：验证标准降级后可继续
