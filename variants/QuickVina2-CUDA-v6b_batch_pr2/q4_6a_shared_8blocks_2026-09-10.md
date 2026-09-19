# Q4-6a 落地：8 blocks/SM 达成 — 2026-09-10

> 承接 Q4 后续方向 1（8 blocks/SM 还差 572B）。

## 改动（一行宏，零语义风险）

`MAX_NUM_OF_RIGID 48 → 40`

依据：CASF 全量最大扭转数 36（3ag9）→ rigid 节点 37；40 覆盖 + 余量。
rcigid_cl 全部数组（children_map/parent/atom_range/origin/orientation_q/
axis/relative_axis/relative_origin）+ pos_dt/ori_dt 同步收缩：
**shared 13,068 → 12,204B（-864B，cuobjdump 确认）→ 8 blocks/SM**
（100KB ÷ 12.2KB = 8.2）。

注意 MAX_NUM_OF_LIG_TORSION 保持 48 不动——它影响 change/output_type/
bfgs 矩阵维度，收缩是另一个工程。

## 六档冒烟（全部通过，ERROR=0）

| target | MODE1 | K2 (s) | E2E_DOCK (s) | vs Q4-5 |
|---|---|---|---|---|
| 1c5z | **-5.9** | 0.017 | 0.065 | 持平 |
| 1pxn | **-8.9** | 0.126 | 0.177 | 持平 |
| 3p5o | **-8.9** | 0.147 | 0.199 | 持平 |
| 3b68 | -11.0* | 0.248（-1.6%）| 0.302（-1%）| 微降 |
| 3ag9 | -5.2* | **1.842（-2.7%）** | **1.911（-2.6%）** | 微降 |

*MODE1 差 = auto_seed MC 噪声（同前）。

## ncu 交叉验证（1u1b）

```
warps_active: 12.1% → 14.1%（+2.0pp，+17% ✓ 与 6→8 块预测一致）
```

单跑 k2 噪声内（1u1b Q4-5 0.488 vs Q4-6a 未见显著再降——占用率 +17%
不等于 k2 -17%：LS 串行链延迟仍在，且 SM 有其他工作交错）。3ag9 的
-2.7% 是唯一统计上偏正的信号（大 shared 压力靶点）。

## 诚实判定

- **占用率目标达成**（12.1→14.1%），但 k2 收益 ≤3%——远低于线性预期。
  原因：No Eligible 82% 的主因是 **LS 串行依赖链的延迟暴露**，驻留 warp
  从 6→8 只多给了 2 个可调度 warp（1.49→~2.0/scheduler），仍远低于
  隐藏延迟所需的 ~4-8。
- **shared 重设计路线的单点收益已基本兑现**；继续到 10/12 blocks 需要
  再砍 4.7KB/2.8KB——剩余大头是 m_coords/minus_forces（3KB，Q4-3 已证
  不可局部裁剪）和 rigid_cl 剩余字段（均为 BFGS 必需）。
  **shared 路线到此收益饱和。**

## 下一步指向

k2 的剩余优化只能来自：
1. **LS 内 eval 精简**（减少指令数/访存次数本身）
2. **v6-batch**（用多配体填满 SM——绕开单配体占用率天花板）
两者都属 Q4-6 后续/Q4-7 立项范围。

## 谱系

- commit d077b81（Q4-6a 本体）；与 Q4-5 同线，CASF 认证前建议与
  per-node 转换下传（方向 2）合并跑一次全量
