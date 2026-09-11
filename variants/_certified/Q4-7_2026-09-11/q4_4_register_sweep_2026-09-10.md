# Q4-4 寄存器/占用率扫描结果 — 2026-09-10

> 假设（q4_ncu_kernel2）：k2 受 shared(14.8KB→6 blocks/SM) + regs(128) 双约束，
> 占用率 12.1%。若把 shared 降到 ≤12.5KB 可到 8 blocks；寄存器降低本身
> **不增加 blocks/SM**（shared 才是绑定约束），但可验证 spilling 代价。

## 一、寄存器上限扫描（1u1b，1 launch，无性能差异即符合预期）

| maxrregcount | 实际 REG | K2 (s) | E2E_DOCK (s) | MODE1 |
|---|---|---|---|---|
| 128（基线）| 128 | 0.494 | 0.552 | -7.0 |
| 112 | 112 | 0.498 | 0.557 | -7.0 |
| 96 | 96 | 0.511 | 0.570 | -6.7 |
| 80 | 80 | 0.505 | 0.563 | -7.0 |

（注意：128 基线 K2=0.494 比常态 0.557 略低——ncu 后单跑波动，同量级。）

**结论：寄存器上限在 80-128 区间对 k2 无影响**（±3% 噪声），无 spilling
（STACK 仅 22-56KB 为编译器临时，未爆）。证实 ncu 判断：
**shared 是绑定约束（6 blocks/SM），寄存器不是。**

## 二、shared 约束的数学

```
4090: shared/SM = 100KB（carveout 上限）
每块 static shared = 14,796B → 6 blocks/SM（6×14.8=88.8KB ≤ 100 < 7×14.8=103.6）
要 7 blocks：≤14.2KB（-0.6KB） ｜ 8 blocks：≤12.5KB（-2.3KB）
10 blocks：≤10KB（-4.8KB） ｜ 12 blocks：≤8.3KB
```

## 三、Q4-3 dynamic-shared 尝试的教训（已回滚，见 q4_3 报告）

m_coords/minus_forces 动态化破坏语义（cand.e 漂移 + quaternion ERROR +
k2 20×劣化）——shared 结构有跨步生命周期依赖，不能按"真实原子数"裁剪。

## 四、修正后的 Q4 kernel 层路线

**占用率 12% 的解锁需要 shared 从 14.8KB 降到 ≤12.5KB（-15%）**，
而静态组成里最大三项：
- m_gpu_ligand.rigid_cl 3.9KB（MAX_RIGID=32 数组，真实 4-17）
- m_coords+minus_forces 3.0KB（MAX_ATOMS=128，真实 13-55）
- orientation_m[32][9] 1.2KB（rigid_cl 内）

可行路径（design-first，Q4 立项）：
1. **rigid_cl 按 num_children 动态化**（-2KB，需 store_node/set/LS 全链
   偏移改造——v11 时代已验证树遍历语义敏感，需 design-first）
2. m_coords/minus_forces 动态化重试（上次失败在初始化语义：`m_coords =
   m_gpu->m_coords` 整结构拷贝被替换成了部分拷贝；正确做法是仅指针化 +
   保持 set_update 写路径不变——需要单变量逐个验证）
3. orientation_m[9] → 用四元数算（-768B/块）
4. **或绕开 shared 约束**：v6-batch 让相邻块的 k2 波次互相填空窗，
   占用率 12% 在批模式下不再致命（GPU 有其他 block 可跑）

## 五、结论

- 寄存器扫描证伪"regs 是约束"：**shared 独占绑定**
- 单配体 redocking 进一步提速必须：要么 shared 重设计（-15% 才 +33%
  occupancy，工程量中等、收益上限 ~33% k2 ≈ -25% dock），要么 kernel
  算法优化（LS 内 eval 精简）
- 工程层（host）已至 ~50ms 底；kernel 层方向已由 ncu+stage_timer 双工具
  定位：**occupancy（shared 约束）与 LS 串行链**
