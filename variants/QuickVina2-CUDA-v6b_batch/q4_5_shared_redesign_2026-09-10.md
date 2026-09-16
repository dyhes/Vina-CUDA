# Q4-5 shared 重设计立项 — design-first 文档 — 2026-09-10

> 目标：kernel2_update 的 shared 14,796B → ≤12,496B（8 blocks/SM，
> occupancy 12.1% → ~16%，k2 上限 -33%）。经源码逐行核查后，方案比
> 初稿更精准——**orientation_m 可以安全移除，其余 rigid_cl 字段必须保留**。

## 一、核查结论（改写了初稿判断）

### 1.1 rigid_cl 全字段使用情况（源码证据）

| 字段 | 大小(MAX_RIGID=48) | 写入方 | 读取方 | 能否删/降 |
|---|---|---|---|---|
| num_children | 4 | host store_node | set_update 循环界 | 保留 |
| children_map | 384 | host | **device 无读取**（仅 host 侧树构建用？待查）| **候选删** |
| parent | 192 | host store_node | set_update 遍历 | 保留 |
| atom_range | 384 | host store_node | set_update/mutate 原子区间 | 保留 |
| origin | 576 | set_update 每步重写 | local_to_lab | 保留 |
| **orientation_m** | **1728** | set_update 每步**重算**（quaternion_to_r3(oq)）| local_to_lab/_direction | **可删：由 oq 现场重算** |
| orientation_q | 768 | set_update 每步重写 + rotate | 树传播（parent_q）| 保留（结构核心）|
| axis | 576 | set_update 每步重算 | mutate 扭转轴 | 保留 |
| relative_axis/origin | 1152 | host 一次性（静态构象数据）| set_update | 保留 |

### 1.2 关键发现：orientation_m 是纯派生量

`set_update`（quasi_newton.cuh L600-660）对每个节点执行：
`quaternion_to_r3(orientation_q[node], orientation_m[node])`——
orientation_m 每一步都从 orientation_q **完整重算**，之后仅在
local_to_lab/_direction 中作为 3×3 矩阵乘系数消费。

**方案：删除 shared 的 orientation_m[48][9]（-1728B）**，在两个消费点
（local_to_lab / local_to_lab_direction 调用处）改为先调用
`quaternion_to_r3(oq[node], om_local[9])`（寄存器/局部 36B）再乘。
数学等价：同一 quaternion → 同一矩阵 → 同一乘积；无跨步状态。

收益核算：
- shared 14796 − 1728 = **13,068B** —— 还差 572B 才到 12,496（8 blocks）
- 继续砍：`children_map`（384B，device 端无读）→ 12,684B（差 188B）
- `d_test[38]`（152B，调试残留）→ 12,532B（差 36B）
- `bfgs_tmp[38]`/`d_test` 与 mutate 的 acc/counter 可合并复用（不同时
  存活：mutate 在 step 开头、bfgs_tmp 在 LS 内）→ 共用 union 再 -152B
  → **12,380B ≤ 12,496 ✓ 8 blocks/SM**

### 1.3 风险清单（v11/Q4-3 教训内化）

- orientation_m 移除涉及 set_update + g_evaluate_update 外所有
  local_to_lab* 调用点（mutate_conf.cuh 无使用，已核）
- local_to_lab 有两处内联展开的变体（L494/L515 手工展开版）——都要改
- children_map 删除前必须 grep device 全量确认零读取
- **冒烟门槛（用户要求，避免孤例）**：六档用例集 5 例全跑
  （1c5z/1pxn/3p5o/3b68/3ag9），affinity 与 Q3 逐例对齐（同 seed 差异
  仅来自 MC 实现——本项目不改 RNG 路径，期望 MODE1 逐位一致）
  + CASF 全量认证通过后才升 MILE

## 二、预期收益

- occupancy 12.1% → 16.1%（6→8 blocks/SM，+33%）
- k2 墙钟：驻留并发 ×1.33 → **k2 -25%（上限）**；e2e_dock -18%（1u1b 型）
- 1pxn（k2 139/197）：-35ms；1c5z（k2 16/66）：-4ms——小靶点几乎无感，
  收益集中在柔度大/原子多的靶点（与六档画像一致）

## 三、实施清单（Q4-5，预计一次会话）

1. [ ] grep 确认 children_map 在 device 端零读取 → 删
2. [ ] 删 shared orientation_m；set_update 内保留重算但写入局部；
       local_to_lab* 调用点改传 q+现场算矩阵（或直接改写两函数签名
       收 q，内部转 m——**推荐后者**，改动集中两函数）
3. [ ] d_test 删除；bfgs_tmp/acc/counter union 化
4. [ ] 编译 → 六档冒烟（5 例）逐例 MODE1 对齐 → ncu 复测 occupancy=16%
       与 shared=12,380B → CASF 全量 → CERT
5. [ ] 若 8 blocks 后 k2 仍受 regs/调度限制（ncu 复查），再做 orientation_q
       精度实验（float4 向量化，为下一步铺路）

## 四、更远的候选（本立项不做）

- orientation_q float4 向量化 + local_to_lab 手工展开已存在
- m_coords/minus_forces 动态化（Q4-3 失败教训：需先解耦整结构拷贝语义）
- v6-batch 合并（与本项目正交，Q4-6）
