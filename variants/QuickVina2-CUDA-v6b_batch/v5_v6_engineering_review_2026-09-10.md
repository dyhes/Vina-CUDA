# v5 / v6 工程层优化检查报告 — 2026-09-10

> 定位：新论文框架（kernel 层 / 算法层 / 工程层）中，v5/v6 归**工程层**。
> 本报告基于源码 diff 实查（milestones/v4→v5→v6 逐文件比对）+ 既有认证数据，
> 作为 Q2 后续迭代的"可信思路"存档。合并目标 Q2-8.0 直接引用本文。

## 一、v5 是什么（= v45 主线的直系祖先）

**实查结论：v5 = v4 − stage_timer 插桩 + 两项工程层修改。**
（v4 的 milestones 拷贝含 9-08 加的 stage_timer；v5 是拆掉插桩后的干净工程版。）

diff 实证（v4→v5，34 行 host 改动 + kernel 微量）：

| # | 修改 | 位置 | 机制 |
|---|---|---|---|
| 1 | **Path B：m_cl 单份拷贝** | kernel2.cu `m_gpu = &mg[0]`（v4 是 `&mg[id]` per-thread）+ host `cudaMalloc(m_size)`（v4 是 `thread*m_size`）+ 单次 memcpy | 8192 线程共享同一 m_cl 副本（m_cl 内容对接期间只读），省 8191 份冗余 HtoD 与显存——**真流量 -30MB/配体** |
| 2 | **grids HtoD 瘦身** | main_procedure_cu.cpp：v4 整块 memcpyAsync(grids_cl 1.14GB) → v5 改 `cudaMemsetAsync` + 只传 header（atu/slope 每 grid 72B）| grid **数据体**由 kernel1 在 GPU 端现场插值生成，host 那份 1.14GB 本来就没用——**HtoD 1.14GB→KB 级** |
| 3 | 写零 HtoD 删除 | result_coords / result_gpus 的 memset-then-copy 全零上传删除 | 纯浪费流量 |
| 4 | stage_timer 拆除 | kernel1/kernel2 的 STAGE_* 全部移除 | 回归零插桩基线 |

认证数据（Sep08_1208，PathB_CLOSE.md）：285/285 OK，rt 1.320（-13% vs v4 1.520），HtoD 0.215→0.000，k2 0.136 持平，aff 噪声带内。Top-1 171（家族带内）。

**可信度**：高。三项都是"搬运策略"而非数值改动，质量等价有双跑认证。

## 二、v6 是什么（= v5 + GPU 级批，三层全在 host/kernel 接口层）

diff 实证（v5→v6，动 3 个文件）：

| # | 修改 | 位置 | 机制 |
|---|---|---|---|
| 1 | **kernel 批处理** | kernel2.cu：签名加 `num_ligands/threads_per_ligand/search_depths[]`；核心映射 `ligand_id = blockNumInGrid / threads_per_ligand; id = % threads_per_ligand; if (ligand_id >= num_ligands) return;`；取数全改 `ric[ric_offset+id]`、`mg[ligand_id]`、`random_maps[ligand_id]`、`torsion_sizes[ligand_id]` | GRID 64×128=8192 blocks 原本 1 block=1 restart，现重映射为 **(ligand_id, restart_id) 二维网格**——单次 launch N 配体 × 8192 restart 并行 |
| 2 | **动态 chunk** | main_procedure_cu.cpp：`cudaMemGetInfo` → `mem_per_ligand`(13.4MB) → `chunk_size = avail/mem_per_ligand`，**cap=64**（4090 实测最优）→ 分配 chunk 级大缓冲（ric/m/rm/coords 各 ×chunk_size）| 显存自适应 + chunk 循环复用同一批缓冲 |
| 3 | **Host 三段式** | 批量 HtoD → chunk 循环（launch + DtoH）→ 批量 free | 消除 N-1 次 `cudaDeviceSynchronize` + DtoH 往返串行 |

认证数据（handoff_2026-07-31/08-01）：单受体 285 配体 screening **380s→58s（6.6×）**，k2 总计 -19.2%，kernel2 占 wall 12.1%→63.9%；redocking（N=1）Top-1 与 v5 等价（差 1，波动内）；chunk cap 扫描 64 最优（128: 59s，285: 66.7s——反而差，chunk 复用缓冲胜过一次全分配）；cross-docking 10×285: 608s，胜 persistent 672s。

**可信度**：高。redocking N=1 时 `ligand_id=0、全部偏移=0`，kernel 数值路径与 v5 完全一致，质量等价有理论依据 + 双跑实证。

## 三、工程层方法论提炼（论文"工程层"一节的骨架）

v5+v6 一脉相承的三条原则：

1. **消灭冗余数据搬运**（v5-1/2/3）：先问"这份拷贝谁读？"——m_cl 8191 份冗余、
   grids 1.14GB 死数据、全零 result。均非算法改动，纯 IO 语义修正。
2. **资源自适应 + 分块复用**（v6-2）：显存查询 → 动态 chunk → 缓冲复用，
   把"能否装下"从编译期 MAX 宏搬到运行时决策（与 grids_cl 静态 1.14GB
   形成对照——后者是待修的同类工程债，见 prep_anatomy）。
3. **并行维度重映射**（v6-1）：不改 kernel 内部计算，只改 block 索引语义
   （restart → (ligand, restart)），用最小代码面积把 GPU 利用率从 ~12% 拉到 64%。

已试尽的工程层否定方向（49 个内，Q2 迭代勿重复）：
stream overlap（CPU 已被 kernel 隐藏）、配体排序（chunk 负载不均反证）、
模板化 MAX_ATOMS、persistent kernel（atomicAdd +21%）、进程并行 pool
（GPU 竞争破坏 BFGS 时序）、pinned memory 3 方案、warp-spec。

## 四、与 Q2 线的接口（Q2-8.0 合并时的已知冲突点）

| 冲突 | Q2-7 现状 | v6 | 合并方向 |
|---|---|---|---|
| ric/randmaps | GPU 生成（ric_gpu.cu，单配体 kernel）| host 批量生成 + 批量 HtoD | **保留 GPU 生成**，改批量化：一次 kernel 生成 N×8192（seed 加 ligand 维）|
| m_cl | mg[0] 单份（单配体语义）| mg[ligand_id] 数组 | 恢复数组布局（N=1 时退化等价，无质量风险）|
| result DtoH | per-ligand 同步读 | chunk 级批量 DtoH | 采 v6 三段式 |
| 进度条/探针 | Q2-7 已修 | v6 无 | 保留 Q2-7 |
| E2E 计时 | Q2-7 三口径 + per-ligand 待补 | 无 | 保留 Q2-7，补 per-ligand 分段 |

## 五、复查结论

- v5/v6 源码与认证报告**互相印证，无矛盾**；两版可直接作为工程层引用
- v6 的 kernel 改动面比预想小（kernel2.cu 签名+映射+取数偏移，计算体不动），
  Q2-8.0 合并工作量集中在 host 侧 ric/m_cl 布局对齐
- 论文可用数据：v5（rt -13%，HtoD→0）、v6（screening 6.6×，k2 占比 12→64%）
  —— 加上 Q2-7（e2e_dock -74.7%），工程层叙事完整
