# Q4 收官 Handoff — QuickVina2-CUDA Q4-7 认证 + 确定性 + 待办 — 2026-09-11

> 新会话从本文件继续。所有路径已验证存在；数字全部来自 experiments/ 实测。

## 一、当前状态（一句话）

Q4-7 已认证（rt 0.340 / e2e_dock 0.118 / k2 0.067，vs v4 -78%/-90%/-50%，
Top-1 179 带中部），代码在 `variants/QuickVina2-CUDA-Q3`（commit 8a27349
+ bf328361 + 9800e4eb），CERT 快照 `_certified/Q4-7_2026-09-11/`。

## 二、全世代数据（CASF-2016 全量 median，285/285 OK）

| 版本 | 日期 | 关键改动 | rt | e2e_dock | k2 | Top-1 | aff\|Δ\|vs v4 |
|---|---|---|---|---|---|---|---|
| v4 | 09-08 | 基线 | 1.520 | — | 0.135 | 184/174* | — |
| v4.5 | 09-09 | 三修复 | 1.440 | — | 0.136 | 175 | 0.141 |
| v4_7 | 09-09 | E2E计时+hoist | 1.430 | 1.225 | 0.134 | 170 | 0.183 |
| v4_8 | 09-10 | ric/randmaps GPU化 | 0.640 | 0.412 | 0.063† | 169 | 0.166 |
| v4_8b | 09-10 | 进度条100ms | 0.600 | 0.384 | 0.127 | 175 | 0.166 |
| Q3 | 09-10 | grids header-only HtoD | 0.410 | 0.181 | 0.128 | 168 | 0.162 |
| Q4-6 | 09-10 | MAX_RIGID 40 + 转换下传 | 0.400 | 0.175 | 0.125 | 177 | 0.158 |
| **Q4-7** | 09-11 | **精度域清理** | **0.340** | **0.118** | **0.067** | **179** | **0.155** |

*v4 两跑极差；†v4_8 k2 作废（缺字段 150/285）。

## 三、Q4-7 核心发现（论文素材）

1. **精度域污染**（新发现，论文级）：`exp()`（非 expf）、
   `printf("%f", double_expr)`、double 宏（M_PI/INFINITY）三类源把
   FP64 拉入热路径——SASS 审计（cuobjdump -sass | grep DFMA/DADD）
   定位，修复后 **k2 -46%、rt -15%**，affinity 分布不变（0.155 带内）
2. **k2/rt 占比演进**：8.9% → 9.4% → 21.2% → 30.0% → 31.2% → 19.7%
   （屏障修复让 k2 本体也 -46%）
3. **Q4-8 屏障修复是 k2 -50% 的真实来源**（受控实验：FP32/FP16 存储
   切换 k2 无差异；屏障缺失→完整 = 0.125→0.063）
4. **Q4-6 认证数字作废**：0.400/0.175/0.125 系屏障部分缺失构建，
   已被 Q4-7 取代

## 四、Q4-8 确定性：已修/未根除

- ✅ pairs 累加 edge-coloring + 按色同步
- ✅ skip_row atomicAnd/Or（RMW 竞态）
- ✅ 3 个 __syncthreads（set_update→GRID、intra→POT、POT→caller）
- ✅ --seed 透传（binary+runner）
- ❌ 未根除：同 seed 双跑 mode2-9 仍有差异（racecheck 49 处 mutate
  分支写 + m_coords 跨 eval 同步缺失）；影响轨迹低位，CASF 分布统计
  有效；完全根治需 1-2 会话（全 shared 写审计）

## 五、新会话待办（按优先级）

### P0 — 立即可做
1. **论文素材整理**：Q4 全世代表 + 精度域污染发现 + 三层框架
   （工程/内存/精度域）已齐备，可写论文 §优化 章节
2. 需更新：论文 Ch4 计时口径段（E2E 三口径 + 多跑中位数±极差声明）

### P1 — 下一轮性能迭代
3. **Q4-9 v6-batch 合并**（screening 6.6×）：v6 batch_chunk 代码在
   `variants/QuickVina2-CUDA-batch_chunk`；与 Q4 线正交；合并时注意
   ric/randmaps 已 GPU 化（v6 的批量 HtoD 路径需适配）
4. **Q4-8 续（确定性根治）**：全 shared 写审计（racecheck 72→0）；
   价值 = 精确 A/B + 可复现性声明；建议在 v6-batch 合并后做

### P2 — 可选
5. M2 FP16 精度归因（被 Q4-8 续阻塞）
6. M4 half2 数学（同上）
7. 剩余 225 条 FP64 SASS（rintf 路径，收益递减）

## 六、关键基础设施（新会话直接用）

- runner：`~/grad/tools/run_experiment.py --variant Q3 --dataset CASF-2016`
  （`--seed N` 已支持透传）
- 构建：`cd variants/QuickVina2-CUDA-Q3 && export PATH=/usr/local/cuda-11.8/bin:$PATH && make OPTION="-DHOST_PROBE"`
- 冒烟：六档用例集（1c5z/1pxn/3p5o/3b68/3ag9）脚本
  `/tmp/q7c_smoke2.sh`（可复用，注意其中 Q46 对比列已过时）
- ncu：免 sudo（RmProfilingAdminOnly=0 已生效）
- CERT：`variants/_certified/Q4-7_2026-09-11/`（58 项）
- GPU 串行纪律：CASF/ncu/smoke 绝不并行

## 七、坑与教训（新会话必读）

1. **探针 mark 打印"上一段"**——跨函数边界的段归靠猜会错，用
   绝对时间戳模式（T=）或直接插桩
2. **冒烟对比必须同构建**——M2 的 -50% 被我误归因 FP16，实为 Q4-8
   屏障（用户抓出）；跨构建比较一律无效
3. **暖页冒烟看不见进程级成本**——VMM first-touch/ctx 只在全量
   （冷进程×285）暴露（Q4-1 教训）
4. **summary.csv 字段完整性**——统计前必查 N（v4_8 的 k2=0.063 作废）
5. **extern __device__ 跨 TU 不可靠**（nvcc 11.8 no-rdc）——设备
   符号定义+使用必须在同一 TU，或用参数传递
6. **stage_timer v1 数值正确**（ns 单位曾误读）；K2WALL 桩已入树
7. **已知残余非确定性**：同 seed 双跑 mode2-9 有差异（MC 噪声量级），
   分布口径有效；论文声明已备

## 八、报告索引（全部双端：本地 ~/Documents/Graduation/literature/handoffs/
 远端 variants/QuickVina2-CUDA-Q3/）
- q_generations_table_2026-09-11.md（本表）
- q4_7_certified_2026-09-11.md（Q4-7 认证）
- q4_8_final_2026-09-11.md（确定性终态）
- q4_8_benefit_analysis_2026-09-11.md（根治收益分析）
- q4_7m_mixed_precision / q4_7m_fp64_source / q4_7m_m2_final（混合精度系列）
- q4_7_summary（LS eval 四方向排查）
- q7_seed_passthrough（--seed 基础设施）
- timing_method_before_after_2026-09-10.md（计时体系）
- smoke_suite_2026-09-10.md（六档用例集）

## 九、当晚勘误（复验后追加，2026-09-11 20:10）— 必读

**全量复验发现：Q4-7 认证有效，但认证之后的提交（bf328361 着色）存在静默回归。**

- CERT 二进制二跑（Sep11_1955_q47cert）：rt 0.340 / e2e_dock 0.119 / k2 0.067 /
  Top-1 176 → **认证声明复现，CERT 快照仍为当前最优与论文口径**。
- HEAD（含 bf328361+32e551fe）全量：Top-1 167，aff vs v4 mean|Δ| 0.291
  （=1.8× 自噪声）且系统性正向偏移 → **分布中性被破坏**。
- 根因：bf328361 的着色管线两处断线（mis 写在 h2d 后 + d_colors 从未
  接到 q4_pair_color 设备符号）→ 设备端 num_colors=0 → 色循环零次 →
  intra-pairs 整体跳过。接线修复（q4_8fix_proposed/）后循环真实执行，
  但分布仍偏（0.293）→ 色序累加本身不满足分布中性。
- **行动项（P0 插入）**：~~Q3 主线回退 bf328361 的着色块~~
  **【已完成 20:35】** 选择性回退落地为提交 8613ee32（Q4-8R）：
  着色管线全拆（设备全局/色循环/参数管道/主机块/mis 字段），
  atomic skip_row RMW + 3 屏障 + M2/FP16 保留。全量门禁通过：
  rt 0.340 / e2e_dock 0.119 / k2 0.068 / **Top-1 178**（回带）/
  aff vs v4a **0.158 对称**（≈自噪声 0.162）。实验
  Sep11_2030_CASF-2016_Q3。当前 Q3 主线 = Q4-8R，与 CERT 快照同分布水平。
- 详见 `q4_7_revalidation_and_q4_8_regression_2026-09-11.md`（双端已同步）。
- runner 新增 `--variant q47cert`（CERT 快照二进制直跑）。
