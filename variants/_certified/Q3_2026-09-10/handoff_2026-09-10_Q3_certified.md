# Q3 里程碑版 — 合入 v5 遗留 + Q2→Q3 变更总结 — 2026-09-10

> Q3 = Q2-7（v4_8b）+ grids header-only HtoD（v5 遗留项回归）
> 目录：`variants/QuickVina2-CUDA-Q3`（fork@Q2-7，git 全程 commit）
> runner：`--variant Q3`；实验 `experiments/Sep10_1358_CASF-2016_Q3/`（285/285 OK，k2 字段 285/285 完整）

## 一、Q2 → Q3 的全部变更（就一项，<10 行）

**Q3-1 grids header-only HtoD**（v5 思路回归，且比 v5 更干净）：
- 删：整块 `cudaMemcpyAsync(grids_gpu, grids_ptr, 1.14GB)`——拷的是只填了
  header 的 host 结构，m_data 全是未初始化堆垃圾
- 删：v5 版本里伴随的 `cudaMemsetAsync(1.14GB)`（kernel1 是 m_data 唯一写入者，
  零初始化同样冗余）
- 换：`atu+slope` 8B + 每 grid 72B 元数据 ×17 —— **总传输 ~1.2KB**

为什么成立：grid 数据体的唯一生产者 = kernel1（GPU 端现场插值），唯一
消费者 = kernel2（读 GPU 侧）。host 那份 m_data 从生到死没人读。

## 二、单配体 redocking 认证数据（CASF-2016 全量 median）

| 口径 | Q2（v4 基线）| Q2-7（v4_8b）| **Q3** | Q3 vs Q2 |
|---|---|---|---|---|
| rt（runner wall）| 1.520 | 0.600 | **0.410** | **-73%** |
| e2e_total | — | 0.552 | 0.352 | — |
| **e2e_dock（主口径）** | ~1.2* | 0.384 | **0.181** | **-85%** |
| gpu_all | 0.368 | 0.347 | **0.145** | -61% |
| k2 | 0.135 | 0.127 | 0.128 | 持平 ✓ |

*Q2 时代无 e2e 口径，1.2 系 v4_7 首测值。

e2e_dock 拆解（Q3）：GPU 时间线 0.145 + host ~0.036 —— **GPU 占比从 v4 的
9% 升至 80%**，host prep 底座从 ~350ms 压到 ~36ms（grids 通路修复 + 探针
变量对齐后的真值，见四）。

## 三、正确性

- aff vs v4：mean|Δ|=0.162，>0.5 共 24 —— 与 v4 自身噪声（0.162/23）持平 ✓
- aff vs Q2-7：0.160（纯 MC 实现噪声量级，HtoD 路径不影响数值）✓
- k2 0.128 vs 0.127：kernel 数值路径未动 ✓
- Top-1：all-atom 168 / heavy 168 —— 低于家族带下沿（171-184）3 个。
  家族带本就宽（极差 13-16），且各次运行 seed=auto_seed()（时钟），
  168 属带外扰动而非退化信号；**最终论文数字建议按"多跑中位数±极差"报**
- smoke 锚点：1pxn -8.8 / 1c5z -5.9 / 1u1b -7.0，ERR=0

## 四、构建/探针变量说明（诚实记录）

Q3 的 e2e_dock=0.181 相比 Q2-7 的 0.384 跳变幅度（-203ms）**大于**
垃圾 HtoD 的直接账面收益——因为对比还叠加了：Q2-7 认证版含进度条线程
（100ms 量子 join），Q3 fork 自"进度条已移除 + 探针三进三出"的最新工作区。
单要素归因以 smoke 对账为准（1pxn 同探针下 0.388→0.193，其中 grids 通路
-195ms 与 PREP 探针实测一致：h2d_start/done 双双归零）。

## 五、里程碑与谱系（按目录纪律 v2 执行）

```
Q2-4 (milestones/v4) ─ Q2-5 (v4.5) ─ Q2-6 (v4_7) ─ Q2-7 (v4_8b, CERT✓)
                                                        │ fork
                                        Q3 = Q2-7 + v5 grids 遗留 (CERT✓)
                                                        │ 后续
                        Q4 规划 = Q3 + v6 batch kernel（GPU 级批，screening 6.6×）
```

- 目录：`variants/QuickVina2-CUDA-Q3`（fork commit b94c2fa → Q3-1 commit）
- CERT 快照：`variants/_certified/Q3_2026-09-10/`（待 tar）
- runner 已注册 `--variant Q3`

## 六、后续（不并入 Q3）

- **v6-batch（Q4）**：GPU 级批（screening 6.6× 已证）——独立立项，届时
  grids header-only 通量改造与批共享通路一并 design-first
- **B. grids 真实 dims**（1.14GB→~123MB）：redocking 单配体下收益仅剩
  malloc 挂账（Q3 已消除其显性成本），优先级降至低
- 单配体 redocking 剩余可挖：entry-gap ~100ms（ar_cl 4MB malloc/streams）、
  kernel2 本体（k2 0.128，属 kernel 层主线）
