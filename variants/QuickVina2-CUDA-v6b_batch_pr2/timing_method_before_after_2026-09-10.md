# 计时方法前后对比 — v4（旧方法）vs v4_8（新方法）— 2026-09-10

> 回答"前后计时方法差异"。两套方法的**代码边界**、**覆盖的成本项**、
> **输出字段**逐项对照，并用同一台 RTX 4090 + CASF-2016 全量的实测数据佐证。

## 一、旧方法（v0~v4_7 全部沿用）

### 1.1 三个层级的计时来源
| 字段 | 计时方 | 边界 |
|---|---|---|
| `runtime_s` | Python runner 秒表（`t0=time.time()` → `subprocess.run` → 差值） | 二进制完整进程生命周期：CUDA ctx init + host 准备 + 全部 HtoD + k1 + k2 + DtoH + 输出 + 退出 |
| `gpu_kernel1/2_s` | CUDA event 对 | kernel1 / kernel2 纯 GPU 执行段 |
| `gpu_htod/dtoh/memset/all_s` | 二进制内 gpu_timer 账本（v4 起有） | 仅被 GT 块包裹的 memcpy/memset；k1/k2 并入 all |

### 1.2 旧方法的三个缺陷
1. **无法拆分 host 与 GPU**：`runtime_s − gpu_all` 得到一个"host 大杂烩"
   （v4 上 = 1.107s/配体，87.9%→88.6% 之争都是在这个黑盒里打转），不知道
   里面是 ctx、随机数、填充、还是进度条。
2. **ctx init 混在 wall 里**：一次性环境成本与 per-ligand 成本混同；
   任何"减少 host 开销"的优化收益都被稀释、且无法归因。
3. **runner 自身开销不可见**：python/sudo/fork 每配体一份，全部计入
   runtime_s，既污染测量、也污染优化决策（我们一度误判它值 0.8s/配体，
   实测 0.04s——见下）。

## 二、新方法（v4_8 起启用）

### 2.1 改动清单
| # | 改动 | 位置 |
|---|---|---|
| 1 | **CUDA ctx warmup**（cudaSetDevice + memset + 空 kernel + 256MB alloc warm + sync），一次性环境成本显式剥离 | `kernel2.cu v7_warmup_cuda()`，main 入口调用 |
| 2 | **二进制内 E2E 三口径**：E2E_TOTAL（进程启动→结束，含 warmup）/ E2E_WARM（warmup 自身）/ E2E_DOCK（warmup 后→结果就绪，**论文主口径**） | `main.cpp`，CLOCK_MONOTONIC |
| 3 | **统一 host 探针** `host_probe.h`：单时间线（extern 状态存 kernel2.cu，修掉两 TU static 分裂），~20 个相位 mark，`-DHOST_PROBE` 激活 | main.cpp + main_procedure_cu.cpp |
| 4 | **runner 解析**：summary.csv 10 列 → 13 列（+e2e_total_s/e2e_warm_s/e2e_dock_s），旧二进制留空向后兼容 | `run_experiment.py` |

### 2.2 新旧覆盖项逐项对照（哪些成本进哪个口径）

| 成本项 | 旧 runtime_s | 新 E2E_DOCK | 备注 |
|---|---|---|---|
| 进程 spawn（python/sudo/fork） | ✅ 计入（污染） | ❌ 排除 | 实测仅 0.04s，旧方法高估为 0.8s |
| **CUDA ctx init + module load** | ✅ 计入（污染） | ❌ **显式 warmup 排除** | 一次性环境成本；warmup 耗时单列 e2e_warm_s（实测 0.081s） |
| parse receptor/ligand | ✅ | ✅ | per-target 必要成本 |
| pre_gpu_setup / cache / prec | ✅ | ✅ | per-receptor 必要成本（不摊薄借口） |
| rand_maps 生成（host→GPU 化后消失） | ✅ | ✅（优化前 214ms→现 1.2ms） | 本轮最大 host 收益来源 |
| ric 随机构象生成 | ✅ | ✅（GPU 化后 host 侧消失） | |
| grids/ar/pa/pre 填充 + HtoD | ✅ | ✅ | per-receptor（redocking 无法摊薄） |
| kernel1 / kernel2 | ✅ | ✅ | |
| DtoH + 输出 PDBQT 写盘 | ✅ | ✅ | 结果交付一部分 |
| 进度条线程 join（sleep 量子） | ✅ 计入（**0.5s UI 污染**） | ✅ 计入但已修（量子 1s→100ms） | 旧方法里它是隐形大头 |

### 2.3 新方法的恒等式（v4_8 CASF 全量中位数实测）

```
runtime_s (0.640) = e2e_total (~0.52) + runner 开销 (0.040) + 残差
e2e_total = e2e_warm (0.081) + e2e_dock (0.412) + 收尾 (~0.03)
e2e_dock (0.412) = GPU 时间线 (0.337) + 二进制内 host (~0.075)
```

旧方法下 1、2、4 三项全部埋在 `runtime_s` 里不可区分；新方法全部显式。

## 三、同数据实测对照（CASF-2016 全量，同一 GPU，median）

| 版本 | 方法 | rt | e2e_dock | k2 | k2/rt | host 侧可解释性 |
|---|---|---|---|---|---|---|
| v4（09-08）| 旧 | 1.520 | 不可测 | 0.135 | 8.9% | 黑盒 1.107s |
| v4_5（09-09）| 旧 | 1.440 | 不可测 | 0.136 | 9.4% | 黑盒 1.007s |
| v4_7（09-09）| **新** | 1.430 | 1.225 | 0.134 | 9.4% | 逐段可测（发现进度条 0.5s） |
| v4_8（09-10）| **新** | **0.640** | **0.412** | 0.063* | 9.9%（median 口径）| 黑盒缩至 0.075s |

*v4_8 k2 median 0.063 vs v4 0.135：host 提速后小 k2 靶点占比改变（长尾 1u1b 类
0.59s 不变），论文引用 k2 时需统一 mean/median 并重算。

### 3.1 优化收益的归因对比（这轮真正赚到的）
- rand_maps host 生成 214ms → 1.2ms（hoist，boost variate_generator 每抽一构）
- ric randomize 213ms → GPU 生成（curand philox，ric_gpu.cu）
- 写零 HtoD（result_coords 12MB + result_gpus 5MB）删除
- 进度条 join 量子 1s → 100ms（-0.5s UI 污染）
合计把 e2e_dock 1.225 → 0.412；rt 1.430 → 0.640。

**若用旧方法测量**：rt 1.43→0.64 的收益会与"runner 波动"混在同一列数字里，
无法回答"收益来自哪一项"——这正是新方法的核心价值。

## 四、Top-1 口径补充（v4_8 认证的遗留项）

- v4_8：all-atom 169 / heavy 170（v4 家族历史带 171-184，median 174）
- v4_8 落在带下沿外 2 个：随机数实现变更（去 omp-race、串行化）→ 不同 MC 实现；
  v4 家族自跑极差 13，单次 169 不构成退化证据，但**必须按中位数±极差口径报告**
- 建议：v4_8 与 v4 各补 1-2 次运行取中位数对比（每次 ~10 分钟）

## 五、口径使用建议（论文）

1. **主口径 = e2e_dock**（warmup 排除已在文中声明；与 Uni-Dock/Vina-GPU 的
   "端到端"惯例兼容，且更严谨——我们显式报告了 warmup 策略）
2. **复现口径 = e2e_total**（含 warmup，透明）
3. **kernel 证据 = k2 / gpu_all**（CUDA event 账本，nsys 校验偏差 <1%）
4. 旧 runtime_s 列保留向后兼容，但论文不再用它做除法分母
5. Top-1 用多次运行中位数 ±极差（171-184，极差 13），不用单次最大值

## 六、坑与陷阱（新增）

- **探针打印的是"上一段"的名字**：`mark(label)` 记录的是自上次 mark 以来的
  时长，所以打印的 label 是上一段的（读数时全体后移一位）——初版据此误判
  "ctx=232ms"，实际那是 model 拷贝+填充段
- **两 TU 各持 static probe 状态** → 时间线分裂、交错误归因（v7 中段踩坑），
  必须用 extern 单例（host_probe.h → kernel2.cu 定义）
- **fprintf 探针不能放进 per-iteration 循环**（20k 次循环 × stderr = 200ms 自污染）
- **make 增量构建可能跳过 .cpp 重编**：验证探针是否生效要用 `strings` +
  `make clean` 双保险
- **探针字符串 ≠ 二进制证据**：`hp_mark("x")` 经 printf("%s") 折叠后 rodata 里
  只剩裸串 "x"；grep "PROBE x" 会误报"探针丢失"

原始数据：
- 旧方法末代：`experiments/Sep09_2154_CASF-2016_v4_7/summary.csv`（13 列新表头，旧二进制语义）
- 新方法首发：`experiments/Sep10_0905_CASF-2016_v4_8/summary.csv`（13 列全）+ `rmsd_results.csv`（Top-1 169/170）
- 更早旧旧方法（10 列前）：`experiments/Jul27_0251_CASF-2016_v4_memcpy_single_m/summary.csv`（6 列）
