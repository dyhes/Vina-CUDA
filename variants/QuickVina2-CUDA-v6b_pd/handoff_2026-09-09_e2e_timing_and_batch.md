# Handoff: 端到端口径修正 + 批量模式（v7 方向）— 2026-09-09

> 背景：HOST_PROFILE.md（同日）定位到 CASF wall 的 88% 是 host 侧开销，其中 CUDA ctx init
> 占 47%。老师/作者要求先纠正"端到端耗时"的口径定义，再做 runner 批量化。

## 1. 端到端口径的修正定义（分子对接场景）

### 1.1 原则：用户视角的"提交任务 → 拿到结果"

端到端耗时的分母应包含**用户为得到一个对接结果必须支付的全部成本**，
排除**与任务数量无关的一次性环境成本**。

### 1.2 逐项裁决（含争议项）

| 项 | 计入端到端？ | 理由 |
|---|---|---|
| 进程 spawn（fork+exec+动态链接） | **是**（若逐配体进程）；**否**（若批量进程） | 随实现方式变化，不是任务本身成本 → 批量模式下自然消失 |
| **CUDA context init** | **否（显式排除）** | 一次性环境成本：驱动初始化 + JIT + 设备枚举，与配体数量无关。批量模式下被摊薄至 0；逐配体进程下是人为惩罚。**这正是 cuda ctx warmup 存在的理由**——把它从测量中剥离，而不是把它算作优化收益 |
| receptor parse + cache/prec 构造 | **是**（首配体）/ **否**（后续配体，同受体复用时） | 同受体多配体是真实工作流（虚拟筛选）；口径上归入"每受体一次性成本"，报告时单列 |
| **pre_gpu_setup（grid/LUT h2d、权重、par.mc）** | **是** | 依赖 receptor+box（每个 CASF 靶点不同），是"准备这次对接"的必要成本，**应计入**。但同 receptor 的批内后续配体可复用 → 归入"每受体一次性" |
| per-ligand prep（rand_maps 生成、m_cl 展开、ric h2d） | **是** | 纯 per-ligand 成本 |
| kernel1 + kernel2 + DtoH | **是** | 核心计算 |
| 输出 PDBQT 写盘 | **是** | 结果交付的一部分 |
| runner 的 python/sudo/fork 开销 | **否（视为测量工具缺陷）** | 不属于对接系统；批量模式 + 二进制内计时后自然消失 |

### 1.3 新口径金字塔（替代 timing_ledger_changelog §1.4）

| 口径 | 定义 | 用途 |
|---|---|---|
| **E2E-per-ligand** | 批进程内：(总 wall − ctx_init) / N_ligands；warmup 首配体不计 | **论文主口径**（真实筛选工作流） |
| E2E-first | 批进程首配体 wall（含 ctx） | 冷启动场景参考 |
| GPU-timeline | k1+k2+h2d+dtoh+memset（CUDA event 账本） | kernel 优化证据 |
| k2 | kernel2 纯 GPU | 与历史数据/论文可比 |

**与 Vina-CUDA 70.5% 口径的关系**：他们没有 ctx 排除问题（未披露），我们报告
E2E-per-ligand 时**显式声明 warmup 策略**（首个配体 warmup，不计入），
这是比 Vina-CUDA 更严谨的口径，写进论文的"计时口径说明"段。

## 2. 实现（v4.5 代码库上，三处改动）

### 2.1 CUDA ctx warmup（main.cpp main_procedure 入口，main_procedure_cl 之前）

```cpp
// [v7] CUDA context warmup: pay driver init ONCE, excluded from per-ligand timing.
// Rationale: ctx init is a one-time environment cost (see handoff §1.2).
{
    int warm_dev = 0; cudaFree(0);                 // force ctx creation
    // small warmup H2D+kernel to page in module code
    float* dummy; cudaMalloc(&dummy, 1024);
    cudaMemset(dummy, 0, 1024); cudaFree(dummy);
    cudaDeviceSynchronize();
}
```
（正式实现把这段放进 `hp_mark("main_procedure_cl_entry")` 之后、`cudaSetDevice` 之前，
并用 event 记录 warmup 耗时打印 `PROBE ctx_warmup : N ms` 供核对。）

### 2.2 二进制内端到端计时（替代 runner 秒表）

在 main.cpp：
- 程序启动最早点 `t_e2e_start`（main 第一行，ctx warmup **之前**打点）
- warmup 后 `t_postwarm`（打印 `PROBE ctx_warmup`）
- main_procedure_cl 返回后、写盘前 `t_results_ready`
- 输出 PDBQT 写完后 `t_e2e_end`
打印：
```
E2E_TOTAL_S=<t_e2e_end - t_e2e_start>          # 含 warmup？见下
E2E_WARM_S=<ctx warmup 耗时>
E2E_DOCK_S=<t_results_ready - t_postwarm>      # 批量模式纯对接段
GPU_ALL_S=<既有账本>
GPU_K2_S=<既有>
```
**E2E_TOTAL 含 warmup、E2E_DOCK 不含**——两个数都打，论文报 E2E_DOCK/N_ligands
并声明 warmup 策略；同时报 E2E_TOTAL 供复现（透明）。

### 2.3 runner 批量模式（run_experiment.py 或新 runner）

- **按 (receptor, box) 分组**：CASF redocking 每个 target 有独立 receptor+box →
  同 target 的 1 个配体本来就一组 = 退化为逐配体！
  → **CASF redocking 批量无收益的根源：每靶点 1 配体**。
  批模式的收益只在 screening（1 受体 × 285 配体）或 decoys 场景。
- 因此 runner 改造分两档：
  - **screening 模式**：真批量，`--ligand_directory` + warmup + E2E 计时，
    预期 per-ligand 1.32 → 0.35s
  - **redocking 模式**：保持逐配体，但口径改为**二进制内 E2E**（剥掉 0.8s runner
    fork 开销 + ctx 显式 warmup 排除）→ rt 1.32 → ~0.42s（仍拿回 ctx+runner 两块）
    且**语义正确**：redocking 本来就是"每靶点一次独立对接"的工作流
- decoys_docking（每靶点多配体）若要跑 → 天然适合批模式

## 3. 实施顺序（建议）

1. main.cpp 加 warmup + E2E 计时（§2.1/2.2，~30 行）→ 编译
2. 单配体 smoke：核对 E2E_DOCK ≈ 旧 PROBE 各段之和 − ctx
3. 新 runner（或改 run_experiment.py 加 `--timing-mode binary`）解析新字段入 summary
4. CASF redocking 全量 → 与 v4/v5/v6 三口径对照表
5. （可选）screening 批量跑一次：验证 per-ligand 0.35s 预测

## 4. 已有资产（不要重造）

- **v6 批内核已存在**：`milestones/v6` kernel2_update 已支持 N 配体单 launch +
  chunk 分批（chunk cap=64），screening 批量可直接用 v6 内核 + v4.5 修复 + 新口径
- **GPU 账本已存在**：gpu_timer/gt_block（账本 vs nsys 偏差 <1%）
- HOST_PROBE 桩已留在 v4.5 源码
- runner 10 列 CSV 已兼容

## 5. 口径相关论文待办（衔接 timing_ledger_changelog §2）

- E2E-per-ligand 加入"计时口径说明"段（与既有四口径并列，成为第五口径=主口径）
- Top-1 用多次运行中位数 ±极差（171-184 极差 13，见 PATHB_CLOSE.md）
- v4.5 / v5_memcpy 的 CASF 数字已认证，可作新基线
