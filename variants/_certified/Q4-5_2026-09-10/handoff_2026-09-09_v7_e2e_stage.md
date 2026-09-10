# v7 E2E 口径 + warmup + host 修复 — 阶段报告（2026-09-09 晚）

> 代码库：`variants/QuickVina2-CUDA-v45`（v4 + 三修复 + 本轮全部 v7 改动）
> 口径定义见 `handoff_2026-09-09_e2e_timing_and_batch.md`（本地 handoffs 已存）

## 已落地

### 1. E2E 计时基建（-DHOST_PROBE 激活，正常构建零开销）
- `inc/host_probe.h`：统一时间线 probe（修复了 main.cpp / main_procedure_cu.cpp
  两 TU 各持 static 状态导致的误归因——这是原 HOST_PROFILE "ctx 232ms" 的错误来源）
- main.cpp：`E2E_TOTAL_S`（含 warmup）/ `E2E_WARM_S` / `E2E_DOCK_S`（排除 warmup，
  **论文主口径**）
- ctx warmup：`v7_warmup_cuda()`（kernel2.cu，CUDA TU）：cudaSetDevice + 4KB memset
  + 空 kernel launch（触发 module load）+ 256MB alloc warm + sync

### 2. Host 热点修复（经 bisect 实测归因）
- **rand_maps 生成 214ms → 1.2ms** ✅：根因是 `random_int/random_fl/random_inside_sphere`
  **每次调用构造 boost::variate_generator**（~12 万次/配体）。hoist 到循环外 +
  sphere 拒绝采样 inline。保持抽随机数序列（uniform 分布为无状态映射）。
  顺带清除 `#pragma omp parallel for` 迭代共享非线程安全 rng 的伪并行/UB。
- **Path B 合并**：m_gpus 单份拷贝（原 8192 份相同 m_cl）+ kernel2_update
  `mg[id] → mg[0]`（配套内核改动）。真 HtoD 流量 -30MB/配体。
- ~~model 拷贝~~：改 const 引用后热点依旧 207ms → **热点实为 grids/ar/pa/pre
  填充段**（bisect 确认），per-receptor 性质，非拷贝问题。
- ~~ric randomize inline~~：**实验回滚**。保持抽随机数流的 inline 复现（Box-Muller
  无缓存尾、4 normals）结果 aff=94.6 全崩 → boost normal_distribution 的引擎消耗
  语义与假设不符。恢复原实现（serial，210ms 保留），后续用 GPU 预生成 rand 表。

## Smoke 结果（三靶点，各配体 centroid box）

| target | AFF mode1 | K2 | E2E_DOCK | 对照 v4 |
|---|---|---|---|---|
| 1pxn | **-9.1** | 0.143 | 1.228 | -9.1 ✓ |
| 1c5z | **-5.9** | ~0.014 | 1.225 | -5.9 ✓ |
| 1u1b | **-6.9** | 0.601 | 1.224 | -7.3~-7.5（±噪声）|

ERR=0；E2E_DOCK 稳定 1.22s（host-bound，与 k2 无关——1c5z k2=0.014 也是 1.22）。

## E2E_DOCK 1.22s 的当前构成（1pxn bisect 实测）

| 项 | 耗时 | 性质 |
|---|---|---|
| grids/ar/pa/pre 填充+h2d | ~207ms | per-receptor（screening 摊薄）|
| ric randomize（8192 confs）| ~213ms | per-ligand；**GPU 化目标** |
| k2 | 142ms（1pxn）| GPU |
| HtoD 账本 | 210ms | 归因仍不清晰（非 m_cl 拷贝，30MB 已除）|
| rand_maps | 1.2ms | ✅ 已修 |
| parse + 输出 + 其他 | ~65ms | — |

注意：`memcpy_single_m` 的 CASF htod=0.000 与此处的 0.21 并不矛盾——账本打点
覆盖范围不同（GT 块包裹 vs 未包裹），实际省的是 30MB 真流量。

## 与 certified 基线的关系
- v4.5（CASF 285/285 ✓，Sep09_1525）不含本轮 host 修复
- v5_memcpy（CASF 285/285 ✓，Sep08_1208）含 Path B 不含三修复
- 本 v7 分支 = v4.5 + Path B + rand_maps hoist + E2E 基建，**需重跑 CASF 认证**
  （预期 Top-1 落 v4 家族 171-184 带内）

## 下一步
1. runner 解析 E2E 字段 → summary 新列（e2e_dock_s 等）
2. CASF 全量认证 v7（预期 wall 1.32 → ~1.0s：省 rand_maps 210ms + runner 口径差）
3. ric randomize GPU 化（curand kernel 生成 rand_maps+ric，省 ~213ms → E2E ~0.8s）
4. screening 批模式验证（v6 内核 + v4.5 修复 + E2E 口径）
