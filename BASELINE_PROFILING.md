# Baseline Profiling — Vina-CUDA (PRE-TC改造)

> 日期: 2026-07-12 | GPU: RTX 4090 (Ada, sm_89) | CUDA 11.8 | 预编译二进制
> 工具: nsys profile (cuda) + sqlite 导出

## 环境澄清（与 CLAUDE.md 不符）
- **实际 GPU**: RTX 4090 24GB (compute cap 8.9), 非 CLAUDE.md 写的 RTX 5090
- **CUDA**: /usr/local/cuda-11.8 (nvcc V11.8.89), 非 12.8
- **sudo**: 不需要 (非 root 限制不存在于本机)
- **PATH**: 需 `export PATH=/usr/local/cuda-11.8/bin:$PATH`

## nsys kernel 时间 (1u1b, source build)
| kernel | grid | block | sharedMem | 占比 |
|---|---|---|---|---|
| 325 (≈kernel2 BFGS) | 128³ | 4³ | 32768 | **99.8%** (1339ms) |
| 314 (≈kernel1 scoring) | — | — | — | 0.2% (2.9ms) |
- → **BFGS (kernel2/kernel2_update) 占 ~99.8% GPU 时间**，远超草稿假设 >80%
- kernel 名被 strip（demangledName=数字），需 NVTX 标记才能精确分离

## 编译选项 (Makefile)
- `NVCC_COMPILER=/usr/local/cuda/bin/nvcc` → 实际 11.8
- `-DGRID_DIM1=64 -DGRID_DIM2=128 -DSMALL_BOX -O3 -DNDEBUG -DBUILD_KERNEL_FROM_SOURCE`
- **无 -arch=sm_xx**（依赖 nvcc 默认，但二进制能跑 4090 → 默认 arch 兼容）

## 精度类型 (kernel2.h:18)
- `typedef float flo;` (FP32)，预留 `// typedef half flo;` (未启用) → FP16 路径待探索

## BFGS 热点函数 (inc/cuda/quasi_newton.cuh)
- `scalar_product` (L1498): 串行点积 → TC 化候选
- `scalar_product_update` (L1510): __shared__ 并行归约 → TC reduction
- `warpRecude` (L352): warp 级归约
- `ig_eval_deriv_update_optimized` / `m_eval_deriv_optimized`: 优化 BFGS 两循环

## 待办
1. [进行中] CASF-2016 全量 redocking baseline (run_experiment.py) → 端到端耗时+RMSD
2. [ ] NVTX 标记 BFGS kernel 重编，精确分离 BFGS vs scoring 占比
3. [ ] 确认 kernel2 vs kernel2_update 哪个占主导

## ncu 深度分析 (BFGS kernel = kernel2_update)
> 工具: Nsight Compute 2022.4.2 (sudo, 因 ERR_NVGPUCTRPERM 需 root)
> 修复: sudo ncu 绝对路径 /usr/local/cuda-11.8/bin/ncu

### BFGS (kernel2_update) SpeedOfLight (10 launch 聚合, 1u1b)
| 指标 | 值 | 解读 |
|---|---|---|
| Compute (SM) [%] | **10.92%** | 计算管道极低利用率 |
| Memory [%] | **17.15%** | 内存带宽未饱和 |
| DRAM Throughput | 8.47% | 非 DRAM-bound |
| L2 Cache Throughput | 17.15% | |
| L1/TEX Cache | 11.41% | |
| Executed IPC Active | 0.43 inst/cycle | 指令吞吐低 |
| SM Busy | 10.85% | SM 大部分空闲 |
| L2 Hit Rate | **99.41%** | 数据命中 L2（不缺带宽） |
| Memory Throughput | 83 GB/s (峰值~1008) | 仅 8% |

### 关键结论
1. **BFGS kernel 既非 compute-bound 也非 memory-bound，而是"双低利用率"**：SM 11% + Mem 17%，远低于峰值 → 典型**低并行度/串行依赖/launch 太小**型 kernel。
2. **对照 G01**：G01 说"memory-bound 核 TC 无效"——但 BFGS 实际是**低利用率**（更糟）。纯 TC 替换单条指令不会提升，因为瓶颈不是计算吞吐而是**并行度/利用率**。
3. **改造方向修正**（重要）：
   - 仅把 `scalar_product` 改成 TC mma **不够**（IPC 0.43 说明根本没喂满 SM）
   - 真正收益来自 **kernel fusion + 提并行度（G03/G04/G09 路线）+ 提 operational intensity** → 让 SM 利用率从 11% 升上去，TC 才能吃到吞吐
   - 或 **增大 BFGS 批处理粒度**（同时优化多个构象的 BFGS，类比 G20 batched）
4. **TC 化仍可做**：当 fusion 后 SM 利用率提升、BFGS 内积成为 compute-bound 热点时，TC mma 替代 FP32 点积才有意义（呼应 G19 Amdahl f≥0.9）。

## CASF-2016 Baseline (改造前黄金基准)
> run_experiment.py --variant vina-cuda --search-mode fast, 285 靶点
- 结果目录: experiments/Jul12_2316_CASF-2016_vina_cuda_baseline_PRE_TC/
- 状态: 284/285 OK
- **平均耗时: 1.66s/靶点, 中位 1.57s**
- RMSD: rmsd_results.csv 已生成（对比改造后用）
- affinity: summary.csv（对比改造后打分）

## 下一步优先级（修正后）
1. **[P0] 提 BFGS 并行度/利用率**：先 fusion + 增大 BFGS 批粒度（batched），把 SM 11%→更高，再做 TC 化才有意义
2. [P1] 加 NVTX 标记，nsys 确认融合后 BFGS 占比变化
3. [P1] TC 化 scalar_product（TF32+EC, G19），对比 roofline
4. [P2] 自适应精度调控（师姐 SwitchDetector）接入
5. [P0] 改造后重跑 CASF baseline 对比精度(≤2Å%)+耗时
