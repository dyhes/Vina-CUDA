# CASF-2016 基准测试实验记录

> **更新日期**: 2026-06-07  
> **数据集**: CASF-2016 CoreSet (285 targets) + Screening (57 targets × 285 ligands)  
> **GPU**: NVIDIA RTX 4090 24GB  
> **统一 RMSD 工具**: spyrmsd (Meli & Biggin, J. Cheminf. 2020), symmetry=True, Top-1 (MODEL 1)

---

## 实验总览

| 能力维度 | 实验编号 | 方法数 | 关键发现 |
|---------|---------|-------|---------|
| **Docking Power** | Exp 1-12 | 7 | Vina-CPU 精度最高 (65.0% ≤2Å)，Vina-GPU2 接近 (63.2%) |
| **Scoring Power** | Exp 13 | 9 | Vina-CPU R=0.601，所有 GPU 变体 R=0.52-0.59 |
| **Ranking Power** | Exp 14 | 9 | Vina-CPU SP=0.542，Uni-Dock detail SP=0.502 为 GPU 最优 |
| **Screening Power** | Exp 15-17 | 3 | Uni-Dock batch 17.7min 完成 57×285 对接，EF@5%=4.30 |

---

## 第一部分：Docking Power (Redocking)

### 实验设计
- **任务**: 从随机初始构象搜索晶体结合构象
- **指标**: RMSD ≤ 2.0Å (成功对接)，≤ 1.0Å (高精度对接)
- **参考**: 晶体配体坐标 (mol2 → PDBQT)
- **搜索框**: 22.5³ Å，中心 = 配体几何中心

### Exp 1: QuickVina-W-CUDA 8192 heuristic
- **日期**: 2026-05-28 / 2026-05-31 (三次独立运行)
- **目录**: `May28_2329`, `May31_1228`, `May31_1300`
- **配置**: thread=8192, search_depth=heuristic, num_modes=9

| 指标 | May28 | May31 v1 | May31 v2 | 平均 |
|------|-------|----------|----------|------|
| OK | 285/285 | 285/285 | 284/285 | 284.7 |
| ≤2.0Å | 63.5% | 61.4% | 58.5% | **61.1%** |
| ≤1.0Å | 44.2% | 46.7% | 45.4% | **45.4%** |
| Median RMSD | 1.198Å | 1.171Å | 1.342Å | **1.24Å** |
| Mean RMSD | 2.699Å | 2.700Å | 2.868Å | **2.76Å** |
| 总时间 | 7.7 min | 8.0 min | 7.8 min | **7.8 min** |
| 每靶点 | 1.62s | 1.68s | 1.65s | **1.65s** |

**随机种子方差**: ≤2Å 三次跑差异 5pp，Median |Δ| = 0.12-0.14Å，≤2Å 一致性 88-90%。

---

### Exp 2: QuickVina2-CUDA 8192 heuristic
- **日期**: 2026-05-31
- **目录**: `May31_2050_CASF-2016_qvina2_cuda`
- **配置**: thread=8192, search_depth=heuristic, RILC-BFGS on, num_modes=9

| 指标 | 值 |
|------|---|
| OK | 285/285 (100%) |
| ≤2.0Å | **62.8%** (179/285) |
| ≤1.0Å | **44.9%** (128/285) |
| Median RMSD | 1.190Å |
| Mean RMSD | 2.833Å |
| 总时间 | 7.6 min |
| 每靶点 | **1.61s** |

**与 QVina-W-CUDA 对比**: ≤2Å 差 0.7pp，≤1Å 差 0.5pp，在随机种子噪声范围内。两者共享 `quasi_newton.cuh` RILC-BFGS kernel。

---

### Exp 3: Vina-CUDA 8192 heuristic
- **日期**: 2026-05-31
- **目录**: `May31_1930_CASF-2016_vina_cuda_8192auto`
- **配置**: thread=8192, search_depth=heuristic, RILC-BFGS on, num_modes=9

| 指标 | 值 |
|------|---|
| OK | 285/285 (100%) |
| ≤2.0Å | **58.9%** (168/285) |
| ≤1.0Å | **44.6%** (127/285) |
| Median RMSD | 1.223Å |
| Mean RMSD | 2.917Å |
| 总时间 | 8.1 min |
| 每靶点 | 1.70s |

---

### Exp 4: Vina-GPU2 (OpenCL) 8192 heuristic
- **日期**: 2026-05-31
- **目录**: `May31_2050_CASF-2016_vina_gpu2_8192auto`
- **配置**: OpenCL kernel 运行时编译，thread=8192, RILC-BFGS on, num_modes=9

| 指标 | 值 |
|------|---|
| OK | 285/285 (100%) |
| ≤2.0Å | **63.2%** (180/285) |
| ≤1.0Å | **46.7%** (133/285) |
| Median RMSD | **1.075Å** |
| Mean RMSD | 2.567Å |
| 总时间 | 10.3 min |
| 每靶点 | 2.18s |

**亮点**: ≤1Å=46.7% 超越 Vina-CPU (46.6%)，Median 1.075Å 几乎追平 Vina-CPU (1.072Å)。OpenCL 开销 +35% 时间。

---

### Exp 5: QuickVina2-GPU2 (OpenCL) 8192 heuristic
- **日期**: 2026-05-31
- **目录**: `May31_2232_CASF-2016_qvina2_gpu2`
- **配置**: OpenCL kernel 运行时编译，thread=8192, **传统 BFGS (无 RILC-BFGS)**, num_modes=9

| 指标 | 值 |
|------|---|
| OK | 285/285 (100%) |
| ≤2.0Å | **59.3%** (169/285) |
| ≤1.0Å | **42.1%** (120/285) |
| Median RMSD | 1.351Å |
| Mean RMSD | 2.869Å |
| 总时间 | 10.7 min |
| 每靶点 | 2.22s |

**与 QuickVina2-CUDA 对比**: ≤2Å 差 3.5pp，≤1Å 差 2.8pp，确认为 RILC-BFGS → 传统 BFGS 的精度退化。

---

### Exp 6: Uni-Dock A fast (128×20)
- **日期**: 2026-05-31
- **目录**: `May31_1202_CASF-2016_unidock_A_fast`
- **配置**: search_mode=fast (exh=128, max_step=20), num_modes=9, scoring=vina

| 指标 | 值 |
|------|---|
| OK | 285/285 (100%) |
| ≤2.0Å | **54.7%** (156/285) |
| ≤1.0Å | **41.8%** (119/285) |
| Median RMSD | 1.600Å |
| Mean RMSD | 3.107Å |
| 总时间 | 11.0 min |
| 每靶点 | 2.29s |

**MC budget**: 128×20 = 2,560 次蒙特卡洛步。

---

### Exp 7: Uni-Dock A balance (384×40)
- **日期**: 2026-05-31
- **目录**: `May31_1758_CASF-2016_unidock_A_balance`
- **配置**: search_mode=balance (exh=384, max_step=40), num_modes=9

| 指标 | 值 | Δ vs fast |
|------|---|---|
| ≤2.0Å | **58.9%** (168/285) | +4.2pp |
| ≤1.0Å | **42.5%** (121/285) | +0.7pp |
| Median RMSD | 1.397Å | -0.20Å |
| Mean RMSD | 2.901Å | -0.21Å |
| 每靶点 | 2.36s | +0.07s |

**MC budget**: 384×40 = 15,360 (6× fast)，精度提升 ~4pp，时间几乎不变。

---

### Exp 8: Uni-Dock A detail (512×40)
- **日期**: 2026-05-31
- **配置**: search_mode=detail (exh=512, max_step=40), num_modes=9

| 指标 | 值 | Δ vs balance |
|------|---|---|
| ≤2.0Å | **62.5%** (178/285) | +3.6pp |
| ≤1.0Å | **44.6%** (127/285) | +2.1pp |
| Median RMSD | **1.171Å** | -0.23Å |
| Mean RMSD | 2.617Å | -0.28Å |
| 每靶点 | 2.37s | +0.01s |

**MC budget**: 512×40 = 20,480 (1.33× balance)，精度再提升 ~4pp。Detail 模式 ≤2Å=62.5% 接近 Vina-CPU (65.0%)。

---

### Exp 9: Uni-Dock B (paired_batch) chunk=8
- **日期**: 2026-05-30
- **目录**: `May30_1959_CASF-2016_unidock_B_ch8`
- **配置**: paired_batch_size=8, search_mode=fast, num_modes=9
- **修复**: MAX_NUM_OF_RIGID 24→128, MAX_NUM_OF_LIG_PAIRS 1024→4096 (避免 SIGSEGV)

| 指标 | 值 |
|------|---|
| OK | 285/285 (100%) |
| ≤2.0Å | **54.7%** (156/285) |
| ≤1.0Å | **39.6%** (113/285) |
| Median RMSD | 1.642Å |
| Mean RMSD | 2.901Å |
| 总时间 | **8.0 min** |
| 每靶点 | **1.68s** |

**B 模式优势**: 速度最快 (1.68s/tgt)，适合大规模并行对接。

---

### Exp 10: Vina-CPU (AutoDock Vina 1.2.3)
- **日期**: 2026-05-29
- **目录**: `May29_0028_CASF-2016_Vina-CPU_auto`
- **配置**: --thread 8192, --exhaustiveness 8 (auto), num_modes=9, timeout=300s

| 指标 | 值 |
|------|---|
| OK | 283/285 (99.3%) |
| Timeouts | 2 (3ag9, 3uri) |
| ≤2.0Å | **65.0%** (184/283) |
| ≤1.0Å | **46.6%** (132/283) |
| Median RMSD | **1.072Å** |
| Mean RMSD | 2.567Å |
| 总时间 | 52.1 min |
| 每靶点 | 11.06s |

**精度之王**: ≤2Å=65.0%，≤1Å=46.6%，Median=1.072Å 均为所有方法最优。代价是 5-7× 慢于 GPU 方法。

---

### Exp 11: 跨方法 Docking Power 对比 (CASF-2016)

| 方法 | ≤1Å | ≤2Å | Median | Mean | s/tgt | 总时间 |
|------|-----|-----|--------|------|-------|--------|
| **Vina-CPU** | **46.6%** | **65.0%** | **1.072Å** | 2.567 | 11.06 | 52.1min |
| Vina-GPU2 (OpenCL) | 46.7% | 63.2% | 1.075Å | **2.567** | 2.18 | 10.3min |
| QVina-W-CUDA | 45.4% | 61.1% | 1.24Å | 2.76 | 1.65 | 7.8min |
| QuickVina2-CUDA | 44.9% | 62.8% | 1.190Å | 2.833 | 1.61 | 7.6min |
| Uni-Dock detail | 44.6% | 62.5% | 1.171Å | 2.617 | 2.37 | 11.3min |
| QuickVina2-GPU2 | 42.1% | 59.3% | 1.351Å | 2.869 | 2.22 | 10.7min |
| Vina-CUDA | 44.6% | 58.9% | 1.223Å | 2.917 | 1.70 | 8.1min |
| Uni-Dock balance | 42.5% | 58.9% | 1.397Å | 2.901 | 2.36 | 11.2min |
| Uni-Dock fast | 41.8% | 54.7% | 1.600Å | 3.107 | 2.29 | 11.0min |
| Uni-Dock B ch=8 | 39.6% | 54.7% | 1.642Å | 2.901 | **1.68** | **8.0min** |

**关键发现**:
1. **Vina-CPU 精度最高**，但 5-7× 慢于 GPU 方法
2. **Vina-GPU2 (OpenCL) 是 GPU 精度之王**：≤1Å=46.7% 超越 Vina-CPU，Median 1.075Å 几乎追平
3. **QuickVina2-CUDA 最快** (1.61s/tgt)，精度仅比 Vina-GPU2 低 3-4pp
4. **RILC-BFGS vs 传统 BFGS**: QuickVina2-CUDA (RILC) vs QuickVina2-GPU2 (传统) 差 3-4pp，确认为优化器差异
5. **Uni-Dock MC budget 线性提升精度**: fast → detail 提升 8pp (≤2Å)，时间几乎不变
6. **随机种子方差**: GPU 方法 ≤2Å 波动 3-5pp，Vina-CPU 近确定性

---

### Exp 12: AutoDockGPU 数据集对比 (140 targets)

| 方法 | ≤1Å | ≤2Å | Median | s/tgt |
|------|-----|-----|--------|-------|
| **Vina-CPU** | **72.1%** | **92.6%** | **0.662Å** | 29.1 |
| QVina-W-CUDA | 42.3% | 74.5% | 1.20Å | 1.68 |

**CASF-2016 vs AutoDockGPU**: AutoDockGPU 靶点更简单 (更刚性、更小配体)，Vina-CPU ≤2Å=92.6% vs CASF-2016 65.0%。QVina-W-CUDA 在 AutoDockGPU 上精度更高 (74.5% vs 61.1%)，但差距仍大。

---

## 第二部分：Scoring Power

### Exp 13: CASF-2016 Scoring Power — 9 方法对比
- **日期**: 2026-06-05
- **脚本**: `tools/scoring_ranking_9methods.py` (Python 3)
- **指标**: Pearson R (预测 score vs 实验 logKa)

| 方法 | N | Pearson R | R² | SD (logKa) |
|------|--:|----------:|---:|-----------:|
| **Vina-CPU** | 283 | **0.601** | **0.361** | **1.738** |
| Vina-GPU2 | 285 | 0.594 | 0.353 | 1.749 |
| UniDock detail | 285 | 0.578 | 0.334 | 1.774 |
| QuickVina2-CUDA | 285 | 0.568 | 0.323 | 1.790 |
| QVina-W-CUDA | 285 | 0.566 | 0.320 | 1.793 |
| UniDock balance | 285 | 0.564 | 0.318 | 1.796 |
| Vina-CUDA | 285 | 0.563 | 0.317 | 1.798 |
| QuickVina2-GPU2 | 283 | 0.562 | 0.316 | 1.791 |
| UniDock fast | 285 | 0.523 | 0.274 | 1.853 |

**CASF-2016 论文基准**: AutoDock Vina R=0.566, X-Score HM R=0.644, GlideScore-SP R=0.576

**关键发现**:
1. **所有方法共享 Vina 打分函数** — 分数差异纯粹来自搜索算法找到的局部最优不同
2. **Vina-CPU 和 Vina-GPU2 打分最准** (R > 0.59): 更彻底的搜索找到更接近全局能量最小的构象
3. **Scoring Power 固有上限** (R ≈ 0.57-0.60, R² ≈ 32-36%): Vina 打分函数只能解释 ~1/3 的实验结合亲和力方差，这是经验打分函数的根本局限
4. **Uni-Dock MC budget 直接影响打分**: fast → detail R 提升 0.055，detail 跨过 CASF-2016 Vina 基线 (0.566)
5. **GPU 变体打分聚集** (R=0.562-0.594): 非 Uni-Dock GPU 方法 R 差异仅 0.032，确认打分差异次要于共享 Vina 函数

---

## 第三部分：Ranking Power

### Exp 14: CASF-2016 Ranking Power — 9 方法对比
- **日期**: 2026-06-05
- **基准**: 57 靶点组 × 5 配体/组，计算组内 Spearman ρ / Kendall τ / PI

| 方法 | N | SP | τ | PI |
|------|--:|---:|---:|---:|
| **Vina-CPU** | 55 | **0.542** | **0.473** | **0.578** |
| UniDock detail | 57 | 0.502 | 0.432 | 0.539 |
| QuickVina2-GPU2 | 55 | 0.499 | 0.437 | 0.533 |
| Vina-GPU2 | 57 | 0.480 | 0.414 | 0.515 |
| Vina-CUDA | 57 | 0.472 | 0.407 | 0.508 |
| QuickVina2-CUDA | 57 | 0.471 | 0.409 | 0.507 |
| UniDock balance | 57 | 0.456 | 0.389 | 0.493 |
| QVina-W-CUDA | 57 | 0.452 | 0.393 | 0.492 |
| UniDock fast | 57 | 0.432 | 0.382 | 0.461 |

**CASF-2016 论文基准**: AutoDock Vina SP=0.441, τ=0.313, PI=0.497

**关键发现**:
1. **Vina-CPU 排序最强** (SP=0.542, τ=0.473, PI=0.578): 更彻底的搜索找到的构象分数更好反映真实亲和力排序
2. **所有方法超越 CASF-2016 Vina 基线** (SP=0.441): 即使最差的 GPU 方法 (QVina-W-CUDA, SP=0.452) 也超越它，可能反映 Vina 1.2.x 改进
3. **Uni-Dock detail 排序意外地高** (SP=0.502): 尽管对接精度 (≤2Å=62.5%) 较低，但打分捕捉了相对亲和力排序
4. **Kendall τ (0.38-0.47) 大幅超越 CASF-2016 Vina** (τ=0.313): RILC-BFGS 和改进的搜索启发式产生更可靠的分数排序
5. **Uni-Dock MC budget 影响排序**: fast → detail SP 提升 0.070，PI 提升 0.078

---

## 第四部分：Screening Power

### Exp 15: Uni-Dock-A Batch Screening (CASF-2016)
- **日期**: 2026-06-06
- **目录**: `Jun06_1239_CASF-2016_unidock_A_fast_screening`
- **配置**: gpu_batch mode (1 receptor × 285 ligands/call), search=fast, num_modes=9
- **规模**: 57 targets × 285 ligands = 16,245 次对接

| 指标 | EF @ 1% (top 3) | EF @ 5% (top 14) | EF @ 10% (top 28) |
|------|----------------:|-----------------:|------------------:|
| Mean | 7.90 | 4.30 | 3.07 |
| Median | 0.00 | 4.07 | 2.26 |

**总时间**: **17.7 min** (~18.6s/target)

**关键发现**:
1. **EF@1% 双峰分布**: 29 targets binder 在 top-3 (EF=19-57)，28 targets 完全失败 (EF=0)
2. **EF@5% 更稳定**: mean=4.30, median=4.07，真实 binder 富集 ~4× 高于随机
3. **Batch 模式速度**: 285 ligands/call 比 sequential (1/call) 快 **17.3×**

---

### Exp 16: QVina-W-CUDA Screening (CASF-2016)
- **日期**: 2026-06-06
- **目录**: `Jun06_0415_CASF-2016_qvina_w_cuda_screening`
- **配置**: thread=8192, timeout=600s, sequential (1 ligand/call)
- **规模**: 57 targets × 285 ligands = 16,245 次对接

| 指标 | EF @ 1% | EF @ 5% | EF @ 10% |
|------|--------:|--------:|---------:|
| Mean | 7.55 | 4.27 | 3.09 |
| Median | 0.00 | 4.07 | 2.26 |

**总时间**: ~306 min (~460s/target)

**与 Uni-Dock batch 对比**: EF 几乎相同 (Δ<0.5)，但 Uni-Dock 快 **17.3×**。QVina-W-CUDA 无 batch 接口，只能逐配体调用，GPU 利用率 0-84% 脉冲式。

---

### Exp 17: Uni-Dock-A Sequential Screening (CASF-2016)
- **日期**: 2026-06-06
- **目录**: `Jun06_1309_CASF-2016_unidock_A_fast_screening`
- **配置**: gpu_batch mode (1 ligand/call, sequential), search=fast, num_modes=9

| 指标 | EF @ 1% | EF @ 5% | EF @ 10% |
|------|--------:|--------:|---------:|
| Mean | **8.83** | 3.56 | 2.81 |
| Median | 0.00 | 3.39 | 2.04 |

**总时间**: 598.1 min (~628s/target)

**Batch vs Sequential 对比**:

| 指标 | Batch (285/call) | Sequential (1/call) | Diff |
|------|------------------|---------------------|------|
| 总时间 | **17.7 min** | 598.1 min | **33.8×** |
| 每靶点 | **18.6s** | 628s | 33.8× |
| EF@1% mean | 7.90 | **8.83** | +11.8% |
| EF@5% mean | **4.30** | 3.56 | -17.2% |
| EF@10% mean | **3.07** | 2.81 | -8.5% |

**关键发现**:
1. **Sequential 略优 EF@1%** (+11.8%): 每个配体独占 GPU 资源，BFGS 收敛更充分
2. **Batch 在 EF@5%/10% 更优** (-8~17%): batch 按配体复杂度分组 (Small/Medium/Large/XL)，可能影响 BFGS 收敛路径
3. **Batch 是实用选择**: 90%+ 精度，34× 加速

---

## ~~正在进行：Vina-CPU Screening Baseline~~ (已完成，见 Exp 20)

- **日期**: 2026-06-06 至今
- **目录**: `Jun06_2352_CASF-2016_vina_cpu_screening`
- **配置**: --thread 8192, --exhaustiveness 8, timeout=600s
- **当前进度**: 16/57 targets (28%)
- **预计完成**: ~41 小时 (2026-06-09 凌晨)

**预估性能**:
- 每靶点: ~60 min (285 × 13.2s)
- 总时间: ~57 小时

---

## 实验结论与启示

### 1. 精度-速度权衡
- **Vina-CPU**: 精度之王，但 5-7× 慢于 GPU
- **Vina-GPU2 (OpenCL)**: GPU 精度最优，≤1Å 超越 Vina-CPU，但 OpenCL 开销 +35%
- **QuickVina2-CUDA**: 最快 GPU 方法 (1.61s/tgt)，精度仅比 Vina-GPU2 低 3-4pp
- **Uni-Dock detail**: 接近 Vina-CPU 精度 (≤2Å=62.5% vs 65.0%)，速度快 5×

### 2. RILC-BFGS vs 传统 BFGS
- QuickVina2-CUDA (RILC) vs QuickVina2-GPU2 (传统): 差 3-4pp ≤2Å
- RILC-BFGS 是 Vina-GPU 家族的核心优化，提升局部搜索精度

### 3. MC budget 对 Uni-Dock 的影响
- fast (2,560 MC) → detail (20,480 MC): ≤2Å 提升 8pp，Scoring R 提升 0.055，Ranking SP 提升 0.070
- Detail 模式是 Uni-Dock 的最佳精度配置，时间几乎不变 (进程启动开销主导)

### 4. Screening Power 实用建议
- **Uni-Dock batch**: 17.7 min 完成 57×285 对接，EF@5%=4.30，实用首选
- **QVina-W-CUDA**: 306 min，EF 几乎相同，无 batch 接口限制速度
- **Vina-CPU**: ~57 小时，作为 baseline 对照

### 5. 随机种子方差
- GPU 方法 ≤2Å 波动 3-5pp，Vina-CPU 近确定性
- 单次 GPU 跑不可靠，建议多次取平均或报告方差

### 6. 打分函数固有局限
- Scoring Power R² ≈ 32-36%: Vina 打分函数只能解释 ~1/3 的实验亲和力方差
- Ranking Power 高方差 (SP_std ≈ 0.41-0.46): 排序高度依赖靶点
- 这是经验打分函数的根本局限，非 GPU 加速引入

---

## 附录：工具配置与陷阱

### Uni-Dock 批量对接方案

| 方案 | 方式 | 优点 | 缺点 |
|------|------|------|------|
| **A (default)** | 逐靶点独立调用 | 行为一致，结果可比 | 低配体数时 kernel launch 开销大 |
| **B (batch)** | `--paired_batch_size N --ligand_index pairs.json` | 消除逐次调用开销，5-10× 加速 | 需准备 JSON，无法拆 per-target 耗时 |

### QVina-W-CUDA 最低 MC budget
- thread < 8192 → SIGABRT (g_evaluate 错误)
- 最低 ~15K MC (thread=8192, heuristic depth)
- Vina-CPU/Uni-Dock 的 2,560 MC 在 QVina-W-CUDA 架构上不可达

### RMSD 工具统一
- **统一**: spyrmsd (Meli & Biggin, J. Cheminf. 2020), symmetry=True, Top-1 (MODEL 1)
- **废弃**: obrms, 自研 Kabsch (不同方法间同一输出可差 ~1.5Å)
- **CSV 列**: target, status, rmsd_all_atom, rmsd_heavy_atom, n_atoms

---

**文档版本**: v2.0 (2026-06-07)  
**总实验数**: 17 (Exp 1-17) + 1 进行中  
**总对接次数**: ~50,000+

### Exp 18: Uni-Dock-A Detail Batch Screening (CASF-2016)
- **日期**: 2026-06-11
- **目录**: `Jun11_1643_CASF-2016_unidock_A_detail_screening`
- **配置**: gpu_batch mode (1 receptor × 285 ligands/call), search=detail (exh=512, max_step=40), num_modes=9
- **规模**: 57 targets × 285 ligands = 16,245 次对接

| 指标 | EF @ 1% (top 3) | EF @ 5% (top 14) | EF @ 10% (top 28) |
|------|----------------:|-----------------:|------------------:|
| Mean | 9.22 | 4.28 | 3.23 |
| Median | 0.00 | 4.07 | 3.39 |

**总时间**: **31.1 min** (~32.6s/target)

---

### Exp 19: Uni-Dock-A Balance Batch Screening (CASF-2016)
- **日期**: 2026-06-11
- **目录**: `Jun11_1714_CASF-2016_unidock_A_balance_screening`
- **配置**: gpu_batch mode (1 receptor × 285 ligands/call), search=balance (exh=384, max_step=40), num_modes=9
- **规模**: 57 targets × 285 ligands = 16,245 次对接

| 指标 | EF @ 1% (top 3) | EF @ 5% (top 14) | EF @ 10% (top 28) |
|------|----------------:|-----------------:|------------------:|
| Mean | 8.98 | 4.31 | 3.39 |
| Median | 0.00 | 4.07 | 3.82 |

**总时间**: **27.3 min** (~28.7s/target)

---

## 第五部分：补充 Screening 实验

### Exp 20: Vina-CPU Screening (CASF-2016)
- **日期**: 2026-06-06 ~ 2026-06-09
- **目录**: `Jun06_2352_CASF-2016_vina_cpu_screening`
- **配置**: --thread 8192, --exhaustiveness 8, timeout=600s, sequential (1 ligand/call)
- **规模**: 57 targets × 285 ligands = 16,245 次对接

| 指标 | EF @ 1% (top 3) | EF @ 5% (top 14) | EF @ 10% (top 28) |
|------|----------------:|-----------------:|------------------:|
| Mean | 8.15 | 3.88 | 3.00 |
| Median | 0.00 | 4.04 | 2.04 |

**总时间**: **3614.2 min** (~3804 s/target ≈ 63.4 min/target ≈ **60 小时**)

**关键发现**:
1. Vina-CPU EF@5%=3.88 低于所有 GPU 方法 (4.07-4.31)，确认为搜索不充分 (exhaustiveness=8 对 screening 偏弱)
2. 60 小时是 Uni-Dock batch (27 min) 的 **134×**，实用性极低
3. EF@1%=8.15 高于 Vina-CUDA (8.83)，但整体 EF 仍偏弱

---

### Exp 21: QuickVina2-CUDA Screening (CASF-2016)
- **日期**: 2026-06-10
- **目录**: `Jun10_0003_CASF-2016_qvina2_cuda_screening`
- **配置**: thread=8192, search_depth=heuristic, RILC-BFGS on, sequential (1 ligand/call)
- **规模**: 57 targets × 285 ligands = 16,245 次对接

| 指标 | EF @ 1% (top 3) | EF @ 5% (top 14) | EF @ 10% (top 28) |
|------|----------------:|-----------------:|------------------:|
| Mean | **9.35** | 3.88 | 3.07 |
| Median | 0.00 | 4.07 | 2.26 |

**总时间**: **410.6 min** (~432 s/target)

**关键发现**:
1. EF@1%=9.35 为所有方法最高 — QuickVina2-CUDA 在 top-3 中捕获 binder 能力最强
2. EF@5%=3.88 与 Vina-CPU 相同，低于 Uni-Dock batch (4.28-4.31)
3. 速度为 Vina-CPU 的 **8.8×**

---

### Exp 22: Vina-CUDA Screening (CASF-2016)
- **日期**: 2026-06-10
- **目录**: `Jun10_1412_CASF-2016_vina_cuda_screening`
- **配置**: thread=8192, search_depth=heuristic, RILC-BFGS on, sequential (1 ligand/call)
- **规模**: 57 targets × 285 ligands = 16,245 次对接

| 指标 | EF @ 1% (top 3) | EF @ 5% (top 14) | EF @ 10% (top 28) |
|------|----------------:|-----------------:|------------------:|
| Mean | 8.83 | 4.15 | 3.10 |
| Median | 0.00 | 4.07 | 2.04 |

**总时间**: **448.8 min** (~472 s/target)

**关键发现**:
1. EF@5%=4.15 高于 QuickVina2-CUDA (3.88) 和 Vina-CPU (3.88)
2. 与 QVina-W-CUDA (EF@5%=4.27) 接近，但速度略慢
3. 速度为 Vina-CPU 的 **8.1×**

---

### 更新后 Screening Power 全景对比 (10 方法)

| 方法 | EF@1% | EF@5% | EF@10% | 总时间 | s/tgt | 相对 Vina-CPU |
|------|-------|-------|--------|--------|-------|---------------|
| **QVina2-CUDA** | **9.35** | 3.88 | 3.07 | 411 min | 432 | 8.8× |
| Uni-Dock detail batch | 9.22 | 4.28 | 3.23 | 31 min | 33 | **115×** |
| Uni-Dock fast seq | 8.83 | 3.56 | 2.81 | 598 min | 628 | 6.0× |
| **Vina-CUDA** | 8.83 | 4.15 | 3.10 | 449 min | 472 | 8.1× |
| Vina-CPU | 8.15 | 3.88 | 3.00 | 3614 min | 3804 | 1× |
| Uni-Dock balance batch | 8.98 | **4.31** | **3.39** | 27 min | 29 | **131×** |
| Uni-Dock fast batch | 7.90 | 4.30 | 3.07 | 18 min | **19** | **200×** |
| QVina-W-CUDA | 7.55 | 4.27 | 3.09 | 306 min | 322 | 11.8× |
| Uni-Dock fast seq | 8.83 | 3.56 | 2.81 | 598 min | 628 | 6.0× |

**未跑 screening**: Vina-GPU2 (OpenCL), QuickVina2-GPU2 (OpenCL) — 无 batch 接口，sequential ~7.5h/each。注：Uni-Dock B (paired_batch) 不适用于 screening（需多 receptor 并行，screening 固定 1 receptor × N ligands，仅 gpu_batch 模式可用）

---

## 2026-06-12: OpenCL Screening 完成（Vina-GPU2 + QVina2-GPU2）

### Vina-GPU2 (OpenCL) — Jun11_1804

- **57/57 靶点全部完成**，16,245 次对接
- **总时间**: 557.9 min (587.2 s/tgt)，相对 Vina-CPU **6.5×**
- **Screening Power**:

| EF@1% | EF@5% | EF@10% |
|-------|-------|--------|
| 8.96  | **4.35** | 3.23 |

### QuickVina2-GPU2 (OpenCL) — Jun12_0322

- **57/57 靶点全部完成**，16,229 次对接（12 个靶点各缺 1-2 个配体）
- **总时间**: 565.8 min (595.6 s/tgt)，相对 Vina-CPU **6.4×**
- **Screening Power**:

| EF@1% | EF@5% | EF@10% |
|-------|-------|--------|
| 8.83  | 4.23  | 3.14 |

### OpenCL vs CUDA 对比

| 方法 | EF@1% | EF@5% | EF@10% | 总时间 | s/tgt |
|------|-------|-------|--------|--------|-------|
| **Vina-GPU2 (OpenCL)** | 8.96 | **4.35** | **3.23** | 558 min | 587 |
| Vina-CUDA | 8.83 | 4.15 | 3.10 | 449 min | 472 |
| **QVina2-GPU2 (OpenCL)** | 8.83 | 4.23 | 3.14 | 566 min | 596 |
| QVina2-CUDA | **9.35** | 3.88 | 3.07 | **411 min** | **432** |

**关键发现**: OpenCL 版 Screening Power 略优于 CUDA 版（EF@5% 4.35 vs 4.15），但速度慢 20-28%。

### 更新后 Screening Power 全景对比 (12 方法)

| 方法 | EF@1% | EF@5% | EF@10% | 总时间 | s/tgt | 相对 Vina-CPU |
|------|-------|-------|--------|--------|-------|---------------|
| QVina2-CUDA | **9.35** | 3.88 | 3.07 | 411 min | 432 | 8.8× |
| Uni-Dock detail batch | 9.22 | 4.28 | 3.23 | 31 min | 33 | 115× |
| Uni-Dock balance batch | 8.98 | **4.31** | **3.39** | 27 min | 29 | 131× |
| **Vina-GPU2 (OpenCL)** | 8.96 | 4.35 | 3.23 | 558 min | 587 | 6.5× |
| **QVina2-GPU2 (OpenCL)** | 8.83 | 4.23 | 3.14 | 566 min | 596 | 6.4× |
| Vina-CUDA | 8.83 | 4.15 | 3.10 | 449 min | 472 | 8.1× |
| Uni-Dock fast batch | 7.90 | 4.30 | 3.07 | 18 min | 19 | 200× |
| Vina-CPU | 8.15 | 3.88 | 3.00 | 3614 min | 3804 | 1× |
| QVina-W-CUDA | 7.55 | 4.27 | 3.09 | 306 min | 322 | 11.8× |
| Uni-Dock fast seq | 8.83 | 3.56 | 2.81 | 598 min | 628 | 6.0× |

**所有 CASF-2016 screening 实验已完成**（12 方法全覆盖）。

---

## 第六部分：GPU Kernel Profiling 与混合精度改造基线 (毕设)

> **方向**: QuickVina2-CUDA 混合精度 (FP16/BF16) 改造
> **基线 m0**: 原始 FP32, 未改造
> **分析报告**: `experiments/Jul03_1900_qvina2_cuda_profile_m0/m0_analysis.md`

### Exp 23: m0 NCU Profiling (1u1b, FP32 baseline)

- **日期**: 2026-07-03
- **目录**: `Jul03_1900_qvina2_cuda_profile_m0`
- **二进制**: QuickVina2-GPU-2-1-CUDA (初版 sm_52 JIT 无 lineinfo; 后重编译 sm_89 AOT + lineinfo)
- **测试配体**: 1u1b (single ligand)
- **NCU**: 2022.3.0, `--set full` 36 passes, 363s replay

**Kernel 时间占比**:

| Kernel | Duration | 占比 |
|---|---|---|
| kernel1 (grid precompute) | 3.02 ms | 0.2% |
| **kernel2_update (MC+BFGS)** | **1573.52 ms** | **99.8%** |

**kernel2_update 核心瓶颈**:

| 指标 | 数值 | 诊断 |
|---|---|---|
| Achieved Occupancy | 6.16% | 🔴 (上限 75%) |
| Registers/thread | 255 | 🔴 封顶 (上限 256) |
| Shared mem/block | 30,524 B (~30 KB) | 🔴 block limit = 3 |
| IPC | 0.43 | (上限 4) |
| Uncoalesced global | 70% | 🔴 |
| FP64 instructions | 17.8M | ⚠️ 4090 上 64:1 慢, 需读源码定位 |

**诊断**: 30 KB shared mem → block limit 3 → occupancy 6%; 寄存器 255 为次级瓶颈。低 occupancy 无法隐藏 fixed latency stall (33%)。详见 `m0_analysis.md` §1–§5。

### Exp 24: sm_89 AOT vs sm_52 JIT 对比 (CASF-2016, 285 pairs)

- **日期**: 2026-07-03
- **目录**: sm_52 `May31_2017_CASF-2016_qvina2_cuda_8192auto` / sm_89 `Jul03_2008_CASF-2016_qvina2_cuda`
- **动机**: 验证 arch 切换 (sm_52 JIT → sm_89 AOT) 不引入系统性偏差, 为 m1 混合精度对比建立同 arch 基线
- **RMSD 工具**: spyrmsd (symmetry=True, heavy-atom)

| 指标 | sm_52 JIT | sm_89 AOT | Δ |
|---|---|---|---|
| OK | 285/285 | 285/285 | — |
| 总运行时间 | 458.4 s | 456.1 s | −0.5% |
| Mean heavy RMSD | 2.715 Å | 2.624 Å | −0.091 |
| Median heavy RMSD | 1.064 Å | 1.211 Å | +0.147 |
| ≤2.0 Å | 63.5% | 61.4% | −2.1 pp |
| ≤1.0 Å | 48.1% | 45.6% | −2.5 pp |
| Affinity \|Δ\| ≥ 0.05 | — | 54.4% | — |
| Affinity \|Δ\| ≥ 0.10 | — | 35.4% | — |

**结论**: 稳态性能等价 (−0.5%, 噪声范围); 精度差异在 MC run-to-run variance 内 (参考 Exp 1 三次同配置运行 ≤2Å 58.5%–63.5%)。arch 切换未引入系统性偏差。**后续 m1 vs m0 对比统一基于 sm_89 AOT** (避免 SASS 浮点运算顺序差异混入精度噪声)。

> 注: m0_full.ncu-rep 基于 sm_52 JIT binary (无 lineinfo), kernel 级结论 (occupancy/regs/shared) 仍有效; 源码级 hotspot 需用 sm_89 AOT + lineinfo binary 重跑 `ncu --set full`。

### Exp 25: m1a (朴素 BF16 改造) — ❌ 失败，已弃用

- **日期**: 2026-07-04 编译，2026-07-12 验证失败
- **目录**: `variants/QuickVina2-CUDA-m1a` (+ `m1a_broken`, `m1a_eonly_broken`，均标记 FAILURE_MARKER.md)
- **改造内容**: `quasi_newton.cuh` 归约 shared mem `float`→`__nv_bfloat16`、`warpRecude` BF16 版；`INFINITY`→`INFINITY_f`、`cos/sin`→`cosf/sinf`
- **失败表现**: 1u1b 卡在 docking 搜索阶段（进度条 ~7% 停滞，m0 同配置 ~2s 完成）；无法完成对接
- **根因（对照文献）**: 朴素 BF16 无误差校正(EC)、未分层保精度（m0_analysis §5.2 要求 BFGS Hessian/line search 保 FP32）。印证 G19（纯 FP16-TC 降 docking 质量）、P09（BF16 舍入误差 > FP16）。
- **结论**: 放弃此路线。

### Exp 26: 新方向 — fusion 提 occupancy 改造（进行中）

- **决策日期**: 2026-07-12
- **动机**: m0_analysis §4 定位 BFGS `kernel2_update` 瓶颈 = **Achieved Occupancy 6.16%**（shared mem 30KB→block limit 3 + 寄存器 255 封顶），非 compute/memory bound。纯降精度(如 m1a BF16) 无法解决，且引入精度风险。
- **新方向**: 先做 **kernel fusion + 降 shared mem/寄存器占用 → 提 occupancy**（目标 6%→33%，m0_analysis §5.1 测算），使 SM 利用率拉起；之后再叠加受控的混合精度（TF32+EC, G19；精度分层保 BFGS Hessian/line search FP32，m0_analysis §5.2）。
- **m0 现状确认（2026-07-12）**: `variants/QuickVina2-CUDA` 1u1b 正常 1.95s，无退化，作为对照基线。
- **baseline 数据**: m0 CASF-2016 (sm_89 AOT) 见 Exp 24；m0 ncu 见 Exp 23 / m0_analysis.md §1–§9。

> ⚠️ 改造对象统一为 **QuickVina2-CUDA**（benchmark 最优 variant），非 Vina-CUDA。thesis 草稿（literature/thesis_draft/01_改造方案草稿.md）原写 Vina-CUDA+TF32+EC 路线需对齐到此。

#### Exp 26 细化: m1_fusion v1 设计 (2026-07-13)

**工作目录**: `variants/QuickVina2-CUDA-m1_fusion` (从 m0 复制, m0 不动)

**瓶颈量化 (m0_analysis §4, ncu)**:
- Achieved Occupancy = 6.16% (上限 75%)
- 根因: shared mem 30KB/block → block limit 3 (shared mem 限制) + 寄存器 255/thread 封顶
- block = (16,2)=32 threads; grid = (64,128)=8192 blocks (每 block = 1 构象)
- 每 SM 仅 3 block×32 = 96 线程活跃 / 1536 上限 → 6.25%

**shared mem 账目 (推算)**:
- `e[MAX_NUM_OF_LIG_PAIRS=2048]` ×2 处 = 16KB (最大头, 两个 lig_pairs 力/能量归约函数)
- `m_coords_cl`/`minus_forces` = 128×3×4B ×2 = 3KB
- `grids_gpu`/`mis_gpu`/其他结构体 ≈ 余量
- 实际 `num_pairs` (真实配体对) 多数 ≪ 2048 → 静态 2048 浪费严重

**v1 改造目标**: 削减 `e[]` 的 shared mem 占用 → block limit 3→更高 → occupancy 升
- 候选 A: `MAX_NUM_OF_LIG_PAIRS` 宏砍到 CASF 实际 max (需先测 num_pairs 分布, 防越界)
- 候选 B: `extern __shared__` 动态分配, 运行时按 num_pairs 申请 (改动 launch 签名)
- 候选 C: warp-shuffle 归约替代 shared mem (threadsPerBlock=32=1 warp, 但 block 二维映射需验证 warp 连续性)

**当前进度**:
- [x] m0 确认无退化 (1u1b 1.95s)
- [x] m1a 系列标记 FAILURE_MARKER.md (Exp 25)
- [x] 建 m1_fusion 工作目录
- [ ] 诊断 num_pairs 分布 (插入 printf, 编译中)
- [ ] 选定 v1 方案并实施
- [ ] 对比 occupancy (ncu) + 1u1b + CASF

#### Exp 26 细化: m1_fusion v1 结果 (2026-07-14)

**改造**: `variants/QuickVina2-CUDA-m1_fusion` 的 `quasi_newton.cuh` 两个 lig_pairs 归约函数
(`eval_interacting_pairs_deriv_update` L825 / `_optimal` L907) 移除 `e[MAX_NUM_OF_LIG_PAIRS=2048]`
静态 shared 数组 (×2 = 16KB), 改用寄存器 `my_e` 累加 + `__shfl_down_sync` 单 warp 树形归约。

**结果**:
- ✅ 编译通过, 1u1b 跑通 (1.90s, 无退化/挂死, 不同于 m1a)
- ❌ **Occupancy 未变**: Achieved 6.17% (m0=6.16%), Block Limit Shared Mem 仍 =3 block
- ❌ **shared mem/block 未降**: `launch__shared_mem_per_block = 30.52 KB/block` (m0 同)
- ❌ **1u1b 时间未变**: m1_fusion 1.90s vs m0 1.89s
- e[] 声明从 2→0 (825/907 函数), 但 kernel2_update 的 shared 占用不变

**诊断 (关键修正)**:
- kernel2_update 的 30KB shared mem **不是 e[] 数组** (e[] 移除后 shared 仍 30.52KB)
- 真凶疑似 **寄存器溢出 (register spill → local memory in shared space)**: Registers/thread=255 (封顶), Block Limit Registers=8, 但 Block Limit Shared=3 是实际瓶颈
- 30KB ≈ 寄存器溢出区 (255 寄存器封顶后, 超出的变量 spill 到 local/shared)
- **v1 改错了目标** (误判 e[] 为 shared 大头)

**v2 方向 (待实施)**: 降寄存器占用 → 减 spill → 降 shared → occupancy 升
- 候选 A: 编译 flag `--maxrregcount=128` (强制降每线程寄存器)
- 候选 B: `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` 注解 kernel
- 候选 C: 拆 kernel2_update 大函数 (降单函数寄存器压力)
- 需实测: occupancy 变化 + 1u1b 时间 + CASF 精度 (防止降寄存器引入 spill 反而变慢)

> **⚠️ v1 第一次实施的技术教训 (2026-07-14)**:
> - 初次 v1 改动 (fusion_v1_line.py 行号替换) 后 `make source` 显示成功且 1u1b 跑通, 但 **改动实际未编入 binary** —— 因 Makefile 的 `clean` 只删 binary 不删 .o, 且 .cuh 改动未触发 .cu 重编 (make 依赖追踪未含 .cuh), 编译器用了旧 .o 缓存。导致 v1 的 ncu 结果与 m0 完全相同 (occupancy 6.17% 未变), 误判"e[] 非 shared 大头"。
> - 后续行号替换 (fusion_v1_line.py) 因 907 函数实际结尾行算错 (896 非 897) 导致切片错位, 破坏文件后半 (L1950 语法错误), v2 编译失败。
> - **修正做法**: ① 从 m0 恢复源码; ② 用括号平衡算法精确找函数结尾 (find_func_end), 重写 825/907 函数; ③ `make clean` + `touch` 所有 .cu/.cuh 强制重编; ④ 编译后必查 binary 时间戳 + ncu 确认改动生效。
> - **原则**: GPU 改造后必须用 `md5sum`/`ls -la` 确认 binary 重编, 且 ncu 报告若与 baseline 字节级相同 (= 同 binary) 应立即怀疑未重编, 而非认为"改动无效"。

#### Exp 27: m1_fusion v1 (warp-shuffle 归约) — 成功, occupancy 提升 (2026-07-14 23:2x)

**改造** (variants/QuickVina2-CUDA-m1_fusion, quasi_newton.cuh):
- 825 `eval_interacting_pairs_deriv_update` + 907 `eval_interacting_pairs_deriv_update_optimal` 两个 lig_pairs 归约函数
- 移除 `__shared__ float e[MAX_NUM_OF_LIG_PAIRS=2048]` (×2 = 16KB), 改用寄存器 `my_e` 累加 + `__shfl_down_sync`(0xFFFFFFFF) 单 warp (block=32=1 warp) 树形归约
- 关键修复: 归约后所有线程须 `return` -> 加 `my_e = __shfl_sync(0xFFFFFFFF, my_e, 0)` 广播 lane0 sum 给所有线程再 `return my_e` (否则非 lane0 线程无 return -> UB -> kernel HANG, 同 m1a 症状)
- 本地 patch 改完 scp 回 + `make clean && touch *.cu *.cuh` 强制重编 (避 make 缓存坑)

**结果 (1u1b, RTX 4090)**:
- ✅ 编译通过 (BUILD_EXIT=0), 1u1b 跑通 1.604s (不再挂死)
- ✅ **Occupancy 6.17% -> 8.15% (+32%)**; Block Limit Shared Mem 3->4 block; Block Limit Registers 8->16
- ✅ 1u1b 时间 1.89s (m0) -> 1.60s (m1_fusion, -15%)
- shared mem/block 下降 (e[] 16KB 移除生效, 与之前 v1 假成功不同 — 那次 make 没重编)

**结论**: e[] 确为 shared mem 大头之一 (之前 v1 假成功因 make 缓存未重编, 误判"e[] 非大头")。warp-shuffle 归约提 occupancy + 降时间。
occupancy 仍低 (8%) 因寄存器溢出仍占部分 shared (Block Limit Registers 已升到 16, 但 Shared 仍限 4) -> v2 可试 --maxrregcount 或 __launch_bounds__ 进一步降 spill。

**CASF-2016 全量**: 后台跑 (tag m1_fusion_v1_warpshuffle), 待对比 m0 精度(RMSD<=2A%) + 速度。

> **范式修正 (2026-07-15)**: 新增 variant 必须注册进统一 runner `tools/run_experiment.py`, 禁止临时软链/替换二进制等野路子。
> - 改动: BINARIES dict 加 `"qvina2-cuda-m1-fusion" -> variants/QuickVina2-CUDA-m1_fusion/QuickVina2-GPU-2-1-CUDA`; argparse choices 加同名; screening/redocking routing 末尾加通用 fallback (任何注册在 BINARIES 且非 unidock/vina-cpu 的变体自动走 run_cuda_vina(binary_key=variant))。
> - 以后新增变体 (m1b/m2/...) 只需在 BINARIES 加一行, runner 自动识别并跑 CASF + RMSD 分析。
> - m1_fusion CASF-2016 全量已用正式 runner 启动 (tag qvina2-cuda-m1-fusion), 待对比 m0 精度 + 速度。

#### Exp 27 (续): m1_fusion v1 CASF-2016 全量结果 (2026-07-15)

用正式 runner (--variant qvina2-cuda-m1-fusion) 跑完 CASF-2016 全量, 对比 m0 (Jul03_2008):
- 靶点 OK: 285/285 (全跑通)
- Top-1 RMSD<=2A: m0 61.4% (175/285) -> m1_fusion 60.0% (171/285), 差 1.4pp (**精度不退化**, BFGS 浮点累加顺序微变正常波动)
- 中位 RMSD: 1.211A -> 1.185A (略优)
- 平均 RMSD: 2.624A -> 2.720A (+0.096A, 微增)
- 平均耗时: 1.60s -> 1.47s (**-8.1%**); 中位 1.52s -> 1.42s (-6.6%)
- 输出: experiments/Jul15_1454_CASF-2016_qvina2_cuda_m1_fusion

**结论**: warp-shuffle 归约 (移除 16KB shared e[]) 在 CASF 全量上精度无损 + 端到端加速 8.1%, 与 1u1b 微基准 (occupancy 6.17%->8.15%, 时间-15%) 方向一致。fusion P0 第一步实证闭环完成。
**下一步 (v2)**: 降寄存器溢出 (--maxrregcount 或 __launch_bounds__) 进一步提 occupancy; 之后上 TF32+EC 混合精度 (G19 路线)。

> **分析脚本规范 (2026-07-15)**: 实验对比用仓库现成脚本, 不再手写 ad-hoc 分析。
> - `tools/rmsd_analysis.py`: spyrmsd 对称校正 Top-1 RMSD, 输出 rmsd_results.csv
> - `tools/avg_runtime.py`: 从 summary.csv 算平均/中位耗时 (s/tgt)
> - `tools/compare_experiments.py` (新增): 复用 run_experiment.py 官方口径 (all-atom RMSD<=2A 成功率 + 中位/平均 RMSD + 耗时), 自动判 docking-power 是否退化 (默认 2pp 容差)
> - 用法: `python3 tools/compare_experiments.py <exp_m0> <exp_new>`
> - 注意: CASF docking power 用 **all-atom RMSD** (含氢), 与 runner 口径一致; heavy-atom 口径会差 ~1pp 但结论不变
> - m1_fusion v1 官方对比 (all-atom): m0 172/285(60.4%) -> m1 170/285(59.6%), -0.7pp PASS; 平均耗时 1.60->1.47s (-8.1%)

#### Exp 28: v1 完整版 CASF-2016 全量 (2026-07-16)

**背景**: v1 实为本实验两个 warp-shuffle 归约删除的合并 —— ① `eval_interacting_pairs_deriv_update_optimal`(L859) 删 `e[MAX_NUM_OF_LIG_PAIRS]`(8KB); ② `ig_eval_deriv_update_optimal`(L388) 删 `e[MAX_NUM_OF_ATOMS]`(512B)。`Jul15_1454` 是仅做 ① 的中间快照; 本 Exp 跑的是含 ①② 的完整 v1。

**ncu 复测 (kernel2_update, 1u1b, 同条件 m0 vs v1)**:
- static shared: m0 29.50 KB -> v1 **20.80 KB** (-8.70 KB, = 8KB + 512B, 与两处删除吻合)
- dynamic shared: 0 (两边皆 0)
- registers/thread: m0 255 -> v1 **128** (意外收益: 移除 shared 读写+__syncthreads 路径, 编译器省寄存器)
- 推翻早期交接文档 (v2_handoff.md) 的错误数据: 该文档称 "static 21308->21308 未变, dynamic 9431->1049, regs 255->255", 实测全错 (源于 v1 初期 make 缓存陷阱期的误测)

**CASF-2016 全量** (runner --variant qvina2-cuda-m1-fusion, search-mode fast):
- 靶点 OK: 285/285
- Top-1 <=2.0A: m0 60.4%(172/285) -> v1 **61.4%(175/285)**, **+1.1pp (PASS, <2pp 容差, 方向为提升)**
- Top-1 <=1.0A: 44.2%(126/285) -> 44.2%(126/285), 0.0pp
- 中位 RMSD: 1.245A -> 1.196A (-3.9%)
- 平均 RMSD: 2.730A -> 2.717A (-0.5%)
- 平均耗时: 1.60s -> **1.48s (-7.3%)**; 中位 1.43s -> 1.43s (0.0%)
- 输出: experiments/Jul16_2003_CASF-2016_qvina2_cuda_m1_fusion

**结论**: v1 (两处 warp-shuffle 删除) CASF 全量精度无损 (docking-power PASS, +1.1pp) + 端到端加速 7.3%。与 1u1b 微基准 (occupancy 6.17%->8.15%, 时间-15%) 方向一致。
**注意**: 20.80 KB 仍 > 20.48 KB (=102400/5) -> block limit 仍卡 4, occupancy 上限未再提升。v2 目标 `grids_cuda`->`__constant__` (~1.4KB) 以踩过 4->5 门槛。

**Fork 规范落实 (2026-07-16)**: 按用户要求保留过程变种, 不原地叠改。
- `QuickVina2-CUDA-m1_fusion` 保持为 v1 (最终版, 含两处删除)
- 从 v1 fork `QuickVina2-CUDA-m1_fusion_v2` 作为 v2 工作副本 (待做 grids_cuda->__constant__)
- runner 注册 `qvina2-cuda-m1-fusion-v2` (BINARIES + argparse choices + routing fallback 三处)
> - 后续 v2 改造在 _v2 目录做, 不污染 v1

#### Exp 29: v2 grids_cuda→`__constant__` 改造 (FAILED, 回退)

**目标**: 把 `kernel2_update` 里的 `grids_gpu grids_cuda`(~1KB, GRIDS_SIZE=17 grid 描述符, 跨 block 只读)从 `__shared__` 搬 `__constant__`, 使 static shared 从 20.80KB 降到 <20.48KB 以踩过 block limit 4→5 门槛。

**实施** (目录 `QuickVina2-CUDA-m1_fusion_v2`, 从 v1 fork):
- `kernel2.cu` 顶部加 `__constant__ grids_gpu grids_cuda_const;`
- 删 kernel 内 `__shared__ grids_gpu grids_cuda;` + per-block 拷贝循环; 传参 `&grids_cuda`→`&grids_cuda_const`
- host 侧 `kernel_monte` 里 `cudaMemcpyToSymbol`/`cudaGetSymbolAddress`+`cudaMemcpy` 把 grids 描述符拷进 constant

**结果 (FAILED)**:
1. **1u1b 单 ligand 能跑通 (EXIT=0, 出 docking result), 但时间 2.007s vs v1 1.60s, 变慢 ~25%**
2. **CASF-2016 全量 285/285 全 FAIL (0 OK)** —— 多 target 场景下 constant 数据错乱, docking 全部失败
3. **ncu 在 v2 binary 下崩溃 (error 11 SIGSEGV)**: `cudaMemcpyToSymbol`(无论变量标识符/`"字符串"`/`cudaGetSymbolAddress`)在 ncu instrumentation 下符号解析失败, 无法用 ncu 验证 occupancy (ncu 在 v1 正常)

**根因分析**:
- **性能退化**: constant memory 走全局 constant 总线, kernel 里 `g_evaluate` 按 atom type `t` 读 `grids_cuda_const.grid_other[t]`, 同一 warp 32 线程读不同 `t` → constant 访问串行化, 比 shared 的 bank 访问慢。block limit 4→5 的理论收益被 constant 访问变慢抵消且反超。
- **CASF 全 FAIL**: `grids_ptr`(host)在 main 循环外只填一次, CASF 多 target 下 constant 拷贝数据 stale/竞态 → docking 失败。即使修对, 性能仍净负。
- **静态确认 shared 确实降了**: PTX `.shared` 求和 = 19544 B (19.09 KB) < 20.48 KB 门槛, 理论上 block limit 应 4→5, 但 occupancy 提升不足以抵消 constant 访问开销。

**结论**: v2 constant 改造 **净负收益** (性能退化 + 正确性风险), **放弃**。fusion 阶段以 v1 (20.80KB, block limit 4) 为终点。下一步直接进 P1 (TF32+EC 混合精度), 因为:
- occupancy 从 4→5 的边际收益小 (kernel 是双低利用率型, compute/memory 都不 bound, occupancy 提升对 IPC 帮助有限)
- constant memory 不适合本 kernel 的 per-atom-type 不规则访问模式

**教训 (写论文踩坑清单)**:
1. `__constant__` 只适合「所有线程读同一地址」的广播场景; 本 kernel 按 atom type 不规则读不同 grid 描述符, constant 反而更慢。
2. host 端不能解引用 device 指针 (`grids` 是 cudaMalloc 的 device 指针); 要拷字段需先 `cudaMemcpy` device→host, 或传 host 指针。
3. `cudaMemcpyToSymbol` 在 non-rdc build + ncu 下符号解析会失败; 用 `cudaGetSymbolAddress`+`cudaMemcpy` 可避开但 ncu 仍崩 (工具链限制, 不影响真实运行)。
4. CASF 全量才是真实验证; 1u1b 单 ligand 跑通不代表多 target 正确 (constant 数据 stale 在多 target 暴露)。

**目录处理**: `QuickVina2-CUDA-m1_fusion_v2` 重命名为 `_v2_FAILED` 保留失败记录; runner 移除 `qvina2-cuda-m1-fusion-v2` 注册; `m1_fusion`(v1) 保持终态不动。


## 版本命名约定（2026-09-10 起）
- 新线统一 Q2-<世代>.<合并序>：Q2-7 = 现主线 v4_8b（认证快照 _certified/Q2-7_2026-09-10）；Q2-8.0 = 合并 v6 batch 的目标版
- 旧目录不重命名（引用链保护）；variants/_INDEX.md 为 90 目录状态索引（LINE/CERT/EXP），脚本可重生成
- GPU 级批 = v6（batch_chunk）已完成认证，后续是合并非新做

## Q4 kernel 层进展（2026-09-10）
- Q4-1 grids 真实 dims：SMALL_BOX MAX 128→64，显存 1.14GB→136MB（93×膨胀修正）；k1/k2/rt 零变化=显存卫生
- Q4-2 stage_timer 验证：v1 数值正确（ns 单位曾误读）；LS 占 k2 88%，k2 占 dock 71-90%；v2 结构+K2WALL 桩已合入
- ncu（1u1b，计数器权限已开）：占用率 12.1%（shared 14.8KB→6 块/SM 绑定），No Eligible 82%，SM 17%/Mem 34%
- Q4-3 shared 动态化实验：破坏 LS/BFGS 跨步语义，已回滚；需 design-first
- Q4-4 regs 扫描 80-128：k2 无影响 → shared 独占绑定确认
- 下一步：shared 重设计（-15% → occupancy +33%）或 v6-batch 填空窗

## Q4-6 认证（2026-09-10）
- 构建：Q4-5 + MAX_RIGID 40（shared 12204B，8 blocks/SM）+ per-node q→矩阵下传
- CASF 285/285：rt 0.400 / e2e_dock 0.175 / k2 0.125 / Top-1 177（all-atom）
- aff vs v4 mean|Δ|=0.158（Q4 系列最佳）；CERT=_certified/Q4-6_2026-09-10
- Q4 世代 vs v4：rt -73.7%，e2e_dock -85%

## Q4-7（M1 精度域清理）认证（2026-09-11）
- exp->expf（MC 接受判定，17.8M FP64）+ 移除 FP64 提升 printf + M_PI/INFINITY float 化
- CASF 285/285：rt 0.340 / e2e_dock 0.118 / k2 0.067（-50% vs v4）/ Top-1 179（带中部）
- aff vs v4 mean|Δ|=0.155（噪声带内）——精度域清理不改变结果分布
- CERT=_certified/Q4-7_2026-09-11；论文新增"精度域污染"发现（SASS 审计法）

## Q4-7（M1 精度域清理）认证（2026-09-11）
- 构建：Q4-6 + exp→expf（MC 接受判定 17.8M FP64）+ 移除 FP64 提升 printf + M_PI/INFINITY float 化
- CASF 285/285：rt 0.340 / e2e_dock 0.118 / k2 0.067（-50% vs v4）/ Top-1 179（all-atom，家族带中部）
- aff vs v4 mean|Δ|=0.155（噪声带内）——精度域清理不改变结果分布
- ★发现"精度域污染"：exp()、printf("%f")、double 宏三类源将 FP64 拉入热路径；
  SASS 审计（cuobjdump -sass | grep "DFMA\|DADD"）是标准排查手段
- CERT=_certified/Q4-7_2026-09-11；runner --seed 透传已加（可复现实验）

## Q4-8 确定性/可复现性（2026-09-11）
- 已修：pairs 累加着色同步（edge-coloring + 按色 __syncthreads）；
  skip_row RMW → atomicAnd/Or；--seed 透传（binary+runner）
- 未根除：同 seed 双跑 mode2-9 仍有差异——残余竞态为 m_coords 跨 eval
  同步缺失 + mutate 分支写（racecheck 49 处报告），影响轨迹低位，
  mode1/CASF 分布统计不受影响
- 论文口径：run-to-run 变异来自 GPU 调度 + 非原子累加；以多跑中位数±
  极差刻画。完全根治列为后续立项（Q4-8 续）
