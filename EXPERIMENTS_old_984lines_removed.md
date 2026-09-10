# Experiments Log

## 0. 对接工具输入输出规范 + AutoDockGPU 执行方案

> ⚠️ **实验前必须对照检查，防止参数错误或输出解析偏差。**

### 0.1 Uni-Dock

**输入参数**

| 参数 | 典型值/含义 | 检查项 |
|------|------|------|
| `--receptor` | 受体 PDBQT（刚性部分） | 确认 protonation 态、是否去水 |
| `--gpu_batch` | 配体 PDBQT / SDF（空格分隔） | **必须直接传文件路径，不能用索引文件** |
| `--scoring` | `vina` / `vinardo` / `ad4` | ad4 需额外 `--maps` |
| `--search_mode` | `fast`(128×20) / `balance`(384×40) / `detail`(512×40) | 三档预设，或手动 `--exhaustiveness --max_step` |
| `--center_x/y/z` | 盒子中心 (Å) | 来源：晶体配体几何中心 / COACH-D / 手动 |
| `--size_x/y/z` | 盒子边长 (Å) | 常用 22.5³；盲对接需更大 |
| `--num_modes` | 默认 9；通常设 1 | 虚拟筛选设 1，redocking 可用 9 |
| `--seed` | 0=随机 | 复现实验时固定 |
| `--max_gpu_memory` | 0=全部可用 | OOM 时手动限制 |
| `--cpu` | 0=自动 | IO 线程数 |

**输出格式**

| 项 | 格式/位置 | 解析方式 |
|------|------|------|
| 对接构象 | `--dir` 下 `<ligand>_out.pdbqt` | 每个配体一个文件 |
| 构象内多模式 | 同一个 pdbqt 文件中 `MODEL` / `ENDMDL` 分段 | `num_modes` 控制数量 |
| 结合能 (score) | PDBQT 文件头部 `REMARK VINA RESULT:` 行 | 第一行 = 最佳模式得分 |
| RMSD 下界/上界 | 同一行 `REMARK` | `rmsd/rank` 信息（如有参考） |
| 日志 | stdout（受 `--verbosity` 控制） | 0=静默，1=正常，2=详细 |

**常见陷阱**

1. **`--gpu_batch` 不能传索引文件**：Uni-Dock v1.1.3 的 `--gpu_batch` 直接解析配体，不要传 `ligand_index` 类索引
2. **score 符号**：Vina 评分函数负值越大结合越强；ad4 正值越小结合越强
3. **`num_modes=1` 时仍然输出 MODEL/ENDMDL**：解析时不要假定无 MODEL 标签
4. **多配体 batch 时输出文件名**：`<原文件名>_out.pdbqt`，注意去后缀逻辑

### 0.2 Uni-Dock 批量对接方案（通用）

Uni-Dock 对**任何数据集**都有两种批量方式：

| 方案 | 方式 | 优点 | 缺点 |
|------|------|------|------|
| **A (default)** | 逐靶点独立调用 `unidock`，与 Vina-CPU/QVina-W-CUDA 对齐 | 行为一致，结果可直接对比 | 低配体数（<10）时 kernel launch 开销占比大 |
| **B (batch)** | `--paired_batch_size N --ligand_index pairs.json`，一次 GPU 调用并行多个 1:1 pair | 消除逐次调用开销，速度提升 5-10× | 需准备 JSON + 每 pair 独立 `ligand_config`；无法拆 per-target 耗时 |

> 例：AutoDockGPU（141 个独立靶点）是 B (batch) 的典型适用场景。

**方案 B (batch) JSON 示例**：
```json
{
    "10gs": {
        "protein": "/path/to/10gs/protein.pdbqt",
        "ligand": "/path/to/10gs/flex-xray.pdbqt",
        "ligand_config": "/path/to/10gs_docking_grid.json"
    },
    "1a30": {
        "protein": "/path/to/1a30/protein.pdbqt",
        "ligand": "/path/to/1a30/flex-xray.pdbqt",
        "ligand_config": "/path/to/1a30_docking_grid.json"
    }
}
```
`ligand_config` 内容示例：`{"center_x": 12.5, "center_y": 8.3, "center_z": -2.1, "size_x": 22.5, "size_y": 22.5, "size_z": 22.5}`

```bash
unidock --paired_batch_size 10 --ligand_index pairs.json \
    --size_x 22.5 --size_y 22.5 --size_z 22.5 \
    --search_mode detail --scoring vina --num_modes 1 --dir ./results/
```

### 0.3 Vina-CPU (AutoDock Vina 1.2.x)

| 参数 | 典型值 |
|------|------|
| `--cpu` | 8-20 |
| `--exhaustiveness` | 8 (默认) / 64 / 128 |
| `--num_modes` | 1 或 9 |
| `--seed` | 固定 |

### 0.4 QVina-W-CUDA

| 参数 | 典型值 |
|------|------|
| `--thread` | 8192 |
| `--search_depth` | 启发式公式自动 |
| `--scoring` | vina |

### 0.5 RMSD 分析

```bash
# 批量 RMSD（参考晶体构象）
for f in results/*_out.pdbqt; do
    name=$(basename "$f" _out.pdbqt)
    rmsd=$(obrms "$f" reference.pdbqt 2>/dev/null | tail -1 | awk '{print $NF}')
    echo "$name $rmsd"
done
```
> `obrms` 需要原子序匹配，必要时用 `--heavy` 仅比较重原子。

---

> **Standard params across all runs**: box 22.5³ Å | center = ligand geometric center | `--thread 8192` | `ulimit -s 8192` | `sudo`

---

## Experiment 1: QuickVina-W-CUDA on CASF-2016 (initial)
**Date**: May 27, 2026 | **Status**: ⚠️ Superseded by Experiment 2
**Directory**: `experiments/May27/QuickVina-W-CUDA_results`

- **Success**: 271/285 (95.1%)
- **Failed**: 14 — 12 illegal atom type `G0` (ADFR artifact), 2 protein `ROOT/BRANCH` tag parse errors
- Fixed in Experiment 2 (repair + re-run with full 285)

---

## Experiment 2: QuickVina-W-CUDA on CASF-2016 (fixed)
**Date**: May 28, 2026 | **Directory**: `experiments/May28_2329_CASF-2016_QVina-W-CUDA`

**Repairs applied**:
- 12 G0-type ligands → re-converted via obabel (script: `tools/convert_to_pdbqt.py`, now defaults to obabel)
- 2 proteins (3utu, 5c2h) → filtered `ROOT/BRANCH/ENDROOT/ENDBRANCH` tags

### Docking

| | |
|---|---|
| Success rate | **285/285 (100%)** |
| Total time | 7.7 min |
| Avg time/target | 1.62s |
| Median time | 1.47s |

### RMSD Accuracy (Kabsch alignment, reference = crystal pose from mol2)

| | All-atom |
|---|---|
| Top-1 ≤2.0Å | 63.9% (182/285) |
| Top-1 ≤1.0Å | 32.3% |
| Median RMSD | 1.47Å |
| Mean RMSD | 2.01Å |
| Pearson r (aff. vs RMSD) | 0.223 |

Output: `summary.csv`, `rmsd_results.csv`

---

## Experiment 3: QuickVina-W-CUDA on AutoDockGPU
**Date**: May 28, 2026 | **Directory**: `experiments/May28_2355_AutoDockGPU_QVina-W-CUDA`

**Dataset**: AutoDockGPU — 140 targets, already PDBQT (no conversion). 17 overlap with CASF-2016, 123 new.
Receptor = `protein.pdbqt`, ligand = `flex-xray.pdbqt` (crystal pose).

### Docking

| | |
|---|---|
| Success rate | **140/140 (100%)** |
| Total time | 3.9 min |
| Avg time/target | 1.68s |
| Median time | 1.56s |
| Avg affinity | -8.60 kcal/mol |

### RMSD Accuracy (Kabsch alignment, reference = flex-xray.pdbqt)

| | All-atom | Heavy-atom |
|---|---|---|
| Top-1 ≤2.0Å | **74.5%** | **80.3%** |
| Top-1 ≤1.0Å | 42.3% | 57.7% |
| Median RMSD | 1.20Å | — |
| Mean RMSD | 1.46Å | — |

Output: `summary.csv`, `rmsd_results.csv`

---

## Cross-Dataset Comparison (QVina-W-CUDA)

| | CASF-2016 | AutoDockGPU |
|---|---|---|
| Targets | 285 | 140 |
| Success | 100% | 100% |
| Avg time | 1.62s | 1.68s |
| ≤2Å (all-atom) | 63.9% | 74.5% |
| ≤1Å (all-atom) | 32.3% | 42.3% |
| Median RMSD | 1.47Å | 1.20Å |

AutoDockGPU accuracy advantage likely reflects easier targets (more rigid binding, smaller ligands).
---

## Experiment 4: AutoDock Vina CPU on CASF-2016
**Date**: May 29, 2026 | **Directory**: `experiments/May29_0028_CASF-2016_Vina-CPU_auto`

**Tool**: Standard AutoDock Vina 1.2.3 CPU | `--thread 8192` | `ulimit -s 8192`
Reference ligands: `CASF-2016-PDBQT/coreset/{tid}/{tid}_ligand.pdbqt` (crystal coords in input PDBQT)

### Docking

| | |
|---|---|
| Success rate | **283/285 (99.3%)** |
| Timeouts (300s) | 2 (3ag9, 3uri) |
| Total wall time | ~62 min |
| Avg time/target | 13.1s |
| Max time | 300.0s (timeout cap) |

### RMSD Accuracy (Kabsch alignment, reference = input PDBQT crystal coords)

| | All-atom |
|---|---|
| Top-1 ≤2.0Å | **68.2%** (193/283) |
| Top-1 ≤1.0Å | **39.9%** (113/283) |
| Median RMSD | 1.255Å |
| Mean RMSD | 1.950Å |

Output: `summary.csv`, `rmsd_results.csv`

---

## Experiment 5: AutoDock Vina CPU on AutoDockGPU
**Date**: May 29, 2026 | **Directory**: `experiments/May29_0028_AutoDockGPU_Vina-CPU`

**Dataset**: AutoDockGPU — 140 targets. Receptor = `protein.pdbqt`, ligand = `flex-xray.pdbqt`.

### Docking

| | |
|---|---|
| Success rate | **136/140 (97.1%)** |
| Timeouts (600s) | 4 (3er5, 4er4, plus 2 in original batch) |
| Avg time/target | 29.1s |
| Max time | 600.0s (timeout cap) |

### RMSD Accuracy (Kabsch alignment, reference = flex-xray.pdbqt)

| | All-atom | Heavy-atom |
|---|---|---|
| Top-1 ≤2.0Å | **92.6%** | **94.1%** |
| Top-1 ≤1.0Å | **72.1%** | **79.4%** |
| Median RMSD | 0.662Å | — |
| Mean RMSD | 0.811Å | — |

Output: `summary.csv`, `rmsd_results.csv`

---

## Full Cross-Dataset Comparison (All Methods)

| | CASF-2016 QVina-W-CUDA | CASF-2016 Vina-CPU | AutoDockGPU QVina-W-CUDA | AutoDockGPU Vina-CPU |
|---|---|---|---|---|
| Targets | 285 | 285 | 140 | 140 |
| Success | 100% | 99.3% | 100% | 97.1% |
| Avg time | 1.62s | 13.1s | 1.68s | 29.1s |
| ≤2Å (all-atom) | 63.9% | **68.2%** | 74.5% | **92.6%** |
| ≤1Å (all-atom) | 32.3% | **39.9%** | 42.3% | **72.1%** |
| Median RMSD | 1.47Å | **1.255Å** | 1.20Å | **0.662Å** |

**Key finding**: Standard Vina CPU consistently outperforms QuickVina-W-CUDA in accuracy on both datasets,
at the cost of ~8× longer runtime on CASF-2016 and ~17× on AutoDockGPU.
The accuracy gap is particularly dramatic on AutoDockGPU (92.6% vs 74.5% ≤2Å),
suggesting that QuickVina-Ws heuristic search-space reduction degrades more on
AutoDockGPU's diverse target set.

---

## Experiment 6: Uni-Dock fast on CASF-2016 (A: default)
**Date**: May 30, 2026 | **Directory**: `experiments/May30_1243_CASF-2016_UniDock_fast_A`

**Tool**: Uni-Dock v1.1.3 | `--search_mode fast` (exh=128, max_step=20) | `--scoring vina` | `--num_modes 1`
**Method**: A (default) — 逐靶点 285 次独立 `unidock` 调用，与 Vina-CPU/QVina-W-CUDA 对齐，center = 配体几何中心
**GPU**: RTX 4090

### Docking

| | |
|---|---|
| Success rate | **285/285 (100%)** |
| Total wall time | ~11 min |
| Avg wall time/target | ~2.3s |
| GPU time/target | 50-130 ms |
| Bottleneck | 进程启动 >90% 墙钟时间，GPU 利用率 0% |

### RMSD Accuracy (Kabsch alignment, reference = ligand PDBQT crystal coords)

| | All-atom |
|---|---|
| Top-1 ≤2.0Å | **50.2%** (143/285) |
| Top-1 ≤1.0Å | **39.3%** (112/285) |
| Median RMSD | 1.954Å |
| Mean RMSD | 3.242Å |

### Cross-Mode Comparison (CASF-2016)

| | Uni-Dock fast | QVina-W-CUDA | Vina-CPU |
|---|---|---|---|
| ≤2.0Å | 50.2% | 63.9% | 68.2% |
| ≤1.0Å | 39.3% | 32.3% | 39.9% |
| Median RMSD | 1.954Å | 1.47Å | 1.255Å |
| Mean RMSD | 3.242Å | 2.01Å | 1.95Å |
| Time | 11 min | 7.7 min | 62 min |

> **Note**: fast 模式的 ≤1.0Å (39.3%) 几乎追平 Vina-CPU (39.9%)，但 ≤2.0Å (50.2%) 差距明显，且 Mean RMSD 被少数极端偏靶点拉高（mean=3.24 >> median=1.95）。fast 的 GPU 实际计算只需 <130ms/靶点，墙钟瓶颈在 285 次独立进程启动开销。

Output: `*_out.pdbqt`, `rmsd_results.csv`

---

## Experiment 7: Uni-Dock fast A on CASF-2016 (num_modes=9, unified runner)
**Date**: May 30, 2026 | **Dir**: `experiments/May30_1451_CASF-2016_unidock_A_fast`

| | |
|---|---|
| Tool | Uni-Dock v1.1.3, `--search_mode fast` (exh=128, max_step=20), `--num_modes 9` |
| Method | A (default) — 285 independent calls via unified runner `run_experiment.py` |
| Success | **285/285 (100%)** |
| Total time | **11.1 min** |

### RMSD

| | All-atom |
|---|---|
| ≤2.0Å | **49.8%** (142/285) |
| ≤1.0Å | **37.5%** (107/285) |
| Median RMSD | 2.021Å |
| Mean RMSD | 3.119Å |

### vs Experiment 6 (num_modes=1)

| | Exp 6 (modes=1) | Exp 7 (modes=9) | Δ |
|---|:---:|:---:|:---:|
| ≤2.0Å | 50.2% | 49.8% | −0.4pp |
| ≤1.0Å | 39.3% | 37.5% | −1.8pp |
| Median | 1.954Å | 2.021Å | +0.067Å |
| Mean | 3.242Å | 3.119Å | −0.123Å |

> Differences within random seed noise (`--seed` not fixed). num_modes does NOT affect Top-1 accuracy — search always explores 9 poses internally; num_modes only controls output count.


---

## Experiment 8: Uni-Dock B (paired_batch) chunk sweep — fixed binary

**Date**: May 30, 2026 | **Binary**: Fixed (MAX_NUM_OF_RIGID 24→128, MAX_NUM_OF_LIG_PAIRS 1024→4096)

**Root cause of previous SIGSEGV**: B mode's non-template path used hardcoded `MAX_NUM_OF_RIGID=24` (3ag9 has 37 rigid segments, 3uri has 30) and `MAX_NUM_OF_LIG_PAIRS=1024` (3ag9 has 1943 pairs), causing array overflow. A mode was unaffected (template SizeConfig with 128/4096). Fixed by patching `src/cuda/kernel.h` and adding `target_link_options` in `CMakeLists.txt` for correct cudart_static linking.

All runs: `--search_mode fast --scoring vina --num_modes 9`, box 22.5^3, center=ligand geometric center, unified runner `run_experiment.py`.

### Active Results (serial runs, no GPU contention)

| Chunk | Dir | OK | Time | s/tgt | <=2.0A | <=1.0A | Median |
|-------|-----|-----|------|-------|--------|--------|--------|
| 8 (#2) | `May30_1959_*` | 285 | 8.0 min | 1.68 | 54.7% | 39.6% | 1.642A |
| 4 | `May30_1916_*` | 285 | 9.0 min | 1.90 | 54.0% | 39.3% | 1.703A |
| 2 | `May30_1845_*` | 285 | 11.7 min | 2.47 | 52.6% | 39.6% | 1.797A |

### B ch=8 Cross-Run Variance

| Run | <=2.0A | <=1.0A | Median |
|-----|--------|--------|--------|
| ch=8 #1 (`May30_1819`, inactive) | 49.5% | 39.3% | 2.070A |
| ch=8 #2 (`May30_1959`, active) | 54.7% | 39.6% | 1.642A |
| Delta | +5.2pp | +0.3pp | -0.428A |

285/285 common targets, 82 identical (29%), Median |Delta RMSD| = 0.075A. Same binary, same config — 5.2pp difference is pure random seed.

---

## Experiment 9: Uni-Dock A fast with fixed binary

**Date**: May 30, 2026 | **Dir (active)**: `May30_1944_CASF-2016_unidock_A_fast`

| | |
|---|---|
| Success | **285/285 (100%)** |
| Total time | 11.1 min |
| Avg time | 2.35 s/tgt |
| <=2.0A | **54.7%** |
| <=1.0A | **41.1%** |
| Median RMSD | **1.541A** |
| Mean RMSD | 3.015A |

### A Cross-Run (binary + random seed)

| Run | Binary | <=2.0A | <=1.0A | Median |
|-----|--------|--------|--------|--------|
| Exp 7 | old binary | 49.8% | 37.5% | 2.021A |
| A #1 (`May30_1929`, inactive) | fixed | 55.4% | 41.4% | 1.432A |
| A #2 (`May30_1944`, active) | fixed | 54.7% | 41.1% | 1.541A |

A #1 vs #2: 285 common, 4 identical, Median |Delta RMSD| = 0.188A. Both fixed-binary runs massively outperform old binary (49.8% to 54.7-55.4%), likely due to build environment / CUDA compilation differences rather than the kernel.h struct changes (which shouldn't affect A mode's template path).

---

## Experiment 10: Final Cross-Method Comparison (CASF-2016, all fixed)

| Method | Time | s/tgt | <=2.0A | <=1.0A | Median |
|--------|------|-------|--------|--------|--------|
| **Vina-CPU** (Exp 4) | 62 min | 13.1 | **68.2%** | **39.9%** | **1.255A** |
| **QVina-W-CUDA** (Exp 2) | 7.7 min | 1.62 | 63.9% | 32.3% | 1.470A |
| **Uni-Dock A** (Exp 9) | 11.1 min | 2.35 | 54.7% | 41.1% | 1.541A |
| **Uni-Dock B ch=8** (Exp 8) | **8.0 min** | **1.68** | 54.7% | 39.6% | 1.642A |

### Key Findings

1. **Vina-CPU still accuracy king** (68.2% <=2.0A), but 5-8x slower than GPU methods
2. **Uni-Dock B ch=8** is the best GPU speed/accuracy tradeoff: 8.0 min, <=2.0A ties A mode at 54.7%
3. **Uni-Dock random seed variance is extreme**: same binary, same config gives +-5pp on <=2.0A across runs. Vina-CPU has near-deterministic output; Uni-Dock's GPU MC search is highly seed-sensitive
4. **Chunk size (2/4/8) has no systematic effect on accuracy** — all within noise band
5. **QVina-W-CUDA** remains competitive (63.9% <=2.0A, fastest at 1.62 s/tgt) but <=1.0A lags
6. **Uni-Dock <=1.0A (39-41%) actually beats Vina-CPU (39.9%)** on some runs — a narrow bright spot for GPU docking precision

### Per-Target Variance Pattern

Both A and B modes show median per-target RMSD difference of 0.07-0.19A between identical-config runs, but <5% of targets produce identical results. Aggregate metrics are more stable than individual targets, but +-5pp swings in <=2.0A suggest that single-run comparisons are unreliable for Uni-Dock on CASF-2016.

---

## Exp 13: Uni-Dock A fast (128×20) — CASF-2016 re-run
**Date**: 2026-05-31  
**Dir**: `experiments/May31_1202_CASF-2016_unidock_A_fast`
**Setup**: Uni-Dock v1.1.3 (repaired binary), fast mode (exh=128, max_step=20), num_modes=9, scoring=vina, box=22.5³ Å, GPU RTX 4090, sudo+ulimit 8192

| Metric | Value |
|--------|-------|
| OK | 285/285 |
| Wall time | 11.0 min |
| s/tgt | 2.29 mean / 2.24 median |
| ≤2Å | **54.7%** (156/285) |
| ≤1Å | **41.8%** (119/285) |
| Median RMSD | 1.600Å |
| Mean RMSD | 3.107Å |

**Cross-run comparison (vs May30_1944)**: Aggregate ≤2Å identical (54.7%). Per-target: only 3/285 identical RMSD; median |Δ| = 0.280Å; ≤2Å agreement = 83.9%. Confirms ±5pp aggregate / ~~~1.1Å median per-target random seed variance documented earlier.

---

## Exp 14: QVina-W-CUDA 8192 heuristic — CASF-2016 (May31 re-run)
**Date**: 2026-05-31  
**Dir**: `experiments/May31_1228_CASF-2016_qvina_w_cuda`  
**Setup**: QVina-W-CUDA, thread=8192 (heuristic depth), box=22.5³ Å, GPU RTX 4090, sudo+ulimit 8192

| Metric | Value |
|--------|-------|
| OK | 285/285 |
| Wall time | 8.0 min |
| s/tgt | ~1.7s |
| ≤2Å | **61.4%** (175/285) |
| ≤1Å | **48.4%** (138/285) |
| ≤0.5Å | 26.3% (75/285) |
| Median RMSD | 1.030Å |
| Mean RMSD | 2.591Å |

**QVina-W-CUDA low-thread sweep** (1a30 single-target):
| thread×depth | MC | Result |
|:---|---:|:---|
| 128×20 | 2,560 | SIGABRT |
| 1000×3 | 3,000 | g_evaluate #2 → SIGABRT |
| 1024×10 | 10,240 | g_evaluate #2 → SIGABRT |
| 2048×10 | 20,480 | g_evaluate #5, massive error spam |
| 8192×heuristic | ~15K | ✅ OK |

Conclusion: QVina-W-CUDA requires ~15K+ MC budget. 2560 MC (Vina-CPU exh=8 global_steps=320, Uni-Dock fast 128×20) is architecturally unreachable on this backend.

---

## RMSD Method Migration: obrms / 自研 Kabsch → spyrmsd

**Date**: 2026-05-31  
**Decision**: 统一所有 RMSD 计算到 spyrmsd (Meli & Biggin, J. Cheminf. 2020)

**动机**:
- 之前三套方法并存：`rmsd_analysis.py`（自研重原子 Kabsch）、`run_experiment.py`（obrms）、手动跑（spyrmsd）
- 不同方法间同一输出可差 ~1.5Å（如 1a30: 自研 Kabsch 2.67 vs obrms 4.44），导致跨实验不可比
- 论文中需要可引用的标准工具

**变更**:
- `rmsd_analysis.py`: spyrmsd.rmsdwrapper(symmetry=True), Top-1 (MODEL 1)
- `run_experiment.py`: 同上，替代 obrms
- CSV 列统一: `target,status,rmsd_all_atom,rmsd_heavy_atom,n_atoms`
- 依赖: `python3-openbabel` (AICloud apt install)

---

## QVina-W-CUDA 8192: Three-run comparison (spyrmsd unified)

**Dir**: `May28_2329`, `May31_1228`, `May31_1300`

Three independent runs of QVina-W-CUDA thread=8192 heuristic, all re-analyzed with spyrmsd:

| Metric | May28 | May31 v1 | May31 v2 |
|--------|-------|----------|----------|
| Valid | 285/285 | 285/285 | 284/285 |
| ≤2.0Å | 63.5% (181) | 61.4% (175) | 58.5% (166) |
| ≤1.0Å | 44.2% (126) | 46.7% (133) | 45.4% (129) |
| Median | 1.198Å | 1.171Å | 1.342Å |
| Mean | 2.699Å | 2.700Å | 2.868Å |

**Pairwise consistency (spyrmsd)**:
- May31v1 vs May31v2: median |Δ| = 0.12Å, ≤2Å agreement = 90.1%
- May28 vs May31v1: median |Δ| = 0.14Å, ≤2Å agreement = 88.1%

**Note**: 之前用自研 Kabsch 分析 May28 得到 ≤1Å 32.3%/≤2Å 63.9%/med 1.47，与 May31 的 obrms 差 16pp——已确认为 RMSD 算法不一致所致，非对接差异。统一 spyrmsd 后三跑完全在随机种子方差范围内。

**Baseline (spyrmsd)**: QVina-W-CUDA 8192 ≈ 62% ≤2Å / 45.5% ≤1Å / median ~1.24Å.

---

## Exp 15: Uni-Dock A balanced (384×40) — CASF-2016
**Date**: 2026-05-31  
**Dir**: `experiments/May31_1758_CASF-2016_unidock_A_balance`  
**Setup**: Uni-Dock v1.1.3 (repaired binary), balanced mode (exh=384, max_step=40), num_modes=9, scoring=vina, box=22.5^3 Å, GPU RTX 4090

| Metric | Value |
|--------|-------|
| OK | 285/285 |
| Total time | 11.2 min |
| s/tgt | 2.36 |
| <=2A | **58.9%** (168/285) |
| <=1A | **42.5%** (121/285) |
| Median RMSD | 1.397A |
| Mean RMSD | 2.901A |

### Fast vs Balance

| | Fast (128×20) | Balance (384×40) | Delta |
|---|---|---|---|
| <=2A | 54.0% | **58.9%** | +4.9pp |
| <=1A | 38.2% | **42.5%** | +4.3pp |
| Median | 1.660A | **1.397A** | -0.26A |
| Mean | 3.107A | **2.901A** | -0.21A |
| s/tgt | 2.29 | 2.36 | +0.07s |
| Total | 11.0 min | 11.2 min | +0.2 min |

MC budget 6x (2560 -> 15360) yields ~5pp accuracy gain with negligible time cost (wall clock bottleneck is process startup, not GPU compute).


---

## Exp 16: Uni-Dock A detail (512×40) — CASF-2016
**Date**: 2026-05-31  
**Dir**: 
**Setup**: Uni-Dock v1.1.3 (repaired binary), detail mode (exh=512, max_step=40), num_modes=9, scoring=vina, box=22.5³ Å, GPU RTX 4090

| Metric | Value |
|--------|-------|
| OK | 285/285 |
| Total time | 11.3 min |
| s/tgt | 2.37 |
| <=2A | **62.5%** (178/285) |
| <=1A | **44.6%** (127/285) |
| Median RMSD | 1.171A |
| Mean RMSD | 2.617A |

### Uni-Dock A Search Mode Sweep

| Mode | MC | <=1A | <=2A | Median | Mean | Time | s/tgt |
|------|-----|------|------|--------|------|------|-------|
| Fast (128×20) | 2,560 | 38.2% | 54.0% | 1.660A | 3.107 | 10.9min | 2.29 |
| Balance (384×40) | 15,360 | 42.5% | 58.9% | 1.397A | 2.901 | 11.2min | 2.36 |
| Detail (512×40) | 20,480 | 44.6% | 62.5% | 1.171A | 2.617 | 11.3min | 2.37 |

Detail <=2A=62.5% approaches QVina-W-CUDA (63.5%) and Vina-CPU (65.0%). Wall time nearly constant across modes — process startup dominates GPU kernel time.

---

## Exp 17: Vina-CUDA 8192 heuristic — CASF-2016
**Date**: 2026-05-31  
**Dir**: `experiments/May31_1930_CASF-2016_vina_cuda_8192auto`
**Setup**: Vina-CUDA (Vina-GPU-2-1-CUDA), thread=8192, search_depth=heuristic, RILC-BFGS on (default), num_modes=9, box=22.5³ Å, GPU RTX 4090, `ulimit -s 8192`

| Metric | Value |
|--------|-------|
| OK | 285/285 |
| Total time | 8.1 min |
| s/tgt | 1.70 |
| ≤2Å | **58.9%** (168/285) |
| ≤1Å | **44.6%** (127/285) |
| Median RMSD | 1.223Å |
| Mean RMSD | 2.917Å |


---

## Exp 20: QuickVina2-GPU2 8192 heuristic — CASF-2016
**Date**: 2026-05-31
**Dir**: `experiments/May31_2232_CASF-2016_qvina2_gpu2`
**Setup**: QuickVina2-GPU 2.1 (QuickVina2-GPU, OpenCL on CUDA), thread=8192, search_depth=heuristic, **traditional BFGS only (no RILC-BFGS)**, num_modes=9, box=22.5^3 A, GPU RTX 4090, `ulimit -s 8192`, OpenCL kernels compiled at runtime (`BUILD_KERNEL_FROM_SOURCE` enabled), `-Werror` removed from OpenCL build options (9 warnings on CL3.0/sm_89 otherwise fail compilation).

| Metric | Value |
|--------|-------|
| OK | 285/285 |
| Total time | 10.7 min |
| s/tgt | 2.22 |
| <=2A | **59.3%** (169/285) |
| <=1A | **42.1%** (120/285) |
| Median RMSD | 1.351A |
| Mean RMSD | 2.869A |

### Observations

QuickVina2-GPU2 is the first OpenCL variant confirmed working on RTX 4090 **without RILC-BFGS**. Two fixes were required:

1. **`-Werror` in OpenCL build options**: NVIDIA OpenCL 3.0 compiler (sm_89) generates 9 warnings for deprecated API calls (e.g., `clCreateCommandQueue`). With `-Werror`, these are treated as errors → `CL_BUILD_PROGRAM_FAILURE`. Remove `-Werror` → 9 warnings but kernel compiles and runs correctly.

2. **`--opencl_binary_path` not supported**: Unlike Vina-GPU2, QuickVina2-GPU2 main.cpp has no `--opencl_binary_path` argument. The runner must not pass this flag. The binary always compiles kernels from source (BUILD_KERNEL_FROM_SOURCE).

Accuracy comparison with QuickVina2-CUDA (same search heuristic, but RILC-BFGS):
- ≤1A: 42.1% vs 44.9% (−2.8pp)
- ≤2A: 59.3% vs 62.8% (−3.5pp)
- Median: 1.351A vs 1.190A (+0.161A)

Note: 3dx1 and 4rfm were transient failures in the initial batch run (likely I/O contention). Both completed successfully on retry with sub-angstrom RMSD (0.87A and 0.36A).
- s/tgt: 2.22 vs 1.61 (+38%)

The 3-4pp accuracy gap is consistent with the RILC-BFGS → traditional BFGS regression. OpenCL runtime overhead adds ~38% to per-target time vs CUDA.

Three key architectural differences from Vina-GPU2:
1. No `--rilc_bfgs` flag — traditional BFGS only
2. No `--opencl_binary_path` — always compiles from source
3. Different search_depth formula: `(0.24N + 0.29·DoF - 5.74)` vs Vina-GPU2's `(0.24N + 0.29·size - 3.41)`

### Fix Log

1. `-Werror` removed → kernel compiles (9 warnings instead of 9 errors)
2. Runner: removed `opencl_binary_path` arg from qvina2-gpu2 call
3. Runner: `cd workdir` before sudo (general fix for all OpenCL variants)
### Cross-GPU Comparison (spyrmsd, CASF-2016 285)

| Variant | <=1A | <=2A | Median | s/tgt | Total |
|---------|------|------|--------|-------|-------|
| Vina-CPU (auto) | 46.6% | **65.0%** | 1.072A | 11.06 | 52.1min |
| Vina-GPU2 8192 | **46.7%** | 63.2% | **1.075A** | 2.18 | 10.3min |
| QVina-W-CUDA 8192 | 44.2% | **63.5%** | 1.20A | 1.62 | 7.7min |
| QuickVina2-CUDA 8192 | 44.9% | 62.8% | 1.190A | 1.61 | 7.6min |
| QuickVina2-GPU2 8192 | 42.1% | 59.3% | 1.351A | 2.22 | 10.7min |
| Uni-Dock A detail | 44.6% | 62.5% | 1.171A | 2.37 | 11.3min |
| Vina-CUDA 8192 | 44.6% | 58.9% | 1.223A | 1.70 | 8.1min |
| Uni-Dock A balance | 42.5% | 58.9% | 1.397A | 2.36 | 11.2min |
| Uni-Dock B ch8 | 39.6% | 54.7% | 1.642A | 1.68 | 8.0min |
| Uni-Dock A fast | 38.2% | 54.0% | 1.660A | 2.29 | 10.9min |

Vina-GPU2 (OpenCL) achieves the best GPU accuracy -- <=1A=46.7% surpasses Vina-CPU (46.6%) and median 1.075A nearly matches Vina-CPU (1.072A) -- at the cost of OpenCL overhead (2.18 s/tgt, +35% vs CUDA variants). QuickVina2-CUDA is the fastest overall (1.61 s/tgt). QVina-W-CUDA leads on <=2A (63.5%). All four Vina-GPU-family variants (Vina-GPU2/QVina-W/QuickVina2/Vina-CUDA) share the same `quasi_newton.cuh` RILC-BFGS kernel -- the accuracy differences stem from search heuristics, G-check/I-check variants, and thread scheduling, not the BFGS implementation itself.

All s/tgt computed via `tools/avg_runtime.py` for consistency: sum(OK runtime_s) / count(OK).
---

## Exp 18: QuickVina2-CUDA 8192 heuristic — CASF-2016
**Date**: 2026-05-31
**Dir**: 
**Setup**: QuickVina2-CUDA (QuickVina2-GPU-2-1-CUDA), thread=8192, search_depth=heuristic, RILC-BFGS on (default, no traditional BFGS fallback), num_modes=9, box=22.5³ Å, GPU RTX 4090, 

| Metric | Value |
|--------|-------|
| OK | 285/285 |
| Total time | 7.6 min |
| s/tgt | 1.61 |
| ≤2Å | **62.8%** (179/285) |
| ≤1Å | **44.9%** (128/285) |
| Median RMSD | 1.190Å |
| Mean RMSD | 2.833Å |

### QVina-W-CUDA vs QuickVina2-CUDA

| | QVina-W-CUDA | QuickVina2-CUDA | Δ |
|---|---|---|---|
| s/tgt | 1.62 | 1.61 | −0.01 |
| ≤1Å | 44.2% | 44.9% | +0.7pp |
| ≤2Å | 63.5% | 62.8% | −0.7pp |
| Median | 1.20Å | 1.190Å | −0.01Å |

Within random seed noise (±2-3pp). Consistent with source-code analysis: both share the same `quasi_newton.cuh` RILC-BFGS kernel; QuickVina2 drops G-check (global Octree) but retains I-check (individual buffer). For CASF-2016 redocking, G-check provides negligible accuracy gain.

All s/tgt computed via `tools/avg_runtime.py`.


---

## Exp 19: Vina-GPU2 8192 heuristic — CASF-2016
**Date**: 2026-05-31
**Dir**: `experiments/May31_2050_CASF-2016_vina_gpu2_8192auto`
**Setup**: Vina-GPU 2.1 (AutoDock-Vina-GPU-2-1, OpenCL on CUDA), thread=8192, search_depth=heuristic, RILC-BFGS on (default), num_modes=9, box=22.5^3 A, GPU RTX 4090, `ulimit -s 8192`, OpenCL kernels compiled at runtime (`BUILD_KERNEL_FROM_SOURCE` enabled)

| Metric | Value |
|--------|-------|
| OK | 285/285 |
| Total time | 10.3 min |
| s/tgt | 2.18 |
| <=2A | **63.2%** (180/285) |
| <=1A | **46.7%** (133/285) |
| Median RMSD | 1.075A |
| Mean RMSD | 2.567A |

### Observations

Vina-GPU2 is the most accurate GPU variant on CASF-2016 — <=1A=46.7% actually **surpasses** Vina-CPU (46.6%), and median 1.075A is nearly identical to Vina-CPU (1.072A). Tradeoff: OpenCL runtime kernel compilation adds overhead (2.18 s/tgt vs 1.61 for QuickVina2-CUDA).

Architecture note: Vina-GPU2 uses OpenCL (not CUDA), with kernels compiled from source at each invocation (`BUILD_KERNEL_FROM_SOURCE` enabled). The `--opencl_binary_path` parameter exists for pre-compiled .bin loading but is bypassed in the shipped binary. Running from the variant directory is required (relative paths to `OpenCL/src/kernels/`).

All s/tgt computed via `tools/avg_runtime.py`.

---

## Exp 21: CASF-2016 Scoring Power Evaluation

**Date**: 2026-06-05
**Script**: `tools/scoring_power_py3.py` (Python 3, no sklearn dependency)
**Benchmark**: CASF-2016 Scoring Power — Pearson correlation between predicted score (negated Vina affinity) and experimental logKa across 285 coreset targets.

### Scoring Power Results

| Method | N | Pearson R | R² | SD (logKa) | p-value |
|--------|--:|----------:|---:|-----------:|--------:|
| **Vina-CPU** | 283 | **0.601** | **0.361** | 1.738 | 3.4e-29 |
| **Vina-GPU2** (OpenCL) | 285 | **0.594** | 0.353 | 1.749 | 1.5e-28 |
| Uni-Dock A detail | 285 | 0.578 | 0.334 | 1.774 | 8.5e-27 |
| QuickVina2-CUDA | 285 | 0.568 | 0.323 | 1.790 | 9.7e-26 |
| QVina-W-CUDA | 285 | 0.566 | 0.320 | 1.793 | 1.6e-25 |
| Vina-CUDA | 285 | 0.563 | 0.317 | 1.798 | 3.4e-25 |
| QuickVina2-GPU2 (OpenCL) | 283 | 0.562 | 0.316 | 1.791 | 5.9e-25 |

**CASF-2016 paper benchmarks**: AutoDock Vina R=0.566, X-Score HM R=0.644, GlideScore-SP R=0.576.

### Pairwise Mean |Δscore| (kcal/mol)

| | Vina-CPU | Vina-GPU2 | QVina-W | Vina-CUDA | QV2-CUDA | QV2-GPU2 | UniDock-dt |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Vina-CPU | 0.000 | 0.165 | 0.234 | 0.239 | 0.245 | 0.270 | 0.168 |
| Vina-GPU2 | 0.165 | 0.000 | 0.169 | 0.170 | 0.179 | 0.189 | 0.143 |
| QVina-W-CUDA | 0.234 | 0.169 | 0.000 | 0.145 | 0.159 | 0.188 | 0.188 |
| Vina-CUDA | 0.239 | 0.170 | 0.145 | 0.000 | 0.162 | 0.163 | 0.187 |
| QV2-CUDA | 0.245 | 0.179 | 0.159 | 0.162 | 0.000 | 0.201 | 0.200 |
| QV2-GPU2 | 0.270 | 0.189 | 0.188 | 0.163 | 0.201 | 0.000 | 0.204 |
| UniDock-dt | 0.168 | 0.143 | 0.188 | 0.187 | 0.200 | 0.204 | 0.000 |

### Per-Target Score Spread (7 methods, 281 common targets)

| Metric | Value |
|--------|-------|
| Mean spread | 0.453 kcal/mol |
| Median spread | 0.210 kcal/mol |
| Max spread | 3.352 kcal/mol |
| Std | 0.535 |
| >1.0 kcal/mol | 38 targets (13.5%) |
| >2.0 kcal/mol | 5 targets (1.8%) |

### Score Distribution

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Vina-CPU | -8.85 | 1.96 | -15.74 | -4.02 |
| Vina-GPU2 | -8.78 | 1.96 | -15.70 | -4.40 |
| QVina-W-CUDA | -8.69 | 1.95 | -15.80 | -4.00 |
| Vina-CUDA | -8.68 | 1.95 | -15.70 | -4.00 |
| QuickVina2-CUDA | -8.68 | 1.97 | -15.80 | -4.10 |
| QuickVina2-GPU2 | -8.64 | 1.92 | -15.70 | -4.10 |
| Uni-Dock detail | -8.76 | 1.93 | -15.76 | -4.21 |

### Key Findings

1. **All methods share the same Vina scoring function** — score differences arise purely from different local optima found by different search algorithms, not from scoring function changes.
2. **Vina-CPU and Vina-GPU2 score most accurately** (R > 0.59): more thorough search finds conformations closer to the global energy minimum, yielding better correlation with experimental affinity.
3. **Score consistency is high**: 87% of targets have <1.0 kcal/mol spread across 7 methods; only 1.8% exceed 2.0 kcal/mol.
4. **Scoring Power is inherently limited** (R ≈ 0.57–0.60, R² ≈ 32–36%): the Vina scoring function explains only ~1/3 of the variance in experimental binding affinity. This is a fundamental limitation of empirical scoring functions, not introduced by GPU acceleration.
5. **Closest score pairs**: Vina-GPU2 ↔ Uni-Dock (0.143 kcal/mol mean |Δ|), Vina-CPU ↔ Vina-GPU2 (0.165). These pairs likely find similar conformational basins.
6. **QVina-W-CUDA ↔ Vina-CUDA ↔ QuickVina2-CUDA cluster** (mean |Δ| ≈ 0.15–0.16): all share the same RILC-BFGS kernel and similar search heuristics, producing nearly identical scores.


---

## Exp 22: CASF-2016 Ranking Power Evaluation

**Date**: 2026-06-05
**Script**: `tools/ranking_power_py3.py` (Python 3, no sklearn dependency)
**Benchmark**: CASF-2016 Ranking Power — 57 target groups × 5 ligands each. For each group, compute Spearman ρ, Kendall τ, and Predictive Index (PI) between score ranking and experimental logKa ranking, then average across groups.

### Ranking Power Results

| Method | N | SP | SP_std | τ | τ_std | PI | PI_std |
|--------|--:|---:|-------:|---:|------:|---:|-------:|
| **Vina-CPU** | 55 | **0.542** | 0.408 | **0.473** | 0.376 | **0.578** | 0.417 |
| Uni-Dock detail | 57 | 0.502 | 0.411 | 0.432 | 0.382 | 0.539 | 0.421 |
| QuickVina2-GPU2 | 55 | 0.499 | 0.444 | 0.437 | 0.406 | 0.533 | 0.441 |
| Vina-GPU2 | 57 | 0.480 | 0.431 | 0.414 | 0.387 | 0.515 | 0.439 |
| Vina-CUDA | 57 | 0.472 | 0.440 | 0.407 | 0.408 | 0.508 | 0.448 |
| QuickVina2-CUDA | 57 | 0.471 | 0.437 | 0.409 | 0.383 | 0.507 | 0.435 |
| QVina-W-CUDA | 57 | 0.452 | 0.458 | 0.393 | 0.406 | 0.492 | 0.457 |

**Note**: Vina-CPU and QuickVina2-GPU2 have N=55 (not 57) due to timeout/parse failures in 2 target groups.

**CASF-2016 paper benchmarks**:
- AutoDock Vina: SP=0.441, τ=0.313, PI=0.497
- X-Score HM: SP=0.577, τ=0.436, PI=0.629
- GlideScore-SP: SP=0.547, τ=0.409, PI=0.601

### Per-Target Spearman Distribution (Vina-CPU, N=55)

| Metric | Value |
|--------|-------|
| Mean | 0.542 |
| Median | 0.700 |
| Min | -0.600 |
| Max | 1.000 |
| SP > 0.5 | 33/55 (60%) |
| SP > 0.0 | 48/55 (87%) |
| SP < -0.5 | 1/55 (2%) |

### Key Findings

1. **Vina-CPU leads ranking power** (SP=0.542, τ=0.473, PI=0.578): more thorough search finds conformations whose scores better reflect true binding affinity ordering.
2. **All methods surpass the CASF-2016 AutoDock Vina baseline** (SP=0.441): even the worst GPU method (QVina-W-CUDA, SP=0.452) outperforms it. This likely reflects improvements in Vina 1.2.x vs the older version used in the CASF-2016 benchmark, or differences in docking parameters.
3. **High per-target variance** (SP_std ≈ 0.41–0.46): ranking is highly target-dependent. Median SP=0.700 vs mean 0.542 shows a right-skewed distribution — most targets rank well, but a few drag down the average.
4. **Uni-Dock detail ranks surprisingly high** (SP=0.502): despite lower docking accuracy (≤2Å=62.5%), its scoring captures the relative affinity ordering well, possibly because detail mode explores enough conformational diversity to find near-optimal poses for most ligands.
5. **QuickVina2-GPU2 (no RILC-BFGS) slightly outperforms Vina-GPU2** in ranking (SP=0.499 vs 0.480), contrary to the docking accuracy trend. This suggests that for relative ranking within a target, the absolute pose quality matters less than consistent conformational sampling across ligands.
6. **Kendall τ values (0.39–0.47) exceed CASF-2016 Vina baseline** (τ=0.313) by a large margin, suggesting that the RILC-BFGS and improved search heuristics in Vina-GPU variants produce more reliable score ordering than classical Vina.


---

## Exp 23: CASF-2016 Scoring & Ranking Power — Full 9-Method Comparison

**Date**: 2026-06-05
**Script**: `tools/scoring_ranking_9methods.py` (Python 3, supersedes Exp 21 & 22 individual runs)

### Scoring Power (score vs experimental logKa)

| Method | N | Pearson R | R² | SD |
|--------|--:|----------:|---:|---:|
| **Vina-CPU** | 283 | **0.601** | **0.361** | **1.738** |
| Vina-GPU2 | 285 | 0.594 | 0.353 | 1.749 |
| UniDock detail (512×40) | 285 | 0.578 | 0.334 | 1.774 |
| QuickVina2-CUDA | 285 | 0.568 | 0.323 | 1.790 |
| QVina-W-CUDA | 285 | 0.566 | 0.320 | 1.793 |
| UniDock balance (384×40) | 285 | 0.564 | 0.318 | 1.796 |
| Vina-CUDA | 285 | 0.563 | 0.317 | 1.798 |
| QuickVina2-GPU2 | 283 | 0.562 | 0.316 | 1.791 |
| UniDock fast (128×20) | 285 | 0.523 | 0.274 | 1.853 |

CASF-2016 Vina baseline: R = 0.566.

### Ranking Power (intra-target score ordering vs logKa ordering)

| Method | N | SP | τ | PI |
|--------|--:|---:|---:|---:|
| **Vina-CPU** | 55 | **0.542** | **0.473** | **0.578** |
| UniDock detail (512×40) | 57 | 0.502 | 0.432 | 0.539 |
| QuickVina2-GPU2 | 55 | 0.499 | 0.437 | 0.533 |
| Vina-GPU2 | 57 | 0.480 | 0.414 | 0.515 |
| Vina-CUDA | 57 | 0.472 | 0.407 | 0.508 |
| QuickVina2-CUDA | 57 | 0.471 | 0.409 | 0.507 |
| UniDock balance (384×40) | 57 | 0.456 | 0.389 | 0.493 |
| QVina-W-CUDA | 57 | 0.452 | 0.393 | 0.492 |
| UniDock fast (128×20) | 57 | 0.432 | 0.382 | 0.461 |

CASF-2016 Vina baseline: SP=0.441, τ=0.313, PI=0.497.

### Uni-Dock MC Budget Effect on Scoring/Ranking

| Mode | MC budget | Scoring R | Ranking SP | Ranking PI |
|------|--------:|----------:|-----------:|-----------:|
| Fast (128×20) | 2,560 | 0.523 | 0.432 | 0.461 |
| Balance (384×40) | 15,360 | 0.564 | 0.456 | 0.493 |
| Detail (512×40) | 20,480 | 0.578 | 0.502 | 0.539 |

8× MC budget (fast → detail): Scoring R +0.055, Ranking SP +0.070, Ranking PI +0.078.
Diminishing returns from balance → detail (1.33× MC): Scoring R +0.014, Ranking SP +0.046, Ranking PI +0.046.

### Key Findings

1. **MC budget directly improves scoring/ranking**: Uni-Dock fast → detail shows monotonic improvement across all metrics. More thorough search finds lower-energy conformations, yielding scores that better correlate with experimental affinity.
2. **Uni-Dock fast is the only method below CASF-2016 Vina baseline** in both scoring (R=0.523 vs 0.566) and ranking (SP=0.432 vs 0.441, PI=0.461 vs 0.497). Its 2,560 MC budget is insufficient for reliable scoring/ranking.
3. **Uni-Dock balance crosses the CASF-2016 Vina threshold** (R=0.564, SP=0.456, PI=0.493), making it the minimum viable MC budget for scoring/ranking tasks.
4. **Uni-Dock detail rivals Vina-CPU in ranking** (SP=0.502 vs 0.542, PI=0.539 vs 0.578) — remarkable for a GPU method running 5× faster.
5. **GPU variant scoring cluster** (R=0.562–0.594): all non-Uni-Dock GPU methods fall within a narrow 0.032 R range, confirming that scoring differences are secondary to the shared Vina scoring function.
6. **All Kendall τ values (0.382–0.473) massively exceed CASF-2016 Vina** (τ=0.313), likely reflecting Vina 1.2.x improvements and RILC-BFGS benefits.


## Experiment 24: Uni-Dock Screening Power (CASF-2016)

**Date**: 2026-06-05  
**Method**: Uni-Dock fast mode, 1 receptor × 285 ligands (GPU batch)  
**Targets**: 57 screening targets from CASF-2016  
**Total dockings**: 16,245 (57 × 285)  
**Runtime**: 17.7 minutes (~18.6s per target)  
**Success**: 57/57 targets, all 285 ligands per target

### Enrichment Factor Results

| Cutoff | Mean EF | Median EF | Targets with EF>0 |
|--------|--------:|----------:|------------------:|
| 1% (top 3) | 8.09 | 0.00 | 29/57 (51%) |
| 5% (top 14) | 4.05 | 4.07 | 36/57 (63%) |
| 10% (top 28) | 2.89 | 2.04 | 45/57 (79%) |

### Key Findings

1. **Bimodal EF @ 1%**: 29 targets have binder in top-3 (EF=19-57), 28 targets fail completely (EF=0). This reflects the fundamental challenge of docking scoring functions.
2. **Top performers** (EF_1% ≥ 19): 3fv1 (57), 3e93 (38), 3kr8 (38), 1e66 (19), 1o3f (19), 1sqa (16), 2p15 (19), 2r9w (19), 2vw5 (19), 2x00 (16), 3ge7 (19), 3gnw (19), 3nw9 (19), 4pcs (19), 5dwr (16).
3. **Complete failures** (all EF=0): 2cet, 2qbp, 2zcq, 3dd0, 3ejr, 3myg, 3qqs, 3u8n, 3uex, 4gid, 4tmn, 5c2h — these targets' true binders score poorly relative to decoys.
4. **EF @ 5% is more stable**: mean=4.05, median=4.07, indicating that with a slightly looser cutoff, Uni-Dock consistently enriches true binders ~4× above random.
5. **Comparison needed**: This is the first screening power result in our CASF-2016 benchmark. Need to run Vina-CPU and other variants for comparison.

### Files
- Score files: experiments/Jun05_2354_CASF-2016_unidock_A_fast_screening/scores/
- Summary: experiments/Jun05_2354_CASF-2016_unidock_A_fast_screening/unidock_fast_screening_summary.txt
- Detail: experiments/Jun05_2354_CASF-2016_unidock_A_fast_screening/unidock_fast_screening_detail.txt 

---

## Experiment: QVina-W-CUDA Screening (CASF-2016)
- **Date:** 2026-06-06
- **Variant:** QuickVina-W-CUDA (Vina-GPU-2-1-W-CUDA)
- **Mode:** screening (57 targets × 285 coreset ligands = 16,245 dockings)
- **Thread:** 8192 | Box: 22.5³ Å | Timeout: 600s
- **Output:** Jun06_0415_CASF-2016_qvina_w_cuda_screening/
- **Runtime:** ~306 min total (avg ~460s/target)
- **Note:** SSH断线后通过resume机制续跑，17靶点SKIP

### Screening Power (CASF-2016 Forward Screening)
| Metric | EF @ 1% | EF @ 5% | EF @ 10% |
|--------|---------|---------|----------|
| Mean   | 7.55    | 4.27    | 3.09     |
| Median | 0.00    | 4.07    | 2.26     |


---

## Experiment: Uni-Dock-A Fast Screening (CASF-2016)
- **Date:** 2026-06-06
- **Variant:** Uni-Dock (gpu_batch mode, 1 receptor × 285 ligands per call)
- **Mode:** screening (57 targets × 285 coreset ligands = 16,245 dockings)
- **Search:** fast | Scoring: vina | num_modes: 9
- **Box:** 22.5³ Å
- **Output:** Jun06_1239_CASF-2016_unidock_A_fast_screening/
- **Runtime:** 17.7 min total (avg ~18.6s/target)

### Screening Power (CASF-2016 Forward Screening)
| Metric | EF @ 1% | EF @ 5% | EF @ 10% |
|--------|---------|---------|----------|
| Mean   | 7.90    | 4.30    | 3.07     |
| Median | 0.00    | 4.07    | 2.26     |

### Comparison: QVina-W-CUDA vs Uni-Dock-A
| Metric | QVina-W-CUDA | Uni-Dock-A | Speedup |
|--------|--------------|------------|---------|
| Total time | 306 min | **17.7 min** | **17.3×** |
| Per-target | 460s | **18.6s** | 24.7× |
| EF@1% mean | 7.55 | 7.90 | — |
| EF@5% mean | 4.27 | 4.30 | — |
| EF@10% mean | 3.09 | 3.07 | — |

**Key insight:** GPU batch mode (285 ligands/call) vs sequential (1 ligand/call) yields ~17× speedup with negligible accuracy loss. The bottleneck is GPU kernel launch overhead, not algorithm.


---

## Experiment: Uni-Dock-A Sequential Screening (CASF-2016)
- **Date:** 2026-06-06
- **Variant:** Uni-Dock (gpu_batch mode, 1 ligand per call, sequential)
- **Mode:** screening (57 targets × 285 coreset ligands = 16,245 dockings)
- **Search:** fast | Scoring: vina | num_modes: 9
- **Box:** 22.5³ Å
- **Output:** Jun06_1309_CASF-2016_unidock_A_fast_screening/
- **Runtime:** 598.1 min total (avg ~628s/target)

### Screening Power (CASF-2016 Forward Screening)
| Metric | EF @ 1% | EF @ 5% | EF @ 10% |
|--------|---------|---------|----------|
| Mean   | 8.83    | 3.56    | 2.81     |
| Median | 0.00    | 3.39    | 2.04     |

### Comparison: Uni-Dock Batch vs Sequential
| Metric | Batch (285/call) | Sequential (1/call) | Diff |
|--------|------------------|---------------------|------|
| Total time | 17.7 min | **598.1 min** | **33.8×** |
| Per-target | 18.6s | **628s** | 33.8× |
| EF@1% mean | 7.90 | 8.83 | +11.8% |
| EF@5% mean | 4.30 | 3.56 | -17.2% |
| EF@10% mean | 3.07 | 2.81 | -8.5% |

**Key insight:** Sequential mode is 33.8× slower but shows slightly better EF@1% (+11.8%) and worse EF@5%/10% (-8~17%). The difference likely stems from Uni-Dock's internal batching strategy: batch mode groups ligands by complexity (Small/Medium/Large/XL), potentially affecting BFGS convergence paths. Sequential mode gives each ligand full GPU resources, leading to more thorough optimization. Batch mode achieves 90%+ accuracy at 34× speedup, making it the practical choice for high-throughput screening.

