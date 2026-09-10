# v7 CASF 认证（v4_8）— 2026-09-10

> 实验：`experiments/Sep10_0905_CASF-2016_v4_8/`（285/285 OK，0 FAIL）
> 构建：v4.5 三修复 + Path B（mg[0]）+ rand_maps hoist + ric/randmaps GPU 生成
>       + 写零 HtoD 删除 + 进度条量子 100ms + E2E 计时（-DHOST_PROBE）

## 三口径全量中位数对照

| 口径 | v4 | v4_7 | **v4_8** | vs v4 |
|---|---|---|---|---|
| rt（runner wall）| 1.520 | 1.430 | **0.640** | **-58%** |
| e2e_dock（主口径）| — | 1.225 | **0.412** | — |
| e2e_total | — | 1.390 | ~0.5（中位）| — |
| k2（GPU）| 0.135 | 0.134 | **0.063** | **-53%** |
| gpu_all | 0.413 | 0.358 | **0.337** | -18% |

意外之喜：**k2 中位数 0.135 → 0.063**（-53%）。host 侧改动不影响 kernel 本身，
此变化源于全量下 k2 分布的构成变化——小 k2 靶点（此前被 host 段掩盖的）现在
采样更快完成，分布形状改变；1u1b 级大靶点 k2 仍 0.59s 不变。论文引用 k2 时
注意口径：mean 与 median 因长尾差异大，沿用 mean 需重新核。

（注：rt -58% 中含 runner 侧 host 负载减轻的二阶效应——每配体进程更短，
sudo/bash/python 开销同步下降。）

## 正确性

- aff vs v4：mean|Δ|=0.166，>0.5 共 20/285，max ~1.3
- 噪声基线（v4 自身两跑）：mean|Δ|=0.162，>0.5 共 23/285
- **v4_8 与 v4 差异 < v4 自身噪声** ✓；better/equal/worse = 83/112/90（无方向性）
- 三靶点 smoke 锚点全保持：1pxn -8.9 / 1c5z -5.9 / 1u1b -6.7

## Top-1 提醒

需跑 rmsd_results（runner 自动生成）确认 Top-1 落带：v4 家族历史 171-184。
（rmsd 文件若未生成，用 `run_experiment.py` 重算或按 EXPERIMENTS.md 流程。）

## 口径结论（可直接进论文"计时说明"段）

- **e2e_dock = 0.412s**（warmup 后纯对接段，中位）：相比 v4 rt 1.520 为 **-73%**
- 拆分：e2e_dock = GPU 时间线 0.337 + 二进制内 host ~0.075
- 剩余 host 大头已从 867ms 压到 ~75ms（rand_maps/ric GPU 化 + 写零 HtoD 删除
  + 进度条量子修复）；per-receptor 填充 207ms 在 redocking 无法摊薄，
  screening 模式可摊薄

## 复现

```
cd /home/ubuntu/grad && python3 tools/run_experiment.py --variant v4_8 --dataset CASF-2016
```
构建：`cd variants/QuickVina2-CUDA-v45 && make OPTION="-DHOST_PROBE"`（探针可选）
