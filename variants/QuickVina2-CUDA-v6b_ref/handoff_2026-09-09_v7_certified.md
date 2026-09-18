# v7 E2E 认证报告 — CASF-2016 全量（2026-09-09 深夜）

> 实验：`experiments/Sep09_2154_CASF-2016_v4_7/`（285/285 OK，0 FAIL）
> 代码：v4.5 三修复 + Path B 合并（mg[0]）+ rand_maps hoist + E2E 计时基建

## 一、三口径对照（全量中位数）

| 口径 | v4 | v4.5 | **v4.7** | 说明 |
|---|---|---|---|---|
| rt（runner wall）| 1.520 | 1.440 | **1.430** | runner 秒表，含 fork/sudo |
| e2e_total（二进制全生命周期）| — | — | **1.390** | 含 ctx warmup 0.081 |
| **e2e_dock（主口径）** | — | — | **1.225** | warmup 后纯对接段 |
| k2（GPU）| 0.135 | 0.136 | **0.134** | kernel2 纯 GPU |
| gpu_all（GPU 时间线）| 0.413 | 0.201 | **0.358** | v4.7 的 HtoD 0.210 回来了（见下）|

## 二、首次拿到的拆分恒等式（全量中位数）

```
rt (1.430) = e2e_total (1.390) + runner fork/sudo (0.040)
e2e_total (1.390) = ctx warmup (0.081) + e2e_dock (1.225) + 残差 0.084
e2e_dock (1.225) = GPU 时间线 (0.358) + 二进制内 host (0.867)
```

**关键结论**：
- 之前归因给"runner fork ~0.8s"的判断**被推翻**——CASF 串行 runner 的真实开销
  只有 0.04s/配体（sudo bash -c 很便宜）。1.32→0.40s 的预测作废。
- 真正的 88% host 开销全部在**二进制内部**：0.867s/配体，构成（1pxn bisect）：
  grids/ar/pa/pre 填充 ~207ms（per-receptor）+ ric randomize ~210ms + rand_maps
  ~~214ms~~（已修至 1.2ms）+ HtoD 同步等待 ~210ms（归因未明）+ k1/k2/DtoH/杂项
- ctx warmup 实测 81ms（不是误判的 232ms——那是 probe 时间线交错的结果）

## 三、正确性

- 285/285 OK，0 FAIL
- aff vs v4：mean|d|=0.183，>0.5 共 26/285，max 1.30
- 噪声基线 v4-vs-v4：mean|d|=0.162，>0.5 共 23/285，max 1.20
- **v4.7 与 v4 的差异 ≈ v4 自身 run-to-run 噪声**（0.183 vs 0.162，26 vs 23）✓
- better/equal/worse = 87/103/95（无方向性偏移）

## 四、性能结论

- **k2 持平**（0.134 vs 0.135）：本轮改动全在 host 侧，符合预期
- **rt 1.430 vs v4 1.520 = -6%**：来自 rand_maps hoist（-210ms）部分被
  HtoD 0.210 的回增抵消（见下）
- **e2e_dock 1.225s 恒定**（p90=1.228）——k2 从 0.014 到 0.6 都是这个数，
  坐实 host-bound；这就是下一阶段的靶子（0.867s 二进制内 host）

## 五、HtoD 0.210 的悬案（下一轮 bisect 对象）

Path B 合并（m_gpus 单份）后真流量应 -30MB，但账本 HtoD 仍 0.210。
两个可能：① 0.210 主要不是 m_cl 拷贝，而是 pre_cl 6MB + ar_cl 4MB + ric 5MB +
coords 12MB 的首拷贝 + 多 stream 同步模式；② v5_memcpy 的 htod=0.000 是账本打点
覆盖差异（GT 块包裹不同）。需 ar/pre/ric 逐块 event 打点定案。

## 六、下一步优先级

1. ric randomize GPU 化（-210ms）：curand kernel 生成 rand_maps + ric
2. HtoD 0.210 bisect（逐 buffer event 对）→ 若为同步模式问题可再省 ~150ms
3. 二者落地后 e2e_dock 预期 1.225 → ~0.85s（-30%），redocking 全量 wall ~1.15s
4. screening 批模式（1 受体摊薄 207ms）另算，v6 内核已备

原始数据：`experiments/Sep09_2154_CASF-2016_v4_7/summary.csv`（13 列含 e2e）
