# C entry-gap 结案：它是 warmup 的重复计入 — 2026-09-10

> 问题：C entry-gap（113ms）能否优化？
> 方法：在 C 段内部（cudaSetDevice / 事件创建 / 8 条 streamCreate）逐点插桩。

## 一、结论：**C 段不存在，此前是 warmup 的双重计入**

C 段内部逐点实测（1pxn，绝对时间轴）：

```
ctx_warmup 结束        T=681.2
  [v7_warmup_cuda 内部]            ← 91ms（已排除在 e2e_dock 外）
pre_gpu_setup mark     T=786.8    ← "main_procedure_cl_entry : 105.6 ms" 的 delta
                                    = warmup(91) + glue(~14.6)
cudaSetDevice          0.0 ms
事件创建 ×2            0.0 ms
streamCreate ×8        0.0 ms
rng / model 引用       0.0 ms
grids malloc 启动      T=787.0
```

即所谓 "C entry-gap 113ms" 实际 = **ctx_warmup(91ms，已在 e2e_dock 口径外)
+ main→main_procedure 之间的胶水(~14.6ms)**。C 内部所有操作全部 ≈0。
之前分解表把它列成"独立 host 大段"是**错误**——warmup 被算了两次。

## 二、修正后的关键路径（1pxn，e2e_dock=197ms，三例同构复核）

```
dock = glue(14.6) + D 共享prep(14) + per-ligand prep+launches(1)
     + k1(6) + k2(139) + sync(2) + post-malloc(7) + G(15)
     ≈ 197ms ✓
host-only 合计 ≈ 50ms（25%），k2 = 139ms（71%）
```

| 段 | ms | 占 dock |
|---|---|---|
| **k2（GPU）** | **139** | **71%** |
| glue（main→procedure 入口）| 14.6 | 7% |
| D 共享 prep（grids header/ar/pa/pre）| 14 | 7% |
| G DtoH+转换 | 15 | 8% |
| post-kernel mallocs | 7 | 4% |
| per-ligand prep | ~1 | <1% |

1c5z（dock 66：k2 16 + host ~50 ✓）、1u1b（dock 616：k2 557 = 90% ✓）同构。

## 三、对优化路线的最终修正

- ~~C entry-gap 优化~~：**不存在这个目标**（胶水 14.6ms 是 parse 后指针
  传递与函数入口，无可挖）
- **host 侧已无大段**：合计 ~50ms，任何池化/复用收益上限 ~20ms
- **k2 = 71-90% 的绝对主体**：单配体 redocking 的剩余优化空间全部在
  **kernel 层**——k2 本体（stage_timer 显示 LS 占 88%）+ grids 真实 dims
  （k1 缩小 + k2 cache 局部性）
- 这也再次印证三层框架：**工程层（v5/v6/Q3）已把 host 挖到只剩 50ms，
  下一阶段的收益只能来自 kernel 层**

## 四、记录

- 探针保留：C 段内部微桩仍在 Q3 源码（-DHOST_PROBE 下 ~0 开销）
- 三份 DAG/分解报告以本报告为准修正：q3_breakdown（C 段作废）、
  q3_dag_correction（结论不变：k2 主导）、本报告（C 段不存在）
