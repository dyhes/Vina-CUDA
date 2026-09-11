# E2E_DOCK 时间构成与比例 — smoke 三例（v4_8b 默认构建，探针版 v9 smoke 日志）

> 说明：构成数据来自 9/10 上午的探针 smoke（进度条已移除、ric/randmaps GPU 化
> 已生效），E2E_DOCK 比认证全量 median (0.384s) 略有出入属正常抖动。
> 探针 mark 打印"上一段"时长；下表已按真实语义重排，三例分段合计与
> E2E_DOCK 对账差 <±56ms（探针自身开销 ~25×1-2ms + 时钟域差）。

## 三例分段表（单位 ms；% = 占该例 e2e_dock）

| 段 | 1pxn (473ms) | 1c5z (321ms) | 1u1b (951ms) | 段含义 |
|---|---|---|---|---|
| A parse+setup | 40 (8%) | 38 (12%) | 29 (3%) | PDBQT 解析 + scoring precalc |
| B ctx warmup | 96 | 86 | 94 | **不入 E2E_DOCK**，单列 e2e_warm |
| C entry-gap | 109 (23%) | 99 (31%) | 106 (11%) | main_procedure 内 pre_gpu_setup：4MB ar_cl 等 malloc + mis/streams 填充 |
| D receptor fill | 197 (42%) | 201 (63%) | 192 (20%) | grids/ar/pa/pre **host 循环填充**（omp 填充 grids）|
| E struct HtoD | 13 (3%) | 13 (4%) | 12 (1%) | 上述结构的上传 |
| F ric launch + kernel1/2 等待 | 147 (31%) | 25 (8%) | 592 (62%) | ric GPU 生成 + **kernel_monte 异步等待**（= 真实 GPU 对接时间）|
| G DtoH + cl_to_vina | 14 (3%) | 9 (3%) | 24 (3%) | 结果回收转换 |
| **prep 小计 (A+C+D+E)** | **359 (76%)** | **351 (109%)** | **338 (36%)** | host 侧准备 |
| **F+G（计算+回收）** | **161 (34%)** | **34 (11%)** | **616 (65%)** | GPU 干活 + 回收 |

（1c5z prep 超 100% 是因为其 e2e_dock 基数小：k2 只有 16ms，GPU 几乎秒完，
整段时间被 host prep 主导。）

## 关键读法

1. **host prep 是常数底座 ~350-360ms/靶点**，与配体大小基本无关
   （三例 A+C+D+E = 359/351/338）。它由受体决定，redocking 同一靶点
   时无法摊薄；screening 多配体共用一受体时可摊成一次性成本。
2. **F 段才是"对接本身"**：小靶点 1c5z 只有 25ms（k2=16ms），
   大靶点 1u1b 592ms（k2=584ms）——F 与 k2 几乎相等，证实 host 已不再
   遮蔽 GPU。
3. **三种画像**：
   - 1c5z 型（k2 << prep）：host 主导，GPU 利用率低，适合 batch/screening 摊薄
   - 1pxn 型（k2 ≈ prep）：过渡区
   - 1u1b 型（k2 >> prep）：GPU 主导，进一步提速只能回 kernel 优化主线
4. 全量 CASF median e2e_dock=384ms、k2=127ms → k2/rt ≈ 21%（v4 时代 9%）：
   GPU 占比翻倍是因为分母（host）被砍，不是 kernel 变快。

## 与 v4 的构成对比（同探针口径，1pxn）

| 段 | v4 | v4_8b | 变化 |
|---|---|---|---|
| rand_maps host 生成 | 214 | 0 | hoist，GPU 化 |
| ric host 生成 | 213 | ~1（launch）| GPU 化 |
| 进度条 join | ~500（隐形）| 0 | 线程移除 |
| prep A+C+D+E | ~350 | ~360 | 持平（未动）|
| kernel_monte+k2 | ~146 | ~147 | 持平 |
| e2e_dock | ~1225 | ~473 | **-61%** |
