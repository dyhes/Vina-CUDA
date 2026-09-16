# prep 底座解剖与优化空间 — 2026-09-10

> 问题：六档 smoke 实证的 host prep 底座 ~350ms 里到底装了什么？
> 方法：PREP 微探针（malloc/cudaMalloc/各 fill 环前后插 mark）逐段剥离，
> 配合两个独立 CUDA 微基准（mallocbench/evbench）交叉验证。

## 结论先行

~350ms 底座 = **3 段：entry-gap ~100ms + cudaMalloc(1.14GB) 后的首事件 208ms + struct 填充 ~40ms**。
**无算法性优化空间，只有"搬运/分配策略"空间；且前两段都是每进程一次，
batch 模式下天然摊薄为 0。**

## 逐段解剖（1pxn，微探针实测）

| 段 | 实测 | 内容 |
|---|---|---|
| entry-gap | 108ms | main_procedure 入口 → malloc(1.14GB) 之间：pre_gpu_setup 的 4MB ar_cl host malloc、mis 填充、7 条 stream 创建、needed 数组填充 |
| malloc(1.14GB) | **0.0ms** | glibc 惰性 mmap，不触页 |
| grids header fill（17 循环）| 0.4-2ms | 只填每 grid 的 m_init/factor/dims 头，**不填 m_data** |
| cudaMalloc(1.14GB) | **0.0-0.4ms** | VMM 惰性映射（standalone 复测 0.2-2ms 一致）|
| **GT_BLOCK_START 首事件** | **208ms** | cudaMalloc 后第一次 stream-0 事件操作触发 1.14GB 物理页挂账（见下方排查链）|
| ar fill（1024×1024 int 循环）| 0.4ms | 1M 次写，快 |
| pa fill（1.4MB）| 0.3ms | |
| pre fill（6.3MB LUT）| 0.8ms | |
| 各 HtoD（async）| 1.4/9.1ms | 页面锁存阶段部分串行化 |

### 208ms 的排查链（关键证据）

1. 微探针把 208ms 夹逼到 [cudaMalloc 返回后，memcpyAsync 启动前] 的
   `GT_BLOCK_START()`（= cudaEventCreate+Record，stream 0）
2. standalone `cudaMalloc(1.14GB)` = **0.2-2ms**（VMM 不挂物理页）
3. standalone `cudaEventCreate+Record` after 1.14GB malloc+7 streams = **0ms**
4. standalone 页面可锁 HtoD 1.14GB = **99ms**（一半：staging buffer 锁页 + DMA）
5. ⇒ 判定：**208ms = VMM 惰性页在第一次 stream-0 全局操作时的物理挂账**
   （约 1.14GB/2MB=570 个 2MB 页的映射+零化），发生在事件记录这个
   "第一个触碰设备状态的操作"上。它不是 cudaMalloc 也不是事件本身的固有成本。

## 优化空间评估（结合论文/代码）

### A. warmup 预挂账（改动最小，收益 208ms→warmup）
在 v7_warmup_cuda 里 cudaMalloc(1.14GB)+memset(触发挂账)+cudaFree。
**已试，无效**（v13：cudaMalloc 仍 208ms）——因为 cudaFree 后 VMM 释放物理页，
主流程重新 cudaMalloc 依旧惰性。**变体 A'**：warmup 里 malloc+挂账后**不 free**，
主流程改为取用该指针（一次性持久分配）。可行，但要处理 box 尺寸变化（缓冲
按 MAX 分配天然适配）。

### B. 缩小 grids_cl（算法层最有依据）
`grids_cl` 按 MAX dims（128³×8 flo×17 grids = 1.14GB）静态定型，但 kernel1
只用**真实 dims**（box 22.5Å → ~61³）。改为按真实 dims 分配：
1.14GB → **~123MB（-89%）**，物理挂账 208→~23ms，HtoD 同步缩小，
GPU 端 cache 命中率还会上升（kernel2 遍历 grid 的工作集变小）。
改动点：grid_cl 的 m_data 从静态数组改柔性指针 + kernel1/kernel2 的索引
改为 dims 步长读取。**中等工作量，触及 kernel1/kernel2 全部 grid 寻址**。

### C. entry-gap 100ms
构成：4MB ar_cl malloc + 7 streams + mis/needed 填充。streams 可复用
（一次创建进程级复用），malloc 可换 pool。收益上限 ~80ms，工程价值低。

### D. batch 模式天然摊薄（论文口径）
三段全部是**每进程一次**：runner `--ligand_directory` 批模式（已在路线图）
下 N 个配体只付 1 次 → per-ligand prep 从 350ms → ~40ms（仅 struct 填充）。
**这是零代码风险、收益最大的路径**，与 e2e handoff 的批量模式设计闭环。

### 优先级建议

1. **D（batch 摊薄）**：无风险，已在计划内，paper 里 screening 口径直接受益
2. **B（真实 dims 分配）**：-185ms + kernel2 cache 收益，作为下一个 kernel
   期工程；需 design-first（索引重写涉及 K1/K2 全部 grid 寻址）
3. **A'（warmup 持久分配）**：若不做 B，A' 可把 208ms 移出 e2e_dock
4. C：暂缓

## 论文表述要点

- prep 底座不是"计算"而是"分配/搬运策略"产物：1.14GB 静态定型 vs ~123MB
  真实需求，是 Vina-GPU 系"MAX 宏定型"工程债的又一例（与 ar_cl 1024×1024
  同根）
- redocking 口径 e2e_dock 0.384s 中 prep 350ms 占 91%？否——全量 median
  e2e_dock=0.384 已含 prep；按六档画像，prep 占比从 p100 的 12% 到
  p0 的 >100%（对账口径差异），全量 median 下 prep ≈ 91%×0.35/0.384
  需重新表述：**e2e_dock=384ms 中 prep ~340ms 占 89%**，GPU 只占 11%
  （median 靶点偏小）——这正是 e2e 口径揭示的、旧 rt 口径看不到的事实
