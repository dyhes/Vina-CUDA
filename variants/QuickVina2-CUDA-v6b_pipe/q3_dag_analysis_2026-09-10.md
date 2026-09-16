# Q3 执行 DAG 与并行性分析 — 2026-09-10

> 依据：Q3 源码逐行核对（main_procedure_cu.cpp / kernel1.cu / kernel2.cu）
> + 绝对时间戳探针实测（1pxn）。回答两问：DAG 长什么样、哪些段实际并行。

## 一、执行 DAG（单配体 redocking，Q3 现状）

```
[host 线程]                                [GPU stream0]              [GPU stream8]
─────────────────────────────────────────────────────────────────────────────────
parse receptor/ligand ─┐
precalculate (scoring) ┤
ctx warmup (B, 在e2e_dock外)
                       ▼
C entry-gap:
  ar_cl 4MB malloc ────┐
  7×streamCreate ──────┤
  mis/needed fill ─────┘
                       ▼
D 共享 prep（顺序 host 代码，H2D 各走不同 stream）:
  grids: cudaMalloc ──► header H2D(s1)          [GPU: 等]
  ar:    fill ─────► cudaMalloc ─► H2D(s2)
  pa:    fill ─────► cudaMalloc ─► H2D(s3)
  pre:   fill ─────► cudaMalloc ─► H2D(s4)
  gb: ──────────────► cudaMalloc ─► H2D(s5)
  mis/needed ───────► cudaMalloc ─► H2D(s6,s7)   ← needed 重复拷 2 次(代码冗余)
                       ▼
  streamSynchronize(s5)   ← 唯一显式 barrier（等 header H2D? 实际等 s5=gb）
                       ▼
  kernel_grid = kernel1 launch + cudaDeviceSynchronize   ← ★硬同步点①
  （kernel1: 128³×4³ 线程现场插值生成全部 grid m_data）
                       ▼                      [GPU: k1 6ms]
F per-ligand prep（host 单线程为主）:
  atoms/coords/minus_forces fill
  pairs fill (omp parallel for)
  ligand copy + store_node 树 walk (递归, 串行)
  m_gpu malloc+H2D; ric/randmaps/result malloc
                       ▼                      [GPU: idle……★空转窗口]
E launch 阶段:
  randmaps kernel launch (s8) ─────────────────────► [s8: 微秒级]
  ric kernel launch (s8) ──────────────────────────► [s8: 微秒级]
  kernel_monte:
    StageStats memset
    kernel2 launch (stream0) ──────────────────────► [s0: k2 139ms]
    cudaDeviceSynchronize                           ← ★硬同步点②（等 k2 完）
                       ▼                            [GPU: k2 139ms……]
  result malloc (host, 与 k2 并行!)
  DtoH results+coords ──► cl_to_vina ──► add_to_output (omp)
                       ▼
  （main 侧: refine 9 modes CPU + write PDBQT）
```

## 二、并行性盘点（实测与代码双重确认）

### 2.1 真并行的（三处）

| 并行对 | 机制 | 实测效果 |
|---|---|---|
| **① kernel1/k2 ‖ F per-ligand prep** | k1 launch 在 F 前、k2 launch 在 F 后但 F 的 144ms 期间 GPU 异步跑 | 1pxn: GPU 145ms 全藏进 F；**sync 只等 7ms**。1u1b: F=555≈k1+k2=564，同样互藏 |
| **② H2D 多 stream** | grids/ar/pa/pre/gb/mis/needed 各 s1-s7 | 理论可重叠；**实际全部 KB~MB 级、host 侧 fill 才是串行主体，多流无感** |
| **③ pairs fill 的 omp** | `#pragma omp parallel for` | pairs 数组填充本身；占 F 段小头 |

### 2.2 假并行/名义并行（值得写进论文的"设计冗余"）

- **7 条 stream 只服务了 ~12KB 的真正传输**（Q3-1 后），而 ar 4MB H2D 走 s2、
  pre 6.3MB 走 s4——由于都在 kernel1 之前被 `cudaDeviceSynchronize`（kernel1
  内部）拦截，stream 并行度对墙钟**零贡献**
- `streamSynchronize(s5)` 只等 gb（~36B）却写在 s5 sync 位置——语义上是
  "等全部前置 H2D"，但 s1-s4/s6-s7 没被显式等待，靠 kernel1 的 deviceSync
  兜底（幸运正确，非设计正确）
- needed 数组**重复 cudaMemcpyAsync 两次**（源码 L442-443 连续两行同参调用）
  ——纯 bug 级冗余

### 2.3 串行链（关键路径）

```
关键路径 = C(113) + D fills(~13) + [k1 6 ‖ F 144] + sync尾(7) + G(14)
         ≈ F 主导（1pxn 144ms）；1u1b 时 F=555 与 k1+k2=564 几乎等长，
           互为边界——再压 F 才会重新露出 GPU 瓶颈
```

## 三、DAG 揭示的优化序（更新版）

1. **F 段拆分重组**（最大收益）：树 walk 串行递归 → 迭代化/omp 任务并行；
   m_gpu HtoD 提前到与 pairs fill 重叠（现在 memcpy 同步接口！`cudaMemcpy`
   非 Async，是 F 段内的隐性串行点）
2. **消除假并行**：7 stream → 2 条（传输/计算）或直接默认流 + 一次
   `cudaMemcpyAsync` 批量小结构（合并 header+ar+pa+pre+mis 成一个 pinned
   打包缓冲，一次 H2D）；删 needed 重复拷贝
3. **硬同步点①重构**：kernel1 的 deviceSync 移除，改事件依赖
   （k1 完成事件 → kernel2 launch 前等待），让 k1 与 F 真重叠而非靠运气
4. **C 段**：ar_cl 4MB malloc 改池化/首次分配（进程级）
5. GPU 空转窗口（F 期间 ~140ms）在 v6-batch 下由其他配体的 k2 填满——
   这就是 batch 的 DAG 本质：**用 N-1 个配体的 GPU 工作填满单配体的 host 串行段**

## 四、论文可用表述

"单配体路径的依赖图呈『host 串行为主干、GPU 计算为旁路』结构：唯一强制
同步点在 kernel1 后与 kernel2 后。v5/v6 的工程层优化分别作用于削短旁路
（HtoD 瘦身）与用并行配体填充主干空窗（batch）；Q3 进一步消除了旁路中的
伪数据传输（grids 垃圾体）。"
