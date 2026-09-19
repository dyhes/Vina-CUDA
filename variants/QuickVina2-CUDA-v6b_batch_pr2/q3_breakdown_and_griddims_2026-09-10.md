# Q3 运行时间分解 v2 + grids 真实 dims 收益评估 — 2026-09-10

> 方法：探针升级为**绝对时间戳模式**（每 mark 附 T=绝对 ms），消除"段名-
> 段长"错位歧义。三例 1pxn/1c5z/1u1b 重跑（Q3 构建，-DHOST_PROBE）。
> 段语义以绝对时间轴为准；CASF 全量数字以 `experiments/Sep10_1358_CASF-2016_Q3` 为准。

## 一、Q3 单配体时间分解（绝对时间轴，1pxn 实测）

```
dock 起点（warmup 结束）T0
  C  entry-gap                    113.1 ms   ← ar_cl 4MB malloc + 7 streams + mis/needed
  D  共享 prep（grids header/       14.6 ms   ← cudaMalloc(grids) 2.5 + header H2D 0.1
     ar/pa/pre fill+HtoD)                     + ar/pa/pre fill 12
  F  per-ligand prep              143.9 ms   ← m_ligand/pairs 填充 + ligand 树 walk
     （ric_gpu_plan 段）                       + m_gpu malloc+HtoD   ★最大 host 段
  E  randmaps/ric launch +         7.0 ms   ← kernel1(6)+kernel2(139) 异步跑在
     kernel_monte(含 sync)                    F 段期间，sync 只等尾巴
  G  DtoH + cl_to_vina             14.3 ms
  e2e_dock ≈ 197ms（与 113+15+144+7+14-overlap 一致）
```

**核心发现（改变后续优化优先级）**：k1(6ms)+k2(139ms) 的 GPU 执行**完全被
F 段的 per-ligand host prep 遮蔽**——k1 在 F 开始前已 launch，k2 的 sync
只等了 7ms 尾巴。1c5z（F=18ms, k2=16）、1u1b（F=555ms, k2=557）同构：
**单配体 redocking 的瓶颈已 100% 是 host 侧 F+C 两段，GPU 完全空闲等待。**

## 二、三例分解表（ms，e2e_dock 为准）

| 段 | 1pxn (197) | 1c5z (66) | 1u1b (616) |
|---|---|---|---|
| C entry-gap | 113 (57%) | 112 (170%*) | 99 (16%) |
| D 共享 prep | 15 (8%) | 13 (20%) | 12 (2%) |
| F per-ligand prep | 144 (73%) | 18 (27%) | 555 (90%) |
| E sync 尾巴 | 7 | 7 | 7 |
| G DtoH+转换 | 14 (7%) | 9 (14%) | 23 (4%) |
| **host 合计（C+D+F+G）** | **286** | **152** | **689** |
| GPU k1+k2 | 145（被遮蔽）| 22（被遮蔽）| 564（近裸露）|

*1c5z 各段占比超 100% 是因为 GPU 与 host 重叠后 e2e_dock 基数小。

F 段与配体柔度强相关（1c5z 18ms → 1u1b 555ms：树 walk/pairs 随
atoms/torsions 增长），C 段与受体相关（~100ms 常数）。

## 三、grids 真实 dims（B 方案）的收益评估

Q3-1（header-only HtoD）已把 grids 的**传输**清零；B 方案针对剩余两处：

| 收益点 | 机制 | 量化 |
|---|---|---|
| cudaMalloc 挂账 | 1.14GB VMM 物理页 → 123MB | 2.5ms → <0.5ms（**~-2ms，可忽略**）|
| host malloc | 1.14GB glibc mmap | 0ms（惰性），不变 |
| 显存占用 | 1.14GB → ~123MB | **-1GB（真收益，但不是时间）** |
| kernel2 grid 读取 cache | m_data 工作集从 MAX³ 压到真实 dims³，插值查询局部性↑ | **潜在 k2 -5~15%（1u1b 级大靶点最明显），需 ncu 实测** |
| kernel1 填 grid | 仍填全 MAX³ 网格点（128³×17 = GPU 端 6ms 不变）| 若 kernel1 也改真实 dims → k1 6→~1ms |

**结论：Q3 之后，B 方案的"时间收益"只剩 kernel 侧（k1 缩小 + k2 cache），
host 侧收益≈0。它从"工程层优化"变成了"kernel 层优化"——归层变了。**

## 四、新的优先级（基于本分解）

1. **F per-ligand prep（1pxn 144ms / 1u1b 555ms）**——当前最大可优化段。
   构成：m_ligand/pairs 填充、ligand 树 store_node walk、m_gpu malloc+HtoD。
   手段：pairs/树结构 GPU 化或批量预分配（ar_cl 4MB→运行时实际大小）、
   CPU 侧 omp 并行填充。这是 F 段的主要成本，1u1b 级配体上收益最大。
2. **C entry-gap（~100ms 常数）**：ar_cl 4MB malloc + 7 stream 创建 +
   mis/needed——streams 复用 + malloc 池化，一次性收益，Q2-8 批模式天然摊薄
3. **kernel 层（k2 本体 + k1 grid 填充）**：与 B 方案合并考虑
   （B 现在的收益主要在这里：k1 6→1ms + k2 cache）
4. **B 单独做不再划算**（host 侧已无收益），并入 kernel 层工作

## 五、数据口径注意

- 探针绝对时间轴与 main.cpp 的 e2e 时钟存在 ~90ms 域差（1pxn：probe 段和
  293ms vs e2e_dock 197ms）——e2e_tres 打点在 main_procedure_cl 返回处，
  部分输出路径在时钟外。**分解表以 probe 绝对轴为准**（段间无歧义）；
  e2e_dock/rt 仍以 main.cpp/runner 为准（与 CASF 全量一致）
- 本分解均为单次运行；CASF 全量 median 见 Q3 认证报告
