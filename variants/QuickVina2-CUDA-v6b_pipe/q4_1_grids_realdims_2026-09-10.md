# Q4-1 grids 真实 dims：执行结果与诚实判定 — 2026-09-10

> 承接：`q3_breakdown_and_griddims` 中 B 方案的"kernel 层收益"假设。
> 分支：`QuickVina2-CUDA-Q3`（git commit fffc594 → 2409e19）

## 一、实测前提（dims probe）

kernel1 内一次性打印全部 17 个 grid 的真实 dims：

```
1pxn / 1u1b / 3ag9: 全部 61,61,61（CASF 统一 22.5Å box => dims 恒 61³）
```

**膨胀倍数**：MAX 128³ / 真实 61³ = **9.07×/grid**；
缓冲总量 1.14GB（MAX）vs **12.3MB**（真实）= **93×**。
kernel1 的 128³ launch 中 88% 线程被 `x>=m_i` guard 直接丢弃。

## 二、改动（极小，编译期常量）

1. `inc/kernel2.h`：SMALL_BOX 分支 `MAX_NUM_OF_GRID_M{I,J,K}` 128→64
   （覆盖 61³ + 余量；LARGE_BOX 分支不动）
2. `inc/cuda/kernel1.cu`：launch grid 从硬编码 128³ 改为跟随 MAX 宏
3. 索引零改动：kernel 侧 m_data 寻址**本来就用运行时真实 dims**
   （`m_i=61` stride；g_evaluate_update 从 header 取 m_i/j/k）——
   MAX 只影响分配上限与 assert，这正是当初"缩 dims"预期低风险的原因

## 三、结果（三例 smoke，vs Q3 基线）

| 指标 | Q3 | **Q4-1** | 判定 |
|---|---|---|---|
| GPU 显存（grid 缓冲）| 1.14GB | **136MB** | **-88%** ✅ |
| k1 | 6ms | 6ms | **不变**（launch/调度受限，非线程数受限；88% 线程早退几乎免费）|
| k2 | 139/557ms | 144/557ms | **不变**（cache 假设证伪，见下）|
| e2e_dock | 197/616ms | 198/617ms | 噪声内 |
| MODE1 | -8.9/-5.9/-6.7 | -8.7/-5.9/-6.7 | 噪声带 ✓ |
| 二进制体积 | -0.5MB | | |

**cache 假设证伪的原因**：k2 只**触碰**真实 61³ 数据（61³×8×4B=725KB/grid
×17 = 12MB），4090 L2=72MB——即便分配是 1.14GB，cache line 只在被访问时
分配，有效 footprint 本来就是 12MB。缩分配改变的是**虚拟/物理占用**，
不是访问局部性。

## 四、诚实判定

- **Q4-1 = 显存卫生 + 分配成本微降**（1.14GB→136MB；cudaMalloc 挂账
  2.5→<1ms），**不是运行时优化**
- "B 方案有 kernel 收益"的假设（q3_breakdown 报告）**正式证伪**
- 保留改动：显存 -1GB 对 screening/batch（Q4 后续）与多实例并发有真实价值，
  且零风险（索引本就按真实 dims）

## 五、k2 内部分布（stage_timer，指向下一目标）

| 靶点 | LS | mutate | eval_init | LS 占比 |
|---|---|---|---|---|
| 1pxn | 81ms | 0.14ms | 5.3ms | **94%** |
| 1u1b | 338ms | 0.22ms | 27.6ms | **92%** |

（LS_TOTAL 为各 thread 累计：8192 线程 × 线搜索总耗时，除以并行度后与
k2 墙钟一致。）

**下一阶段的真实目标 = 线搜索（LS）**：占 k2 92-94%，而 k2 占 e2e_dock
71-90%。LS 内部是 BFGS 回溯线搜索的逐 eval 串行链——这属于 kernel 层
主战场（v11 tile-LS 已证伪一条路，可挖的还有：早停策略、eval 精简、
shared 布局）。Q4 后续立项时先跑 ncu stall-reason 再定方向（按重规划纪律）。

## 六、谱系

- commit：fffc594（dims probe）→ 2409e19（Q4-1 本体）
- 未做 CERT（无性能里程碑）；_INDEX 中 Q3 仍为 LINE 头
- dims probe 代码保留（`grids_dims_printed` 一次性打印，可随时复测）
