# v5 遗留项合入核查 + 与 grids 真实 dims 的叠加性 — 2026-09-10

> 问题①：v5 的优化是否全部在 Q2-7（v45 现线）里？
> 问题②：v5 的 grids HtoD 瘦身与"B. grids_cl 缩到真实 dims"能否叠加？

## 一、逐项核查（v5 三项 vs Q2-7 现状，diff 实证）

| v5 优化项 | Q2-7 现状 | 结论 |
|---|---|---|
| **Path B：m_cl 单份拷贝**（mg[0] + m_size 单份 malloc）| ✅ 已合（kernel2.cu `m_gpu = &mg[0]`，注释 "[Path B merged]"）| 已收 |
| **grids HtoD 瘦身**（memsetAsync + header 72B/grid）| ❌ **未合，已回退**：Q2-7 恢复了 v4 的整块 `memcpyAsync(grids_gpu, grids_ptr, 1.14GB)` | **遗漏 #1** |
| 写零 HtoD 删除（result_coords/result_gpus）| ✅ 已合（"[v7-GPU] skipped" 注释在）| 已收 |
| stage_timer 拆除 | ✅ 语义上已超越（探针改 -DHOST_PROBE 开关，默认关）| 已收 |

### 遗漏 #1 的实际代价（Q2-7 每 target 白做的两件事）

1. **1.14GB 垃圾 HtoD**：host `grids_ptr` 只填了 header，m_data 是未初始化堆内存，
   却整块 async 上传（GPU 端马上被 kernel1 全量覆盖）——纯粹搬运垃圾。
   async 不挡 host 时钟，但占 DMA 引擎 + staging buffer，与其它拷贝争带宽。
2. **v5 的 memsetAsync 也别照抄**：v5 每次 target 还 `cudaMemsetAsync(grids_gpu, 0, 1.14GB)`
   ——kernel1 反正会全量覆盖写，memset 同样多余（~1.1ms 设备时间 + 一次全显存扫过）。

### 正确形态（两者合取，v5 思路的完全体）

```
cudaMalloc(grids_gpu, REAL_SIZE)      // 见下节 B，1.14GB → ~123MB
// header: 一小块 cudaMemcpyAsync（atu/slope + 每 grid 72B 元数据）
// m_data: 完全不传 —— kernel1 现场生成，唯一写入者
// 不要 memset：kernel1 覆盖全部 m_data，零初始化是冗余
```
预期收益（相对 Q2-7 现状）：DMA 带宽释放 + 消除 1.14GB 垃圾拷贝；
配合 B 后 malloc 挂账 208ms → ~23ms、显存占用 -89%。

## 二、与"B. grids_cl 缩真实 dims"的叠加性：**可以，且天然互补**

两者作用在**同一条数据通路的正交两段**：

```
[host malloc 1.14GB] → [header 填充] → [HtoD 1.14GB 垃圾] → [GPU 1.14GB 驻留]
        ↑ B 管这段（分配大小）         ↑ v5 思路管这段（传什么）
```

- **B（缩分配）**：`grid_cl.m_data` 从静态 `MAX_DIM³×8` 改真实 dims 动态分配，
  sizeof(grids_cl) 1.14GB → ~123MB（22.5Å box ≈ 61³）。malloc 挂账 208→~23ms，
  显存 -1GB，kernel2 grid 工作集变小（cache 命中率↑，间接收益）
- **v5 瘦身（改传输）**：header-only HtoD。B 之后"垃圾体"从 1.14GB 缩到
  ~123MB，但**性质不变**——仍是拷贝无用数据；v5 思路把它归零
- **先后顺序**：先 B 后瘦身最顺（B 改了 struct 布局，瘦身代码要按新布局
  写 header 偏移；反着做要改两遍）。若走 v6 批合并（Q2-8.0），三件事一次
  设计到位：grids 是 per-process 共享的（同 receptor 多配体共用一份 grid），
  批模式下 B+瘦身只做一次、N 配体摊零

### 依赖与风险

- kernel1 索引从静态 stride（`m_data[MAX³×8]` 常量偏移）改 dims 步长 →
  kernel1/kernel2 所有 grid 寻址点都要过一遍（design-first，ncu 前后对比）
- kernel2 读 grid 的插值边界（m_init/factor）已经在 header 里，不受影响
- `grids_front` 语义（dim0!=0 的第一个 grid）保持

## 三、行动建议（挂入 Q2 路线）

1. **立即（半小时）**：把 v5 的 grids header-only HtoD 移植进 Q2-7
   （不要 memset）——这是 v5 遗留的纯收益，改动 <10 行
2. **Q2-8.0 合并时**：B（真实 dims 动态分配）与 v6 batch 一起 design-first，
   三者（B/瘦身/批）共享同一次 grid 通路重构
3. 论文工程层叙事顺势补一条：**"分配-传输-生成"三级解耦**
   （v5 修传输、B 修分配、kernel1 本就负责生成）
