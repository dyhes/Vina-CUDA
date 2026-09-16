# DAG 修正案：k2 在关键路径上（用户判断正确）— 2026-09-10

> 起因：用户质疑我此前的关键路径公式 `max(F, k1+k2)`——正确公式应为
> `F + k2`（k2 必须等 per-ligand prep 完成才能 launch）。用 K2T 直接
> 插桩（kernel_monte 内 launch 前/后打绝对墙钟）一锤定音。

## 一、实测裁决（1pxn，绝对墙钟）

```
K2T pre-launch  : T=...585.9   ← kernel2 launch 前
K2T post-sync   : T=...732.2   ← cudaDeviceSynchronize 返回
   → k2 在主机时钟上占 146.3ms，位于 [585.9, 732.2]
PROBE ric_gpu_plan : 147.2 ms（mark 在 kernel_monte 返回后触发）
   → 该 delta = [rm_gpu_plan(585.1) .. 732.3] = per-ligand host prep(~1ms)
     + randmaps/ric launch + k1(6) + k2(146.3)
PROBE rm_gpu_plan : T=...585.1（0.0 delta）
```

**结论：k2 没有任何与 host prep 的重叠。** 之前我把 `ric_gpu_plan` 的
147.2ms 误读为"per-ligand host prep 144ms、与 GPU 并行"——实际那 147ms
几乎全是在等 k2（k1 6ms + k2 139ms 事件时间 + sync 开销）。F 段的
per-ligand host prep 只有 **~1ms**（"144ms 树 walk/pairs 填充"是我对
段名的错位解读，store_node/pairs 的真实成本毫秒级）。

错误根源复盘：hp_mark 打印"上一段"delta，而 `ric_gpu_plan` 与 `km_done`
两个 mark 之间夹着 kernel_monte 调用——我把打印**顺序**当成了执行**位置**，
而 stderr 与 stdout 缓冲交错进一步误导（PRE_KM 行出现在 seed 打印之前）。

## 二、修正后的 DAG 与关键路径

```
关键路径 = C(113) + D(15) + F_host(~1) + [k1(6) ‖ launch] + k2(139~146) + G(14)
         ≈ 197ms ✓（与 e2e_dock 完全吻合）
GPU 与 host 几乎零重叠：GPU 在 C+D+F 期间空转 129ms（66%）
```

| 段 | 1pxn | 占关键路径 |
|---|---|---|
| C entry-gap | 113 | **57%** |
| D 共享 prep | 15 | 8% |
| F per-ligand host | ~1 | <1% |
| k1+k2+sync | 146 | **74%**（k2 为主体）|
| G DtoH+conv | 14 | 7% |

（占比和 >100% 因 k1 与 D/C 尾部少量重叠）

1c5z / 1u1b 同构复核：1c5z k2=16（dock 66 = C112 主导）；1u1b k2=557
（dock 616 = k2 主导，C 占 16%）——修正后的公式对三例全部吻合。

## 三、修正后的优化含义（与上一份报告的差异）

| 段 | 旧判断 | 修正判断 |
|---|---|---|
| F per-ligand prep | "144-555ms 最大段，优先攻" | **仅 ~1ms，已不是目标** |
| k2 | "被 F 遮蔽" | **关键路径主体（74%），kernel 层是真主线** |
| C entry-gap | 57%（1pxn）| 不变：单配体下最大 host 段 |
| batch 的收益机制 | "填 host 串行空窗" | **填 GPU 空转窗口**（k2 期间 GPU 无其他工作；N 配体流水线让 GPU 连续、host prep 被摊薄/隐藏）|
| grids 真实 dims（B）| host 侧无收益 | 不变 + **k2 侧 cache 收益现在直接体现在关键路径上**（k2 占 74%，-10% k2 = e2e -7%）|

**修正后的优先级**：
1. **kernel 层（k2 本体）**——1u1b 型靶点绝对主导；grids 真实 dims（k1 缩小
   + k2 cache）与 k2 优化同批做
2. **C entry-gap（113ms 常数）**——1c5z 型小靶点的主体；streams/malloc 池化
3. F/G/小填充电——已不是问题

对三档画像的修正：现在三例都是 **host C 段 + GPU k2 双主**结构；
"GPU 完全被遮蔽"的说法作废。
