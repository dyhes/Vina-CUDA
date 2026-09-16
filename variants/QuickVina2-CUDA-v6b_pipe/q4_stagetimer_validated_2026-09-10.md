# stage_timer 验证结案 + Q4-2 — 2026-09-10

> 起因：per-block 调试数据与全局计数器"矛盾"引发的正确性怀疑。
> 结果：**stage_timer v1 的数值是正确的**——错的是我的单位换算。

## 一、验证过程（三步）

1. **per-block 侧写**：给前 64 块加独立 LS 累计器（先修了 per-TU 符号副本
   缺陷：dbg 数组定义进 header 导致 kernel2.cu 读到另一份未初始化副本）
2. **绝对墙钟插桩**：K2WALL —— block0 在 kernel2_update 内的驻留时间
3. **矛盾消解**（单位复核）：

```
K2WALL block0 = 16.9 ms（block0 在 kernel 内的全程驻留）
LS_TOTAL_NS   = 82.96e9 ns = 82.96 秒（总和，跨 8192 blocks）
  → per-block LS 均值 = 82.96s / 8192 = 10.1 ms ✓（与 per-block 实测 9.9-14.6ms 吻合）
  → 之前"0.19µs/次线搜索不可能"的验算是我的换算错误（把 ns 当 ms 除）
驻留比：block 驻留 16.9ms，kernel 墙钟 143ms
  → 并发驻留 ≈ 8192 × 16.9 / 143 ≈ 968 块（~1024，占 4090 容量的一半）
```

**v1 遗留的真实缺陷（已修）**：
- `stage_timer.cu` 不在 Makefile SRC_CUDA（从未链接；d_stage 靠 nvcc 跨 TU
  合并侥幸工作）→ 已删该文件，定义统一进 header（单包含域保证），Makefile 还原
- dbg 数组 per-TU 副本缺陷 → 已清理全部 v1 调试代码
- v2 结构保留：header 单一定义域 + K2WALL 墙钟桩（-DHOST_PROBE 下常驻）

## 二、验证后的 k2 构成（1pxn，数字终于全部自洽）

| 段 | per-block | 占 block 忙时 |
|---|---|---|
| **LS（线搜索）** | **10.1ms** | **60%** |
| eval_init | 0.66ms | 4% |
| twop（BFGS 双循环）| 0.71ms | 4% |
| mutate | 0.017ms | <1% |
| 其余（grid/intra/pot 计入 evals）| ~5.4ms | 32% |
| block 驻留合计 | 16.9ms | 100% |

- k2 墙钟 143ms = 8192 块 ÷ ~970 并发 × 16.9ms —— **受限的不是算力，
  是驻留并发度**（每块 32 线程 × 970 块 ≈ 31k 线程，4090 可容 ~150k；
  block 数固定 8192 是 thread=8192 的.restart 数）
- 1u1b 复算：LS 总 337.6s/8192 = 41.2ms/块 × 驻留比 ≈ k2 557ms ✓

## 三、对优化方向的最终指向

1. **LS 60% of block busy**：LS 内部的 E eval（grid 插值+pair）每次
   ~0.2-0.4µs × 232 evals/块。攻击点：LS 里 eval 的访存模式与 early-exit
   分歧（ncu 现在可跑）
2. **驻留并发度 ~970/8192**：若把 32 线程/块的 occupancy 提上去
   （寄存器 128 上限、shared 用量），更多块并发 → k2 墙钟直接缩短——
   这是"未用满 GPU"的真实机制，与 7 月 ncu 的 SM 10.9% 自洽
3. **stage_timer 现在可信**：后续 ncu 结果可用它做跨验证

## 四、状态

- Q3 分支新增 commit（stage_timer v2 + 清理），未动任何数值路径
- Makefile 恢复原样（stage_timer.cu 已删除，header 单源自洽）
- GPU 计数器权限：重启后 `RmProfilingAdminOnly=0` ✓，ncu 无需 sudo
