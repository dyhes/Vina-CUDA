# Q5R-SERIAL: 冗余隐性串行 → 显式 Thread0 串行改造 — 2026-09-12

> 用户指令: 先把所有的冗余隐性串行改造为显式 Thread0 串行, 消除额外的工作负载。
> 基线: Q5 (cd10ca24)。改动仅 `inc/cuda/quasi_newton.cuh`, mutate_conf 未动。

## 一、改造内容(4 处, 全部 [Q5R-SERIAL] 注释标记)

### P1. set_update 刚体树传播 (最大冗余源)
- **原状**: 全部 64 线程冗余执行同一段串行 DFS(节点四元数链 + 128 原子
  local_to_lab), 写相同的 shared 值。每次 eval 调用(含每次 LS trial)。
- **改法**: 函数体内 `if (threadIdx.x==0 && threadIdx.y==0) {...}` 门控
  + 末尾 `__syncthreads()` 发布 m_coords/rigid。
- **同步正确性**: 原子级证据——eval 路径中 ig/pairs/POT 对 m_coords 的
  读取全部发生在 set_update 尾随 sync 之后; LS trial 的 x 更新(P4)由
  thread0 执行, set_update 也是 thread0 读 x → 程序序天然满足。

### P2. RILC init: d=-g 与 to_norm
- `get_to_minus_update`: 全线程冗余 → thread0 + sync。
- `to_norm` (~55 次串行迭代): 全线程冗余 → thread0 算入 `__shared__ flo
  d_norm` + sync, 块内读取。

### P3. RILC 低秩更新块 (k2 主循环内)
- **原状**: lm_s/lm_y 填充、ys/yy、cau、two-loop 递归(~275 次串行迭代
  × 5 段)全部线程冗余执行, 每 BFGS 步一次。
- **改法**: 整块 `if (threadNumInBlock == 0) {...}` + 尾随 sync(发布 d
  供下一轮 LS 全线程读取)。
- **注意**: ys>cau 分支判定输入为 thread0 局部值, 分支均匀(块内一致),
  门控后语义不变。

### P4. LS trial 构象增量
- `*x = *xp; output_type_cl_increment(...)` 全线程冗余 → thread0 门控。
- 消费者: set_update(thread0 读 x)+ m_coords(sync 发布)→ 安全。

### 未动(记录理由)
- mutate_conf quaternion_increment/torsion 写: ~20 FLOP×63 线程, 且
  mutate 有多早退分支, 加 sync 的分支风险大于收益。
- 共享 struct 拷贝(candidate=tmp, xp=*x 等): 零成本同值写, 门控反而
  增加同步点。

## 二、验证结果

### 冒烟(锚点, seed 42)
| target | k2 | aff | vs Q5 |
|---|---|---|---|
| 1pxn | 0.051 | -9.0 | 同(-9.0, k2 0.050-052)|
| 1c5z | 0.010 | -5.9 | 同(-5.9)|
| 3ag9 | 1.243-1.278 (3 seeds) | -5.5/-5.3/-5.5 | 带内(-5.6/-5.0)|

资源: SHARED 12684 不变, REG 128, LOCAL 0 — 8 blocks/SM 不受影响。

### 2×CASF 全量门禁(285/285, 无 FAIL)

| 指标 | Q5-cert-A | Q5-cert-B | **Q5R-s42** | **Q5R-s43** |
|---|---|---|---|---|
| rt 中位 | 0.320 | 0.320 | **0.300** | **0.300** |
| k2 中位 | 0.052 | 0.052 | **0.049** | **0.048** |
| Top-1 ≤2Å (all-atom) | 180 | 178 | 170 | 177 |
| Top-1 ≤2Å (heavy) | 179 | 177 | 173 | 177 |
| aff 对称(中位 Δ) | — (cert 噪声 0.100) | | **0.100** | — |

### 解读
1. **性能**: rt -6.3% (0.320→0.300), k2 -6~-8% (0.052→0.048-049)。
   超出预期(预期 0~-3%)——冗余负载消除在小配体上释放了 issue 槽。
2. **aff 完全对称**: 中位 Δ=0.100 = Q5 自身 cert 噪声 → 数值语义等价。
3. **Top-1**: s43=177 在 Q5 带内 {178,180} 边缘; s42=170 低于带。
   Top-1 单跑极差 9-10 是已记录特性(文献+cert 历史); s42/s43 两跑
   中位 = 173.5, Q5 两跑中位 = 179 — 差 5.5, 需谨慎解读:
   - aff 对称性(0.100=噪声)说明搜索未破坏;
   - Top-1 分布的 target 级重合度需要逐 target 对比才能定性
     (见 §三 后续)。
4. **同步安全**: 285×2 跑零 FAIL 零 CUDA error, 三锚点位模式与 Q5
   一致 — 门控未引入竞态。

## 三、结论与后续

**Q5R-SERIAL 是一个净改进**: 负载卫生(消除 63/64 冗余执行)+ 实测
rt/k2 双降 6-8% + aff 严格对称。建议:
1. Top-1 s42=170 需第三个 seed 确认(或逐 target 对比 Q5 cert 的
   fail 集合, 判断是噪声还是真实位移);
2. 若第三跑 Top-1 ≥175, 可将 Q5R-SERIAL 提交为新主线版本
   (Q5.1), 走 CERT 快照流程;
3. 改动文件: `inc/cuda/quasi_newton.cuh` 仅此一个, 4 处
   [Q5R-SERIAL] 标记, 可整体 revert。

## 四、构建复现
```
ssh ai_cloud
cd ~/grad/variants/QuickVina2-CUDA-Q3
export PATH=/usr/local/cuda-11.8/bin:$PATH
make clean && make OPTION="-DHOST_PROBE" -j$(nproc)
# 门禁: ulimit -s 8192; runner --variant Q3 --dataset CASF-2016 --thread 8192
```
