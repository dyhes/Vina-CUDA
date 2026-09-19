# Q4-9R: ncu 复测与混合精度方向重规划 — 2026-09-11 晚

> 用户指令: 基于最新版本(Q4-8R 8613ee32)做 ncu, 按结果规划混合精度优化。
> 结果: ncu 推翻了"混合精度是下一步"的前提——真正热点是 RILC 局部内存
> 溢出。已实测修复(Q4-9R b67729d1), k2 -26%。

## 一、ncu 复测(Q4-8R, 1pxn, kernel2_update, 3-launch 中位)

### SpeedOfLight / Occupancy
| 指标 | 值 | 判读 |
|---|---|---|
| Compute (SM) | 28.5% | 非 compute-bound |
| **DRAM Throughput** | **70.3%** | ← 异常: Q4 代基线只有 8-27% |
| Memory Throughput | 685 GB/s | |
| L2 Hit | 99.2% | |
| Achieved/Theoretical Occ | 13.98/14.58% | 7 blocks/SM, shared 限 |
| Block Limit Shared/Regs | 7 / 16 | shared 先到顶 |
| No Eligible | 83.5% | 延迟暴露型 |

### Stall 分解(904-hazard 结构未变, 但屏障占比可忽略)
long_scoreboard 22.1 / wait 21.0 / short_scoreboard 15.2 / selected 9.5 /
barrier **0.5** / lg_throttle 1.2 / mio 1.4 / math_pipe 0.15

### 内存单元分解(决定性)
| 层 | % | |
|---|---|---|
| DRAM | 70.3% | 读仅 897MB, **写 44.79GB/launch** |
| L2 | 53.6% | |
| L1/tex | 28.4% | |

### 与指令数的矛盾 → 溢出定位
- global ST 指令仅 3.6M, 解释不了 44.8GB 写
- SASS: `kernel2_update` **STL=2071 条**
- ncu: `local_op_st` = **18.9 亿 sector**(60GB L1 级)/launch, 每线程
  ~225KB 写
- res-usage: **STACK 22,176 B/线程**
- FP64 pipe: **2.2%**(Q4-7 清理后精度域已归零级)

## 二、根因: RILC 的 lm_s/lm_y 溢出数组

`rilc_bfgs_update`(活路径)头部:
```c
flo lm_s[MAX_HESSIAN_MATRIX_SIZE] = {0};   // 1485 floats = 5.9KB
flo lm_y[MAX_HESSIAN_MATRIX_SIZE] = {0};   // 两个 = 11.9KB/线程
```
- `MAX_HESSIAN_MATRIX_SIZE = (6+48+1)(6+48+2)/2 = 1485` 是**三角矩阵**尺寸
- 但实际用法是**向量** `lm_s[0..n-1]`, n = 6+torsions(CASF ≤54)
- 即 **27 倍超配**; `= {0}` + 跨迭代存活 ⇒ 编译器落地 local memory
  (DRAM 后备), 每次 BFGS 调用全量写
- 8192 块 × 32 线程 × 每线程 ~72KB(两数组+其余栈) = 写洪流

## 三、修复(Q4-9R, 单变量, commit b67729d1)

```c
__shared__ flo lm_s[MAX_NUM_OF_DIMENSIONS_OF_MATIRX];   // 55 floats
__shared__ flo lm_y[MAX_NUM_OF_DIMENSIONS_OF_MATIRX];
if (threadNumInBlock == 0) { /* zero once */ }
```
- 右尺寸(55×4×2=440B/block, block 级本来 32 线程冗余算同一向量)
- shared 12468→12916B, 7 blocks/SM 不变(8 blocks 需 ≤12544, 差 372B)
- shared 零初始化只在 kernel 入口做一次(skip_row 同款)

### 实测
| 层面 | Q4-8R | Q4-9R | Δ |
|---|---|---|---|
| 1pxn local_op_st (sector/launch) | 18.9 亿 | 6.64 亿 | -65% |
| 1pxn DRAM 写 | 44.8 GB | 2.57 GB | **-94%** |
| 1pxn DRAM% | 70.3% | 4.3% | 归零级 |
| 1pxn k2 (ncu) | 66.0 ms | 59.1 ms | -10% |
| 1pxn k2 (runner) | — | — | |
| **CASF k2 median** | 0.068 | **0.050** | **-26%** |
| CASF e2e_dock | 0.119 | 0.101 | -15% |
| CASF rt | 0.340 | 0.330 | -3% |
| 3ag9(尾部) k2 | 1.66 | 1.604 | -3~4% |
| Top-1 | 178 | 177 | 带内 |
| aff vs Q4-8R | — | 0.147 对称(80/80) | 分布中性 ✓ |
| STL (kernel2_update) | 2071 | 519 | -75% |

注意: 局部溢出还剩 ~0.66G sectors(其余栈 9.9KB/线程, 主要是别处),
k2 没有跌到 0.035 的原因 = DRAM 不是唯一瓶颈, wait/scoreboard 延迟链
仍在(occupancy 14% 未动)。

### 补强证据(迟到扫参, Q4-8R 源码)
寄存器上限 128→160→192→255 的 STL 计数: 2071→2060→2039→2024(-2%,
平坦)——独立证实溢出由 STACK 22KB 绑定而非寄存器预算, Q4-4 旧结论
在新源码上复现成立; 排除"调 cap 消溢出"路线, Q4-9R 的 shared 化是
唯一正解。(该扫参与后续构建存在目录并发窗口, 已核验最终树未被扰动:
盘上二进制 md5 == /tmp/qv2_q49r 存档, Makefile=128, HEAD=b67729d1。)

## 四、混合精度方向重规划(基于 Q4-9R 的 ncu)

### 8-blocks/SM 快验结论(2026-09-11 深夜, 决定性)

**并发探针**(Q4-9R 二进制, 3 进程 pool: 1pxn+1u1b+3bv9, 诊断性质非 benchmark):
- serial: k2 = 0.055 / 0.423 / 0.432 (与单跑一致)
- 3-way 并行: 各 k2 退化到 ~2x (0.145/0.912/0.903), **总吞吐仅 1.30x/3 进程**
- 读法: 一个 kernel2 已占满 block 槽(8192 块), 新增 warp 的边际收益
  ≈ 0.15x/进程 ⇒ **8-blocks(+14% warps) 预期收益 ≤ 3-5% k2, 大概率 ~0**,
  且与 Q4-6a 的实测(occupancy +17% → k2 -2~3%)完全一致 ⇒ **8-blocks
  手术(需 3 处耦合改动)不做**, 这就是 LS 依赖链绑定的第三次独立确认

### v6-batch 预期修正(重要)
v6 的 screening 6.6× 认证在旧臃肿构建时代(k2 0.135, DRAM 70%, e2e 多肥肉)。
Q4-9R 后 GPU 侧 slack 已缩水(DRAM 4.3%/SM 28.5%/并发 1.30x), batch 的
**kernel 级共享收益变小**; 但 **进程级 entry-gap ~100ms×N**(C 段) 仍是
batch 免疫的常数 —— 合并前必须重新推导收益模型, 别拿 6.6× 当预期。

### 优先级重排(旧→新)
1. ~~FP16 grid 存储(M2)~~ → 已做, k2 无带宽收益(L2 99%)
2. ~~M4 half2 数学~~ → **降级为 P3**: SM 28.5%、FMA 5.7%、math_pipe
   throttle 0.15% —— 算术管线远未饱和, 换低精度数学无杠杆
3. ~~FP64 清理(M1 续)~~ → FP64 pipe 2.2%, 已无收益空间
4. **新 P1: 继续清溢出** — STACK 仍有 9.9KB/线程:
   - `lm_alpha/lm_ys/ys/yy/beta/cau` 等标量常驻(寄存器化)
   - 找出剩余 STL 519 条的来源(nvdisasm -g 行映射)
   - `visited_cl`/`ele_cl list` 若有线程级副本同样处理
5. **新 P2: 8 blocks/SM** — shared 12916B, 差 372B 到 12544×8 的门槛:
   - mis_cl 搬 `__constant__`(Q3 已有模板) 或 MAX_RIGID 再压
   - 或 barrier 语义不变下去掉 q4_bitmap_init 双字段合并
6. **P2 并行: wait/long_scoreboard 延迟链** = v6-batch(填 occupancy
   缺口 14%→理论值) — 与 Q4-9 合并(Q4-9 v6-batch 计划不变, 且现在
   batch 的收益预期更强: 单 ligand 内存流已干净)
7. TC/Tensor Core: 维持 REFUTED 判定(K=1, 无 GEMM 形), 除非 v6-batch
   后把 scoring 改造成 batched GEMM(那是 C 线的延伸, 不是现在)

### 论文叙事更新
- 三层框架(工程/内存/精度域) → 补第四层实例: **生命周期/尺寸错配**
  (array sized by wrong MAX + zero-init => local 溢出), ncu DRAM 写
  指令数对不上 → SASS STL 计数 → res-usage STACK 的定位链是方法学
- Q4 世代表加 Q4-8R/Q4-9R 行: k2 0.067 → 0.068 → **0.050**,
  rt 0.340 → 0.330, Top-1 177-179 带内
- 混合精度章节的结论: **在延迟暴露型内核上, 存储精度/数学精度的
  收益上限由瓶颈层级决定** —— 本项目实测: 精度域清理(-46%)只
  在 DRAM 写错配修复后才能看清后续方向

## 五、实验索引

- Sep11_2030_CASF-2016_Q3 (Q4-8R 门禁, k2 0.068/Top-1 178)
- Sep11_2109_CASF-2016_Q3 (Q4-9R 门禁, k2 0.050/Top-1 177)
- /tmp/q48r_full.ncu-rep (Q4-8R 全 section 3-launch)
- /tmp/qv2_q48r_ref (Q4-8R 参照二进制), /tmp/qv2_q49r (Q4-9R)
- commits: 8613ee32 (Q4-8R) → b67729d1 (Q4-9R)
