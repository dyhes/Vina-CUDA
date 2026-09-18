# Q5 ncu 瓶颈分析与提升方向 — 2026-09-12

> 基于 Q5(tag Q5 = 344d9f52, 64 线程)的 ncu 全套剖面, 对比 Q4-9R(32t)。
> 目标: 判定 64 线程后的新瓶颈与剩余提升方向。

## 一、Q5 剖面 vs Q4-9R 剖面(1pxn, kernel2_update)

| 指标 | Q4-9R (32t) | Q5 (64t) | 变化 |
|---|---|---|---|
| duration | 59.1ms | 57.8ms | -2.2% |
| **SM throughput** | 30.5% | **52.9%** | +73%! |
| L1 throughput | 30.9% | 30.5%(hit 57.5%) | ≈ |
| L2 throughput | 29.5% | 29.8% | ≈ |
| DRAM | 4.25% | 12.3% | ↑(local 流量) |
| issue_active | 18.4% | **30.7%** | +67% |
| Occupancy | 14.0%(7w) | **27.8%(14w)** | 2× |
| Eligible/sched | 0.42 | 0.42→ | |
| **barrier stall** | **0.5%** | **9.1%** | **×18** |
| long_sb | 15.5% | 14.0% | ↓ |
| short_sb | 15.8% | 11.5% | ↓ |
| wait | 23.6% | 19.7% | ↓ |
| local_op_st | 0.66G | **1.32G** | ×2(线程翻倍) |
| bank conflicts | 6.5M | 17.4M | ×2.7 |
| DRAM bytes | 2.57GB | 6.89GB | ×2.7 |

## 二、解读: 64 线程的收益与代价

**收益**: SM 30.5→52.9%(+73%), issue 18.4→30.7%(+67%)——warp 翻倍把
延迟暴露的空转发射槽填上了, 这正是预期的机制。

**代价(两个新瓶颈浮出)**:
1. **barrier 税 0.5%→9.1%(×18)**: 64t = 同一轨迹 2 warps, 每次
   eval/POT/gyration 的相位屏障都让 warp1 等 warp0(树走/归约是
   thread0 串行段), 小配体 E 循环短, 屏障次数/有效功比例高
2. **local 溢出流量 ×2**: 线程翻倍, 每线程剩余溢出(visited_cl 4.3KB
   等)不变 → local_op_st 0.66G→1.32G, DRAM bytes 2.6→6.9GB

**净效果**: SM +73% 只换来 k2 -2.2%(1pxn)——增量被 barrier 税和
LSU 压力吃掉。但在**大配体上完全不同**:

## 三、大配体端(3ag9, n=91): Q5 的真实收益区

| 指标 | 3ag9 @64t |
|---|---|
| SM / L1 / L2 | 44.4 / 44.2 / 48.3% |
| issue_active | 40.1% |
| **barrier** | **1.6%**(大 E 循环摊薄) |
| long_sb / wait | 20.6 / 25.0 |

大 E 循环(91 原子 + ~700 pairs)摊薄屏障后, 2 warps 的并行收益
完全兑现(全量 -17%)。**barrier 税与配体尺寸成反比**。

## 四、瓶颈判定与提升方向(更新后)

### 瓶颈迁移图
```
Q4-9R: 延迟暴露型(warp 饥饿: 1.75/sched, No Eligible 83.5%)
Q5:    双瓶颈转移:
       小配体 → barrier 税(9.1%) + LSU(local 1.32G)
       大配体 → 正向 throughput 转型中(issue 40%, 管线 44-48%)
```

### 方向 A: barrier 税消除(小/中配体, 预期 k2 -5~10%)
- 相位屏障让 warp1 全程陪跑 thread0 的串行段(树走/归约)
- 方案: **warp 专业化**——warp0 跑 GRID+INTRA gather, warp1 跑
  POT 树走与标量归约, 中间用生产者-消费者 shared 队列替代全块
  屏障(文献: Syncopate 2601.20595 / FLUX 2406.06868 的 chunk 级
  overlap 思路); 或把串行段改为 warp 协作(树走按层双 warp 分摊)
- 风险: 高(改动 eval 数据流), 必须 design-first

### 方向 B: 剩余溢出减半(全尺寸段, 预期 k2 -3~6%)
- local_op_st 1.32G × ~225KB/线程: visited_cl(4.3KB 栈)是最大项
- 方案: visited_cl 的 list_cl 指针数组改 shared 索引数组(+2.2KB
  shared → 仍 7 blocks? 12684+2200=14884 > 14628 差 256B, 需配对
  shave); 或 visited 深度限制(cut len)
- 风险: 中(数据结构改动, 需语义核对)

### 方向 C: bank 冲突整形(≈1%, 低优先)
- conflicts 17.4M vs 19.3 亿 wavefronts ≈ 0.9%; q4_skip_row/
  m_coords 访问模式微调即可, 收益太小

### 方向 D: 大配体专项(如果筛选负载以大配体为主)
- 3ag9 已在 44-48% 管线, 继续提 occupancy(8 blocks: 再削 900B)
  或 ILP 展开(等待 25% 中可挖)→ 有 -5~10% 空间
- 这是唯一"继续吃 64t 红利"的路线(配对 shave 已在 S4a-1 清单)

### 已关闭方向(维持)
- 精度域(FP64 2.2%)、TC(K=1)、8-blocks@32t(MPS 证伪)、
  early-term(已覆盖)、S2/S3(S1 平坦证伪)

## 五、执行建议(优先序)

1. **B: 剩余溢出减半** — 最低风险、全尺寸受益、shared 账目清晰
   (visited_cl 需 -256B 配对 shave, 可用 MAX_RIGID 38 已省的部分)
2. **D: 大配体专项** — 若论文/应用侧关心 screening 尾部成本
3. **A: warp 专业化** — 只有当 B+D 后仍需 <40ms k2 时才立项
   (结构性重构, 1-2 会话, 预期 -5~10%)
4. **小配体不再单独优化** — barrier 税是其结构成本, 摊薄靠大配体
   路线反向受益有限; 论文口径: 分复杂度报告


## 十七、方向 B 执行后修正(2026-09-12 下午): 目标是死代码, B 取消

按计划执行 B(visited_cl 溢出减半)前的代码核对推翻了前提:

1. **interesting() 的 dist/notPicked 分支是运行期死代码**:
   visited->full 需要 `10 × n_variable`(≈220+)次 add() 才置位, 而每
   轨迹只有 ~12 个 episode(97831/8192)→ full 永远为 false →
   interesting() 永远在 `!full` 提前返回 true → dist[540]/notPicked[540]
   从未执行。溢出大头在别处。

2. **剩余 1.32G local_st 的真实构成 = 分布式临时量**: 每次 eval 的
   x/g 候选拷贝(candidate = tmp / tmp = candidate)、LS 回溯状态、
   编译器寄存器溢出——分散在数十个站点, 没有单点可修。消除 = 重构
   LS 状态机(S1 同类教训: latency-bound 内核删 local 不缩短关键路径)。

3. **顺带发现一个潜在 bug**: `add()` 把**栈局部变量** `ele_cl e` 的
   指针存进 `visited->list_cl[index]`——add 返回即悬空! 当前无害仅因
   full 永不饱和(悬空读取不会发生); 若未来调大 episode 数/轨迹深度,
   会变成真正的悬空读。**记录为已知潜在地雷, 不在本次修改。**

### B 判决: 取消(目标死代码 + 收益 S1 类 ≈ 0)

## 十八、Q5 之后的方向终版(全部收敛)

| 方向 | 状态 | 一句话 |
|---|---|---|
| kernel 内微优化(S1/S2/S3/B) | **全部证伪/取消** | latency-bound 不为指令减量付费 |
| 64t(已落地) | ✓ | 大配体 -15~20%, 小配体屏障税 |
| 8 blocks @64t | 可选 | 需 ~900B shave, 预期 +5~10% |
| early-term | 关账 | 机制已存在(A2/Lewis-Overton) |
| v6-batch | 待解冻 | 收益=entry-gap 摊销, 非内核共享 |
| 确定性根治 | 待立项 | racecheck 环境受限 |
| 论文 | **下一步** | 素材完整: 四层框架+全世代表+否定结论边界证明 |

**单配体 redocking 的 kernel 与 host 侧均已到实际地板。下一步 = 论文
写作或 v6-batch(按修正后收益模型)。**

## 十九、Q4-5 遗产与 Q4-7C 价值评估(2026-09-12, 实测)

### Q4-7C skip-bitmap: 仍有 0~2% 价值, 保留
A/B 实测(64t, 同 seed/盒, 禁用 skip 检查 vs 启用):
| target | bitmap ON | OFF | bitmap 贡献 |
|---|---|---|---|
| 1pxn | 0.055 | 0.054 | ~1.8% |
| 3ag9 | 1.346 | 1.348 | ~0.1% |
| 1c5z | 0.010 | 0.010 | 0 |

大配体的 pairs 在精修期大多处于 cutoff 内, bitmap 很少跳过 → 收益
随配体增大而衰减(引入时 -0~6% 是 32t 单 warp 稀缺时代的数字)。
成本仅 256B shared + 每 pair 一次 atomic。**判定: 保留(不是杠杆,
但是廉价正收益)。**

### Q4-5 遗产(POT tmp 数组): 结构性冗余可消除, 是 8-blocks 的最便宜钥匙
phase1(原子分步并行累加)写 `*_tmp` 共享数组; phase2(S1 后 thread0
串行树走)把它们读入**独立的** position/orientation_derivative 数组再
原地累加。但 DFS 序(子节点 index > 父节点, 现有代码已依赖此序)保证
自底向上时每个节点先于其父被终算 → **可以就地直接在 tmp 上累加**,
两个独立 deriv 数组(456×2=912B shared)纯冗余。

- 单独做: 性能 ≈0(S1 类), 但 shared 12684→11772B
- **作为 8-blocks@64t 的钥匙**: 恰好覆盖所需的 -908B(11776 门槛),
  使 8 块 × 2 warps = 16 warps/SM(+14%)成立——这是全表最便宜的
  一块拼图, 且行为零变化(同序同值)

### 建议组合(Q4-12 续, 半会话级工作量)
tmp 合并(912B) ⇒ 8 blocks ⇒ 16 warps/SM ⇒ 大配体再 -3~6%
全部在既有审计框架内(无新 sync 语义, 只删数组与拷贝)。
