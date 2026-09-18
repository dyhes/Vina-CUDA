# Q4-10 规划基线: 单 ligand redocking 穷尽性审计 — 2026-09-11 深夜

> 用户问题: 单 ligand redocking 是否穷尽? 若没有, 暂不考虑 v6-batch。
> 结论: **没有穷尽 — kernel 层确实接近地板, 但 rt 330ms 里 host 侧占 85%,
> 其中 ctx warmup ~85ms + 输出/teardown ~83ms 是两块未被任何世代动过的肥肉。**

## 一、Q4-9R 全量 rt=330ms 完整解剖 (medians, 全口径对账)

| 段 | ms | 占 rt | 内容 | 状态 |
|---|---|---|---|---|
| runner tail | 52 | 16% | subprocess/sudo/exit, runner 侧 | **未优化过** |
| ctx warmup (C 内) | ~84 | 25% | v7_warmup_cuda: 1.14GB malloc+free 探针 + 256MB memset + module load + ctx init | **未优化过** |
| dock | 101 | 31% | k1 6 + **k2 50** + host glue 45 | k2 已修 4 代; glue 恒定 |
| post-dock (G) | ~83 | 25% | 输出 PDBQT 写 + log flush + teardown; corr(k2)=0.46 部分复杂度相关 | **未优化过** |
| 合计 | 330 | | | |

host 侧合计 = 330 - 50(k2) = **280ms = 85% of rt**。
dock 内 glue 45ms 恒定 (min 39/max 63), 与 k2 无关 — 四代优化从没动过。

## 二、kernel 层 (k2) — 接近地板, 判据三条

1. DRAM 4.3% / L1 28% / SM 28.5% — 无单点饱和
2. 并发探针: 3 进程仅 1.30x → 加 warp 边际 ~0.15x/进程
3. LS 串行依赖链: Q4-6a (+occ → -2~3%) / Q4-4 (regs 扫平) / 并发探针, 三次独立确认

剩余 k2 菜单 (全部 ≤5%, 共计乐观 ~10%):
- interesting() 32 线程冗余计算 → thread0+广播 (-3~5%)
- 剩余 STL 519 行级定位 (≤2%)
- 8-blocks: 枪毙 (并发探针)
- M4 half2 / TC: 冻结 (SM 28.5%, K=1 无 GEMM 形)

## 三、host 层 — 未开发的三大块 (本轮审计的真正发现)

### H1. ctx warmup ~85ms (25% rt) — 设计冗余, 最大的单项
v7_warmup_cuda 干了三件事:
a. `cudaMalloc(1.14GB)+cudaFree` 探针 — **Q4-1 之后已无意义** (真实 grids
   只有 136MB, 这是在给一个不存在的尺寸热身页表!)
b. 256MB memset — 给 L2/页表热身, 代价 ~20-40ms
c. module load + ctx init + 4KB memset probe
动作: (a) 直接删 (Q4-1 后僵尸); (b) 缩到 136MB 或删 (实验决定);
(c) 保留 (真实成本)。预期: -40~60ms (-12~18% rt)。风险低 (探针语义)。

### H2. post-dock 输出/teardown ~83ms (25% rt) — 从未测量过
构成待插桩: PDBQT 文本写 (输出模型 N 个 pose)、gpu_runtime.log ofstreams、
output_type Cl→host 转换。corr(k2)=0.46 ⇒ 部分随 pose 数/原子数走。
动作: 先 HOST_PROBE 插桩 (write_models / log flush / teardown 三段),
再决定: 缓冲一次性写 / 并行格式化 / 砍冗余 log。
预期: 未知 → 先测量 (prep_floor_anatomy 的 host_probe 阶梯法)。

### H3. dock 内 glue 45ms (14% rt) — 恒定, 含 D2H 7ms + launch 排队
Q3 解剖遗留: C 段(113) 已被 Q4 削到 warm 84; D 段 mallocs 现在在哪里
待查。动作: PREP 段插桩复查后定。优先级低于 H1/H2。

### H4. runner tail 52ms (16% rt) — 工具链, 不是被测程序
sudo+python subprocess。论文口径用 binary e2e_total (276ms) 可避开;
若 CASF 逐目标 rt 必须保留, 可用 binary 侧 rt 输出替代 runner 计时。
动作: 口径决策 (论文), 非优化项。

## 四、执行顺序 (v6-batch 冻结前提下)

1. **Q4-10a: 删 1.14GB malloc 探针 + 重审 256MB memset** (单变量, 一次
   构建, 预期 -12~18% rt) — 风险: 几乎零 (探针是 HOST_PROBE 内)
2. **Q4-10b: post-dock 插桩测量** (只加探针不改行为) → 定 H2 内容
3. **Q4-10c: 按 b 的结果拆 output 写路径** (大概率 -5~15% rt)
4. 之后再看: H3 glue / k2 长尾 (interesting 广播) 是否还值得
5. 每步门禁: 全量 CASF (aff 对称性 + Top-1 带内) + 冒烟锚点

预期路线图: 330 → 285 (a) → ~255-270 (c) — 单 ligand redocking 还有
~20% rt 可拿, 全在 host 侧, 不碰 kernel。

## 五、Q4-10a 实测结果(当晚 23:30) — 预测修正

诚实记录: 实测收益 **远小于预测**。
- 预测: 删两个僵尸探针 -40~60ms (推理: 它们占了 warm 84ms 的大头)
- 实测: 1c5z fresh warm 97→83ms (**只 -14ms**); CASF warm 84→78ms,
  rt 330→320ms (**-3% rt, 非预测的 -12~18%**)
- 原因: 4090 的 VMM 物理化在纯 malloc+free (无真实写入) 下本来就快
  (prep_floor_anatomy 的 208ms 是 Q4-1 时代对 1.14GB **真实写**的成本);
  剩余 83ms 是 **真实的 ctx init + module load**, 属环境成本, 不可再压
  (warmup kernel launch 触发 fatbin 装载是必要的)
- 判定: 探针代码仍是该删的(僵尸+误导), 但 H1 的可优化余量已归零。
  **ctx warmup 83ms 改判为地板。**
- 门禁: 285/285, k2 0.050 不变, Top-1 174(带内), aff vs Q4-9R 0.179
  对称(95/85) — 通过
- commit 394bed75; 实验 Sep11_2327_CASF-2016_Q3

方法论教训(再次): 预测收益必须先跑微基准(malloc-free 一次的成本)再
报数字——"僵尸"不等于"贵"。

## 六、Q4-10b 实测结果(深夜) — "83ms post-dock" 被解剖证伪

插桩(PD_REFINE/PD_WRITE, HOST_PROBE-only)后三分量实测:

| target | PD_REFINE | PD_WRITE | 备注 |
|---|---|---|---|
| 1c5z | 3ms | ~0 | |
| 1pxn | 36ms | ~0 | |
| 3ag9 | **144ms** | ~0 | 随复杂度缩放, corr(k2)=0.46 的来源 |

**判决: "83ms post-dock" 不是单一可优化块**, 实际由:
1. **PD_REFINE = CPU 端 refine_structure 对 top modes 的 BFGS 精修**
   (3-144ms, 随复杂度缩放) —— 这是 Vina 语义本体, 砍它=砍精度,
   且 CASF Top-1 正依赖 refined affinity; **不是免费收益, 是语义决策**
2. parse/reading ~20ms (在 t0..warm 段内, 之前被误计入 post-dock)
3. log/teardown ~20ms 常数
4. **PD_WRITE ≈ 0** (ofile 缓冲写) → **Q4-10c(拆输出写路径)取消** —
   写根本不花钱

### Q4-10b 全量门禁(两跑, 插桩行为中性验证)
- run1: rt 0.32 / k2 0.049 / Top-1 165; run2: rt 0.31 / k2 0.051 / Top-1 174
- 两跑中位 169.5, 极差 9 < 家族带宽 13-16 → 165 是种子实现噪声
  (runner 默认 auto_seed, 每跑不同实现; 插桩不改行为)
- aff vs Q4-9R: 0.174 / 0.171 对称 → 通过

## 七、① 的算术复核(深夜) — interesting() 亦证伪, 菜单清空

动手前算术复核推翻了我给的 -3~5% 估计:
- interesting() 每个 BFGS 只调一次(非每 eval); dist2_cl ≈50 FLOP ×
  len(典型 ~100) ≈ 5K FLOP/线程, ×32 冗余 = 160K FLOP/block/BFGS
- 对照: 一次 eval ≈ 50K FLOP(block 并行), 每 BFGS ~300 次 eval ≈ 15M
- **冗余占比 ≈ 1%**, 非 3-5%
- 且修复代价: dist(2160B)+notPicked(540B) 要进 shared → shared
  12916+2700=15616B > 7-block 预算 14628 → **掉到 6 blocks/SM(-14% occ)**
- 延迟暴露型内核里 warp 是最稀缺资源, -14% occupancy 换 ~1% 冗余 =
  **净负收益**

**判定: 不建实验, 直接证伪。** ncu 菜单至此清空——redocking 关账成立,
且比"留一个 3-5% 尾巴"更干净。

## 八、ncu 菜单终版(全部清空或冻结)

| 方向 | 判定 | 依据 |
|---|---|---|
| 管线百分比类(精度/数学/带宽) | 无对象 | 全部单元 ≤31% |
| 8-blocks/SM | 枪毙 | 并发探针 1.30x/3proc |
| interesting() 冗余 | 证伪 | 算术 ~1% + occ 代价 |
| STL 519 残余 | ≤1-2% | 递减 |
| 128 线程块 | 审计门禁 | v4.5b 6 处 sync 隐患未清 |
| PD_REFINE | 语义 | 砍=砍精度 |
| batch | 冻结 | 用户指令 |

**redocking 唯一剩余 >10% 的路 = 算法级重构(eval 链缩短/跨 eval 增量),
必须 design-first 立项; 以及 v6-batch 解冻(收益模型需重推)。**

## 九、终局判定: 单 ligand redocking 优化穷尽 ✅

| 层 | 状态 | 证据 |
|---|---|---|
| k2 kernel | 地板 | DRAM4%/SM28% 无饱和 + 并发1.30x + Q4-6a/Q4-4 三确认 |
| ctx warmup 78ms | 地板 | 真实 ctx init + module load (Q4-10a 实测余量 0) |
| 输出写 | ~0ms | Q4-10b 实测, 无杠杆 |
| PD_REFINE | 语义 | 砍=精度决策, 非优化 |
| parse/teardown | ~40ms | 量级小, 无杠杆 |
| runner tail 52ms | 工具链 | 口径问题, 非程序问题 |

| 层 | 状态 | 证据 |
|---|---|---|
| k2 kernel | 地板 | DRAM4%/SM28% 无饱和 + 并发1.30x + Q4-6a/Q4-4 三确认 |
| ctx warmup 78ms | 地板 | 真实 ctx init + module load (Q4-10a 实测余量 0) |
| 输出写 | ~0ms | Q4-10b 实测, 无杠杆 |
| PD_REFINE | 语义 | 砍=精度决策, 非优化 |
| parse/teardown | ~40ms | 量级小, 无杠杆 |
| runner tail 52ms | 工具链 | 口径问题, 非程序问题 |

**redocking 关账。下一步按用户指令解冻 v6-batch 时, 先重推收益模型
(kernel 级收益已缩水, entry-gap 100ms×N 是主收益)。**

- 世代表可再加 Q4-9R (k2 0.050) 与 Q4-10 行
- 三层框架补第 4 实例: 生命周期/尺寸错配 (lm 数组) + **第 5 实例:
  僵尸探针成本** (warmup 探针按旧 VRAM 尺寸工作, 底层优化后未回收)
- "redocking 穷尽性" 一节可直接引用本审计: kernel 层地板证据链
  (三次独立确认) + host 层残余构成
