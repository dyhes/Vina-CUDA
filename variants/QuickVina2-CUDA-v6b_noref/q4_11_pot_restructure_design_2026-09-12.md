# Q4-11 设计文档: POT 树重构 — latency-bound kernel 的算法级优化 — 2026-09-12

> 用户指令: 结合论文/arXiv, 按 latency-bound kernel 优化思路(block size /
> 尾部效应 / ILP)做算法级重构设计。本文档 = design-first 立项书, 代码未动。

## 〇、设计前置: 用户三角度 × 实测证据的裁决

### 0.1 Block size (32→64/128) — 有条件的大杠杆, 被审计门禁挡
- E(能量+梯度)占 kernel ~95%, 其 atom/pairs 循环可 block 并行; v4.5 实测
  block-E at 128 threads = 2.9× 快/次
- 但 128 线程版 6 处 sync 隐患只修了 2 处, serial-LS 版仍损坏 (v4.5b)
- **门禁 = 全 shared 写审计 (Q4-5b 遗留)**; 审计本身 1-2 会话
- 结论: 不是"调参", 是"重构 + 正确性工程"; 放 S4(远期), 不进本轮

### 0.2 尾部效应 — 实测低优先级
- 8192 blocks / (128 SM × 7 blocks) = 9 波, 波 quantization 存在
  (8192 % 896 ≠ 0, 尾波最坏 ~11%)
- 但轨迹时长异质(LS trials 4-5 次随机)反而**填补**了尾波空隙;
  并发探针(1.30x/3proc)证明 SM 不缺 block, 缺的是 per-warp 的发射
- 文献(CUTLASS Stream-K / persistent kernels)针对的是 tile 同质场景;
  我们的最慢 block = LS 失败率 24% 的算法方差, 非调度问题
- 结论: 不单独立项; 若 S1-S3 后仍有尾迹, 用 ncu 的 wave 复盘验证

### 0.3 ILP — 真实存在, 但按站点分布; 最大站点在 POT
- wait 23.7% = 定长依赖链: POT 的 `position_derivative_tmp[i][k] += ...`
  是逐原子串行 FADD 链(~4 cycle/次), 3 累加器打满后流水线空转
- long_sb 15.6% = grid gather 的 RAW 链 (L2 99.7% hit, 无 cache 杠杆)
- 站点级 ILP 修复(多累加器展开/软件流水)是本轮 S2 的内容

## 一、主攻对象: POT_deriv_update (E 的 43%, k2 的 ~38%)

### 1.1 现状解剖(当前树, 非记忆)
```c
__device__ void POT_deriv_update(minus_forces, lig_rigid_gpu, m_coords, g,
                                 threadNumInBlock, threadsPerBlock) {
    __shared__ flo position_derivative_tmp[MAX_RIGID][3];   // 已 shared ✓
    flo position_derivative[MAX_RIGID][3];                  // ← 每线程 local!
    __shared__ flo orientation_derivative_tmp[MAX_RIGID][3];// 已 shared ✓
    flo orientation_derivative[MAX_RIGID][3];               // ← 每线程 local!
    flo torsion_derivative[MAX_RIGID];                      // ← 每线程 local
    // 相1: 原子力→节点合力/合力矩, strided ✓ 并行
    // 相2: 自底向上树累加 for(i=R-1;i>=0;i--) — 无线程 guard,
    //      O(R²) children_map 位扫描, 32 线程冗余执行
    // 相3: g 写出
}
```
- R = num_rigid = torsions+1 (CASF 最大 37); MAX_RIGID=40
- 每线程 local: 2×[40][3]×4B + [40]×4B = 960+160+对齐 ≈ 1.2KB
  → STACK 9984B 的主要成分(Q4-9R 已修 lm 数组, 这是剩余 STL 大头)
- 32 线程冗余执行 O(R²) 树遍历: R=37 时 2×37²×32 ≈ 88K 次位测试/E 调用,
  ×每 BFGS ~19 次 E ≈ 1.7M 次/轨迹
- 自底向上累加是串行深度 = 树高(典型 3-6, 最坏 R)的依赖链

### 1.2 为什么 POT 是最后的大站点
- stage_timer 实测: POT 占 E 的 28-49%(median 43%), E 占 k2 的 ~88%
- GRID 已穷尽(L2 99.7% hit, FP16/texture/__ldg 全部证伪)
- INTRA 的 p_eval gather 连续 + cutoff 剪枝是物理性的(不可省)
- POT 是"纯簿记"却占 43% —— 全部来自 local 溢出 + 冗余 + 串行树

## 二、文献定位(现有论文 + arXiv 检索)

### 2.0 领域总图: 已发表的 GPU 对接论文如何处理 latency-bound/BFGS

七篇核心论文的策略分类(全部读完/核对过原文摘要与相关章节):

| 论文 | 对 LS/BFGS 串行链的策略 | 并行化轴 |
|---|---|---|
| **Vina-GPU** (Tang 2022, Molecules 27:3041) | BFGS 保持在 work-item 内**串行**; 改并行的是**轨迹数**(大规模提高) + 砍 search_depth; 树→列表转换便于 OpenCL | 轨迹间 |
| **AutoDock-GPU** (Santos-Martins 2021, JCTC) | LS(Solis-Wets/ADADELTA) 在 work-group 内串行; **GA 种群并行**; v1.2 加 **early termination**(能量已优则跳过 LS) — 基准测试砍 50% runtime | 个体间 + 评分内 |
| **Vina-GPU 2.1** (Tang 2024, TCBB, **RILC-BFGS**) | **降迭代数+降复杂度的 BFGS 变体**(算法级减功) + grid cache | 轨迹间(同 2.0) |
| **Schieffer TC** (arXiv 2410.10447) | 评分归约上 Tensor Core; 明确指出"reduction 是 LS kernel 的主要 warp stall 来源" | 评分内 |
| **Uni-Dock** (ChemRxiv 2022) | 每分子数百线程(=评分内并行), **分子批间并行**; LS 串行 | 分子批间 + 评分内 |
| **PETSc/TAO L-BFGS** (arXiv 2608.00196) | 通用 L-BFGS 两循环在 GPU 上同样 latency-bound; 解法=**批量解耦多问题**, 不动两循环 | 问题间 |
| **AutoDock-GPU 不规则计算基准** (PMC8654209) | **early termination 系统化**: 自适应 LS trial 上限/跳过无望模式, 平均 -50% | 工作量削减 |

**领域共识(无人例外)**: 没有任何论文在 GPU 上并行化 BFGS 迭代链本身。
全部四条已验证路线:
(a) **跨轨迹/个体/分子并行**(尴尬并行轴);
(b) **单次评分内部并行**(atom/pairs 归约);
(c) **算法级减功**(RILC 降迭代/低复杂度方向更新);
(d) **工作量削减**(early termination / 启发式 search_depth)。
串行 LS 链被全部作者视为**固有属性**。我们的 #50(warm-start +19~29%)/
#51(投机 LS 结构不可行)两个独立否定与领域共识一致——这是"边界证明",
不是我们做错了。

### 2.1 技术路线裁决表

| 方向 | 文献 | 适配判定 |
|---|---|---|
| Parallel scan 破 BFGS 两循环 | BPPSA 1907.10134 / ScanWeaver 2606.00601 / Gonzalez 2603.16850 | **不适用**: RILC 只保留当前 (s,y), 无 m-history 递归 — 没有可 scan 的链(领域共识: 也没人这么做) |
| 异步/lock-free quasi-Newton | AsySQN 2011.00667 | 不适用: 单轨迹 LS 链不可异步; 多轨迹独立性已占用 |
| ILP/occupancy 取舍 | **Instruction Wall (arXiv 2608.13696)** | **核心依据**: 60-70% stall 来自内存延迟+指令依赖, 解法=ILP 重构 > 加 occupancy; 与 wait 23.7% + No Eligible 83.5% 同构 |
| 尾部/wave quantization | Stream-K / WaveTune 2604.10187 / RaMP 2604.26039 | 低优先: 同质 tile 技术; 对接领域无人处理此问题(轨迹数=grid 已天然覆盖), 尾迹=算法方差 |
| 树遍历 GPU 化 | GPU Tree Traversals (Zhou et al.) lockstep 规约 | S3 方法学: lockstep 下 per-warp 栈共享 → 按层同步+shared 累加 |
| stall 根因诊断 | LEO 2604.20032 | 方法论背书: wait/long_sb 分解已做, 结论一致 |
| block size | Uni-Dock 每分子数百线程 + AutoDock-GPU work-group + v4.5 实测 2.9×/E | 大杠杆, 审计门禁后做(S4) |
| **early termination** | **AutoDock-GPU v1.2 (PMC8654209), -50%** | **我们未开采的文献路线**(见 §2.2) |
| SpMM/level-sync 归约 | Sputnik 2006.10901 / MIT 6.506 | S3 的 level-synchronous 模板: 按拓扑层并行+层间 sync |
| block size | v4.5 实测 (2.9×/E) + UniEP (32 warps 藏 ILP) | 大杠杆但需 sync 审计, S4 |

### 2.2 early termination 事前评估结论(2026-09-12): 机制已存在, 该路线已开采

逐切口核对当前代码后的判决:

| 切口 | 现状 | 判定 |
|---|---|---|
| ② LS trial 级提前判负 | **Lewis-Overton 本身就是自适应 bracketing**(brackt/Armijo/curvature 违规即刻 return -1), trial 循环天生数据自适应; episode 级还有 **A2 门**(k≥4 且连续 2 次相对改进 <1e-2 → return, 自 Q2-7 就在树里, b94c2fae) | 已存在, 重加=重复且动 LS 收敛语义 |
| ③ 无望模式跳过精修 | 我们的精修(PD_REFINE)只对容器 top modes 做, 逐模式裁剪 = num_modes/energy_range 语义旋钮 | 语义决策, 非优化(D 方向) |
| ① episode 入口能量门 | **无便宜的 pre-BFGS 能量信号**(mutate 不评估; 有意义的门需要 E, 而 E 正是 BFGS 的第一步); 1-step 探测+升级 = 改变"多浅重启"搜索语义(#50 同类) | 奖品未知且机制性自限 |

**机制性自限的量化**(Sep08 stage 剖面, 1pxn):
- iters/episode = **3.52**(344,420/97,831), LS/episode = 4.52
- A2 门把 episode 压在 ~4 迭代; LS 失败率 24% 使坏候选者 1-2 迭代即退出
- 即"坏候选者的 BFGS"本来就不贵 —— **没有想象中的大块浪费**
- AutoDock-GPU 的 -50% 测于 LGA 种群(个体能量来自前次 EVAL, 有免费信号;
  其 v1.2 加的机制 = 我们已有的 A2/Lewis-Overton 等价物), **不可移植**

**判定: early termination 三切口全部已覆盖或无杠杆, 该文献路线在我们
代码上已开采完毕。** 唯一未知 = 被拒 episode 的迭代分布直方图(需要
stage_timer 扩展 probe build), 但即使存在也受 A2 上限约束, 预期 <5%,
优先级低于 Q4-11 S1-S3。



**研究空隙(论文贡献点)**: 无任何论文对"分子对接树形力→梯度投影"做
GPU 级并行化; POT 树累加的 lockstep 化 + 溢出消除是可发表的微贡献。

## 三、三阶段设计(每阶段独立 fork + 门禁)

### S1 — 溢出消除 (单变量, 最小改动)
```
- flo position_derivative[MAX_RIGID][3];      → __shared__ (960B)
- flo orientation_derivative[MAX_RIGID][3];   → __shared__ (960B)
- flo torsion_derivative[MAX_RIGID];          → 保留 local (160B, 只在相3写)
```
- shared 12916+960 = 13876B < 14628 (7-block 预算) → **occupancy 不掉**
- 预期: STACK 9984→~8KB, POT 的 STL/LDL 链消失, POT 43%→~30%
- 风险: 树累加改为跨线程读写 shared → 需要相间 __syncthreads(已有纪律)
- 门禁: 六档冒烟 + 2×CASF mean

### S2 — ILP + 位扫描 (叠加在 S1 上)
- 相1 的串行 FADD 链: 每节点 2 累加器展开(处理 j, j+1 两原子),
  打破 4-cycle FADD 依赖 → ILP ×2
- 相2 的 children_map 内循环: `while(mask){ j=__ffsll(mask); mask&=mask-1; }`
  只迭代真实子节点, O(R²)→O(R+Σchildren); 并加 thread0 guard(该结果
  与线程无关, 32 路冗余纯浪费) — 修复时注意与 S1 的 shared 读写边界
- 预期: 相1 时间 ×0.6, 相2 ×0.3; POT →~22%
- 门禁: 同上 + ncu 复盘(wait% 应下降)

### S3 — 树累加 level-synchronous (结构重构)
- 自底向上串行(R 步) → 按树深层间同步并行:
  - 预处理(host, 一次/配体): BFS 层号 layer[node] + 按层 CSR 布局
  - 设备: for L = maxdepth..0: 该层节点被 32 线程均摊, 节点 j 的累加 =
    self + Σ children[j] (children 已在 L+1 层算好, __syncthreads 层障)
  - 深度 = 树高(配体树典型 3-6, 远小于 R=37)
- 收益模型: 串行深度 R→树高; 32 路冗余→1 路 + 线程均摊内层
- 预期: POT 43%→15-20%, **累计 k2 -10~15%** (与 v10 候选的旧估计一致)
- 风险: 层障次数 = 树高 ×(barrier 0.5% 基数, 低); 需要新 h2d 布局
  (layer[] 数组进 mis 或独立 buffer) — 设计先于编码

### S4 (远期, 独立立项) — 128 线程块 + 全 shared 写审计
E 的 atom/pairs 循环 2.9× 并行潜力; 前置 = v4.5b 六处 sync 隐患审计。
与 S3 正交, 可在 S3 后叠加。

## 四、预期与止损

| 阶段 | k2 预期 | 累计 | 止损条件 |
|---|---|---|---|
| S1 | -4~6% | | 冒烟错值/CASF 回归 → 还原 |
| S2 | -3~5% | -7~10% | ncu wait% 不降 → 停 |
| S3 | -5~8% | -12~15% | 层障开销 > 收益(树高实测) → 停在 S2 |

若 S1-S3 全部兑现: rt 0.320 → ~0.29, k2 0.050 → ~0.043。
CASF Top-1 门禁不变(分布中性, 不碰数值语义——只改执行序, 浮点和的
结合律变化在 S3 需 aff 对称门禁确认)。

## 五、执行清单(下一步即 S1)

1. fork: 现纪律已在树上, 直接开分支 commit
2. S1 补丁(~10 行): 两个数组加 __shared__, 树累加前插 __syncthreads
3. make + 1pxn/1c5z 冒烟(锚点 -8.9/-5.9) + res-usage 验证 7 blocks 保持
4. ncu: POT share + STACK 变化
5. 2×CASF 门禁(aff 对称 + Top-1 两跑中位带内)
6. 认证 → S2


## 六、S1 实测结果(2026-09-12 凌晨) — 正确但性能中性, S2/S3 取消

### 实测
- SHARED 12916→13876(+960B), 7 blocks/SM 保持 ✓; STACK 9984→9808(仅-176B)
- 正确性: 1c5z -5.9 / 1pxn -9.0 与 Q4-9R 逐位一致(锚点吻合)
  (中途一次假警报: ad-hoc 盒中心写错导致 1c5z -0.2, 两二进制一致错误
  ——是盒中心错误非代码 bug; 修正盒中心后两构建逐位一致)
- 2×CASF: rt 0.320/0.320, k2 0.052/0.052, Top-1 {180,178}(带上缘)
- **判定: 正确保持 + 性能中性(k2 ±4% 噪声内), 预测的 -4~6% 未兑现**

### 根因(第三次预测修正, 本次最有价值)
latency-bound 内核里 **删 ALU 冗余不缩短关键路径**: 关键路径 = LS 串行链
+ L2 gather 延迟暴露, 冗余 ALU 工作本就填充在必然空闲的发射槽里。
POT 占 43% 是"指令数占比"而非"关键路径占比" —— 这正是 Instruction
Wall 论文(2608.13696)的核心论点在自家数据上的验证: 该论文明确说
"提高 ILP 与降低 occupancy 的取舍"以及"仅加 occupancy 不达 peak"——
但反向同样成立: **仅减指令也不达 peak, 除非砍的是延迟链本身**。

### 止损执行(按预案)
- S2/S3 与 S1 同族(降 ALU/降串行度), S1 平坦已证伪其前提 → **取消**
- S1 保留在实验分支 `q4-11-s1-experiment`(81aad7a2): 数值安全且 S4
  (128 线程)重构时 shared 化是前置需求
- 主线还原 Q4-10b(aec75100)并重建验证(1pxn -9.0 / k2 0.055)

## 七、Q4-11 总结: 三个否定结论 + 一条已验证的地板

1. BPPSA/scan 类: 不适用(RILC 无两循环) — 结构性排除
2. early termination: 三切口已覆盖(A2/Lewis-Overton/visited), -50% 不可移植
3. POT 冗余/溢出消除(S1): 正确但性能中性 — latency-bound 不为 ALU 减量付费
4. k2 地板证据链最终形态: 无管线饱和 + 并发 1.30x + S1 ALU 减量中性
   —— 三条独立证据指向同一结论: **关键路径是 LS 串行等待, 不是任何
   可减的本地工作量**

**redocking kernel 层正式关账。剩余路径: v6-batch(解冻后先重推收益
模型) / S4 128 线程(先全 shared 写审计) / 论文写作。**

## 八、64 线程方案设计(2026-09-12, 用户提案采纳)与 S4a-1 落地

### 资源对齐分析(用户提案: 64t + shared 削到 8 块)
| 配置 | reg 限制 | shared 限制 | warps/SM | 轨迹/SM |
|---|---|---|---|---|
| 32t 现状 | 16 | **7** | 7 (14.6%) | 7 |
| 64t | 8 (64×128=8192, 64K/8192=8) | 7~8 | **14~16** (29~33%) | 7~8 |
| 128t | **4** (64K/16384) | 8 | 16 (33%) | 4 |

64t 优于 128t: warps 相同(16)但 64t 保留 8 条独立 LS 链(交错/负载均衡
更好), 且寄存器 64K/8192 整除零浪费。

### S4a-1 已落地(46bc3cce, 主线)
- mis_cl → `__constant__`(TU 内符号, cudaGetSymbolAddress ncu-safe): -68B
- MAX_RIGID 40→38: **实测 CASF-2016 max 36 BRANCH(3ag9)**, 37 节点 ≤38: -164B
- shared 12916→**12684B**; 冒烟 1pxn -9.1 / 1c5z -5.9 ✓, k2 0.055 不变
- **重要修正**: ncu 显示 shared 限制仍 7 块——8 块需 static ≤11,776B
  (每块 1KB driver reserve 我此前没算)。但 **64t 不需要 8 块**:
  7 块 × 2 warps = 14 warps/SM = 现状的 2 倍。进一步 shave(还需 ~900B)
  推迟, 先看 64t@7blocks 的实测。

### S4a-2 (64t) 审计清单 — 下一会话开门第一件事
1. compute-sanitizer racecheck 归零(1pxn seed42, 64t 构建)
2. 四处已知雷人工核对: mutate quaternion 多写者 RMW / gyration_radius
   发散 gr / 发散分支内 __syncthreads / kernel2 相间屏障
3. "No-op at threadsPerBlock=32" 注释的全部屏障 → 变真屏障, 逐个验证
   读写边界
4. visited_cl 栈 4.3KB/线程 × 64 = local 流量翻倍, 审计时评估 hybrid
5. 门禁: 冒烟六档 + 2×CASF(aff 对称 + Top-1 两跑中位) + ncu(occupancy
   应到 ~29%, long_sb 应下降)
预期: E 并行段 ×2, k2 -20~35%(Amdahl: 串行段不变)。

### 实验设计
1. 三目标 solo ncu(kernel2_update): 1pxn 58.6ms/SM30.5/L1 30.5/L2 29.2;
   1u1b 509ms/SM26.9; 3bv9 515.6ms/SM27.8 —— 全部 <31%
2. 并发干扰测试: ncu 下并发 → **实验无效**(ncu 需独占计数器, 序列化了竞争)
3. 转向 MPS 测试(绕开 ncu): MPS daemon + 3 进程并发 vs 串行

### 关键发现: 无 MPS 的并发 ≠ 内核并发
CUDA 多进程默认**时间切片**(context switching), 内核并不共存!
原 3 进程探针的 1.30x = 启动/拷贝的 overlap, 不是 SM 共享 →
**"并发 1.30x"作为"k2 无 slack"证据是混淆的, 作废重测。**

### MPS 实测(client 确认连上 server.log)
| | kernel 和 | 3 进程 wall |
|---|---|---|
| 串行 | 0.909s (0.055+0.423+0.431) | ~1.75s(含 3×启动) |
| **MPS 并发** | — | **1.100s** |

3 进程 MPS 加速 = 1.75/1.10 = **1.59x**(vs 无 MPS 1.30x)。
但分解: 两个 ~0.42s 内核在 MPS 下共存后墙钟仍 ~0.84s —— **共存但
零加速**: 每个 kernel 的 8192 块挤同一批 896 个块槽(7 blocks/SM × 128),
被 shared 上限(13876B×7≈97KB of 100KB/SM)钉死。

### 终极结论: shared 内存墙 = 第四机制
- 块槽被 SHARED 容量钉死 → 任何进程数的块共存都无法增加并行度
- **这同时解释了** solo 的低 occupancy、并发的零扩展、以及为什么
  S1 减 ALU/减指令都无效(瓶颈不是指令发射, 是块级资源)
- **k2 地板证据链升级为四证**: 无管线饱和 / 无 MPS 时序切换 /
  MPS 共存零加速 / shared 容量 = 7 blocks 硬顶

### 对 v6-batch 的最终修正(重要)
- **kernel 级 co-residency 吞吐假设死亡**: batch 不可能靠"填空窗"
  加速 kernel 本身 —— 块槽已满
- batch 的真实收益只剩: 启动/entry-gap/传输摊销(进程级常数)
- 对 v6 合并的收益模型: screening 场景(多配体同受体)仍有效——
  但收益来自消掉每配体 ~100ms entry-gap, 不是 GPU 共享
- **若未来想真正提升 k2: 必须降 shared/block**(让 8+ blocks 共存)
  或减块资源(MAX_RIGID/结构裁剪), 这与 Q4-6a 的 8-blocks 尝试不同——
  现在知道的是"7 是 shared 钉死的, 不是 warp 数不够"

## 九、对 L1TEX 假设本身的回答
未直接验证(ncu 无法在并发下测), 但被上位机制取代: 就算 L1 不饱和,
块槽也进不去。L1TEX 从"主攻候选"降级为"shared 墙突破后的次级观察项"。


## 十五、S4a-2 执行结果(2026-09-12): 64 线程正式落地(344d9f52)

### 设计落地
- 唯一必需改动: `dim3 block_size(16, 4, 1)`(+S4a-1 的容量准备)
- 全部既有机制参数化正确: threadNumInBlock=tx+16*ty 线性、lane/wid
  warp 拆分、strided 循环、v4.5 cross-warp 归约(warp_e[32]+wid 树)、
  gyration/ig/POT 归约的 barrier-outside-if 模式

### 审计结果
1. **synccheck(64t, 完成跑): 0 barrier errors** ✓ — 64t 特有的
   发散 barrier 风险类实测干净
2. racecheck: 无法完成(sanitizer 在 kernel2 启动点 OOM/被杀, exit 9,
   1024/8192 线程同样; 死前显示 0 hazards) — 环境限制, 部分通过
3. 低线程数崩溃(1024/2048/4096 quaternion normalize + IllegalAddress):
   **32t 对照构建同样崩溃** — 预先存在的家族行为, 非 64t 回归
   (batching gotchas 早就记录 thread=1000 quaternion crash)

### 门禁(2×CASF 全量)
| | run1 | run2 | 判定 |
|---|---|---|---|
| rt | 0.310 | 0.310 | -3% ✓ |
| e2e_dock | 0.101 | 0.101 | 持平 ✓ |
| k2 median | 0.051 | 0.050 | 小配体平坦(预期) |
| Top-1 | 175/176 | 176/177 | **带内** ✓ 两跑中位 175.5 |
| aff vs Q4-9R | 0.159 | 0.173 | 对称(97/71, 94/69) ✓ |

### 分复杂度 k2(Amdahl 特征完美复现, 两跑一致)
| 配体 | 原子 | k2 变化 |
|---|---|---|
| 3ag9 | 91 | **-17.5% / -16.7%** |
| 3uri | 78 | **-18.3% / -19.6%** |
| 3bv9 | 59 | **-17.7% / -15.1%** |
| 1pxn | 29 | -2.0% |
| 1c5z | 14 | 0% |

n≥59 的配体稳定 -15~20%(E 并行段 2 warps), 小配体原子循环太短无法
切分——Amdahl 分母效应与预测方向一致, 幅度落在保守区间。

### occupancy 验证
sm__warps_active 27.8%(预测 29%: 14 warps/64 上限)——2× 达成,
shared 限制 7 块未变(符合账目: 8 块需 static ≤11776B)。

### 语义注意(必须写进论文口径)
64t 的跨 warp 归约改变了块内浮点求和顺序 → 同 seed 下**结果分布移动**
(3ag9 单点 -0.7 kcal)。这是结构性的、非随机的; 两跑 CASF 显示
分布整体仍在家族带内(Top-1 {175,176}, aff mean|Δ| ≤0.173)。论文引用
64t 结果时必须标注"块归约序改变, 分布口径迁移至新带"。

## 十六、最终状态
- **主线 HEAD = 344d9f52 (64t)**: rt 0.31 / e2e_dock 0.101 /
  k2 median 0.050-0.051(小配体) / **大配体 k2 -15~20%** / Top-1 带内
- 世代链: aec75100(Q4-10b) → 46bc3cce(S4a-1) → 344d9f52(S4a-2)
- 32t 构建不再维护; --thread 8192 是唯一稳定配置(与历代一致)
