# BFGS 全流程 × FP32 vs FP16 能量面分析 — 2026-09-12

> 回答: "结合代码详细梳理 BFGS 的全过程, 讨论 FP32 和 FP16 的能量面有什么区别"
> 所有代码引用来自 Q5 主线 (cd10ca24, 64t), 行号可复核。

## 一、BFGS 完整计算流程(代码级)

### 1.1 调用层级

```
kernel2_update (每块 = 1 条 MC 轨迹, 64 线程 = 2 warps)
│
├── for step in 0..search_depth-1:               # MC 外循环, search_depth=8
│   ├── candidate = tmp                           # 拷贝当前最优
│   ├── mutate_conf_cl_update(&candidate)         # 随机扰动(POT 空间)
│   ├── rilc_bfgs_update(&candidate, ...)         # BFGS 精修 ← 主要耗时
│   │   │
│   │   ├── fx = m_eval_deriv_update(x, g, ...)   # 能量+梯度 E()
│   │   │   ├── set_update(c, rigid, m_coords)    # 构象→原子坐标
│   │   │   │   └── local_to_lab(...)             # 树传播 + 四元数变换
│   │   │   ├── ig_eval_deriv_update(...)          # grid 插值(PER-ATOM)
│   │   │   │   └── g_evaluate_update(...)         # 三线性插值
│   │   │   ├── eval_interacting_pairs(...)        # INTRA pairs
│   │   │   │   └── p_eval_deriv(...)              # 预计算表插值
│   │   │   └── POT_deriv_update(...)              # 力→梯度(树走)
│   │   │
│   │   ├── if !interesting(): return              # 去重门
│   │   ├── d = -g                                 # 初始方向
│   │   ├── while true:                            # BFGS 迭代循环
│   │   │   ├── ls = line_search_lewisoverton(...) # 线搜索
│   │   │   │   ├── x = xp + stp × d               # 试探步
│   │   │   │   ├── set_update(x)                  # 又一次树传播
│   │   │   │   ├── f, g = m_eval_deriv_update(x)  # 又一次 E()
│   │   │   │   ├── Armijo: f ≤ finit + c×stp×g₀·d ?
│   │   │   │   ├── curvature: g·d ≥ dstest ?
│   │   │   │   ├── brackt → stp = (mu+nu)/2       # 二分
|   |   |   |   └── count ≥ 8 → return             # trial 上限
│   │   │   ├── A2: 连续 2 次相对改进 < 1e-2 → return
│   │   │   ├── ‖g‖ < 1e-5 → return               # 收敛门
│   │   │   └── RILC 方向更新(单对 s/y, O(n))
│   │   └── (episode 结束: LS fail / 收敛 / max_steps)
│   ├── metropolis_accept(tmp.e, candidate.e)      # MC 接受判定
│   └── if tmp.e < best_e: best_out = tmp          # 记录最优
│
└── results[block] = best_out                      # 写回结果
```

### 1.2 每步的计算量和精度关键点

| 阶段 | 每 episode 调用次数 | 计算内容 | 精度敏感量 |
|---|---|---|---|
| set_update | ~19(每次 E 前) | 树传播: 坐标=父坐标+旋转变换 | **坐标累计量化误差**(深度×量化) |
| grid 插值 | 同上(每原子) | 三线性插值(grid FP16) | 插值坐标量化 → 权重跳变 |
| INTRA pairs | 同上(~500 pairs) | p_eval 表插值 + r² 计算 | r² 对坐标量化敏感(近 cutoff 翻转) |
| POT 树走 | 同上(每次 E 后) | 力→梯度投影 | 梯度对坐标误差敏感(树深×链式) |
| RILC 方向 | 每迭代 | O(n) 单对更新 | 精度要求中等(单对) |
| ‖g‖ 收敛门 | 每迭代 | scalar_product | **噪声地板 vs 门限** |

## 二、能量面: FP32 vs FP16 的数学差异

### 2.1 能量面的连续性

**FP32 坐标路径**: 
- m_coords 以 ~1e-7Å 精度表示(23 位尾数)
- set_update 的树传播: 误差累积 ~1e-6Å(6 层深度 × 每层 1e-7)
- grid 插值: s = (coord - init) × factor, s 连续变化
- 插值权重连续变化 → 能量面**连续可导**(在网格精度内)
- BFGS 二次模型有效: Armijo 和曲率条件在光滑面上正常工作

**FP16 坐标路径(E1)**:
- m_coords 以 0.0078Å 步长量化(10 位尾数在 ±11Å 范围)
- set_update 树传播: 每层写入时截断 → 深度 6 累计 ~0.02-0.05Å
- grid 插值: s 跳变步长 = 0.0078Å × 2.71 = **0.021 grid 单位**
- 插值能量跳变 ≈ Δgrid_value × 0.021 ≈ **0.002-0.02 kcal**/坐标跳变
- **能量面 = 阶梯函数**(不连续), 台阶尺寸 ~0.002 kcal

### 2.2 梯度的数学

Vina 打分函数的梯度 = dE/dr × dr/dx(链式法则):
- dE/dr: 来自 p_eval_deriv 的预计算表(光滑)
- dr/dx: 来自 set_update 的坐标变换(旋转矩阵 × 内部坐标)

**FP32**: dr/dx 是连续函数, BFGS 的拟牛顿方向 d = -H⁻¹g 精确指向下降方向。

**FP16**: dr/dx 在量化边界处不连续 — 但这不是主要问题。主要问题是:

```
‖g‖ = sqrt(Σᵢ (dE/dxᵢ)²)

其中每个 dE/dxᵢ 包含来自 FP16 坐标量化的噪声:
δgᵢ ≈ (d²E/dx²) × δx  其中 δx = 0.0039Å(半步量化)

在接近极小值处, d²E/dx² ≈ Hessian 对角 ≈ 1-10 kcal/mol/Å²
δgᵢ ≈ 1-10 × 0.0039 ≈ 0.004-0.04 kcal/mol/Å

‖g‖_noise ≈ sqrt(n) × 0.02 ≈ 0.1-0.2 kcal/mol/Å (n=55 维)
```

**收敛门 ‖g‖ < 1e-5 永远不会被 FP16 噪声地板以下的 ‖g‖ 触发** — 
即使真实梯度为零, 计算出的 ‖g‖ ≈ 0.1(量化噪声)。

### 2.3 Armijo 条件在阶梯面上的行为

```
Armijo: f(x + stp×d) ≤ f(x) + c₁ × stp × g·d

FP32: f 连续可微 → 当 stp 足够小时, Armijo 必然满足
      → LS 总能找到有效步长 → episode 正常收敛

FP16: f 是阶梯函数 → f(x + stp×d) 的值取决于 x 落在哪个量化格子里
      → 当步长小于量化步长(0.0078Å / ‖d‖)时, f 不随 stp 变化
      → Armijo 变成 f(new) ≤ f(old) - ε, 其中 ε ~ 0.002 kcal(量化噪声)
      → 如果 f(old) 和 f(new) 在同一量化格子内: f(new) = f(old), 
        Armijo 要求 0 ≤ -ε, 永远不满足 → LS 失败
      → 如果跨了格子: f 跳变可能向上也可能向下, 随机

     → episode 的 LS 失败率从 24%(FP32) 上升到接近 100%(FP16, 
       当 ‖g‖ 已经小于量化噪声水平时)
```

**这解释了 E1 的 "No results"**: 
所有 episode 的 LS 在几步内就失败 → episode 立即结束 → 
返回的构象就是 mutate 后的初始构象(未经优化) → 
这些构象的能量等于 mutate 后的能量(通常很高, 因为 mutation 是随机的)
→ metropolis 以温度 1.2 接受了一些 → 但最终 best_e 也是很高的值
→ out_cont 非空但能量太差 → "No results" 来自质量过滤

**等一下, E1 的日志确实说 "No results found for ligand"** — 
回看 main.cpp L483: `if (!out_cont.empty()) {...} else { "No results" }`
out_cont 是从 results 缓冲区构建的。如果 best_out.e = 初始的
`mutate 后未经 BFGS 精修的能量` (~+40 kcal 或更高), 那不是 NaN,
out_cont 应该非空... 除非: E1 的 26ms k2 说明 BFGS episode 极短
(A2 门或 LS 立即失败), best_out.e = eval_init 的 fx(第一次 E 的值),
而 E1 的 m_coords 是 FP16 的, 初始坐标来自 ric(随机构象)经 set_update — 
这些构象能量是 ~+10², 不是空。所以 "No results" 不是 out_cont 空,
是 out_cont 非空但**所有构象都未通过后续的质量检查**。

让我重新检查: "No results" 打印在 main.cpp 的 out_cont 为空的分支。
out_cont 来自 `output_container out_cont = out_conts[ligand_count]`, 
而 out_conts 是从 results 缓冲区排序去重后的输出。如果所有 8192 
轨迹的 best_out 都相同(因为 metropolis 全拒绝, best_out = 初始
ric 构象), 去重后只剩 1 个, 仍然非空... 

**所以 "No results" 的机制需要更仔细核实** — 但这不影响本节的核心
分析(FP16 能量面的阶梯性 vs FP32 的光滑性)。

## 三、核心结论: FP32 与 FP16 能量面的区别

| 性质 | FP32 | FP16 |
|---|---|---|
| 连续性 | 连续(在网格精度内) | **阶梯函数**(量化步长 0.0078Å) |
| 可导性 | 光滑(BFGS 二次模型假设成立) | 不连续(在量化边界处跳变) |
| 梯度噪声地板 | ~1e-7 | **~0.1 kcal/mol/Å** |
| 收敛门 1e-5 | 可达 | **永不可达**(噪声地板 >> 门限) |
| Armijo 行为 | 光滑下降 → 可满足 | 量化跳变 → 随机通过/失败 |
| LS 收敛率 | 76%(24% 失败) | ≈100% 失败(近极小值处) |
| BFGS 有效性 | 方向精确 | 方向被量化噪声淹没 |

**核心区别**: FP32 能量面是 BFGS 可以导航的**光滑曲面**; FP16 
坐标量化把能量面变成了 **BFGS 无法导航的阶梯地形** — 不是因为
"方向不好", 而是因为"脚下的地面本身在跳动"。

## 四、如果坚持 FP16: 需要什么条件?

1. **收敛门适配噪声地板**: ‖g‖ 门从 1e-5 → 0.1(FP16 噪声量级)
   → episode 变得更浅 → 搜索质量下降(这就是 D 方向, 但动机不同)
2. **量化噪声平滑**: 对 FP16 坐标做插值/平滑 — 但这需要额外的
   FP32 计算, 违背初衷
3. **随机舍入(stochastic rounding)**: 用随机舍入替代截断, 使量化
   误差在多次迭代中统计 unbiased → 不改变单次不连续性, 但改变
   长期统计行为 — 这是研究级改动
4. **最小可分辨距离保护**: r² < ε 时跳过 pairs — 但这不解决
   坐标量化本身的问题

以上全部是**语义变更**, 不是工程优化 — 它们改变搜索算法本身。
按用户之前"RMSD/affinity 不显著劣化即可"的标准, 它们**可以测试**,
但预期需要大幅调整搜索参数(search_depth, mutation_amplitude,
temperature)才能恢复等效搜索质量 — 这等于重新调参一个新算法。
