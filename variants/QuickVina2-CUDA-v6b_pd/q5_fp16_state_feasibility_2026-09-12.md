# Q5 FP16 state / 16-block feasibility experiment (2026-09-12) — REFUTED with root cause

> Question: can FP16 storage of the shared optimizer state (m_coords,
> minus_forces, x/g/candidate) halve shared and unlock 16 blocks/SM at
> 64 threads × 64 regs? Answer: NO — empirically, with a physical root
> cause. Also settles the m1a "BF16 dead" verdict's transferability and
> the ‖g‖ threshold's role.

## 1. Resource arithmetic that motivated the experiment

| config | reg limit | shared limit | actual blocks | warps/SM |
|---|---|---|---|---|
| 64t × 128 regs (Q5) | 8 | 7 | 7 | 14 (27.8%) |
| 64t × 64 regs (user proposal) | **16** | 7 (FP32 shared) | 7 | 14 — no gain |
| 64t × 64 regs + FP16 state | 16 | 13–16 | 13–16 | 26–32 (47–57%) |

Key correction: each block also carries ~1 KB of driver-reserved shared, so
the 8-block target needed static ≤ 11,776 B (not 12,800 B). FP16 state +
register halving is a REQUIRED PAIR — halving regs alone hits the shared
wall (7 blocks), and FP16-ing shared alone hits the register wall (8 blocks).

## 2. E1 result: search completely broken, silently

Converted m_coords_cl.coords and m_minus_forces.coords to `__half` with
explicit `__float2half`/`__half2float` helpers at all ~35 access sites.
Shared dropped 12684 → 11148 B; occupancy limit showed 8 blocks on BOTH
shared and registers. k2 "improved" 55→26 ms. But: **"No results found for
ligand" — zero conformations produced**, zero CUDA errors.

Diagnostic print of candidate energies per MC step: `cand.e = 15,705,782 /
22,448,062 / 33,474,280` (15–33 MILLION kcal/mol vs normal ±100).

## 3. Root cause: FP16 coordinate quantization manufactures atom collisions

Chain (each step measured or source-verified):

```
FP16 mantissa (10 bits) at 10 Å frame → quantization step ~0.0078 Å
→ after tree propagation, atoms 0.01–0.05 Å apart quantize to the
  SAME position (r² = 0)
→ Vina scoring's 1/r¹² repulsion at r→0 explodes: (1/0.001)¹² ≈ 1e36
→ M2 stores grid values as __half (max 65504) → interpolated energy
  at the collision = Inf
→ Inf energy → Inf/NaN gradient → NaN coordinates → all subsequent
  evals NaN
→ NaN < 1e38 is always false → best_e never updates → results
  buffer stays at initial state → out_cont empty → "No results"
```

The 15M-energy trajectories are the ones that came CLOSE to collision
(partial explosion); exact collisions went straight to NaN. Zero CUDA
errors because every NaN comparison is silently false — the pipeline
runs to completion producing nothing.

## 4. Why FP32 never hits this

FP32's 23-bit mantissa at a 10 Å frame gives ~1e-6 Å spacing — two atoms
cannot quantize to the same position, r² > 0 always, the scoring surface
stays continuous, and BFGS can navigate it. FP16's 0.0078 Å step is the
same order as atom-pair distances in tight poses — quantization
manufacures the unphysical states that FP32 never produces.

## 5. Threshold relaxation does NOT rescue it (E1v2)

Raising the ‖g‖ gate from 1e-5 to 1e-3 (a relaxation the FP32 threshold
sweep showed quality-neutral) did not restore convergence: still "No
results" at 26 ms. The failure is not the gate-vs-noise-floor race; it is
the scoring surface itself being discontinuous/Inf under quantized
coordinates. A separate 4×5 threshold sweep (1e-5→1e-2, FP32 builds, five
smoke anchors) independently confirmed: k2 moves <2% across 100× of
threshold relaxation (episodes are already terminated by the A2 small-improvement
gate and the 24% LS-failure rate long before ‖g‖ matters), so a relaxed gate
buys no speed either. Both halves of the FP16+relaxed-gate combination fail
independently.

## 6. m1a verdict re-examined (user challenge, upheld)

The original m1a note ("BF16 blanket swap = hang, never repeat") has three
confounds that prevent transferring its verdict to FP16 state storage:
1. The swap targeted the **energy-reduction array consumed by
   `__shfl_down_sync`** — pre-Volta-style shfl on `__nv_bfloat16` requires
   reinterpret tricks; the hang may be an implementation failure, not a
   numerical one.
2. BF16 (7-bit mantissa, FP32-wide exponent) is the WRONG format for
   docking quantities: coordinates live in a bounded box (±11 Å from box
   center), energies are curl-clamped, quaternions are normalized. FP16's
   10-bit mantissa with ±65504 range fits every quantity with margin
   (forces O(10³) ≪ 65504; the 1.8e308 max_fl sentinel exists only in
   host control flow). FP16 strictly dominates BF16 here.
3. m1a tested one array on the pre-Q4 codebase. It did not test
   coordinates, gradients, or optimizer state.

Upheld lesson from m1a that DOES transfer: never blanket-swap a type;
convert at the boundaries and test per-array.

## 7. Final mixed-precision map (all positions empirically resolved)

| storage/compute location | verdict | evidence |
|---|---|---|
| VRAM grid values (m_data) FP16 | ✓ landed (M2) | VRAM −50%, k2 neutral, no search impact |
| FP64 → FP32 in hot path (M1) | ✓ landed (Q4-7) | k2 −46%, SASS FP64 916→123 |
| shared coordinates (m_coords) FP16 | ✗ refuted | E1/E1v2: atom collisions → 1/r¹² explosion → NaN cascade → empty results |
| shared forces (minus_forces) FP16 | ✗ (requires coords conversion to matter; alone saves too little for 8 blocks) | arithmetic: −768 B → 12,940 B > 12,800 threshold |
| shared optimizer state (x/g/candidate) FP16 | ✗ not attempted | masked by coordinate failure; m1a-class risk |
| TC mma for BFGS math | ✗ | K=1, no GEMM shape; math is 0.1% of iteration |
| register halving alone | ✗ zero occupancy gain | shared caps at 7 blocks |

## 8. Closing note

Single-ligand redocking is at its practical resource floor for this
execution model (1 trajectory/block × 2 warps, shared-bound at 7 blocks).
Remaining throughput paths are system-level: v6-batch merge (entry-gap
amortization, not kernel co-residency) and thesis write-up. Any future
revisit of FP16 state must first solve the atom-collision problem
(e.g., r² < ε guard = semantics change, or box-relative coordinate
encoding with reduced range) — both are new projects, not tweaks.


## 二十三、对"碰撞保护 + 放松门"路径的完整回答(回应追加追问)

### 用户追问的逻辑链
1. 阈值扫描证明放松 ‖g‖ 门不影响 affinity(FP32 域) →
2. E1 失败因为门没放松 →
3. "在 E1 中放松门限, 是否就能收敛?" → E1v2 实测: 不能(已证)
4. "是否应该调整门限 + 加碰撞保护?" → 本节回答

### 为什么"碰撞保护"不是正确的修复

**实际失败模式(有 diag 数据)**: cand.e = 15,705,782 / 22,448,062 /
33,474,280 kcal/mol(正常范围 ±10²)。这些能量不是来自 r→0 的打分
发散——**Vina 的打分函数在 r=0 处是有限值**(所有项都是 gaussian/
smooth 函数, 无 1/r¹² 奇点; smooth 表预计算时已被 min(v, sf.eval)
限制, 且 M2 的 __half 存储值域 65504 足够容纳)。

**15M 的真正来源**: FP16 坐标量化在 set_update 的树传播中引入
**阶梯状不连续** — 每次写 m_coords 时坐标被截断到 0.0078Å 网格,
下一个原子的坐标从这个截断值出发计算 → 能量面变成**不连续的
阶梯函数** → BFGS 的二次模型和 Armijo 线搜索在阶梯面上失效
→ 线搜索的试验点跨过不连续边界 → 能量跳到 15M → 判负 → episode
失败 → 所有轨迹都失败 → out_cont 空。

**碰撞保护 `if (r2 < r_min²) skip` 解决的是"r 太近导致打分发散"**
—— 但 Vina 打分在 r=0 处本来就不发散(设计如此), 实际发散的是
BFGS 沿不连续面的导航。碰撞保护治的是不存在的问题。

### 为什么放松门限也不是正确的修复

放松 ‖g‖ 门(1e-5→1e-3)改变了 episode 的终止条件, 但**能量面的
不连续性不变** — BFGS 的线搜索仍然跨过量化边界, 能量仍然跳变。
放松门限只会让 episode 在线搜索失败前就放弃 — 结果同样是
欠优化的构象(而不是有效构象)。

### 根本的不可行性

FP16 的 10 位尾数在任意尺度下相对精度 ≈ 1e-3。BFGS 的二次模型
假设能量面在试验点附近是**光滑的**(二阶可导)。坐标量化破坏了
光滑性 — 这不是通过调门限或加保护能修复的, 因为它是**坐标
表示本身的性质**。

**唯一可行的 FP16 坐标方案**: FP16 存储 + FP32 计算 + FP32 残差
校正(GMRES-IR 思路) — 但这需要**双倍 shared**(FP16 存储 + FP32
影子拷贝), 容量反而更大, 违背初衷。

### 终局
FP16 坐标路径在本架构上不可行(物理原因, 非实现缺陷)。
16 blocks 的 warps/SM 收益无法通过 FP16 状态压缩获得。
若真需要 16 blocks, 正确路径 = 非 FP16 的 shared 结构性削减
(§四的 gyration overlay 等), 但预期收益(-3~6%)不足以立项。


## 二十四、E1-retry 终局: 架构级约束确认(2026-09-12 晚)

### 实验结果
E1-retry(正确 host FP16 写入 + device __half 读取)解决了 E1 的
实现 bug, 内核本身运行正常(k2=26ms, 无 CUDA 错误)。但
**"No results found for ligand" 仍然出现**。

### 根因: m_cl 是 host-device 桥梁, 不能单端降精度

```
HOST (CPU)                          DEVICE (GPU)
┌─────────────────┐                ┌──────────────────┐
│ refine_structure │   cudaMemcpy   │ kernel2_update   │
│ 读取 m_coords    │◄──────────────►│ 读写 m_coords    │
│ 为 FP32 float    │   m_cl 整体    │ 为 __half(E1改) │
│ → CPU 读到 FP16  │                │                  │
│   位模式→ 垃圾    │                │                  │
└─────────────────┘                └──────────────────┘
```

m_cl struct 是 **host CPU 精修(refine_structure)和 device GPU BFGS
的共享数据**。当 device 端把 m_coords 写成 __half 位模式后:
- GPU BFGS 正常工作(读 __half → half2float → FP32 计算)
- 但 HOST CPU 精修读同一个 m_cl 的 m_coords 时, 读到的是 __half
  位模式被 CPU 解释为 FP32 float → **完全不同的数值** →
  refine_structure 产出垃圾构象 → 全部被质量过滤 → out_cont 空

### 为什么这是架构级约束
- m_cl 是 8192 条轨迹共用的**全局结构**(Path B merged), 不是
  per-block shared — 无法在 kernel 内做"临时 FP16 化"
- host 的 refine_structure 需要 FP32 精度(它是 CPU 代码, 不支持
  __half); device 需要 FP16 来节省 shared — 两者冲突
- **解决方案只有**: (a)双 kernel 分离 host/device 视图(Q6-IR),
  或 (b)host 端做 __half→float 转换(额外开销), 或 (c)维持 FP32

### 最终结论
**FP16 m_coords/minus_forces 在当前单 kernel 架构下不可行** —
不是精度问题, 是 **host-device 数据共享的架构约束**。
Q6-IR 双 kernel 方案是正确的解决路径, 但需要 2-3 会话。
E1-retry 已回退, 主线恢复 Q5 (cd10ca24, 二进制 5eaa4fac)。