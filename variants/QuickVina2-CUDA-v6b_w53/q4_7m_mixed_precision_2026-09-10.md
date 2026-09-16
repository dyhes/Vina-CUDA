# RTX 4090 混合精度能力 × BFGS 算法层改造分析 — 2026-09-10

> 应要求先分析硬件能力，再对照 BFGS 流程找算法层改造点。
> **历史 trap 修正前提**：此前 FP16/BF16 结论（m1a 失败、FP16 敏感）
> 产生于 v4 系（occupancy 6%、寄存器 255、ncu 旧口径），计时与瓶颈
> 定位体系均与 Q4 现状不同，需在 Q4-6 基线上重新评估。

## 一、RTX 4090 (sm_89 Ada) 混合精度硬件能力

| 精度路径 | 峰值吞吐 | 说明 |
|---|---|---|
| FP32 CUDA cores | 82.6 TFLOPS | 现状基准 |
| FP16 CUDA cores（half2 打包）| **165 TFLOPS（2×）** | 不需 TC，普通 FMA 即可 |
| FP16 Tensor Core（FP16 累加）| 330 TFLOPS | mma.m16n8k16 |
| FP16 TC / FP32 累加 | 165 TFLOPS | 精度安全的 TC 路径 |
| BF16 TC | 165 TFLOPS | 同 FP16 形状，动态范围更大 |
| TF32 TC | 82.6 TFLOPS | 输入 FP32，仅尾数截断，10 位尾数 |
| FP64 | 1/64 速率 | **ncu 发现 kernel 里有 17.8M FP64 指令——先揪出这个** |

硬件要点：
- 4 TC/SM，shared 100KB/SM；mma.m16n8k16 需要 16×16 tile 的规整 GEMM 形状
- **half2 打包的 CUDA-core 路径不需要 TC**，也能拿 2×，且无 tile 形状约束

## 二、BFGS/LS 流程的算力账（Q4-6 实测语境）

以 1u1b（n = 6+16 = 22 维）为例，每 BFGS 迭代：

```
scalar_product ×2        2n ≈ 44 FLOPs
H·y (dense mat-vec)      n² ≈ 484 FLOPs        ← 最大矩阵运算
rank-2 H 更新            2n² ≈ 968 FLOPs        ← 最大矩阵运算
合计 BFGS 数学           ~1.5 kFLOP/iter
```

对比：每迭代 ~5.5 次 eval，每次 eval = 树遍历 + 61³ 插值 gather +
pairs 遍历 ≈ 数千次访存操作、几百 FLOPs——**eval 是成本主体且是
访存/gather 型**，BFGS 数学只占每迭代算力的 ~0.1%。

## 三、结论：混合精度的可行落点（重新排序）

### 结论 1：TC 化 BFGS 数学（旧主线）在本 kernel 是伪命题
- 数学占比 0.1% × 利用率 14%，TC 吞吐再高也无意义（Amdahl）
- mma 需要 16×16 tile；n=22 的 GEMV 要 pad 到 16×32 tile，浪费过半
- v4 时代"SM 11% 下 TC 无效"的判断在 Q4 依然成立——**瓶颈不在算力**

### 结论 2：真正可行的是 half2 CUDA-core 路径，落点有三处

#### 2a. GRID 插值数据的 FP16 存储（★ 最高性价比）
- 现状：grid m_data 是 8 个 FP32/格点（Q4-1 后 12.3MB）
- FP16 化：**显存/带宽减半**（12.3→6.2MB），插值 gather 的
  long_scoreboard 停顿直接受益；fp16_grid_v5 变体证明写入端可行
  （kernel1 已有 __float2half 管线残留）
- 精度：插值用的亲和能值域 ±几十 kcal/mol，FP16 相对精度 2^-11
  ≈ 5e-4——**远高于 MC 噪声（affinity |Δ|~0.16）**，风险极低
- 这是**存储精度**不是计算精度，不涉及 LS/BFGS 语义
- Q4-6 上 61³×17 grid 常驻 L2 的 footprint 减半 → L2 命中率进一步提升

#### 2b. pairs p_eval_deriv 查表的 FP16（★）
- pre->m_data LUT（fast/smooth 数组）同样可 FP16 存储：查表插值
  精度损失与 2a 同量级，收益同上（LUT ~6.3MB）

#### 2c. eval 内数学的 half2（★ 需实验）
- p_eval_deriv 插值系数乘加、pairs 力累加等逐元素乘加可 half2 化
- **但减法敏感**：coords_b - coords_a 大数相减得小 r 向量，
  FP16 的 10 位尾数会放大相对误差——**坐标本身必须保 FP32**，
  仅 r2 之后的多项式求值可降
- 风险中等，需要 3b68/3ag9 全量精度验证

### 结论 3：BFGS 本体的"精度分层"（旧 m0_analysis §5.2）在 Q4 的意义
- H 矩阵/线搜索判定保 FP32（状态量，误差会跨步累积）
- 上述 2a/2b 都是**数据存储降精度 + 计算保 FP32**的分层模式——
  正是 m1a 失败后总结的正确路线（m1a 错在把归约状态也降成 BF16）

### 结论 4：TC/mma 的真实机会在 v6-batch 之后
- batch 合并后每 block 处理多配体 → H·y 变成 batched GEMV
  （16 个轨迹 × n=22 的 H 打包 = 22×352 矩阵），才勉强够一个 mma
  tile；且此时 GPU 被填满，eval 变成瓶颈——TC 才有讨论意义
- **顺序：先 batch（Q4-8），再 TC（Q4-9）**，与 7 月"先 fusion 后 TC"
  的判断一致但理由更新

## 四、推荐的 Q4-7 续作（新编号 Q4-7M）

| 步骤 | 内容 | 预期 | 风险 |
|---|---|---|---|
| M1 | 排查 kernel 内 FP64 指令（17.8M）来源，消除（double 字面量/ powf 等）| k2 直接受益 | 低 |
| M2 | grid m_data FP16 存储 + 插值读端 half2 | gather 带宽减半，k2 -5~10% | 低（存储精度）|
| M3 | pre LUT FP16 化 | 同上 | 低 |
| M4 | p_eval_deriv half2 数学 | 再 -5% | 中（需精度全量验证）|

验证协议：六档冒烟（同 seed 逐位不要求——插值路径变化会改变数值，
但 affinity 分布须在噪声带）+ CASF 全量（aff ≤0.16 + Top-1 带内）+ ncu
（DRAM/long_scoreboard 前后对比）。

**注意**：M2 改变数值路径（FP16 截断），MODE1 期望在噪声带内但不逐位
一致——这是与 Q4-7C 最大的验证差异。
