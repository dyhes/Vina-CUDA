# Q4-7 立项设计：LS eval 精简（算法/kernel 层）— 2026-09-10

> 依据：stage_timer（LS 占 k2 60-88%）+ ncu（占用率 14.1%，wait 26% +
> long_scoreboard 21%，No Eligible 82%）。目标：降低 LS 内每次 eval 的
> 成本与次数，或提高其延迟隐藏度。

## 一、LS 内一次 eval 的解剖（m_eval_deriv_update）

```
set_update(c, ...)                     坐标重算：树遍历 48 节点 × local_to_lab
ig_eval_deriv_update_optimal(...)      GRID：逐原子 g_evaluate_update（插值）
  + warp 归约
eval_interacting_pairs_deriv_optimal  INTRA：逐 pair p_eval_deriv
  + warp 归约
POT_deriv_update(...)                  force→change（树遍历再一次）
```

每个 BFGS episode 调用 evals = 1(init) + Σ_LS(count)；1pxn 全局
evals=1.9M / 8192 traj = 231 evals/traj，EVAL_INIT 只占 1/ep。

## 二、候选优化（按风险×收益排序）

### A. set_update 增量坐标重算（★★★ 最高收益）
- 现状：LS 每次试步 `x = xp + stp*d` 后 set_update **全树重算**所有原子
  坐标（48 节点 local_to_lab）。但试探步只改 1 个扭转角（部分步骤）或
  position+orientation（根节点）——**未受影响子树的原子坐标不变**
- 方案：set_update 增量化——输入改为"变化的节点集合"（root 扭转 → 全树；
  child 扭转 → 该节点及其子树），仅重算受影响子树
- 收益估计：1pxn 树 6 节点，平均子树 ~1/3 → set_update -60%；set_update
  占 eval ~25%（树遍历×2：set + POT）→ eval -15%，LS 段 -15%
- 风险：中。需维护"脏节点"标记；increment 后原子坐标增量传播已有
  rigid 树结构可复用（parent 链）。**必须六档冒烟 + 双 seed 位比**

### B. g_evaluate_update 的 4 角点 gather 合并（★★）
- 现状：每次插值 8 个角点 × 2 分量（e+dor），地址计算 `x0 + m_i*(y0+m_j*z0)`
  产生 4 条独立 load 链（ncu long_scoreboard 21% 的主要来源之一）
- 方案：三线性插值改为 **pre->m_data 与 grid m_data 打包布局**（SOA→AOS
  调整）使 8 角点落在连续 128B；或用 `__ldg` + float4
- 收益：long_scoreboard 减半 → eval -10~15%（若 LS 是访存延迟受限）
- 风险：低-中（布局改动波及 kernel1 写端）

### C. INTRA pairs 预筛（cutoff 预计算）（★★）
- 现状：每 eval 遍历全部 pairs（2048 cap），每对算 r2 再判 cutoff；
  但**原子对列表在 LS 过程中静态**，只有坐标变——距离超 cutoff 的对
  在构象小扰动下大多持续超限
- 方案：episode 初始化时按 r2 排序/分桶，LS 内跳过明显超限桶；
  或用 Kepler 式 compact 列表（首 eval 时生成 active list，N 步刷新）
- 收益：3b68/3ag9 类大 pairs 数靶点 eval -20~40%；小配体收益小
- 风险：中高（动态列表 + 正确性敏感：cutoff 边界原子必须保守纳入）

### D. evals 次数削减（算法层，谨慎）
- Lewis-Overton LS 的 count≤8 上限已存在；可试初始步长预测
  （用上一 episode 的最终 stp 作初始，减少 1-2 次试步 eval）
- 收益：evals -10~20%；风险：改变搜索轨迹 → **Top-1 可能漂移**，
  必须全量认证 + 多跑中位口径

## 三、验证协议（每项独立立项，控制变量）

1. 六档冒烟 5 例：MODE1 与 Q4-6 **同 seed 对比**（需加 --seed 透传，
   当前 auto_seed；A/B/C 不改 RNG 路径，期望逐位一致）
2. stage_timer v2：LS_TOTAL/evals_n → **ns/eval 直接对比**
3. ncu 复测：wait/long_scoreboard 占比变化
4. CASF 全量：aff mean|Δ|（期望 ≤0.16 噪声带）+ Top-1 带内 + rt/e2e_dock

## 四、推荐执行序

**A → C → B → D**（A 收益最大且与 v11 已有树遍历经验衔接；
C 需 3b68/3ag9 类靶点验证；B 依赖 A 落地后的新 profile；D 改变
搜索语义，最后做且需多跑统计）。

预期合计：k2 -25~40%（若 A+C 兑现），e2e_dock -20~30%
（1u1b 型靶点受益最大），rt 再 -10~15%。
