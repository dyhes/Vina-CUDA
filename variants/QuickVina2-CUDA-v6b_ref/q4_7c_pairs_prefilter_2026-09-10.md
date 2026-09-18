# Q4-7C 设计：INTRA pairs 预筛（active list）— 2026-09-10

> Q4-7 执行序头名（A 已证伪）。

## 一、现状与成本

`eval_interacting_pairs_deriv_update_optimal` 每 eval 遍历
`pairs->num_pairs`（1pxn ~150 / 1u1b ~1500 / 3ag9 ~4000 量级），每对：
3 loads(type,a,b) + 6 loads(坐标) + r2 计算 + cutoff 判定；
超限对直接跳过。**r2 超限的对占比 60-90%**（柔性配体多数原子对彼此
远离），这些对每次 eval 都白付 r2 计算与两坐标 load。

## 二、方案：episode 级 active list（保守阈值）

1. **首 eval 后构建**：BFGS 每 episode 的第一次 eval 之后，用当前构象
   计算每对 r2，把 `r2 < (cutoff + margin)²` 的对索引写入 shared
   active list（margin 覆盖 LS 期间坐标漂移：step ≤ 1Å 量级，
   margin 取 2Å → r2 阈值 (cutoff+2)²）
2. **episode 内后续 eval 只遍历 active list**（通常 10-40% 长度）
3. **保守性论证**：LS 内单步位移 ≤ stp·|d|∞，BFGS step 归一化后
   ≤1Å；margin 2Å 意味着一个对只有在"从 >cutoff+2 距离进入
   <cutoff"时才可能漏算——需要相对位移 >2Å/步，物理上被
   minimize 步长排除；且 episode 结束即重建列表，误差不跨 episode
4. **正确性等价性**：漏选只会发生在"从远处拉近"的对，而其在进入
   cutoff 前贡献为 0（curl 前 e>0 才平滑，超限 e=0）；首个进入
   cutoff 的 episode/eval 会因 margin 而捕获。**数值差 ≤ MC 噪声**，
   冒烟用同 seed 逐位对比验证

## 三、实现要点

- shared 预算：active list 索引 int16 × num_pairs 上限 2048 = 4KB
  ——**超预算**！改为：不存列表，改为在原遍历中加
  `if (pair_r2_cache[i] > thresh) continue;` 的 **r2 cache 数组**
  （float ×2048 = 8KB 也超）。
  → **最终形态：把 active list 放 global（每 traj 一份，episode 重建），
  LS 内只读**；或缩小 pairs 本身——**host 预筛**：CASF 配体的 pair 表
  在 host 生成时即可按"晶体构象 + 最大内摆幅"保守裁剪？不行，
 构象自由度大时不可预筛。
- **修正实现**：global active list（int16，8192 traj × cap 512 = 8MB，
  超；改为 per-traj cap 256 → 4MB）——显存可承受，但 kernel1 后需
  一遍构建 kernel……复杂度上升。
- **降级方案（推荐先做）**：不加列表，**仅调整遍历顺序**：把 pairs 按
  host 侧晶体构象 r2 降序排序（近对在前），利用 early-exit 无效果——
  无分支预测收益。→ 真正低风险版本：**episode 内 r2 缓存进 shared 的
  压缩位图**（1 bit/pair × 2048 = 256B shared！）：首 eval 时把超限对
  的 bit 置 1，后续 eval 先查 bit，连续 2 个 episode 未激活则置 0 重试
  ——无显存增量、shared +256B（预算内：12204+256=12460 ≤ 12496）✓

## 四、验证

- 同 seed 逐位对比（需先加 --seed 透传）
- 六档冒烟：3ag9/1u1b 应显著（pair 多），1c5z 无感
- ncu：INTRA 段指令数下降
- CASF 全量认证

## 五、收益估计

1u1b/3ag9 eval -30~50%（超限对跳过），k2 -10~25%（这些靶点）；
全量 median 收益取决于 pair 分布，估计 -5~15%。
