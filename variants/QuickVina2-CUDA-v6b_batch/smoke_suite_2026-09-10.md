# 冒烟用例集（smoke suite）v1 — 2026-09-10 定稿

> 用途：后续每次代码改动（kernel/host/runner）后快速验证正确性 + 立即读出
> 性能影响面的标准用例集。全部来自 CASF-2016 coreset，规范调用见文末。

## 选样方法（一句话）

以全量 285 配体拟合的复杂度模型 `log k2 = -7.59 + 1.33·log(atoms) + 0.69·log(tors)`
（R²=0.97，系数为本次实验数据实测拟合，v4 与 v4_8b 两代独立数据系数一致：
-7.62/+1.36/+0.69，说明刻画的是程序结构而非单次噪声）的预测值作一维
复杂度轴，在其 p0/p20/p40/p60/p80/p100 取靶。

## 六档用例表（已跑通，v4_8b 探针构建）

| 档 | 靶点 | atoms | torsions | k2 (s) | E2E_DOCK (s) | 画像 |
|---|---|---|---|---|---|---|
| p0 | 1c5z | 13 | 1 | 0.016 | 0.265 | host-bound（GPU 占比 ~8%）|
| p20 | 3f3c | 16 | 4 | 0.059 | *未跑探针* | 低端过渡 |
| p40 | 1pxn | 25 | 5 | 0.127 | 0.388 | 平衡区（prep≈GPU）|
| p60 | 3p5o | 31 | 5 | 0.170 | 0.428 | 过渡（GPU ~40%）|
| p80 | 3b68 | 34 | 9 | 0.255 | 0.516 | GPU 渐主（~50%）|
| p100 | 3ag9 | 91 | 36 | 1.981 | 2.257 | GPU-bound（88%，prep 仅 12%）|

（p20 的 3f3c 后续需要时补跑探针；正确性锚点用 MODE1 值随时可验）

### 正确性锚点（MODE1 affinity，v4_8b 构建）

```
1c5z -5.9 | 1pxn -8.9 | 3p5o -9.0 | 3b68 -11.4 | 3ag9 -4.7 | (1u1b -7.5/-6.7)
```
任何改动后这些值偏离 v4 噪声带（mean|Δ|>0.5 的靶点数应 <25/285 全量口径；
smoke 上单点偏离 >0.5 即需排查）。

## 关键结构发现（用例集已实证）

1. **host prep 底座 ~350ms 与配体无关**：A+C+D+E 段六档实测
   359/351/338/~340/~330/~330（p60-p100 略降因 start 段波动），受体侧常数
   ——screening 摊薄论断成立；3ag9 上 prep 只占 15%
2. **F 段（ric+kernel_monte 等待）≈ k2**：六档全部吻合（如 3ag9 1982 vs 1981ms）
   —— host 已完全不遮蔽 GPU
3. **GPU 占比曲线**：8% (p0) → 40% (p40-60) → 50% (p80) → 88% (p100)。
   p60-p80 是 host/GPU 交棒区
4. **极端尾部**：3ag9（91a/36t）k2=1.98s，超出 1u1b 3.5×——torsions>16 的
   稀有尾部（n=8）应单列讨论，任何 kernel 优化在该桶的绝对收益最大

## 标准测试流程（后续改动沿用）

```bash
# 1. 构建
cd variants/QuickVina2-CUDA-v45 && export PATH=/usr/local/cuda-11.8/bin:$PATH
make OPTION="-DHOST_PROBE"        # 探针版（测量用）
# 或 make OPTION=""               # 无探针（正式跑分用，探针零开销设计）

# 2. smoke（GPU 串行规则：绝不并行，逐个跑）
for T in 1c5z 1pxn 3p5o 3b68 3ag9; do
  python3 - <<PY  # 算 box center
...（见 /tmp/v9_*.log 的调用记录或 runner）
PY
done
# 3. 读数：MODE1（锚点）、E2E_DOCK_S、E2E_WARM_S、GPU_KERNEL2_TIME、PROBE 分段
# 4. 全量：python3 tools/run_experiment.py --variant v4_8b --dataset CASF-2016
```

## 回归模型来源说明（防止误引）

- 模型为**本次研究从实验数据拟合**（v4_8b 与 v4 两次独立全量），非 Vina
  论文或任何代码内置。Vina 原论文只给搜索算法框架，无此类 cost model
- 用途仅限：①选样分档 ②解释 k2 随复杂度的伸缩 ③论文中作经验观测
  （empirical observation）呈现，注明拟合数据来源；不可当作理论推导使用
- 弹性系数的物理意义：k2 对 atoms 弹性 1.33（单 eval 成本，超线性），
  对 torsions 弹性 0.69（DOF 维数），两者乘法耦合

## 待办

- [ ] 3f3c（p20）补探针跑一次，补全六档分段表
- [ ] 六档数据入论文时：散点图（全量 k2 vs 模型预测线）+ 六档分段堆叠图
- [ ] smoke 结果记录模板（每列：target/构建/MODE1/E2E_DOCK/k2/prep/F/G）
