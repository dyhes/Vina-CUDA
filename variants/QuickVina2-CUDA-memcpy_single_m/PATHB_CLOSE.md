# Path B（memcpy_single_m）收尾报告 — 2026-09-09

## 结论：Path B 已于 Sep08_1208 完成全量认证，无需再跑

### 认证数据（experiments/Sep08_1208_CASF-2016_v5_memcpy_single_m）
- 285/285 OK（此前 Aug01 rerun 唯一 FAIL 的 3bv9 在本次 OK，rt 1.35s）
- rt median **1.320s vs v4 1.520s = -13%**（smoke 预期 -18%，全量下 -13%）
- HtoD **0.215s → 0.000s**（289MB→37KB 的修复直接可见，账本列已齐全）
- k2 median 0.136s = v4 0.135s 持平（改动只在传输侧，符合预期）
- 亲和力 vs v4：mean |d| 0.159 / >0.5 共 18/285 / max 1.40
  对照 v4 自身 run-to-run 噪声（mean 0.162 / 23 / 1.20）→ 差异在 MC 噪声内 ✓

### Top-1 口径提醒（写论文时注意）
v4 家族历次全量 Top-1(<2Å)：171/172/173/174/174/175/181/184 —— 极差 13 个。
- 论文引用的 180/285 是这个噪声带的上沿；
- v4.5=175、v5_memcpy=171 都在带内；
- **建议论文口径改为多次运行的中位数（174）±极差**，否则 180 vs 171 这类
  差异会被审稿人用同样的噪声反驳。

### Path B 状态：CLOSED
合并建议：memcpy_single_m 变体 + v4.5 三修复互不冲突（传输侧 vs 计算侧），
下一个 milestone 可直接 memcpy_single_m ⊕ v4.5 合并出 v6 基线。

## 附：host 侧 wall 构成（v5_memcpy 全量中位数）
rt 1.320 = GPU 总账 0.158（k1 0.010 + k2 0.136 + dtoh 0.007 + memset 0.004，htod≈0）
**host 侧未入账 1.162s = 88% wall** —— 下一优化主战场在 host（历史最佳 batch_pool2
rt_med 1.230 仍未动它）。
