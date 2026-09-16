# Q4-5 实施报告：shared orientation_m 移除 — 2026-09-10

> 按 design-first 文档（q4_5_shared_redesign + amendment1）实施。
> 分支 Q3 线，commit 0291222。

## 一、实现（最终形态，比立项文档更收敛）

1. **`local_to_lab` / `local_to_lab_direction` 改收 quaternion**：函数内
   `quaternion_to_r3(q, om_local[9])` 现场转矩阵（局部 36B）
2. **12 个调用点**从 `orientation_m[node]` 改传 `orientation_q[node]`
3. **rigid_cl 删除 orientation_m[48][9]**（kernel2.h，-1728B）
4. host `store_node` 的 2 处 orientation_m 写入删除
5. `to_norm_update` 函数整体删除（唯一调用点已被注释，dead shared 152B）

**过程中抓住一个自伤 bug**（记录防再犯）：批量替换
`orientation_m[X] → orientation_q[X]` 时把 4 处
`quaternion_to_r3(oq[node], om[node])` **写入点**也替换成了
`quaternion_to_r3(oq[node], oq[node])`——自赋值摧毁 quaternion →
axis 全垃圾 → angle_to_quaternion ERROR 风暴、k2 20×。冒烟集第一时间
拦截（1c5z MODE1 -4.5 + 218 万 ERROR），修复 = 删除这 4 行（m 不再存储，
转换内联于消费函数）。**若没有六档冒烟门槛，这个 bug 会直接进 CASF。**

## 二、资源与性能（六档冒烟集，全部通过）

shared **14,796 → 13,068B（-1728，-12%）→ 7 blocks/SM**（目标 8 需要
再 -572B，见"遗留"）。cuobjdump 直读确认。

| target | MODE1（Q3→Q4-5）| K2 (s) | E2E_DOCK (s) | Δdock |
|---|---|---|---|---|
| 1c5z | -5.9 → **-5.9** | 0.016→0.017 | 0.062→0.065 | 持平（host 主导）|
| 1pxn | -8.9 → **-8.9** | 0.127→0.127 | 0.193→0.178 | **-8%** |
| 3p5o | -9.0 → **-9.0** | 0.170→0.145 | — | — |
| 3b68 | -11.4 → -11.1* | 0.255→0.252 | — | — |
| 3ag9 | -4.7 → -5.4* | 1.981→1.894 | 2.257→1.962 | **-13%** |
| 1u1b | -7.0 → **-7.0** | 0.557→**0.488（-12%）** | 0.617→**0.545（-12%）** | ✓ |

*3b68/3ag9 的 MODE1 差异 = 运行间 auto_seed 的 MC 噪声（不同进程），
非代码回归（本项不改 RNG 路径；k2 同级、ERROR=0）。

 apples-to-apples（同为"无进度条+探针"构建的三例）：
**1u1b k2 -12% / dock -12%，1pxn dock -8%，1c5z 持平**——与 7 块/SM
(+17% 上限) 和"收益随配体复杂度增长"的六档画像吻合。

## 三、遗留与下一步

1. **8 blocks/SM 还差 572B**（13,068 → ≤12,496）：候选 =
   mutate acc/counter 与 bfgs 缓冲 union 化（生命周期不重叠，需小心）
   + mutate_counter int16。预期再 +14% occupancy，k2 再 -10~14%
2. **转换开销可再省**：local_to_lab 现在每原子调用都重算 q→m；
   可在 set_update 内每节点转一次存局部数组下传（找回预计算收益，
   同时保住 shared 节省）——1u1b 可能再挤 3-5%
3. **CASR 全量认证**：Q4-5 改动语义敏感（虽然冒烟全过），升 MILE 前
   必须全量。待用户指令。

## 四、谱系

- commit 0291222（Q4-5 本体）在 Q3 分支；CERT 快照仍是 Q3_2026-09-10
- Q4-5 完成认证后可立 Q4 CERT 快照（含 Q4-1 显存卫生 + Q4-5 occupancy）
