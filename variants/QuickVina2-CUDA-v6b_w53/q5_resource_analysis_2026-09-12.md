# Q5 资源使用分析: 寄存器 / shared / local — 2026-09-12

> 基于 Q5(tag Q5 = 344d9f52)构建的 cuobjdump -res-usage + PTX .shared
> 逐变量分解 + ncu 交叉验证。目标: 判定当前资源瓶颈与 8-blocks 路径。

## 一、总账(每块 64 线程)

| 资源 | 每块用量 | SM 预算 | 块数限制 |
|---|---|---|---|
| 寄存器 | 64×128 = 8,192 | 65,536 | **8** |
| shared | 12,684 + ~1,024(reserve) = 13,708 | 102,400 | **7** ← 瓶颈 |
| warps/SM @7 块 | — | — | 14(27.8%, ncu 实测吻合) |

**shared 是唯一的块数瓶颈**(寄存器还有 1 块余量)。要 8 块(16 warps/SM,
+14%): static shared 需 ≤ 11,776B(每块 1KB driver reserve 必须计入),
即再削 **908B**。

## 二、shared 12,664B 逐变量分解(PTX 实测)

| 字节 | 归属 | 性质 |
|---|---|---|
| **3,216** | m_gpu_ligand(rigid 树拷贝: children_map/parent/atom_range/origin/**orientation_q 608B**/axis) | orientation_q 是**每轨迹可变状态**(set_update 写), 其余 ~2.6KB 只读 |
| **1,536+1,536** | m_coords + minus_forces [128][3] | 每块可变工作区(BFGS 必需) |
| **1,224** | 输出临时簇 | |
| **1,024** | gyration_radius 的 acc[128]+counter[128] | **稀有分支**(仅 which==0 的振幅变异路径)但常驻 |
| 912 | POT position/orientation_derivative_tmp[38][3] | Q4-5 遗产 |
| ~2,700 | rilc/scalar/ig/eval_interacting 各处 out_type/change 临时(228-64B × ~12) | 作用域重叠, union 风险高 |
| 260 | q4_skip_row(256)+q4_epoch_i(4) | Q4-7C |

## 三、寄存器与栈

- **128 regs/thread** = Makefile `-maxrregcount=128` 主动上限。
  7 块 × 8,192 = 57,344 / 65,536 → **有 8,192 个寄存槽空闲**;
  提到 144/thread(7×9,216=64,512)仍 7 块, 可消化部分栈溢出
- **STACK 9,984B/线程 × 64 = 624KB/块**(32t 时 312KB)——64t 使每 SM
  活跃 local 翻倍(≈4.5MB), 这就是 Q5 剖面 local_st 1.32G、DRAM
  2.6→6.9GB 的来源
- 栈构成: LS 回溯状态 + output_type/change 临时 + visited_cl 4.3KB
  (static 保留, 运行期**从不饱和**——full 需 220+ add, 每轨迹仅 ~12)

## 四、8-blocks 路径评估(需 -908B)

| shave | 字节 | 风险 |
|---|---|---|
| gyration acc/counter [128]→[91](CASF max 91 原子) | -296 | 低(独立数组, 与 m_coords 的 128 无耦合) |
| **gyration 1024B 与 eval 系临时 overlay**(互斥执行: mutate 相 vs BFGS 相) | -1,024 | **高**(手工 union, 须证明作用域互斥+插桩验证) |
| m_gpu_ligand 拆分: 只读树数据 ~2.6KB → __constant__(每配体一次 h2d) | -2,600 | 中(需新常量符号+绑定, TU 内; orientation_q 留 shared) |

**组合**: (c)只读树→constant(-2,600) 单独就够 8 块(12684-2600=10,084,
甚至 12,800/12,684→…→ 可达 9-10 块? 受寄存器 8 块封顶→正好 8);
(a)+(d)组合(-1,320)也够且不动数据流。**推荐 (a)+(c) 或 (a)+(d)**,
不推荐单做 (d)。

**收益核算**: 8 块 @64t = 16 warps/SM(33.3%, +14% vs 现状 27.8%),
S4a-2 实测 14→27.8% 带来 k2 -17%(大配体), 但那是 0→2 warps 的
质变; 14→16 warps 是量变, 且 barrier 税(9.1%)随块内同步不变——
**预期 k2 -3~6%(大配体), 不是 -17% 的重复**。

## 五、结论

1. **当前瓶颈 = shared(7 块)**, 寄存器有余量(8 块)、栈可容忍
2. 8-blocks 的现实路径存在(三种组合, 推荐 mis/树常量化或 gyration
   overlay), 但收益只有 +14% warps → **k2 预期 -3~6%, 非重复性大赢**
3. 栈 9.8KB/线程是 local 流量(1.32G)的来源, 但 S1 证据表明消除它
   在 latency-bound 内核上不付费
4. **资源侧结论: 不存在"再砍一刀就大赢"的单点**; Q5 的资源布局已
   与其执行模型(1 轨迹/块 × 2 warps)匹配。真正的下一步在
   v6-batch(吞吐场景)或论文, 而不是继续抠 shared 字节
