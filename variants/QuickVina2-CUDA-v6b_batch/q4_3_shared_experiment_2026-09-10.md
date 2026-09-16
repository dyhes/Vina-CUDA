# Q4-3 shared 内存瘦身实验：尝试 → 回滚 → 结论 — 2026-09-10

> 目标：ncu 判定 k2 受 shared 限制（14.8KB/块 → 6 blocks/SM，占用率 12.1%）。
> 尝试把 m_coords/minus_forces（3KB，按 MAX_ATOMS=128 静态分配）改为按真实
> 可动原子数的 dynamic shared，把 shared 降到 ~11.8KB → 8 blocks/SM。

## 一、实现（kernel2.cu + kernel2.cuh + main_procedure_cu.cpp）

- 两个 shared 结构 → `extern __shared__` carve-out，指针化（`m_coords.` →
  `m_coords->`），kernel_monte 增加 nat_movable 参数，launch 传动态大小
- 16B 对齐修正（minus_forces offset 向上取整到 4-flo 边界）

## 二、结果：回滚

| 指标 | 结果 |
|---|---|
| 功能 | **破坏**：首构象 cand.e 0.0→5.6，angle_to_quaternion ERROR，k2 143ms→2776ms（20×）|
| 判定 | shared 结构在 LS/BFGS 内有超出"真实原子数"范围的隐式依赖
       （m_coords 由 set_update 每步重写、get_heavy_atom... 遍历、以及
       跨步残留语义），按真实数裁剪破坏了这些不变量 |

回滚后复验：MODE1=-8.9 / K2=0.143 / 0 ERROR ✓（基线恢复）

## 三、结论与修正后的认知

1. **shared 瘦身不是"改两个数组"能完成的**——shared 里 14.8KB 有隐式语义
   （整结构拷贝、跨步状态残留），必须整体 design-first（重排 kernel 内存
   布局，明确各 shared 变量的生命周期）。列入 Q4 kernel 层立项的设计输入。
2. **occupancy 提升的另一条腿仍有效**：寄存器 128 顶满（`-maxrregcount`）
   与 shared 双重约束；shared 需整体重设计，寄存器上限则可独立试
   （`__launch_bounds__(32, N)` 编译期实验）——**下一步先做寄存器扫描**，
   因为它不改语义、纯编译指令
3. 4090 shared/SM=100KB：7 blocks 需 ≤14.2KB/块、8 需 ≤12.5KB、10 需 ≤10KB

## 四、当前 Q4 状态

| 项 | 状态 |
|---|---|
| Q4-1 真实 dims（136MB）| ✅ 已合入（显存卫生）|
| Q4-2 stage_timer v2 + K2WALL | ✅ 已合入（验证工具）|
| ncu 画像 | ✅ 占用率 12.1%，shared+regs 双约束，LS wait/scoreboard 主停顿 |
| Q4-3 shared 瘦身 | ❌ 尝试后回滚（语义破坏），转 design-first |
| Q4-4 寄存器/占用率扫描 | **下一步**（纯编译实验）|
