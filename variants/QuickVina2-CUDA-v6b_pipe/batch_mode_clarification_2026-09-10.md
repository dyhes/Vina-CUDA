# runner 调用方法 vs 代码侧 batch —— 澄清 — 2026-09-10

> **勘误（2026-09-10 晚）**：下文"层级 2 GPU 级批另立项"的说法**不准确**。
> GPU 级批 = **v6 已完成并认证的改造**（`milestones/v6` = `variants/QuickVina2-CUDA-batch_chunk`，
> 见 `handoff_2026-08-01_v6_verified.md`）：`kernel2_update` 单次 launch 处理 N 配体、
> block 映射 `(ligand_id, restart_id)`、动态 chunk（cap=64）、host 三段式消除 N-1 次
> deviceSync。screening 6.6×（380s→58s），redocking Top-1 等价。
> 当前 v45 主线**不含** v6 batch；正确的剩余工作是**把 v6 batch kernel 合并进 v45 线**
> （v4.5 三修复 + v7/v8 host 优化 + v6 batch = 最终统一版），而非新立项。
> 另：stream overlap / persistent kernel / 配体排序等"GPU 级批的延伸优化"
> 均已在 v6 时代试尽（49 个否定方向内），勿重复。

## 1. 目前 runner 是怎么调对接程序的（实证）

`tools/run_experiment.py` CASF 流程（L704-716）：**每个配体单独 fork 一个
对接二进制进程**：

```python
for tid in targets:                      # 285 次
    cmd = ["sudo", "bash", "-c",
      f'ulimit -s 8192 && exec "{binary}" '
      f'--receptor "{protein}" --ligand "{ligand}" '   # 单配体
      f'--thread 8192 --center_x {cx} ... --out "{output_file}"']
    r = subprocess.run(cmd, ...)         # 阻塞直到进程退出
```

要点：
- `--ligand <单个文件>`：走的是二进制的**单配体路径**（main.cpp L940-947：
  `ligand_names.push_back({ligand_name})`，vector 长度 1）
- 进程生命周期 = ctx 初始化 + 全部 prep + 对接 1 个配体 + 退出，×285
- runner 端自身的实测开销只有 ~0.04s/配体（此前已证伪 0.8s 初判）

## 2. 二进制其实**已有**多配体入口：`--ligand_directory`

main.cpp L674/L911-937：`--ligand_directory <dir>` 一次读入目录内全部 PDBQT，
`ms.push_back(parse_bundle(...))` 逐个解析 → `ligand_num = ms.size()`，
后续 per-ligand 循环（main.cpp L436 refine/write、main_procedure_cu.cpp
L766 kernel_monte 每 ligand_count 一次 launch）**本来就是循环多配体设计的**。
这是 Vina-GPU 系为 virtual screening 留的路径，不是我们要新写的。

但注意：**多配体 ≠ 一次 GPU 执行**。当前代码对每个配体仍然是：
per-ligand 分配 m_gpu/ric/randmaps/result 缓冲 → 单独 launch kernel_monte
→ `cudaDeviceSynchronize()`（kernel_monte 内部 L487）→ DtoH → 串行到下一个。
即"一个进程里顺序跑 N 个配体"，而不是 N 个配体同时上 GPU。

## 3. 所以"批模式"的两个层级，别混淆

### 层级 1：进程级批（runner/调用侧改，或直接用现有 --ligand_directory）
- 一进程 N 配体：ctx 初始化、grids/ar/pa/pre 的 prep、1.14GB 分配与挂账、
  streams 创建 —— **这些每进程一次的成本从 N 次变 1 次**
- 好处：CASF 285 靶点若同受体共享则省；但 CASF redocking 恰恰**每靶点
  受体不同** → grids/prep 不可复用，只有 ctx+warmup+分配策略部分可摊
- 这就是 prep 解剖报告里 D 项"batch 摊薄"的确切含义：**摊的是进程级
  常数（ctx+warmup ~180ms + VMM 挂账 208ms），不是 receptor prep**

### 层级 2：GPU 级批（必须改代码）
- 把 N 个配体的 kernel_monte 合并/并发 launch，或 kernel2 内多配体并行
- 收益：填满 GPU（单配体 8192 threads 只占 4090 的 ~15% SM 占用率）
- 这属于 kernel 优化主线，改动大，须 design-first

### runner 侧怎么配合
两条路：
- **A. 直接调用现有 `--ligand_directory`**：runner 改为按受体分组目录
  → 一次进程跑同受体全部配体。零代码改动，立刻生效；但输出解析要适配
  （输出目录 `<dir>_out/`，stdout 每配体一段 E2E/GPU 打印，runner 解析器
  需按段切分）
- **B. runner 不变，仅作为对照口径**：redocking 逐配体调用保留（每靶点
  独立进程反而隔离性好、崩一个不连坐）

## 4. 对当前测量口径的影响（重要）

- e2e_dock 是**二进制内**计时：批进程里 per-ligand 的 E2E_DOCK 会混入
  "第一个配体额外承担的 ctx/挂账" —— 需要新增 per-ligand 分段打点
  （每个 kernel_monte 前后）才能把批模式的摊薄效果量化出来
- runner 的 rt 口径在批模式下 = 全目录总时间，需除以 N 或按配体分段
- **CASF redocking 的公平性**：论文若声称 per-ligand 成本，应说明分母
  口径（独占进程 vs 共享进程）。批模式数字更好看，但 redocking 场景
  用户就是逐靶点跑的 —— 两个口径都报，注明适用场景

## 5. 结论（回答原问题）

- "批模式在代码侧还是 runner 侧"：**两者都有份**。进程级批 = 现成
  `--ligand_directory` + runner 改调用方式（A 路线）；GPU 级批 = 代码侧
  kernel 改造（层级 2，另立项）
- prep 350ms 里能被层级 1 摊掉的：ctx+warmup（~180ms，且已在 e2e_dock 外）
  + VMM 挂账 208ms（warmup 前置或进程内第一配体承担）≈ 每进程一次性
  ~390ms；**不能摊**的：受体 parse/precalc/fill（受体不同则必须重做）
- 因此 CASF redocking 下层级 1 收益有限（285 个不同受体）；真正的大头
  场景是 screening（同受体多配体）——这也与六档画像结论一致
