# 进度条线程治理（v4_8 → v4_8b）— 2026-09-10

## 决策：进度条改为**默认关闭、`-DSHOW_PROGRESS_BAR` 显式开启**

（不是"带标识才打印"以外的第三种方案；也不是物理删除源码。）

## 理由

1. **测量独立性**：runner 统计只依赖 stdout 的结构化行（E2E_*、GPU_KERNEL*_TIME）
   与输出文件，与 UI 动画完全无关——UI 线程对统计无贡献，留在默认构建里
   只会持续往所有测量里掺 50-100ms 噪声
2. **防遗忘**：显式关闭的默认值意味着"忘了传宏"的后果是**测量更干净**而不是
   **被污染**；方向正确。若默认开、靠加宏关闭，忘传宏 = 全量数据掺 0.5s 级
   UI 噪声且难以察觉
3. **家族背景**（已逐一 grep 实证）：`print_process`+`sleep(1)`+`join` 是
   Vina-CUDA→Vina-GPU→QuickVina 系一脉相承的祖传代码（Vina-GPU L175/L637、
   Vina-GPU2 L31/L682、Vina-CUDA L46/L972、QuickVina-W L254/L703 全部同构），
   唯 Uni-Dock 重构掉了（parallel_progress 仅 26 行桩 + verbosity 门控）。
   我们与 Uni-Dock 对齐其行为
4. **不改 NDEBUG**：spawn/join 原包裹在 `#ifdef NDEBUG`，Makefile 固定
   `-DNDEBUG`；不能靠去 NDEBUG 关进度条（会激活全库 assert，改变性能路径）

## 改动（main_procedure_cu.cpp 三处）

```cpp
#if defined(NDEBUG) && defined(SHOW_PROGRESS_BAR)
    std::thread console_thread(print_process);
#endif
// L786 abort 路径 join / L835 收尾 join 同条件
```

`status` 全局枚举（L445/784/833 的赋值与 L71/72/83 的读取）保留不动：
无 UI 线程时它只是写给自己看的标志，零成本。

## 验证

- 默认构建（`-DHOST_PROBE`）：`objdump` 证实 `pthread_create` 仅剩 1 个调用点
  （CUDA runtime 内部线程 `__cudart1088`），console spawn+join 已编译掉；
  `print_process` 符号保留但无引用
- 交互构建（`+ -DSHOW_PROGRESS_BAR`）：`strings` 含 "Perform docking"，UI 恢复
- smoke（默认构建，对比 v4_8 认证基线）：

| target | v4_8（100ms 量子）| v4_8b（无 UI 线程）|
|---|---|---|
| 1pxn | 0.424 | **0.388** |
| 1c5z | 0.326 | **0.265** |
| 1u1b | 0.923 | **0.835** |

  affinity 不变（-8.9/-5.9/-7.5），收益 = join 残留 ~36-61ms，与
  "期望 50ms、最坏 100ms" 的 U(0,100ms) 预测吻合

## 遗留

- v4_8 CASF 认证数字（rt 0.640 / e2e_dock 0.412）基于 100ms 量子构建，
  仍有效；v4_8b 为最终测量基线，如需更干净的论文数字可重跑一次全量
  （预期 rt ~0.60 / e2e_dock ~0.37）
- 其余家族成员（Vina-GPU 2.1 基线）如做同机对比，建议打同款 opt-in 补丁
  保证双方口径一致
