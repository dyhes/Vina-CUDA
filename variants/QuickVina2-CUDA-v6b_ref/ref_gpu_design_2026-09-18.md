# REF-G (GPU refine) 设计 — 2026-09-18

> 承接: 三真杠杆① (q53_v6b_batch_desC §七点七)。③ NO_REFINEMENT 已门禁否决
> (q53_v6b_noref_gate_2026-09-18.md)。本文档取代未落盘的 ref_gpu_design_2026-09-16.md
> (该 session 在 fork 验证后中断, WIP 草稿从未运行)。

## 一、WIP 遗稿验尸 (v6b_pd 工作树未提交 diff, 2026-09-18 16:00 实查)

结论: **死稿, 不缝补, 重写**。四个结构性缺陷:

1. **读空容器**: REF-G 段 (main_procedure_cu.cpp L979) 从 `outs[]` 打包 top-20,
   但 `outs[]` 的填充在 L1067 消费循环 — 打包内容是未初始化内存。
2. **无 DtoH 回读**: kernel 跑完后从 `h_pack` (host 侧、还是 HtoD 时的内容)
   写回 `outs[]` — "refined" 位姿 = 输入位姿, 整个 kernel 等于白跑。
3. **线程契约违反**: `refine_update_kernel` 只让 tid0 进 `rilc_bfgs_update`,
   但该机器是全块协作 (入口 `xp = *x` 全线程执行, 内部分区+同步) —
   tid0-only = R1a 死锁同款错误 (q53_w6cL_r1a: 分支内 __syncwarp)。
   另外 `__syncthreads()` 写在 `if (tid==0)` 分支内 = 发散同步 UB;
   `visited_init` 错放 tid0 (kernel2 模板中 visited 是 per-thread 局部)。
4. **未运行**: 二进制从未构建, 草稿从未冒烟。

## 二、机制发现 (决定 REF-G 的真正语义目标)

细读 CPU `refine_structure` (main.cpp L148-167):

```cpp
VINA_FOR(p, 5) {
    nc.slope = 100 * pow(10, 2*p);   // 100 → 1e10 软→硬 5 级
    quasi_newton_par(m, prec, nc, out, g, cap);
    m.set(out.c);
    if (nc.within(m)) break;          // 入盒即停
}
```

与 GPU 搜索的两个本质差异:

1. **精确势 vs 网格插值**: CPU refine 走 `non_cache` (配体原子 × 受体原子
   直接枚举, prec 样条查表), GPU 搜索走预计算网格插值。REF-G 若复用
   rilc_bfgs_update (网格势), 清除不了插值误差。
2. **软→硬坡度阶梯**: 第一轮 slope=100 (软墙), 入盒即停 — 多数盒内姿态
   只做一轮; 边界外姿态被软墙拉回。这是 5 轮存在的意义。
3. (附) 新 BFGS 状态 + 最多 5× BFGS 步数预算。

**关键推论**: 崖组靶 (1u1b/2zda/3ozt/3uri/4djv — ③门禁实锤 refine 可救回)
的救援机制来自上述某一项或组合。WIP 的语义 (网格势 + 单一 slope 续跑) 三项
全不占 — 对已收敛的 GPU 姿态几乎不动能量、不改排序 → **预测 REF-G.0 会
复现 ③ 的失败**。必须先用判别实验确定机制, 再定架构。

## 三、D0 判别实验 (CPU-only, 便宜, 先于一切 kernel 工程)

固定 seed, 崖靶矩阵 (1u1b, 2zda, 3ozt, 3uri, 4djv, 1w4o, 1z9g, 3ueu, 4gkm,
2yfe, 1k1i, 3jvs) + 锚点 (1pxn, 1c5z)。四变体 (independent binaries, ifdef):

| 变体 | 势 | 阶梯 | 检验 |
|---|---|---|---|
| ctl | exact (non_cache) | 5 级 | 现状 |
| E-a (REF_PROBE_GRID) | **CPU cache 网格** | 5 级 | 网格势够不够 |
| E-b (REF_PROBE_AUTH) | exact | 单轮 slope=slope_orig | 阶梯是不是机制 |
| E-c (REF_PROBE_GRIDAUTH) | 网格 | 单轮 | REF-G.0 字面语义 |

读法: E-a 保留救援 → REF-G.0 (复用 rilc, 1 天级) 可行;
E-a 失 + E-b 保 → 阶梯是机制 → REF-G 需 GPU 端 slope 阶梯 (中等);
E-a/E-b 全失 → 精确势是机制 → REF-G 需 GPU 端受体原子枚举精确势
(REF-G.2, 多 session 工程: 受体原子上传 + 配体原子分区的 inter-pair eval,
prec 样条表已在 GPU 上可复用)。

注意: ①跨运行搜索非确定性 (同 seed 33/96 aff>0.3) 在崖靶 1Å vs 5-10Å 的
信号面前可忽略; 歧义靶 3 跑定夺。②cuda 流程无 CPU cache — E-a/E-c 需在
PD1 前按 gd 每配体建一次 CPU cache (ms 级, 探针可承受)。

## 四、既定契约 (重写时必须逐条遵守)

1. `rilc_bfgs_update` = 全块协作; `set`/`set_update` 在 kernel2 模板中
   全线程无守卫调用; `visited` = per-thread 局部 + `visited_init` 各自调。
2. `q4_skip_row`/`q4_epoch_i`: tid0 清零 + `__syncthreads()` 在分支外。
3. mis_cuda_const 绑定是 kernel 发射的隐式前置 (v6b_pd ERROR#6 教训)。
4. REF-G 段必须放在 `outs[]` 填充之后 (消费循环后), DtoH 回读显式。
5. `_syncthreads` 永不在发散分支内。
6. 构建必须 `-DHOST_PROBE` + 形状宏与认证形态一致 (09-15/09-18 两次踩坑)。
7. 最终能量由既有 CPU eval_adjusted rescore 重算 — REF-G 只需产出精修
   构象, 能量语义自动与现路径对齐。

## 五、资产位置
- v6b_ref fork: 6a2cf1cd (干净, v6b_pipe 基座, 无 P-D 机器)。
- WIP 遗稿: v6b_pd 工作树 git diff (验尸完可丢弃/存档 /tmp/refg_wip.diff)。
- 消费循环锚点 (v6b_ref): main_procedure_cu.cpp L854-921 (outs 填充完成后 =
  REF-G 插入点); PD1 调用点: main.cpp L461-470。
