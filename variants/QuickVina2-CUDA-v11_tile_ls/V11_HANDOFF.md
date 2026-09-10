
# v11 Tile-Parallel LS — 实现交接（2026-09-08）

## 状态：架构设计完成，基础设施已起步（fork 于 variants/QuickVina2-CUDA-v11_tile_ls，Stage A 已编译通过：128 线程 block + cg 头文件）

## XHB 参照实现（源码级，quasi_newton_gpu.cu L272-330）
- cg::tiled_partition<8>，128 线程 = 16 tiles
- 10 个 trial 并行：trial = tile.meta_group_rank()（0-9），alpha = 0.5^trial
- 每 tile 用 8 线程算完整 E：ig/pairs/POT 均有 <tile_size> 版本（model_gpu.cuh eval_deriv<tile_size>）
- tile 内归约：temp_storage_fl + tile.sync（tile_sum, common_gpu.cuh L393）
- 接受判定：f1 - f0 < c0 * alpha * pg → temp_storage_int[trial] = trial
- block_min(temp_storage_int, 10) 选最小通过 trial → 该 tile 广播 alpha/f1/x_new/g_new

## 我方需 tile 化的 6 个函数（v4 → v11）
1. scalar_product_update → tile 版：shared tmp 改 per-tile 槽（temp_storage[n] × 16 tiles），
   归约用 tile-shfl（8 lanes 内），去掉 __syncthreads（tile.sync 替代）
2. ig_eval_deriv_update_optimal：stride threadsPerBlock→8（tile.rank() 起），
   全 warp shfl 归约 → tile 内 8-lane shfl（tile.shfl_down，自动 mask）
3. eval_interacting_pairs：同上
4. POT_deriv_update：树遍历由 tile-rank-0 执行（每 tile 独立树），shared 数组改 per-tile 槽
5. get_to_minus_update / output_type_cl_increment：elementwise，stride 改 8
6. LS 本体：XHB 模式重写（L272-330 可逐行参照）

## kernel2_update 重构要点
- block 128 线程（已改 dim3 block_size(128,1,1)），16 tiles of 8
- 所有 shared 数组拷贝循环 stride 32→128
- stage_timer：threadIdx.x==0 语义不变
- BFGS 非_LS 部分（two-loop/标量积）：可全 block 128 线程（n=15 维小，
  scalar_product 的 block 归约天然支持）或 warp0-only（需 others syncthreads 等待）

## 预期收益（用实测数字推演，见 changelog §11.2）
- 1c5z(13at): LS 2.75× → kernel ~1.7×
- 1pxn(25at): LS 1.4× → kernel ~1.2×  
- 1u1b(55at): 负优化 → 建议运行时回退串行 LS（atoms>40 时走原路径）
- CASF 加权预期: kernel ~1.15-1.3×（大配体不劣化前提下）

## 风险与门禁
- E tile 化触碰全部热点路径 → smoke 亲和力对照 + CASF 双 run（TRAP51）
- shared 预算：+per-tile temp（~16×n×4B≈1KB）→ 16.9+1 ≈ 18KB < 27.7KB ✓
- 与 v6 batch 兼容性：block 映射 (ligand_id, restart_id) 不变，仅 block 内线程数变化 → 兼容

## 剩余工作量估计：6 个函数 tile 化 + LS 重写 + kernel2 收尾 ≈ 1 个整天

---
## v11 实现日志（2026-09-09，tile-v1 实测否证 + v2 方案）

### 已落地（inc/cuda/quasi_newton.cuh, mutate_conf.cuh）
1. Stage A 三颗 128 线程雷已修：
   - ig/pairs 归约单 warp shfl → warp1-3 部分和被丢 → 改 block 级归约（warp shfl + shared 树）
   - pairs 力累加跨 warp 竞争 → shared atomicAdd
   - POT_deriv_update accumulate→tree 无 barrier（32线程单warp侥幸）→ 加 __syncthreads
   - mutate_conf quaternion_increment 多 warp 并发 read-modify-write → 单 writer + barrier
2. tile-v1 已实现并编译通过：line_search_lewisoverton_tile（8 trial/tile，Lewis-Overton 语义），
   接入 rilc_bfgs_update（k>1 时 eval_reset 守卫），v4 串行路径保留（V11_LS_GRID=0 可回退）。

### tile-v1 实测否证（1pxn thread 8192，stage_timer）
- K2 0.146s(v4) → 1.614s（9.2× 慢）；EVALS 1.94M→100k（单 E 慢 ~180×）；affinity -9.0 vs -9.1 基本对
- STAGE_EVAL_INIT 块版 E 反而更快（5.48→3.51s）→ 128 线程修复版块 E 是对的
- 根因A（性能）：per-trial 状态（m_coords+forces+rigid ≈10KB）在 lane0 local memory →
  spill 序列化。根因B（正确性）：POT_deriv 的 __shared__ position_derivative_tmp[48][3]
  被 8 个并发 lane0 写同一块 → 1u1b ERROR 22 万条、affinity=0.0
- 结论：lane0-串行-E + local/静态shared 状态 = 死路（实测，非推测）

### v2 方案（下一步，方案B：dynamic shared per-tile 槽）
- extern __shared__ 切 per-tile 槽：m_coords 1.5K+forces 1.5K+rigid动态 3.6K+POT tmp 1.2K ≈ 7.8KB/tile
  ×8 tile ≈ 63KB dynamic + 17KB static = 80KB < 99KB（sm_89 opt-in 上限）✓
- kernel_monte 加 cudaFuncSetAttribute(kernel2_update, cudaFuncAttributeMaxDynamicSharedMemorySize, 80KB)
- POT_deriv → POT_deriv_tile(…, per-tile tmp 指针)；ig 串行版已就绪（v11_ig_eval_serial）
- 风险：80KB/block → RTX4090 1 block/SM（v4 16 block/SM）；靠 8-trial 并行 + 128 线程块 E 换回。
  若仍负优化 → 方案C（XHB 原生真 tile 化 E，ig/pairs 8-lane 分工，工作量大）
- smoke 命令规范：cd 变体目录 + ulimit -s 8192 + --receptor/--ligand 直传 + --thread 8192
  （--config 方式会崩，v4 二进制同样崩，勿用）

---
## v11 实现日志（续，2026-09-09 下午：v2 dynamic-smem 版跑通，实测结论）

### v2 修复（quasi_newton.cuh + kernel2.cu）
1. shfl mask 移位 UB：0xFFu<<(threadIdx.x&~7) 对 thread>=32 是 UB → 0xFFu<<((threadIdx.x&31)&~7)
2. kernel_monte 重复 launch（我改的 opt-in + 原版遗留）→ 合并为单 launch + opt-in 64.8KB
3. tiles 8-15 越界写 slot[8..15]（状态初始化未按 tno<V11_LS_GRID 门控）→ 已门控
4. trial 循环重排：整个 octet（8 lane）集体进入 v11_m_eval_tile（8-lane eval 是集体操作，
   lane0-only 循环会让 lanes1-7 永远到不了 __syncwarp）；bookkeeping 全部 warp-uniform
   （floc 蝶形归约一致，dgd lane0 算完 __shfl_sync 广播），步进规则与 v4 一致
5. snapshot 8-lane 并行化（6.6KB×53万次 LS 的 smem 写流量 /8）
6. per-E 计数补进 tile 路径（原来 EVALS_N 只数块版 E，E/LS=0.19 是假象，实为 4.43）

### v2 实测（1pxn 同 box 同 seed 环境对照 v4 新测 K2=0.144s）
- 亲和力：1pxn -9.1 / 1c5z -5.9 = 与 v4 完全一致，ERR=0 → 正确性达标 ✓
- K2: 1.577s vs v4 0.144s = **11.0x 慢**
- 分解：E/LS 4.43 vs 4.32（+2%，搜索行为等价）；LS 次数 +20%（8 trial 探索更多）
  per-E 0.668us vs 0.075us = **8.9x 慢**（tile E 只有 8 lane 并行 vs 块 E 128 线程 + 
  2×syncwarp + smem 槽流量 + lane0 串行 set/POT）
- 块版 E（eval_init）19us vs v4 56us = 2.9x 快（128 线程修复有效）
- 结论：**LS 内并行 8 trial 的收益(理论 4.3/8≈0.54x E 次数) 被单次 tile E 8.9x 减速吞没**
  → 净效果 11x 慢。tile-parallel-LS 路线在"1 block=1 ligand、E 需全原子坐标"的架构下
  被证伪（第 24+1 个否定方向）：XHB 能成立是因为它把整个 E 做成 tile_size 并行且
  coords/forces 跟 tile 走，我方 set/POT 的树遍历本质串行、pairs 力累加需原子操作，
  8 lane 无法有效摊开。

### 有效的副产品
- 128 线程正确性修复（块版归约/POT barrier/mutate 单写者）让块 E 2.9x 提速，
  这部分已验证有效，可单独保留（v4 逻辑 + 128 线程块 = 未来的 v4.5 方向）
- v4 串行 LS 路径完整保留（V11_LS_GRID=0 即回退），tile 代码可作为失败实验存档

### 下一步建议
1. 保留全部 128 线程修复 + 撤 LS dispatch 到 v4 串行路径（= v4+fixes），K2 预期 <0.144s
   （块 E 2.9x 提速直接兑现在 LS 里），跑 CASF 验证后可立为 v4.5 milestone
2. tile 代码留档不删（V11_LS_GRID 编译开关控制）
