# Q6-batch 合并设计文档 — Q5 + v6_batch_chunk — 2026-09-12

> 用户指令: 基于 Q5 做 v6-batch。
> 这是 design-first 立项书, 代码未动。

## 〇、背景

Q5 (cd10ca24) 是单配体 redocking 的实际地板 (rt 0.310, k2 0.050)。
v6_batch_chunk 已认证 screening 6.6× (rt 1.24-1.31 时代), 但从未
合入主线。合并后 Q5 的全部 Q4 修复(溢出修复/精度域/64t/僵尸探针
清除)将与 v6 的批量架构结合。

## 一、v6 批量架构(Q5 需要新增的部分)

v6 kernel2_update 的批量签名:
```c
__global__ void kernel2_update(
    const output_type_cl *ric,       // batch: N × threads_per_ligand
    m_cl *mg,                        // batch: N 个 m_cl
    pre_cl *pre,                     // 共享(同一受体)
    grids_cl *grids,                 // 共享(同一受体)
    random_maps *random_maps,        // batch: N 份
    ligand_atom_coords_cl *coords,   // batch: N × threads_per_ligand
    output_type_cl *results,         // batch: N × threads_per_ligand
    const mis_cl *mis,               // 共享
    const int *torsion_sizes,        // batch: N
    const int *search_depths,        // batch: N
    const int *bfgs_steps,           // batch: N
    const int rilc_bfgs_enable,
    const int num_ligands,           // 批大小
    const int threads_per_ligand)    // 每配体的 MC 重启数(8192)
{
    int ligand_id = blockNumInGrid / threads_per_ligand;
    int id = blockNumInGrid % threads_per_ligand;
    m_cl* m_gpu = &mg[ligand_id];    // 每配体独立 m_cl
    ...
}
```

关键差异 vs Q5:
1. `mg` 从单个 `m_cl*` 变为 `m_cl[]` 数组(每配体独立)
2. `ligand_id = blockIdx / threads_per_ligand` 替代 `blockIdx`
3. `ric[ric_offset + id]` 替代 `ric[id]` — 每配体独立的初始构象
4. `torsion_sizes[]`/`search_depths[]`/`bfgs_steps[]` 每配体独立

## 二、Q5 → Q6 合并清单

### Host 端 (main_procedure_cu.cpp) — 最大改动
1. **批量 m_cl 分配**: 为 N 个配体各分配一个 m_cl (cudaMalloc N 份)
2. **ric**: 已在 GPU 生成(v7_gen_ric_gpu), 无需 host 批量工作 —
   只需连续布局(或指针数组)供批量 kernel 用 ric[ric_offset+id] 寻址
3. **random_maps 共享**: 事实上已是共享行为! 现行代码每配体均用
   (seed+77) 同一 seed 生成 → pi_map/sphere_map 各配体字节级相同,
   仅 int_map 因 n_tors 缩放不同(U{0..n_tors})。共享方案:
   - pi_map/sphere_map: 单份共享(字节级相同, 零风险)
   - int_map: 保留 per-ligand(80KB×N, 微不足道)或共享均匀数
     + 使用处 %(n_tors+1)(轻微分布偏差, 可忽略)
   → 批量 kernel 签名可少一个 per-ligand 数组
4. **单次 kernel launch**: grid_size = N × 8192 / threads_per_block
5. **结果收集**: N × 8192 个 results, 按 ligand_id 拆分
6. **entry-gap 摊销**: N 个配体共享一次 ctx warmup + precalculate

### Device 端 (kernel2.cu + quasi_newton.cuh)
1. kernel2_update 签名: +num_ligands, +threads_per_ligand
2. ligand_id/id 计算替代直接 block id
3. m_gpu = &mg[ligand_id] 替代 &mg[0]
4. 其余逻辑(Q4-9R 溢出修复/Q4-12 64t/S4a-1 mis const)保持不变
5. **GRIDS 共享**: 同一受体的 N 个配体共享同一 grid — 已经是

### 需要保留的 Q5 修复(合并时不能丢失)
- Q4-9R: lm_s/lm_y __shared__ 右尺寸(非 FP16, 非 local)
- Q4-12: 64 线程 block(16,4)
- S4a-1: mis_cl → __constant__
- Q4-7: 精度域清理(expf/printf/宏)
- Q4-8R: 无着色(已回退), atomic skip_row 保留
- E1-retry 回退: m_coords/minus_forces 保持 FP32

### 需要注意的冲突
- v6 的 `mg[ligand_id]` 与 Q5 的 `m_gpu = &mg[0]` — 改为 mg[ligand_id]
- v6 的 batch ric 与 Q5 的 per-ligand ric — ric 已 GPU 生成, 只需布局
- v6 没有 Q4 的 stage_timer/mis_cuda_const — 需要合并

## 三、收益模型(基于修正后的理解)

### screening 场景(多配体同一受体)
- **entry-gap 摊销**: ctx warmup 84ms + precalculate 5ms + parse 20ms
  = ~110ms / N 个配体 → N=100 时每配体仅 1.1ms(省 109ms/配体)
- **kernel 级共享**: 同一受体 grid, N 个配体的 kernel 共存 →
  无共享墙问题(不同配体的块可以共存, shared 是 per-block 的)
  → 但 SM 块槽仍然有限(128 SM × 7 = 896 slots), N×8192 块
  排队执行 → 每个配体的 k2 不会加速, 但 GPU 利用率提高
- **正确的收益模型**: screening 总时间 ≈ (N × k2) / GPU_utilization
  + 1× entry_gap。GPU_utilization 从 solo 的 ~30% 提升到并发下
  的 ~60-90% → 总时间 ≈ N × k2 / 0.6-0.9 + 110ms
  vs 串行: N × (k2 + 320ms)

### redocking 场景(单配体)
- **无收益**: 单配体不需要 batch — Q5 已经是地板

## 四、执行计划(分阶段)

### Phase 1: 单受体多配体批量(核心)
- 批量 m_cl/ric 分配 + 单 kernel launch
- 验证: 同一受体 5 个配体的 CASF 子集 vs 串行 Q5
- 门禁: aff 对称 + Top-1 带内

### Phase 2: 全量 CASF screening 模式
- 批量运行 285 个配体(可能需要分批, 每批 N=8-16)
- 门禁: Top-1 + aff 分布 + RT 对比

### Phase 3: v6 的 screening 6.6× 重新验证
- 修正后预期: 不再是 6.6×(那是旧构建时代), 而是基于 entry-gap
  摊销的真实加速比

## 五、工作量估计

| 阶段 | 内容 | 时间 |
|---|---|---|
| Phase 1a | host 端批量分配+launch | 1 会话 |
| Phase 1b | device 端签名改造+测试 | 1 会话 |
| Phase 2 | 全量门禁+修正 | 1 会话 |
| 合计 | | **3 会话** |

## 六、执行前提

- Q5 (cd10ca24) 作为基线, 所有 Q4 修复已合入
- v6_batch_chunk 目录作为参考(旧代码, 不直接 cherry-pick)
- 需要设计 host 端批量数据布局(数组 of structs vs struct of arrays)
