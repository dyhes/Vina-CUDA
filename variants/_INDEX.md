# variants/ 索引 — 自动生成 2026-09-10

| 目录 | 状态 | 说明 |
|---|---|---|
| _certified | CERT | 认证基线快照区（只读） |
| AutoDock-GPU | 外部基线 | 对比用第三方代码 |
| AutoDock-Vina | 外部基线 | 对比用第三方代码 |
| QuickVina-W-CUDA | 外部基线 | 对比用第三方代码 |
| QuickVina-W-GPU | 外部基线 | 对比用第三方代码 |
| QuickVina-W-GPU2 | 外部基线 | 对比用第三方代码 |
| QuickVina2-CUDA | 外部基线 | 对比用第三方代码 |
| QuickVina2-CUDA-batch_chunk | LINE | v6 batch kernel 载体（GPU级批，screening 6.6x），Q2-8.0 合并源 |
| QuickVina2-CUDA-batch_p2 | EXP-未结 | v6 系 batch 变体 p2 |
| QuickVina2-CUDA-batch_stream | EXP-否定 | v6 系 stream overlap，已证伪 |
| QuickVina2-CUDA-fp16_grid_v5 | EXP-否定 | 混合精度（BFGS 敏感，证伪） |
| QuickVina2-CUDA-fp16_sr_grid_v5 | EXP-否定 | 混合精度（BFGS 敏感，证伪） |
| QuickVina2-CUDA-hostreg | EXP-否定 | host/persistent/warp 实验（否定方向） |
| QuickVina2-CUDA-m0 | EXP-未结 | 有二进制，未在主线报告出现——待归类 |
| QuickVina2-CUDA-m1_fusion | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_bfgs_early | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_ldg | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_ls_dyn2 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_ls_dyn3 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_ls_dyn4 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_ls_dynamic | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_lsmax6 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_lsmax8 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_lsmax8_bfgs_early | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p05_bankconflict | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_b | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_c | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_c_batch | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate_32t | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate_b1 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate_b3 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate_b3cb | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate_prefetch | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_globstate_reg64 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_nwarp | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b1_g0a_sharedgrid | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b2 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b2_g0a | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_gamma_b2_g0c | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_prefetch | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_tc | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_p1_warpspec | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_stats | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v2_FAILED | EXP-否定 | m1 fusion v2 失败标记 |
| QuickVina2-CUDA-m1_fusion_v2_adals | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v2_d1 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v2_d1b | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v2_d1c | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v2_d1d | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v2_d1e | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v2_pairsort | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v2_parallel_scan | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v3_lbfgs3 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v3_lbfgs8 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1_fusion_v3_lbfgs8_pscan3 | EXP-否定 | m1 世代融合实验（49 否定方向内） |
| QuickVina2-CUDA-m1a | EXP-未结 | 有二进制，未在主线报告出现——待归类 |
| QuickVina2-CUDA-m1a_broken | EXP-否定 | m1a broken |
| QuickVina2-CUDA-m1a_eonly_broken | EXP-否定 | m1a e-only broken |
| QuickVina2-CUDA-memcpy_single_m | CERT | Path B（mg[0] 单份拷贝），已并入 v45 |
| QuickVina2-CUDA-no_result_h2d | EXP-否定 | host/persistent/warp 实验（否定方向） |
| QuickVina2-CUDA-persistent | EXP-否定 | host/persistent/warp 实验（否定方向） |
| QuickVina2-CUDA-pinned_host | EXP-否定 | host/persistent/warp 实验（否定方向） |
| QuickVina2-CUDA-pinned_small | EXP-否定 | host/persistent/warp 实验（否定方向） |
| QuickVina2-CUDA-v11_tile_ls | EXP-否定 | tile-LS（XHB 模式），v1 spill/v2 smem 双重证伪 |
| QuickVina2-CUDA-v45 | LINE | Q2-7 现役主线（v4.5修复+v7 E2E+v8 host GPU化+无进度条），CASF 认证 285/285 rt 0.600s |
| QuickVina2-CUDA-v6_trilinear_ilp | EXP-否定 | legacy v6 trilinear ILP |
| QuickVina2-CUDA-v7_children_bitpack | EXP-否定 | legacy v7 children bitpack |
| QuickVina2-CUDA-v8_visited_local | EXP-否定 | legacy v8 visited-local（4期否定） |
| QuickVina2-CUDA-vdeserted5_sigtest | EXP-未结 | 有二进制，未在主线报告出现——待归类 |
| QuickVina2-CUDA-warp_spec | EXP-否定 | host/persistent/warp 实验（否定方向） |
| QuickVina2-GPU | 外部基线 | 对比用第三方代码 |
| QuickVina2-GPU2 | 外部基线 | 对比用第三方代码 |
| Uni-Dock | 外部基线 | 对比用第三方代码 |
| Uni-Dock2 | 外部基线 | 对比用第三方代码 |
| UniDock-Pro | 外部基线 | 对比用第三方代码 |
| Vina-CUDA | 外部基线 | 对比用第三方代码 |
| Vina-CUDA-pre | 外部基线 | 对比用第三方代码 |
| Vina-GPU | 外部基线 | 对比用第三方代码 |
| Vina-GPU+ | 外部基线 | 对比用第三方代码 |
| Vina-GPU2 | 外部基线 | 对比用第三方代码 |
| Vina-UP | 外部基线 | 对比用第三方代码 |
| quickvina-w-GPU-Deprecated | 外部基线 | 对比用第三方代码 |
| tmp_bug | EXP-否定 | 临时 bug 复现 |
| tmp_qg | EXP-否定 | 临时实验 |
| v4_bfgs_instrumented | EXP-否定 | host/persistent/warp 实验（否定方向） |
| v4_mixed_fp16 | EXP-否定 | 混合精度（BFGS 敏感，证伪） |
| v4_resync | EXP-否定 | host/persistent/warp 实验（否定方向） |

## 统计

- m1 世代融合实验（49 否定方向内）: 44
- 对比用第三方代码: 18
- host/persistent/warp 实验（否定方向）: 8
- 混合精度（BFGS 敏感，证伪）: 3
- 有二进制，未在主线报告出现——待归类: 3
- v6 batch kernel 载体（GPU级批，screening 6.6x），Q2-8.0 合并源: 1
- v6 系 batch 变体 p2: 1
- v6 系 stream overlap，已证伪: 1
- m1 fusion v2 失败标记: 1
- m1a broken: 1
- m1a e-only broken: 1
- Path B（mg[0] 单份拷贝），已并入 v45: 1
- tile-LS（XHB 模式），v1 spill/v2 smem 双重证伪: 1
- Q2-7 现役主线（v4.5修复+v7 E2E+v8 host GPU化+无进度条），CASF 认证 285/285 rt 0.600s: 1
- legacy v6 trilinear ILP: 1
- legacy v7 children bitpack: 1
- legacy v8 visited-local（4期否定）: 1
- 临时 bug 复现: 1
- 临时实验: 1
