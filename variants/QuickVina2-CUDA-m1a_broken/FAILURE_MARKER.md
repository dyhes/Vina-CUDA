# FAILURE MARKER

- **标记日期**: 2026-07-12
- **变体**: QuickVina2-CUDA-m1a_broken
- **改造类型**: 朴素 BF16 混合精度（quasi_newton.cuh 归约 shared mem float→__nv_bfloat16, warpRecude BF16 版）
- **状态**: ❌ 失败（已弃用）
- **失败表现**: 1u1b 卡在 docking 搜索阶段（进度条 ~7% 停滞，m0 同配置 ~2s 完成）；无法完成对接
- **根因（对照文献）**: 朴素 BF16 改造未做误差校正(EC)且未分层保精度（m0_analysis §5.2 要求 BFGS Hessian/line search 保 FP32）。印证 G19（纯 FP16-TC 降 docking 质量）、P09（BF16 舍入误差大于 FP16）。
- **后续方向**: 放弃此路线，改为 fusion 提 occupancy 方向改造（见新 m1 设计）。
- **保留原因**: 留作对照，勿删。
