# v4.5 = v4 + 三修复（2026-09-09）

## 内容（均对 blockDim=32 行为等价，对 >32 线程是正确性前提）
1. ig_eval_deriv_update_optimal 归约：单warp shfl -> 块级(warp shfl + shared 树)
2. eval_interacting_pairs_deriv_update_optimal 归约：同上
3. POT_deriv_update accumulate->tree 相位间加 __syncthreads
（pairs 力累加保持 v4 原样；mutate/gation 原样——128 线程下它们仍有坑，见 v11 HANDOFF）

## Smoke（同 box 三例，v4 新测对照）
- K2: -0.7% / +6.3% / -1.8%（噪声内）；aff |d| <= 0.2 kcal
- ERR=0 全部

## CASF-2016 redocking 285/285 OK（experiments/Sep09_1525_CASF-2016_v4_5）
- mean rt 1.459s vs v4 1.531s；median 1.440 vs 1.520（-5%，主要来自机器状态差异，
  k2 median 0.135 vs 0.136 = +0.7% 持平）
- 亲和力：mean |delta| 0.141，>0.5 的 13/285；对照 v4 自身两次运行（Sep08_1706 vs
  Sep08_1745）：mean |d| 0.162，>0.5 的 23/285 —— v4.5 与 v4 的差异小于 v4 自身的
  run-to-run 噪声，判定为 Monte Carlo 随机性，无系统性漂移
- better/equal/worse: 79/113/93（对称，无方向性）

## 结论
v4.5 = v4 的稳健超集：性能持平、数值在 MC 噪声内、285/285 OK。
价值：清掉 128 线程正确性地雷（归约/相位barrier），未来任何 blockDim>32 的实验
不必重踩；本身不提速（32 线程下行为等价）。
