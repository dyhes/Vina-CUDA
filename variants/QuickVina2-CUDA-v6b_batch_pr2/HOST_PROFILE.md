# Host 侧 wall 构成 profile 报告 — 2026-09-09

## 方法
v4.5 源码注入 HOST_PROBE 计时桩（`-DHOST_PROBE` 编译开关，无性能影响），
1c5z（GPU 最小 case，k2=0.014s）单配体 standalone + 批模式实测。

## 单配体 standalone 分解（总计 0.497s，GPU 仅 4.4%）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| **CUDA ctx init + streams + 首次 malloc/memcpy** | **232 ms** | **47%** |
| per-ligand host prep（随机数 20000×3 + 结构展开 + h2d）| 95 ms | 19% |
| pre_gpu_setup（par.mc/weights 准备）| 70 ms | 14% |
| 输出写出 + teardown | 35 ms | 7% |
| 进程启动 + parse（受体+配体 PDBQT）| 19 ms | 4% |
| everything_t（打分项构造）| 17 ms | 3% |
| precalculate 插值表 | 7 ms | 1% |
| **GPU kernels（k1+k2）** | **22 ms** | **4%** |

## CASF 实测构成（rt_med 1.32s）
- 0.50s = 二进制内部（上表）
- **~0.80s = runner 侧**：python `subprocess.run` + `sudo bash -c` fork，
  **每配体一个新进程** → CUDA ctx 232ms + setup 70ms 每配体重付
- 实测对照：standalone 0.497 vs CASF 1.27~1.39 ✓ 吻合

## 批模式实测（--ligand_directory，一个进程跑多配体）
- batch-3: 1.515s（0.505 s/lig）；batch-7: 2.582s（**0.369 s/lig**）
- ctx 摊薄有效但 per-ligand prep 串行仍在

## 优化杠杆（按收益排序）
1. **runner 改批模式**（最大杠杆）：CASF runner 已逐配体 spawn 进程；改用
   `--ligand_directory` 一次进程跑全目录，每配体立省 ctx 232ms + setup 70ms
   + runner fork 0.8s → **rt 1.32 → ~0.4s（-70%）**，零内核改动
   - 注意：CASF 每配体 box 不同（center 逐配体）→ 需按 box 分组跑批，
     或给二进制加多 box 支持的小改动
2. **CUDA ctx 复用**：若必须逐配体进程，用 cudaIPC/持久 daemon 保活 ctx
3. **per-ligand prep 95ms**：rand_maps 20000×(3+1) float 全部 host 生成再 h2d，
   可 GPU 生成（curand）或减量（search_depth 相关即可）
4. k2 kernel 本身（0.014~0.6s）只在 1u1b 这类大配体上才显著

## 与既有变体的关系
- batch_pool2（Jul28, rt 1.230）已验证批方向，但当时未做 probe 归因；
- v6_batch_chunk（Sep08, rt 1.310）合并了 memcpy 修复但仍是逐配体 runner。
