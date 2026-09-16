# v6b 批量改造(DESIGN C)执行纪要 — 2026-09-16

> 指令: 基于 q53_w6cL_r1a_2026-09-15.md 梳理现状, 进行 v6-batch 改造。
> 基线: q53-w6-192t 头 b725964e (W6c+P4); fork commit b38d5183; 批量落地 ec3b5d63。

## 〇、结果总览

- **批量路径 (DESIGN C) 落地并双跑过门禁**: kernel2_update 字节级未动
  (b725964e 原样), 批量 = N 次异步 launch + 单次尾 sync, host 侧摊销。
- **关键架构否决 (ncu 实证)**: DESIGN A (单 kernel 批量签名 + device 指针数组)
  尾巴配体 k2 **+29~37%** (3ag9 948/963µs vs 同日对照 694/700µs) — 机制为
  寄存器再分配 (local-ld +31%), `__restrict__` 与 shared-ctx (FIX8) 均救不回。
- 批量收益 (N=5 同受体): e2e_dock 0.997s vs 串行和 1.34s (**−26%**),
  k2 0.862 vs 单跑和 0.917 (同流排队, 无共驻加速 — 符合 MPS 教训预期)。

## 一、现状梳理 (r1a handoff 承接)

| 项 | 状态 |
|---|---|
| Q5.3 (W4 128t/64reg) | redocking champion (k2 0.047, CERT) |
| W6c (192t/40reg+shave) | screening 特化 (Σk2 −12.7%), 未晋升 |
| W6c-L P1/P2/P3 | 全否决 (搜索轨迹雷); P4 已落地 (b725964e, 惰性) |
| v6-batch 合并 | 本 session 执行 (r1a handoff §六.2 主线优先级) |

## 二、实现 (fork: variants/QuickVina2-CUDA-v6b_batch)

### 改动面 (kernel 未动!)
1. `kernel2.cuh`/`kernel_monte`: 15 参批量签名 (指针数组+标量数组+num_ligands+threads_per_ligand+lig_valid)。
2. `lib/main_procedure_cu.cpp`: per-ligand 循环拆三段 **prepare | batch-kernel | consume**:
   - prepare: 原 fills/mallocs/gen (含 v7_gen_randmaps/ric), 加 staging 向量 + `lig_valid=1`;
   - kernel: 一次 `kernel_monte` (N 次异步 launch, 一次 sync);
   - consume: DtoH/cl_to_vina/add_to_output + per-ligand cudaFree 移到 kernel 后
     (**FIX5: 原 free 在循环内 = use-after-free; consume 在 kernel 前 = 读空**)。
3. kernel_monte 内: mis_cuda_const 绑定 1 次/批; per-ligand 标量 pack (4N int 单 h2d)
   在 event 窗外 (防 10µs cudaMalloc 污染 52µs 锚点口径); lig_valid=0 的配体跳过 launch。
4. ric/randmaps: 零新代码 (v7 GPU 生成, stream8→stream0 隐式同步依赖成立);
   random_maps 事实共享 (seed+77) 保留 per-ligand buffer (共享化是可选优化非正确性项)。
5. runner: `Q3-v6b` variant 注册 (BINARIES + choices)。

### 被否决的 DESIGN A (单 kernel 批量) — 完整证据链
- 实现: kernel2_update 签名改指针数组+标量数组, `ligand_id = blockNumInGrid / threads_per_ligand`。
- 调试链: IllegalAddress → memcheck 定位 (host 数组当 device 数组 → FIX; stash 在
  cudaMalloc 前 → NULL 指针 → FIX2; use-after-free + consume 顺序 → FIX5 结构拆分)。
- 全修好后 **性能裁决失败**: 3ag9 k2 948/963µs vs 同日 Q3 对照 694/700µs (+37%/+37%)。
- ncu 归因: local-ld 1.353G→1.778G (+31%), local-st +40%, latency 33.96→35.28,
  REG 恒 40 — **每轨迹常量从参数空间 (0 reg) 变寄存器驻留局部量 (~8-15 reg),
  在固定 40reg 帽下把热循环 BFGS 状态挤进 stack**。
- FIX7 (`__restrict__` 参数+别名): local 不变 (1.76G) — 无效。
- FIX8 (块一致 8 值进 shared ctx, +52B): local-ld 1.62G, k2 903µs — 只收复 1/3。
- 结论: 不是别名问题, 是**寄存器再分配**。批量签名下每个 use 点都变成
  `数组读+间接` (哪怕 shared ctx), 短活跃区间仍迫使重分配。40reg 帽 + W6c
  形状对插装零容忍。DESIGN C (kernel 不动) 是唯一无回归路径。

## 三、门禁数据

### 冒烟 (5 桶, 同日对照 Q3CTL = Q3 目录同日重建)
| target | Q3CTL k2 | V6B k2 | MODE1 (CTL/V6B) |
|---|---|---|---|
| 1c5z | 15µs | 14µs | -5.9 / -5.9 |
| 1pxn | 54µs | 49µs | -9.0 / -9.1 |
| 3p5o | 67µs | 60µs | -8.7 / -9.1 |
| 3b68 | 82µs | 84µs | -11.4 / -11.4 |
| 3ag9 | 692µs | 710µs (+2.6%) | -5.1 / -4.7 |

### 批量正确性/稳定性 (DESIGN C)
- 3×N=5 (1c5z 受体): rc=0, 零错误, 5/5 输出; k2 0.848/0.830/0.828 (spread 2.4%)。
- N=1 (回归形态) 与 N=16: rc=0 全输出。
- 批量 MODE1 与单跑偏移 = 测试设计 (共用 1c5z box), 非缺陷; 正确性由 CASF 全量判定。

### CASF 285 redocking ×2 跑 (随机种子实现)
| 指标 | W6c run1/2/3 (9-15) | 今日 CTL (Q3) | **V6B run1** | **V6B run2** |
|---|---|---|---|---|
| n OK | 285×3 | 285 | 285 | 285 |
| k2 med | 0.049/0.049/0.048 | 0.0500 | 0.0500 | 0.0500 |
| Σk2 | 18.39/18.52/18.46 | 18.86 | 18.88 | 18.87 |
| e2e_dock | 0.098 | 0.100 | 0.100 | 0.100 |
| Top-1 | 171/169/166 | 178 | 166 | 171 |
| aff\|Δ\|>0.5 vs 今日CTL | — | — | 26 | 33 |
| vs 自噪声基线 | flips -24/+22, -19/+16, -22/+17; aff>0.5 19/21/26 | | 包络内 | 包络内 |

- **k2 med / Σk2 与对照完全一致** (kernel 字节级相同 → codegen 相同, 符合预期)。
- Top-1 {166,171} 中位 168.5; vs 今日 CTL 178 跨度 9.5 < 家族自噪声 13
  (v4 家族 8 跑 171..184 协议) → RNG 噪声非回归; V6B 同二进制自噪声
  flips -22/+27 与 W6c 三跑自噪声 (19-24) 同带。
- aff |Δ| med 0.100 = 自噪声; P3 红旗 (Σk2 异常快 16.58, evals 减半, 悬崖列表)
  在 V6B **全部不存在**: Σk2 18.88=对照, 冒烟 evals 5.69M vs 对照 5.60M (+1.6%)。
- 门禁判定: **过** (redocking 无回归 + 批量路径功能正确)。

## 四、当前状态
- 分支 q53-w6-192t 工作树干净 (b725964e); v6b 工作: fork b38d5183 → ec3b5d63。
- 二进制 md5 7c797793; 指纹 REG:40 STACK:10160 SHARED:11520 (= W6c 原样)。
- runner Q3-v6b 可用; smoke 脚本 /tmp/v6b_{smoke,ctl_smoke,batch_check,scale}.sh。
- CASF: Sep16_1201 (v6b r1), Sep16_1213 (v6b r2), Sep16_1206 (同日 CTL)。

## 五、教训沉淀
1. **批量签名 (指针数组+间接) 在固定 reg 帽 kernel 上有真实代码gen代价**:
   参数空间标量是零寄存器; 运行期间接 ~8-15 reg → 挤压热循环 → local 溢出税。
   ncu local-ld/st 是最灵敏的探针。`__restrict__`/shared-ctx 都救不回。
2. **认证形态优先**: v6 的 `kernel2PK...` 单参数 kernel 是被 CASF 认证的
   codegen; 批量化的安全形态是 host 侧循环 launch (DESIGN C), 不是 kernel
   内 ligand_id 分支 (DESIGN A)。同流 N-launch 与单 launch 无共驻差异
   (后续 kernel 必须等前 kernel 排空), 批量收益本来就在 host 侧。
3. **consume/free 必须随 kernel 调用点一起搬家** — 结构性重构 (循环拆段)
   时, 数据流两端 (生产者/消费者) 和生命周期 (free) 都要跟着调用点走。
4. memcheck 首个报告点 ≠ 根因点 (shared 拷贝模式骗了我一次); 
   readback 探针 (host 端 cudaMemcpy 回读 device 数组) 比设备 printf 可靠
   (printf 在 fault 时丢失)。
5. Top-1 单跑带缘 {166} 需第二跑定带 ({166,171} 中位 168.5 = 带内);
   家族自噪声基线 (同二进制双跑翻转数/aff>0.5) 是区分噪声与回归的必要对照。

## 六、下一步 (未执行)
1. Screening 模式全量 (57 靶 × 285 配体, 同受体批量) — v6b 的主战场,
   期望 Σk2 不变 + e2e 摊销 350ms→~40ms/配体 (Level-1 部分) + N-1 sync 消除。
2. W6c (192t) screening 构建选择: v6b 与 W6c 块形状正交 (launch 参数),
   screening 构建可在 v6b 上切 192t/40reg。
3. random_maps 共享化 (pi/sphere 单份) — 可选优化, 非正确性项。
4. per-ligand 事件对 (批量内每配体 k2 归因) — 需要时再加。
