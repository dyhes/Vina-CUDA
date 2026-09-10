# 版本谱系梳理与新命名约定 — 2026-09-10

## 一、为什么乱（盘点 91 个 variants + 13 个 milestones 的成因）

1. **两套并行目录**：`milestones/v0-v6`（v6 时代的"里程碑"快照，各自有二进制）
   与 `variants/QuickVina2-CUDA-*`（FORK 纪律的实验目录，80+ 个）。
   `milestones/v4` 7 月 25 日建、9 月 8 日又重编过——同一名字下内容漂移。
2. **编号三义**：`v4` 既是 milestones 快照，又是"v4 家族"统称（v4/v4.5/v4.7/
   v4_8/v4_8b 五个认证基线全叫 v4.x，但**物理目录只有一个 `QuickVina2-CUDA-v45`**，
   靠 git/时间线区分），还与 runner 的 `--variant v4_8b` 标签耦合。
3. **失败/探针目录混居**：`*_broken`、`_FAILED`、`tmp_bug`、`vdeserted*`、
   `m1_fusion_*` 40+ 个否定方向目录与认证目录平铺同级，无法一眼分辨"活的线"。
4. **v6 消失感**：v6（GPU 批）只在 `milestones/v6`/`variants/QuickVina2-CUDA-batch_chunk`，
   而 v45 主线不含它——导致"GPU 级批是不是要新做"的误判（刚发生）。

## 二、命名约定（新线自今日起执行；旧目录**不重命名**——runner/论文/交接文档
   的引用链比目录名贵，动它风险大）

### 2.1 主线编号（going forward）

```
Q2-<世代>.<合并序>[.<增量>]     # 例：Q2-8.0, Q2-8.1
```
- **世代 = 统一版序号**：每完成一次"结构级合并/重基线"进一代。
  当前合并目标 = v4.5 修复 + v8 host 优化 + v6 batch → **Q2-8.0**
  （8 = 第 8 次重基线；与旧 v8 无关，旧 v8_visited_local 改称 legacy）
- **合并序**：同世代内的认证小步（修 bug、微优化）+0.1
- **增量**：未认证的实验目录 +.x.y（如 Q2-8.1.3），认证后砍掉尾号
- runner `--variant` 标签与目录名同名（`Q2-8.0`），论文引用同符号

### 2.2 目录三色分层（variants/ 内加 README 标注，物理不动）

- **LINE**（活的线）：`QuickVina2-CUDA-v45`（= 现役主线，合并后更名 Q2-8.0）、
  `QuickVina2-CUDA-batch_chunk`（= v6 batch kernel 载体，合并源）
- **CERT**（认证基线快照，只读）：每次 CASF 认证后 tar 一份到
  `variants/_certified/<标签>_<日期>/`，与活目录解耦（防 drift，正是
  milestones/v4 被 9 月重编教训的解法）
- **EXP**（实验/否定）：其余 80+ 目录按现名不动；顶层加
  `variants/_INDEX.md` 一行一目录（状态：LINE/CERT/EXP-否定/EXP-未结）

### 2.3 旧名 → 新符号对照（论文与交接文档引用用右列）

| 旧名/符号 | 新符号 | 状态 |
|---|---|---|
| milestones/v4（7-25 重编版）| Q2-4 | CERT 快照待补 |
| QuickVina2-CUDA-v45 现内容（v4.5+修复）| Q2-5（= v4.5 认证版）| CERT 待补 |
| v45 现内容 + v7 E2E（v4_7）| Q2-6 | CERT 待补 |
| v45 现内容 + ric GPU + 无进度条（v4_8b）| **Q2-7（当前活线头）** | 现 LINE |
| batch_chunk / milestones/v6 | Q2-8.0 的 batch 来源 | LINE（合并源）|
| v11_tile_ls | Q2-7.1（tile-LS，已证伪，EXP）| EXP-否定 |
| m1_fusion_* 40+ 目录 | Q2-3 系（m1 世代，已收尾）| EXP-否定 |
| v8_visited_local / v6_trilinear / v7_children | legacy-v8/v6/v7 | EXP-否定 |

## 三、立刻要做的三件事（防止再次混乱）

1. **CERT 快照**：把 v4_8b 当前二进制+源码 tar 到 `variants/_certified/Q2-7_2026-09-10/`
   （含 13 份报告 md）——此后 v45 目录随便改，认证基线不再漂移
2. **`variants/_INDEX.md`**：91 目录一行表（我可以用脚本从时间线+报告自动生成初稿）
3. **合并立项按新名**：`Q2-8.0 = v45 + v6 batch` 合并分支叫
   `variants/QuickVina2-CUDA-Q2-8.0-merge`，handoff/论文统一用 Q2-* 符号

## 四、不做什么

- 不批量重命名旧目录（引用链风险 > 收益）
- 不动 `milestones/`（它是历史快照，CERT 机制替代其职能）
- 不改 runner 里旧 variant 标签（v4_8b 等历史实验数据要能复现）
