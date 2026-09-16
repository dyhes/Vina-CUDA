# 共用目录的真相 + 目录管理规约 v2 — 2026-09-10

## 一、"v4 家族共用一个目录"是怎么发生的（诚实复盘）

物理事实：`variants/QuickVina2-CUDA-v45/` 这一个目录，从 9 月 9 日至今被
**原地持续修改**，先后承载了 5 个逻辑版本：

| 逻辑版本 | 时间 | 该目录当时的状态 |
|---|---|---|
| v4.5 | 9-09 凌晨 | v4 + 三修复（归约/POT barrier/syncthreads）|
| v4_7 | 9-09 晚 | + E2E 计时基建 + Fix A/B（host hoist）|
| v4_8 | 9-10 早 | + ric/randmaps GPU 化 + 写零 HtoD 删除 |
| v4_8b | 9-10 中午 | + 进度条线程默认移除（opt-in 宏）|
| 探针微调 | 9-10 下午 | PREP 微探针三进三出（现仍带探针桩）|

**为什么会这样**：违反了我们自己 7-16 定的 FORK 纪律。当时图快——每个
增量都小（几处 patch），`ssh + patch + make` 的循环里没有 fork，靠
experiments 时间戳 + runner variant 标签做"软版本化"。代价今天暴露：

1. **无法 checkout 回 v4.5 复现**（源码已被覆盖 4 代，只剩 .bak 残片和
   CERT 快照——而 CERT 机制今天才建）
2. **误判风险**：昨天的 "kernel_monte 探针消失" 一案，根因之一就是
   目录内源码 vs 二进制的版本不同步（make 增量 + 原地改）
3. **git 没用起来**：v45 目录有 git（ahead 2 commits），但 9 月所有修改
   全部未提交——git status 显示 1318 files changed，等于没有版本管理

## 二、规约 v2（吸收你的两条修正，取代我昨天的方案）

你的两点都对，采纳：
- ✅ **milestones 只记"有明确提升的里程碑"**：认证过的、论文要引用的
- ✅ **variants 管中间版本，每次修改先 fork**：中间过程比结论更常回看

### 2.1 fork 纪律（硬性，恢复执行）

```
新逻辑版本 = 新目录（从上一认证点 fork）
  cp -a QuickVina2-CUDA-v45  QuickVina2-CUDA-Q2-8.0-merge   # 例
```
- fork 时机：**任何语义修改前**（探针/日志类注入除外，但必须当日回滚）
- 目录内 git：fork 后 `git add -A && git commit -m "fork from <src> @<date>"`
  每次 patch 前后各一 commit（今天已把 v45 现状 commit 掉，补上欠账）
- runner `--variant` 标签 = 目录名（不再出现 v45 目录跑出 v4_8 标签的错位）

### 2.2 三层目录职责

| 层 | 位置 | 放什么 | 何时写 |
|---|---|---|---|
| **EXP** | `variants/QuickVina2-CUDA-*` | 中间版本、失败尝试（fork 出来的每个都留）| 每次 fork |
| **MILE** | `milestones/` | 只有认证+明确提升的版本（快照，只读）| CASF 认证后 |
| **CERT** | `variants/_certified/` | 认证当刻的源码+二进制+报告 tar（防 milestones 重演漂移）| 同上 |

- MILE 命名跟随 Q2-*：`milestones/Q2-7/`、`milestones/Q2-8.0/`（v0-v6 旧目录保留不动）
- _INDEX.md 三色（LINE/CERT/EXP）不变，MILE 由 CERT 对应

### 2.3 现状修补（已执行）

1. ✅ v45 目录 git commit（9 月全部修改一次性入库，message 标明含 v4.5→v4_8b 四代）
2. ✅ CERT 快照 Q2-7_2026-09-10（昨日已做）
3. ⏳ MILE 补录：从 CERT tar 恢复四个历史点到 `milestones/`（v4.5/v4_7/v4_8
   无法精确恢复——**承认损失**：这三代只有 CERT-style 快照缺失，能恢复的
   只有 v4_8b=Q2-7；v4.5/v4_7/v4_8 依赖 experiments 的 summary+报告存档
   作"数值快照"，源码级不可复现，记入 _INDEX 损失说明）

## 三、执行清单（Q2-8.0 起）

```
1. cp -a v45 → Q2-8.0-merge          # fork
2. cd Q2-8.0-merge && git add -A && git commit -m "fork@Q2-7"
3. 合并 v6 batch kernel（ric 批生成 + m_cl 布局对齐）
4. smoke 五例 + CASF 全量
5. 认证通过 → tar 到 _certified/Q2-8.0_<date> + cp 到 milestones/Q2-8.0/
6. v45 目录转为只读 EXP（_INDEX 改 LINE→CERT-linked）
```
