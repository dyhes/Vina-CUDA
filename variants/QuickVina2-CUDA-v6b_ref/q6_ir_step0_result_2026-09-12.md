# Q6-IR Step 0 结果: FP16 精度损失容忍性验证 — 2026-09-12

## 实验
在 Q5 (cd10ca24) 的 set_update 末尾加一段 FP16 round-trip 量化
（`__float2half(__half2float(v))`），模拟 __half m_coords 存储的精度损失。
**不改变 struct 类型**（shared 保持 12684B），**不改变 occupancy**。
这隔离了"精度问题"和"容量问题"。

## 结果
| target | k2 | VINA RESULT | 判定 |
|---|---|---|---|
| 1pxn | 0.052 | **-9.1** | 与 Q5 一致 ✓ |
| 1c5z | 0.011 | **-5.9** | 锚点吻合 ✓ |

FP16 量化噪声（0.0039Å 半步）**不影响 BFGS 收敛到正确构象**。

## 关键推论

**E1 失败不是精度问题，是实现问题。**

E1 失败（15M 能量、No results）的原因 = struct 类型变更导致的
m_cl 布局/类型不一致（host FP32 写入 → device __half 读出的
位模式不匹配），而非 FP16 精度本身不足以支撑 BFGS。

Step 0 证明了：**只要 struct 类型不变，仅对值做 FP16 round-trip
量化，BFGS 完全正常工作。**

## 对 Q6-IR 的意义

Q6-IR 的双 kernel 设计恰好避免了 E1 的实现问题：
- 主线 struct 类型保持 FP32 不变
- Phase A kernel 内部使用局部 __half 表示
- kernel 边界做显式 FP16↔FP32 转换

**Q6-IR 的 FP16 精度前提已验证可行。** 剩余工作 = 双 kernel
架构实现（2-3 会话）。
