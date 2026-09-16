# Q4-8 Determinism/Reproducibility (2026-09-11)

> Status: PARTIAL FIX landed, residual documented. Read before making any
> determinism claim, reproducibility statement, same-seed bit-exact comparison,
> or "is this optimization change safe" A/B plan.

## What was found

1. **Same-seed runs are NOT bit-exact** in the Q4-6/Q4-7 line. Verified by
   RESCHK FNV-1a checksum of the raw results buffer (8192 × output_type_cl)
   after DtoH: two runs with `--seed 42` produce different checksums.
2. **Block0 is bit-deterministic** (BESTOUT probe: identical coords/energy
   across runs; LSTRACE per-LS `ls`/`fx` sequence identical; mutate-phase
   `V1 coord` trace identical). The divergence is in **other blocks**, and it
   is macroscopic (mode-2 rmsd 0.44 vs 0.35, mode-3..9 affinities ±0.1–0.4)
   — LSB flips in the GPU kernel flip LS threshold decisions and fork whole
   trajectories.
3. **racecheck (compute-sanitizer) census on 1pxn/seed42: 72 races.**
   - 49 × mutate_conf.cuh:306 (mutate_conf_cl_update divergent shared writes)
   - remainder: local_to_lab writes vs g_evaluate reads (m_coords cross-eval
     sync), quasi_newton.cuh:211–213/422/506/510/514/612/613/646

## Fixes landed (commit bf32836, Q3 line)

1. **Pairs-force accumulation race (the main one) FIXED via edge-coloring.**
   Host greedy edge-coloring of the pair graph (pairs sharing an atom get
   different colors; `std::vector<int> last_color(na,-1)` per ligand, colors
   h2d'd with mis) + device loop `for c in 0..num_colors { for pairs of
   color c (strided) { compute; minus_forces[a]-=f; minus_forces[b]+=f; }
   __syncthreads(); }`. No two same-color pairs touch an atom → no race;
   per-atom accumulation order is FIXED (color-major) → deterministic.
   num_colors travels in `mis_cl.q4_num_colors` (added to the h2d struct —
   NOT via `cudaMemcpyToSymbol`, see trap below).
2. **skip-row bitmap RMW → atomicAnd/atomicOr** (multi-thread RMW on the
   same shared word was a lost-update race even with disjoint bits).
3. **Barriers added**: set_update→GRID loop (all threads finish writing
   shared m_coords before any reads), pairs→POT_deriv (pair-force writes
   complete before tree reduction reads minus_forces across atoms),
   POT_deriv_update end (g complete before caller consumes).
4. **--seed passthrough**: binary supported it already (main.cpp L693,
   `auto_seed()` if absent); runner `run_cuda_vina` + `run_qvina_w_cuda`
   now pass `--seed {seed}` (`--seed 0` = auto). Verified: log prints
   "Using random seed: 42".

## Residual (NOT fixed)

Same-seed runs still differ in mode-2..9 rmsd (~0.1–0.4) — remaining races:
- **m_coords cross-eval sync**: ig_eval GRID loop writes minus_forces[i]
  (per-atom, no race) but set_update writes m_coords from ALL threads
  (same values — benign WAW) and racecheck still flags local_to_lab writes
  vs g_evaluate reads — the barrier coverage is not proven complete.
- **mutate divergent writes** (49 reports): `c->position[i] += ...` with
  `i` strided is benign (disjoint), but racecheck cannot distinguish
  same-value multi-writer from real races — needs a manual per-line audit.
- CASF distribution stats (aff mean|Δ|, Top-1 counts, medians) REMAIN VALID
  — the noise is MC-noise-magnitude.

## Traps hit (generalizable)

- **`extern __device__` in a host TU demotes to a host variable.** Declaring
  `extern __device__ unsigned char* q4_pair_color;` in main_procedure_cu.cpp
  (compiled as host code by nvcc, no -rdc) makes `cudaMemcpyToSymbol(sym,...)`
  fail with "invalid conversion from int to const void*" — the host compiler
  treats it as a plain extern int. Fixes: use
  `cudaGetSymbolAddress(&ptr, sym)` + plain `cudaMemcpy`, or carry scalars in
  a struct that is h2d'd anyway (mis_cl->q4_num_colors), or define the symbol
  in the kernel TU and pass it as a kernel PARAMETER.
- **gpu-vina-optimization SKILL.md's reference list can drift from disk** —
  `references/q4_8_determinism_2026-09-11.md` was cited in SKILL.md before the
  file existed; the write was refused until the file was created. Keep
  SKILL.md reference lists in sync with actual files at the end of a session.
- **`fmodf` vs while-loop for angle normalization**: the original
  `normalize_angle` while-loop with `M_PI` (double macro) promoted the whole
  angle reduction to FP64 (DSETP.F64/DFMA chains in SASS). Float-PI rewrite +
  `rintf`/`nearbyintf` still emitted I2F.F64.S64 (int→double→float path).
  Recheck SASS after any such rewrite.

## Validation standard for kernel changes (post-Q4-8)

- "Same-seed bit-exact" is NOT a valid standard (residual races).
- Use: same-seed affinity **distribution stats** — per-target |Δ| vs the
  self-noise band (v4 self-noise mean|Δ|=0.162, >0.5 count ~19–24), plus
  CASF Top-1 in the 171–184 family band, quoted as **median ± spread over
  ≥2 runs** (single runs cannot resolve band-edge placement; family spread
  is 13–16).
- Determinism work is a SEPARATE project (full shared-write audit), not a
  gate for performance/precision changes.
