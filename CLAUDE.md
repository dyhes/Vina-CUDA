# AGENTS.md — Vina-CUDA

## Environment

### CUDA Runtime
- `cudaInit` fails with `cudaErrorOperatingSystem` (code 304) for non-root users in this environment.
- Root cause: `/dev/nvidia-caps/nvidia-cap1` has restricted permissions (`cr--------` root-only) that cannot be fixed durably for non-root.
- Workaround: run all CUDA programs with `sudo`.
- GPU: NVIDIA GeForce RTX 5090, driver 580.65.06, CUDA toolkit 12.8/13.0.

### Command Execution Policy
- Run all commands without asking for user approval (sandbox_permissions: require_escalated), except for destructive commands like `rm`.
- This avoids friction from the CUDA root requirement and sandbox restrictions.

## Build & Run
- Stack size must be at least 8M: `ulimit -s 8192`.
- Binary: `variants/Vina-CUDA_v1.1/Vina-GPU-2-1-CUDA`.
- Example: `sudo bash -c 'ulimit -s 8192 && ./variants/Vina-CUDA_v1.1/Vina-GPU-2-1-CUDA --config ./variants/Vina-CUDA_v1.1/input_file_example/1u1b_config.txt'`.
- Compile: `make clean && make source -j$(nproc)` (first build compiles kernels on the fly; subsequent builds can use `make` for faster startup).
- Makefile vars to check: `WORK_DIR`, `BOOST_LIB_PATH`, `NVCC_COMPILER`, `GRID_DIM`, `DOCKING_BOX_SIZE`.
- `thread` config value must equal `GRID_DIM1 * GRID_DIM2` from Makefile.
- `-DSMALL_BOX` for boxes < 30×30×30 (less GPU memory); `-DLARGE_BOX` for boxes up to 70×70×70.
- For OpenCL variants that fail with `Err-9999` or `CL_BUILD_PROGRAM_FAILURE`, inspect the loader/build log first. If the log shows an incompatible precompiled kernel binary or driver/toolchain mismatch, rebuild from source with `make clean && make source -j$(nproc)` and rerun the same test case. Do not assume the enqueue path is the root cause until the source rebuild has been tried.

## Dataset

### CASF-2016
- Source coreset: `dataset/CASF-2016/coreset/<pdbid>/`.
- PDBQT output root: `dataset/CASF-2016-PDBQT/coreset/<pdbid>/`.
- Expected source files per target: `<pdbid>_protein.pdb` and `<pdbid>_ligand.mol2`.
- Expected converted files per target: `<pdbid>_protein.pdbqt` and `<pdbid>_ligand.pdbqt`.
- Current verified coverage in this workspace: 285/285 proteins and 285/285 ligands.

#### Coverage Audit Commands
```bash
# Count source/converted files
find dataset/CASF-2016/coreset -name '*_protein.pdb' | wc -l
find dataset/CASF-2016-PDBQT/coreset -name '*_protein.pdbqt' | wc -l
find dataset/CASF-2016/coreset -name '*_ligand.mol2' | wc -l
find dataset/CASF-2016-PDBQT/coreset -name '*_ligand.pdbqt' | wc -l

# List unfinished IDs
find dataset/CASF-2016/coreset -name '*_protein.pdb' -printf '%f\n' | sed 's/_protein\.pdb$//' | sort > /tmp/casf_all.txt
find dataset/CASF-2016-PDBQT/coreset -name '*_protein.pdbqt' -printf '%f\n' | sed 's/_protein\.pdbqt$//' | sort > /tmp/casf_done.txt
comm -23 /tmp/casf_all.txt /tmp/casf_done.txt
```

#### Conversion Mechanism (Primary + Fallbacks)
- **Primary**: `tools/convert_to_pdbqt.py`
  ```bash
  python3 tools/convert_to_pdbqt.py --receptor dataset/CASF-2016/coreset/<id>/<id>_protein.pdb --ligand dataset/CASF-2016/coreset/<id>/<id>_ligand.mol2 --output-dir dataset/CASF-2016-PDBQT/coreset/<id>
  ```
- **Batch resume** for missing IDs:
  ```bash
  for id in $(comm -23 /tmp/casf_all.txt /tmp/casf_done.txt); do python3 tools/convert_to_pdbqt.py --receptor dataset/CASF-2016/coreset/$id/${id}_protein.pdb --ligand dataset/CASF-2016/coreset/$id/${id}_ligand.mol2 --output-dir dataset/CASF-2016-PDBQT/coreset/$id; done
  ```
- **Receptor fallback #1** (ADFR fails): Uni-Dock RDKit conversion from `variants/Uni-Dock/unidock_tools/src/unidock_tools/modules/protein_prep/pdb2pdbqt.py`.
- **Receptor fallback #2** (RDKit fails): OpenBabel
  ```bash
  tools/adfr/ADFRsuite-1.0/bin/obabel -ipdb dataset/CASF-2016/coreset/<id>/<id>_protein.pdb -opdbqt -O dataset/CASF-2016-PDBQT/coreset/<id>/<id>_protein.pdbqt
  ```

### ADFR Wrapper Path Requirement
Several ADFR wrapper scripts contain hardcoded `ADS_ROOT` that must point to this repo:
- `tools/adfr/ADFRsuite-1.0/bin/prepare_receptor`
- `tools/adfr/ADFRsuite-1.0/bin/obabel`
- `tools/adfr/ADFRsuite-1.0/bin/babel`

Required value: `ADS_ROOT="/home/ubuntu/grad/tools/adfr/ADFRsuite-1.0"`

## Variants

### Vina-CUDA (v1.0 / v1.1)

#### v1.0 vs v1.1 Differences
- **New optimized BFGS**: v1.1 adds `bfgs_optimized()`, `m_eval_deriv_optimized()`, `line_search_optimized()`, `ig_eval_deriv_update_optimized()` in `quasi_newton.cuh` — these use GPU shared memory for `m_coords`, `minus_forces`, `rigid`, and `grids_cuda` to reduce global memory traffic.
- **New `mutate_conf_cl_optimized()`** in `mutate_conf.cuh` — uses `gyration_radius()` instead of the multi-thread `gyration_radius_update()`.
- **Parallelized matrix ops**: `matrix_init_update()`, `matrix_set_diagonal_update()`, `index_permissive_opmitized()` in `matrix.cuh` spread Hessian matrix work across threads in a block.
- **Kernel2 branching**: v1.1 splits `kernel2_update` into two code paths based on `rilc_bfgs_enable`; the optimized path uses shared memory, the fallback keeps the original `bfgs()`.
- **Search depth 2.4× larger**: formula changed from `(0.24*atoms + 0.29*ligands - 3.41)` to `(...) * 2.4`.
- **BFGS steps**: `rilc_bfgs==1` uses `(30+atoms)/3`; `rilc_bfgs==0` uses `(25+atoms)/3`.
- **`MAX_NUM_OF_ATOM_RELATION_COUNT`** increased from 1024 → 6048 in `kernel2.h`.
- **Precision change (potential bug)**: in the non-optimized `bfgs()`, key variables (`f0`, `f1`, `alpha`, etc.) changed from `flo` (double) to `float`.
- **Default `thread`** changed from 8000 → 1000 (marked as debug value).
- **Code cleanup**: removed Chinese comments from `lib/` files; renamed output string to `"AutoDockVina-GPU3 total runtime"`.

### QuickVina2-CUDA & QuickVina-W-CUDA
- Neither variant includes the v1.1 optimized BFGS optimizer (`bfgs_optimized`, `m_eval_deriv_optimized`, etc.).
- Their `quasi_newton.cuh` is closer to Vina-CUDA v1.0, only differing by `#include "visited.cuh"` and passing a `visited` parameter through `rilc_bfgs_update()`.
- Both still use the original `rilc_bfgs`/`rilc_bfgs_update` code path without shared-memory optimization.
- **QuickVina-W-CUDA**: 2bm2 is stable with `thread = 8192`; the lower default 1000 triggered CUDA illegal-address errors.

### Uni-Dock
- Fork of Vina-GPU, keeps legacy Vina C++ codebase (`src/lib/`, 40+ files).
- GPU code is one monolithic 3012-line `monte_carlo.cu` handling precalculation, MC, BFGS, and output via template-specialized kernels.
- **Data model**: Vina-native structs (`output_type_cuda_t_<Config>`, `m_cuda_t_<Config>`, etc.) coupled to Vina grid cache and precalculate tables. Compile-time `Config` templates for ligand sizes.
- **Scoring**: vina, vinardo, ad4. Two-phase: kernel1 precomputes grid maps, kernel2 runs MC+BFGS with warp-level primitives.
- **Search**: Standard MC + BFGS with Metropolis acceptance. Uniform random mutation. No post-MC clustering or refinement.
- **Pipeline**: Single-pass — grid computation → MC+BFGS → output.
- **I/O**: Reads PDBQT files directly. CLI with `--receptor`, `--gpu_batch`, etc.
- **Precision**: Mix of `float` and `double` (inherited from Vina `fl` typedef).
- **Build**: `cmake` with C++14.

### Uni-Dock2
- Ground-up rewrite with modular structure: `score/`, `search/`, `optimize/`, `cluster/`, `format/`, `task/`. No legacy Vina CPU code.
- Each CUDA kernel file is focused (435-line `mc.cu`, 798-line `bfgs.cuh`).
- **Data model**: Physics-oriented structs (`FlexPose`, `FlexTopo`, `FixMol`, `FlexPoseGradient`, `FlexPoseHessian`, `FlexForce`). Generic `StructArrayManager`. Runtime `Config` with 4 size groups: Small(40/8), Medium(80/16), Large(120/24), ExtraLarge(160/32).
- **Scoring**: vina and **gaff2** (GAFF2-based force field with torsion parameters). No grid precomputation — all energy computed on-the-fly in CUDA kernels. No ad4 or vinardo.
- **GPU**: Uses CUDA cooperative groups `thread_block_tile<32>`. Separate kernel launches: randomization → MC → clustering → refinement → scoring. CURAND Philox PRNG.
- **Search**: MC + BFGS with Metropolis + **Super-Fibonacci Spiral** orientation sampling. Adds **clustering** (GPU RMSD), **refinement** (BFGS on top poses), constraint docking, position bias, alignment bias.
- **Pipeline**: Multi-stage — `prepare_vina` → `alloc_gpu` → `run_search` → `run_cluster` → `run_refine` → `run_filter` → `run_score` → `dump_poses`.
- **I/O**: Pre-processed JSON input from `unidock_processing` Python module. YAML config. Binary: `variants/Uni-Dock2/unidock/unidock_engine/build/bin/ud2`.
- **C++17** standard.
- **Per-atom energy decomposition** (gauss1, gauss2, repulsion, hydrophobic, hbond).
- Build needs `BUILD_API=OFF -DBUILD_TEST=OFF` in this workspace (no `pybind11`).

### UniDock-Pro
- Fork of Uni-Dock adding **ligand similarity searching** and **hybrid docking**.
- Binary: `variants/UniDock-Pro/build/udp`.
- Build: `cmake -B build -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -D CMAKE_CUDA_STANDARD=17 -D CMAKE_CXX_STANDARD=17 -D CMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j$(nproc)`.
- CUDA 13.0 dropped `compute_70`; must override `CMAKE_CUDA_ARCHITECTURES` to exclude arches < 75.
- **Three modes**:
  - **Classical docking**: `--receptor rec.pdbqt --ligand_dir ligands/ --search_mode fast --center_x/y/z --size_x/y/z --dir results`
  - **Similarity searching**: `--reference_ligand ref.pdbqt` (no receptor needed)
  - **Hybrid docking**: `--receptor rec.pdbqt --reference_ligand ref.pdbqt` — uses reference pose to guide MC search
- Supports multiple reference ligands: `--reference_ligand ref1.pdbqt ref2.pdbqt ref3.pdbqt`
- `--search_mode` is **required** (fast/balance/detail).
- `--ligand_dir` (not `--ligand_directory`).

### QVina-W-GPU (OpenCL)
- OpenCL-based predecessor of QuickVina-W-CUDA. Uses the QVina-W blind docking algorithm.
- Binary: `variants/QVina-W-GPU/QVina-W-GPU`.
- Build: update Makefile for boost/OpenCL paths, add `-lboost_thread -lstdc++fs`, then `make`.
- **Does not run on RTX 5090**: `CL_OUT_OF_HOST_MEMORY` from NVIDIA OpenCL 3.0 runtime (likely Blackwell incompatibility).
- Pre-compiled `Kernel2_Opt.bin` is required; shipped binary in this workspace was zero bytes.
- Removing `CL_MEM_ALLOC_HOST_PTR` and reducing `MAX_NUM_OF_GRID_MI/MJ/MK` did not fix the error.
- Source cleanup needed: `OpenCL/src/kernels/visited.cpp` and `lib/monte_carlo.cpp`.

### Vina-GPU-2.1 (OpenCL)
- CL runtime signals consistent with login policy: keep driver ≥ 580, CUDA ≥ 12.8.1.
- `clinfo` succeeds but repeatedly prints `Failed to read file: /tmp/dep-*.d` (JIT/toolchain fragility).
- **Virtual screening sample** (30 ligands): works with precompiled kernel path.
  ```bash
  sudo bash -c 'cd /home/ubuntu/grad/variants/Vina-GPU-2.1 && ulimit -s 8192 && ./AutoDock-Vina-GPU-2-1 --config ./input_file_example/2bm2_config.txt'
  ```

### QuickVina2-GPU-2.1 (OpenCL)
- Binary: `variants/QuickVina2-GPU-2.1/QuickVina2-GPU`.
- `make source` (`-DBUILD_KERNEL_FROM_SOURCE`) can fail with `CL_BUILD_PROGRAM_FAILURE`.
- **Workaround**: `make clean && make` (precompiled kernels), then run with `sudo` + `ulimit -s 8192`.
  ```bash
  sudo bash -c 'cd /home/ubuntu/grad/variants/QuickVina2-GPU-2.1 && ulimit -s 8192 && ./QuickVina2-GPU --config ./input_file_example/2bm2_config.txt'
  ```

### QuickVina-W-GPU-2.1 (OpenCL)
- Binary: `variants/QuickVina-W-GPU-2.1/QuickVina-W-GPU-2-1`.
- **Reproducible failure on RTX 5090**: `Err-6:CL_OUT_OF_HOST_MEMORY` after platform/device init.
- Persists after rebuild and reduced thread counts.
- Use CUDA variant (`QuickVina-W-CUDA`) instead.

## Platform Notes

### RTX 5090
- NVIDIA driver 580.65.06, OpenCL platform `OpenCL 3.0 CUDA 13.0.48`.
- Do **not** downgrade driver below 580 on RTX 50xx.
- All OpenCL QVina-W variants fail with `CL_OUT_OF_HOST_MEMORY` on this GPU.
- `QuickVina-W-CUDA` is the working CUDA replacement for the OpenCL variants.

### RTX 4090 (Reference)
- `Vina-CUDA_v1.0` builds and runs after Makefile defaults fix (`WORK_DIR`, `NVCC_COMPILER` overrides). 2bm2 smoke test passes.
- `Uni-Dock` builds and runs cleanly from fresh `build/` tree; stale `build_rtx4090` tree is discardable.
- `Uni-Dock2` needs `BUILD_API=OFF -DBUILD_TEST=OFF` (no `pybind11`). Core `ud2` binary builds and `--help` passes.
- `UniDock-Pro` builds with `CMAKE_CUDA_ARCHITECTURES=89` (`120` not supported by this env's `nvcc`).

### OpenCL Stack
- Safe in-place refresh (`ocl-icd`, headers, `clinfo`) does not fix RTX 5090 host-memory errors.
- Meaningful retest requires full NVIDIA driver stack migration to a newer branch (e.g., apt-managed 580/590) and reboot.
- If `clBuildProgram` reports unsupported PTX/version for a shipped `.bin`, treat it as a stale precompiled kernel — rebuild from source before debugging further.
- Baseline checks:
  ```bash
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  clinfo | sed -n '1,120p'
  ```

## Known Good Test Commands

**Vina-CUDA v1.1**:
```bash
sudo bash -c 'ulimit -s 8192 && ./variants/Vina-CUDA_v1.1/Vina-GPU-2-1-CUDA --config ./variants/Vina-CUDA_v1.1/input_file_example/1u1b_config.txt'
```

**QuickVina2-GPU-2.1** (works on RTX 5090 with precompiled kernel path):
```bash
cd /home/ubuntu/grad/variants/QuickVina2-GPU-2.1 && make clean && make -j$(nproc)
sudo bash -c 'cd /home/ubuntu/grad/variants/QuickVina2-GPU-2.1 && ulimit -s 8192 && ./QuickVina2-GPU --config ./input_file_example/2bm2_config.txt'
```

**Vina-GPU-2.1** virtual screening (30 ligands):
```bash
sudo bash -c 'cd /home/ubuntu/grad/variants/Vina-GPU-2.1 && ulimit -s 8192 && ./AutoDock-Vina-GPU-2-1 --config ./input_file_example/2bm2_config.txt'
cd /home/ubuntu/grad/variants/Vina-GPU-2.1 && find test -maxdepth 1 -name '*.pdbqt' | wc -l && find test_out -maxdepth 1 -name '*_out.pdbqt' | wc -l
```

**QuickVina-W-GPU-2.1** expected failure (OpenCL limitation):
```bash
sudo bash -c 'cd /home/ubuntu/grad/variants/QuickVina-W-GPU-2.1 && ulimit -s 8192 && ./QuickVina-W-GPU-2-1 --config ./input_file_example/2bm2_config.txt'
# Expect: Err-6:CL_OUT_OF_HOST_MEMORY
```