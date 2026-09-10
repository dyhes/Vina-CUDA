import re

# ============ A1: per-warp localize + __syncwarp + warp-slotting ============
# Apply to quasi_newton.cuh and kernel2.cu in m1_fusion_p1_gamma

Q = "inc/cuda/quasi_newton.cuh"
K = "inc/cuda/kernel2.cu"
qs = open(Q).read()
ks = open(K).read()

# --- 1. global __syncthreads() -> __syncwarp(0xFFFFFFFF) in quasi_newton.cuh ---
n_q = qs.count("__syncthreads()")
qs = qs.replace("__syncthreads()", "__syncwarp(0xFFFFFFFF)")
print(f"quasi_newton.cuh: replaced {n_q} __syncthreads() -> __syncwarp")

# --- 2. find_change_index_read: warp-slot the two __shared__ scratch arrays ---
old_fc = ('\t__shared__ flo position_derivative_tmp[MAX_NUM_OF_RIGID][3];\n'
          '\tflo position_derivative[MAX_NUM_OF_RIGID][3];\n'
          '\t__shared__ flo orientation_derivative_tmp[MAX_NUM_OF_RIGID][3];\n'
          '\tflo orientation_derivative[MAX_NUM_OF_RIGID][3];\n')
new_fc = ('\tint _wid = threadIdx.x / 32;\n'
          '\t__shared__ flo position_derivative_tmp[4][MAX_NUM_OF_RIGID][3];\n'
          '\tflo position_derivative[MAX_NUM_OF_RIGID][3];\n'
          '\t__shared__ flo orientation_derivative_tmp[4][MAX_NUM_OF_RIGID][3];\n'
          '\tflo orientation_derivative[MAX_NUM_OF_RIGID][3];\n')
assert old_fc in qs, "find_change_index_read shared block not found"
qs = qs.replace(old_fc, new_fc, 1)

# index the two scratch arrays with _wid
qs = qs.replace("position_derivative_tmp[i][k]", "position_derivative_tmp[_wid][i][k]")
qs = qs.replace("orientation_derivative_tmp[i][k]", "orientation_derivative_tmp[_wid][i][k]")

# --- 3. POT_deriv_update: warp-slot tmp / d_test ---
# find the function's __shared__ tmp[MAX...] and d_test[MAX...]
old_pot = ('\t__shared__ flo tmp[MAX_NUM_OF_DIMENSIONS_OF_MATIRX];\n')
new_pot = ('\tint _wid = threadIdx.x / 32;\n'
           '\t__shared__ flo tmp[4][MAX_NUM_OF_DIMENSIONS_OF_MATIRX];\n')
assert old_pot in qs, "POT tmp shared not found"
qs = qs.replace(old_pot, new_pot, 1)
# d_test
old_dt = ('\t__shared__ float d_test[MAX_NUM_OF_DIMENSIONS_OF_MATIRX];\n')
new_dt = ('\t__shared__ float d_test[4][MAX_NUM_OF_DIMENSIONS_OF_MATIRX];\n')
assert old_dt in qs, "POT d_test shared not found"
qs = qs.replace(old_dt, new_dt, 1)
# index usages: tmp[threadNumInBlock] -> tmp[_wid][threadNumInBlock]; tmp[threadNumInBlock + threadsPerBlock*i] -> tmp[_wid][...]
# careful: only within POT_deriv_update. Use a targeted replace on the specific lines.
qs = qs.replace("tmp[threadNumInBlock] += tmp[threadNumInBlock + threadsPerBlock * i];",
                "tmp[_wid][threadNumInBlock] += tmp[_wid][threadNumInBlock + threadsPerBlock * i];")
qs = qs.replace("if (threadNumInBlock < ( n % threadsPerBlock)) {\n\t\t\ttmp[threadNumInBlock] += tmp[threadNumInBlock + tmp_1 * threadsPerBlock];",
                "if (threadNumInBlock < ( n % threadsPerBlock)) {\n\t\t\ttmp[_wid][threadNumInBlock] += tmp[_wid][threadNumInBlock + tmp_1 * threadsPerBlock];")
# d_test indexing
qs = qs.replace("d_test[threadNumInBlock] = d->coords[i][k];",
                "d_test[_wid][threadNumInBlock] = d->coords[i][k];")
qs = qs.replace("d_test[threadNumInBlock + threadsPerBlock * j]",
                "d_test[_wid][threadNumInBlock + threadsPerBlock * j]")

# --- 4. rilc_bfgs_update: __shared__ state -> local ---
qs = qs.replace("\t__shared__ output_type_cl xp;\n\txp = *x;\n",
                "\toutput_type_cl xp = *x;\n")
qs = qs.replace("\t__shared__ change_cl gp;\n\tgp = *g;\n",
                "\tchange_cl gp = *g;\n")
qs = qs.replace("\t\t__shared__ change_cl g_orig;\n\t\tg_orig = *g;\n",
                "\t\tchange_cl g_orig = *g;\n")
qs = qs.replace("\t\t__shared__ output_type_cl x_orig;\n\t\tx_orig = *x;\n",
                "\t\toutput_type_cl x_orig = *x;\n")
qs = qs.replace("\t\t__shared__ change_cl d;\n\t\td = *g;\n",
                "\t\tchange_cl d = *g;\n")

open(Q, "w").write(qs)
print("quasi_newton.cuh: A1 edits applied")

# ============ kernel2.cu ============
# --- 5. warp-id / lane compute + remove __shared__ state in kernel2_update ---
old_top = ('\tint threadsPerBlock = blockDim.x * blockDim.y;\n'
           '\tint threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;\n'
           '\tint blockNumInGrid = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y +  blockIdx.x;\n'
           '\t// int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;\n\n'
           '\tint id = blockNumInGrid;\n'
           '\t// int id = globalThreadNum;\n\n'
           '\tflo best_e = INFINITY;\n'
           '\tligand_atom_coords_cl best_coords;\n\n'
           '\tm_cl* m_gpu = &mg[id];\n\n'
           '\t__shared__ visited_cl visited;\n'
           '\tvisited_init(&visited);\n\n')
new_top = ('\tint threadsPerBlock = blockDim.x * blockDim.y;\n'
           '\tint threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;\n'
           '\tint warpId = threadNumInBlock / 32;\n'
           '\tint lane = threadNumInBlock % 32;\n'
           '\tint wThreads = 32;\n'
           '\tint blockNumInGrid = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;\n'
           '\t// int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;\n\n'
           '\tint id = blockNumInGrid * 4 + warpId;   // 4 warps/block, each warp = 1 conformation\n'
           '\t// int id = globalThreadNum;\n\n'
           '\tflo best_e = INFINITY;\n'
           '\tligand_atom_coords_cl best_coords;\n\n'
           '\tm_cl* m_gpu = &mg[id];\n\n'
           '\tvisited_cl visited;\n'
           '\tif (lane == 0) visited_init(&visited);\n'
           '\t__syncwarp(0xFFFFFFFF);\n\n')
assert old_top in ks, "kernel2_update top not found"
ks = ks.replace(old_top, new_top, 1)

# remaining __shared__ in kernel2_update body (m_coords, minus_forces, tmp, g, best_out, candidate, m_gpu_ligand, mis_gpu, grids_cuda)
ks = ks.replace("\t__shared__ m_coords_cl m_coords;\n\tm_coords = m_gpu->m_coords;\n",
                "\tm_coords_cl m_coords = m_gpu->m_coords;\n")
ks = ks.replace("\t__shared__ m_minus_forces minus_forces;\n\tminus_forces = m_gpu->minus_forces;\n",
                "\tm_minus_forces minus_forces = m_gpu->minus_forces;\n")
ks = ks.replace("\t__shared__ output_type_cl tmp;\n\ttmp = ric[id];\n",
                "\toutput_type_cl tmp = ric[id];\n")
ks = ks.replace("\t__shared__ change_cl g;\n",
                "\tchange_cl g;\n")
ks = ks.replace("\t__shared__ output_type_cl best_out;\n",
                "\toutput_type_cl best_out;\n")
ks = ks.replace("\t__shared__ output_type_cl candidate;\n",
                "\toutput_type_cl candidate;\n")
ks = ks.replace("\t__shared__ ligand_gpu m_gpu_ligand;\n",
                "\tligand_gpu m_gpu_ligand;\n")
ks = ks.replace("\t__shared__ mis_cl mis_gpu;\n\tmis_gpu = *mis;\n",
                "\tmis_cl mis_gpu = *mis;\n")
ks = ks.replace("\t__shared__ grids_gpu grids_cuda;\n",
                "\tgrids_gpu grids_cuda;\n")

# grid copy loop: use lane/wThreads instead of threadNumInBlock/threadsPerBlock
ks = ks.replace("for(int i = threadNumInBlock; i<GRIDS_SIZE; i+=threadsPerBlock){",
                "for(int i = lane; i<GRIDS_SIZE; i+=wThreads){")
# after the grid copy loop there is __syncthreads(); -> already converted globally? kernel2.cu not yet.
# convert __syncthreads in kernel2.cu
n_k = ks.count("__syncthreads()")
ks = ks.replace("__syncthreads()", "__syncwarp(0xFFFFFFFF)")
print(f"kernel2.cu: replaced {n_k} __syncthreads() -> __syncwarp")

# pass lane/wThreads to helper calls inside kernel2_update (mutate_conf, rilc_bfgs_update, set_update, get_heavy_atom)
# These calls currently pass threadNumInBlock, threadsPerBlock -> replace with lane, wThreads
# But threadNumInBlock/threadsPerBlock are still used for warp-meta; we replace only the helper arg pairs.
# Safer: replace the specific 2-arg sequences in the calls we know.
for old, new in [
    ("m_gpu->atoms,\n\t\t\t\t\t\t  &m_coords,\n\t\t\t\t\t\t  m_gpu->ligand.rigid.origin[0],\n\t\t\t\t\t\t  mis_gpu.epsilon_fl,\n\t\t\t\t\t\t  mis_gpu.mutation_amplitude,\n\t\t\t\t\t\t  torsion_size,\n\t\t\t\t\t\t  threadNumInBlock,\n\t\t\t\t\t\t  threadsPerBlock);",
     "m_gpu->atoms,\n\t\t\t\t\t\t  &m_coords,\n\t\t\t\t\t\t  m_gpu->ligand.rigid.origin[0],\n\t\t\t\t\t\t  mis_gpu.epsilon_fl,\n\t\t\t\t\t\t  mis_gpu.mutation_amplitude,\n\t\t\t\t\t\t  torsion_size,\n\t\t\t\t\t\t  lane,\n\t\t\t\t\t\t  wThreads);"),
]:
    if old in ks:
        ks = ks.replace(old, new, 1)
        print("patched mutate_conf call")

# rilc_bfgs_update calls (two of them) pass threadNumInBlock, threadsPerBlock
ks = ks.replace("&m_coords,\t\t// shared memory\n\t\t\t\t\t\t\t\t\t &minus_forces, // shared memory\n\t\t\t\t\t\t\t\t\t threadNumInBlock,\n\t\t\t\t\t\t\t\t\t threadsPerBlock,",
                "&m_coords,\t\t// per-warp local\n\t\t\t\t\t\t\t\t\t &minus_forces, // per-warp local\n\t\t\t\t\t\t\t\t\t lane,\n\t\t\t\t\t\t\t\t\t wThreads,")
# set_update calls pass threadNumInBlock,threadsPerBlock
ks = ks.replace("m_gpu->atoms, m_gpu->m_num_movable_atoms, mis_gpu.epsilon_fl);",
                "m_gpu->atoms, m_gpu->m_num_movable_atoms, mis_gpu.epsilon_fl);")  # no-op guard

# launch config: 4 warps/block, grid = NCONF/4
ks = ks.replace("\tdim3 grid_size(GRID_DIM1, GRID_DIM2, 1);\n\tdim3 block_size(16, 2, 1);",
                "\tdim3 grid_size(GRID_DIM1/2, GRID_DIM2, 1);\n\tdim3 block_size(32, 4, 1);")

open(K, "w").write(ks)
print("kernel2.cu: A1 edits applied")
print("DONE")
