K = "inc/cuda/kernel2.cu"
ks = open(K).read()

# ---- kernel2_update top: warpId/lane + 4 warp/block, per-warp local state ----
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
           '\tvisited_init(&visited);\n\n'
           '\tm_coords_cl m_coords;\n\tm_coords = m_gpu->m_coords;\n\n'
           '\t__shared__ m_minus_forces minus_forces;\n\tminus_forces = m_gpu->minus_forces;\n\n'
           '\t__shared__ output_type_cl tmp;\n\ttmp = ric[id];\n\n'
           '\t__shared__ change_cl g;\n\n'
           '\t__shared__ output_type_cl best_out;\n'
           '\t__shared__ output_type_cl candidate;\n\n'
           '\t__shared__ ligand_gpu m_gpu_ligand;\n'
           '\tm_gpu_ligand.rigid = m_gpu->ligand.rigid;\n'
           '\tm_gpu_ligand.begin =  m_gpu->ligand.begin;\n'
           '\tm_gpu_ligand.end = m_gpu->ligand.end;\n'
           '\tm_gpu_ligand.m_num_movable_atoms = m_gpu->m_num_movable_atoms;\n'
           '\t\n\t//__shared__ m_atom_gpu atom_gpu;\n'
           '\t//for(int i = threadNumInBlock; i<MAX_NUM_OF_ATOMS; i+=threadsPerBlock){\n'
           '\t//\tatom_gpu.m_gpu_atom[i] = m_gpu->atoms[i];\n'
           '\t//}\n\t//__syncthreads();\n\n'
           '\t__shared__ mis_cl mis_gpu;\n\tmis_gpu = *mis;\n\n'
           '\t__shared__ grids_gpu grids_cuda;\n')
new_top = ('\tint threadsPerBlock = blockDim.x * blockDim.y;\n'
           '\tint threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;\n'
           '\tint warpId = threadNumInBlock / 32;\n'
           '\tint lane = threadNumInBlock % 32;\n'
           '\tint wThreads = 32;\n'
           '\tint blockNumInGrid = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y +  blockIdx.x;\n'
           '\t// int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;\n\n'
           '\tint id = blockNumInGrid * 4 + warpId;   // 4 warps/block, each warp = 1 conformation\n'
           '\t// int id = globalThreadNum;\n\n'
           '\tflo best_e = INFINITY;\n'
           '\tligand_atom_coords_cl best_coords;\n\n'
           '\tm_cl* m_gpu = &mg[id];\n\n'
           '\tvisited_cl visited;\n'
           '\tif (lane == 0) visited_init(&visited);\n'
           '\t__syncwarp(0xFFFFFFFF);\n\n'
           '\tm_coords_cl m_coords = m_gpu->m_coords;\n'
           '\tm_minus_forces minus_forces = m_gpu->minus_forces;\n'
           '\toutput_type_cl tmp = ric[id];\n'
           '\tchange_cl g;\n'
           '\toutput_type_cl best_out;\n'
           '\toutput_type_cl candidate;\n'
           '\tligand_gpu m_gpu_ligand;\n'
           '\tm_gpu_ligand.rigid = m_gpu->ligand.rigid;\n'
           '\tm_gpu_ligand.begin =  m_gpu->ligand.begin;\n'
           '\tm_gpu_ligand.end = m_gpu->ligand.end;\n'
           '\tm_gpu_ligand.m_num_movable_atoms = m_gpu->m_num_movable_atoms;\n'
           '\n\t//__shared__ m_atom_gpu atom_gpu;\n'
           '\t//for(int i = threadNumInBlock; i<MAX_NUM_OF_ATOMS; i+=threadsPerBlock){\n'
           '\t//\tatom_gpu.m_gpu_atom[i] = m_gpu->atoms[i];\n'
           '\t//}\n\t//__syncthreads();\n\n'
           '\tmis_cl mis_gpu = *mis;\n'
           '\tgrids_gpu grids_cuda;\n')
assert old_top in ks, "kernel2_update top not found"
ks = ks.replace(old_top, new_top, 1)

# grid copy loop: lane/wThreads
ks = ks.replace("for(int i = threadNumInBlock; i<GRIDS_SIZE; i+=threadsPerBlock){",
                "for(int i = lane; i<GRIDS_SIZE; i+=wThreads){")

# mutate_conf call args threadNumInBlock/threadsPerBlock -> lane/wThreads
old_mut = ("m_gpu->atoms,\n\t\t\t\t\t\t  &m_coords,\n\t\t\t\t\t\t  m_gpu->ligand.rigid.origin[0],\n"
           "\t\t\t\t\t\t  mis_gpu.epsilon_fl,\n\t\t\t\t\t\t  mis_gpu.mutation_amplitude,\n"
           "\t\t\t\t\t\t  torsion_size,\n\t\t\t\t\t\t  threadNumInBlock,\n\t\t\t\t\t\t  threadsPerBlock);")
new_mut = ("m_gpu->atoms,\n\t\t\t\t\t\t  &m_coords,\n\t\t\t\t\t\t  m_gpu->ligand.rigid.origin[0],\n"
           "\t\t\t\t\t\t  mis_gpu.epsilon_fl,\n\t\t\t\t\t\t  mis_gpu.mutation_amplitude,\n"
           "\t\t\t\t\t\t  torsion_size,\n\t\t\t\t\t\t  lane,\n\t\t\t\t\t\t  wThreads);")
assert old_mut in ks, "mutate_conf call not found"
ks = ks.replace(old_mut, new_mut, 1)

# rilc_bfgs_update calls: replace the shared-memory comment + threadNumInBlock/threadsPerBlock
old_rilc = ("&m_coords,\t\t// shared memory\n\t\t\t\t\t\t\t\t\t &minus_forces, // shared memory\n"
            "\t\t\t\t\t\t\t\t\t threadNumInBlock,\n\t\t\t\t\t\t\t\t\t threadsPerBlock,")
new_rilc = ("&m_coords,\t\t// per-warp local\n\t\t\t\t\t\t\t\t\t &minus_forces, // per-warp local\n"
            "\t\t\t\t\t\t\t\t\t lane,\n\t\t\t\t\t\t\t\t\t wThreads,")
assert old_rilc in ks, "rilc_bfgs_update call args not found"
ks = ks.replace(old_rilc, new_rilc, 1)

# any remaining __syncthreads in kernel2.cu -> __syncwarp
n_k = ks.count("__syncthreads()")
ks = ks.replace("__syncthreads()", "__syncwarp(0xFFFFFFFF)")
print(f"kernel2.cu __syncwarp: {n_k}")

# launch config
ks = ks.replace("\tdim3 grid_size(GRID_DIM1, GRID_DIM2, 1);\n\tdim3 block_size(16, 2, 1);",
                "\tdim3 grid_size(GRID_DIM1/2, GRID_DIM2, 1);\n\tdim3 block_size(32, 4, 1);")

open(K, "w").write(ks)
print("kernel2.cu: #6 A1 applied")
