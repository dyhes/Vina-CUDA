#include "stage_timer.h"
#include "gpu_timer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel2.h"
#include "mutate_conf.cuh"
#include "quasi_newton.cuh"
#include "matrix.cuh"
#include "assert.h"
#include "visited.cuh"
#include "host_probe.h"

// [v7] unified host probe: single definition (all TUs share one timeline)
#ifdef HOST_PROBE
HostProbe g_host_probe;
void HostProbe::mark(const char* name) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    double t = ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
    if (last && last[0]) fprintf(stderr, "PROBE %s : %.1f ms | T=%.1f\n", last, t - t0, t);
    t0 = t; last = name;
}
#endif

__constant__ mis_cl mis_cuda_const;  // [Q4-12] per-ligand scalar params (read-only)

// [v7] one-time CUDA warmup: driver init, allocator mapping, module load.
// Runs BEFORE per-target timing; excluded from E2E_DOCK (see handoff
// e2e_timing_and_batch §1.2 — ctx/module load is environment cost).
#ifdef HOST_PROBE
__global__ void __v7_warmup_kernel() {}
extern "C" void v7_warmup_cuda() {
	// [Q4-10a] zombie probes removed: the 1.14GB cudaMalloc+Free probe sized
	// the OLD grids buffer (Q4-1 shrank real grids to 136MB) and cudaFree
	// releases the pages anyway (prep_floor_anatomy: warmup malloc+free does
	// NOT help the real allocation); the 256MB memset was L2/page warmup for
	// a footprint that no longer exists. Kept: real ctx init semantics
	// (cudaSetDevice, small probe alloc+memset, warmup kernel launch for
	// module load, device sync) — measured to cost ~2-5ms total.
	checkCUDA(cudaSetDevice(0));
	float* dummy = NULL; checkCUDA(cudaMalloc((void**)&dummy, 4096));
	checkCUDA(cudaMemset(dummy, 0, 4096));
	__v7_warmup_kernel<<<1, 32>>>();
	checkCUDA(cudaFree(dummy));
	checkCUDA(cudaDeviceSynchronize());
}
#endif

__device__ inline void get_heavy_atom_movable_coords(output_type_cl *tmp, const m_cl *m, ligand_atom_coords_cl *coords)
{
	int counter = 0;
	for (int i = 0; i < m->m_num_movable_atoms; i++)
	{
		if (m->atoms[i].types[0] != EL_TYPE_H)
		{
			for (int j = 0; j < 3; j++)
				coords->coords[counter][j] = m->m_coords.coords[i][j];
			counter++;
		}
		else
		{
			// printf("\n kernel2 --> 12: removed H atom coords in get_heavy_atom_movable_coords()!");
		}
	}
	// assign 0 for others
	for (int i = counter; i < MAX_NUM_OF_ATOMS; i++)
	{
		for (int j = 0; j < 3; j++)
			coords->coords[i][j] = 0;
	}
}

__device__ inline void get_heavy_atom_movable_coords_update(output_type_cl *tmp, const m_cl *m, ligand_atom_coords_cl * coords, m_coords_cl* m_coords)
{
	int counter = 0;
	for (int i = 0; i < m->m_num_movable_atoms; i++)
	{
		if (m->atoms[i].types[0] != EL_TYPE_H)
		{
			for (int j = 0; j < 3; j++)
				coords->coords[counter][j] = m_coords->coords[i][j];
			counter++;
		}
		else
		{
			// printf("\n kernel2 --> 12: removed H atom coords in get_heavy_atom_movable_coords()!");
		}
	}
	// assign 0 for others
	for (int i = counter; i < MAX_NUM_OF_ATOMS; i++)
	{
		for (int j = 0; j < 3; j++)
			coords->coords[i][j] = 0;
	}
}

// Generate a random number according to step
__device__ inline flo generate_n(flo *pi_map, const int step)
{ // // delete the constant
	return fabs(pi_map[step]) / M_PI;
}

__device__ inline bool metropolis_accept(flo old_f, flo new_f, flo temperature, flo n)
{
	if (new_f < old_f)
		return true;
	const flo acceptance_probability = expf((old_f - new_f) / temperature); // [Q4-7M-M1] FP64 exp -> FP32 expf (17.8M FP64 ops eliminated, ncu-confirmed)
	bool res = n < acceptance_probability;
	// return n < acceptance_probability;  // why not return the res ????
	return res;
}

//__device__ flo m_cutoff_sqr_r =  555.000;
//__device__ flo m_cutoff_sqr_rr =  0;
//__device__ int n = 11;
//__device__ int m = 22;
//__device__ int q = 33;
//__device__ int t = 44;

__global__ void kernel2(
	const output_type_cl *ric,	   // delete the __global
	m_cl *mg,					   // delete the __global
	pre_cl *pre,				   // delete the __constant
	grids_cl *grids,			   // delete the __constant
	random_maps *random_maps,	   // delete the __constant
	ligand_atom_coords_cl *coords, // delete the __global
	output_type_cl *results,	   // delete the __global
	const mis_cl *mis,			   // delete the __global
	const int torsion_size,
	const int search_depth,
	const int max_bfgs_steps,
	const int rilc_bfgs_enable)
{
	int gx = blockIdx.x * blockDim.x + threadIdx.x;
	int gy = blockIdx.y * blockDim.y + threadIdx.y;
	int gs = gridDim.x * blockDim.x;
	int gl = gx + gy * gs;

	// printf("the gl is : %d \n,", gl);

	flo best_e = 1.0e38f;
	output_type_cl best_out;
	ligand_atom_coords_cl best_coords;

	for (int gll = gl; gll < mis->thread; gll += mis->total_wi)
	{
		m_cl m = *mg;

		output_type_cl tmp = ric[gll];

		change_cl g;

		output_type_cl candidate;

		for (int step = 0; step < search_depth; step++)
		{
			candidate = tmp;

			int map_index = (step + gll * search_depth) % MAX_NUM_OF_RANDOM_MAP;
			mutate_conf_cl(map_index,
						   &candidate,
						   random_maps->int_map,
						   random_maps->sphere_map,
						   random_maps->pi_map,
						   m.ligand.begin,
						   m.ligand.end,
						   m.atoms,
						   &m.m_coords,
						   m.ligand.rigid.origin[0],
						   mis->epsilon_fl,
						   mis->mutation_amplitude,
						   torsion_size);
			// printf(" stop the program running"); assert(true);
			__syncthreads();
			if (rilc_bfgs_enable == 1)
			{
				rilc_bfgs(&candidate,
						  &g,
						  &m,
						  pre,
						  grids,
						  mis,
						  torsion_size,
						  max_bfgs_steps);
			}
			else
			{
				bfgs(&candidate,
					 &g,
					 &m,
					 pre,
					 grids,
					 mis,
					 torsion_size,
					 max_bfgs_steps);
			}

			flo n = generate_n(random_maps->pi_map, map_index);

			if (step == 0 || metropolis_accept(tmp.e, candidate.e, 1.2, n))
			{

				tmp = candidate;

				set(&tmp, &m.ligand.rigid, &m.m_coords,
					m.atoms, m.m_num_movable_atoms, mis->epsilon_fl);

				if (tmp.e < best_e)
				{
					if (rilc_bfgs_enable == 1)
					{
						rilc_bfgs(&tmp,
								  &g,
								  &m,
								  pre,
								  grids,
								  mis,
								  torsion_size,
								  max_bfgs_steps);
					}
					else
					{
						bfgs(&tmp,
							 &g,
							 &m,
							 pre,
							 grids,
							 mis,
							 torsion_size,
							 max_bfgs_steps);
					}

					// set
					if (tmp.e < best_e)
					{
						set(&tmp, &m.ligand.rigid, &m.m_coords,
							m.atoms, m.m_num_movable_atoms, mis->epsilon_fl);

						best_out = tmp;
						// printf("best_out -->> e = %f \n",best_out.e);

						// printf("best_out -->> position = %f , %f, %f \n", best_out.position[0], best_out.position[1], best_out.position[2]);

						// printf("best_out -->> orientation = %f , %f, %f,%f \n", best_out.orientation[0], best_out.orientation[1], best_out.orientation[2],best_out.orientation[3]);
						get_heavy_atom_movable_coords(&best_out, &m, &best_coords); // get coords
						best_e = tmp.e;
					}
				}
			}
		}
		// write the best conformation back to CPU
		results[gll] = best_out;
		coords[gll] = best_coords;
	}
}


__global__ void kernel2_update(
	const output_type_cl *ric,
	m_cl *mg,
	pre_cl *pre,
	grids_cl *grids,
	random_maps *random_maps,
	ligand_atom_coords_cl *coords,
	output_type_cl *results,
	const mis_cl *mis,
	const int torsion_size,
	const int search_depth,
	const int max_bfgs_steps,
	const int rilc_bfgs_enable)
{
	{ if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0)
	    d_k2_wall = st_clock_ns(); }


	int threadsPerBlock = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y +  blockIdx.x;
	// int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;        

	int id = blockNumInGrid;
	// int id = globalThreadNum;

	flo best_e = 1.0e38f;
	ligand_atom_coords_cl best_coords;

	visited_cl visited;
	visited_init(&visited);

	m_cl* m_gpu = &mg[0];   // [Path B merged] single shared copy (read-only in kernel)

	__shared__ m_coords_cl m_coords;
	m_coords = m_gpu->m_coords;

	__shared__ m_minus_forces minus_forces;
	minus_forces = m_gpu->minus_forces;

	__shared__ output_type_cl tmp;
	tmp = ric[id];

	__shared__ change_cl g;

	__shared__ output_type_cl best_out;
	__shared__ output_type_cl candidate;

	__shared__ ligand_gpu m_gpu_ligand;
	m_gpu_ligand.rigid = m_gpu->ligand.rigid;
	m_gpu_ligand.begin =  m_gpu->ligand.begin;
	m_gpu_ligand.end = m_gpu->ligand.end;
	m_gpu_ligand.m_num_movable_atoms = m_gpu->m_num_movable_atoms;
	
	//__shared__ m_atom_gpu atom_gpu;
	//for(int i = threadNumInBlock; i<MAX_NUM_OF_ATOMS; i+=threadsPerBlock){
	//	atom_gpu.m_gpu_atom[i] = m_gpu->atoms[i];
	//}
	//__syncthreads();

	// [Q4-12] mis scalars now in __constant__ mis_cuda_const (-68B shared)

	// [Q4-7C] per-trajectory pairs skip-bitmap (shared, zeroed once per kernel)
	__shared__ unsigned long long q4_skip_row[(MAX_NUM_OF_LIG_PAIRS + 63) / 64];
	__shared__ int q4_epoch_i;
	if (threadNumInBlock == 0) {
		for (int w = 0; w < (MAX_NUM_OF_LIG_PAIRS + 63) / 64; w++) q4_skip_row[w] = 0ULL;
		q4_epoch_i = 0;
	}
	__syncthreads();

	__shared__ grids_gpu grids_cuda;

	for(int i = threadNumInBlock; i<GRIDS_SIZE; i+=threadsPerBlock){
		grids_cuda.grid_other[i].m_i = grids->grids[i].m_i;
		grids_cuda.grid_other[i].m_j= grids->grids[i].m_j;
		grids_cuda.grid_other[i].m_k = grids->grids[i].m_k;
		for(int n= 0; n<3;n++){
			grids_cuda.grid_other[i].m_init[n] = grids->grids[i].m_init[n];
			grids_cuda.grid_other[i].m_range[n]  = grids->grids[i].m_range[n] ;
			grids_cuda.grid_other[i].m_factor[n]  = grids->grids[i].m_factor[n] ;
			grids_cuda.grid_other[i].m_dim_fl_minus_1[n]  = grids->grids[i].m_dim_fl_minus_1[n] ;
			grids_cuda.grid_other[i].m_factor_inv[n]  = grids->grids[i].m_factor_inv[n] ;
		}
	}
	__syncthreads();
	

	for (int step = 0; step < search_depth; step++)
	{
		candidate = tmp;
	

		int map_index = (step + id * search_depth) % MAX_NUM_OF_RANDOM_MAP;
		// printf("\nstep = %d, id = %d, search_depth = %d, map_index = %d,", step, id, search_depth, map_index);
		STAGE_T0;
		mutate_conf_cl_update(map_index,
							  &candidate,
							  random_maps->int_map,
							  random_maps->sphere_map,
							  random_maps->pi_map,
							  m_gpu_ligand.begin,
							  m_gpu_ligand.end,
							  m_gpu->atoms,
							  &m_coords,
							  m_gpu->ligand.rigid.origin[0],
							  mis_cuda_const.epsilon_fl,
							  mis_cuda_const.mutation_amplitude,
							  torsion_size,
							  threadNumInBlock,
							  threadsPerBlock);STAGE_ADD(mutate);;

		if (rilc_bfgs_enable == 1)
		{
			if (threadNumInBlock == 0 && blockNumInGrid < 1) printf("V1 coord[0]=%.3f,%.3f,%.3f cand.e=%.1f\n", m_coords.coords[0][0], m_coords.coords[0][1], m_coords.coords[0][2], candidate.e);
		rilc_bfgs_update(&candidate, // shared memory
							 &g,		 // shared memory
							 m_gpu,	 // global memory
							 pre,
							 grids,
							 &mis_cuda_const,
							 torsion_size,
							 max_bfgs_steps,
							 &m_coords,		// shared memory
							 &minus_forces, // shared memory
							 threadNumInBlock,
							 threadsPerBlock,
							 &m_gpu_ligand.rigid,
							 &grids_cuda,
							&visited,
							 q4_skip_row,
							 (unsigned long long)q4_epoch_i);
		}
		else
		{
			//bfgs(&candidate, // shared memory
			//	 &g,		 // shared memory
			//	 m_gpu,	 // global memory
			//	 pre,
			//	 grids,
			//	 mis,
			//	 torsion_size,
			//	 max_bfgs_steps);
			printf("the bfgs function cannot work,please using rilc_bfgs_enable == 1 in command");
		}

		flo n = generate_n(random_maps->pi_map, map_index);

		if (step == 0 || metropolis_accept(tmp.e, candidate.e, 1.2, n))
		{

			tmp = candidate;

			set_update(&tmp, &m_gpu_ligand.rigid, &m_coords,
				m_gpu->atoms, m_gpu->m_num_movable_atoms, mis_cuda_const.epsilon_fl);

			if (tmp.e < best_e)
			{
				if (rilc_bfgs_enable == 1)
				{
					rilc_bfgs_update(&tmp,
									 &g,
									 m_gpu,
									 pre,
									 grids,
									 &mis_cuda_const,
									 torsion_size,
									 max_bfgs_steps,
									 &m_coords,		// shared memory
									 &minus_forces, // shared memory
									 threadNumInBlock,
									 threadsPerBlock,
									 &m_gpu_ligand.rigid,
									 &grids_cuda,
									 &visited,
									 q4_skip_row,
									 (unsigned long long)q4_epoch_i);
				}
				else
				{
					//bfgs(&tmp,
					//	 &g,
					//	 m_gpu,
					//	 pre,
					//	 grids,
					//	 mis,
					//	 torsion_size,
					//	 max_bfgs_steps);
					printf("the bfgs function cannot work,please using rilc_bfgs_enable == 1 in command");
				}

				// set
				if (tmp.e < best_e)
				{
					set_update(&tmp, &m_gpu_ligand.rigid, &m_coords,
						m_gpu->atoms, m_gpu->m_num_movable_atoms, mis_cuda_const.epsilon_fl);

					best_out = tmp;

					// printf("best_out -->> e = %f \n",best_out.e);

					// printf("best_out -->> position = %f , %f, %f \n", best_out.position[0], best_out.position[1], best_out.position[2]);

					// printf("best_out -->> orientation = %f , %f, %f,%f \n", best_out.orientation[0], best_out.orientation[1], best_out.orientation[2],best_out.orientation[3]);
					get_heavy_atom_movable_coords_update(&best_out, m_gpu, &best_coords,&m_coords); // get coords
					best_e = tmp.e;
				}
			}
		}
	}
	// write the best conformation back to CPU
	results[id] = best_out;
	coords[id] = best_coords;
	{ if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0) {
	    unsigned long long w = st_clock_ns() - d_k2_wall;
	    atomicAdd(&d_k2_wall, 0); // no-op keeps symbol referenced
	    printf("K2WALL block0 = %.3f ms\n", w/1e6);
	  } }
}

extern "C"
void kernel_monte(
	output_type_cl **ric_batch_in,
	m_cl **mg_batch_in,
	pre_cl *pre,
	grids_cl *grids,
	random_maps **random_maps_batch_in,
	ligand_atom_coords_cl **coords_batch_in,
	output_type_cl **results_batch_in,
	const mis_cl *mis,
	const int *torsion_sizes,
	const int *search_depths,
	const int *bfgs_steps,
	const int rilc_bfgs_enable,
	const int num_ligands,
	const int threads_per_ligand,
	const int *lig_valid)
{
	// [v6b-DESIGN-C] per-ligand launches, async on stream 0, ONE sync at the
	// end. The kernel itself is the UNCHANGED single-ligand kernel2_update
	// (scalars in param space — certified codegen, zero k2-regression risk);
	// the batch value is host-side: N-1 per-ligand deviceSync removed and the
	// per-process entry-gap amortized. Per-ligand event pairs give the k2 sum
	// plus per-ligand attribution (batch numbers need per-ligand marks).
	dim3 grid_size(GRID_DIM1, GRID_DIM2, 1);
	dim3 block_size(48, 4, 1); // [Q5.3-W6] 192 threads: 6 warps per trajectory (still 1 conf/block)

	StageStats* d_stage_ptr;
	cudaGetSymbolAddress((void**)&d_stage_ptr, d_stage);
	cudaMemset(d_stage_ptr, 0, sizeof(StageStats));

		// [Q4-12] ligand-independent scalars -> constant memory, ONCE per batch
		{
			void* mc = nullptr;
			checkCUDA(cudaGetSymbolAddress(&mc, mis_cuda_const));
			checkCUDA(cudaMemcpy(mc, mis, sizeof(mis_cl), cudaMemcpyHostToDevice));
		}

	int launched = 0;
	cudaEvent_t ev[2];
	cudaEventCreate(&ev[0]);
	cudaEventCreate(&ev[1]);
	cudaEventRecord(ev[0], 0);
	for (int lg = 0; lg < num_ligands; lg++) {
		if (lig_valid[lg] == 0) continue;
		kernel2_update<<<grid_size, block_size>>>(ric_batch_in[lg], mg_batch_in[lg], pre, grids,
			random_maps_batch_in[lg], coords_batch_in[lg], results_batch_in[lg], mis,
			torsion_sizes[lg], search_depths[lg], bfgs_steps[lg], rilc_bfgs_enable);
		launched++;
	}
	checkCUDA(cudaDeviceSynchronize()); // the ONLY sync: end of the batch

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) printf("kernel2 was wrong [%s]\n", cudaGetErrorString(error));

	cudaEventRecord(ev[1], 0);
	cudaEventSynchronize(ev[0]);
	cudaEventSynchronize(ev[1]);
	flo kernalExecutionTime = 0.0f;
	cudaEventElapsedTime(&kernalExecutionTime, ev[0], ev[1]);
	cudaEventDestroy(ev[0]);
	cudaEventDestroy(ev[1]);

	gt_add_kernel(2, kernalExecutionTime);
	printf("GPU_KERNEL2_TIME=%.3f\n", kernalExecutionTime / 1000.0f);
	printf("V6B_LAUNCHED=%d\n", launched);

	StageStats h_stage;
	cudaMemcpy(&h_stage, d_stage_ptr, sizeof(StageStats), cudaMemcpyDeviceToHost);
	printf("STAGE_MUTATE_NS=%llu\n", h_stage.mutate);
	printf("STAGE_EVAL_INIT_NS=%llu\n", h_stage.eval_init);
	printf("STAGE_LS_TOTAL_NS=%llu\n", h_stage.ls_total);
	printf("STAGE_TWOP_NS=%llu\n", h_stage.twop);
	printf("STAGE_GRID_NS=%llu\n", h_stage.grid);
	printf("STAGE_INTRA_NS=%llu\n", h_stage.intra);
	printf("STAGE_POT_NS=%llu\n", h_stage.pot);
	printf("STAGE_EVALS_N=%llu\n", h_stage.evals_n);
	printf("STAGE_EPISODES_N=%llu\n", h_stage.n_ep);
	printf("STAGE_ITER_N=%llu\n", h_stage.n_iter);
	printf("STAGE_LS_N=%llu\n", h_stage.n_ls);
}

// [STREAM] split-form launchers: ligand-level pipeline. NO batch-tail sync:
// the pump thread keeps launching while consumers drain finished ligands.
extern "C"
void kernel_monte_ranged(
	cudaStream_t s3_kernel_stream,
	output_type_cl **ric_batch_in,
	m_cl **mg_batch_in,
	pre_cl *pre,
	grids_cl *grids,
	random_maps **random_maps_batch_in,
	ligand_atom_coords_cl **coords_batch_in,
	output_type_cl **results_batch_in,
	const mis_cl *mis,
	const int *torsion_sizes,
	const int *search_depths,
	const int *bfgs_steps,
	const int rilc_bfgs_enable,
	const int num_ligands,
	const int threads_per_ligand,
	const int *lig_valid,
	cudaEvent_t ev_range) // records event after ALL launches; no sync inside
{
	dim3 grid_size(GRID_DIM1, GRID_DIM2, 1);
	dim3 block_size(48, 4, 1);

	StageStats* d_stage_ptr;
	cudaGetSymbolAddress((void**)&d_stage_ptr, d_stage);
	cudaMemset(d_stage_ptr, 0, sizeof(StageStats));

	{
		void* mc = nullptr;
		checkCUDA(cudaGetSymbolAddress(&mc, mis_cuda_const));
		checkCUDA(cudaMemcpy(mc, mis, sizeof(mis_cl), cudaMemcpyHostToDevice));
	}

	for (int lg = 0; lg < num_ligands; lg++) {
		if (lig_valid[lg] == 0) continue;
		kernel2_update<<<grid_size, block_size, 0, s3_kernel_stream>>>(ric_batch_in[lg], mg_batch_in[lg], pre, grids,
			random_maps_batch_in[lg], coords_batch_in[lg], results_batch_in[lg], mis,
			torsion_sizes[lg], search_depths[lg], bfgs_steps[lg], rilc_bfgs_enable);
	}
	checkCUDA(cudaEventRecord(ev_range, s3_kernel_stream)); // fire when range kernels done
}

void kernel_monte_begin_mis(const mis_cl *mis)
{
	void* mc = nullptr;
	checkCUDA(cudaGetSymbolAddress(&mc, mis_cuda_const));
	checkCUDA(cudaMemcpy(mc, mis, sizeof(mis_cl), cudaMemcpyHostToDevice));
	StageStats* d_stage_ptr;
	cudaGetSymbolAddress((void**)&d_stage_ptr, d_stage);
	cudaMemset(d_stage_ptr, 0, sizeof(StageStats));
}

extern "C"
void kernel_monte_stream_lig(
	output_type_cl *ric, m_cl *mg, pre_cl *pre, grids_cl *grids,
	random_maps *random_maps, ligand_atom_coords_cl *coords,
	output_type_cl *results, const mis_cl *mis,
	const int torsion_size, const int search_depth, const int max_bfgs_steps,
	const int rilc_bfgs_enable, cudaEvent_t ev_done)
{
	dim3 grid_size(GRID_DIM1, GRID_DIM2, 1);
	dim3 block_size(48, 4, 1);
	kernel2_update<<<grid_size, block_size>>>(ric, mg, pre, grids,
		random_maps, coords, results, mis, torsion_size, search_depth,
		max_bfgs_steps, rilc_bfgs_enable);
	cudaEventRecord(ev_done, 0);
}

