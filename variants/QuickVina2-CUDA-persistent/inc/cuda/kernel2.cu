#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel2.h"
#include "mutate_conf.cuh"
#include "quasi_newton.cuh"
#include "matrix.cuh"
#include "assert.h"
#include "visited.cuh"

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
	const flo acceptance_probability = exp((old_f - new_f) / temperature);
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

	flo best_e = INFINITY;
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


// [Persistent] Global counter for persistent kernel — atomically incremented by each block
__device__ int persistent_ligand_counter = 0;

__global__ void kernel2_update(
	const output_type_cl *ric,		   // batch: N * threads_per_ligand
	m_cl *mg,					   // batch: N m_cl structs
	pre_cl *pre,				   // shared
	grids_cl *grids,			   // shared
	random_maps *random_maps,   // batch: N random_maps
	ligand_atom_coords_cl *coords,    // batch: N * threads_per_ligand
	output_type_cl *results,	   // batch: N * threads_per_ligand
	const mis_cl *mis,			   // shared
	const int *torsion_sizes,	   // batch: N
	const int *search_depths,	   // batch: N
	const int *bfgs_steps,		   // batch: N
	const int rilc_bfgs_enable,
	const int num_ligands,		   // batch size
	const int threads_per_ligand)	   // MC restarts per ligand (8192)
{
	int threadsPerBlock = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y +  blockIdx.x;

	// Each block handles a stride of restarts for one ligand
	// Persistent: block loops over ligands via atomic counter
	// Block stride = number of blocks in one "wave" = total blocks / num_ligands
	// But simpler: each block claims restart slots via atomicAdd on persistent counter
	// persistent_counter maps to (ligand_id, restart_id) pair

	while (1)
	{
		// Only thread 0 claims the slot, broadcast to all threads via shared memory
		__shared__ int s_slot;
		if (threadNumInBlock == 0)
			s_slot = atomicAdd(&persistent_ligand_counter, 1);
		__syncthreads();
		int slot = s_slot;

		int ligand_id = slot / threads_per_ligand;
		int id = slot % threads_per_ligand;

		if (ligand_id >= num_ligands) return;

		int ric_offset = ligand_id * threads_per_ligand;

	flo best_e = INFINITY;
	ligand_atom_coords_cl best_coords;

	visited_cl visited;
	visited_init(&visited);

	m_cl* m_gpu = &mg[ligand_id]; // [Batch] per-ligand m_cl

	__shared__ m_coords_cl m_coords;
	m_coords = m_gpu->m_coords;

	__shared__ m_minus_forces minus_forces;
	minus_forces = m_gpu->minus_forces;

	__shared__ output_type_cl tmp;
	tmp = ric[ric_offset + id];

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

	__shared__ mis_cl mis_gpu;
	mis_gpu = *mis;

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
	

	int search_depth = search_depths[ligand_id];
	for (int step = 0; step < search_depth; step++)
	{
		candidate = tmp;
	

		int map_index = (step + id * search_depth) % MAX_NUM_OF_RANDOM_MAP;
		// printf("\nstep = %d, id = %d, search_depth = %d, map_index = %d,", step, id, search_depth, map_index);
		mutate_conf_cl_update(map_index,
							  &candidate,
								  random_maps[ligand_id].int_map,
								  random_maps[ligand_id].sphere_map,
								  random_maps[ligand_id].pi_map,
							  m_gpu_ligand.begin,
							  m_gpu_ligand.end,
							  m_gpu->atoms,
							  &m_coords,
							  m_gpu->ligand.rigid.origin[0],
							  mis_gpu.epsilon_fl,
							  mis_gpu.mutation_amplitude,
								  torsion_sizes[ligand_id],
							  threadNumInBlock,
							  threadsPerBlock);

		if (rilc_bfgs_enable == 1)
		{
			if (threadNumInBlock == 0 && blockNumInGrid < 1) printf("V1 coord[0]=%.3f,%.3f,%.3f cand.e=%.1f\n", m_coords.coords[0][0], m_coords.coords[0][1], m_coords.coords[0][2], candidate.e);
		rilc_bfgs_update(&candidate, // shared memory
							 &g,		 // shared memory
							 m_gpu,	 // global memory
							 pre,
							 grids,
							 &mis_gpu,
								 torsion_sizes[ligand_id],
								 bfgs_steps[ligand_id],
							 &m_coords,		// shared memory
							 &minus_forces, // shared memory
							 threadNumInBlock,
							 threadsPerBlock,
							 &m_gpu_ligand.rigid,
							 &grids_cuda,
							&visited);
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

	flo n = generate_n(random_maps[ligand_id].pi_map, map_index);

		if (step == 0 || metropolis_accept(tmp.e, candidate.e, 1.2, n))
		{

			tmp = candidate;

			set_update(&tmp, &m_gpu_ligand.rigid, &m_coords,
				m_gpu->atoms, m_gpu->m_num_movable_atoms, mis_gpu.epsilon_fl);

			if (tmp.e < best_e)
			{
				if (rilc_bfgs_enable == 1)
				{
					rilc_bfgs_update(&tmp,
									 &g,
									 m_gpu,
									 pre,
									 grids,
									 &mis_gpu,
										 torsion_sizes[ligand_id],
										 bfgs_steps[ligand_id],
									 &m_coords,		// shared memory
									 &minus_forces, // shared memory
									 threadNumInBlock,
									 threadsPerBlock,
									 &m_gpu_ligand.rigid,
									 &grids_cuda,
									 &visited);
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
						m_gpu->atoms, m_gpu->m_num_movable_atoms, mis_gpu.epsilon_fl);

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
	results[ric_offset + id] = best_out;
	coords[ric_offset + id] = best_coords;

	} // end persistent while loop
}

extern "C" 
void kernel_monte(
	const output_type_cl *ric,
	m_cl *mg,
	pre_cl *pre,
	grids_cl *grids,
	random_maps *random_maps,
	ligand_atom_coords_cl *coords,
	output_type_cl *results,
	const mis_cl *mis,
	const int *torsion_sizes,
	const int *search_depths,
	const int *bfgs_steps,
	const int rilc_bfgs_enable,
	const int num_ligands,
	const int threads_per_ligand)
{
	dim3 grid_size(64, 128, 1);  // 8192 blocks — persistent kernel loops internally
	dim3 block_size(16, 2, 1);
	cudaEvent_t time1, time2;
	cudaEventCreate(&time1);
	cudaEventCreate(&time2);

	cudaEventRecord(time1, 0);

	// Reset persistent counter via a tiny kernel
	{
		int *d_counter;
		checkCUDA(cudaGetSymbolAddress((void**)&d_counter, persistent_ligand_counter));
		checkCUDA(cudaMemset(d_counter, 0, sizeof(int)));
	}

	kernel2_update<<<grid_size, block_size>>>(ric, mg, pre, grids, random_maps, coords, results, mis,
						  torsion_sizes, search_depths, bfgs_steps,
						  rilc_bfgs_enable, num_ligands, threads_per_ligand);
	checkCUDA(cudaDeviceSynchronize());

	flo kernalExecutionTime;
	cudaError_t error;
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("kernel2 was wrong [%s]\n", cudaGetErrorString(error));
	}
	cudaEventRecord(time2, 0);

	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);

	cudaEventElapsedTime(&kernalExecutionTime, time1, time2);

	printf("GPU_KERNEL2_TIME=%.3f\n", kernalExecutionTime / 1000.0f);

	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
}
