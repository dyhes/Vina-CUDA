#include "cache.h"
#include "host_probe.h"
// #include "wrapcl.h"
#include "monte_carlo.h"
#include "coords.h"
#include "mutate.h"
#include "quasi_newton.h"
#include "parallel_mc.h"
#include "szv_grid.h"
#include <thread>
#include <execinfo.h>
static struct timespec _g_tv;
#include <signal.h>
#include <unistd.h>
static void _stream_segv(int sig) {
	void *bt[24];
	int n = backtrace(bt, 24);
	fprintf(stderr, "\n=== STREAM_CRASH sig=%d ===\n", sig);
	backtrace_symbols_fd(bt, n, 2);
	_exit(99);
}
#include <atomic>
#include <boost/progress.hpp>

#include "commonMacros.h"
// #include "wrapcl.h"
#include "kernel1.cuh"
#include "gpu_timer.h"
#include "gt_block.h"
#include "kernel2.cuh"
#include "random.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <stdio.h>
#include "omp.h"

/**
 * #include <boost/timer/timer.hpp>
 * using boost::timer::cpu_timer
 */
#include <boost/timer.hpp>

boost::timer boost_t;
#ifdef HOST_PROBE
// [v7] GPU-side rand_maps + ric generation (ric_gpu.cu)
extern "C" void v7_gen_randmaps_gpu(random_maps* d_rm, int n_tors,
                                    unsigned long long seed, cudaStream_t s);
extern "C" void kernel_monte_begin_mis(const mis_cl *mis);
extern "C" void kernel_monte_stream_lig(
	output_type_cl *ric, m_cl *mg, pre_cl *pre, grids_cl *grids,
	random_maps *random_maps, ligand_atom_coords_cl *coords,
	output_type_cl *results, const mis_cl *mis,
	const int torsion_size, const int search_depth, const int max_bfgs_steps,
	const int rilc_bfgs_enable, cudaEvent_t ev_done, cudaStream_t s);
extern "C" void v7_gen_ric_gpu(output_type_cl* d_ric, int thread,
                               float cx, float cy, float cz,
                               float sx, float sy, float sz,
                               int n_tors, unsigned long long seed, cudaStream_t s);
#endif

// extern __constant__ mis_cl mis_cuda;

using namespace std;
volatile enum { FINISH,
				DOCKING,
				ABORT } status;
void print_process()
{
	int count = 0;
	printf("\n");
	do
	{
#ifdef WIN32
		Sleep(100);
#else
		usleep(100000);   // [v7] 100ms quantum: progress UI should not add ~0.5s avg to E2E
#endif
		printf("\rPerform docking|");
		for (int i = 0; i < count; i++)
			printf(" ");
		printf("=======");
		for (int i = 0; i < 30 - count; i++)
			printf(" ");
		printf("|");
		fflush(stdout);

		count++;
		count %= 30;
	} while (status == DOCKING);
	if (status == FINISH)
	{
		printf("\rPerform docking|");
		for (int i = 0; i < 16; i++)
			printf("=");
		printf("done");
		for (int i = 0; i < 17; i++)
			printf("=");
		printf("|\n");
		fflush(stdout);
	}
	else if (status == ABORT)
	{
		printf("\rPerform docking|");
		for (int i = 0; i < 16; i++)
			printf("=");
		printf("error");
		for (int i = 0; i < 16; i++)
			printf("=");
		printf("|\n");
		fflush(stdout);
	}
}

std::vector<output_type> cl_to_vina(output_type_cl result_ptr[],
									ligand_atom_coords_cl result_coords_ptr[],
									int thread, int lig_torsion_size)
{
	std::vector<output_type> results_vina;
	int num_atoms;

#pragma omp parallel for
	for (int i = 0; i < thread; i++)
	{
		output_type_cl tmp = result_ptr[i];
		ligand_atom_coords_cl tmp_coords = result_coords_ptr[i];
		conf tmp_c;
		tmp_c.ligands.resize(1);
		// Position
		for (int j = 0; j < 3; j++)
			tmp_c.ligands[0].rigid.position[j] = tmp.position[j];
		// Orientation
		qt q(tmp.orientation[0], tmp.orientation[1], tmp.orientation[2], tmp.orientation[3]);
		tmp_c.ligands[0].rigid.orientation = q;
		output_type tmp_vina(tmp_c, tmp.e);
		// torsion
		for (int j = 0; j < lig_torsion_size; j++)
			tmp_vina.c.ligands[0].torsions.push_back(tmp.lig_torsion[j]);
		// coords
		for (int j = 0; j < MAX_NUM_OF_ATOMS; j++)
		{
			vec v_tmp(tmp_coords.coords[j][0], tmp_coords.coords[j][1], tmp_coords.coords[j][2]);
			if ((v_tmp[0] != 0 || v_tmp[1] != 0) || (v_tmp[2] != 0))
				tmp_vina.coords.push_back(v_tmp);
		}
		results_vina.push_back(tmp_vina);
		if (i == 0)
			num_atoms = tmp_vina.coords.size();
		if (num_atoms != tmp_vina.coords.size())
		{
			fprintf(stderr, "CL2V_MISMATCH i=%d\n", i);
		}
	}
	return results_vina;
}

void main_procedure_cl(cache &c, const std::vector<model> &ms, const precalculate &p, const parallel_mc par,
					   const vec &corner1, const vec &corner2, const int seed, std::vector<output_container> &outs,
					   const std::vector<std::vector<std::string>> ligand_names, const int rilc_bfgs)
{

	/**************************************************************************/
	/***************************    GPU Init    ****************************/
	/**************************************************************************/

	// static std::ofstream g_log("out.log");
	// std::cout.rdbuf(g_log.rdbuf());

	int device = 0; // Assuming device 0
	cudaDeviceProp props;
	checkCUDA(cudaGetDeviceProperties(&props, device));
	int max_wg_size = props.maxThreadsPerBlock;

	std::cout << "Device name: " << props.name << " || " << "Compute capability: " << props.major << "." << props.minor << std::endl;
	// std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
	// std::cout << "The maxThreadsPerBlock is : " << max_wg_size << std::endl;

	// std::cout<<"The maxThreadsPerBlock is : "<<max_wg_size<<std::endl;

	/**************************************************************************/
	/************************    Original Vina code    ************************/
	/**************************************************************************/

	printf("\nUsing random seed: %d", seed);

	sz nat = num_atom_types(c.atu);

	szv needed;
	for (int i = 0; i < nat; i++)
	{
		if (!c.grids[i].initialized())
		{
			needed.push_back(i);
			c.grids[i].init(c.gd);
		}
	}

	flv affinities(needed.size());

	grid &g = c.grids[needed.front()];

	const fl cutoff_sqr = p.cutoff_sqr();

	grid_dims gd_reduced = szv_grid_dims(c.gd);
	szv_grid ig(ms[0], gd_reduced, cutoff_sqr); // use ms[0]
	// szv_grid ig2(ms[1], gd_reduced, cutoff_sqr); // use ms[0]

	for (int i = 0; i < 3; i++)
	{
		if (ig.m_init[i] != g.m_init[i])
		{
			printf("m_init not equal!");
			exit(-1);
		}
		if (ig.m_range[i] != g.m_range[i])
		{
			printf("m_range not equal!");
			exit(-1);
		}
	}

	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl

	std::vector<conf_size> s;
	std::vector<output_type> tmps;
	for (int i = 0; i < ms.size(); i++)
	{
		s.push_back(ms[i].get_size());
		output_type tmp(s[i], 0);
		tmps.push_back(tmp);
	}
	// change g(s);

	/**********************************************************************************/
	/************************ Kernel1  & Allocate GPU memory **************************/
	/**********************************************************************************/
	boost_t.restart();

	hp_mark("pre_gpu_setup");
hp_mark("C_setdev_start");
checkCUDA(cudaSetDevice(0));
hp_mark("C_setdev_done");
	hp_mark("cudaSetDevice");
	cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8;
	// [GT] init gpu_timer events
	cudaEventCreate(&g_timer.start_ev);
	cudaEventCreate(&g_timer.stop_ev);

hp_mark("C_stream1_start");
checkCUDA(cudaStreamCreate(&stream1));
	hp_mark("stream1");
	checkCUDA(cudaStreamCreate(&stream2));
	checkCUDA(cudaStreamCreate(&stream3));
	checkCUDA(cudaStreamCreate(&stream4));
	checkCUDA(cudaStreamCreate(&stream5));
	checkCUDA(cudaStreamCreate(&stream6));
	checkCUDA(cudaStreamCreate(&stream7));
	checkCUDA(cudaStreamCreate(&stream8));
	hp_mark("C_streams_done");
hp_mark("streams_all");

	hp_mark("rng_ctor");
	rng generator(static_cast<rng::result_type>(seed));
	hp_mark("rng_done");
	const model& m = ms[0];   // [v7] reference: copy ctor was 220ms (atom bonds heap allocs); all uses are reads
	hp_mark("model_copy");
// Preparing grid related data
	if (GRIDS_SIZE != c.grids.size())
	{
		throw std::runtime_error("grid_size has to be 17!");
	} // grid_size has to be 17
	hp_mark("PREP_grids_malloc_start");
	grids_cl *grids_ptr = (grids_cl *)malloc(sizeof(grids_cl));
	hp_mark("PREP_grids_malloc_done");
	grid *tmp_grid_ptr = &c.grids[0];

	grids_ptr->atu = c.atu;		// atu
	grids_ptr->slope = c.slope; // slope
	int grids_front;
#pragma omp parallel for
	for (int i = GRIDS_SIZE - 1; i >= 0; i--)
	{
		for (int j = 0; j < 3; j++)
		{
			grids_ptr->grids[i].m_init[j] = tmp_grid_ptr[i].m_init[j];
			grids_ptr->grids[i].m_factor[j] = tmp_grid_ptr[i].m_factor[j];
			grids_ptr->grids[i].m_dim_fl_minus_1[j] = tmp_grid_ptr[i].m_dim_fl_minus_1[j];
			grids_ptr->grids[i].m_factor_inv[j] = tmp_grid_ptr[i].m_factor_inv[j];
		}
		if (tmp_grid_ptr[i].m_data.dim0() != 0)
		{
			grids_ptr->grids[i].m_i = tmp_grid_ptr[i].m_data.dim0();
			assert(MAX_NUM_OF_GRID_MI >= grids_ptr->grids[i].m_i);
			grids_ptr->grids[i].m_j = tmp_grid_ptr[i].m_data.dim1();
			assert(MAX_NUM_OF_GRID_MJ >= grids_ptr->grids[i].m_j);
			grids_ptr->grids[i].m_k = tmp_grid_ptr[i].m_data.dim2();
			assert(MAX_NUM_OF_GRID_MK >= grids_ptr->grids[i].m_k);
			grids_front = i;
		}
		else
		{
			grids_ptr->grids[i].m_i = 0;
			grids_ptr->grids[i].m_j = 0;
			grids_ptr->grids[i].m_k = 0;
		}
	}
	size_t grids_size = sizeof(grids_cl);

	grids_cl *grids_gpu;
	hp_mark("PREP_cudamalloc_start");
	checkCUDA(cudaMalloc((void **)&grids_gpu, grids_size));
	hp_mark("PREP_cudamalloc_done");
	GT_BLOCK_START();
	hp_mark("PREP_grids_h2d_start");
	// [Q3-1] grids header-only HtoD (v5 idea, minus the redundant memset):
	// kernel1 generates ALL m_data on device; host ships only atu/slope (8B)
	// + per-grid metadata (first 72B of grid_cl). No m_data, no zero-fill.
	checkCUDA(cudaMemcpyAsync(&grids_gpu->atu, &grids_ptr->atu, sizeof(int) + sizeof(flo), cudaMemcpyHostToDevice, stream1));
	for (int gi = 0; gi < GRIDS_SIZE; gi++)
		checkCUDA(cudaMemcpyAsync(&grids_gpu->grids[gi], &grids_ptr->grids[gi], 72, cudaMemcpyHostToDevice, stream1));
	hp_mark("PREP_grids_h2d_done");
	hp_mark("grids_fill_h2d");

	// Preparing atom relationship
	if (MAX_NUM_OF_ATOM_RELATION_COUNT < ig.m_data.m_data.size())
	{
		throw std::runtime_error("relation too large! please use -DLARGE_BOX in makefile");
	}
	assert(ig.m_data.m_i <= 10);
	assert(ig.m_data.m_j <= 10);
	assert(ig.m_data.m_k <= 10);
	ar_cl *ar_ptr = (ar_cl *)malloc(sizeof(ar_cl));
	hp_mark("PREP_ar_start");
	for (int i = 0; i < ig.m_data.m_data.size(); i++)
	{
		ar_ptr->relation_size[i] = ig.m_data.m_data[i].size();
		if (MAX_NUM_OF_ATOM_RELATION_COUNT < ar_ptr->relation_size[i])
		{
			throw std::runtime_error("relation too large!");
		}
		for (int j = 0; j < ar_ptr->relation_size[i]; j++)
		{
			ar_ptr->relation[i][j] = ig.m_data.m_data[i][j];
		}
	}
	hp_mark("PREP_ar_done");
	size_t ar_size = sizeof(ar_cl);

	ar_cl *ar_gpu;
	checkCUDA(cudaMalloc((void **)&ar_gpu, ar_size));
	hp_mark("PREP_cudamalloc_grids_ar");
	checkCUDA(cudaMemcpyAsync(ar_gpu, ar_ptr, ar_size, cudaMemcpyHostToDevice, stream2));
	hp_mark("ar_fill_h2d");

	// Preparing protein atoms related data
	pa_cl *pa_ptr = (pa_cl *)malloc(sizeof(pa_cl));
	if (MAX_NUM_OF_PROTEIN_ATOMS <= m.grid_atoms.size())
	{
		throw std::runtime_error("pocket too large!");
	}
	for (int i = 0; i < m.grid_atoms.size(); i++)
	{
		pa_ptr->atoms[i].types[0] = m.grid_atoms[i].el;
		pa_ptr->atoms[i].types[1] = m.grid_atoms[i].ad;
		pa_ptr->atoms[i].types[2] = m.grid_atoms[i].xs;
		pa_ptr->atoms[i].types[3] = m.grid_atoms[i].sy;
		for (int j = 0; j < 3; j++)
			pa_ptr->atoms[i].coords[j] = m.grid_atoms[i].coords.data[j];
	}
	hp_mark("PREP_pa_done");
	size_t pa_size = sizeof(pa_cl);

	pa_cl *pa_gpu;
	checkCUDA(cudaMalloc((void **)&pa_gpu, pa_size));
	checkCUDA(cudaMemcpyAsync(pa_gpu, pa_ptr, pa_size, cudaMemcpyHostToDevice, stream3));
	hp_mark("pa_fill_h2d");

	// Preaparing precalculated look up table related data
	pre_cl *pre_ptr = (pre_cl *)malloc(sizeof(pre_cl));
	pre_ptr->m_cutoff_sqr = p.cutoff_sqr();
	pre_ptr->factor = p.factor;
	pre_ptr->n = p.n;
	if (MAX_P_DATA_M_DATA_SIZE <= p.data.m_data.size())
	{
		throw std::runtime_error("LUT too large!");
	}

	for (int i = 0; i < p.data.m_data.size(); i++)
	{
		pre_ptr->m_data[i].factor = p.data.m_data[i].factor;
		if (FAST_SIZE != p.data.m_data[i].fast.size())
		{
			throw std::runtime_error("fast too large!");
		}
		if (SMOOTH_SIZE != p.data.m_data[i].smooth.size())
		{
			throw std::runtime_error("smooth too large!");
		}

		for (int j = 0; j < FAST_SIZE; j++)
		{
			pre_ptr->m_data[i].fast[j] = p.data.m_data[i].fast[j];
		}
		for (int j = 0; j < SMOOTH_SIZE; j++)
		{
			pre_ptr->m_data[i].smooth[j][0] = p.data.m_data[i].smooth[j].first;
			pre_ptr->m_data[i].smooth[j][1] = p.data.m_data[i].smooth[j].second;
		}
	}
	hp_mark("PREP_pre_done");
	size_t pre_size = sizeof(pre_cl);

	pre_cl *pre_gpu;
	checkCUDA(cudaMalloc((void **)&pre_gpu, pre_size));
	checkCUDA(cudaMemcpyAsync(pre_gpu, pre_ptr, pre_size, cudaMemcpyHostToDevice, stream4));
	hp_mark("pre_fill_h2d");

	// Preparing grid boundries
	gb_cl *gb_ptr = (gb_cl *)malloc(sizeof(gb_cl));
	gb_ptr->dims[0] = ig.m_data.dim0();
	gb_ptr->dims[1] = ig.m_data.dim1();
	gb_ptr->dims[2] = ig.m_data.dim2();
	for (int i = 0; i < 3; i++)
		gb_ptr->init[i] = ig.m_init.data[i];
	for (int i = 0; i < 3; i++)
		gb_ptr->range[i] = ig.m_range.data[i];
	size_t gb_size = sizeof(gb_cl);

	gb_cl *gb_gpu;
	checkCUDA(cudaMalloc((void **)&gb_gpu, gb_size));
	checkCUDA(cudaMemcpyAsync(gb_gpu, gb_ptr, gb_size, cudaMemcpyHostToDevice, stream5));

	// miscellaneous
	mis_cl *mis_ptr = (mis_cl *)malloc(sizeof(mis_cl));
	mis_ptr->needed_size = needed.size();
	mis_ptr->epsilon_fl = epsilon_fl;
	mis_ptr->cutoff_sqr = cutoff_sqr;
	mis_ptr->max_fl = max_fl;
	// mis_ptr->torsion_size = tmp.c.ligands[0].torsions.size();
	// mis_ptr->max_bfgs_steps = quasi_newton_par_max_steps;
	// mis_ptr->search_depth = par.mc.search_depth;
	mis_ptr->mutation_amplitude = par.mc.mutation_amplitude;
	// mis_ptr->total_wi = max_wi_size[0] * max_wi_size[1];
	mis_ptr->thread = par.mc.thread;
	mis_ptr->ar_mi = ig.m_data.m_i;
	mis_ptr->ar_mj = ig.m_data.m_j;
	mis_ptr->ar_mk = ig.m_data.m_k;
	mis_ptr->grids_front = grids_front;
	for (int i = 0; i < 3; i++)
		mis_ptr->authentic_v[i] = authentic_v[i];
	for (int i = 0; i < 3; i++)
		mis_ptr->hunt_cap[i] = par.mc.hunt_cap[i];
	size_t mis_size = sizeof(mis_cl);

	mis_cl *mis_gpu;
	checkCUDA(cudaMalloc((void **)&mis_gpu, mis_size));
	checkCUDA(cudaMemcpyAsync(mis_gpu, mis_ptr, mis_size, cudaMemcpyHostToDevice, stream6));

	float *needed_ptr = (float *)malloc(mis_ptr->needed_size * sizeof(float));
	for (int i = 0; i < mis_ptr->needed_size; i++)
		needed_ptr[i] = needed[i];

	float *needed_gpu;
	checkCUDA(cudaMalloc((void **)&needed_gpu, mis_ptr->needed_size * sizeof(float)));
	checkCUDA(cudaMemcpyAsync(needed_gpu, needed_ptr, mis_ptr->needed_size * sizeof(float), cudaMemcpyHostToDevice, stream7));
	checkCUDA(cudaMemcpyAsync(needed_gpu, needed_ptr, mis_ptr->needed_size * sizeof(float), cudaMemcpyHostToDevice, stream7));
GT_BLOCK_END(0);

	/**************************************************************************/
	/************************    Allocate GPU memory    ***********************/
	/**************************************************************************/

	checkCUDA(cudaStreamSynchronize(stream5));

	checkCUDA(cudaStreamDestroy(stream1));
	checkCUDA(cudaStreamDestroy(stream2));
	checkCUDA(cudaStreamDestroy(stream3));
	checkCUDA(cudaStreamDestroy(stream4));
	checkCUDA(cudaStreamDestroy(stream5));
	checkCUDA(cudaStreamDestroy(stream6));
	checkCUDA(cudaStreamDestroy(stream7));

	// std::cout << "time of memcpy for kernel_grid: " << boost_t.elapsed() << std::endl;
	status = DOCKING;
#if defined(NDEBUG) && defined(SHOW_PROGRESS_BAR) // [v7] progress UI is opt-in: interprocess stats must not depend on a UI thread
	std::thread console_thread(print_process);
#endif

	/**************************************************************************/
	/************************  Call to kernel function  ***********************/
	/**************************************************************************/

	// std::cout << "<<Call to kernel_grid function>> " << std::endl;
	kernel_grid(pre_gpu, pa_gpu, gb_gpu, ar_gpu, grids_gpu, mis_gpu, needed_gpu, c.atu, nat);

	free(pa_ptr);
	free(gb_ptr);
	free(ar_ptr);
	free(needed_ptr);
	free(pre_ptr);
	free(grids_ptr);

	checkCUDA(cudaFree(pa_gpu));
	checkCUDA(cudaFree(gb_gpu));
	checkCUDA(cudaFree(ar_gpu));
	checkCUDA(cudaFree(needed_gpu));

	/**************************************************************************/
	/************************    Kernel2    ***********************/
	/**************************************************************************/
	int num_ligands = ms.size();
	hp_mark("grids_lut_h2d");
std::vector<random_maps *> rand_maps_ptrs(num_ligands);
	std::vector<output_type_cl *> ric_ptrs(num_ligands);
	std::vector<m_cl *> m_ptrs(num_ligands);
	std::vector<ligand_atom_coords_cl *> result_coords_ptrs(num_ligands);
	std::vector<output_type_cl *> result_ptrs(num_ligands);
	std::vector<int> torsion_sizes(num_ligands);
	// [v6b] batch staging: pointer-of-buffers (v6 architecture, zero struct churn)
	// + per-ligand scalars; arrays filled in the per-ligand loop, kernel called once after it
	std::vector<output_type_cl *> ric_batch_arr(num_ligands, nullptr);
	std::vector<m_cl *> mg_batch_arr(num_ligands, nullptr);
	std::vector<random_maps *> rm_batch_arr(num_ligands, nullptr);
	std::vector<ligand_atom_coords_cl *> coords_batch_arr(num_ligands, nullptr);
	std::vector<output_type_cl *> results_batch_arr(num_ligands, nullptr);
	std::vector<int> search_depths_arr(num_ligands, 0);
	std::vector<int> bfgs_steps_arr(num_ligands, 0);
	std::vector<int> lig_valid_arr(num_ligands, 0);

	// std::vector<cl_mem> ric_gpus			(num_ligands);
	// std::vector<cl_mem> m_gpus				(num_ligands);
	// std::vector<cl_mem> random_maps_gpus	(num_ligands);
	// std::vector<cl_mem> result_coords_gpus	(num_ligands);
	// std::vector<cl_mem> result_gpus			(num_ligands);

	std::vector<output_type_cl *> ric_gpus(num_ligands);
	std::vector<m_cl *> m_gpus(num_ligands);
	std::vector<random_maps *> random_maps_gpus(num_ligands);
	std::vector<ligand_atom_coords_cl *> result_coords_gpus(num_ligands);
	std::vector<output_type_cl *> result_gpus(num_ligands);

#ifdef STREAMING_PIPE
	// [S1] ligand-granularity streaming: events + consumers set up before the
	// prep loop; kernel2[j] launches happen INLINE at the end of ligand j's
	// prep, so k2[j] runs on the GPU while ligands j+1.. are being prepped.
	std::vector<cudaEvent_t> ev_done(num_ligands);
	std::vector<cudaEvent_t> ev_gen(num_ligands);
	std::vector<cudaEvent_t> ev_dtoh(num_ligands);
	for (int j = 0; j < num_ligands; j++) {
		checkCUDA(cudaEventCreate(&ev_done[j]));
		checkCUDA(cudaEventCreate(&ev_gen[j]));
		checkCUDA(cudaEventCreateWithFlags(&ev_dtoh[j], cudaEventDisableTiming));
	}
	int s1_win = 32;
	if (const char *wenv = getenv("V6B_STREAM_WINDOW")) s1_win = atoi(wenv);
	if (s1_win < 1) s1_win = 1;
	int s1_workers = 4;
	if (const char *wenv = getenv("V6B_STREAM_WORKERS")) s1_workers = atoi(wenv);
	if (s1_workers < 1) s1_workers = 1;
	std::atomic<int> s1_launched{0};
	cudaStream_t s1_comp;
	checkCUDA(cudaStreamCreate(&s1_comp)); // [S1-FIX2] non-default compute stream
	// [S1-FIX5] pre-extend the MallocAsync pool: without this, every per-ligand
	// allocation that grows the pool triggers a device-synchronizing expansion
	// (measured 22ms p50 == waiting for the running kernel2). Pre-touch 2GB.
	{
		void* pre = nullptr;
		if (cudaMallocAsync(&pre, (size_t)2 << 30, s1_comp) == cudaSuccess) {
			cudaFreeAsync(pre, s1_comp);
		}
		cudaStreamSynchronize(s1_comp);
	}
	int s1_inflight_base = 0;
	printf("V6B_STREAM_S1 window=%d workers=%d\n", s1_win, s1_workers);
	clock_gettime(CLOCK_MONOTONIC, &_g_tv);
	// consumers: striped (j = w, w+N), ev-sync + DtoH only; cl2v on main thread
	auto s1_consumer = [&](int w) {
		for (int j = w; j < num_ligands; j += s1_workers) {
			if (lig_valid_arr[j] == 0) continue;
			// [S1-FIX] wait until this ligand has actually been launched:
			// a never-recorded event synchronizes IMMEDIATELY (CUDA semantics),
			// so consumers started before the pump would DtoH garbage.
			while (s1_launched.load() <= j) {
				std::this_thread::yield();
			}
			// [S1-FIX7] DtoH async on s1_comp: the legacy default stream would
			// serialize against the whole s1_comp queue. Events keep ordering.
			while (cudaEventQuery(ev_done[j]) == cudaErrorNotReady)
				std::this_thread::yield();
			checkCUDA(cudaMemcpyAsync(result_ptrs[j], result_gpus[j], par.mc.thread * sizeof(output_type_cl), cudaMemcpyDeviceToHost, s1_comp));
			checkCUDA(cudaMemcpyAsync(result_coords_ptrs[j], result_coords_gpus[j], par.mc.thread * sizeof(ligand_atom_coords_cl), cudaMemcpyDeviceToHost, s1_comp));
			checkCUDA(cudaEventRecord(ev_dtoh[j], s1_comp));
			// [S1-FIX2] sanity: first pose energy must be finite; otherwise the
			// DtoH raced something (observed ~1/285 processes). Re-copy once —
			// the device buffer still holds the kernel result.
			if (!(result_ptrs[j][0].e == result_ptrs[j][0].e) ||
			    result_ptrs[j][0].e > 1e30f || result_ptrs[j][0].e < -1e30f) {
				checkCUDA(cudaMemcpy(result_ptrs[j], result_gpus[j], par.mc.thread * sizeof(output_type_cl), cudaMemcpyDeviceToHost));
				checkCUDA(cudaMemcpy(result_coords_ptrs[j], result_coords_gpus[j], par.mc.thread * sizeof(ligand_atom_coords_cl), cudaMemcpyDeviceToHost));
				fprintf(stderr, "S1_REDTOH j=%d\n", j);
			}
		}
	};
	std::vector<std::thread> s1_consumers;
	for (int w = 0; w < s1_workers; w++)
		s1_consumers.emplace_back(s1_consumer, w);
	// [S1] setup-phase H2D (grids/pre/mis/needed on streams 1/4/6/7) must be
	// complete before the first inline kernel2 launch — the old batch path was
		// saved by its tail deviceSync; streaming lost that when the launch moved
		// into the prep loop.
	checkCUDA(cudaDeviceSynchronize());
	kernel_monte_begin_mis(mis_ptr); // [S1] bind mis constants + reset stage stats once
#endif


	// std::vector<cl_event> ligands_events	(num_ligands);

	for (int ligand_count = 0; ligand_count < num_ligands; ligand_count++)
	{
		hp_mark("rand_maps_gen_host");

		// printf("\nDocking ligand: name: [%s] , the Atom size is: [%lu] , the torsion_sizes is : [%lu] \n", ligand_names[ligand_count][0].c_str(), ms[ligand_count].atoms.size(), tmps[ligand_count].c.ligands[0].torsions.size());
		try
		{
			const model& m = ms[ligand_count];   // [v7] reference (was 220ms copy)
			if (m.atoms.size() >= MAX_NUM_OF_ATOMS)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: Ligand NUM_OF_ATOMS too large!");
			}

			// assert(m.atoms.size() < MAX_NUM_OF_ATOMS);

			output_type tmp = tmps[ligand_count];
			torsion_sizes[ligand_count] = tmp.c.ligands[0].torsions.size();
			const int n_tors_gpu = int(tmp.c.ligands[0].torsions.size());
			if (tmp.c.ligands[0].torsions.size() >= MAX_NUM_OF_LIG_TORSION)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: Ligand NUM_OF_LIG_TORSION too large!");
			}

			// assert(tmp.c.ligands[0].torsions.size() < MAX_NUM_OF_LIG_TORSION);

			// Preparing m related data
			checkCUDA(cudaMallocHost((void **)&m_ptrs[ligand_count], sizeof(m_cl))); // [S1-FIX6] pinned: async H2D must not block on pageable copy
			m_cl *m_ptr = m_ptrs[ligand_count];

			for (int i = 0; i < m.atoms.size(); i++)
			{
				m_ptr->atoms[i].types[0] = m.atoms[i].el; // To store 4 atoms types (el, ad, xs, sy)
				m_ptr->atoms[i].types[1] = m.atoms[i].ad;
				m_ptr->atoms[i].types[2] = m.atoms[i].xs;
				m_ptr->atoms[i].types[3] = m.atoms[i].sy;
				for (int j = 0; j < 3; j++)
				{
					m_ptr->atoms[i].coords[j] = m.atoms[i].coords[j]; // To store atom coords
				}
			}

			// To store atoms coords
			for (int i = 0; i < m.coords.size(); i++)
			{
				for (int j = 0; j < 3; j++)
				{
					m_ptr->m_coords.coords[i][j] = m.coords[i].data[j];
				}
			}

			// To store minus forces
			for (int i = 0; i < m.coords.size(); i++)
			{
				for (int j = 0; j < 3; j++)
				{
					m_ptr->minus_forces.coords[i][j] = m.minus_forces[i].data[j];
				}
			}

			// Generate random maps
			rand_maps_ptrs[ligand_count] = (random_maps *)malloc(sizeof(random_maps));
			// [v7-GPU] rand_maps generated on device (see v7_gen_randmaps_gpu below,
			// after the device buffer is allocated). Host copy unused.
#ifdef HOST_PROBE
			hp_mark("rm_gpu_plan");
#endif
			size_t rand_maps_size = sizeof(random_maps);

			// Preparing random initial conformations
		ric_ptrs[ligand_count] = (output_type_cl *)malloc(par.mc.thread * sizeof(output_type_cl));
		output_type_cl *ric_ptr = ric_ptrs[ligand_count];
		// [v6b] second malloc below was leak+dead (pointer overwritten before use); removed

			// [v7-GPU] ric (random initial conformations) generated on device by
			// v7_gen_ric_gpu right after the ric buffer is allocated below.
			// Host tmp.c.randomize loop removed (~210ms/ligand).
#ifdef HOST_PROBE
			hp_mark("ric_gpu_plan");
#endif
		size_t ric_size = par.mc.thread * sizeof(output_type_cl);

			// Preparing ligand data
			if (m.num_other_pairs() != 0)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: m.other_paris is not supported!");
			} // m.other_paris is not supported!
			if (m.ligands.size() != 1)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: Only one ligand supported!");
			} // Only one ligand supported!
			m_ptr->ligand.pairs.num_pairs = m.ligands[0].pairs.size();

			// std::cout << "m_ptr->ligand.pairs.num_pairs = " << m_ptr->ligand.pairs.num_pairs << std::endl;
			if (m.ligands[0].pairs.size() >= MAX_NUM_OF_LIG_PAIRS)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: Ligand NUM_OF_LIG_PAIRS Too Large!");
			}
			// std::cout << "m_ptr->ligand.pairs.num_pairs = " << m_ptr->ligand.pairs.num_pairs << std::endl;
#pragma omp parallel for
			for (int i = 0; i < m_ptr->ligand.pairs.num_pairs; i++)
			{
				m_ptr->ligand.pairs.type_pair_index[i] = m.ligands[0].pairs[i].type_pair_index;
				m_ptr->ligand.pairs.a[i] = m.ligands[0].pairs[i].a;
				m_ptr->ligand.pairs.b[i] = m.ligands[0].pairs[i].b;
			}

			m_ptr->ligand.begin = m.ligands[0].begin; // 0
			m_ptr->ligand.end = m.ligands[0].end;	  // 29
			ligand m_ligand = m.ligands[0];			  // [v7] kept as copy: store_node needs non-const tree&; ligand copy is small
			if (m_ligand.end >= MAX_NUM_OF_ATOMS)
			{
				throw std::runtime_error("/lib/main_procedure_cu.cpp: NUM_OF_ATOMS Ligand Too large!");
			}

			// Store root node
			m_ptr->ligand.rigid.atom_range[0][0] = m_ligand.node.begin;
			m_ptr->ligand.rigid.atom_range[0][1] = m_ligand.node.end;
			for (int i = 0; i < 3; i++)
				m_ptr->ligand.rigid.origin[0][i] = m_ligand.node.get_origin()[i];

#pragma omp parallel for
			for (int i = 0; i < 9; i++)
			{
				}

			m_ptr->ligand.rigid.orientation_q[0][0] = m_ligand.node.orientation().R_component_1();
			m_ptr->ligand.rigid.orientation_q[0][1] = m_ligand.node.orientation().R_component_2();
			m_ptr->ligand.rigid.orientation_q[0][2] = m_ligand.node.orientation().R_component_3();
			m_ptr->ligand.rigid.orientation_q[0][3] = m_ligand.node.orientation().R_component_4();

			for (int i = 0; i < 3; i++)
			{
				m_ptr->ligand.rigid.axis[0][i] = 0;
				m_ptr->ligand.rigid.relative_axis[0][i] = 0;
				m_ptr->ligand.rigid.relative_origin[0][i] = 0;
			}

			// Store children nodes (in depth-first order)
			struct tmp_struct
			{
				int start_index = 0;
				int parent_index = 0;
				void store_node(tree<segment> &child_ptr, rigid_cl &rigid)
				{
					start_index++; // start with index 1, index 0 is root node
					rigid.parent[start_index] = parent_index;
					rigid.atom_range[start_index][0] = child_ptr.node.begin;
					rigid.atom_range[start_index][1] = child_ptr.node.end;
					for (int i = 0; i < 9; i++)
					{
						}
					rigid.orientation_q[start_index][0] = child_ptr.node.orientation().R_component_1();
					rigid.orientation_q[start_index][1] = child_ptr.node.orientation().R_component_2();
					rigid.orientation_q[start_index][2] = child_ptr.node.orientation().R_component_3();
					rigid.orientation_q[start_index][3] = child_ptr.node.orientation().R_component_4();
					for (int i = 0; i < 3; i++)
					{
						rigid.origin[start_index][i] = child_ptr.node.get_origin()[i];
						rigid.axis[start_index][i] = child_ptr.node.get_axis()[i];
						rigid.relative_axis[start_index][i] = child_ptr.node.relative_axis[i];
						rigid.relative_origin[start_index][i] = child_ptr.node.relative_origin[i];
					}
					if (child_ptr.children.size() == 0)
						return;
					else
					{
						if (start_index >= MAX_NUM_OF_RIGID)
						{
							throw std::runtime_error("/lib/main_procedure_cu.cpp: Children map too large!");
						}
						int parent_index_tmp = start_index;
						for (int i = 0; i < child_ptr.children.size(); i++)
						{
							this->parent_index = parent_index_tmp; // Update parent index
							this->store_node(child_ptr.children[i], rigid);
						}
					}
				}
			};
			tmp_struct ts;
			for (int i = 0; i < m_ligand.children.size(); i++)
			{
				ts.parent_index = 0; // Start a new branch, whose parent is 0
				ts.store_node(m_ligand.children[i], m_ptr->ligand.rigid);
			}
			m_ptr->ligand.rigid.num_children = ts.start_index;
			// std::cout << "m_ptr->ligand.rigid.num_children [num_torsion] = " << ts.start_index << std::endl;

			// set children_map
#pragma omp parallel for
			for (int i = 0; i < MAX_NUM_OF_RIGID; i++)
			{
				m_ptr->ligand.rigid.children_map[i] = 0ULL;
			}

#pragma omp parallel for
			for (int i = 1; i < m_ptr->ligand.rigid.num_children + 1; i++)
			{
				int parent_index = m_ptr->ligand.rigid.parent[i];
				m_ptr->ligand.rigid.children_map[parent_index] |= (1ULL << i);
			}
			m_ptr->m_num_movable_atoms = m.num_movable_atoms();
			// std::cout << "m_ptr->m_num_movable_atoms = " << m_ptr->m_num_movable_atoms << std::endl;
			size_t m_size = sizeof(m_cl);

			// Init ligand atom coords
			result_coords_ptrs[ligand_count] = (ligand_atom_coords_cl *)malloc(par.mc.thread * sizeof(ligand_atom_coords_cl));
			// memset(result_coords_ptrs[ligand_count], 0, sizeof(ligand_atom_coords_cl));

			// Init results
			result_ptrs[ligand_count] = (output_type_cl *)malloc(par.mc.thread * sizeof(output_type_cl));
			// memset(result_ptrs[ligand_count],0,sizeof(output_type_cl));
			// cl_mem ric_gpu;
			// output_type_cl* ric_gpus[ligand_count];
			boost_t.restart();
			struct timespec _m0; clock_gettime(CLOCK_MONOTONIC, &_m0);
			#ifdef STREAMING_PIPE
			checkCUDA(cudaMallocAsync((void **)&ric_gpus[ligand_count], ric_size, s1_comp));
			#else
			checkCUDA(cudaMalloc((void **)&ric_gpus[ligand_count], ric_size));
			#endif
			// GT_BLOCK_START();
			// [v7-GPU] ric HtoD skipped: generated on device

			// checkCUDA(cudaMalloc((void **)&ric_gpus[ligand_count], ric_size));

			// [Path B merged] m_gpus is read-only in kernel2_update — all writes go to
			// shared memory. Allocate a single copy instead of 8192 identical copies
			// (saves ~289 MB HtoD per ligand; CASF-certified in memcpy_single_m).
			#ifdef STREAMING_PIPE
			checkCUDA(cudaMallocAsync((void **)&m_gpus[ligand_count], m_size, s1_comp));
			#else
			checkCUDA(cudaMalloc((void **)&m_gpus[ligand_count], m_size));
			#endif
			#ifdef STREAMING_PIPE
			checkCUDA(cudaMemcpyAsync(m_gpus[ligand_count], m_ptr, m_size, cudaMemcpyHostToDevice, s1_comp));
			#else
			checkCUDA(cudaMemcpy(m_gpus[ligand_count], m_ptr, m_size, cudaMemcpyHostToDevice));
			#endif
			{ struct timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t); fprintf(stderr, "PP2=%.2f\n", (_t.tv_sec-_g_tv.tv_sec)*1e3+(_t.tv_nsec-_g_tv.tv_nsec)/1e6); _g_tv = _t; } // [S1-FIX3] async on s1_comp: no CPU wait, ordered before kernel2[j]

			// random_maps* random_maps_gpus[ligand_count];
			#ifdef STREAMING_PIPE
			checkCUDA(cudaMallocAsync((void **)&random_maps_gpus[ligand_count], rand_maps_size, s1_comp));
			#else
			checkCUDA(cudaMalloc((void **)&random_maps_gpus[ligand_count], rand_maps_size));
			#endif
			// [v7-GPU] random_maps generated on device (HtoD skipped)
			v7_gen_randmaps_gpu(random_maps_gpus[ligand_count], n_tors_gpu,
					(unsigned long long)seed + 77u, stream8);
			{ struct timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t); fprintf(stderr, "PP1=%.2f\n", (_t.tv_sec-_g_tv.tv_sec)*1e3+(_t.tv_nsec-_g_tv.tv_nsec)/1e6); _g_tv = _t; }

			#ifdef STREAMING_PIPE
			checkCUDA(cudaMallocAsync((void **)&result_coords_gpus[ligand_count], mis_ptr->thread * sizeof(ligand_atom_coords_cl), s1_comp));
			#else
			checkCUDA(cudaMalloc((void **)&result_coords_gpus[ligand_count], mis_ptr->thread * sizeof(ligand_atom_coords_cl)));
			#endif
			// checkCUDA(cudaMemcpy(result_coords_gpus[ligand_count], result_coords_ptrs[ligand_count], mis_ptr->thread * sizeof(ligand_atom_coords_cl), cudaMemcpyHostToDevice));
			// [v7-GPU] result_coords HtoD skipped (device writes coords)

			// output_type_cl* result_gpus[ligand_count];
			#ifdef STREAMING_PIPE
			checkCUDA(cudaMallocAsync((void **)&result_gpus[ligand_count], par.mc.thread * sizeof(output_type_cl), s1_comp));
			#else
			checkCUDA(cudaMalloc((void **)&result_gpus[ligand_count], par.mc.thread * sizeof(output_type_cl)));
			#endif
			{ struct timespec _m1; clock_gettime(CLOCK_MONOTONIC, &_m1);
			  fprintf(stderr, "MALLOC_MS=%.2f\n", (_m1.tv_sec-_m0.tv_sec)*1e3+(_m1.tv_nsec-_m0.tv_nsec)/1e6); }
			// [v6b] staging for the post-loop batch kernel call — MUST sit after the
			// LAST cudaMalloc of this ligand (FIX2: early stash captured NULL for
			// rm/coords/results because their cudaMallocs come later in the block)
			mg_batch_arr[ligand_count] = m_gpus[ligand_count];
			rm_batch_arr[ligand_count] = random_maps_gpus[ligand_count];
			ric_batch_arr[ligand_count] = ric_gpus[ligand_count];
			coords_batch_arr[ligand_count] = result_coords_gpus[ligand_count];
			results_batch_arr[ligand_count] = result_gpus[ligand_count];
			search_depths_arr[ligand_count] = par.mc.search_depth[ligand_count];
			bfgs_steps_arr[ligand_count] = par.mc.ssd_par.bfgs_steps[ligand_count];
			// checkCUDA(cudaMemcpy(result_gpus[ligand_count], result_ptrs[ligand_count], par.mc.thread * sizeof(output_type_cl), cudaMemcpyHostToDevice));
			// checkCUDA(cudaMemcpy(result_gpus[ligand_count], result_ptrs[ligand_count], par.mc.thread * sizeof(output_type_cl), cudaMemcpyHostToDevice));
GT_BLOCK_END(0);
			// [v7-GPU] result HtoD skipped (result_ptrs all zero; device writes results)
			{
				const float cx = corner1[0], cy = corner1[1], cz = corner1[2];
				const float sx = corner2[0] - corner1[0], sy = corner2[1] - corner1[1], sz2 = corner2[2] - corner1[2];
				v7_gen_ric_gpu(ric_gpus[ligand_count], par.mc.thread, cx, cy, cz, sx, sy, sz2,
						n_tors_gpu, (unsigned long long)seed + 991u, stream8);
			{ struct timespec _t; clock_gettime(CLOCK_MONOTONIC, &_t); fprintf(stderr, "PP3=%.2f\n", (_t.tv_sec-_g_tv.tv_sec)*1e3+(_t.tv_nsec-_g_tv.tv_nsec)/1e6); _g_tv = _t; }
			}
			// std::cout << "time of memcpy for kernel monte carlo: " << boost_t.elapsed() << std::endl;

			/**************************************************************************/
			/************************  Call to kernel function  ***********************/
			/**************************************************************************/
			// Call to kernel function

#ifdef HOST_PROBE
			fprintf(stderr, "PRE_KM\n");
#endif
			// [v6b] kernel launch moved AFTER the per-ligand loop: one batched
			// kernel_monte call for all fully-prepared ligands (v6 architecture)
			lig_valid_arr[ligand_count] = 1;
#ifdef STREAMING_PIPE
			// [S1] inline launch. The randmaps/ric generator kernels for THIS
			// ligand were enqueued on stream8 within its prep; gate stream0 on
			// ev_gen[ligand_count] (per-ligand cross-stream dependency), then
			// launch kernel2 on stream0 and record ev_done. Backpressure via
			// ev_done queries on the window base cursor.
			while (s1_launched.load() - s1_inflight_base >= s1_win) {
				if (cudaEventQuery(ev_done[s1_inflight_base]) == cudaSuccess)
					s1_inflight_base++;
				else
					break;
			}
			struct timespec _pp; clock_gettime(CLOCK_MONOTONIC, &_pp);
			fprintf(stderr, "PREPT[j]=%.3f\n", (_pp.tv_sec - _g_tv.tv_sec) * 1e3 + (_pp.tv_nsec - _g_tv.tv_nsec) / 1e6);
			_g_tv = _pp;
			cudaEventRecord(ev_gen[ligand_count], stream8);
			cudaStreamWaitEvent(s1_comp, ev_gen[ligand_count], 0);
			kernel_monte_stream_lig(ric_batch_arr[ligand_count], mg_batch_arr[ligand_count],
				pre_gpu, grids_gpu, rm_batch_arr[ligand_count], coords_batch_arr[ligand_count],
				results_batch_arr[ligand_count], mis_gpu, torsion_sizes[ligand_count],
				search_depths_arr[ligand_count], bfgs_steps_arr[ligand_count], rilc_bfgs,
				ev_done[ligand_count], s1_comp);
			s1_launched++;
#endif
			hp_mark("km_done");
#ifdef HOST_PROBE
			fprintf(stderr, "POST_KM\n");
#endif

		}
		catch (...)
		{
			continue;
		}
	}

	// [STREAM] ligand-level software pipeline. The prep loop above kept its
	// in-loop launches REMOVED (launches now happen here, windowed). Main
	// thread pumps prep->launch; 4 consumers drain finished ligands.
	// Window bounds in-flight device buffers (W6c density x SM headroom).
#ifdef STREAMING_PIPE
	// [S1-DEBUG] launch via the certified kernel_monte path instead of inline
	fprintf(stderr, "KMARGS j0 ric=%p mg=%p rm=%p co=%p rs=%p mis=%p tors=%d sd=%d bfgs=%d rilc=%d\n",
		(void*)ric_batch_arr[0], (void*)mg_batch_arr[0], (void*)rm_batch_arr[0],
		(void*)coords_batch_arr[0], (void*)results_batch_arr[0], (void*)mis_gpu,
		torsion_sizes[0], search_depths_arr[0], bfgs_steps_arr[0], rilc_bfgs);
	kernel_monte(ric_batch_arr.data(), mg_batch_arr.data(), pre_gpu, grids_gpu,
				 rm_batch_arr.data(), coords_batch_arr.data(), results_batch_arr.data(),
				 mis_gpu, torsion_sizes.data(), search_depths_arr.data(),
				 bfgs_steps_arr.data(), rilc_bfgs, num_ligands, par.mc.thread,
				 lig_valid_arr.data());
	// consumers already did their DtoH pass BEFORE this launch (they were not
	// gated on real kernels in bypass mode) -> their data is garbage; redo DtoH
	for (int j = 0; j < num_ligands; j++) {
		if (lig_valid_arr[j] == 0) continue;
		checkCUDA(cudaMemcpy(result_ptrs[j], result_gpus[j], par.mc.thread * sizeof(output_type_cl), cudaMemcpyDeviceToHost));
		checkCUDA(cudaMemcpy(result_coords_ptrs[j], result_coords_gpus[j], par.mc.thread * sizeof(ligand_atom_coords_cl), cudaMemcpyDeviceToHost));
	}
	for (auto &c : s1_consumers)
		c.join();
	printf("V6B_STREAM_S1_LAUNCHED=%d\n", s1_launched.load());
	for (int j = 0; j < num_ligands; j++)
		if (lig_valid_arr[j] == 0) continue;
		else checkCUDA(cudaEventSynchronize(ev_dtoh[j]));
	for (int j = 0; j < num_ligands; j++) {
		if (lig_valid_arr[j] == 0) continue;
		std::vector<output_type> result_vina = cl_to_vina(
			result_ptrs[j], result_coords_ptrs[j], par.mc.thread, torsion_sizes[j]);
		if (result_vina.size() == 0) {
			status = ABORT;
			exit(-1);
		}
		for (int i = 0; i < par.mc.thread; i++)
			add_to_output_container(outs[j], result_vina[i], par.mc.min_rmsd, par.mc.num_saved_mins);
		checkCUDA(cudaFreeAsync(ric_gpus[j], s1_comp));
		checkCUDA(cudaFreeAsync(m_gpus[j], s1_comp));
		checkCUDA(cudaFreeAsync(random_maps_gpus[j], s1_comp));
		checkCUDA(cudaFreeAsync(result_coords_gpus[j], s1_comp));
		checkCUDA(cudaFreeAsync(result_gpus[j], s1_comp));
	}
	for (int j = 0; j < num_ligands; j++) {
		checkCUDA(cudaEventDestroy(ev_done[j]));
		checkCUDA(cudaEventDestroy(ev_gen[j]));
		checkCUDA(cudaEventDestroy(ev_dtoh[j]));
	}

#else
		kernel_monte(ric_batch_arr.data(), mg_batch_arr.data(), pre_gpu, grids_gpu,
					 rm_batch_arr.data(), coords_batch_arr.data(), results_batch_arr.data(),
					 mis_gpu, torsion_sizes.data(), search_depths_arr.data(),
					 bfgs_steps_arr.data(), rilc_bfgs, num_ligands, par.mc.thread,
					 lig_valid_arr.data());
#endif

#ifndef STREAMING_PIPE
	// [v6b-FIX5] per-ligand result consumption AFTER the batched kernel:
	// DtoH/cl_to_vina/add_to_output + the per-ligand device frees (buffers
	// must stay alive until the batched kernel has finished).
	for (int ligand_count = 0; ligand_count < num_ligands; ligand_count++)
	{
		if (lig_valid_arr[ligand_count] == 0) continue;
		try
		{
			// Init ligand atom coords
			result_coords_ptrs[ligand_count] = (ligand_atom_coords_cl *)malloc(par.mc.thread * sizeof(ligand_atom_coords_cl));

			// Init results
			result_ptrs[ligand_count] = (output_type_cl *)malloc(par.mc.thread * sizeof(output_type_cl));
			// std::vector<output_type_cl*>		result_ptrs			(num_ligands);
			// std::vector<ligand_atom_coords_cl*> result_coords_ptrs	(num_ligands);

			GT_BLOCK_START();
			checkCUDA(cudaMemcpy(result_ptrs[ligand_count], result_gpus[ligand_count], par.mc.thread * sizeof(output_type_cl), cudaMemcpyDeviceToHost));

			checkCUDA(cudaMemcpy(result_coords_ptrs[ligand_count], result_coords_gpus[ligand_count], par.mc.thread * sizeof(ligand_atom_coords_cl), cudaMemcpyDeviceToHost));
			checkCUDA(cudaMemcpy(result_coords_ptrs[ligand_count], result_coords_gpus[ligand_count], par.mc.thread * sizeof(ligand_atom_coords_cl), cudaMemcpyDeviceToHost));
GT_BLOCK_END(1);

			hp_mark("dtoh_results");
			std::vector<output_type> result_vina =
cl_to_vina(result_ptrs[ligand_count], result_coords_ptrs[ligand_count],
															  par.mc.thread, torsion_sizes[ligand_count]);
			// if empty, something goes wrong in the device part
			if (result_vina.size() == 0)
			{
				status = ABORT;
#if defined(NDEBUG) && defined(SHOW_PROGRESS_BAR)
				console_thread.join();
#endif
				exit(-1);
			}

			hp_mark("cl_to_vina");
			// Write back to vina
#pragma omp parallel for
			for (int i = 0; i < par.mc.thread; i++)
			{
				// assert(result_vina[i].coords.size()== mis_ptr->)
				add_to_output_container(outs[ligand_count], result_vina[i], par.mc.min_rmsd, par.mc.num_saved_mins);
			}
			// VINA_CHECK(!outs[ligand_count].empty());
			// VINA_CHECK(outs[ligand_count].front().e <= outs[ligand_count].back().e);

			hp_mark("add_to_out");
			free(rand_maps_ptrs[ligand_count]);
			free(ric_ptrs[ligand_count]);
			checkCUDA(cudaFreeHost(m_ptrs[ligand_count]));
			free(result_coords_ptrs[ligand_count]);
			free(result_ptrs[ligand_count]);

			checkCUDA(cudaFree(result_coords_gpus[ligand_count]));
			checkCUDA(cudaFree(result_gpus[ligand_count]));
			checkCUDA(cudaFree(ric_gpus[ligand_count]));
			checkCUDA(cudaFree(m_gpus[ligand_count]));
			checkCUDA(cudaFree(random_maps_gpus[ligand_count]));
		}
		catch (...)
		{
			continue;
		}
	}
#endif // !STREAMING_PIPE

	free(mis_ptr);
	checkCUDA(cudaFree(mis_gpu));
	checkCUDA(cudaFree(grids_gpu));
	checkCUDA(cudaFree(pre_gpu));
	// free(mis_gpu_res);
	// free(grids_gpu_res);
	// free(pre_gpu_res);

// [GT] settle and report GPU-timeline accounting
	cudaDeviceSynchronize();
	gt_report();

	status = FINISH;
#if defined(NDEBUG) && defined(SHOW_PROGRESS_BAR)
	console_thread.join(); // wait the thread finish (only if UI thread was spawned)
#endif
#ifdef TIME_ANALYSIS
	// Output Analysis
	cl_ulong time_start, time_end;
	double total_time = 0;
	std::string log_file = "gpu_runtime.log";
	// delete file if file exists
	if (std::ifstream(log_file))
	{
		std::remove(log_file.c_str());
	}
	for (int ligand_count = 0; ligand_count < ms.size(); ligand_count++)
	{
		cl_event event_tmp = ligands_events[ligand_count];
		err = clGetEventProfilingInfo(event_tmp, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		checkErr(err);
		err = clGetEventProfilingInfo(event_tmp, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		checkErr(err);
		total_time = time_end - time_start;
		err = clReleaseEvent(ligands_events[ligand_count]);
		checkErr(err);
		printf("\nAutoDockVina-GPU3 ligand %d runtime = %0.3f s", ligand_count, (total_time / 1000000000.0));

		std::cout << std::setiosflags(std::ios::fixed);
		std::ofstream file(log_file, std::ios::app);
		if (file.is_open())
		{
			file << "AutoDockVina-GPU3 monte carlo runtime of ligand " << ligand_names[ligand_count][0] << " = " << std::setprecision(5) << (total_time / 1000000000.0) << " s" << std::endl;
			file.close();
		}
	}
	printf("\n");
#endif
}