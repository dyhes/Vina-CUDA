// bh_shared.h — [Q7-BH] dynamic shared-memory carve for 8-conf blocks
#ifndef BH_SHARED_H
#define BH_SHARED_H
#ifndef BH_NCONF
#define BH_NCONF 8
#endif
#define BH_TPB 128

struct BhConfState {
	float warp_e[32];
	unsigned long long skip_row[(MAX_NUM_OF_LIG_PAIRS + 63) / 64];
	int q4_probe_epoch;
	int q4_bitmap_init;
	flo pos_der_tmp[MAX_NUM_OF_RIGID][3];
	flo orient_der_tmp[MAX_NUM_OF_RIGID][3];
	int q4_eval_idx;
	flo sp_tmp[MAX_NUM_OF_DIMENSIONS_OF_MATIRX];
	output_type_cl xp;
	change_cl gp;
	flo lm_s[MAX_NUM_OF_DIMENSIONS_OF_MATIRX];
	flo lm_y[MAX_NUM_OF_DIMENSIONS_OF_MATIRX];
	change_cl g_orig;
	output_type_cl x_orig;
	change_cl d;
	flo d_norm;
	float gyr_wacc[32];
	int gyr_wcnt[32];
	float mut_rotation[3];
	// kernel2-level per-conf state
	m_coords_cl m_coords;
	m_minus_forces minus_forces;
	output_type_cl tmp;
	change_cl g;
	output_type_cl best_out;
	output_type_cl candidate;
	ligand_gpu m_gpu_ligand;
	unsigned long long q4_skip_row[(MAX_NUM_OF_LIG_PAIRS + 63) / 64];
	int q4_epoch_i;
};

extern __shared__ BhConfState bh[];

__device__ __forceinline__ void conf_barrier(int conf_id) {
	__syncwarp(); // [Q7-BH] reconverge warp (thread0 gates diverge it)
	__asm__ volatile("bar.sync %0, %1;" :: "r"(conf_id + 1), "n"(BH_TPB));
}
__device__ __forceinline__ void conf_barrier_m(int conf_id) {
	conf_barrier(conf_id);
}
#endif