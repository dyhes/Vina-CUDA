#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "stage_timer.h"
#ifndef _QUASI_NEWTON_H_
#define _QUASI_NEWTON_H_


#include "commonMacros.h"
#include "matrix.cuh"
#include "visited.cuh"

// [v11] tile-parallel line search configuration (compile-time; 0 disables -> v4 serial path)
#ifndef V11_LS_GRID
#define V11_LS_GRID 8        // concurrent trial tiles: alpha_t = stp * 0.5^t, t = 0..V11_LS_GRID-1
#endif
#ifndef V11_LS_MAX_ATOMS
#define V11_LS_MAX_ATOMS 40  // ligands with more movable atoms keep the v4 serial LS (large-ligand guard)
#endif

// [v11 v2] dynamic shared memory: V11_LS_GRID per-tile slots (declared once per TU;
// kernel2_update must be launched with V11_TILE_SMEM_FLO*sizeof(flo) dynamic smem,
// opt-in via cudaFuncSetAttribute in kernel_monte)
extern __shared__ flo v11_slots[];

// [v11 v2] lightweight reference view of rigid_cl: array members -> pointers, so the
// tile's per-trial rigid state can live in its dynamic-smem slot while read-only
// skeleton fields alias the master. Member names mirror rigid_cl.
struct v11_rigid_ref {
	int num_children;
	const uint64_t* children_map;   // [48] alias master
	const int*      parent;         // [48] alias master
	const int (*atom_range)[2];     // [48][2] alias master
	flo (*origin)[3];               // [48][3] -> slot
	flo (*orientation_m)[9];        // [48][9] -> slot
	flo (*orientation_q)[4];        // [48][4] -> slot
	flo (*axis)[3];                 // [48][3] -> slot
	const flo (*relative_axis)[3];  // [48][3] alias master
	const flo (*relative_origin)[3];// [48][3] alias master
};


__device__ inline int num_atom_types(int atu) {
	switch (atu) {
	case 0: return EL_TYPE_SIZE;
	case 1: return AD_TYPE_SIZE;
	case 2: return XS_TYPE_SIZE;
	case 3: return SY_TYPE_SIZE;
	default: printf("Kernel1:num_atom_types() ERROR!"); return INFINITY;
	}
}

__device__ inline void elementwise_product(flo* out, const flo* a, const flo* b) {
	out[0] = a[0] * b[0];
	out[1] = a[1] * b[1];
	out[2] = a[2] * b[2];
}

__device__ inline flo elementwise_product_sum(const flo* a, const flo* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

//__device__ inline flo access_m_data(  flo* m_data, int m_i, int m_j, int i, int j, int k) {  //delete the contant flo* m_data
//	return m_data[i + m_i * (j + m_j * k)];
//}

//__device__ inline bool not_max(flo x) {
//	return (x < 0.1 * INFINITY);// Problem: replace max_fl with INFINITY?
//}

__device__ inline void curl_with_deriv(flo* e, flo* deriv, flo v, const flo epsilon_fl) {
	if (*e > 0 && v < 0.1 * INFINITY) {
		flo tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		*e *= tmp;
		for (int i = 0; i < 3; i++) deriv[i] *= powf(tmp, 2);
	}
}

__device__ inline void curl_without_deriv(flo* e, flo v, const flo epsilon_fl) {
	if (*e > 0 && v < 0.1 * INFINITY) {
		flo tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		*e *= tmp;
	}
}

__device__ flo g_evaluate(   grid_cl*	g,  // delete the __constant
					const				flo*		m_coords,			// double[3]
					const				flo		slope,				// double
					const				flo		v,					// double
										flo*		deriv,				// double[3]
					const				flo		epsilon_fl
) {
	int m_i = g->m_i;
	int m_j = g->m_j;
	int m_k = g->m_k;
	//flo tmp_vec[3] = { m_coords[0] - g->m_init[0],m_coords[1] - g->m_init[1] ,m_coords[2] - g->m_init[2] };
	//flo tmp_vec2[3] = { g->m_factor[0],g->m_factor[1] ,g->m_factor[2] };
	flo s[3];
	//elementwise_product(s, tmp_vec, tmp_vec2); // 
	s[0] = (m_coords[0] - g->m_init[0]) *  g->m_factor[0];
	s[1] = (m_coords[1] - g->m_init[1]) *  g->m_factor[1];
	s[2] = (m_coords[2] - g->m_init[2]) *  g->m_factor[2];

	flo miss[3] = { 0,0,0 };
	int region[3];
	int a[3];
	int m_data_dims[3] = { m_i,m_j,m_k };
	for (int i = 0; i < 3; i++){
		if (s[i] < 0) {
			miss[i] = -s[i];
			region[i] = -1;
			a[i] = 0;
			s[i] = 0;
		}
		else if (s[i] >= g->m_dim_fl_minus_1[i]) {
			miss[i] = s[i] - g->m_dim_fl_minus_1[i];
			region[i] = 1;
			if (m_data_dims[i] < 2)printf("\nKernel2: g_evaluate ERROR!#2");
			a[i] = m_data_dims[i] - 2;
			s[i] = 1;
		}
		else {
			region[i] = 0;
			a[i] = (int)s[i];
			s[i] -= a[i];
		}
		if (s[i] < 0 || s[i] > 1) printf("\nKernel2: g_evaluate ERROR! #3 and #4");
		//if (s[i] > 1)printf("\nKernel2: g_evaluate ERROR!#4");
		//if (a[i] < 0)printf("\nKernel2: g_evaluate ERROR!#5");
		if (a[i] < 0 || a[i] + 1 >= m_data_dims[i]) printf("\nKernel2: g_evaluate ERROR!#5");
	}

	//flo tmp_m_factor_inv[3] = { g->m_factor_inv[0],g->m_factor_inv[1],g->m_factor_inv[2] };
	//const flo penalty = slope * elementwise_product_sum(miss, tmp_m_factor_inv);
	const flo penalty = slope * (miss[0] * g->m_factor_inv[0] + miss[1] * g->m_factor_inv[1] + miss[2] * g->m_factor_inv[2]);

	const int x0 = a[0];
	const int y0 = a[1];
	const int z0 = a[2];

	int base = (x0 + m_i * (y0 + m_j * z0)) * 8;
	 flo* base_ptr = &g->m_data[base]; // delete the constant

	const flo f000 = *base_ptr;
	const flo f100 = *(base_ptr + 1);
	const flo f010 = *(base_ptr + 2);
	const flo f110 = *(base_ptr + 3);
	const flo f001 = *(base_ptr + 4);
	const flo f101 = *(base_ptr + 5);
	const flo f011 = *(base_ptr + 6);
	const flo f111 = *(base_ptr + 7);

	const flo x = s[0];
	const flo y = s[1];
	const flo z = s[2];
		  
	const flo mx = 1 - x;
	const flo my = 1 - y;
	const flo mz = 1 - z;

	flo f =
		f000 * mx * my * mz +
		f100 * x  * my * mz +
		f010 * mx * y  * mz +
		f110 * x  * y  * mz +
		f001 * mx * my * z	+
		f101 * x  * my * z	+
		f011 * mx * y  * z	+
		f111 * x  * y  * z  ;

	{ // deriv always valid
		const flo x_g =
			f000 * (-1) * my * mz +
			f100 *   1  * my * mz +
			f010 * (-1) * y  * mz +
			f110 *	 1  * y  * mz +
			f001 * (-1) * my * z  +
			f101 *   1  * my * z  +
			f011 * (-1) * y  * z  +
			f111 *   1  * y  * z  ;


		const flo y_g =
			f000 * mx * (-1) * mz +
			f100 * x  * (-1) * mz +
			f010 * mx *   1  * mz +
			f110 * x  *   1  * mz +
			f001 * mx * (-1) * z  +
			f101 * x  * (-1) * z  +
			f011 * mx *   1  * z  +
			f111 * x  *   1  * z  ;


		const flo z_g =
			f000 * mx * my * (-1) +
			f100 * x  * my * (-1) +
			f010 * mx * y  * (-1) +
			f110 * x  * y  * (-1) +
			f001 * mx * my *   1  +
			f101 * x  * my *   1  +
			f011 * mx * y  *   1  +
			f111 * x  * y  *   1  ;

		flo gradient[3] = { x_g, y_g, z_g };

		curl_with_deriv(&f, gradient, v, epsilon_fl);

		flo gradient_everywhere[3];

		for (int i = 0; i < 3; i++) {
			gradient_everywhere[i] = ((region[i] == 0) ? gradient[i] : 0);
			deriv[i] = g->m_factor[i] * gradient_everywhere[i] + slope * region[i];
		}


		return f + penalty;
	}
}

__device__ flo g_evaluate_update(   grid_cl*	g,  // delete the __constant
					const				flo*		m_coords,			// double[3]
					const				flo		slope,				// double
					const				flo		v,					// double
										flo*		deriv,				// double[3]
					const				flo		epsilon_fl,
					grids_cl_other* grids_cuda
					
) {
	int m_i = grids_cuda->m_i;
	int m_j = grids_cuda->m_j;
	int m_k = grids_cuda->m_k;
	if(m_i * m_j * m_k == 0)printf("\nkernel2: g_evaluate ERROR!#1");
	
	flo s[3];
	//elementwise_product(s, tmp_vec, tmp_vec2); // 

	s[0] = (m_coords[0] - grids_cuda->m_init[0]) *  grids_cuda->m_factor[0];
	s[1] = (m_coords[1] - grids_cuda->m_init[1]) *  grids_cuda->m_factor[1];
	s[2] = (m_coords[2] - grids_cuda->m_init[2]) *  grids_cuda->m_factor[2];
	//__threadfence;

	flo miss[3] = { 0,0,0 };
	 int region[3];
	 int a[3];
	int m_data_dims[3] = { m_i,m_j,m_k };

	//for (int i = threadNumInBlock; i < 3; i += threadsPerBlock){
	for (int i = 0; i < 3; i++){
		flo dim_minus_1 = grids_cuda->m_dim_fl_minus_1[i];
		bool below = (s[i] < 0);
		bool above = (s[i] >= dim_minus_1);
		flo miss_below = -s[i];
		flo miss_above = s[i] - dim_minus_1;
		int a_inside = (int)s[i];
		flo s_inside = s[i] - (flo)a_inside;
		miss[i] = below ? miss_below : (above ? miss_above : 0.0f);
		region[i] = below ? -1 : (above ? 1 : 0);
		a[i] = below ? 0 : (above ? (m_data_dims[i] - 2) : a_inside);
		s[i] = below ? 0.0f : (above ? 1.0f : s_inside);
	}

	//flo tmp_m_factor_inv[3] = { g->m_factor_inv[0],g->m_factor_inv[1],g->m_factor_inv[2] };
	//const flo penalty = slope * elementwise_product_sum(miss, tmp_m_factor_inv);
	const flo penalty = slope * (miss[0] * grids_cuda->m_factor_inv[0] + miss[1] * grids_cuda->m_factor_inv[1] + miss[2] * grids_cuda->m_factor_inv[2]);
	if (penalty <= -epsilon_fl) printf("\nKernel2: g_evaluate ERROR!#6");

	const int x0 = a[0];
	const int y0 = a[1];
	const int z0 = a[2];

	int base = (x0 + m_i * (y0 + m_j * z0)) * 8;
	 flo* base_ptr = &g->m_data[base]; // delete the constant

	const flo f000 = *base_ptr;
	const flo f100 = *(base_ptr + 1);
	const flo f010 = *(base_ptr + 2);
	const flo f110 = *(base_ptr + 3);
	const flo f001 = *(base_ptr + 4);
	const flo f101 = *(base_ptr + 5);
	const flo f011 = *(base_ptr + 6);
	const flo f111 = *(base_ptr + 7);

	const flo x = s[0];
	const flo y = s[1];
	const flo z = s[2];
		  
	const flo mx = 1 - x;
	const flo my = 1 - y;
	const flo mz = 1 - z;

	flo f =
		f000 * mx * my * mz +
		f100 * x  * my * mz +
		f010 * mx * y  * mz +
		f110 * x  * y  * mz +
		f001 * mx * my * z	+
		f101 * x  * my * z	+
		f011 * mx * y  * z	+
		f111 * x  * y  * z  ;

	if (deriv) { // valid pointer
		const flo x_g =
			f000 * (-1) * my * mz +
			f100 *   1  * my * mz +
			f010 * (-1) * y  * mz +
			f110 *	 1  * y  * mz +
			f001 * (-1) * my * z  +
			f101 *   1  * my * z  +
			f011 * (-1) * y  * z  +
			f111 *   1  * y  * z  ;


		const flo y_g =
			f000 * mx * (-1) * mz +
			f100 * x  * (-1) * mz +
			f010 * mx *   1  * mz +
			f110 * x  *   1  * mz +
			f001 * mx * (-1) * z  +
			f101 * x  * (-1) * z  +
			f011 * mx *   1  * z  +
			f111 * x  *   1  * z  ;


		const flo z_g =
			f000 * mx * my * (-1) +
			f100 * x  * my * (-1) +
			f010 * mx * y  * (-1) +
			f110 * x  * y  * (-1) +
			f001 * mx * my *   1  +
			f101 * x  * my *   1  +
			f011 * mx * y  *   1  +
			f111 * x  * y  *   1  ;

		flo gradient[3] = { x_g, y_g, z_g };

		curl_with_deriv(&f, gradient, v, epsilon_fl);

		flo gradient_everywhere[3];

		for (int i = 0; i < 3; i++) {
			gradient_everywhere[i] = ((region[i] == 0) ? gradient[i] : 0);
			deriv[i] = grids_cuda->m_factor[i] * gradient_everywhere[i] + slope * region[i];
		}


		return f + penalty;
	}	
	else {  // none valid pointer
		printf("\nKernel2: g_evaluate ERROR!#7");
		curl_without_deriv(&f, v, epsilon_fl);
		return f + penalty;
	}
}


__device__ flo ig_eval_deriv(				output_type_cl*		x,
											change_cl*			g, 
						const				flo				v,
											grids_cl*			grids,  // delete the constant
											m_cl*				m,
						const				flo				epsilon_fl
) {
	flo e = 0;
	int nat = num_atom_types(grids->atu);
	for (int i = 0; i < m->m_num_movable_atoms; i++) {
		int t = m->atoms[i].types[grids->atu];
		if (t >= nat) {
			for (int j = 0; j < 3; j++)m->minus_forces.coords[i][j] = 0;
			continue;
		}
		flo deriv[3];

		e = e + g_evaluate(&grids->grids[t], m->m_coords.coords[i], grids->slope, v, deriv, epsilon_fl);

		for (int j = 0; j < 3; j++) m->minus_forces.coords[i][j] = deriv[j];
	}
	return e;
}

__device__ flo ig_eval_deriv_update_optimal(		output_type_cl*		x,
											change_cl*			g, 
						const				flo				v,
											grids_cl*	grids,  // delete the constant
											m_cl*				m,
						const				flo				epsilon_fl,
											m_coords_cl*	m_coords,
											m_minus_forces* minus_forces,
						const				int				threadNumInBlock,
						const				int				threadsPerBlock,
						                    grids_gpu*      grids_cuda
) {

	int nat = num_atom_types(grids->atu);

	// [FUSION v2] warp-shuffle reduction: register accumulation + single-warp tree, remove 512B shared e[]
	float my_e = 0.0f;
	float deriv[3] = {0};

	
	for (int i = threadNumInBlock;
		i < m->m_num_movable_atoms;
		i = i + threadsPerBlock
	){
		if (i < m->m_num_movable_atoms) {
			int t = m->atoms[i].types[grids->atu];
			if (t >= nat) {
				//for (int j = 0; j < 3; j++)
				minus_forces->coords[i][0] = 0;
				minus_forces->coords[i][1] = 0;
				minus_forces->coords[i][2] = 0;
			} else {
				
				my_e += g_evaluate_update(&grids->grids[t], m_coords->coords[i], grids->slope, v, deriv, epsilon_fl,&grids_cuda->grid_other[t]);

				//for (int j = 0; j < 3; j++)
				minus_forces->coords[i][0] = deriv[0];
				minus_forces->coords[i][1] = deriv[1];
				minus_forces->coords[i][2] = deriv[2];
			}
		}
	}__syncthreads();
	//Sum intermolecular energy
	// [v11 fix] block-level reduction: in-warp shfl + cross-warp shared tree.
	// The old single-warp shfl version silently dropped warps 1..3 at blockDim=128 (Stage A bug).
	{
		__shared__ flo warp_e[32];
		const int lane = threadNumInBlock & 31;
		const int wid = threadNumInBlock >> 5;
		for (int offset = 16; offset > 0; offset >>= 1)
			my_e += __shfl_down_sync(0xFFFFFFFF, my_e, offset);
		if (lane == 0) warp_e[wid] = my_e;
		__syncthreads();
		if (wid == 0) {
			flo v = (lane < (threadsPerBlock >> 5)) ? warp_e[lane] : 0.0f;
			for (int offset = 16; offset > 0; offset >>= 1)
				v += __shfl_down_sync(0xFFFFFFFF, v, offset);
			if (lane == 0) warp_e[0] = v;
		}
		__syncthreads();
		my_e = warp_e[0];
	}
	return my_e;
}


__device__ inline void quaternion_to_r3(const flo* q, flo* orientation_m) {
	// Omit assert(quaternion_is_normalized(q));
	const flo a = q[0];
	const flo b = q[1];
	const flo c = q[2];
	const flo d = q[3];

	const flo aa = a * a;
	const flo ab = a * b;
	const flo ac = a * c;
	const flo ad = a * d;
	const flo bb = b * b;
	const flo bc = b * c;
	const flo bd = b * d;
	const flo cc = c * c;
	const flo cd = c * d;
	const flo dd = d * d;

	
	
	orientation_m[0] = aa + bb - cc - dd;
	orientation_m[3] = 2 * (-ad + bc);
	orientation_m[6] = 2 * (ac + bd);

	orientation_m[1] = 2 * (ad + bc);
	orientation_m[4] = (aa - bb + cc - dd);
	orientation_m[7] = 2 * (-ab + cd);

	orientation_m[2] = 2 * (-ac + bd);
	orientation_m[5] = 2 * (ab + cd);
	orientation_m[8] = (aa - bb - cc + dd);
	

	
}

__device__ inline void local_to_lab_direction(			flo* out,
									const	flo* local_direction,
									const	flo* orientation_m
) {
	/*
	out[0] =	orientation_m[0] * local_direction[0] +
				orientation_m[3] * local_direction[1] +
				orientation_m[6] * local_direction[2];
	out[1] =	orientation_m[1] * local_direction[0] +
				orientation_m[4] * local_direction[1] +
				orientation_m[7] * local_direction[2];
	out[2] =	orientation_m[2] * local_direction[0] +
				orientation_m[5] * local_direction[1] +
				orientation_m[8] * local_direction[2];*/


    flo ld0 = local_direction[0];
    flo ld1 = local_direction[1];
    flo ld2 = local_direction[2];


    flo om0_ld0 = orientation_m[0] * ld0;
    flo om1_ld0 = orientation_m[1] * ld0;
    flo om2_ld0 = orientation_m[2] * ld0;

    flo om3_ld1 = orientation_m[3] * ld1;
    flo om4_ld1 = orientation_m[4] * ld1;
    flo om5_ld1 = orientation_m[5] * ld1;

    flo om6_ld2 = orientation_m[6] * ld2;
    flo om7_ld2 = orientation_m[7] * ld2;
    flo om8_ld2 = orientation_m[8] * ld2;


    out[0] = om0_ld0 + om3_ld1 + om6_ld2;
    out[1] = om1_ld0 + om4_ld1 + om7_ld2;
    out[2] = om2_ld0 + om5_ld1 + om8_ld2;

}

__device__ inline void local_to_lab(						flo*		out,
							const				flo*		origin,
							const				flo*		local_coords,
							const				flo*		orientation_m
) {
	
	flo lc0 = local_coords[0];
    flo lc1 = local_coords[1];
    flo lc2 = local_coords[2];

	out[0] = origin[0] + (	orientation_m[0] * lc0 +
							orientation_m[3] * lc1 +
							orientation_m[6] * lc2
							);			 
	out[1] = origin[1] + (	orientation_m[1] * lc0 +
							orientation_m[4] * lc1 +
							orientation_m[7] * lc2
							);			 
	out[2] = origin[2] + (	orientation_m[2] * lc0 +
							orientation_m[5] * lc1 +
							orientation_m[8] * lc2
							);


}

__device__ inline void angle_to_quaternion2(	flo*		out,
									const		flo*		axis,
												flo		angle
) {
	if (sqrtf(powf(axis[0], 2) + powf(axis[1], 2) + powf(axis[2], 2)) - 1 >= 0.001) printf("\nkernel2: angle_to_quaternion() ERROR! --> norm3(axis) - 1 = %f",norm3(axis) - 1); // Replace assert(eq(axis.norm(), 1));
	normalize_angle(&angle);
	flo c = cos(angle / 2);
	flo s = sin(angle / 2);
	out[0] = c;
	out[1] = s * axis[0];
	out[2] = s * axis[1];
	out[3] = s * axis[2];
}

__device__ void set(	const				output_type_cl* x,
								rigid_cl*		lig_rigid_gpu,
								m_coords_cl*	m_coords_gpu,	
			const				atom_cl*		atoms,				
			const				int				m_num_movable_atoms,
			const				flo			epsilon_fl
) {
	//************** (root --> origin[0]) node.set_conf **************// 
	for (int i = 0; i < 3; i++) lig_rigid_gpu->origin[0][i] = x->position[i]; // set origin
	for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[0][i] = x->orientation[i]; // set orientation_q
	quaternion_to_r3(lig_rigid_gpu->orientation_q[0], lig_rigid_gpu->orientation_m[0]);// set orientation_m
	// set coords
	int begin = lig_rigid_gpu->atom_range[0][0];
	int end =	lig_rigid_gpu->atom_range[0][1];
	for (int i = begin; i < end; i++) {
		local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[0], &atoms[i].coords[0], lig_rigid_gpu->orientation_m[0]);
	}
	//************** end node.set_conf **************//

	//************** branches_set_conf **************//
	//update nodes in depth-first order
	for (int current = 1; current < lig_rigid_gpu->num_children + 1; current++) { // current starts from 1 (namely starts from first child node)
		int parent = lig_rigid_gpu->parent[current];
		flo torsion = x->lig_torsion[current - 1]; 
		local_to_lab(	lig_rigid_gpu->origin[current], 
						lig_rigid_gpu->origin[parent],
						lig_rigid_gpu->relative_origin[current],
						lig_rigid_gpu->orientation_m[parent]
						); // set origin
		local_to_lab_direction(	lig_rigid_gpu->axis[current], 
								lig_rigid_gpu->relative_axis[current],
								lig_rigid_gpu->orientation_m[parent]
								); // set axis
		flo tmp[4];
		flo parent_q[4] = {	lig_rigid_gpu->orientation_q[parent][0],
								lig_rigid_gpu->orientation_q[parent][1] ,
								lig_rigid_gpu->orientation_q[parent][2] ,
								lig_rigid_gpu->orientation_q[parent][3] };
		flo current_axis[3] = {	lig_rigid_gpu->axis[current][0],
									lig_rigid_gpu->axis[current][1],
									lig_rigid_gpu->axis[current][2] };

		angle_to_quaternion2(tmp, current_axis, torsion);
		angle_to_quaternion_multi(tmp, parent_q);
		quaternion_normalize_approx(tmp, epsilon_fl);

		for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[current][i] = tmp[i]; // set orientation_q
		quaternion_to_r3(lig_rigid_gpu->orientation_q[current], lig_rigid_gpu->orientation_m[current]); // set orientation_m

		// set coords
		begin = lig_rigid_gpu->atom_range[current][0];
		end =	lig_rigid_gpu->atom_range[current][1];
		for (int i = begin; i < end; i++) {
			local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[current], &atoms[i].coords[0], lig_rigid_gpu->orientation_m[current]);
		}
	}
	//************** end branches_set_conf **************//
}


__device__ void set_update(	const				output_type_cl* x,
								rigid_cl*		lig_rigid_gpu,
								m_coords_cl*	m_coords_gpu,	
			const				atom_cl*		atoms,				
			const				int				m_num_movable_atoms,
			const				flo			epsilon_fl
			
) {
	//************** (root --> origin[0]) node.set_conf **************// (CHECKED)
	
	lig_rigid_gpu->origin[0][0] = x->position[0]; // set origin
	lig_rigid_gpu->origin[0][1] = x->position[1]; // set origin
	lig_rigid_gpu->origin[0][2] = x->position[2]; // set origin
	
	for (int i = 0; i < 4; i+=2) {
		lig_rigid_gpu->orientation_q[0][i] = x->orientation[i];
		lig_rigid_gpu->orientation_q[0][i+1] = x->orientation[i+1]; // set orientation_q
	}
	quaternion_to_r3(lig_rigid_gpu->orientation_q[0], lig_rigid_gpu->orientation_m[0]);// set orientation_m
	// set coords
	int begin = lig_rigid_gpu->atom_range[0][0];
	int end =	lig_rigid_gpu->atom_range[0][1];
	for (int i = begin; i < end; i++) {
		local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[0], &atoms[i].coords[0], lig_rigid_gpu->orientation_m[0]); 
	}
	//************** end node.set_conf **************//

	//************** branches_set_conf **************//
	//update nodes in depth-first order
	for (int current = 1; current < lig_rigid_gpu->num_children + 1; current++) { // current starts from 1 (namely starts from first child node)
		int parent = lig_rigid_gpu->parent[current];
		flo torsion = x->lig_torsion[current - 1]; // torsions are all related to child nodes
		local_to_lab(	lig_rigid_gpu->origin[current], 
						lig_rigid_gpu->origin[parent],
						lig_rigid_gpu->relative_origin[current],
						lig_rigid_gpu->orientation_m[parent]
						); // set origin
		local_to_lab_direction(	lig_rigid_gpu->axis[current], 
								lig_rigid_gpu->relative_axis[current],
								lig_rigid_gpu->orientation_m[parent]
								); // set axis
		flo tmp[4];
		flo parent_q[4] = {	lig_rigid_gpu->orientation_q[parent][0],
								lig_rigid_gpu->orientation_q[parent][1] ,
								lig_rigid_gpu->orientation_q[parent][2] ,
								lig_rigid_gpu->orientation_q[parent][3] };
		flo current_axis[3] = {	lig_rigid_gpu->axis[current][0],
									lig_rigid_gpu->axis[current][1],
									lig_rigid_gpu->axis[current][2] };

		angle_to_quaternion2(tmp, current_axis, torsion);
		angle_to_quaternion_multi(tmp, parent_q);
		quaternion_normalize_approx(tmp, epsilon_fl);

		for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[current][i] = tmp[i]; // set orientation_q

		quaternion_to_r3(lig_rigid_gpu->orientation_q[current], lig_rigid_gpu->orientation_m[current]); // set orientation_m

		// set coords
		begin = lig_rigid_gpu->atom_range[current][0];
		end =	lig_rigid_gpu->atom_range[current][1];
		for (int i = begin; i < end; i++) {
			local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[current], &atoms[i].coords[0], lig_rigid_gpu->orientation_m[current]);
		}
	}
	//************** end branches_set_conf **************//
}

__device__ inline void p_eval_deriv(						flo*		out,
										int			type_pair_index,
										flo		r2,
						pre_cl*		pre,  // delete the constant 
					const				flo		epsilon_fl
) {
	const flo cutoff_sqr = pre->m_cutoff_sqr;
	p_m_data_cl* tmp = &pre->m_data[type_pair_index];
	flo r2_factored = tmp->factor * r2;
	int i1 = (int)(r2_factored);
	int i2 = i1 + 1;
	flo rem = r2_factored - i1;
	flo p1[2] = { tmp->smooth[i1][0], tmp->smooth[i1][1] };
	flo p2[2] = { tmp->smooth[i2][0], tmp->smooth[i2][1] };
	flo e = p1[0] + rem * (p2[0] - p1[0]);
	flo dor = p1[1] + rem * (p2[1] - p1[1]);
	out[0] = e;
	out[1] = dor;
}

__device__ inline void curl(flo* e, flo* deriv, flo v, const flo epsilon_fl) {
	if (*e > 0 && v < 0.1 * INFINITY) {
		flo tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		(*e) = tmp * (*e);
		for (int i = 0; i < 3; i++)deriv[i] = deriv[i] * (tmp * tmp);
	}
}

__device__ flo eval_interacting_pairs_deriv(  pre_cl*			pre,  // delete the constant
									const				flo			v,
									const				lig_pairs_cl*   pairs,
									const			 	m_coords_cl*	m_coords,
														m_minus_forces* minus_forces,
									const				flo			epsilon_fl
) {
	flo e = 0;

	for (int i = 0; i < pairs->num_pairs; i++) {
		const int ip[3] = { pairs->type_pair_index[i], pairs->a[i] ,pairs->b[i] };
		flo coords_b[3] = { m_coords->coords[ip[2]][0], m_coords->coords[ip[2]][1], m_coords->coords[ip[2]][2] };
		flo coords_a[3] = { m_coords->coords[ip[1]][0], m_coords->coords[ip[1]][1], m_coords->coords[ip[1]][2] };
		flo r[3] = { coords_b[0] - coords_a[0], coords_b[1] - coords_a[1] ,coords_b[2] - coords_a[2] };
		flo r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
	
		if (r2 < pre->m_cutoff_sqr) {
			flo tmp[2];
			p_eval_deriv(tmp, ip[0], r2, pre, epsilon_fl);
			flo force[3] = { r[0] * tmp[1], r[1] * tmp[1] ,r[2] * tmp[1] };
			curl(&tmp[0], force, v, epsilon_fl);
			e += tmp[0];
			for (int j = 0; j < 3; j++){
				minus_forces->coords[ip[1]][j] -= force[j];
				minus_forces->coords[ip[2]][j] += force[j];
			}
			//for (int j = 0; j < 3; j++)minus_forces->coords[ip[2]][j] += force[j];
		}
	}
	return e;
}

__device__ flo eval_interacting_pairs_deriv_update_optimal(  pre_cl*			 pre,  // delete the constant
									const				flo			v,
									const				lig_pairs_cl*    pairs,
									const			 	m_coords_cl*	 m_coords,
														m_minus_forces*  minus_forces,
									const				flo			     epsilon_fl,
									const				int				 threadNumInBlock,
									const				int				 threadsPerBlock
) {
	// [FUSION v1] force accumulation via shared-memory atomics (v11: 4 warps touch the same atom)
	float my_e = 0.0f;

	for (int i = threadNumInBlock; i < pairs->num_pairs; i = i + threadsPerBlock){
		const int ip[3] = { pairs->type_pair_index[i], pairs->a[i] ,pairs->b[i] };
		float coords_b[3] = { m_coords->coords[ip[2]][0], m_coords->coords[ip[2]][1], m_coords->coords[ip[2]][2] };
		float coords_a[3] = { m_coords->coords[ip[1]][0], m_coords->coords[ip[1]][1], m_coords->coords[ip[1]][2] };
		float r[3] = { coords_b[0] - coords_a[0], coords_b[1] - coords_a[1] ,coords_b[2] - coords_a[2] };
		float r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

		if (r2 < pre->m_cutoff_sqr) {
			float tmp[2];
			p_eval_deriv(tmp, ip[0], r2, pre, epsilon_fl);
			float force[3] = { r[0] * tmp[1], r[1] * tmp[1] ,r[2] * tmp[1] };
			curl(&tmp[0], force, v, epsilon_fl);
			my_e += tmp[0];
			for (int j = 0; j < 3; j++){
				atomicAdd(&minus_forces->coords[ip[1]][j], -force[j]);
				atomicAdd(&minus_forces->coords[ip[2]][j], +force[j]);
			}
		}

	}

	// [v11 fix] block-level energy reduction: in-warp shfl + cross-warp shared tree
	// (the old single-warp shfl dropped warps 1..3 at blockDim=128)
	{
		__shared__ float warp_e[32];
		const int lane = threadNumInBlock & 31;
		const int wid = threadNumInBlock >> 5;
		for (int offset = 16; offset > 0; offset >>= 1)
			my_e += __shfl_down_sync(0xFFFFFFFF, my_e, offset);
		if (lane == 0) warp_e[wid] = my_e;
		__syncthreads();
		if (wid == 0) {
			float v = (lane < (threadsPerBlock >> 5)) ? warp_e[lane] : 0.0f;
			for (int offset = 16; offset > 0; offset >>= 1)
				v += __shfl_down_sync(0xFFFFFFFF, v, offset);
			if (lane == 0) warp_e[0] = v;
		}
		__syncthreads();
		my_e = warp_e[0];
	}
	return my_e;
}

__device__ inline void product(flo* res, const flo*a,const flo*b) {
    flo a0 = a[0];
    flo a1 = a[1];
    flo a2 = a[2];

    flo b0 = b[0];
    flo b1 = b[1];
    flo b2 = b[2];

    res[0] = a1 * b2 - a2 * b1;
    res[1] = a2 * b0 - a0 * b2;
    res[2] = a0 * b1 - a1 * b0;
}

__device__  void POT_deriv(	const					m_minus_forces* minus_forces,
				const					rigid_cl*		lig_rigid_gpu,
				const					m_coords_cl*		m_coords,
										change_cl*		g
) {
	int num_torsion = lig_rigid_gpu->num_children;
	int num_rigid = num_torsion + 1;
	flo position_derivative_tmp[MAX_NUM_OF_RIGID][3];
	flo position_derivative[MAX_NUM_OF_RIGID][3];
	flo orientation_derivative_tmp[MAX_NUM_OF_RIGID][3];
	flo orientation_derivative[MAX_NUM_OF_RIGID][3];
	flo torsion_derivative[MAX_NUM_OF_RIGID]; // torsion_derivative[0] has no meaning(root node has no torsion)

	for (int i = 0; i < num_rigid; i++) {
		int begin = lig_rigid_gpu->atom_range[i][0];
		int end = lig_rigid_gpu->atom_range[i][1];
		for (int k = 0; k < 3; k++){
			position_derivative_tmp[i][k] = 0;
			orientation_derivative_tmp[i][k] = 0; 
		}
		for (int j = begin; j < end; j++) {
			for (int k = 0; k < 3; k++)position_derivative_tmp[i][k] += minus_forces->coords[j][k];

			flo tmp1[3] = {	m_coords->coords[j][0] - lig_rigid_gpu->origin[i][0],
								m_coords->coords[j][1] - lig_rigid_gpu->origin[i][1],
								m_coords->coords[j][2] - lig_rigid_gpu->origin[i][2] };
			flo tmp2[3] = {  minus_forces->coords[j][0],
								minus_forces->coords[j][1],
								minus_forces->coords[j][2] };
			flo tmp3[3];
			//product(tmp3, tmp1, tmp2);
			tmp3[0] = tmp1[1] * tmp2[2] - tmp1[2] * tmp2[1];
			tmp3[1] = tmp1[2] * tmp2[0] - tmp1[0] * tmp2[2];
			tmp3[2] = tmp1[0] * tmp2[1] - tmp1[1] * tmp2[0];
			for (int k = 0; k < 3; k++)orientation_derivative_tmp[i][k] += tmp3[k];
		}
	}

	// position_derivative 
	for (int i = num_rigid - 1; i >= 0; i--) {// from bottom to top
		for (int k = 0; k < 3; k++)position_derivative[i][k] = position_derivative_tmp[i][k]; // self
		// looking for chidren node
		for (int j = 0; j < num_rigid; j++) {
			if (lig_rigid_gpu->children_map[i] & (1ULL << j)) {
				for (int k = 0; k < 3; k++)position_derivative[i][k] += position_derivative[j][k]; // self+children node
			}
		}
	}

	// orientation_derivetive
	for (int i = num_rigid - 1; i >= 0; i--) { // from bottom to top
		for (int k = 0; k < 3; k++) orientation_derivative[i][k] = orientation_derivative_tmp[i][k]; // self
		// looking for chidren node
		for (int j = 0; j < num_rigid; j++) {
			if (lig_rigid_gpu->children_map[i] & (1ULL << j)) { // self + children node + product
				for (int k = 0; k < 3; k++) orientation_derivative[i][k] += orientation_derivative[j][k];
				flo product_out[3];
				flo origin_temp[3] = {	lig_rigid_gpu->origin[j][0] - lig_rigid_gpu->origin[i][0],
											lig_rigid_gpu->origin[j][1] - lig_rigid_gpu->origin[i][1],
											lig_rigid_gpu->origin[j][2] - lig_rigid_gpu->origin[i][2] };
				product(product_out, origin_temp, position_derivative[j]);
				for (int k = 0; k < 3; k++)orientation_derivative[i][k] += product_out[k];
			}
		}
	}

	// torsion_derivative
	for (int i = num_rigid - 1; i >= 0; i--) { // from bottom to top
		flo sum = 0;
		for (int j = 0; j < 3; j++) sum += orientation_derivative[i][j] * lig_rigid_gpu->axis[i][j];
		torsion_derivative[i] = sum;
	}

	for (int k = 0; k < 3; k++)	g->position[k] = position_derivative[0][k];
	for (int k = 0; k < 3; k++) g->orientation[k] = orientation_derivative[0][k];
	for (int k = 0; k < num_torsion; k++) g->lig_torsion[k] = torsion_derivative[k + 1];// no meaning for node 0
}

__device__  void POT_deriv_update(	
				const					m_minus_forces* 	minus_forces,
				const					rigid_cl*			lig_rigid_gpu,
				const					m_coords_cl*		m_coords,
										change_cl*			g,
				const		            int				    threadNumInBlock, 
				const					int				    threadsPerBlock						
) {
	int num_torsion = lig_rigid_gpu->num_children;
	int num_rigid = num_torsion + 1;
	__shared__ flo position_derivative_tmp[MAX_NUM_OF_RIGID][3];
	flo position_derivative[MAX_NUM_OF_RIGID][3];
	__shared__ flo orientation_derivative_tmp[MAX_NUM_OF_RIGID][3];
	flo orientation_derivative[MAX_NUM_OF_RIGID][3];
	flo torsion_derivative[MAX_NUM_OF_RIGID]; // torsion_derivative[0] has no meaning(root node has no torsion)

	for (int i = threadNumInBlock; i < num_rigid; i = i + threadsPerBlock) {
		int begin = lig_rigid_gpu->atom_range[i][0];
		int end = lig_rigid_gpu->atom_range[i][1];
		for (int k = 0; k < 3; k++)
		{
			position_derivative_tmp[i][k] = 0; 
			orientation_derivative_tmp[i][k] = 0;
		}
		for (int j = begin; j < end; j++) {
			for (int k = 0; k < 3; k++)position_derivative_tmp[i][k] += minus_forces->coords[j][k];

			flo tmp1[3] = {	m_coords->coords[j][0] - lig_rigid_gpu->origin[i][0],
								m_coords->coords[j][1] - lig_rigid_gpu->origin[i][1],
								m_coords->coords[j][2] - lig_rigid_gpu->origin[i][2] };
			flo tmp2[3] = {  minus_forces->coords[j][0],
								minus_forces->coords[j][1],
								minus_forces->coords[j][2] };
			flo tmp3[3];
			product(tmp3, tmp1, tmp2);
			for (int k = 0; k < 3; k++)orientation_derivative_tmp[i][k] += tmp3[k];
		}
	}

	// [v11] phase barrier: with blockDim>32 the accumulate loop above runs across 4 warps;
	// the tree walk below reads position_derivative_tmp/orientation_derivative_tmp written by
	// other warps (parent nodes). v4 was implicitly safe only at blockDim=32 (single warp lockstep).
	__syncthreads();

	// position_derivative + orientation_derivetive + torsion_derivative
	for (int i = num_rigid - 1; i >= 0; i--) {// from bottom to top
		for (int k = 0; k < 3; k++)
		{
			position_derivative[i][k] = position_derivative_tmp[i][k]; // self
			orientation_derivative[i][k] = orientation_derivative_tmp[i][k];
		}

		// looking for chidren node
		for (int j = 0; j < num_rigid; j++) {
			if (lig_rigid_gpu->children_map[i] & (1ULL << j)) {
				for (int k = 0; k < 3; k++)
				{
					position_derivative[i][k] += position_derivative[j][k]; 
					orientation_derivative[i][k] += orientation_derivative[j][k];
				}
				flo product_out[3];
				flo origin_temp[3] = {	lig_rigid_gpu->origin[j][0] - lig_rigid_gpu->origin[i][0],
											lig_rigid_gpu->origin[j][1] - lig_rigid_gpu->origin[i][1],
											lig_rigid_gpu->origin[j][2] - lig_rigid_gpu->origin[i][2] };
				product(product_out, origin_temp, position_derivative[j]);
				for (int k = 0; k < 3; k++)orientation_derivative[i][k] += product_out[k];
				 // self+children node
			}
		}

		// [v11 note] no second barrier needed: position_derivative/orientation_derivative are
		// per-thread local arrays (every thread walks the tree redundantly); the only
		// cross-thread data is *_tmp (shared), covered by the barrier above.

		flo sum = 0;
		for (int j = 0; j < 3; j++) sum += orientation_derivative[i][j] * lig_rigid_gpu->axis[i][j];
		torsion_derivative[i] = sum;

	}

	for (int k = 0; k < 3; k++)	
	{
		g->position[k] = position_derivative[0][k];
		g->orientation[k] = orientation_derivative[0][k];
	}
	for (int k = 0; k < num_torsion; k++) g->lig_torsion[k] = torsion_derivative[k + 1];// no meaning for node 0
}



__device__ flo m_eval_deriv(			output_type_cl*		c,
										change_cl*			g,
										m_cl*				m,
										pre_cl*				pre,  //delete the constant 
										grids_cl*			grids, //delete the constant 
					const				flo*				v,     // delete the __global
					const				flo				epsilon_fl
) {
	set(c, &m->ligand.rigid, &m->m_coords, m->atoms, m->m_num_movable_atoms, epsilon_fl);

	flo e = ig_eval_deriv(	c,
							g, 
							v[1],				
							grids,
							m,
							epsilon_fl							
							);
	
	e += eval_interacting_pairs_deriv(	pre,
										v[0],
										&m->ligand.pairs,
										&m->m_coords,
										&m->minus_forces,
										epsilon_fl
									);

	POT_deriv(&m->minus_forces, &m->ligand.rigid, &m->m_coords, g);

	return e;
}

__device__ flo m_eval_deriv_update(		output_type_cl*		c,
										change_cl*			g,
										m_cl*				m,
										pre_cl*				pre,  //delete the constant 
										grids_cl*		grids,    //delete the constant 
					const				flo*				v,     // delete the __global
					const				flo					epsilon_fl,
										m_coords_cl*		m_coords,
										m_minus_forces* 	minus_forces,
					const				int					threadNumInBlock,
					const				int					threadsPerBlock,
									rigid_cl*       	m_gpu_rigid,
									grids_gpu*          grids_cuda

) {

	set_update(c, m_gpu_rigid, m_coords, m->atoms, m->m_num_movable_atoms, epsilon_fl);
	if (threadIdx.x==0 && threadIdx.y==0) atomicAdd(&d_stage.evals_n, 1ULL);

	STAGE_T0;
	flo e = ig_eval_deriv_update_optimal(c,
							g, 
							v[1],				
							grids,
							m,
							epsilon_fl,
							m_coords,
							minus_forces,
							threadNumInBlock,
							threadsPerBlock,							
							grids_cuda);
	STAGE_ADD(grid);
							
	
	unsigned long long st_t0b = st_clock_ns();
	e += eval_interacting_pairs_deriv_update_optimal(	
										pre,
										v[0],
										&m->ligand.pairs,
										m_coords,
										minus_forces,
										epsilon_fl,
										threadNumInBlock,
										threadsPerBlock
									);
	STAGE_ADD2(st_t0b, intra);

	unsigned long long st_t0c = st_clock_ns();
	POT_deriv_update(minus_forces, m_gpu_rigid, m_coords, g, threadNumInBlock, threadsPerBlock);
	STAGE_ADD2(st_t0c, pot);

	return e;
}


// Only support one ligand, no flex !
__device__ inline flo find_change_index_read(const change_cl* g, int index, int lig_torsion_size) {
	
	if (index < 3)return g->position[index];
	index -= 3;
	if (index < 3)return g->orientation[index];
	index -= 3;
	if (index < lig_torsion_size)return g->lig_torsion[index]; 
	printf("\nKernel2:find_change_index_read() ERROR!"); // Shouldn't be here
	return -1;
}

__device__ inline flo find_change_index_read_update(const change_cl* g, int index, int lig_torsion_size) {
    int offset = index / 3;  // Integer division by 3
    int remainder = index % 3; // Modulo operation by 3
	int tmp = index - 6;

    switch (offset) {
        case 0:
            // Access position array
            return g->position[remainder];
        case 1:
            // Access orientation array
            return g->orientation[remainder];
        case 2:
		default:
            // Access lig_torsion array with boundary check
            if (tmp < lig_torsion_size) {
                return g->lig_torsion[tmp];
            } else {
                // Handle out-of-bounds access (optional)
                printf("\nKernel2:find_change_index_read() ERROR: Index out of bounds!");
                return -1;  // or return a default value
            }
    }
}


__device__ inline void find_change_index_write(change_cl* g, int index, flo data, int lig_torsion_size) {
	if (index < 3) { g->position[index] = data; return; }
	index -= 3;
	if (index < 3) { g->orientation[index] = data; return; }
	index -= 3;
	if (index < lig_torsion_size) { g->lig_torsion[index] = data; return; } 
	printf("\nKernel2:find_change_index_write() ERROR!"); // Shouldn't be here
}

__device__ inline void find_change_index_write_update(change_cl* g, int index, flo data, int lig_torsion_size) {
	int offset = index / 3;  // Integer division by 3
    int remainder = index % 3; // Modulo operation by 3
	int tmp = index - 6;

    switch (offset) {
        case 0:
            // Access position array
            g->position[remainder] = data; return;
        case 1:
            // Access orientation array
            g->orientation[remainder] = data; return;
        case 2:
		default:
            // Access lig_torsion array with boundary check
            if (tmp < lig_torsion_size) {
                g->lig_torsion[tmp] = data; return;
            } else {
                // Handle out-of-bounds access (optional)
                printf("\nKernel2:find_change_index_write_update() ERROR: Index out of bounds!");
                return ;  // or return a default value
            }
    }
}

__device__ inline void minus_mat_vec_product(	const		matrix_gpu*		h,
							const		change_cl*	in,
										change_cl*  out,
							const		int			lig_torsion_size
) {
	int n = h->dim;
	for (int i = 0; i < n; i++) {
		flo sum = 0;
		for (int j = 0; j < n; j++) {
			sum += h->data[index_permissive(h, i, j)] * find_change_index_read(in, j, lig_torsion_size);
		}
		find_change_index_write(out, i, -sum, lig_torsion_size);
	}
}


__device__ inline flo scalar_product(	const	change_cl*	a,
								const	change_cl*	b,
								const	int			n,
								const	int			lig_torsion_size
) {
	flo tmp = 0;
	for (int i = 0; i < n; i++) {
		tmp += find_change_index_read(a, i, lig_torsion_size) * find_change_index_read(b, i, lig_torsion_size);
	}
	return tmp;
}

__device__ inline flo scalar_product_update(	const	change_cl*	a,
								const	change_cl*	b,
								const	int			n,
								const	int			lig_torsion_size,
								const	int			threadNumInBlock,
								const   int			threadsPerBlock

) {
	if( n > MAX_NUM_OF_DIMENSIONS_OF_MATIRX) printf("\nKernel2: scalar_product() ERROR!"); 
	__shared__ flo tmp[MAX_NUM_OF_DIMENSIONS_OF_MATIRX];
	for (int i = threadNumInBlock; i < MAX_NUM_OF_DIMENSIONS_OF_MATIRX; i = i + threadsPerBlock)
	{ 
		tmp[i] = 0.0;
	}

	for (int i = threadNumInBlock; i < n; i = i + threadsPerBlock) {
		tmp[i]= find_change_index_read_update(a, i, lig_torsion_size) * find_change_index_read_update(b, i, lig_torsion_size);
	}

	if (n > threadsPerBlock) {
		int tmp_1 = n / threadsPerBlock;
		for (int i = 1; i < tmp_1; i++) {
			tmp[threadNumInBlock] += tmp[threadNumInBlock + threadsPerBlock * i];
		}
		if (threadNumInBlock < ( n % threadsPerBlock)) {
			tmp[threadNumInBlock] += tmp[threadNumInBlock + tmp_1 * threadsPerBlock];
		}
		__syncthreads();
	}
	
	for (int off = (n) >> 1; off > 0; off >>= 1)
		{
			if (threadNumInBlock < off)
			{
				tmp[threadNumInBlock] += tmp[threadNumInBlock + off];
			}
			__syncthreads();
		}

	return tmp[0];
}

__device__ inline void get_to_minus(change_cl* a, change_cl* b, const int n)
{
	for (int i = 0; i < 3; i++)a->position[i] = -b->position[i];
	for (int i = 0; i < 3; i++)a->orientation[i] = -b->orientation[i];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)a->lig_torsion[i] = -b->lig_torsion[i];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)a->flex_torsion[i] = -b->flex_torsion[i];
}

__device__ inline void get_to_minus_update(change_cl* a, change_cl* b, const int threadNumInBlock, const int threadsPerBlock)
{
	
		a->position[0] = -b->position[0];
		a->position[1] = -b->position[1];
		a->position[2] = -b->position[2];

		a->orientation[0] = -b->orientation[0];
		a->orientation[1] = -b->orientation[1];
		a->orientation[2] = -b->orientation[2];
	
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i += 2) {
		a->lig_torsion[i] = -b->lig_torsion[i];
		a->lig_torsion[i+1] = -b->lig_torsion[i+1];
		a->flex_torsion[i] = -b->flex_torsion[i];
		a->lig_torsion[i+1] = -b->lig_torsion[i+1];
	}
	
}
__device__ inline flo to_norm(const change_cl* in, const int n, const int torsion_size)
{
	flo d_test = 0;
	for (int i = 0; i < n; i++)
	{
		d_test += find_change_index_read_update(in, i, torsion_size) * find_change_index_read_update(in, i, torsion_size);
	}
	d_test = sqrt(d_test);
	return d_test;
}

__device__ inline float to_norm_update(const change_cl* in, const int n, const int torsion_size, const int threadNumInBlock, const int threadsPerBlock)
{
	__shared__ float d_test[MAX_NUM_OF_DIMENSIONS_OF_MATIRX];

	for (int i = threadNumInBlock; i < MAX_NUM_OF_DIMENSIONS_OF_MATIRX; i = i + threadsPerBlock)
	{ 
		d_test[i] = 0.0;
	}

	for (int i = threadNumInBlock; i < n; i = i + threadsPerBlock) 
	{
			d_test[i]= find_change_index_read(in, i, torsion_size) * find_change_index_read(in, i, torsion_size);
	}
	__syncthreads();

	flo temp = 0;
	for(int i = 0; i<n; i++)
	
		temp += d_test[i];

	return sqrt(temp);
}


// =====================================================================================
// [v11] TILE-PARALLEL LINE SEARCH (XHB Vina-UP mode, quasi_newton_gpu.cu L255-337)
// blockDim.x = 128 = 16 tiles of 8 (cg::tiled_partition<8>).
// v1: 8 trial steps (alpha_t = stp * 0.5^t, t = 0..7) evaluated CONCURRENTLY, one per
// tile 0..7; tiles 8..15 idle (assist/dyn-scheduling deferred to v2).
// Serial fallback (V11_LS_GRID=0 or movable atoms > V11_LS_MAX_ATOMS): original
// line_search_lewisoverton_update — bit-identical to v4.
//
// Numerical notes (vs v4):
// - E/forces are computed by the ORIGINAL per-thread-serial set_update / ig / pairs /
//   POT routines, run redundantly by tile-lane 0. We deliberately do NOT split the
//   loops across lanes: POT tree traversal is serial by construction, and ig/pairs
//   block-level reductions rely on whole-block __syncthreads(). Only the SEARCH is
//   parallel; one E evaluation costs the same as in v4 (stage GRID/INTRA/POT stay
//   comparable) — the win is 8 E's instead of ~1-8 serial E's per line search.
// - pairs forces accumulate via shared-memory atomicAdd (order-nondeterministic,
//   ~1e-7 noise; CASF-double-run gates this).
// - all per-trial state (candidate conf, m_coords, minus_forces, rigid dynamic
//   fields) lives in LOCAL MEMORY of lane 0 (~4.4KB/lane); no block-level shared
//   arrays are touched during trials -> no cross-tile sync, no SMEM growth.
// - Lewis-Overton bookkeeping is per-tile private (mu/nu/brackt/count); a trial
//   publishes (f1, absolute accepted step) only when it passes the curvature test
//   (or the count-8 guard with f1 <= finit). Winner = min f1, ties -> lower trial
//   (larger step), matching v4's first-accepted ordering.
// =====================================================================================

__device__ inline bool v11_ls_enabled(int nat_atoms) {
	return (V11_LS_GRID > 0) && (nat_atoms <= V11_LS_MAX_ATOMS);
}

#if V11_LS_GRID > 0

// exact 0.5^t via repeated multiplication (powf is not exact for powers of two)
__device__ inline flo v11_half_pow(int t) {
	flo h = 1.0f;
	for (int i = 0; i < t; i++) h *= 0.5f;
	return h;
}

// [v11] ig (grid) evaluation, SINGLE-THREAD serial port of ig_eval_deriv_update_optimal.
// The block version's __syncthreads + static __shared__ reduction is unusable here:
// only 8 lane-0 threads (one per trial tile) execute concurrently while the rest of the
// block waits at the tile-LS barrier -> mismatched barriers would deadlock, and a static
// shared accumulator would collide across the 8 concurrent callers. A single thread needs
// no reduction at all; math is otherwise identical (g_evaluate_update + grid_other).
__device__ inline flo v11_ig_eval_serial(
	output_type_cl* x, change_cl* g, const flo v, grids_cl* grids, m_cl* m,
	const flo epsilon_fl, m_coords_cl* m_coords, m_minus_forces* minus_forces,
	grids_gpu* grids_cuda
) {
	int nat = num_atom_types(grids->atu);
	flo e = 0;
	for (int i = 0; i < m->m_num_movable_atoms; i++) {
		int t = m->atoms[i].types[grids->atu];
		if (t >= nat) {
			minus_forces->coords[i][0] = 0;
			minus_forces->coords[i][1] = 0;
			minus_forces->coords[i][2] = 0;
		} else {
			flo deriv[3];
			e += g_evaluate_update(&grids->grids[t], m_coords->coords[i], grids->slope, v,
					deriv, epsilon_fl, &grids_cuda->grid_other[t]);
			minus_forces->coords[i][0] = deriv[0];
			minus_forces->coords[i][1] = deriv[1];
			minus_forces->coords[i][2] = deriv[2];
		}
	}
	return e;
}

// per-tile E evaluation: 8-lane parallel (v2).
//   set_update / POT_deriv : lane 0 serial (single-thread code in v4 too, no relative loss;
//                            per-tile rigid dynamic fields + POT tmp arrays live in the
//                            tile's dynamic-smem slot -> no cross-tile collision)
//   ig / pairs             : lane-strided with 8-lane shfl reductions (the mass of the E work;
//                            8x the lanes of one v4 "thread" on the hot loops)
// The tile's slot pointer (flo*) layout: [coords 384 | forces 384 | rs_dyn 912 | pot_pos 144
//                                        | pot_ori 144 | pot_red 40 | spare 16] flo
#define V11_TILE_SLOT_FLO (384 + 384 + 912 + 144 + 144 + 40 + 16)
#define V11_TILE_SMEM_FLO (V11_LS_GRID * V11_TILE_SLOT_FLO)   // dynamic smem, opt-in at launch

// [v11 v2] reinterpret a tile's flo* slot as its per-trial state structures
__device__ inline void v11_slot_split(flo* slot,
		m_coords_cl** coords, m_minus_forces** forces, flo** rs_dyn,
		flo** pot_pos, flo** pot_ori, flo** pot_red) {
	flo* p = slot;
	*coords   = (m_coords_cl*)(p);        p += 384;
	*forces   = (m_minus_forces*)(p);     p += 384;
	*rs_dyn   = p;                        p += 912;
	*pot_pos  = p;                        p += 144;
	*pot_ori  = p;                        p += 144;
	*pot_red  = p;
}

// ---- v2 tile-parallel ig (grid) evaluation: lane-strided + 8-lane shfl reduction ----
__device__ inline flo v11_ig_eval_tile(
	output_type_cl* x, change_cl* g, const flo v, grids_cl* grids, m_cl* m,
	const flo epsilon_fl, m_coords_cl* m_coords, m_minus_forces* minus_forces,
	grids_gpu* grids_cuda, int lane
) {
	const int nat = num_atom_types(grids->atu);
	const int nt = m->m_num_movable_atoms;
	flo my_e = 0;
	for (int i = lane; i < nt; i += 8) {
		int t = m->atoms[i].types[grids->atu];
		if (t >= nat) {
			minus_forces->coords[i][0] = 0;
			minus_forces->coords[i][1] = 0;
			minus_forces->coords[i][2] = 0;
		} else {
			flo deriv[3];
			my_e += g_evaluate_update(&grids->grids[t], m_coords->coords[i], grids->slope, v,
					deriv, epsilon_fl, &grids_cuda->grid_other[t]);
			minus_forces->coords[i][0] = deriv[0];
			minus_forces->coords[i][1] = deriv[1];
			minus_forces->coords[i][2] = deriv[2];
		}
	}
	// 8-lane butterfly reduction + broadcast (no shared, no barriers beyond the warp)
	// [v11] mask is WARP-relative: this tile's octet = lanes (warp_lane & ~7)..+7.
	// (0xFFu << (lane & ~7) was wrong for tiles 1..7 of warp 0 and UB for threads >= 32)
	const unsigned oct_mask = 0xFFu << ((threadIdx.x & 31) & ~7);
	my_e += __shfl_xor_sync(oct_mask, my_e, 4, 8);
	my_e += __shfl_xor_sync(oct_mask, my_e, 2, 8);
	my_e += __shfl_xor_sync(oct_mask, my_e, 1, 8);
	return my_e;
}

// ---- v2 tile-parallel pairs evaluation: lane-strided + atomic force accumulation ----
__device__ inline flo v11_pairs_eval_tile(
	pre_cl* pre, const flo v, lig_pairs_cl* pairs, m_coords_cl* m_coords,
	m_minus_forces* minus_forces, const flo epsilon_fl, int lane
) {
	const int np = pairs->num_pairs;
	flo my_e = 0;
	for (int i = lane; i < np; i += 8) {
		const int ip[3] = { pairs->type_pair_index[i], pairs->a[i], pairs->b[i] };
		float coords_b[3] = { m_coords->coords[ip[2]][0], m_coords->coords[ip[2]][1], m_coords->coords[ip[2]][2] };
		float coords_a[3] = { m_coords->coords[ip[1]][0], m_coords->coords[ip[1]][1], m_coords->coords[ip[1]][2] };
		float r[3] = { coords_b[0] - coords_a[0], coords_b[1] - coords_a[1], coords_b[2] - coords_a[2] };
		float r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
		if (r2 < pre->m_cutoff_sqr) {
			float tmp[2];
			p_eval_deriv(tmp, ip[0], r2, pre, epsilon_fl);
			float force[3] = { r[0] * tmp[1], r[1] * tmp[1], r[2] * tmp[1] };
			curl(&tmp[0], force, v, epsilon_fl);
			my_e += tmp[0];
			for (int j = 0; j < 3; j++){
				minus_forces->coords[ip[1]][j] -= force[j];
				minus_forces->coords[ip[2]][j] += force[j];
			}
		}
	}
	const unsigned oct_mask = 0xFFu << ((threadIdx.x & 31) & ~7);   // warp-relative octet
	my_e += __shfl_xor_sync(oct_mask, my_e, 4, 8);
	my_e += __shfl_xor_sync(oct_mask, my_e, 2, 8);
	my_e += __shfl_xor_sync(oct_mask, my_e, 1, 8);
	return my_e;
}

// ---- v2 POT on per-tile buffers: lane0 serial over per-tile tmp arrays ----
// (identical math to POT_deriv; the only change is that the __shared__ tmp arrays
//  become the tile's slot-resident per-tile arrays)
__device__ void v11_POT_deriv_tile(
	const m_minus_forces* minus_forces, v11_rigid_ref* lig_rigid_gpu, const m_coords_cl* m_coords,
	change_cl* g, flo* pot_pos, flo* pot_ori, flo* pot_red
) {
	flo (*position_derivative_tmp)[3] = (flo(*)[3])pot_pos;   // [48][3]
	flo (*orientation_derivative_tmp)[3] = (flo(*)[3])pot_ori; // [48][3]
	flo* torsion_derivative = pot_red;                          // [48]

	int num_torsion = lig_rigid_gpu->num_children;
	int num_rigid = num_torsion + 1;
	flo position_derivative[MAX_NUM_OF_RIGID][3];
	flo orientation_derivative[MAX_NUM_OF_RIGID][3];

	for (int i = 0; i < num_rigid; i++) {
		int begin = lig_rigid_gpu->atom_range[i][0];
		int end = lig_rigid_gpu->atom_range[i][1];
		for (int k = 0; k < 3; k++){
			position_derivative_tmp[i][k] = 0;
			orientation_derivative_tmp[i][k] = 0;
		}
		for (int j = begin; j < end; j++) {
			for (int k = 0; k < 3; k++)position_derivative_tmp[i][k] += minus_forces->coords[j][k];

			flo tmp1[3] = {	m_coords->coords[j][0] - lig_rigid_gpu->origin[i][0],
							m_coords->coords[j][1] - lig_rigid_gpu->origin[i][1],
							m_coords->coords[j][2] - lig_rigid_gpu->origin[i][2] };
			flo tmp2[3] = {  minus_forces->coords[j][0],
							minus_forces->coords[j][1],
							minus_forces->coords[j][2] };
			flo tmp3[3];
			tmp3[0] = tmp1[1] * tmp2[2] - tmp1[2] * tmp2[1];
			tmp3[1] = tmp1[2] * tmp2[0] - tmp1[0] * tmp2[2];
			tmp3[2] = tmp1[0] * tmp2[1] - tmp1[1] * tmp2[0];
			for (int k = 0; k < 3; k++)orientation_derivative_tmp[i][k] += tmp3[k];
		}
	}

	for (int i = num_rigid - 1; i >= 0; i--) {// from bottom to top
		for (int k = 0; k < 3; k++)position_derivative[i][k] = position_derivative_tmp[i][k]; // self
		// looking for chidren node
		for (int j = 0; j < num_rigid; j++) {
			if (lig_rigid_gpu->children_map[i] & (1ULL << j)) {
				for (int k = 0; k < 3; k++)position_derivative[i][k] += position_derivative[j][k]; // self+children node
			}
		}
	}

	// orientation_derivetive
	for (int i = num_rigid - 1; i >= 0; i--) { // from bottom to top
		for (int k = 0; k < 3; k++) orientation_derivative[i][k] = orientation_derivative_tmp[i][k]; // self
		// looking for chidren node
		for (int j = 0; j < num_rigid; j++) {
			if (lig_rigid_gpu->children_map[i] & (1ULL << j)) { // self + children node + product
				for (int k = 0; k < 3; k++) orientation_derivative[i][k] += orientation_derivative[j][k];
				flo product_out[3];
				flo origin_temp[3] = {	lig_rigid_gpu->origin[j][0] - lig_rigid_gpu->origin[i][0],
										lig_rigid_gpu->origin[j][1] - lig_rigid_gpu->origin[i][1],
										lig_rigid_gpu->origin[j][2] - lig_rigid_gpu->origin[i][2] };
				product(product_out, origin_temp, position_derivative[j]);
				for (int k = 0; k < 3; k++)orientation_derivative[i][k] += product_out[k];
			}
		}
	}

	// torsion_derivative
	for (int i = num_rigid - 1; i >= 0; i--) { // from bottom to top
		flo sum = 0;
		for (int j = 0; j < 3; j++) sum += orientation_derivative[i][j] * lig_rigid_gpu->axis[i][j];
		torsion_derivative[i] = sum;
	}

	for (int k = 0; k < 3; k++)	g->position[k] = position_derivative[0][k];
	for (int k = 0; k < 3; k++) g->orientation[k] = orientation_derivative[0][k];
	for (int k = 0; k < num_torsion; k++) g->lig_torsion[k] = torsion_derivative[k + 1];// no meaning for node 0
}

// per-tile E evaluation (v2): 8-lane parallel ig/pairs; lane0-serial set/POT on the
// tile's private slot. slot = V11_TILE_SLOT_FLO flo in DYNAMIC shared memory.
// Intra-octet sync via __syncwarp(mask) (tiles are 8 consecutive threads inside a warp):
//   S1: lane0's set_update writes coords -> barrier -> all lanes read coords in ig/pairs
//   S2: all lanes' atomicAdd forces done -> barrier -> lane0's POT reads forces
__device__ void v11_set_tile(
	const output_type_cl* x, v11_rigid_ref* lig_rigid_gpu, m_coords_cl* m_coords_gpu,
	const atom_cl* atoms, const int m_num_movable_atoms, const flo epsilon_fl
) {
	// per-node set_conf, identical math to set_update; per-trial rigid via ref
	lig_rigid_gpu->origin[0][0] = x->position[0];
	lig_rigid_gpu->origin[0][1] = x->position[1];
	lig_rigid_gpu->origin[0][2] = x->position[2];

	for (int i = 0; i < 4; i+=2) {
		lig_rigid_gpu->orientation_q[0][i] = x->orientation[i];
		lig_rigid_gpu->orientation_q[0][i+1] = x->orientation[i+1];
	}
	quaternion_to_r3(lig_rigid_gpu->orientation_q[0], lig_rigid_gpu->orientation_m[0]);
	{
		int begin = lig_rigid_gpu->atom_range[0][0];
		int end =	lig_rigid_gpu->atom_range[0][1];
		for (int i = begin; i < end; i++) {
			local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[0], &atoms[i].coords[0], lig_rigid_gpu->orientation_m[0]);
		}
	}

	for (int current = 1; current < lig_rigid_gpu->num_children + 1; current++) {
		int parent = lig_rigid_gpu->parent[current];
		flo torsion = x->lig_torsion[current - 1];
		local_to_lab(	lig_rigid_gpu->origin[current],
						lig_rigid_gpu->origin[parent],
						lig_rigid_gpu->relative_origin[current],
						lig_rigid_gpu->orientation_m[parent]
						);
		local_to_lab_direction(	lig_rigid_gpu->axis[current],
								lig_rigid_gpu->relative_axis[current],
								lig_rigid_gpu->orientation_m[parent]
								);
		flo tmp[4];
		flo parent_q[4] = {	lig_rigid_gpu->orientation_q[parent][0],
							lig_rigid_gpu->orientation_q[parent][1] ,
							lig_rigid_gpu->orientation_q[parent][2] ,
							lig_rigid_gpu->orientation_q[parent][3] };
		flo current_axis[3] = {	lig_rigid_gpu->axis[current][0],
								lig_rigid_gpu->axis[current][1],
								lig_rigid_gpu->axis[current][2] };

		angle_to_quaternion2(tmp, current_axis, torsion);
		angle_to_quaternion_multi(tmp, parent_q);
		quaternion_normalize_approx(tmp, epsilon_fl);

		for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[current][i] = tmp[i];

		quaternion_to_r3(lig_rigid_gpu->orientation_q[current], lig_rigid_gpu->orientation_m[current]);

		int begin = lig_rigid_gpu->atom_range[current][0];
		int end =	lig_rigid_gpu->atom_range[current][1];
		for (int i = begin; i < end; i++) {
			local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[current], &atoms[i].coords[0], lig_rigid_gpu->orientation_m[current]);
		}
	}
}

__device__ inline flo v11_m_eval_tile(
	output_type_cl* c, change_cl* g,
	m_cl* m, pre_cl* pre, grids_cl* grids, const flo* v, flo epsilon_fl,
	m_coords_cl* m_coords, m_minus_forces* minus_forces,
	v11_rigid_ref* rs, grids_gpu* grids_cuda, flo* slot, int lane
) {
	const unsigned mask = 0xFFu << ((threadIdx.x & 31) & ~7);   // warp-relative octet (fix UB shift)
	if (lane == 0)
		v11_set_tile(c, rs, m_coords, m->atoms, m->m_num_movable_atoms, epsilon_fl);
	__syncwarp(mask);                                    // S1
	flo e = v11_ig_eval_tile(c, g, v[1], grids, m, epsilon_fl, m_coords, minus_forces, grids_cuda, lane);
	e += v11_pairs_eval_tile(pre, v[0], &m->ligand.pairs, m_coords, minus_forces, epsilon_fl, lane);
	__syncwarp(mask);                                    // S2
	if (lane == 0) {
		m_coords_cl* tc; m_minus_forces* tf; flo* rd, *pp, *po, *pr;
		v11_slot_split(slot, &tc, &tf, &rd, &pp, &po, &pr);
		v11_POT_deriv_tile(minus_forces, rs, m_coords, g, pp, po, pr);
	}
	__syncwarp(mask);                                    // broadcast-barrier (g is per-slot anyway)
	return e;
}

// [v11] tile-parallel Lewis-Overton line search. Same (dginit/dgtest/dstest, mu/nu,
// MAX_LINESEARCH=8) semantics as line_search_lewisoverton_update; on success *f and
// *stp are the winner's (energy, absolute accepted step) and 1 is returned;
// -1 = no trial passed (block-level x/g/m_coords/minus_forces/rigid untouched).
__device__ int line_search_lewisoverton_tile(
				m_cl* m_cl_gpu, pre_cl* p_cl_gpu, grids_cl* ig_cl_gpu,
				int n, flo* stp, output_type_cl* x, flo* f, change_cl* g,
				const change_cl* d, const output_type_cl* xp, const change_cl* gp,
				const flo epsilon_fl, const flo* hunt_cap, const int torsion_size,
				m_coords_cl* m_coords, m_minus_forces* minus_forces,
				const int threadNumInBlock, const int threadsPerBlock,
				rigid_cl* m_gpu_rigid, grids_gpu* grids_cuda
)
{
	cg::thread_block bg = cg::this_thread_block();
	cg::thread_block_tile<8> tile = cg::tiled_partition<8>(bg);
	const int lane = tile.thread_rank();
	const int tno = tile.meta_group_rank();
	const int tr = tno;   // trial index == tile index, 0..7

	// ---- phase 0: block-wide scalars (serial, identical to v4) ----
	__shared__ flo s_dginit, s_finit, s_f1, s_stp;
	if (threadIdx.x == 0) {
		flo dg = 0;
		for (int i = 0; i < n; i++)
			dg += find_change_index_read_update(gp, i, torsion_size) * find_change_index_read_update(d, i, torsion_size);
		s_dginit = dg; s_finit = *f;
	}
	__syncthreads();
	const flo dginit = s_dginit, finit = s_finit;
	if (0.0 < dginit) return -1;
	const flo dgtest = 1.0e-4 * dginit;   // F_DEC_COEFF
	const flo dstest = 0.1 * dginit;      // S_CURV_COEFF

	// ---- per-trial state: DYNAMIC SHARED MEMORY slots, one per tile (v2) ----
	// extern __shared__ flo v11_slots[];  (declared at top of quasi_newton.cuh)
	// slot t: [coords 384 | forces 384 | rs_dyn 912 | pot 328 | spare 16] flo = 7.8KB
	__shared__ volatile flo found_f1[V11_LS_GRID];   // INFINITY = not found
	__shared__ volatile flo found_stp[V11_LS_GRID];  // absolute accepted step of that trial
	__shared__ int s_best;

	flo* my_slot = v11_slots + tno * V11_TILE_SLOT_FLO;
	m_coords_cl* xs; m_minus_forces* mf; flo* rd, *pp, *po, *pr;
	v11_slot_split(my_slot, &xs, &mf, &rd, &pp, &po, &pr);   // valid pointers only for tno<8

	// [v11 v2] rs: reference view over the tile's slot.
	// rigid_cl embeds fixed-size arrays, so we build a lightweight ref struct with the
	// same field layout (identical member names) whose array members point into the
	// slot: static/readonly fields (children_map/parent/atom_range/relative_*) alias
	// the MASTER (const, never written during LS); dynamic fields (origin/axis/
	// orientation_*) live in rd. set_update/v11_POT_deriv_tile consume it read-only
	// via the same member syntax.
	// ONLY tiles 0..7 have slots (V11_LS_GRID); tiles 8..15 skip all state work.
	v11_rigid_ref rs;
	if (tno < V11_LS_GRID) {
		rs.num_children = m_gpu_rigid->num_children;
		rs.children_map = m_gpu_rigid->children_map;
		rs.parent       = m_gpu_rigid->parent;
		rs.atom_range   = m_gpu_rigid->atom_range;
		rs.relative_axis   = m_gpu_rigid->relative_axis;
		rs.relative_origin = m_gpu_rigid->relative_origin;
		rs.origin         = (flo(*)[3])rd;
		rs.axis           = (flo(*)[3])(rd + 144);
		rs.orientation_q  = (flo(*)[4])(rd + 288);
		rs.orientation_m  = (flo(*)[9])(rd + 320);
		{
			const int nr = m_gpu_rigid->num_children + 1;
			for (int i = 0; i < nr; i++) {
				for (int k = 0; k < 3; k++) { rs.origin[i][k] = m_gpu_rigid->origin[i][k]; rs.axis[i][k] = m_gpu_rigid->axis[i][k]; }
				for (int k = 0; k < 4; k++) rs.orientation_q[i][k] = m_gpu_rigid->orientation_q[i][k];
				for (int k = 0; k < 9; k++) rs.orientation_m[i][k] = m_gpu_rigid->orientation_m[i][k];
			}
		}
	}
	output_type_cl xc; change_cl gc;

	if (threadIdx.x == 0)
		#pragma unroll
		for (int t = 0; t < V11_LS_GRID; t++) found_f1[t] = INFINITY;
	__syncthreads();

	// ---- trial loop: the WHOLE octet (all 8 lanes of tiles 0..7) enters together ----
	// The 8-lane eval (v11_m_eval_tile) is a collective operation over the octet; the
	// Lewis-Overton bookkeeping is uniform across the octet (floc is butterfly-reduced
	// to an identical value on every lane; dgd is computed by lane0 and broadcast), so
	// every branch below is warp-uniform and __syncwarp inside eval is well-defined.
	const change_cl* dp = d;      // direction is read-only during LS (master's, no copy)
	const unsigned oct_mask = 0xFFu << ((threadIdx.x & 31) & ~7);
	if (tno < V11_LS_GRID) {
		// snapshot master state into my slot — 8-lane parallel (was lane0-serial 6.6KB,
		// ~530k LS calls x 6.6KB = 3.5GB smem-write traffic; now divided by 8)
		for (int i = lane; i < MAX_NUM_OF_ATOMS; i += 8)
			for (int k = 0; k < 3; k++) {
				xs->coords[i][k] = m_coords->coords[i][k];
				mf->coords[i][k] = minus_forces->coords[i][k];
			}
		{
			const int nr = m_gpu_rigid->num_children + 1;
			for (int i = lane; i < nr; i += 8) {
				for (int k = 0; k < 3; k++) { rs.origin[i][k] = m_gpu_rigid->origin[i][k]; rs.axis[i][k] = m_gpu_rigid->axis[i][k]; }
				for (int k = 0; k < 4; k++) rs.orientation_q[i][k] = m_gpu_rigid->orientation_q[i][k];
				for (int k = 0; k < 9; k++) rs.orientation_m[i][k] = m_gpu_rigid->orientation_m[i][k];
			}
		}
		__syncwarp(oct_mask);                      // snapshot complete before trials read slots
		flo floc = *f;                             // uniform across the octet

		flo stp_t = (*stp) * v11_half_pow(tr);     // alpha_t = stp * 0.5^trial
		flo mu = 0.0f, nu = 1.0e+20f;
		int count = 0;
		bool brackt = false;
		bool run = true;

		while (run) {
			xc = *xp;                                   // reset to xp every iteration (v4 semantics)
			if (lane == 0)
				output_type_cl_increment(&xc, dp, stp_t, epsilon_fl, torsion_size);
			floc = v11_m_eval_tile(&xc, &gc, m_cl_gpu, p_cl_gpu, ig_cl_gpu, hunt_cap, epsilon_fl,
					xs, mf, &rs, grids_cuda, my_slot, lane);
			if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(&d_stage.evals_n, 1ULL);
			++count;
			if (floc > finit + stp_t * dgtest) {        // Armijo violated (uniform)
				nu = stp_t; brackt = true;
			} else {
				flo dgd = 0;                            // curvature on tile-local gradient
				if (lane == 0) {
					for (int i = 0; i < n; i++)
						dgd += find_change_index_read_update(&gc, i, torsion_size)
							 * find_change_index_read_update(dp, i, torsion_size);
				}
				dgd = __shfl_sync(oct_mask, dgd, 0, 8);   // broadcast lane0's dgd
				if (dgd < dstest) mu = stp_t;           // descend further (v4: mu update only)
				else { found_f1[tr] = floc; found_stp[tr] = stp_t; run = false; }  // publish
			}
			if (run && count >= 8) {                    // MAX_LINESEARCH (v4 guard)
				if (floc <= finit) { found_f1[tr] = floc; found_stp[tr] = stp_t; }
				run = false;                            // else: trial failed, publishes nothing
			}
			if (run) stp_t = brackt ? (0.5f * (mu + nu)) : (stp_t * 2.0f);   // v4 step rule
		}
	}
	__syncthreads();

	// ---- pick winner (thread 0 serial; ties -> lower trial = larger step) ----
	if (threadIdx.x == 0) {
		int best = -1; flo bf = INFINITY;
		#pragma unroll
		for (int t = 0; t < V11_LS_GRID; t++) {
			flo fv = found_f1[t];
			if (fv < bf) { bf = fv; best = t; }
		}
		s_best = best;
		s_f1 = bf;
		s_stp = (best >= 0) ? found_stp[best] : 0.0f;
	}
	__syncthreads();

	if (s_best < 0) return -1;   // no trial passed; block-level state untouched

	// ---- winner copy-back ----
	// winner's conf/grad live in the winner lane0's registers -> it publishes them into
	// its own slot: xc -> spare area (16 flo), gc -> pot area (dead after its last eval).
	// BLOCK barrier, then the whole block parallel-copies slot -> block-level structures.
	{
		flo* w_slot = v11_slots + s_best * V11_TILE_SLOT_FLO;
		m_coords_cl* wc; m_minus_forces* wf; flo* wrd, *wpp, *wpo, *wpr;
		v11_slot_split(w_slot, &wc, &wf, &wrd, &wpp, &wpo, &wpr);
		if (tno == s_best && lane == 0) {
			*(output_type_cl*)(w_slot + V11_TILE_SLOT_FLO - 16) = xc;   // conf -> spare
			*(change_cl*)wpp = gc;                                      // grad -> pot area
		}
		__syncthreads();   // publish visible block-wide

		// 128-thread parallel copy: coords + forces + rigid dynamic fields
		const int nr = m_gpu_rigid->num_children + 1;
		for (int i = threadNumInBlock; i < MAX_NUM_OF_ATOMS; i += threadsPerBlock)
			for (int k = 0; k < 3; k++) {
				m_coords->coords[i][k] = wc->coords[i][k];
				minus_forces->coords[i][k] = wf->coords[i][k];
			}
		for (int i = threadNumInBlock; i < nr; i += threadsPerBlock) {
			for (int k = 0; k < 3; k++) m_gpu_rigid->origin[i][k] = wrd[i*3 + k];
			for (int k = 0; k < 3; k++) m_gpu_rigid->axis[i][k] = wrd[144 + i*3 + k];
			for (int k = 0; k < 4; k++) m_gpu_rigid->orientation_q[i][k] = wrd[288 + i*4 + k];
			for (int k = 0; k < 9; k++) m_gpu_rigid->orientation_m[i][k] = wrd[320 + i*9 + k];
		}
		if (threadNumInBlock == 0) {
			*x = *(output_type_cl*)(w_slot + V11_TILE_SLOT_FLO - 16);
			*g = *(change_cl*)wpp;
		}
	}
	__syncthreads();
	*f = s_f1;
	*stp = s_stp;
	return 1;
}

#endif // V11_LS_GRID > 0

__device__ int line_search_lewisoverton(					m_cl* 			m_cl_gpu,
	                        				pre_cl* 		p_cl_gpu,  // delete the constant 
											grids_cl* 		ig_cl_gpu, // delete the constant
														int 			n,
														flo* 			stp,
														output_type_cl* x,
														flo* 			f,
														change_cl* 		g,
									const 				change_cl* 		d,
									const 				output_type_cl* xp,
									const 				change_cl* 		gp,
									const 				flo 			epsilon_fl,
									const 			 	flo* 			hunt_cap,  // delete the __global
									const 				int 			torsion_size
	)
{
	int count = 0;
	bool brackt = false;
	flo finit, dginit, dgtest, dstest;
	flo mu = 0.0, nu = 1.0e+20;

	dginit = scalar_product(gp, d, n, torsion_size);

	/* Make sure that s points to a descent direction. */
	if (0.0 < dginit)
	{
		return -1;
	}
	/* The initial value of the cost function. */
	finit = *f;
	//F_DEC_COEFF
	dgtest = 1.0e-4 * dginit;
	// S_CURV_COEFF 
	dstest = 0.1 * dginit;

	while (true)
	{
		//output_type_cl_init_with_output(x, xp);
		*x = *xp;
		output_type_cl_increment(x, d, *stp, epsilon_fl, torsion_size);
		/* Evaluate the function and gradient values. */
		*f = m_eval_deriv(x,
			g,
			m_cl_gpu,
			p_cl_gpu,
			ig_cl_gpu,
			hunt_cap,
			epsilon_fl
		);
		++count;
		/* Check the Armijo condition. */
		if (*f > finit + *stp * dgtest)
		{
			nu = *stp;
			brackt = true;
		}
		else
		{
			if (scalar_product(g, d, n, torsion_size) < dstest)
			{
				mu = *stp;
			}
			else
			{
				return count;
			}
		}
		// MAX_LINESEARCH
		if (8 <= count)
		{
			if (*f > finit)
				return -1;
			else
				return count;
		}
		if (brackt)
		{
			(*stp) = 0.5 * (mu + nu);
		}
		else
		{
			(*stp) *= 2.0;
		}
	}
}

__device__ int line_search_lewisoverton_update(			m_cl* 			m_cl_gpu,
	                        							pre_cl* 		p_cl_gpu,  // delete the constant 
														grids_cl* 		ig_cl_gpu, // delete the constant
														int 			n,
														flo* 			stp,
														output_type_cl* x,
														flo* 			f,
														change_cl* 		g,
									const 				change_cl* 		d,
									const 				output_type_cl* xp,
									const 				change_cl* 		gp,
									const 				flo 			epsilon_fl,
									const 			 	flo* 			hunt_cap,  // delete the __global
									const 				int 			torsion_size,
														m_coords_cl*	m_coords, // shared memory
														m_minus_forces*	minus_forces,// shared memory
									const				int				threadNumInBlock,
									const		        int				threadsPerBlock,
									               rigid_cl*   	m_gpu_rigid,
												   grids_gpu*   grids_cuda
	)
{
	int count = 0;
	bool brackt = false;
	flo finit, dginit, dgtest, dstest;
	flo mu = 0.0, nu = 1.0e+20;

	dginit = scalar_product_update(gp, d, n, torsion_size,threadNumInBlock,threadsPerBlock);

	/* Make sure that s points to a descent direction. */
	if (0.0 < dginit)
	{
		return -1;
	}
	/* The initial value of the cost function. */
	finit = *f;
	//F_DEC_COEFF
	dgtest = 1.0e-4 * dginit;
	// S_CURV_COEFF 
	dstest = 0.1 * dginit;

	while (true)
	{
		//output_type_cl_init_with_output(x, xp);
		*x = *xp;
		output_type_cl_increment(x, d, *stp, epsilon_fl, torsion_size);
		/* Evaluate the function and gradient values. */
		*f = m_eval_deriv_update(x,
			g,
			m_cl_gpu,
			p_cl_gpu,
			ig_cl_gpu,
			hunt_cap,
			epsilon_fl,
			m_coords,
			minus_forces,
			threadNumInBlock,
			threadsPerBlock,
			m_gpu_rigid,
			grids_cuda
		);
		++count;
		/* Check the Armijo condition. */
		if (*f > finit + *stp * dgtest)
		{
			nu = *stp;
			brackt = true;
		}
		else
		{
			if (scalar_product_update(g, d, n, torsion_size,threadNumInBlock,threadsPerBlock) < dstest)
			{
				mu = *stp;
			}
			else
			{
				return count;
			}
		}
		// MAX_LINESEARCH
		if (8 <= count)
		{
			if (*f > finit)
				return -1;
			else
				return count;
		}
		if (brackt)
		{
			(*stp) = 0.5 * (mu + nu);
		}
		else
		{
			(*stp) *= 2.0;
		}
	}
}


__device__  flo line_search(					 	m_cl*				m,
							pre_cl*				pre, // delete the constant
							grids_cl*			grids, // delete the constant
										int					n,
					const				output_type_cl*		x,
					const				change_cl*			g,
					const				flo				f0,
					const				change_cl*			p,
										output_type_cl*		x_new,
										change_cl*			g_new,
										flo*				f1,
					const				flo				epsilon_fl,
					const		flo*				hunt_cap,  // delete the __global
					const				int					lig_torsion_size
) {
	const flo c0 = 0.0001;
	const int max_trials = 10;
	const flo multiplier = 0.5;
	flo alpha = 1;

	const flo pg = scalar_product(p, g, n, lig_torsion_size);

	for (int trial = 0; trial < max_trials; trial++) {

		*x_new =  *x;

		output_type_cl_increment(x_new, p, alpha, epsilon_fl, lig_torsion_size);

		*f1 =  m_eval_deriv(x_new,
							g_new,
							m,
							pre,
							grids,
							hunt_cap,
							epsilon_fl
							);
		// printf("alpha: %f", alpha);


		if (*f1 - f0 < c0 * alpha * pg)
			break;
		alpha *= multiplier;
	}
	// if (*f1 - f0> 0)
	// 	printf("True \n");
	// else
	// 	printf("False \n");
	return alpha;
}


__device__ bool bfgs_update(			matrix_gpu*			h,
					const	change_cl*		p,
					const	change_cl*		y,
					const	flo			alpha,
					const	mis_cl*			mis,
					const	int				torsion_size
) {
	const flo yp = scalar_product(y, p, h->dim, torsion_size);

	if (alpha * yp < mis->epsilon_fl) return false;

	change_cl minus_hy = *y;
	minus_mat_vec_product(h, y, &minus_hy, torsion_size);
	const flo yhy = -scalar_product(y, &minus_hy, h->dim, torsion_size);
	const flo r = 1 / (alpha * yp);
	const int n = 6 + torsion_size;
	int s = torsion_size;
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			flo tmp = alpha * r * (find_change_index_read(&minus_hy, i, s) * find_change_index_read(p, j,s)
									+ find_change_index_read(&minus_hy, j, s) * find_change_index_read(p, i,s)) +
									+alpha * alpha * (r * r * yhy + r) * find_change_index_read(p, i,s) * find_change_index_read(p, j, s);

			h->data[i + j * (j + 1) / 2] += tmp;
		}
	}

	return true;
}

__device__ void rilc_bfgs(				output_type_cl* 	x,
						change_cl* 			g,
						m_cl* 				m_cl_gpu,
						pre_cl* 			p_cl_gpu, // delete the constant
						grids_cl* 			ig_cl_gpu, // delete the constant
	const				mis_cl*				mis,       // delete the global 
	const				int					torsion_size,
	const				int					max_steps
)
{

	int n = 3 + 3 + torsion_size; // the dimensions of matirx

	// rilc_bfgs init
	int k, ls;
	flo step, fx, ys, yy;
	flo beta = 0, cau;
	flo lm_s_dot_d, cau_t;
	//flo d_update_tmp = 0;

	output_type_cl xp = *x;
	change_cl gp = *g;


	flo lm_alpha = 0;
	flo lm_s[MAX_HESSIAN_MATRIX_SIZE] = { 0 };
	flo lm_y[MAX_HESSIAN_MATRIX_SIZE] = { 0 };
	flo lm_ys = 0;
	fx = m_eval_deriv(	x,
						g,
						m_cl_gpu,
						p_cl_gpu,
						ig_cl_gpu,
						mis->hunt_cap,
						mis->epsilon_fl
	);

	flo fxp = fx;
	flo fx_orig = fx;
	change_cl g_orig = *g;
	output_type_cl x_orig = *x;

	change_cl d = *g;

	//d = -g
	get_to_minus(&d, g, n);

	if (!(sqrt(scalar_product(g, g, n, torsion_size)) >= 1e-5))
	{
		x->e = fx;
		return;
	}
	else
	{
		step = 1.0 / to_norm(&d, n, torsion_size);
		k = 1;

		while (true) {

			xp = *x;
			gp = *g;

			fxp = fx;


			ls = line_search_lewisoverton(m_cl_gpu,
				p_cl_gpu,
				ig_cl_gpu,
				n,
				&step,
				x,
				&fx,
				g,
				&d,
				&xp,
				&gp,
				mis->epsilon_fl,
				mis->hunt_cap,
				torsion_size
			);
			if (ls < 0) {

				*x = xp;
				*g = gp;

				fx = fxp;
				break;
			}
			if (!(sqrt(scalar_product(g, g, n, torsion_size)) >= 1e-5)) {
				x->e = fx;
				return;
			}
			if (max_steps != 0 && max_steps <= k) {
				break;
			}
			/* Count the iteration number. */
			++k;
			if (threadIdx.x==0 && threadIdx.y==0) atomicAdd(&d_stage.n_iter, 1ULL);

			for (int i = 0; i < n; i++) {
				lm_s[i] = step * find_change_index_read(&d, i, torsion_size);
			}
			for (int i = 0; i < n; i++) {
				lm_y[i] = find_change_index_read(g, i, torsion_size) - find_change_index_read(&gp, i, torsion_size);
			}

			ys = 0;
			for (int i = 0; i < n; i++) {
				ys += lm_y[i] * lm_s[i];
			}

			yy = 0;
			for (int i = 0; i < n; i++) {
				yy += lm_y[i] * lm_y[i];
			}

			lm_ys = ys;

			/* Compute the negative of gradients. */
			// d = -g
			get_to_minus(&d, g, n);
			cau = 0;cau_t = 0;
			for (int i = 0; i < n; i++) {
				cau += lm_s[i] * lm_s[i];
				cau_t = cau_t + (find_change_index_read(&gp, i, torsion_size) * find_change_index_read(&gp, i, torsion_size));
			}
			cau = cau * sqrt(cau_t);
			// CAUTIOUS_FACTOR
			cau = cau * 1.0e-6;

			if (ys > cau) {
				/* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
				lm_s_dot_d = 0;
				for (int a = 0; a < n; a++)
				{
					lm_s_dot_d = lm_s_dot_d + lm_s[a] * find_change_index_read(&d, a, torsion_size);
				}
				lm_alpha = lm_s_dot_d / lm_ys;
				for (int b = 0; b < n; b++)
				{
					//flo mimus_lm_alpha_mulit_lm_y = (-lm_alpha * lm_y[b]); 
					find_change_index_write(&d, b, find_change_index_read(&d, b, torsion_size) + lm_alpha * lm_y[b], torsion_size);
					// d *= ys / yy;
					find_change_index_write(&d, b, find_change_index_read(&d, b, torsion_size) * (ys / yy), torsion_size);
				
				}

				beta = 0;
				for (int a = 0; a < n; a++)
				{
					beta += (lm_y[a] * find_change_index_read(&d, a, torsion_size));
				}
				beta /= lm_ys;


				for (int i = 0; i < n; i++)
				{
					find_change_index_write(&d, i, find_change_index_read(&d, i, torsion_size) + ((lm_alpha - beta) * lm_s[i]), torsion_size);
				}

			}
			step = 1.0;
		}

	}

	//  rilc_bfgs init

	if (!(fx <= fx_orig)) {
		fx = fx_orig;

		*x = x_orig;

		*g = g_orig;

	}

	// write output_type_cl energy
	x->e = fx;
}

__device__ void rilc_bfgs_update(			output_type_cl* 	x,
											change_cl* 			g,
											m_cl* 				m_cl_gpu,
											pre_cl* 			p_cl_gpu, // delete the constant
											grids_cl* 		ig_cl_gpu, // delete the constant
						const				mis_cl*				mis,       // delete the global 
						const				int					torsion_size,
						const				int					max_steps,
											m_coords_cl*		m_coords,
											m_minus_forces* 	minus_forces,
						const				int					threadNumInBlock,
						const				int					threadsPerBlock,
						 				rigid_cl*       	m_gpu_rigid,
										grids_gpu*          grids_cuda,
										visited_cl*         visited
)
{
	int lig_torsion_size = torsion_size;
	int n = 3 + 3 + torsion_size;
	if (threadIdx.x==0 && threadIdx.y==0) atomicAdd(&d_stage.n_ep, 1ULL); // the dimensions of matirx

	// rilc_bfgs init
	int k, ls;
	flo step, fx, ys, yy;
	flo beta = 0, cau;
	flo lm_s_dot_d, cau_t;
	//flo d_update_tmp = 0;

	__shared__ output_type_cl xp;
	xp = *x;
	__shared__ change_cl gp;
	gp = *g;


	flo lm_alpha = 0;


	flo lm_s[MAX_HESSIAN_MATRIX_SIZE] = {0}; 
	flo lm_y[MAX_HESSIAN_MATRIX_SIZE] = {0}; 

	flo lm_ys = 0;
	STAGE_T0;
	fx = m_eval_deriv_update(x,
						g,
						m_cl_gpu,
						p_cl_gpu,
						ig_cl_gpu,
						mis->hunt_cap,
						mis->epsilon_fl,
						m_coords, // shared memory
						minus_forces,// shared memory
						threadNumInBlock,
						threadsPerBlock,
						m_gpu_rigid,
						grids_cuda
	);
	STAGE_ADD(eval_init);
	__syncthreads();

	if(!interesting(x, fx, g, visited, torsion_size)){
		x->e = fx;
	}
	else
	{

		add(visited, x, fx, g, lig_torsion_size);
		flo fxp = fx;
		flo fx_orig = fx;
		// Init g_orig, x_orig
		__shared__ change_cl g_orig;
		g_orig = *g;
		__shared__ output_type_cl x_orig;
		x_orig = *x;

		__shared__ change_cl d;
		d = *g;

		//d = -g
		get_to_minus_update(&d, g, threadNumInBlock, threadsPerBlock);
		//__syncthreads();

		if (!(sqrt(scalar_product_update(g, g, n, torsion_size,threadNumInBlock,threadsPerBlock)) >= 1e-5))
			{
				x->e = fx;
				return;
			}
		else
		{
			step = 1.0 / to_norm(&d, n, torsion_size);
			//step = 1.0 / to_norm_update(&d, n, torsion_size,threadNumInBlock,threadsPerBlock);
			k = 1;
		int small_imp_count = 0;

		while (true) {

			xp = *x;
			gp = *g;

			fxp = fx;


			STAGE_T0;
		if (v11_ls_enabled(m_cl_gpu->m_num_movable_atoms)) {
#if defined(V11_LS_GRID) && V11_LS_GRID > 0
			// [v11 tile path — MEASURED AS NET LOSS (11x), kept for the record;
			//  the k>1 reset below is also unused by the serial path because
			//  line_search_lewisoverton_update recomputes E at *xp itself.]
			if (k > 1) {   // after iter 1 the master coords/forces are STALE (trials never touch them)
				set_update(x, m_gpu_rigid, m_coords, m_cl_gpu->atoms,
						m_cl_gpu->m_num_movable_atoms, mis->epsilon_fl);
				fx = ig_eval_deriv_update_optimal(x, g, mis->hunt_cap[1], ig_cl_gpu, m_cl_gpu,
						mis->epsilon_fl, m_coords, minus_forces,
						threadNumInBlock, threadsPerBlock, grids_cuda);
				fx += eval_interacting_pairs_deriv_update_optimal(p_cl_gpu, mis->hunt_cap[0],
						&m_cl_gpu->ligand.pairs, m_coords, minus_forces, mis->epsilon_fl,
						threadNumInBlock, threadsPerBlock);
				POT_deriv_update(minus_forces, m_gpu_rigid, m_coords, g, threadNumInBlock, threadsPerBlock);
				gp = *g;   // two-loop below diffs against the gradient AT the accepted point
			}
			ls = line_search_lewisoverton_tile(m_cl_gpu,
					p_cl_gpu,
					ig_cl_gpu,
					n,
					&step,
					x,
					&fx,
					g,
					&d,
					&xp,
					&gp,
					mis->epsilon_fl,
					mis->hunt_cap,
					torsion_size,
					m_coords,
					minus_forces,
					threadNumInBlock,
					threadsPerBlock,
					m_gpu_rigid,
					grids_cuda
			);
		} else {
			ls = line_search_lewisoverton_update(m_cl_gpu,
					p_cl_gpu,
					ig_cl_gpu,
					n,
					&step,
					x,
					&fx,
					g,
					&d,
					&xp,
					&gp,
					mis->epsilon_fl,
					mis->hunt_cap,
					torsion_size,
					m_coords,
					minus_forces,
					threadNumInBlock,
					threadsPerBlock,
					m_gpu_rigid,
					grids_cuda
			);
#else
			// [v4.5] v4 serial LS (block=128 with the correctness fixes; tile path compiled out)
			ls = line_search_lewisoverton_update(m_cl_gpu,
					p_cl_gpu,
					ig_cl_gpu,
					n,
					&step,
					x,
					&fx,
					g,
					&d,
					&xp,
					&gp,
					mis->epsilon_fl,
					mis->hunt_cap,
					torsion_size,
					m_coords,
					minus_forces,
					threadNumInBlock,
					threadsPerBlock,
					m_gpu_rigid,
					grids_cuda
			);
#endif
		} // [v11] end tile/serial dispatch
		STAGE_ADD(ls_total);
		if (threadIdx.x==0 && threadIdx.y==0) atomicAdd(&d_stage.n_ls, 1ULL);
			// __syncthreads();
			if (ls < 0) {

				*x = xp;
				*g = gp;

				fx = fxp;
				break;
			}
			// A2: BFGS early termination on small relative improvement
			if (k >= 4) {
				flo rel_imp = fabs(fxp - fx) / fmax(fabs(fxp), 1.0f);
				if (rel_imp < 1e-2) {
					small_imp_count++;
					if (small_imp_count >= 2) {
						x->e = fx;
						return;
					}
				} else {
					small_imp_count = 0;
				}
			}
			if (!(sqrt(scalar_product_update(g, g, n, torsion_size,threadNumInBlock,threadsPerBlock)) >= 1e-5)) {
				x->e = fx;
				return;
			}
			if (max_steps != 0 && max_steps <= k) {
				break;
			}
			/* Count the iteration number. */
			++k;
			if (threadIdx.x==0 && threadIdx.y==0) atomicAdd(&d_stage.n_iter, 1ULL);

unsigned long long st_t0w = st_clock_ns();
			for (int i = 0; i < n; i++) {
				lm_s[i] = step * find_change_index_read_update(&d, i, torsion_size);
				lm_y[i] = find_change_index_read_update(g, i, torsion_size) - find_change_index_read_update(&gp, i, torsion_size);
			}

			ys = 0;
			yy = 0;

			for (int i = 0; i < n; i++) {
				ys += lm_y[i] * lm_s[i];
				yy += lm_y[i] * lm_y[i];
			}


			lm_ys = ys;

			/* Compute the negative of gradients. */
			// d = -g
			get_to_minus_update(&d, g, threadNumInBlock,threadsPerBlock);
			// __syncthreads();
			cau = 0;
			cau_t = 0;
			for (int i = 0; i < n; i++) {
				cau += lm_s[i] * lm_s[i];
				cau_t = cau_t + (find_change_index_read_update(&gp, i, torsion_size) * find_change_index_read_update(&gp, i, torsion_size));
			}
			
			
			cau = cau * sqrt(cau_t);
			// CAUTIOUS_FACTOR
			cau = cau * 1.0e-6;

			if (ys > cau) {
				/* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
				lm_s_dot_d = 0;
				
				for (int a = 0; a < n; a++)
				{
					lm_s_dot_d = lm_s_dot_d + lm_s[a] * find_change_index_read_update(&d, a, torsion_size);
				}
				
				lm_alpha = lm_s_dot_d / lm_ys;
				
				for (int b = 0; b < n; b ++)
				{
				//flo mimus_lm_alpha_mulit_lm_y = (-lm_alpha * lm_y[b]); 
					find_change_index_write_update(&d, b, find_change_index_read_update(&d, b, torsion_size) + lm_alpha * lm_y[b], torsion_size);
				}

				// d *= ys / yy;
				for (int i = 0; i < n; i ++)
				{
					find_change_index_write_update(&d, i, find_change_index_read_update(&d, i, torsion_size) * (ys / yy), torsion_size);
				}

				beta = 0;
				for (int a = 0; a < n; a++)
				{
					beta += (lm_y[a] * find_change_index_read_update(&d, a, torsion_size));
				}
					beta /= lm_ys;


				for (int i = 0; i < n; i ++)
				{
					find_change_index_write_update(&d, i, find_change_index_read_update(&d, i, torsion_size) + ((lm_alpha - beta) * lm_s[i]), torsion_size);
				}

			}
				STAGE_ADD2(st_t0w, twop);
				step = 1.0;
		}

		}

		//  rilc_bfgs init

		if (!(fx <= fx_orig)) {
			fx = fx_orig;

			*x = x_orig;

			*g = g_orig;

		}

		// write output_type_cl energy
		x->e = fx;
	}
}

__device__ void bfgs(			output_type_cl*			x,
								change_cl*			g,
								m_cl*				m,
								pre_cl*				pre,  // delete the constant 
								grids_cl*			grids,  // delete the constant 
			const				mis_cl*				mis,   // delete the __global
			const				int					torsion_size,
			const				int					max_bfgs_steps
) 
{
	int lig_torsion_size = torsion_size;
	int n = 3 + 3 + lig_torsion_size; // the dimensions of matirx

	matrix_gpu h;
	matrix_init(&h, n, 0);
	matrix_set_diagonal(&h, 1);

	change_cl g_new = *g;

	output_type_cl x_new = *x ;
	flo f0 = m_eval_deriv(	x,
								g,
								m,
								pre,
								grids,
								mis->hunt_cap,
								mis->epsilon_fl
							);

	flo f_orig = f0;

	change_cl g_orig = *g;
	output_type_cl x_orig = *x;

	change_cl p = *g;

	for (int step = 0; step < max_bfgs_steps; step++) {

		minus_mat_vec_product(&h, g, &p, torsion_size);
		flo f1 = 0;

		const flo alpha = line_search(	m,
											pre,
											grids,
											n,
											x,
											g,
											f0,
											&p,
											&x_new,
											&g_new,
											&f1,
											mis->epsilon_fl,
											mis->hunt_cap,
											torsion_size
										);
		
		change_cl y = g_new;
		// subtract_change
		for (int i = 0; i < n; i++) {
			flo tmp = find_change_index_read(&y, i, torsion_size) - find_change_index_read(g, i, torsion_size);
			find_change_index_write(&y, i, tmp, torsion_size);
		}
		
		f0 = f1;
		*x = x_new;
		if (!(sqrt(scalar_product(g, g, n, torsion_size)) >= 1e-5))break;
		*g = g_new;

		if (step == 0) {
			flo yy = scalar_product(&y, &y, n, torsion_size);
			if (fabs(yy) > mis->epsilon_fl) {
				matrix_set_diagonal(&h, alpha * scalar_product(&y, &p, n, torsion_size) / yy);
			}
		}
		bool h_updated = bfgs_update(&h, &p, &y, alpha, mis, torsion_size);
	}

	if (!(f0 <= f_orig)) {
		f0 = f_orig;
		*x = x_orig;
		*g = g_orig;
	}

	// write output_type_cl energy
	x->e = f0;
}

#endif