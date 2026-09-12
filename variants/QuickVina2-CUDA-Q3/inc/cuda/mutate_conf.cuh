#ifndef _MUTATE_CONF_H_
#define _MUTATE_CONF_H_
#include "commonMacros.h"

__device__ void quaternion_increment(flo* q, const flo* rotation, flo epsilon_fl);
__device__ inline void normalize_angle(flo* x);

__device__ void output_type_cl_increment(output_type_cl* x, const change_cl* c, flo factor, flo epsilon_fl, int lig_torsion_size) {
	
	flo rotation[3];
	// position increment and orientation increment
	for (int k = 0; k < 3; k++) {
		x->position[k] += factor * c->position[k];
		rotation[k] = factor * c->orientation[k];
	}
	quaternion_increment(x->orientation, rotation, epsilon_fl);
	
	// torsion increment
	for (int k = 0; k < lig_torsion_size; k++) {
		flo tmp = factor * c->lig_torsion[k];
		normalize_angle(&tmp);
		x->lig_torsion[k] += tmp;
		normalize_angle(&(x->lig_torsion[k]));
	}
}

__device__ inline flo norm3(const flo* a) {
	return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

__device__ inline void normalize_angle(flo* x) {
	// [Q4-7M] was while-loop with double M_PI promotions (FP64 DSETP chains);
	// float PI, branchless-ish reduction (max 3 wraps for |angle|<~10)
	const flo PI_f = 3.14159265358979f;
	const flo TWO_PI_f = 2 * PI_f;
	while (*x > 3 * PI_f)  { flo n = nearbyintf((*x - PI_f) / TWO_PI_f); *x -= n * TWO_PI_f; if (n == 0) break; }
	while (*x < -3 * PI_f) { flo n = nearbyintf((-*x - PI_f) / TWO_PI_f); *x += n * TWO_PI_f; if (n == 0) break; }
	if (*x > PI_f)   *x -= TWO_PI_f;
	else if (*x < -PI_f) *x += TWO_PI_f;
}

__device__ inline void normalize_angle_update(float* x) {
    *x = fmodf(*x + M_PI, 2 * M_PI) - M_PI;
}


__device__ inline bool quaternion_is_normalized(flo* q) {
	flo q_pow = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
	flo sqrt_q_pow = sqrt(q_pow);
	return (q_pow - 1 < 0.001f) && (sqrt_q_pow - 1 < 0.001f);
}

__device__ inline void angle_to_quaternion(flo* q, const flo* rotation, flo epsilon_fl) {
	//flo angle = norm3(rotation);
	flo angle = sqrtf(rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2]);
	if (angle > epsilon_fl) {
		flo axis[3] = { rotation[0] / angle, rotation[1] / angle ,rotation[2] / angle };
		// [Q4-7M] axis-norm ERROR printf removed (FP64 promotion)
		normalize_angle(&angle);
		flo c = cosf(angle / 2);
		flo s = sinf(angle / 2);
		q[0] = c; q[1] = s * axis[0]; q[2] = s * axis[1]; q[3] = s * axis[2];
		return;
	}
	
	q[0] = 1; q[1] = 0; q[2] = 0; q[3] = 0;
	
	return;
}

__device__ inline void angle_to_quaternion_update(flo* q, const flo* rotation, flo epsilon_fl) {
	flo a0 = rotation[0], a1 = rotation[1], a2 = rotation[2];
	flo angle = sqrtf(a0 * a0 + a1 * a1 + a2 * a2);
	if (angle > epsilon_fl) {
		flo axis[3] = { a0 / angle, a1 / angle , a2 / angle };
		// [Q4-7M] norm3 ERROR printf removed (FP64 promotion via 0.001 literal)
		normalize_angle(&angle);
		flo c = cosf(angle / 2);
		flo s = sinf(angle / 2);
		q[0] = c; q[1] = s * axis[0]; q[2] = s * axis[1]; q[3] = s * axis[2];
		return;
	}
}

// quaternion multiplication
__device__ inline void angle_to_quaternion_multi(flo* qa, const flo* qb) {
	flo tmp[4] = { qa[0],qa[1],qa[2],qa[3] };
	qa[0] = tmp[0] * qb[0] - tmp[1] * qb[1] - tmp[2] * qb[2] - tmp[3] * qb[3];
	qa[1] = tmp[0] * qb[1] + tmp[1] * qb[0] + tmp[2] * qb[3] - tmp[3] * qb[2];
	qa[2] = tmp[0] * qb[2] - tmp[1] * qb[3] + tmp[2] * qb[0] + tmp[3] * qb[1];
	qa[3] = tmp[0] * qb[3] + tmp[1] * qb[2] - tmp[2] * qb[1] + tmp[3] * qb[0];
}


__device__ inline void angle_to_quaternion_multi_update(flo* qa, const flo* qb) {
    flo a0 = qa[0], a1 = qa[1], a2 = qa[2], a3 = qa[3];

    flo b0 = qb[0], b1 = qb[1], b2 = qb[2], b3 = qb[3];

    qa[0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3;
    qa[1] = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2;
    qa[2] = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1;
    qa[3] = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0;
}

__device__ inline void quaternion_normalize_approx(flo* q, flo epsilon_fl) {
	const flo s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
	// Omit one assert()
	if (fabs(s - 1) >= TOLERANCE){
		const flo a = sqrt(s);
		if (a <= epsilon_fl) printf("\nmutate: quaternion_normalize_approx ERROR!"); // Replace assert(a > epsilon_fl);
		for (int i = 0; i < 4; i += 2) {
			q[i] *= (1 / a);
			q[i+1] *= (1 / a);
		}
		if (quaternion_is_normalized(q) != true)printf("\nmutate: quaternion_normalize_approx() ERROR!");// Replace assert(quaternion_is_normalized(q));
	}else
	{
		//printf("fabs(s - 1) < TOLERANCE)");
	}
}


__device__ void quaternion_increment(float* q, const float* rotation, float epsilon_fl) {
	//if (quaternion_is_normalized(q) != true)printf("\nmutate: quaternion_increment() ERROR!"); // Replace assert(quaternion_is_normalized(q))
	float q_old[4] = { q[0],q[1],q[2],q[3] };
	angle_to_quaternion(q, rotation, epsilon_fl);
	angle_to_quaternion_multi_update(q, q_old);
	quaternion_normalize_approx(q, epsilon_fl);
}


__device__ inline flo vec_distance_sqr(const flo* a, const flo* b) {
	flo dx = a[0] - b[0];
    flo dy = a[1] - b[1];
    flo dz = a[2] - b[2];
    
    return dx * dx + dy * dy + dz * dz;
	//return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]);
}

__device__  flo gyration_radius(				int				m_lig_begin,
									int				m_lig_end,
						const		atom_cl*		atoms,
						const		m_coords_cl*	m_coords_gpu,
						const		flo*			m_lig_node_origin
) {
	flo acc = 0;
	int counter = 0;
	flo origin[3] = { m_lig_node_origin[0], m_lig_node_origin[1], m_lig_node_origin[2] };
	for (int i = m_lig_begin; i < m_lig_end; i++) {
		flo current_coords[3] = { m_coords_gpu->coords[i][0], m_coords_gpu->coords[i][1], m_coords_gpu->coords[i][2] };
		if (atoms[i].types[0] != EL_TYPE_H) { // for el, we use the first element (atoms[i].types[0])
			acc += vec_distance_sqr(current_coords, origin);
			++counter;
		}
	}
	return (counter > 0) ? sqrt(acc / counter) : 0;
}

__device__  flo gyration_radius_update(				int				m_lig_begin,
									int				m_lig_end,
						atom_cl*			atoms,
					const		m_coords_cl*	m_coords_gpu,
					const		flo*			m_lig_node_origin,
						const		int				threadNumInBlock,
						const		int				threadsPerBlock
) {
	// [Q5.2] register + shuffle reduction replaces acc[128]/counter[128]
	// shared arrays (-1,024B/block). Each lane accumulates its strided
	// subset in registers, in-warp shfl tree, cross-warp combine through
	// a 2-entry shared partial. Semantics: same atom set (i in
	// [0, m_lig_end) strided, non-H only), same sum/count; float
	// addition order may differ from the array version (within the
	// accepted reduction-order noise band, cf. Q5.1 aff symmetry).
	const flo ox = m_lig_node_origin[0];
	const flo oy = m_lig_node_origin[1];
	const flo oz = m_lig_node_origin[2];

	flo my_acc = 0.0f;
	int  my_cnt = 0;
	for (int i = threadNumInBlock; i < m_lig_end; i = i + threadsPerBlock) {
		if (atoms[i].types[0] != EL_TYPE_H) {
			const flo cx = m_coords_gpu->coords[i][0] - ox;
			const flo cy = m_coords_gpu->coords[i][1] - oy;
			const flo cz = m_coords_gpu->coords[i][2] - oz;
			my_acc += cx * cx + cy * cy + cz * cz;
			my_cnt += 1;
		}
	}

	// in-warp shuffle reduction (full-warp mask; all lanes converged here)
	for (int off = 16; off > 0; off >>= 1) {
		my_acc += __shfl_down_sync(0xFFFFFFFFu, my_acc, off);
		my_cnt += __shfl_down_sync(0xFFFFFFFFu, my_cnt, off);
	}

	// cross-warp combine via minimal shared scratch
	__shared__ float warp_acc[32];
	__shared__ int   warp_cnt[32];
	const int wid = threadNumInBlock >> 5;
	const int lane = threadNumInBlock & 31;
	const int nwarps = (threadsPerBlock + 31) >> 5;
	if (lane == 0) { warp_acc[wid] = my_acc; warp_cnt[wid] = my_cnt; }
	__syncthreads();

	if (wid == 0) {
		float a = (lane < nwarps) ? warp_acc[lane] : 0.0f;
		int   c = (lane < nwarps) ? warp_cnt[lane] : 0;
		for (int off = 16; off > 0; off >>= 1) {
			a += __shfl_down_sync(0xFFFFFFFFu, a, off);
			c += __shfl_down_sync(0xFFFFFFFFu, c, off);
		}
		if (lane == 0) { warp_acc[0] = a; warp_cnt[0] = c; }
	}
	__syncthreads();

	return (warp_cnt[0] > 0) ? sqrtf(warp_acc[0] / (flo)warp_cnt[0]) : 0.0f;
}

__device__ void mutate_conf_cl(const		int				step,
											output_type_cl*	c,
										    int*			random_int_map, // delete the  __constant
											flo			random_inside_sphere_map[][3], // delete the  __constant
											flo*			random_fl_pi_map, // delete the  __constant
					const					int				m_lig_begin,
					const					int				m_lig_end,
					const					atom_cl*		atoms,
					const					m_coords_cl*	m_coords_gpu,
					const					flo*			m_lig_node_origin_gpu,
					const					flo			epsilon_fl,
					const					flo			amplitude,
					const					int				lig_torsion_size
) {

	int index = step; // global index (among all threads)
	int which = random_int_map[index];
	int flex_torsion_size = 0;  
		if (which == 0) {
			for (int i = 0; i < 3; i++)
				c->position[i] += amplitude * random_inside_sphere_map[index][i];
			return;
		}
		--which;
		if (which == 0) {
			flo gr = gyration_radius(m_lig_begin, m_lig_end, atoms, m_coords_gpu, m_lig_node_origin_gpu);
			if (gr > epsilon_fl) {
				flo rotation[3];
				for (int i = 0; i < 3; i++)rotation[i] = amplitude / gr * random_inside_sphere_map[index][i];
				quaternion_increment(c->orientation, rotation, epsilon_fl);
			}
			return;
		}
		--which;
		if (which < lig_torsion_size) { c->lig_torsion[which] = random_fl_pi_map[index]; return; }
		which -= lig_torsion_size;

	if (flex_torsion_size != 0) {
		if (which < flex_torsion_size) { c->flex_torsion[which] = random_fl_pi_map[index]; return; }
		which -= flex_torsion_size;
	}
}

__device__ void mutate_conf_cl_update(const		int				step,
											output_type_cl*	c,
											int*			random_int_map, // delete the  __constant
											flo			random_inside_sphere_map[][3], // delete the  __constant
											flo*			random_fl_pi_map, // delete the  __constant
					const					int				m_lig_begin,
					const					int				m_lig_end,
											atom_cl*		atoms,
					const					m_coords_cl*	m_coords_gpu,
										flo*			m_lig_node_origin_gpu,
					const					flo			epsilon_fl,
					const					flo			amplitude,
					const					int				lig_torsion_size,
					const					int				threadNumInBlock,
					const					int				threadsPerBlock
) {

		int index = step; // global index (among all threads)
		int which = random_int_map[index];
		int flex_torsion_size = 0;  
		if (which == 0) {
			for (int i = threadNumInBlock; i < 3; i += threadsPerBlock)
				c->position[i] += amplitude * random_inside_sphere_map[index][i];
			return;
		}
		--which;
		if (which == 0) {
			flo gr = gyration_radius_update(m_lig_begin, m_lig_end, atoms, m_coords_gpu, m_lig_node_origin_gpu,threadNumInBlock,threadsPerBlock);
			//flo gr = gyration_radius(m_lig_begin, m_lig_end, atoms, m_coords_gpu, m_lig_node_origin_gpu);
			
			if (gr > epsilon_fl) {
				__shared__ flo rotation[3];
				for (int i = threadNumInBlock; i < 3; i += threadsPerBlock)
				{
					rotation[i] = amplitude / gr * random_inside_sphere_map[index][i];
				}
				    __syncthreads();	
					quaternion_increment(c->orientation, rotation, epsilon_fl);		
			}
			return;
		}
				
		--which;
		if (which < lig_torsion_size) { c->lig_torsion[which] = random_fl_pi_map[index]; return; }
		which -= lig_torsion_size;

		if (flex_torsion_size != 0) {
			if (which < flex_torsion_size) { c->flex_torsion[which] = random_fl_pi_map[index]; return; }
			which -= flex_torsion_size;
		}
	
}
#endif