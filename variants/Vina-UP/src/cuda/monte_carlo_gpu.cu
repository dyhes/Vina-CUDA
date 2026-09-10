#include "monte_carlo_gpu.cuh"
#include "quasi_newton_gpu.cuh"
#include "model_gpu.cuh"
#include "precalculate_gpu.cuh"
#include "output_type_gpu.cuh"
#include "common_gpu.cuh"

#include <stdio.h>

__device__ bool metropolis_accept_gpu(
	fl old_f,
	fl new_f,
	fl temperature,
	curandState & generator
) {
	if(new_f < old_f) return true;
	const fl acceptance_probability = std::exp((old_f - new_f) / temperature);
	return random_fl_gpu(0, 1, generator) < acceptance_probability;
}

template<class Size>
__device__ void mutate_conf_gpu(
	conf_gpu<Size> & c,
	fl amplitude,
	curandState & generator
) { // ONE OF: 2A for position, similar amp for orientation, randomize torsion
	sz mutable_entities_num = c.ligand.torsions_size + 2;
	if(mutable_entities_num == 0) return;
	
	int which_int = random_int_gpu(0, int(mutable_entities_num - 1), generator);
	assert(which_int >= 0);
	sz which = sz(which_int);
	assert(which < mutable_entities_num);

	switch(which) {
		case 0: {
			c.ligand.rigid.position += amplitude * random_inside_sphere_gpu(generator);
			break;
		}
		case 1: {
			vec_gpu rotation; 
			rotation = amplitude * random_inside_sphere_gpu(generator); 
			quaternion_increment_gpu(c.ligand.rigid.orientation, rotation);
			break;
		}
		default:
			c.ligand.torsions[which - 2] = random_fl_gpu(-pi_gpu, pi_gpu, generator);
	}
}

// out is sorted
template<>
__device__ void monte_carlo_gpu::operator()<Small>(
	model_gpu<Small> & m,
	output_type_gpu<Small> & out,
	const precalculate_byatom_gpu<Small> & p,
	const cache_gpu<Small> & ig,
	const vec_gpu & corner1,
	const vec_gpu & corner2,
	curandState & generator
) const {
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
    int evalcount = 0;
	vec_gpu authentic_v;
	authentic_v.init(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl
	change_gpu<Small> g;
	g.init(m.ligand.flexible_body.tree_nodes_size);
	output_type_gpu<Small> tmp;
	tmp.init(m.ligand.flexible_body.tree_nodes_size, 0);
	tmp.c.randomize(corner1, corner2, generator);
	fl best_e = max_fl_gpu;
	quasi_newton_gpu quasi_newton_par;
    quasi_newton_par.max_steps = local_steps;
	quasi_newton_par(m, p, ig, tmp, g, hunt_cap, evalcount);
	out = tmp; // always initialize out with a valid optimized conformation
	for(int step = 0; step < global_steps; step++) {
		if((max_evals > 0) & (evalcount > max_evals))
			break;
		
		output_type_gpu<Small> candidate = tmp;
		mutate_conf_gpu(candidate.c, mutation_amplitude, generator);

		quasi_newton_par(m, p, ig, candidate, g, hunt_cap, evalcount);

		if(step == 0 || metropolis_accept_gpu(tmp.e, candidate.e, temperature, generator)) {
			tmp = candidate;

			m.set(tmp.c); // FIXME? useless?

			// FIXME only for very promising ones
			if(tmp.e < best_e) {
				quasi_newton_par(m, p, ig, tmp, g, authentic_v, evalcount);
				m.set(tmp.c); // FIXME? useless?
				// m.get_heavy_atom_movable_coords(tmp);
				out = tmp; // 20 - max size
				if(tmp.e < best_e)
					best_e = tmp.e;
			}
		}

		// temp_storage_fl[threadIdx.x] = best_e;
		// __syncthreads();
		// fl _e = block_min(temp_storage_fl, (Small::atoms_size / 32 + 1) * 32);

		if(thread == 0) {
			if(step % 50 == 0)
				printf("thread: %d, step: %d, e: %.3f\n", thread, step, best_e);
			if(step + 1 == global_steps)
				printf("thread: %d, step: %d, e: %.3f\n", thread, step, best_e);
		}

		// if(step % 100 == 0) printf("step: %d, e: %.3f\n", step, best_e);
		// else if(step + 1 == global_steps) printf("step: %d, e: %10.3f\n", step, best_e);
		// if(thread == 0) printf("thread: %d, step: %d, e: %.3f\n", thread, step, best_e);
	}
}
