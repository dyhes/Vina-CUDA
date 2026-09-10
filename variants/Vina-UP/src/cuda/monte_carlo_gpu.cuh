
#ifndef MONTE_CARLO_GPU_H_
#define MONTE_CARLO_GPU_H_

#include "common_gpu.cuh"

struct Small;

typedef double fl;
typedef unsigned long sz;

template<class Size>
struct model_gpu;

template<class Size>
struct cache_gpu;

template<class Size>
struct precalculate_byatom_gpu;

template<class Size>
struct output_type_gpu;

struct monte_carlo_gpu {
    unsigned max_evals;
	unsigned global_steps;
	fl temperature;
	vec_gpu hunt_cap;
	fl min_rmsd;
	sz num_saved_mins;
	fl mutation_amplitude;
	unsigned local_steps;

    // monte_carlo() : max_evals(0), global_steps(2500), temperature(1.2), hunt_cap(10, 1.5, 10), min_rmsd(0.5), num_saved_mins(50), mutation_amplitude(2) {}

	// out is sorted
    template<class Size>
	__device__ void operator()(
        model_gpu<Size> & m,
        output_type_gpu<Size> & out,
        const precalculate_byatom_gpu<Size> & p,
        const cache_gpu<Size> & ig,
        const vec_gpu & corner1,
        const vec_gpu & corner2,
        curandState & generator
    ) const;
};


// template specialize, so no need to provide implementation
template<>
__device__ void monte_carlo_gpu::operator()<Small>(
    model_gpu<Small> & m,
    output_type_gpu<Small> & out,
    const precalculate_byatom_gpu<Small> & p,
    const cache_gpu<Small> & ig,
    const vec_gpu & corner1,
    const vec_gpu & corner2,
    curandState & generator
) const;

#endif

