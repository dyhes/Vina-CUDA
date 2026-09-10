#ifndef QUASI_NEWTON_GPU_H_
#define QUASI_NEWTON_GPU_H_

typedef double fl;

struct vec_gpu;
struct Small;

template<class Size>
struct model_gpu;

template<class Size>
struct cache_gpu; 

template<class Size>
struct precalculate_byatom_gpu;

template<class Size>
struct output_type_gpu;

template<class Size>
struct change_gpu;
struct quasi_newton_gpu {
    unsigned max_steps;
    fl average_required_improvement;
    // quasi_newton() : max_steps(1000), average_required_improvement(0.0) {}
    __device__ void init();
    // clean up
    template<class Size>
    __device__ void operator()(
        model_gpu<Size> & m,
        const precalculate_byatom_gpu<Size> & p,
        const cache_gpu<Size> & ig,
        output_type_gpu<Size> & out,
        change_gpu<Size> & g,
        const vec_gpu & v,
        int & evalcount
    ) const; // g must have correct size
};

template<>
__device__ void quasi_newton_gpu::operator()<Small>(
    model_gpu<Small> & m,
    const precalculate_byatom_gpu<Small> & p,
    const cache_gpu<Small> & ig,
    output_type_gpu<Small> & out,
    change_gpu<Small> & g,
    const vec_gpu & v,
    int & evalcount
) const; // g must have correct size

#endif
