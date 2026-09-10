#include "quasi_newton_gpu.cuh"
#include "model_gpu.cuh"
#include "cache_gpu.cuh"
#include "precalculate_gpu.cuh"
#include "output_type_gpu.cuh"

#include <cuda/std/cmath>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

typedef unsigned long sz;

template<class Size>
struct quasi_newton_aux_gpu {
    model_gpu<Size> * m;
    const precalculate_byatom_gpu<Size> * p;
    const cache_gpu<Size> * ig;
    const vec_gpu * v;

    // quasi_newton_aux(model* m_, const precalculate_byatom* p_, const igrid* ig_, const vec& v_) : m(m_), p(p_), ig(ig_), v(v_) {}

    __device__ void init(
        model_gpu<Size> * _m,
        const precalculate_byatom_gpu<Size> * _p,
        const cache_gpu<Size> * _ig,
        const vec_gpu & _v
    );
    
    __device__ fl operator()(
        const conf_gpu<Size> & c,
        change_gpu<Size> & g
    );

	template<int tile_size>
	__device__ fl operator()(
        const conf_gpu<Size> & c,
        change_gpu<Size> & g,
		cg::thread_block_tile<tile_size> & tile,
		fl * temp_storage_fl,
		int async
    );
};

template<class Size>
__device__ void quasi_newton_aux_gpu<Size>::init(
    model_gpu<Size> * _m,
    const precalculate_byatom_gpu<Size> * _p,
    const cache_gpu<Size> * _ig,
    const vec_gpu & _v
) {
    // COPY:
    m = _m;
    p = _p;
    ig = _ig;
    v = &_v;
}

template<class Size>
__device__ fl quasi_newton_aux_gpu<Size>::operator()(
    const conf_gpu<Size> & c,
    change_gpu<Size> & g
) {
    // Before evaluating conf, we have to update model
    m->set(c);

    const fl tmp = m->eval_deriv(*p, *ig, *v, g);
    return tmp;
}

template<class Size>
template<int tile_size>
__device__ fl quasi_newton_aux_gpu<Size>::operator()(
    const conf_gpu<Size> & c,
    change_gpu<Size> & g,
	cg::thread_block_tile<tile_size> & tile,
	fl * temp_storage_fl,
	int async
) {
    // Before evaluating conf, we have to update model
    (*m).set<tile_size>(c, tile, 0);
	tile.sync();

    const fl tmp = (*m).eval_deriv<tile_size>(*p, *ig, *v, g, tile, temp_storage_fl, 0);
	tile.sync();
    return tmp;
}

template<int size>
__device__ inline void set_diagonal_gpu(
    triangular_matrix_gpu<fl, size> & m,
    fl x
) {
	for(int i = 0; i < m.m_dim; i++)
		m(i, i) = x;
}

template<int size>
__device__ inline void set_diagonal_gpu(
    triangular_matrix_gpu<fl, size> & m,
    fl x,
	int async
) {
	for(int i = threadIdx.x; i < m.m_dim; i += blockDim.x)
		m(i, i) = x;
}

template<class Size>
__device__ void minus_mat_vec_product_gpu(
    const triangular_matrix_gpu<fl, Size::torsions_size + 6> & m,
    const change_gpu<Size> & in,
    change_gpu<Size> & out
) {
	sz n = m.m_dim;
	for(int i = 0; i < n; i++) {
		fl sum = 0;
		for(int j = 0; j < n; j++)
			sum += m(i, j) * in(j);
		out(i) = -sum;
	}
}

template<class Size>
__device__ void minus_mat_vec_product_gpu(
    const triangular_matrix_gpu<fl, Size::torsions_size + 6> & m,
    const change_gpu<Size> & in,
    change_gpu<Size> & out,
	int async
) {
	sz n = m.m_dim;
	for(int i = threadIdx.x; i < n; i += blockDim.x) {
		fl sum = 0;
		for(int j = 0; j < n; j++)
			sum += m(i, j) * in(j);
		out(i) = -sum;
	}
}

template<class Size>
__device__ inline fl scalar_product_gpu(
    const change_gpu<Size> & a,
    const change_gpu<Size> & b,
    sz n
) {
	fl tmp = 0;
	for(int i = 0; i < n; i++)
		tmp += a(i) * b(i);
	return tmp;
}

template<class Size>
__device__ inline fl scalar_product_gpu(
    const change_gpu<Size> & a,
    const change_gpu<Size> & b,
    sz n,
	fl * temp_storage_fl,
	int async
) {
	for(int i = threadIdx.x; i < n; i += blockDim.x)
		temp_storage_fl[i] = a(i) * b(i);
	__syncthreads();
	fl tmp = block_sum(temp_storage_fl, n);
	return tmp;
}

template<class Size>
__device__ inline bool bfgs_update_gpu(
    triangular_matrix_gpu<fl, Size::torsions_size + 6> & h,
    const change_gpu<Size> & p,
    const change_gpu<Size> & y,
    const fl alpha
) {
	const fl yp  = scalar_product_gpu(y, p, h.m_dim);
	if(alpha * yp < epsilon_fl_gpu) return false; // FIXME?
	change_gpu<Size> minus_hy(y);
	minus_mat_vec_product_gpu(h, y, minus_hy);
	const fl yhy = - scalar_product_gpu(y, minus_hy, h.m_dim);
	const fl r = 1 / (alpha * yp); // 1 / (s^T * y) , where s = alpha * p // FIXME   ... < epsilon
	const sz n = p.num_floats();
	for(int i = 0; i < n; i++)
		for(int j = i; j < n; j++) // includes i
			h(i, j) +=   alpha * r * (minus_hy(i) * p(j)
	                                + minus_hy(j) * p(i)) +
			           + alpha * alpha * (r*r * yhy  + r) * p(i) * p(j); // s * s == alpha * alpha * p * p
	return true;
}

template<class Size>
__device__ inline bool bfgs_update_gpu(
    triangular_matrix_gpu<fl, Size::torsions_size + 6> & h,
    const change_gpu<Size> & p,
    const change_gpu<Size> & y,
    const fl alpha,
	fl * temp_storage_fl,
	change_gpu<Size> & minus_hy,
	int async
) {
	const fl yp  = scalar_product_gpu(y, p, h.m_dim, temp_storage_fl, 0);
	if(alpha * yp < epsilon_fl_gpu) return false; // FIXME?

	minus_hy.init(y, 0);
	__syncthreads();

	minus_mat_vec_product_gpu(h, y, minus_hy, 0);
	__syncthreads();

	const fl yhy = - scalar_product_gpu(y, minus_hy, h.m_dim, temp_storage_fl, 0);
	const fl r = 1 / (alpha * yp); // 1 / (s^T * y) , where s = alpha * p // FIXME   ... < epsilon
	const sz n = p.num_floats();
	const int mat_size = (n + 1) * n / 2;

	for(int m = threadIdx.x; m < mat_size; m += blockDim.x) {
		int i = m / n;
		int j = m % n;
		if(j >= i) h(i, j) += 
			alpha * r * (minus_hy(i) * p(j) + minus_hy(j) * p(i)) +
			alpha * alpha * (r*r * yhy  + r) * p(i) * p(j);
	}
	
	return true;
}

template<class Size>
__device__ fl line_search_gpu(
    quasi_newton_aux_gpu<Size> & f,
    sz n,
    const conf_gpu<Size> & x,
    const change_gpu<Size> & g,
    const fl f0,
    const change_gpu<Size> & p,
    conf_gpu<Size> & x_new,
    change_gpu<Size> & g_new,
    fl & f1,
    int & evalcount
) { // returns alpha
	const fl c0 = 0.0001;
	const unsigned max_trials = 10;
	const fl multiplier = 0.5;
	fl alpha = 1;

	const fl pg = scalar_product_gpu(p, g, n);

	for(int trial = 0; trial < max_trials; trial++) {
		x_new = x;
		x_new.increment(p, alpha);
		f1 = f(x_new, g_new);
		evalcount++;
		if(f1 - f0 < c0 * alpha * pg) // FIXME check - div by norm(p) ? no?
			break;
		alpha *= multiplier;
	}
	return alpha;
}

template<class Size, int tile_size>
__device__ fl line_search_gpu(
    quasi_newton_aux_gpu<Size> & f,
    sz n,
    const conf_gpu<Size> & x,
    const change_gpu<Size> & g,
    const fl f0,
    const change_gpu<Size> & p,
    conf_gpu<Size> & x_new,
    change_gpu<Size> & g_new,
    fl & f1,
    int & evalcount,
	fl * temp_storage_fl,
	int * temp_storage_int,
	int async
) { // returns alpha
	const fl c0 = 0.0001;
	const unsigned max_trials = 10;
	const fl multiplier = 0.5;
	fl alpha = 1;

	const fl pg = scalar_product_gpu(p, g, n, temp_storage_fl, 0);
	__syncthreads();

	conf_gpu<Size> x_tmp;
	change_gpu<Size> g_tmp;
	g_tmp = g;

	temp_storage_int[threadIdx.x] = max_trials;
	__syncthreads();

	cg::thread_block bg = cg::this_thread_block();
	cg::thread_block_tile<tile_size> tile = cg::tiled_partition<tile_size>(bg);

	__shared__ quasi_newton_aux_gpu<Size> f_tile_sh[(Size::atoms_size / 32 + 1) * 32 / tile_size];
	__shared__ change_gpu<Size> g_tile_sh[(Size::atoms_size / 32 + 1) * 32 / tile_size];
	if(tile.thread_rank() == 0) f_tile_sh[tile.meta_group_rank()] = f;
	if(tile.thread_rank() == 0) g_tile_sh[tile.meta_group_rank()] = g;
	tile.sync();

	for(int trial = tile.meta_group_rank(); trial < max_trials; trial += tile.meta_group_size()) {
		alpha = cuda::std::powf(multiplier, trial);
		x_tmp = x;
		x_tmp.increment(p, alpha);
		f1 = f_tile_sh[tile.meta_group_rank()].operator()<tile_size>(
			x_tmp,
			g_tile_sh[tile.meta_group_rank()],
			tile,
			temp_storage_fl + tile.meta_group_rank()*tile_size,
			0
		);
		evalcount++;

		tile.sync();

		if(f1 - f0 < c0 * alpha * pg) {
			if(tile.thread_rank() == 0) temp_storage_int[trial] = trial;
			break;
		}
	}
	__syncthreads();

	int min_trial = block_min(temp_storage_int, max_trials);
	if(tile.meta_group_rank() == min_trial % tile.meta_group_size()) {
		if(tile.thread_rank() == 0) temp_storage_fl[0] = alpha;
		if(tile.thread_rank() == 0) temp_storage_fl[1] = f1;
		if(tile.thread_rank() == 0) x_new = x_tmp;
		if(tile.thread_rank() == 0) g_new = g_tile_sh[tile.meta_group_rank()];
	}
	__syncthreads();

	alpha = temp_storage_fl[0];
	f1 = temp_storage_fl[1];

	return alpha;
}

template<class Size>
__device__ void subtract_change_gpu(
    change_gpu<Size> & b,
    const change_gpu<Size> & a,
    sz n
) { // b -= a
	for(int i = 0; i < n; i++)
		b(i) -= a(i);
}

template<class Size>
__device__ void subtract_change_gpu(
    change_gpu<Size> & b,
    const change_gpu<Size> & a,
    sz n,
	int async
) { // b -= a
	for(int i = threadIdx.x; i < n; i += blockDim.x)
		b(i) -= a(i);
}

template<class Size>
__device__ fl bfgs_gpu(
    quasi_newton_aux_gpu<Size> & f,
    conf_gpu<Size> & x,
    change_gpu<Size> & g,
    const unsigned max_steps,
    const fl average_required_improvement,
    const sz over,
	int& evalcount
) { // x is I/O, final value is returned
	sz n = g.num_floats();
	triangular_matrix_gpu<fl, Size::torsions_size + 6> h;
	h.init(n, 0);
	set_diagonal_gpu(h, 1);

	change_gpu<Size> g_new;
	g_new.init(g);
	conf_gpu<Size> x_new;
	x_new.init(x);
	// std::cout << "bfgs_gpu, tag: 1" << std::endl;
	fl f0 = f(x, g);
	// std::cout << "bfgs_gpu, tag: 2" << std::endl;
	evalcount++;

	fl f_orig = f0;
	change_gpu<Size> g_orig;
	g_orig.init(g);
	conf_gpu<Size> x_orig;
	x_orig.init(x);
	change_gpu<Size> p;
	p.init(g);

	for(int step = 0; step < max_steps; step++) {
		minus_mat_vec_product_gpu(h, g, p);
		fl f1 = 0;
		const fl alpha =
			line_search_gpu(f, n, x, g, f0, p, x_new, g_new, f1, evalcount, 0);
		change_gpu<Size> y;
		y.init(g_new);
		subtract_change_gpu(y, g, n);

		f0 = f1;
		x = x_new;
		if(!(std::sqrt(scalar_product_gpu(g, g, n)) >= 1e-5)) break; // breaks for nans too // FIXME !!?? 
		g = g_new; // ?

		if(step == 0) {
			const fl yy = scalar_product_gpu(y, y, n);
			if(std::abs(yy) > epsilon_fl_gpu)
				set_diagonal_gpu(h, alpha * scalar_product_gpu(y, p, n) / yy);
		}

		bool h_updated = bfgs_update_gpu(h, p, y, alpha);
	}
	if(!(f0 <= f_orig)) { // succeeds for nans too
		f0 = f_orig;
		x = x_orig;
		g = g_orig;
	}
	return f0;
}

template<class Size>
__device__ fl bfgs_gpu(
    quasi_newton_aux_gpu<Size> & f,
    conf_gpu<Size> & x,
    change_gpu<Size> & g,
    const unsigned max_steps,
    const fl average_required_improvement,
    const sz over,
	int & evalcount,
	int async
) { // x is I/O, final value is returned
	sz n = g.num_floats();

	fl f0;
	f0 = f(x, g);
	evalcount++;
	__syncthreads();

	__shared__ conf_gpu<Size> x_shared;
	__shared__ change_gpu<Size> g_shared;
	__shared__ fl f0_shared;
	if(threadIdx.x == 0) x_shared = x;
	if(threadIdx.x == 0) g_shared = g;
	if(threadIdx.x == 0) f0_shared = f0;
	__syncthreads();

	fl f_orig = f0_shared;
	change_gpu<Size> g_orig;
	g_orig.init(g_shared);
	conf_gpu<Size> x_orig;
	x_orig.init(x_shared);

	__shared__ change_gpu<Size> g_new;
	__shared__ conf_gpu<Size> x_new;
	__shared__ change_gpu<Size> p;
	g_new.init(g_shared, 0);
	x_new.init(x_shared, 0);
	p.init(g_shared, 0);

	__shared__ triangular_matrix_gpu<fl, Size::torsions_size + 6> h;
	h.init(n, 0, 0);
	__syncthreads();
	set_diagonal_gpu(h, 1, 0);
	__syncthreads();

	__shared__ fl temp_storage_fl[(Size::atoms_size / 32 + 1) * 32];
	__shared__ int temp_storage_int[(Size::atoms_size / 32 + 1) * 32];
	__shared__ change_gpu<Size> y;
	__shared__ change_gpu<Size> minus_hy;
	if(threadIdx.x == 0 && blockIdx.x == 0) printf("[SH] x_shared=%p g_shared=%p g_new=%p x_new=%p p=%p h=%p temp_fl=%p temp_int=%p y=%p mhy=%p\n", &x_shared, &g_shared, &g_new, &x_new, &p, &h, temp_storage_fl, temp_storage_int, &y, &minus_hy);

	for(int step = 0; step < max_steps; step++) {
		minus_mat_vec_product_gpu(h, g_shared, p, 0);
		__syncthreads();

		fl f1 = 0;
		fl alpha = line_search_gpu<Size, 8>(
			f, n, x_shared, g_shared, f0_shared, p, x_new, g_new, f1, evalcount,
			temp_storage_fl, temp_storage_int, 0
		);
		__syncthreads();

		
		y.init(g_new, 0);
		__syncthreads();

		subtract_change_gpu(y, g_shared, n, 0);

		f0_shared = f1;
		x_shared.init(x_new, 0);
		if(!(std::sqrt(scalar_product_gpu(g_shared, g_shared, n, temp_storage_fl, 0)) >= 1e-5)) break; // breaks for nans too // FIXME !!?? 
		g_shared.init(g_new, 0); // ?
		__syncthreads();

		if(step == 0) {
			const fl yy = scalar_product_gpu(y, y, n, temp_storage_fl, 0);
			__syncthreads();

			if(std::abs(yy) > epsilon_fl_gpu)
				set_diagonal_gpu(
					h,
					alpha * scalar_product_gpu(y, p, n, temp_storage_fl, 0) / yy,
					0
				);
			__syncthreads();
		}

		bool h_updated = bfgs_update_gpu(h, p, y, alpha, temp_storage_fl, minus_hy, 0);
		__syncthreads();
	}
	if(!(f0_shared <= f_orig)) { // succeeds for nans too
		if(threadIdx.x == 0) f0_shared = f_orig;
		if(threadIdx.x == 0) x_shared = x_orig;
		if(threadIdx.x == 0) g_shared = g_orig;
		__syncthreads();
	}

	x = x_shared;
	g = g_shared;

	return f0_shared;
}

template<>
__device__ void quasi_newton_gpu::operator()<Small>(
    model_gpu<Small> & m,
    const precalculate_byatom_gpu<Small> & p,
    const cache_gpu<Small> & ig,
    output_type_gpu<Small> & out,
    change_gpu<Small> & g,
    const vec_gpu & v,
    int & evalcount
) const { // g must have correct size
    quasi_newton_aux_gpu<Small> aux;
    aux.init(&m, &p, &ig, v);
	
    fl res = bfgs_gpu(aux, out.c, g, max_steps, average_required_improvement, 10, evalcount, 0);
	
    // Update model a last time after optimization
    m.set(out.c);
    out.e = res;
}

__device__ void quasi_newton_gpu::init() {
	max_steps = 1000;
	average_required_improvement = 0.0;
}
