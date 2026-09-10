#ifndef PRECALCULATE_GPU_H_
#define PRECALCULATE_GPU_H_

#include <cuda/std/utility>
#include <cuda/std/cassert>
#include "size.h"

typedef double fl;
typedef unsigned long sz;

template<class T, int size>
struct triangular_matrix_gpu {
    T m_data[size * (size + 1) / 2];
    sz m_data_size;
    sz m_dim;

    __device__ inline const T & operator()(
        sz i,
        sz j
    ) const;

    __device__ inline T & operator()(
        sz i,
        sz j
    );

    __device__ inline void init(int _m_dim, int val);
    __device__ inline void init(int _m_dim, int val, int async);
};

template<class T, int size>
__device__ inline void triangular_matrix_gpu<T, size>::init(int _m_dim, int val) {
    m_dim = _m_dim;
    m_data_size = (m_dim + 1) * m_dim / 2;
    for(int i = 0; i < m_data_size; i++)
        m_data[i] = val;
}

template<class T, int size>
__device__ inline void triangular_matrix_gpu<T, size>::init(int _m_dim, int val, int async) {
    if(threadIdx.x == 0) m_dim = _m_dim;
    if(threadIdx.x == 0) m_data_size = (m_dim + 1) * m_dim / 2;
    int loop_size = (_m_dim + 1) * _m_dim / 2;
    for(int i = threadIdx.x; i < loop_size; i += blockDim.x)
        m_data[i] = val;
}

template<class T, int size>
__device__ inline const T & triangular_matrix_gpu<T, size>::operator()(
    sz i,
    sz j
) const {
    assert(j < m_dim && i < m_dim);
	if(i <= j)
        return m_data[i + j*(j+1)/2];
    else
        return m_data[j + i*(i+1)/2];
}

template<class T, int size>
__device__ inline T & triangular_matrix_gpu<T, size>::operator()(
    sz i,
    sz j
) {
    assert(j < m_dim && i < m_dim);
	if(i <= j)
        return m_data[i + j*(j+1)/2];
    else
        return m_data[j + i*(i+1)/2];
}

struct precalculate_element_gpu {
    fl factor;
    cuda::std::pair<fl ,fl> * smooth;
    int smooth_size;

    __device__ inline cuda::std::pair<fl ,fl> eval_deriv(fl r2) const;
};

__device__ inline cuda::std::pair<fl ,fl> precalculate_element_gpu::eval_deriv(
    fl r2
) const {
    fl r2_factored = factor * r2;
    assert(r2_factored + 1 < smooth_size);
    sz i1 = sz(r2_factored);
    sz i2 = i1 + 1; // r2 is expected < cutoff_sqr, and cutoff_sqr * factor + 1 < n, so no overflow
    assert(i1 < smooth_size);
    assert(i2 < smooth_size);
    fl rem = r2_factored - i1;
    assert(rem >= -epsilon_fl_gpu);
    assert(rem < 1 + epsilon_fl_gpu);
    const cuda::std::pair<fl ,fl> & p1 = smooth[i1];
    const cuda::std::pair<fl ,fl> & p2 = smooth[i2];
    fl e = p1.first + rem * (p2.first - p1.first);
    fl dor = p1.second + rem * (p2.second - p1.second);
    return cuda::std::pair<fl ,fl>(e, dor);
};

template<class Size>
struct precalculate_byatom_gpu {
    fl m_cutoff_sqr;
    fl m_max_cutoff_sqr;
    triangular_matrix_gpu<precalculate_element_gpu, Size::atoms_size> m_data;

    __device__ inline fl cutoff_sqr() const;
    __device__ inline fl max_cutoff_sqr() const;
    __device__ inline cuda::std::pair<fl ,fl> eval_deriv(sz i, sz j, fl r2) const;
};

template<class Size>
__device__ inline fl precalculate_byatom_gpu<Size>::cutoff_sqr() const {
    return m_cutoff_sqr;
}

template<class Size>
__device__ inline fl precalculate_byatom_gpu<Size>::max_cutoff_sqr() const {
    return m_max_cutoff_sqr;
}

template<class Size>
__device__ inline cuda::std::pair<fl ,fl> precalculate_byatom_gpu<Size>::eval_deriv(
    sz i,
    sz j,
    fl r2
) const {
    assert(r2 <= m_max_cutoff_sqr);
    return m_data(i, j).eval_deriv(r2);
};

#endif