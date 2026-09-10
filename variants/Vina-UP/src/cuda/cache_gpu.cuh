#ifndef CACHE_GPU_H_
#define CACHE_GPU_H_

#include "size.h"
#include "common_gpu.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

typedef double fl;

struct atom_gpu;

struct array_3d_gpu {
    unsigned int m_i;
    unsigned int m_j;
    unsigned int m_k;
    fl * m_data;
    unsigned int m_data_size;

    __device__ inline sz dim(sz i) const;
    
    __device__ inline const fl & operator()(
        sz i,
        sz j,
        sz k
    ) const;
};

__device__ inline sz array_3d_gpu::dim(sz i) const {
	switch(i) {
		case 0: return m_i;
		case 1: return m_j;
		case 2: return m_k;
		default: assert(false); return 0; // to get rid of the warning
	}
}

__device__ inline const fl & array_3d_gpu::operator()(
    sz i,
    sz j,
    sz k
) const {
    return m_data[i + m_i*(j + m_j*k)];
}


struct grid_gpu {
    vec_gpu m_init;
    vec_gpu m_factor;
    vec_gpu m_dim_fl_minus_1;
    array_3d_gpu m_data;
    vec_gpu m_factor_inv;

    __device__ fl evaluate(
        const vec_gpu & location,
        fl slope,
        fl c,
        vec_gpu * deriv
    ) const;
};

template<class Size>
struct cache_gpu {
    grid_gpu m_grids[Size::atom_types_size];
    int m_grids_size;
    fl m_slope;

    __device__ fl eval_deriv(
	    atom_gpu * m_atoms,
	    vec_gpu * m_coords,
	    vec_gpu * m_minus_forces,
	    int m_atoms_size,
	    fl v
    ) const;

    template<int tile_size>
    __device__ fl eval_deriv(
	    atom_gpu * m_atoms,
	    vec_gpu * m_coords,
	    vec_gpu * m_minus_forces,
	    int m_atoms_size,
	    fl v,
        cg::thread_block_tile<tile_size> & tile,
        fl * temp_storage_fl,
        int async
    ) const;
};

template<>
struct cache_gpu<Small> {
    grid_gpu m_grids[Small::atom_types_size];
    int m_grids_size;
    fl m_slope;

    __device__ fl eval_deriv(
	    atom_gpu * m_atoms,
	    vec_gpu * m_coords,
	    vec_gpu * m_minus_forces,
	    int m_atoms_size,
	    fl v
    ) const;
    
    template<int tile_size>
    __device__ fl eval_deriv(
	    atom_gpu * m_atoms,
	    vec_gpu * m_coords,
	    vec_gpu * m_minus_forces,
	    int m_atoms_size,
	    fl v,
        cg::thread_block_tile<tile_size> & tile,
        fl * temp_storage_fl,
        int async
    ) const;
};


template<int tile_size>
__device__ fl cache_gpu<Small>::eval_deriv(
    atom_gpu * m_atoms,
    vec_gpu * m_coords,
    vec_gpu * m_minus_forces,
    int m_atoms_size,
    fl v,
    cg::thread_block_tile<tile_size> & tile,
    fl * temp_storage_fl,
    int async
) const { // needs m.coords, sets m.minus_forces
	fl e = 0;
	sz nat = num_atom_types(atom_gpu::XS);

	for(int i = tile.thread_rank(); i < m_atoms_size; i += tile.num_threads()) {
		const atom_gpu & a = m_atoms[i];
		sz t = a.get(atom_gpu::XS);

		if (t >= nat) { m_minus_forces[i].assign(0); continue; }
		switch (t)
		{
			case XS_TYPE_G0_GPU:
			case XS_TYPE_G1_GPU:
			case XS_TYPE_G2_GPU:
			case XS_TYPE_G3_GPU:
				m_minus_forces[i].assign(0);
				continue;
			case XS_TYPE_C_H_CG0_GPU:
			case XS_TYPE_C_H_CG1_GPU:
			case XS_TYPE_C_H_CG2_GPU:
			case XS_TYPE_C_H_CG3_GPU:
				t = XS_TYPE_C_H_GPU;
				break;
			case XS_TYPE_C_P_CG0_GPU:
			case XS_TYPE_C_P_CG1_GPU:
			case XS_TYPE_C_P_CG2_GPU:
			case XS_TYPE_C_P_CG3_GPU:
				t = XS_TYPE_C_P_GPU;
				break;
		}

		vec_gpu deriv;
		const grid_gpu & g = m_grids[t];
		e += g.evaluate(m_coords[i], m_slope, v, &deriv);
		m_minus_forces[i] = deriv;
	}
	tile.sync();

	temp_storage_fl[tile.thread_rank()] = e;
	tile.sync();

	fl e_sum = tile_sum<tile_size>(tile, temp_storage_fl,
		m_atoms_size > tile.num_threads() ? tile.num_threads() : m_atoms_size);
	
	return e_sum;
}

#endif