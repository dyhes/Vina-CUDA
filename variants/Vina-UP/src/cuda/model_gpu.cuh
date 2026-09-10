#ifndef MODEL_GPU_H_
#define MODEL_GPU_H_

#include "size.h"
#include "cache_gpu.cuh"
#include "precalculate_gpu.cuh"
#include "output_type_gpu.cuh"
#include "common_gpu.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
struct interacting_pair_gpu {
	sz a;
	sz b;
};

struct frame_gpu {
    vec_gpu origin;
    mat_gpu orientation_m;
    qt_gpu orientation_q;
    int atom_begin;
    int atom_end;

    __device__ inline vec_gpu local_to_lab(const vec_gpu & local_coords) const;
    __device__ inline vec_gpu local_to_lab_direction(const vec_gpu & local_direction) const;
    __device__ inline void set_orientation(const qt_gpu & q);
    __device__ inline void set_coords(const atom_gpu * atoms, vec_gpu * coords) const;
    template<int tile_size>
    __device__ inline void set_coords(const atom_gpu * atoms, vec_gpu * coords, cg::thread_block_tile<tile_size> & tile, int async) const;
    __device__ inline cuda::std::pair<vec_gpu, vec_gpu> sum_force_and_torque(const vec_gpu * coords, const vec_gpu * forces) const;
    template<int tile_size>
    __device__ inline cuda::std::pair<vec_gpu, vec_gpu> sum_force_and_torque(const vec_gpu * coords, const vec_gpu * forces, cg::thread_block_tile<tile_size> & tile, fl * temp_storage_fl, int async) const;
};

__device__ inline vec_gpu frame_gpu::local_to_lab(const vec_gpu & local_coords) const {
	vec_gpu tmp;
	tmp = origin + orientation_m*local_coords; 
	return tmp;
}

__device__ inline vec_gpu frame_gpu::local_to_lab_direction(const vec_gpu & local_direction) const {
	vec_gpu tmp;
	tmp = orientation_m * local_direction;
	return tmp;
}

__device__ inline void frame_gpu::set_orientation(const qt_gpu & q) { // does not normalize the orientation
	orientation_q = q;
	orientation_m = quaternion_to_r3_gpu(orientation_q);
}

__device__ inline void frame_gpu::set_coords(
    const atom_gpu * atoms,
    vec_gpu * coords
) const {
	for(int i = atom_begin; i < atom_end; i++)
		coords[i] = local_to_lab(atoms[i].coords);
}

template<int tile_size>
__device__ inline void frame_gpu::set_coords(
    const atom_gpu * atoms,
    vec_gpu * coords,
    cg::thread_block_tile<tile_size> & tile,
    int async
) const {
    for(int i = tile.thread_rank(); i < atom_end - atom_begin; i += tile.num_threads())
        coords[i + atom_begin] = local_to_lab(atoms[i + atom_begin].coords);
}

__device__ inline cuda::std::pair<vec_gpu, vec_gpu> frame_gpu::sum_force_and_torque(
    const vec_gpu * coords,
    const vec_gpu * forces
) const {
	cuda::std::pair<vec_gpu, vec_gpu> tmp;
	tmp.first.assign(0);
	tmp.second.assign(0);
	for(int i = atom_begin; i < atom_end; i++) {
		tmp.first  += forces[i]; 
		tmp.second += cross_product_gpu(coords[i] - origin, forces[i]);
	}
	return tmp;
}

template<int tile_size>
__device__ inline cuda::std::pair<vec_gpu, vec_gpu> frame_gpu::sum_force_and_torque(
    const vec_gpu * coords,
    const vec_gpu * forces,
    cg::thread_block_tile<tile_size> & tile,
    fl * temp_storage_fl,
    int async
) const {
	cuda::std::pair<vec_gpu, vec_gpu> tmp;
	tmp.first.assign(0);
	tmp.second.assign(0);
	for(int i = tile.thread_rank(); i < atom_end - atom_begin; i += tile.num_threads()) {
		tmp.first  += forces[i + atom_begin]; 
		tmp.second += cross_product_gpu(coords[i + atom_begin] - origin, forces[i + atom_begin]);
	}
    tile.sync();

    cuda::std::pair<vec_gpu, vec_gpu> t_sum;
    for(int i = 0; i < 3; i++) {
        temp_storage_fl[tile.thread_rank()] = tmp.first[i];
        tile.sync();
        t_sum.first[i] = tile_sum<tile_size>(tile, temp_storage_fl, 
            atom_end - atom_begin > tile.num_threads() ? tile.num_threads() : atom_end - atom_begin);
        tile.sync();
    }

    for(int i = 0; i < 3; i++) {
        temp_storage_fl[tile.thread_rank()] = tmp.second[i];
        tile.sync();
        t_sum.second[i] = tile_sum<tile_size>(tile, temp_storage_fl,
            atom_end - atom_begin > tile.num_threads() ? tile.num_threads() : atom_end - atom_begin);
        tile.sync();
    }

	return t_sum;
}

struct rigid_body_gpu {
    frame_gpu frame;
    
    __device__ inline void set_conf(
        const atom_gpu * atoms,
        vec_gpu * coords,
        const rigid_conf_gpu & c
    );

    template<int tile_size>
    __device__ inline void set_conf(
        const atom_gpu * atoms,
        vec_gpu * coords,
        const rigid_conf_gpu & c,
        cg::thread_block_tile<tile_size> & tile,
        int async
    );

    __device__ inline void set_derivative(
        const cuda::std::pair<vec_gpu, vec_gpu>& force_torque,
        rigid_change_gpu & c
    ) const;
};

__device__ inline void rigid_body_gpu::set_conf(
    const atom_gpu * atoms,
    vec_gpu * coords,
    const rigid_conf_gpu & c
) {
    frame.origin = c.position;
	frame.set_orientation(c.orientation);
	frame.set_coords(atoms, coords);
}

template<int tile_size>
__device__ inline void rigid_body_gpu::set_conf(
    const atom_gpu * atoms,
    vec_gpu * coords,
    const rigid_conf_gpu & c,
    cg::thread_block_tile<tile_size> & tile,
    int async
) {
    if(tile.thread_rank() == 0) frame.origin = c.position;
	if(tile.thread_rank() == 0) frame.set_orientation(c.orientation);
    tile.sync();

	frame.set_coords<tile_size>(atoms, coords, tile, async);
}

__device__ inline void rigid_body_gpu::set_derivative(
    const cuda::std::pair<vec_gpu, vec_gpu> & force_torque,
    rigid_change_gpu & c
) const{
	c.position     = force_torque.first;
	c.orientation  = force_torque.second;
}

struct segment_gpu {
    frame_gpu frame;
    vec_gpu relative_axis;
	vec_gpu relative_origin;
    vec_gpu axis;

    __device__ inline void set_conf(
        const frame_gpu & parent,
        const atom_gpu * atoms,
        vec_gpu * coords,
        const fl * c
    );

    __device__ inline void set_derivative(
        const cuda::std::pair<vec_gpu, vec_gpu> & force_torque,
        fl & c
    ) const;
};

__device__ inline void segment_gpu::set_conf(
    const frame_gpu & parent,
    const atom_gpu * atoms,
    vec_gpu * coords,
    const fl * c
)  {
	const fl torsion = *c;
	frame.origin = parent.local_to_lab(relative_origin);
	axis = parent.local_to_lab_direction(relative_axis);
	qt_gpu tmp = angle_to_quaternion_gpu(axis, torsion) * parent.orientation_q;
	quaternion_normalize_approx_gpu(tmp); // normalization added in 1.1.2
	//quaternion_normalize(tmp); // normalization added in 1.1.2
	frame.set_orientation(tmp);
	frame.set_coords(atoms, coords);
}

__device__ inline void segment_gpu::set_derivative(
    const cuda::std::pair<vec_gpu, vec_gpu> & force_torque,
    fl & c
) const {
	c = force_torque.second * axis;
}

struct tree_gpu {
    segment_gpu node;
    int ch_begin;
    int ch_end;
    frame_gpu * parent;

    __device__ inline void set_conf(
        const atom_gpu * atoms,
        vec_gpu * coords,
        const fl * c
    );

    __device__ inline cuda::std::pair<vec_gpu, vec_gpu> derivative(
        const vec_gpu * coords,
        const vec_gpu * forces
    ) const;
};

__device__ void branches_set_conf_gpu(
	tree_gpu * tree_nodes,
	cuda::std::pair<int, int> * ch_BE,
    int ch_BE_size,
	const atom_gpu * atoms,
	vec_gpu * coords,
	const fl * c
);

template<int tile_size>
__device__ void branches_set_conf_gpu(
	tree_gpu * tree_nodes,
	cuda::std::pair<int, int> * ch_BE,
    int ch_BE_size,
	const atom_gpu * atoms,
	vec_gpu * coords,
	const fl * c,
    cg::thread_block_tile<tile_size> & tile,
    int async
) {
	for(int l = 0; l < ch_BE_size; l++) {
		int n_begin = ch_BE[l].first; int n_end = ch_BE[l].second;

		for(int i = tile.thread_rank(); i < n_end - n_begin; i += tile.num_threads()) {
			const fl * ci = c + i + n_begin;
			tree_nodes[i + n_begin].set_conf(atoms, coords, ci);
		}
        tile.sync();
	}
}

__device__ void branches_derivative_gpu(
    const tree_gpu * tree_nodes,
    int tree_nodes_size,
    const vec_gpu * coords,
    const vec_gpu * forces,
    cuda::std::pair<vec_gpu, vec_gpu> * nodes_force_torque,
    const cuda::std::pair<int, int> * ch_BE,
    const int ch_BE_size,
    fl * d
);

template<int tile_size>
__device__ void branches_derivative_gpu(
    const tree_gpu * tree_nodes,
    int tree_nodes_size,
    const vec_gpu * coords,
    const vec_gpu * forces,
    cuda::std::pair<vec_gpu, vec_gpu> * nodes_force_torque,
    const cuda::std::pair<int, int> * ch_BE,
    const int ch_BE_size,
    fl * d,
    cg::thread_block_tile<tile_size> & tile,
    int async
) { // adds to out
	for(int i = tile.thread_rank(); i < tree_nodes_size; i += tile.num_threads())
		nodes_force_torque[i] = tree_nodes[i].derivative(coords, forces);
	tile.sync();

	for(int l = ch_BE_size - 1; l >= 0; l--) {
		int n_begin = ch_BE[l].first; int n_end = ch_BE[l].second;
		for(int j = tile.thread_rank(); j < n_end - n_begin; j += tile.num_threads()) {
			for(int ch = tree_nodes[j + n_begin].ch_begin; ch < tree_nodes[j + n_begin].ch_end; ch++) {
				nodes_force_torque[j + n_begin].first += nodes_force_torque[ch].first;

				vec_gpu r;
        		r = tree_nodes[ch].node.frame.origin - (*tree_nodes[ch].parent).origin;
				
				nodes_force_torque[j + n_begin].second +=
					cross_product_gpu(r, nodes_force_torque[ch].first) +
					nodes_force_torque[ch].second;
			}
			fl * p = d + j + n_begin;
			tree_nodes[j + n_begin].node.set_derivative(nodes_force_torque[j + n_begin], *p);
		}
		tile.sync();
	}
}

__device__ inline void tree_gpu::set_conf(
    const atom_gpu * atoms,
    vec_gpu * coords,
    const fl * c
) {
    node.set_conf(*parent, atoms, coords, c);
}

__device__ inline cuda::std::pair<vec_gpu, vec_gpu> tree_gpu::derivative(
    const vec_gpu * coords,
    const vec_gpu * forces
) const {
	cuda::std::pair<vec_gpu, vec_gpu> force_torque = node.frame.sum_force_and_torque(coords, forces);
	return force_torque;
}

template<class Size>
struct heterotree_gpu {
    rigid_body_gpu node;
    cuda::std::pair<vec_gpu, vec_gpu> nodes_force_torque[Size::torsions_size];
    tree_gpu tree_nodes[Size::torsions_size];
    int tree_nodes_size;
    int ch_begin;
    int ch_end;
    cuda::std::pair<int, int> ch_BE[Size::torsions_size];
    int ch_BE_size;

    __device__ inline void set_conf(
        const atom_gpu * atoms,
        vec_gpu * coords,
        const ligand_conf_gpu<Size> & c
    );

    template<int tile_size>
    __device__ inline void set_conf(
        const atom_gpu * atoms,
        vec_gpu * coords,
        const ligand_conf_gpu<Size> & c,
        cg::thread_block_tile<tile_size> & tile,
        int async
    );

    __device__ inline void derivative(
        const vec_gpu * coords,
        const vec_gpu * forces,
        ligand_change_gpu<Size> & c
    ) const;

    template<int tile_size>
    __device__ inline void derivative(
        const vec_gpu * coords,
        const vec_gpu * forces,
        ligand_change_gpu<Size> & c,
        cg::thread_block_tile<tile_size> & tile,
        fl * temp_storage_fl,
        int async
    );
};

template<class Size>
__device__ inline void heterotree_gpu<Size>::set_conf(
    const atom_gpu * atoms,
    vec_gpu * coords,
    const ligand_conf_gpu<Size> & c
) {
    node.set_conf(atoms, coords, c.rigid);
	branches_set_conf_gpu(
    	tree_nodes,
        ch_BE,
        ch_BE_size,
		atoms,
		coords,
		c.torsions
    );
}

template<class Size>
template<int tile_size>
__device__ inline void heterotree_gpu<Size>::set_conf(
    const atom_gpu * atoms,
    vec_gpu * coords,
    const ligand_conf_gpu<Size> & c,
    cg::thread_block_tile<tile_size> & tile,
    int async
) {
    node.set_conf<tile_size>(atoms, coords, c.rigid, tile, 0);
    tile.sync();
	branches_set_conf_gpu<tile_size>(
    	tree_nodes,
        ch_BE,
        ch_BE_size,
		atoms,
		coords,
		c.torsions,
        tile,
        0
    );
}

template<class Size>
__device__ inline void heterotree_gpu<Size>::derivative(
    const vec_gpu * coords,
    const vec_gpu * forces,
    ligand_change_gpu<Size> & c
) const {
    cuda::std::pair<vec_gpu, vec_gpu> nodes_force_torque[Size::torsions_size];
	fl * p = c.torsions;
    branches_derivative_gpu(
        tree_nodes,
        tree_nodes_size,
        coords,
        forces,
        nodes_force_torque,
        ch_BE,
        ch_BE_size,
        p
    );

    cuda::std::pair<vec_gpu, vec_gpu> force_torque = node.frame.sum_force_and_torque(coords, forces);
    for(int i = ch_begin; i < ch_end; i++) {
        force_torque.first += nodes_force_torque[i].first;
        
        vec_gpu r;
        r = tree_nodes[i].node.frame.origin - node.frame.origin;
				
		force_torque.second +=
		    cross_product_gpu(r, nodes_force_torque[i].first) +
		    nodes_force_torque[i].second;
    }
    
	node.set_derivative(force_torque, c.rigid);
}

template<class Size>
template<int tile_size>
__device__ inline void heterotree_gpu<Size>::derivative(
    const vec_gpu * coords,
    const vec_gpu * forces,
    ligand_change_gpu<Size> & c,
    cg::thread_block_tile<tile_size> & tile,
    fl * temp_storage_fl,
    int async
) {
	fl * p = c.torsions;
    branches_derivative_gpu<tile_size>(
        tree_nodes,
        tree_nodes_size,
        coords,
        forces,
        nodes_force_torque,
        ch_BE,
        ch_BE_size,
        p,
        tile,
        0
    );
    tile.sync();

    cuda::std::pair<vec_gpu, vec_gpu> force_torque = node.frame.sum_force_and_torque<tile_size>(coords, forces, tile, temp_storage_fl, 0);
    if(tile.thread_rank() != 0) {
        force_torque.first.assign(0);
        force_torque.second.assign(0);
    }
    tile.sync();

    for(int i = tile.thread_rank(); i < ch_end - ch_begin; i += tile.num_threads()) {
        force_torque.first += nodes_force_torque[i + ch_begin].first;
        
        vec_gpu r;
        r = tree_nodes[i + ch_begin].node.frame.origin - node.frame.origin;
				
		force_torque.second +=
		    cross_product_gpu(r, nodes_force_torque[i + ch_begin].first) +
		    nodes_force_torque[i + ch_begin].second;
    }
    tile.sync();

    for(int i = 0; i < 3; i++) {
        temp_storage_fl[tile.thread_rank()] = force_torque.first[i];
        tile.sync();
        force_torque.first[i] = tile_sum<tile_size>(tile, temp_storage_fl,
            ch_begin - ch_end > tile.num_threads() ? tile.num_threads() : ch_begin - ch_end);
        tile.sync();
    }

    for(int i = 0; i < 3; i++) {
        temp_storage_fl[tile.thread_rank()] = force_torque.second[i];
        tile.sync();
        force_torque.second[i] = tile_sum<tile_size>(tile, temp_storage_fl,
            ch_begin - ch_end > tile.num_threads() ? tile.num_threads() : ch_begin - ch_end);
        tile.sync();
    }
    
	if(tile.thread_rank() == 0) node.set_derivative(force_torque, c.rigid);
    tile.sync();
}

template<class Size>
struct ligand_gpu {
    heterotree_gpu<Size> flexible_body;
    int atom_begin;
    int atom_end;

    interacting_pair_gpu pairs[Size::pairs_size];
    int pairs_size;

    int one_atom_pairs[Size::atoms_size];
    int one_atom_pairs_size;
};

template<class Size>
struct model_gpu {
    ligand_gpu<Size> ligand;
    atom_gpu atoms[Size::atoms_size];       int atoms_size;
    vec_gpu coords[Size::atoms_size];
    vec_gpu minus_forces[Size::atoms_size];

    __device__ void set(const conf_gpu<Size> & c);

    template<int tile_size>
    __device__ void set(const conf_gpu<Size> & c, cg::thread_block_tile<tile_size> & tile, int async);

    __device__ fl eval_deriv(
        const precalculate_byatom_gpu<Size> & p,
        const cache_gpu<Size> & ig,
        const vec_gpu & v,
        change_gpu<Size> & g
    );
    
    template<int tile_size>
    __device__ fl eval_deriv(
        const precalculate_byatom_gpu<Size> & p,
        const cache_gpu<Size> & ig,
        const vec_gpu & v,
        change_gpu<Size> & g,
        cg::thread_block_tile<tile_size> & tile,
        fl * temp_storage_fl,
        int async
    );

    __device__ void get_heavy_atom_movable_coords(output_type_gpu<Size> & tmp) const;

    __device__ fl gyration_radius() const;
};

template<class Size>
__device__ void model_gpu<Size>::set(const conf_gpu<Size> & c) {
	ligand.flexible_body.set_conf(atoms, coords, c.ligand);
}

template<class Size>
template<int tile_size>
__device__ void model_gpu<Size>::set(const conf_gpu<Size> & c, cg::thread_block_tile<tile_size> & tile, int async) {
	ligand.flexible_body.set_conf<tile_size>(atoms, coords, c.ligand, tile, 0);
}

template<class Size>
__device__ fl eval_interacting_pairs_deriv_gpu(
    const precalculate_byatom_gpu<Size> & p,
    fl v,
    const interacting_pair_gpu * pairs,
    int pairs_size,
    const vec_gpu * coords,
    vec_gpu * forces,
    const bool with_max_cutoff=false
) { // adds to forces  // clean up
	fl e = 0;
	fl cutoff_sqr = p.cutoff_sqr();

	if (with_max_cutoff)
		cutoff_sqr = p.max_cutoff_sqr();

	for( int i = 0; i < pairs_size; i++) {
		const interacting_pair_gpu & ip = pairs[i];
		vec_gpu r = coords[ip.b] - coords[ip.a]; // a -> b
		fl r2 = r.norm_sqr();
		if(r2 < cutoff_sqr) {
			cuda::std::pair<fl, fl> tmp = p.eval_deriv(ip.a, ip.b, r2);
			vec_gpu force; force = tmp.second * r;
			curl_gpu(tmp.first, force, v);
			e += tmp.first;

			// FIXME inefficient, if using hard curl
			forces[ip.a] -= force; // we could omit forces on inflex here
			forces[ip.b] += force;
		}
	}
	return e;
}

template<class Size, int tile_size>
__device__ fl eval_interacting_pairs_deriv_gpu(
    const precalculate_byatom_gpu<Size> & p,
    fl v,
    const interacting_pair_gpu * pairs,
    int pairs_size,
    const vec_gpu * coords,
    vec_gpu * forces,
    cg::thread_block_tile<tile_size> & tile,
    fl * temp_storage_fl,
    int * one_atom_pairs,
    int one_atom_pairs_size,
    int async,
    const bool with_max_cutoff=false
) { // adds to forces  // clean up
	fl e = 0;
	fl cutoff_sqr = p.cutoff_sqr();

	if (with_max_cutoff)
		cutoff_sqr = p.max_cutoff_sqr();
    
    int step = 0;
    int pair_begin = 0;
    while(step < one_atom_pairs_size) {
        int pair_end = one_atom_pairs[step];

        vec_gpu force_a;
        force_a.assign(0);

        for(int i = tile.thread_rank(); i < pair_end - pair_begin; i += tile.num_threads()) {
            const interacting_pair_gpu & ip = pairs[i + pair_begin];
            vec_gpu r = coords[ip.b] - coords[ip.a];
            fl r2 = r.norm_sqr();

            if(r2 < cutoff_sqr) {
                cuda::std::pair<fl, fl> tmp = p.eval_deriv(ip.a, ip.b, r2);
			    vec_gpu force;
                force = tmp.second * r;
			    curl_gpu(tmp.first, force, v);
			    e += tmp.first;

			    // FIXME inefficient, if using hard curl
                force_a -= force;
			    // forces[ip.a] -= force; // we could omit forces on inflex here
			    forces[ip.b] += force;
            }
        }
        tile.sync();

        sz ipa = pairs[pair_begin].a;
        if(tile.thread_rank() == 0) force_a += forces[ipa];
        tile.sync();
        for(int i = 0; i < 3; i++) {
            temp_storage_fl[tile.thread_rank()] = force_a[i];
            tile.sync();
            forces[ipa][i] = tile_sum<tile_size>(tile, temp_storage_fl,
                pair_end - pair_begin > tile.num_threads() ? tile.num_threads() : pair_end - pair_begin);
            tile.sync();
        }

        pair_begin = pair_end;
        step++;
    }

    temp_storage_fl[tile.thread_rank()] = e;
    tile.sync();
    fl e_sum = tile_sum<tile_size>(tile, temp_storage_fl, tile.num_threads());

	return e_sum;
}

template<class Size>
__device__ fl model_gpu<Size>::eval_deriv(
    const precalculate_byatom_gpu<Size> & p,
    const cache_gpu<Size> & ig,
    const vec_gpu & v,
    change_gpu<Size> & g
) { // clean up
	// INTER ligand - grid
	fl e = ig.eval_deriv(atoms, coords, minus_forces, atoms_size, v[1]); // sets minus_forces, except inflex

	// INTRA ligand_i - ligand_i
	e += eval_interacting_pairs_deriv_gpu(
        p,
        v[0],
        ligand.pairs,
        ligand.pairs_size,
        coords,
        minus_forces
    ); // adds to minus_forces
    
	// calculate derivatives
	ligand.flexible_body.derivative(coords, minus_forces, g.ligand);

	return e;
}

template<class Size>
template<int tile_size>
__device__ fl model_gpu<Size>::eval_deriv(
    const precalculate_byatom_gpu<Size> & p,
    const cache_gpu<Size> & ig,
    const vec_gpu & v,
    change_gpu<Size> & g,
    cg::thread_block_tile<tile_size> & tile,
    fl * temp_storage_fl,
    int async
) { // clean up
	// INTER ligand - grid
	fl e = ig.eval_deriv<tile_size>(atoms, coords, minus_forces, atoms_size, v[1], tile, temp_storage_fl, 0); // sets minus_forces, except inflex
    tile.sync();

	// INTRA ligand_i - ligand_i
	e += eval_interacting_pairs_deriv_gpu<Size, tile_size>(
        p,
        v[0],
        ligand.pairs,
        ligand.pairs_size,
        coords,
        minus_forces,
        tile,
        temp_storage_fl,
        ligand.one_atom_pairs,
        ligand.one_atom_pairs_size,
        0
    ); // adds to minus_forces
    tile.sync();
    
	// calculate derivatives
	ligand.flexible_body.derivative<tile_size>(coords, minus_forces, g.ligand, tile, temp_storage_fl, 0);

	return e;
}

template<class Size>
__device__ void model_gpu<Size>::get_heavy_atom_movable_coords(output_type_gpu<Size> & tmp) const { // FIXME mv
    int cnt = 0;
	for(int i = 0; i < atoms_size; i++)
		if (atoms[i].el != EL_TYPE_H_GPU)
			tmp.coords[cnt++] = coords[i];
    tmp.coords_size = cnt;
}

template<class Size>
__device__ fl model_gpu<Size>::gyration_radius() const {
	const ligand_gpu<Size> & lig = ligand;
	fl acc = 0;
	unsigned counter = 0;
	for(int i = lig.atom_begin; i< lig.atom_end; i++) {
		if(atoms[i].el != EL_TYPE_H_GPU) { // only heavy atoms are used
			acc += vec_distance_sqr(coords[i], lig.flexible_body.node.frame.origin); // FIXME? check!
			++counter;
		}
	}
	return (counter > 0) ? std::sqrt(acc/counter) : 0;
}


#endif