#include <cstdio>
#include "copy_utils.cuh"
#include "model_gpu.cuh"
#include "cache_gpu.cuh"
#include "precalculate_gpu.cuh"
#include "output_type_gpu.cuh"
#include "monte_carlo_gpu.cuh"

#include "../lib/monte_carlo.h"
#include "../lib/model.h"
#include "../lib/cache.h"
#include "../lib/precalculate.h"
#include "../lib/conf.h"

#include <boost/chrono.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>

void vec_copy(vec_gpu & v_gpu, const vec & v) {
	v_gpu.data[0] = v.data[0]; v_gpu.data[1] = v.data[1]; v_gpu.data[2] = v.data[2];
}

void vec_copy(vec & v, const vec_gpu & v_gpu) {
	v.data[0] = v_gpu.data[0]; v.data[1] = v_gpu.data[1]; v.data[2] = v_gpu.data[2];
}

void matrix_copy(mat_gpu & m_gpu, const mat & m) {
	m_gpu.data[0] = m.data[0]; m_gpu.data[1] = m.data[1]; m_gpu.data[2] = m.data[2];
	m_gpu.data[3] = m.data[3]; m_gpu.data[4] = m.data[4]; m_gpu.data[5] = m.data[5];
	m_gpu.data[6] = m.data[6]; m_gpu.data[7] = m.data[7]; m_gpu.data[8] = m.data[8];
}

void quaternion_copy(qt_gpu & q_gpu, const qt & q) {
	q_gpu.a = q.R_component_1(); q_gpu.b = q.R_component_2(); q_gpu.c = q.R_component_3(); q_gpu.d = q.R_component_4();
}

void quaternion_copy(qt & q, const qt_gpu & q_gpu) {
	qt _q(q_gpu.a, q_gpu.b, q_gpu.c, q_gpu.d);
	q = _q;
}

template<class Size>
void model_copy(model_gpu<Size> * m_gpu_ptr, const model & m, int m_size) {
	std::vector<model_gpu<Size>> m_array(m_size);
for(int mi = 0; mi < m_size; mi++) {
    model_gpu<Size> & m_gpu = m_array[mi];
	m_gpu.atoms_size = m.num_movable_atoms();
	for(int i = 0; i < m_gpu.atoms_size; i++) {
		m_gpu.atoms[i].el = m.atoms[i].el;
		m_gpu.atoms[i].ad = m.atoms[i].ad;
		m_gpu.atoms[i].xs = m.atoms[i].xs;
		m_gpu.atoms[i].sy = m.atoms[i].sy;
		vec_copy(m_gpu.atoms[i].coords, m.atoms[i].coords);
		vec_copy(m_gpu.coords[i], m.coords[i]);
		vec_copy(m_gpu.minus_forces[i], m.minus_forces[i]);
	}

	std::vector<interacting_pair_gpu> pairs_array;
	m_gpu.ligand.pairs_size = m.ligands[0].pairs.size();
	for(int i = 0; i < m_gpu.ligand.pairs_size; i++) {
		m_gpu.ligand.pairs[i].a = m.ligands[0].pairs[i].a;
		m_gpu.ligand.pairs[i].b = m.ligands[0].pairs[i].b;
		pairs_array.push_back(m_gpu.ligand.pairs[i]);
	}
	m_gpu.ligand.atom_begin = m.ligands[0].begin;
	m_gpu.ligand.atom_end = m.ligands[0].end;

	std::sort(pairs_array.begin(), pairs_array.end(),
		[](interacting_pair_gpu & a, interacting_pair_gpu & b){ return a.a < b.a; });
	for(int i = 0; i < pairs_array.size(); i++)
		m_gpu.ligand.pairs[i] = pairs_array[i];
	int i = 0, j = 0;
	m_gpu.ligand.one_atom_pairs_size = 0;
	while(i < m_gpu.ligand.pairs_size) {
		while(j < m_gpu.ligand.pairs_size && m_gpu.ligand.pairs[i].a == m_gpu.ligand.pairs[j].a) j++;
		m_gpu.ligand.one_atom_pairs[m_gpu.ligand.one_atom_pairs_size++] = j;
		i = j;
	}

	m_gpu.ligand.flexible_body.node.frame.atom_begin = m.ligands[0].node.begin;
	m_gpu.ligand.flexible_body.node.frame.atom_end = m.ligands[0].node.end;
	vec_copy(m_gpu.ligand.flexible_body.node.frame.origin, m.ligands[0].node.origin);
	matrix_copy(m_gpu.ligand.flexible_body.node.frame.orientation_m, m.ligands[0].node.orientation_m);
	quaternion_copy(m_gpu.ligand.flexible_body.node.frame.orientation_q, m.ligands[0].node.orientation_q);

	std::vector<tree<segment>> tree_nodes_h;
	for(int i = 0; i < m.ligands[0].children.size(); i++)
		tree_nodes_h.push_back(m.ligands[0].children[i]);

	int begin = 0; int end = tree_nodes_h.size();
	int begin_next; int end_next;
	int ch_BE_size = 0;
	while(begin < end) {
		m_gpu.ligand.flexible_body.ch_BE[ch_BE_size++] = cuda::std::make_pair(begin, end);

		begin_next = end;
		for(int i = begin; i < end; i++)
			for(int j = 0; j < tree_nodes_h[i].children.size(); j++)
				tree_nodes_h.push_back(tree_nodes_h[i].children[j]);
		end_next = tree_nodes_h.size();

		begin = begin_next;
		end = end_next;
	}
	m_gpu.ligand.flexible_body.ch_BE_size = ch_BE_size;

	std::vector<tree_gpu> tree_nodes_d;
	tree_nodes_d.resize(tree_nodes_h.size());
	int base = m.ligands[0].children.size();
	for(int i = 0; i < tree_nodes_h.size(); i++) {
		tree_gpu & node_d = tree_nodes_d[i];
		tree<segment> & node_h = tree_nodes_h[i];

		vec_copy(node_d.node.relative_axis, node_h.node.relative_axis);
		vec_copy(node_d.node.relative_origin, node_h.node.relative_origin);
		vec_copy(node_d.node.axis, node_h.node.axis);

		node_d.node.frame.atom_begin = node_h.node.begin;
		node_d.node.frame.atom_end = node_h.node.end;
		vec_copy(node_d.node.frame.origin, node_h.node.origin);
		matrix_copy(node_d.node.frame.orientation_m, node_h.node.orientation_m);
		quaternion_copy(node_d.node.frame.orientation_q, node_h.node.orientation_q);

		node_d.ch_begin = base;
		base += node_h.children.size();
		node_d.ch_end = base;

		for(int j = node_d.ch_begin; j < node_d.ch_end; j++)
			tree_nodes_d[j].parent = &(m_gpu_ptr[mi].ligand.flexible_body.tree_nodes[i].node.frame);
				// &m_gpu.ligand.flexible_body.tree_nodes[i].node.frame;
	}

	m_gpu.ligand.flexible_body.ch_begin = 0;
	m_gpu.ligand.flexible_body.ch_end = m.ligands[0].children.size();
	for(int i = m_gpu.ligand.flexible_body.ch_begin; i < m_gpu.ligand.flexible_body.ch_end; i++)
		tree_nodes_d[i].parent = &(m_gpu_ptr[mi].ligand.flexible_body.node.frame);
			// &m_gpu.ligand.flexible_body.node.frame;
	m_gpu.ligand.flexible_body.tree_nodes_size = tree_nodes_d.size();
	fprintf(stderr, "[DBG] model_gpu<Small>=%zu B, tree_nodes=%zu, atoms=%d, pairs=%d, ch_BE=%d\n", sizeof(model_gpu<Small>), tree_nodes_d.size(), m_gpu.atoms_size, m_gpu.ligand.pairs_size, m_gpu.ligand.flexible_body.ch_BE_size);
	for(int i = 0; i < m_gpu.ligand.flexible_body.tree_nodes_size; i++)
		m_gpu.ligand.flexible_body.tree_nodes[i] = tree_nodes_d[i];
}
	cudaMemcpy(m_gpu_ptr, m_array.data(), sizeof(model_gpu<Size>) * m_size, cudaMemcpyHostToDevice);
	// memcpy(m_gpu_ptr, &m_gpu, sizeof(model_gpu<Small>));
}

template<class Size>
void cache_copy(cache_gpu<Size> * c_gpu_ptr, fl * c_m_data_m_data_gpu_ptr, const cache & c) {
    cache_gpu<Size> c_gpu;
	c_gpu.m_slope = c.m_slope;
	c_gpu.m_grids_size = c.m_grids.size();

	std::vector<fl> fl_array(c_gpu.m_grids_size * Size::grids_size);

	for(int i = 0; i < c_gpu.m_grids_size; i++) {
		grid_gpu & g_gpu = c_gpu.m_grids[i];
		const grid & g = c.m_grids[i];

		vec_copy(g_gpu.m_init, g.m_init);
		vec_copy(g_gpu.m_factor, g.m_factor);
		vec_copy(g_gpu.m_dim_fl_minus_1, g.m_dim_fl_minus_1);
		vec_copy(g_gpu.m_factor_inv, g.m_factor_inv);

		g_gpu.m_data.m_i = g.m_data.m_i;
		g_gpu.m_data.m_j = g.m_data.m_j;
		g_gpu.m_data.m_k = g.m_data.m_k;
		g_gpu.m_data.m_data_size = g.m_data.m_i * g.m_data.m_j * g.m_data.m_k;
		g_gpu.m_data.m_data = c_m_data_m_data_gpu_ptr + i * Size::grids_size;

		for(int j = 0; j < g_gpu.m_data.m_data_size; j++)
			fl_array[j + i * Size::grids_size] = g.m_data.m_data[j];
	}

	cudaMemcpy(c_m_data_m_data_gpu_ptr, fl_array.data(), sizeof(fl) * fl_array.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(c_gpu_ptr, &c_gpu, sizeof(cache_gpu<Size>), cudaMemcpyHostToDevice);
	// memcpy(c_m_data_m_data_gpu_ptr, fl_array.data(), sizeof(fl) * fl_array.size());
	// memcpy(c_gpu_ptr, &c_gpu, sizeof(cache_gpu<Size>));
}

template<class Size>
void precalculate_byatom_copy(precalculate_byatom_gpu<Size> * p_gpu_ptr, cuda::std::pair<fl, fl> * p_m_data_m_data_gpu_ptr, const precalculate_byatom & p) {
    precalculate_byatom_gpu<Size> p_gpu;
	p_gpu.m_cutoff_sqr = p.m_cutoff_sqr;
	p_gpu.m_max_cutoff_sqr = p.m_max_cutoff_sqr;

	p_gpu.m_data.m_dim = p.m_data.m_dim;
	p_gpu.m_data.m_data_size = p.m_data.m_data.size();

	std::vector<cuda::std::pair<fl, fl>> pr_array(p_gpu.m_data.m_data_size * Size::r2_smooth_size);

	for(int i = 0; i < p_gpu.m_data.m_data_size; i++) {
		precalculate_element_gpu & pe_gpu = p_gpu.m_data.m_data[i];
		const precalculate_element & pe = p.m_data.m_data[i];

		pe_gpu.factor = pe.factor;
		pe_gpu.smooth_size = pe.smooth.size();
		pe_gpu.smooth = p_m_data_m_data_gpu_ptr + i * Size::r2_smooth_size;

		for(int j = 0; j < pe_gpu.smooth_size; j++) {
			pr_array[j + i * Size::r2_smooth_size].first = pe.smooth[j].first;
			pr_array[j + i * Size::r2_smooth_size].second = pe.smooth[j].second;
		}
	}

	cudaMemcpy(p_m_data_m_data_gpu_ptr, pr_array.data(), sizeof(cuda::std::pair<fl, fl>) * pr_array.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(p_gpu_ptr, &p_gpu, sizeof(precalculate_byatom_gpu<Size>), cudaMemcpyHostToDevice);
	// memcpy(p_m_data_m_data_gpu_ptr, pr_array.data(), sizeof(std::pair<fl, fl>) * pr_array.size());
	// memcpy(p_gpu_ptr, &p_gpu, sizeof(precalculate_byatom_gpu<Size>));
}

template<class Size>
void output_type_copy(
	output_container & out,
	const int out_size,
	output_type_gpu<Size> * out_gpu_ptr
) {
    std::vector<output_type_gpu<Size>> out_gpu(out_size);
	cudaMemcpy(out_gpu.data(), out_gpu_ptr, sizeof(output_type_gpu<Size>) * out_size, cudaMemcpyDeviceToHost);
	// memcpy(out_gpu.data(), out_gpu_ptr, sizeof(output_type_gpu<Size>) * out_size);
	cudaDeviceSynchronize();

	int n_kept = 0; int tmin = 1<<30, tmax = -1;
	for(int i = 0; i < out_size; i++) {
		int tsize = out_gpu[i].c.ligand.torsions_size;
		if(i < 5) fprintf(stderr, "[DBG-out %d] tsize=%d e=%f\n", i, tsize, out_gpu[i].e);
		if(tsize > tmax) tmax = tsize;
		if(tsize < tmin) tmin = tsize;
		if(tsize <= 0 || tsize > Size::torsions_size) continue;
		conf_size s;
		s.ligands.push_back(tsize);
		
		output_type ot(s, 0);
		ot.e = out_gpu[i].e;
		vec_copy(ot.c.ligands[0].rigid.position, out_gpu[i].c.ligand.rigid.position);
		quaternion_copy(ot.c.ligands[0].rigid.orientation, out_gpu[i].c.ligand.rigid.orientation);
		for(int j = 0; j < tsize; j++)
			ot.c.ligands[0].torsions[j] = out_gpu[i].c.ligand.torsions[j];

		// for(int j = 0; j < out_gpu[i].coords_size; j++)
		// 	ot.coords.push_back(vec(out_gpu[i].coords[j][0], out_gpu[i].coords[j][1], out_gpu[i].coords[j][2]));

		out.push_back(new output_type(ot));
		n_kept++;
	}
	fprintf(stderr, "[DBG] out_size=%d kept=%d tsize_range=[%d,%d]\n", out_size, n_kept, tmin, tmax);
	out.sort();
}

template<class Size>
__global__ void mc_bfgs_kernel(
	const monte_carlo_gpu * mc,
	model_gpu<Size> * m,
    output_type_gpu<Size> * out,
    const precalculate_byatom_gpu<Size> * p,
    const cache_gpu<Size> * ig,
    const vec_gpu * corner1,
    const vec_gpu * corner2,
	curandState * generator,
	int * seeds
	// rng * generator
) {
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seeds[thread], 0, 0, generator + thread);
	(*mc).operator()(
		*(m + thread),
		*(out + thread),
		*p,
		*ig,
		*corner1,
		*corner2,
		*(generator + thread)
	);
}

void mc_bfgs_kernel_wrapper(
	const monte_carlo & mc,
	const model & m,
    output_container & out,
    const precalculate_byatom & p,
    const igrid & ig,
    const vec& corner1,
    const vec& corner2,
	int num_tasks
) {
	const cache & _ig = dynamic_cast<const cache &>(ig);
	cache_gpu<Small> * ig_gpu_ptr;
	cudaMalloc(&ig_gpu_ptr, sizeof(cache_gpu<Small>));
	// ig_gpu_ptr = static_cast<cache_gpu<Small> *>(malloc(sizeof(cache_gpu<Small>)));
	fl * ig_m_data_m_data_gpu_ptr;
	cudaMalloc(&ig_m_data_m_data_gpu_ptr, sizeof(fl) * _ig.m_grids.size() * Small::grids_size);
	// ig_m_data_m_data_gpu_ptr = static_cast<fl *>(malloc(sizeof(fl) * _ig.m_grids.size() * Small::grids_size));
	cache_copy(ig_gpu_ptr, ig_m_data_m_data_gpu_ptr, _ig);

	precalculate_byatom_gpu<Small> * p_gpu_ptr;
	cudaMalloc(&p_gpu_ptr, sizeof(precalculate_byatom_gpu<Small>));
	// p_gpu_ptr = static_cast<precalculate_byatom_gpu<Small> *>(malloc(sizeof(precalculate_byatom_gpu<Small>)));
	cuda::std::pair<fl, fl> * p_m_data_m_data_gpu_ptr;
	cudaMalloc(&p_m_data_m_data_gpu_ptr, sizeof(cuda::std::pair<fl, fl>) * p.m_data.m_data.size() * Small::r2_smooth_size);
	// p_m_data_m_data_gpu_ptr =
	// 	static_cast<cuda::std::pair<fl, fl> *>(malloc(sizeof(cuda::std::pair<fl, fl>) * p.m_data.m_data.size() * Small::r2_smooth_size));
	precalculate_byatom_copy(p_gpu_ptr, p_m_data_m_data_gpu_ptr, p);

	vec_gpu corner_gpu[2];
	vec_copy(corner_gpu[0], corner1);
	vec_copy(corner_gpu[1], corner2);
	vec_gpu * cor_gpu_ptr;
	cudaMalloc(&cor_gpu_ptr, sizeof(vec_gpu) * 2);
	cudaMemcpy(cor_gpu_ptr, corner_gpu, sizeof(vec_gpu) * 2, cudaMemcpyHostToDevice);
	// cor_gpu_ptr = static_cast<vec_gpu *>(malloc(sizeof(vec_gpu) * 2));
	// memcpy(cor_gpu_ptr, corner_gpu, sizeof(vec_gpu) * 2);

	dim3 blockSize((Small::atoms_size / 32 + 1) * 32, 1, 1);
	dim3 blocks(200, 1, 1);
	num_tasks = blocks.x * blockSize.x;

	std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned int> distr;
	std::vector<int> seeds(num_tasks);
	for(int i = 0; i < seeds.size(); i++)
		seeds[i] = distr(eng);
	int * seeds_gpu;
	cudaMalloc(&seeds_gpu, sizeof(int) * num_tasks);
	cudaMemcpy(seeds_gpu, seeds.data(), sizeof(int) * num_tasks, cudaMemcpyHostToDevice);

	curandState * generator;
	cudaMalloc(&generator, sizeof(curandState) * num_tasks);

	model_gpu<Small> * m_gpu_ptr;
	cudaMalloc(&m_gpu_ptr, sizeof(model_gpu<Small>) * num_tasks);
	// m_gpu_ptr = static_cast<model_gpu<Small> *>(malloc(sizeof(model_gpu<Small>)));
	model_copy(m_gpu_ptr, m, num_tasks);

	output_type_gpu<Small> * out_gpu_ptr;
	cudaMalloc(&out_gpu_ptr, sizeof(output_type_gpu<Small>) * num_tasks);
	// out_gpu_ptr = static_cast<output_type_gpu<Small> *>(malloc(sizeof(output_type_gpu<Small>)));

	monte_carlo_gpu mc_gpu;
	mc_gpu.max_evals = mc.max_evals;
	mc_gpu.global_steps = 100; // mc.global_steps / (blocks.x * blockSize.x) + 1;
	mc_gpu.temperature = mc.temperature;
	vec_copy(mc_gpu.hunt_cap, mc.hunt_cap);
	mc_gpu.min_rmsd = mc.min_rmsd;
	mc_gpu.num_saved_mins = mc.num_saved_mins;
	mc_gpu.mutation_amplitude = mc.mutation_amplitude / m.gyration_radius(0);
	mc_gpu.local_steps = mc.local_steps;
	monte_carlo_gpu * mc_gpu_ptr;
	cudaMalloc(&mc_gpu_ptr, sizeof(monte_carlo_gpu));
	cudaMemcpy(mc_gpu_ptr, &mc_gpu, sizeof(monte_carlo_gpu), cudaMemcpyHostToDevice);
	// mc_gpu_ptr = static_cast<monte_carlo_gpu *>(malloc(sizeof(monte_carlo_gpu)));
	// memcpy(mc_gpu_ptr, &mc_gpu, sizeof(monte_carlo_gpu));

	std::cout << std::endl << "monte-carlo begin ..." << std::endl;
	boost::chrono::system_clock::time_point start = boost::chrono::system_clock::now();
	mc_bfgs_kernel<<<blocks, blockSize>>>(
		mc_gpu_ptr,
		m_gpu_ptr,
		out_gpu_ptr,
		p_gpu_ptr,
		ig_gpu_ptr,
		&cor_gpu_ptr[0],
		&cor_gpu_ptr[1],
		generator,
		seeds_gpu
	);
	cudaDeviceSynchronize();
	boost::chrono::duration<double> sec = boost::chrono::system_clock::now() - start;
	std::cout << "AutoDock Vina v1.2.5 took " << sec.count() << " seconds" << std::endl;
	std::cout << "monte-carlo end ..." << std::endl;

	output_type_copy(out, num_tasks, out_gpu_ptr);

	cudaFree(m_gpu_ptr);
	cudaFree(ig_gpu_ptr);
	cudaFree(p_gpu_ptr);
	cudaFree(out_gpu_ptr);
	cudaFree(p_m_data_m_data_gpu_ptr);
	cudaFree(ig_m_data_m_data_gpu_ptr);
	cudaFree(mc_gpu_ptr);
	cudaFree(cor_gpu_ptr);
	cudaFree(generator);
	cudaFree(seeds_gpu);
	// free(m_gpu_ptr);
	// free(ig_gpu_ptr);
	// free(p_gpu_ptr);
	// free(out_gpu_ptr);
	// free(p_m_data_m_data_gpu_ptr);
	// free(ig_m_data_m_data_gpu_ptr);
	// free(mc_gpu_ptr);
	// free(cor_gpu_ptr);
}