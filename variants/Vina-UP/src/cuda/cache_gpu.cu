#include "cache_gpu.cuh"
#include "common_gpu.cuh"

__device__ fl grid_gpu::evaluate(
    const vec_gpu & location,
    fl slope,
    fl v,
    vec_gpu * deriv
) const { // sets *deriv if not NULL	
	vec_gpu s  = elementwise_product_gpu(location - m_init, m_factor); 

	vec_gpu miss;
	miss.init(0, 0, 0);
	int region[3];
	int a[3];
	// std::cout << "grid::evaluate_aux: tag 1" << std::endl;
	for(int i = 0; i < 3; i++) {
		if(s[i] < 0) {
			miss[i] = -s[i];
			region[i] = -1;
			a[i] = 0; 
			s[i] = 0;
		}       
		else if(s[i] >= m_dim_fl_minus_1[i]) {
			miss[i] = s[i] - m_dim_fl_minus_1[i];
			region[i] = 1;
			assert(m_data.dim(i) >= 2);
			a[i] = m_data.dim(i) -  2; 
			s[i] = 1;
		}
		else {
			region[i] = 0; // now that region is boost::array, it's not initialized
			a[i] = sz(s[i]);
			s[i] -= a[i];
		}
		assert(s[i] >= 0);
		assert(s[i] <= 1);
		assert(a[i] >= 0);
		assert(a[i]+1 < m_data.dim(i));
	}
	const fl penalty = slope * (miss * m_factor_inv); // FIXME check that inv_factor is correctly initialized and serialized
	assert(penalty > -epsilon_fl_gpu);
	// std::cout << "grid::evaluate_aux: tag 2" << std::endl;
	// std::cout << a[0] << " " << a[1] << " " << a[2] << std::endl;
	const sz x0 = a[0];
	const sz y0 = a[1];
	const sz z0 = a[2];

	const sz x1 = x0+1;
	const sz y1 = y0+1;
	const sz z1 = z0+1;


	const fl f000 = m_data(x0, y0, z0);
	const fl f100 = m_data(x1, y0, z0);
	const fl f010 = m_data(x0, y1, z0);
	const fl f110 = m_data(x1, y1, z0);
	const fl f001 = m_data(x0, y0, z1);
	const fl f101 = m_data(x1, y0, z1);
	const fl f011 = m_data(x0, y1, z1);
	const fl f111 = m_data(x1, y1, z1);
	// std::cout << "grid::evaluate_aux: tag 3" << std::endl;
	const fl x = s[0];
	const fl y = s[1];
	const fl z = s[2];

	const fl mx = 1-x;
	const fl my = 1-y;
	const fl mz = 1-z;

	fl f = 
		f000 *  mx * my * mz  +
		f100 *   x * my * mz  +
		f010 *  mx *  y * mz  + 
		f110 *   x *  y * mz  +
		f001 *  mx * my *  z  +
		f101 *   x * my *  z  +
		f011 *  mx *  y *  z  +
		f111 *   x *  y *  z  ;

	if(deriv) { // valid pointer
		const fl x_g = 
			f000 * (-1)* my * mz  +
			f100 *   1 * my * mz  +
			f010 * (-1)*  y * mz  + 
			f110 *   1 *  y * mz  +
			f001 * (-1)* my *  z  +
			f101 *   1 * my *  z  +
			f011 * (-1)*  y *  z  +
			f111 *   1 *  y *  z  ;


		const fl y_g = 
			f000 *  mx *(-1)* mz  +
			f100 *   x *(-1)* mz  +
			f010 *  mx *  1 * mz  + 
			f110 *   x *  1 * mz  +
			f001 *  mx *(-1)*  z  +
			f101 *   x *(-1)*  z  +
			f011 *  mx *  1 *  z  +
			f111 *   x *  1 *  z  ;


		const fl z_g =  
			f000 *  mx * my *(-1) +
			f100 *   x * my *(-1) +
			f010 *  mx *  y *(-1) + 
			f110 *   x *  y *(-1) +
			f001 *  mx * my *  1  +
			f101 *   x * my *  1  +
			f011 *  mx *  y *  1  +
			f111 *   x *  y *  1  ;

		vec_gpu gradient;
		gradient.init(x_g, y_g, z_g);
		curl_gpu(f, gradient, v);
		vec_gpu gradient_everywhere;

		for(int i = 0; i < 3; i++) {
			gradient_everywhere[i] = ((region[i] == 0) ? gradient[i] : 0);
			(*deriv)[i] = m_factor[i] * gradient_everywhere[i] + slope * region[i];
		}

		return f + penalty;
	}
	else {
		curl_gpu(f, v);
		return f + penalty;
	}
} 

__device__ fl cache_gpu<Small>::eval_deriv(
	atom_gpu * m_atoms,
	vec_gpu * m_coords,
	vec_gpu * m_minus_forces,
	int m_atoms_size,
	fl v
) const { // needs m.coords, sets m.minus_forces
	fl e = 0;
	sz nat = num_atom_types(atom_gpu::XS);

	for(int i = 0; i < m_atoms_size; i++) {
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
		// std::cout << "cache::eval_deriv, tag: 1" << std::endl;
		e += g.evaluate(m_coords[i], m_slope, v, &deriv);
		// std::cout << "cache::eval_deriv, tag: 2" << std::endl;
		m_minus_forces[i] = deriv;
	}
	return e;
}

// template<>
// __device__ fl cache_gpu<Small>::eval_deriv<32>(
// 	atom_gpu * m_atoms,
// 	vec_gpu * m_coords,
// 	vec_gpu * m_minus_forces,
// 	int m_atoms_size,
// 	fl v,
// 	cg::thread_block_tile<32> & tile,
// 	fl * temp_storage_fl,
// 	int async
// ) const { // needs m.coords, sets m.minus_forces
// 	fl e = 0;
// 	sz nat = num_atom_types(atom_gpu::XS);

// 	for(int i = tile.thread_rank(); i < m_atoms_size; i += tile.num_threads()) {
// 		const atom_gpu & a = m_atoms[i];
// 		sz t = a.get(atom_gpu::XS);

// 		if (t >= nat) { m_minus_forces[i].assign(0); continue; }
// 		switch (t)
// 		{
// 			case XS_TYPE_G0_GPU:
// 			case XS_TYPE_G1_GPU:
// 			case XS_TYPE_G2_GPU:
// 			case XS_TYPE_G3_GPU:
// 				m_minus_forces[i].assign(0);
// 				continue;
// 			case XS_TYPE_C_H_CG0_GPU:
// 			case XS_TYPE_C_H_CG1_GPU:
// 			case XS_TYPE_C_H_CG2_GPU:
// 			case XS_TYPE_C_H_CG3_GPU:
// 				t = XS_TYPE_C_H_GPU;
// 				break;
// 			case XS_TYPE_C_P_CG0_GPU:
// 			case XS_TYPE_C_P_CG1_GPU:
// 			case XS_TYPE_C_P_CG2_GPU:
// 			case XS_TYPE_C_P_CG3_GPU:
// 				t = XS_TYPE_C_P_GPU;
// 				break;
// 		}

// 		vec_gpu deriv;
// 		const grid_gpu & g = m_grids[t];
// 		e += g.evaluate(m_coords[i], m_slope, v, &deriv);
// 		m_minus_forces[i] = deriv;
// 	}
// 	tile.sync();

// 	temp_storage_fl[tile.thread_rank()] = e;
// 	tile.sync();

// 	fl e_sum = tile_sum<32>(tile, temp_storage_fl,
// 		m_atoms_size > tile.num_threads() ? tile.num_threads() : m_atoms_size);
	
// 	return e_sum;
// }