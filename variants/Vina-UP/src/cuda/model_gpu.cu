#include "model_gpu.cuh"

__device__ void branches_set_conf_gpu(
	tree_gpu * tree_nodes,
	cuda::std::pair<int, int> * ch_BE,
	int ch_BE_size,
	const atom_gpu * atoms,
	vec_gpu * coords,
	const fl * c)
{
	for(int l = 0; l < ch_BE_size; l++) {
		int n_begin = ch_BE[l].first; int n_end = ch_BE[l].second;

		for(int i = n_begin; i < n_end; i++) {
			const fl * ci = c + i;
			tree_nodes[i].set_conf(atoms, coords, ci);
		}
	}
}

// __device__ void branches_set_conf_gpu(
// 	tree_gpu * tree_nodes,
// 	cuda::std::pair<int, int> * ch_BE,
// 	int ch_BE_size,
// 	const atom_gpu * atoms,
// 	vec_gpu * coords,
// 	const fl * c,
// 	int async
// ) {
// 	for(int l = 0; l < ch_BE_size; l++) {
// 		int n_begin = ch_BE[l].first; int n_end = ch_BE[l].second;

// 		for(int i = threadIdx.x; i < n_end - n_begin; i += blockDim.x) {
// 			const fl * ci = c + i;
// 			tree_nodes[i].set_conf(atoms, coords, ci);
// 		}
// 	}
// }


__device__ void branches_derivative_gpu(
    const tree_gpu * tree_nodes,
	int tree_nodes_size,
    const vec_gpu * coords,
    const vec_gpu * forces,
    cuda::std::pair<vec_gpu, vec_gpu> * nodes_force_torque,
	const cuda::std::pair<int, int> * ch_BE,
	const int ch_BE_size,
    fl * d
) { // adds to out
	for(int i = 0; i < tree_nodes_size; i++)
		nodes_force_torque[i] = tree_nodes[i].derivative(coords, forces);

	for(int l = ch_BE_size - 1; l >= 0; l--) {
		int n_begin = ch_BE[l].first; int n_end = ch_BE[l].second;
		for(int j = n_begin; j < n_end; j++) {
			for(int ch = tree_nodes[j].ch_begin; ch < tree_nodes[j].ch_end; ch++) {
				nodes_force_torque[j].first += nodes_force_torque[ch].first;

				vec_gpu r;
        		r = tree_nodes[ch].node.frame.origin - (*tree_nodes[ch].parent).origin;
				
				nodes_force_torque[j].second +=
					cross_product_gpu(r, nodes_force_torque[ch].first) +
					nodes_force_torque[ch].second;
			}
			fl * p = d + j;
			tree_nodes[j].node.set_derivative(nodes_force_torque[j], *p);
		}
	}
}

// __device__ void branches_derivative_gpu(
//     const tree_gpu * tree_nodes,
// 	int tree_nodes_size,
//     const vec_gpu * coords,
//     const vec_gpu * forces,
//     cuda::std::pair<vec_gpu, vec_gpu> * nodes_force_torque,
// 	const cuda::std::pair<int, int> * ch_BE,
// 	const int ch_BE_size,
//     fl * d,
// 	int async
// ) { // adds to out
// 	for(int i = threadIdx.x; i < tree_nodes_size; i += blockDim.x)
// 		nodes_force_torque[i] = tree_nodes[i].derivative(coords, forces);
// 	__syncthreads();

// 	for(int l = ch_BE_size - 1; l >= 0; l--) {
// 		int n_begin = ch_BE[l].first; int n_end = ch_BE[l].second;
// 		for(int j = threadIdx.x; j < n_end - n_begin; j += blockDim.x) {
// 			for(int ch = tree_nodes[j].ch_begin; ch < tree_nodes[j].ch_end; ch++) {
// 				nodes_force_torque[j].first += nodes_force_torque[ch].first;

// 				vec_gpu r;
//         		r = tree_nodes[ch].node.frame.origin - (*tree_nodes[ch].parent).origin;
				
// 				nodes_force_torque[j].second +=
// 					cross_product_gpu(r, nodes_force_torque[ch].first) +
// 					nodes_force_torque[ch].second;
// 			}
// 			fl * p = d + j;
// 			tree_nodes[j].node.set_derivative(nodes_force_torque[j], *p);
// 		}
// 		__syncthreads();
// 	}
// }

