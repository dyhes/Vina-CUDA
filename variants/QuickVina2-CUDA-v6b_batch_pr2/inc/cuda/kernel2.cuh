extern "C"
void kernel_monte(
				output_type_cl**				ric_batch,
				m_cl**							mg_batch,
				pre_cl*							pre,
				grids_cl*						grids,
				random_maps**					random_maps_batch,
				ligand_atom_coords_cl**			coords_batch,
				output_type_cl**				results_batch,
				const mis_cl*					mis,
				const int*						torsion_sizes,
				const int*						search_depths,
				const int*						bfgs_steps,
				const int						rilc_bfgs_enable,
				const int						num_ligands,
				const int						threads_per_ligand,
				const int*						lig_valid
);

extern "C"
void kernel_monte_ranged(
	cudaStream_t s3_kernel_stream,
	output_type_cl **ric_batch_in,
	m_cl **mg_batch_in,
	pre_cl *pre,
	grids_cl *grids,
	random_maps **random_maps_batch_in,
	ligand_atom_coords_cl **coords_batch_in,
	output_type_cl **results_batch_in,
	const mis_cl *mis,
	const int *torsion_sizes,
	const int *search_depths,
	const int *bfgs_steps,
	const int rilc_bfgs_enable,
	const int num_ligands,
	const int threads_per_ligand,
	const int *lig_valid,
	cudaEvent_t ev_range);