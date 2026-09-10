extern "C"
void kernel_monte(
				const		output_type_cl*		ric,
					  		m_cl*		mg,
      		pre_cl*		pre,
					grids_cl*		grids,
					const random_maps*	random_maps,
					ligand_atom_coords_cl*	coords,
					output_type_cl*		results,
const					mis_cl*		mis,
const					int*		torsion_sizes,
const					int*		search_depths,
const					int*		bfgs_steps,
const 			int		rilc_bfgs_enable,
const 			int		num_ligands,
const 			int		threads_per_ligand
);

// Stream version
void kernel_monte_stream(
				const		output_type_cl*		ric,
					  		m_cl*		mg,
      		pre_cl*		pre,
					grids_cl*		grids,
					random_maps*	random_maps,
					ligand_atom_coords_cl*	coords,
					output_type_cl*		results,
const					mis_cl*		mis,
const					int*		torsion_sizes,
const					int*		search_depths,
const					int*		bfgs_steps,
const 			int		rilc_bfgs_enable,
const 			int		num_ligands,
const 			int		threads_per_ligand,
				cudaStream_t	stream
);
