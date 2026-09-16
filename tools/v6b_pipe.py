#!/usr/bin/env python3
# v6b-P pipe patches, IDEMPOTENT (skip already-applied edits). Idempotent:
# P-A may already be applied (grep), P-B/P-C likewise; re-run safe.
import sys
D = sys.argv[1]

def patch(path, old, new, tag, n=1, skip_if=None):
    src = open(path, encoding='utf-8').read()
    if skip_if and skip_if in src:
        print(tag, 'already applied, skip'); return
    c = src.count(old)
    assert c == n, '%s: anchor x%d (want %d): %r' % (path, c, n, old[:60])
    open(path, 'w', encoding='utf-8').write(src.replace(old, new))
    print(tag, 'patched')

P1 = D + '/lib/main_procedure_cu.cpp'
patch(P1,
'''			checkCUDA(cudaMemcpy(result_coords_ptrs[ligand_count], result_coords_gpus[ligand_count], par.mc.thread * sizeof(ligand_atom_coords_cl), cudaMemcpyDeviceToHost));
			checkCUDA(cudaMemcpy(result_coords_ptrs[ligand_count], result_coords_gpus[ligand_count], par.mc.thread * sizeof(ligand_atom_coords_cl), cudaMemcpyDeviceToHost));''',
'''			checkCUDA(cudaMemcpy(result_coords_ptrs[ligand_count], result_coords_gpus[ligand_count], par.mc.thread * sizeof(ligand_atom_coords_cl), cudaMemcpyDeviceToHost));
			// [v6b-P P-A] second (duplicate) coords DtoH removed: it re-copied the
			// identical buffer back-to-back (~0.3s per 96-ligand chunk, pure waste).''',
tag='P-A', skip_if='[v6b-P P-A]')

P2 = D + '/main/main.cpp'
patch(P2,
'''#ifndef NO_REFINEMENT
		VINA_FOR_IN(i, out_cont)
		refine_structure(m, prec, nc[ligand_count], out_cont[i], authentic_v, par.mc.ssd_par.bfgs_steps[ligand_count]);
#endif''',
'''#ifndef NO_REFINEMENT
		// [v6b-P P-B v2] refine_structure is NOT pose-parallel-safe as written:
		// it mutates the SHARED model m (m.set(out.c)) and the SHARED nc.slope
		// (5-slope sweep) — parallelizing across poses races on both (the
		// double-free/corruption we hit). Keep the loop serial; the pipelining
		// win for this segment moves to P-D (overlap with next chunk kernel),
		// not intra-ligand parallelism.
		VINA_FOR_IN(i, out_cont)
		refine_structure(m, prec, nc[ligand_count], out_cont[i], authentic_v, par.mc.ssd_par.bfgs_steps[ligand_count]);
#endif''',
tag='P-B', skip_if='[v6b-P P-B]')

patch(P2,
'''	for (int ligand_count = 0; ligand_count < ligand_num; ligand_count++)
	{
		hp_mark("bfgs_steps_vec");
		non_cache tmp(ms[ligand_count], gd, &prec, slope);
		nc.push_back(tmp); // if gd has 0 n's, this will not constrain anything
	}''',
'''	// [v6b-P P-C] non_cache construction is per-ligand independent (96 x 3.9ms
	// serial). non_cache has no default ctor -> build into unique_ptr slots in
	// parallel, then move out in index order (nc[ligand_count] indexing kept).
	{
		std::vector<std::unique_ptr<non_cache>> tmp_nc(ligand_num);
		#pragma omp parallel for schedule(dynamic, 1)
		for (int ligand_count = 0; ligand_count < ligand_num; ligand_count++)
		{
			tmp_nc[ligand_count].reset(new non_cache(ms[ligand_count], gd, &prec, slope));
		}
		for (int ligand_count = 0; ligand_count < ligand_num; ligand_count++)
			nc.push_back(std::move(*tmp_nc[ligand_count]));
	}
	hp_mark("bfgs_steps_vec");''',
tag='P-C', skip_if='[v6b-P P-C]')
