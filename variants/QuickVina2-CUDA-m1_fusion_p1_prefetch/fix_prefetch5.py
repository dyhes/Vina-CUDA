f = "inc/cuda/quasi_newton.cuh"
s = open(f).read()

old = ('''__device__ flo ig_eval_deriv(\t\t\t\toutput_type_cl*\t\tx,
\t\t\t\t\t\t\t\t\t\t\tchange_cl*\t\t\tg, 
\t\t\t\t\t\tconst\t\t\t\tflo\t\t\t\tv,
\t\t\t\t\t\t\t\t\t\t\tgrids_cl*\t\t\tgrids,  // delete the constant
\t\t\t\t\t\t\t\t\t\t\tm_cl*\t\t\t\tm,
\t\t\t\t\t\tconst\t\t\t\tflo\t\t\t\tepsilon_fl
) {
\tflo e = 0;
\tint nat = num_atom_types(grids->atu);
\t__shared__ flo stage[2][8];
\tauto pipeline = cuda::make_pipeline();
\tint n = m->m_num_movable_atoms;
\t// initial prefetch for atom 0 (if valid)
\tbool has_last = false;
\tdecltype(pipeline)::commit_t last_commit;
\tif (n > 0) {
\t\tint t0 = m->atoms[0].types[grids->atu];
\t\tif (t0 < nat) {
\t\t\tint b0 = grid_base_of(&grids->grids[t0], m->m_coords.coords[0]);
\t\t\tdecltype(pipeline)::copy(pipeline, &stage[0][0], &grids->grids[t0].m_data[b0], 8);
\t\t\tlast_commit = pipeline.producer_commit();
\t\t\thas_last = true;
\t\t}
\t}
\tfor (int i = 0; i < n; i++) {
\t\tint t = m->atoms[i].types[grids->atu];
\t\tif (t >= nat) {
\t\t\tfor (int j = 0; j < 3; j++)m->minus_forces.coords[i][j] = 0;
\t\t\tcontinue;
\t\t}
\t\tflo deriv[3];
\t\tconst flo* buf = nullptr;
\t\tif (has_last) {
\t\t\tpipeline.consumer_wait(last_commit);
\t\t\tbuf = stage[i % 2];
\t\t\tpipeline.consumer_release(last_commit);
\t\t}
\t\te = e + g_evaluate(&grids->grids[t], m->m_coords.coords[i], grids->slope, v, deriv, epsilon_fl, buf);
\t\tfor (int j = 0; j < 3; j++) m->minus_forces.coords[i][j] = deriv[j];
\t\t// prefetch next atom (i+1)
\t\thas_last = false;
\t\tif (i + 1 < n) {
\t\t\tint t1 = m->atoms[i + 1].types[grids->atu];
\t\t\tif (t1 < nat) {
\t\t\t\tint b1 = grid_base_of(&grids->grids[t1], m->m_coords.coords[i + 1]);
\t\t\t\tdecltype(pipeline)::copy(pipeline, &stage[(i + 1) % 2][0], &grids->grids[t1].m_data[b1], 8);
\t\t\t\tlast_commit = pipeline.producer_commit();
\t\t\t\thas_last = true;
\t\t\t}
\t\t}
\t}
\treturn e;
}''')

new = ('''__device__ flo ig_eval_deriv(\t\t\t\toutput_type_cl*\t\tx,
\t\t\t\t\t\t\t\t\t\t\tchange_cl*\t\t\tg, 
\t\t\t\t\t\tconst\t\t\t\tflo\t\t\t\tv,
\t\t\t\t\t\t\t\t\t\t\tgrids_cl*\t\t\tgrids,  // delete the constant
\t\t\t\t\t\t\t\t\t\t\tm_cl*\t\t\t\tm,
\t\t\t\t\t\tconst\t\t\t\tflo\t\t\t\tepsilon_fl
) {
\tflo e = 0;
\tint nat = num_atom_types(grids->atu);
\t__shared__ flo stage[2][8];
\tauto pipeline = cuda::make_pipeline();
\tauto last_commit = pipeline.producer_commit();  // placeholder no-op commit
\tint n = m->m_num_movable_atoms;
\t// initial prefetch for atom 0 (if valid)
\tbool has_last = false;
\tif (n > 0) {
\t\tint t0 = m->atoms[0].types[grids->atu];
\t\tif (t0 < nat) {
\t\t\tint b0 = grid_base_of(&grids->grids[t0], m->m_coords.coords[0]);
\t\t\tcuda::memcpy_async(&stage[0][0], &grids->grids[t0].m_data[b0], 8, pipeline);
\t\t\tlast_commit = pipeline.producer_commit();
\t\t\thas_last = true;
\t\t}
\t}
\tfor (int i = 0; i < n; i++) {
\t\tint t = m->atoms[i].types[grids->atu];
\t\tif (t >= nat) {
\t\t\tfor (int j = 0; j < 3; j++)m->minus_forces.coords[i][j] = 0;
\t\t\tcontinue;
\t\t}
\t\tflo deriv[3];
\t\tconst flo* buf = nullptr;
\t\tif (has_last) {
\t\t\tpipeline.consumer_wait(last_commit);
\t\t\tbuf = stage[i % 2];
\t\t\tpipeline.consumer_release(last_commit);
\t\t}
\t\te = e + g_evaluate(&grids->grids[t], m->m_coords.coords[i], grids->slope, v, deriv, epsilon_fl, buf);
\t\tfor (int j = 0; j < 3; j++) m->minus_forces.coords[i][j] = deriv[j];
\t\t// prefetch next atom (i+1)
\t\thas_last = false;
\t\tif (i + 1 < n) {
\t\t\tint t1 = m->atoms[i + 1].types[grids->atu];
\t\t\tif (t1 < nat) {
\t\t\t\tint b1 = grid_base_of(&grids->grids[t1], m->m_coords.coords[i + 1]);
\t\t\t\tcuda::memcpy_async(&stage[(i + 1) % 2][0], &grids->grids[t1].m_data[b1], 8, pipeline);
\t\t\t\tlast_commit = pipeline.producer_commit();
\t\t\t\thas_last = true;
\t\t\t}
\t\t}
\t}
\treturn e;
}''')

assert old in s, "ig_eval_deriv body not found for rewrite"
s = s.replace(old, new, 1)
open(f, "w").write(s)
print("OK: ig_eval_deriv rewritten with cuda::memcpy_async + auto token")
