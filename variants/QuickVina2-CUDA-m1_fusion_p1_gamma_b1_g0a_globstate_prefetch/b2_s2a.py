import re
F = "inc/cuda/quasi_newton.cuh"
s = open(F).read()

# === B2-step2: g_evaluate_update accepts prefetched m_data pointer ===
# Change signature: add `const flo* m_data_ptr` (8 prefetched floats)
old_sig = """__device__ flo g_evaluate_update(   grid_cl*\tg,  // delete the __constant
\t\t\t\t\tconst\t\t\t\tflo*\t\tm_coords,\t\t\t// double[3]
\t\t\t\t\tconst\t\t\t\tflo\t\t\t\tslope,\t\t\t\t// double
\t\t\t\t\tconst\t\t\t\tflo\t\t\t\tv,\t\t\t\t\t// double
\t\t\t\t\t\t\t\t\t\tflo*\t\tderiv,\t\t\t\t// double[3]
\t\t\t\t\tconst\t\t\t\tflo\t\t\t\tepsilon_fl,
\t\t\t\t\tgrids_cl_other* grids_cuda
\t\t\t\t\t
) {"""
new_sig = """__device__ flo g_evaluate_update(   grid_cl*\tg,  // delete the __constant
\t\t\t\t\tconst\t\t\t\tflo*\t\tm_coords,\t\t\t// double[3]
\t\t\t\t\tconst\t\t\t\tflo\t\t\t\tslope,\t\t\t\t// double
\t\t\t\t\tconst\t\t\t\tflo\t\t\t\tv,\t\t\t\t\t// double
\t\t\t\t\t\t\t\t\t\tflo*\t\tderiv,\t\t\t\t// double[3]
\t\t\t\t\tconst\t\t\t\tflo\t\t\t\tepsilon_fl,
\t\t\t\t\tgrids_cl_other* grids_cuda,
\t\t\t\t\tconst flo* m_data_ptr   // B2: prefetched m_data[base..base+7]
) {"""
assert old_sig in s, "g_evaluate_update signature not found"
s = s.replace(old_sig, new_sig, 1)

# Remove base computation + replace 8 gathers with m_data_ptr reads
old_gather = """\tconst int x0 = a[0];
\tconst int y0 = a[1];
\tconst int z0 = a[2];

\tint base = (x0 + m_i * (y0 + m_j * z0)) * 8;
\t flo* base_ptr = &g->m_data[base]; // delete the constant

\tconst flo f000 = *base_ptr;
\tconst flo f100 = *(base_ptr + 1);
\tconst flo f010 = *(base_ptr + 2);
\tconst flo f110 = *(base_ptr + 3);
\tconst flo f001 = *(base_ptr + 4);
\tconst flo f101 = *(base_ptr + 5);
\tconst flo f011 = *(base_ptr + 6);
\tconst flo f111 = *(base_ptr + 7);"""
new_gather = """\tconst int x0 = a[0];
\tconst int y0 = a[1];
\tconst int z0 = a[2];

\t// B2: read from prefetched m_data (producer already gathered base..base+7)
\tconst flo f000 = m_data_ptr[0];
\tconst flo f100 = m_data_ptr[1];
\tconst flo f010 = m_data_ptr[2];
\tconst flo f110 = m_data_ptr[3];
\tconst flo f001 = m_data_ptr[4];
\tconst flo f101 = m_data_ptr[5];
\tconst flo f011 = m_data_ptr[6];
\tconst flo f111 = m_data_ptr[7];"""
assert old_gather in s, "g_evaluate_update gather block not found"
s = s.replace(old_gather, new_gather, 1)

open(F, "w").write(s)
print("B2-step2a: g_evaluate_update now takes prefetched m_data_ptr (interface refactored)")
