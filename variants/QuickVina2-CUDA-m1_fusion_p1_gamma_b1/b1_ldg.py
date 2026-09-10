f = "inc/cuda/quasi_newton.cuh"
s = open(f).read()

# B1: g_evaluate_update grid gathers via __ldg (read-only L1 cache)
old = "\tflo* base_ptr = &g->m_data[base]; // delete the constant\n\n\tconst flo f000 = *base_ptr;\n\tconst flo f100 = *(base_ptr + 1);\n\tconst flo f010 = *(base_ptr + 2);\n\tconst flo f110 = *(base_ptr + 3);\n\tconst flo f001 = *(base_ptr + 4);\n\tconst flo f101 = *(base_ptr + 5);\n\tconst flo f011 = *(base_ptr + 6);\n\tconst flo f111 = *(base_ptr + 7);"

new = "\tconst flo* base_ptr = &g->m_data[base]; // delete the constant\n\n\tconst flo f000 = __ldg(base_ptr);\n\tconst flo f100 = __ldg(base_ptr + 1);\n\tconst flo f010 = __ldg(base_ptr + 2);\n\tconst flo f110 = __ldg(base_ptr + 3);\n\tconst flo f001 = __ldg(base_ptr + 4);\n\tconst flo f101 = __ldg(base_ptr + 5);\n\tconst flo f011 = __ldg(base_ptr + 6);\n\tconst flo f111 = __ldg(base_ptr + 7);"

assert old in s, "g_evaluate_update gather block not found"
s = s.replace(old, new, 1)
open(f, "w").write(s)
print("B1: g_evaluate_update 8 grid gathers -> __ldg")
