f = "inc/cuda/quasi_newton.cuh"
s = open(f).read()

# The 8-corner read block (appears twice: g_evaluate + g_evaluate_update)
old = (
    "const flo f000 = *base_ptr;\n"
    "const flo f100 = *(base_ptr + 1);\n"
    "const flo f010 = *(base_ptr + 2);\n"
    "const flo f110 = *(base_ptr + 3);\n"
    "const flo f001 = *(base_ptr + 4);\n"
    "const flo f101 = *(base_ptr + 5);\n"
    "const flo f011 = *(base_ptr + 6);\n"
    "const flo f111 = *(base_ptr + 7);\n"
)
new = (
    "const flo f000 = __ldg(base_ptr);\n"
    "const flo f100 = __ldg(base_ptr + 1);\n"
    "const flo f010 = __ldg(base_ptr + 2);\n"
    "const flo f110 = __ldg(base_ptr + 3);\n"
    "const flo f001 = __ldg(base_ptr + 4);\n"
    "const flo f101 = __ldg(base_ptr + 5);\n"
    "const flo f011 = __ldg(base_ptr + 6);\n"
    "const flo f111 = __ldg(base_ptr + 7);\n"
)
cnt = s.count(old)
assert cnt >= 1, "8-corner read block not found"
s = s.replace(old, new)
open(f, "w").write(s)
print(f"OK: replaced {cnt} 8-corner read block(s) with __ldg read-only-cache hint")
