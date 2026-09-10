f = "inc/cuda/quasi_newton.cuh"
s = open(f).read()

# Fix: commit_t type -> auto
old1 = "cuda::pipeline::commit_t last_commit;"
new1 = "auto last_commit;"
assert old1 in s, "commit_t pattern not found"
s = s.replace(old1, new1, 1)

# Fix: ::cuda::pipeline::copy free function (force namespace)
old2 = "decltype(pipeline)::copy(pipeline,"
new2 = "::cuda::pipeline::copy(pipeline,"
assert old2 in s, "copy pattern not found"
s = s.replace(old2, new2, 1)

open(f, "w").write(s)
print("OK: commit_t->auto, ::cuda::pipeline::copy")
