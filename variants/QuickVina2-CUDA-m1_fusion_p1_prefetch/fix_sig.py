f = "inc/cuda/quasi_newton.cuh"
s = open(f).read()
# Fix: the broken insertion put ", const flo* cached = nullptr" AFTER ") {"
bad = ('\t\t\t\t\tconst\t\t\t\tflo\t\t\tepsilon_fl\n) {\n'
       '\t\t\t\t\t, const flo* cached = nullptr\n')
good = ('\t\t\t\t\tconst\t\t\t\tflo\t\t\tepsilon_fl,\n'
        '\t\t\t\t\tconst flo* cached = nullptr\n) {\n')
assert bad in s, "broken-insertion pattern not found"
s = s.replace(bad, good, 1)
open(f, "w").write(s)
print("OK: fixed g_evaluate signature (cached param before ) {)")
