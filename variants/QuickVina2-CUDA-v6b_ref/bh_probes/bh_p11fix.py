#!/usr/bin/env python3
# P11 fixes v2: brace via tail anchor + quasi renumber
Q = "/home/ubuntu/grad/variants/QuickVina2-CUDA-Q3"

p = Q + "/inc/cuda/bh_shared.h"
s = open(p).read()
old = "\t\t}\n\t}\n#endif\n\n#endif"
new = "\t\t}\n\t}\n#endif\n}\n#endif"
assert old in s, "tail anchor"
s = s.replace(old, new)
open(p, "w").write(s)
print("BRACE-FIXED")

import re
p = Q + "/inc/cuda/quasi_newton.cuh"
s = open(p).read()
cnt = [0]
def rep(m):
    cnt[0] += 1
    return f"conf_barrier((conf_id), {2000 + cnt[0] - 1})"
s2 = re.sub(r"conf_barrier\(\(conf_id\), 200000 \+ __LINE__\)", rep, s)
assert cnt[0] == 23, f"expected 23 sites, got {cnt[0]}"
open(p, "w").write(s2)
print("QUASI slots renumbered:", cnt[0])

# slot -> line map
map_lines = []
ln = 0
for i, line in enumerate(s2.split("\n")):
    if "conf_barrier((conf_id), 200" in line and "200000" not in line:
        map_lines.append(i + 1)
open("/tmp/bh_quasi_slot_lines.txt", "w").write("\n".join(f"{i} {l}" for i, l in enumerate(map_lines)))
print("slot->line:", list(enumerate(map_lines)))
