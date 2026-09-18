#ifndef HOST_PROBE_H
#define HOST_PROBE_H
// Unified host-phase probe: state lives in kernel2.cu (single definition),
// declared extern here. Active only with -DHOST_PROBE; zero cost otherwise.
#ifdef HOST_PROBE
#include <cstdio>
#include <ctime>
struct HostProbe {
    double t0 = 0; const char* last = "";
    void mark(const char* name);
};
extern HostProbe g_host_probe;
inline void hp_mark(const char* name) { g_host_probe.mark(name); }
#else
inline void hp_mark(const char*) {}
#endif
#endif
