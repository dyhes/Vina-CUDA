// [v7] ric_gpu.cu — GPU-side generation of rand_maps + ric (random initial conformations)
// Replaces the host serial loop (~210ms/ligand: 8192 confs x ~13 draws through
// boost::variate_generator constructed per draw). GPU version: one kernel,
// 8192 threads, curand_MTGP32, identical DRAW SEMANTICS to the host path:
//   random_in_box:   3x uniform(a,b)   (corner1[k] + u*(corner2[k]-corner1[k]))
//   random_orientation: 4x normal(0,1) via Box-Muller WITHOUT tail caching
//                      (matches host ctor-per-call: each normal consumes 2 draws)
//   torsions:        1x uniform(-pi, pi) each
// rand_maps (per ligand): int_map[i] ~ U{0..n_tors}, pi_map[i] ~ U(-pi,pi),
//   sphere_map[i] ~ unit ball via rejection (3 uniforms per attempt).
// Output layout matches host structs exactly.
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "kernel2.h"

#define PI_F 3.14159265358979f

// generate rand_maps for one ligand: one thread per map slot
__global__ void v7_randmaps_kernel(random_maps* rm, int n_tors,
                                   unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAX_NUM_OF_RANDOM_MAP) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, 0, &st);
    // int in [0, n_tors] inclusive — host used uniform_int(0, n_tors)
    float u1 = curand_uniform(&st);
    int iv = (int)(u1 * (n_tors + 1));
    if (iv > n_tors) iv = n_tors;
    rm->int_map[i] = iv;
    float u2 = curand_uniform(&st);
    rm->pi_map[i] = -PI_F + u2 * 2.0f * PI_F;
    // unit-ball rejection
    float r1, r2, r3;
    do {
        r1 = 2.0f * curand_uniform(&st) - 1.0f;
        r2 = 2.0f * curand_uniform(&st) - 1.0f;
        r3 = 2.0f * curand_uniform(&st) - 1.0f;
    } while (r1*r1 + r2*r2 + r3*r3 >= 1.0f);
    rm->sphere_map[i][0] = r1;
    rm->sphere_map[i][1] = r2;
    rm->sphere_map[i][2] = r3;
}

// generate ric (random initial conformations): one thread per conf
__global__ void v7_ric_kernel(output_type_cl* ric, int thread,
                              float cx, float cy, float cz,
                              float sx, float sy, float sz,
                              int n_tors, unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= thread) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, 1000000u + i, 0, &st);

    // random_in_box: 3 uniforms mapped per axis
    float px = cx + curand_uniform(&st) * sx;
    float py = cy + curand_uniform(&st) * sy;
    float pz = cz + curand_uniform(&st) * sz;
    ric[i].position[0] = px;
    ric[i].position[1] = py;
    ric[i].position[2] = pz;

    // random_orientation: 4 normals (Box-Muller, no tail cache), normalized
    float q[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        float u1 = curand_uniform(&st);
        float u2 = curand_uniform(&st);
        if (u1 < 1e-12f) u1 = 1e-12f;
        q[k] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI_F * u2);
    }
    float nrm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (nrm > 2.2e-16f) {
        ric[i].orientation[0] = q[0]/nrm;
        ric[i].orientation[1] = q[1]/nrm;
        ric[i].orientation[2] = q[2]/nrm;
        ric[i].orientation[3] = q[3]/nrm;
    } else {
        ric[i].orientation[0] = 1.0f;
        ric[i].orientation[1] = 0.0f;
        ric[i].orientation[2] = 0.0f;
        ric[i].orientation[3] = 0.0f;
    }

    for (int t = 0; t < n_tors; t++) {
        float u = curand_uniform(&st);
        ric[i].lig_torsion[t] = -PI_F + u * 2.0f * PI_F;
    }
    ric[i].e = 0.0f;
}

extern "C" void v7_gen_randmaps_gpu(random_maps* d_rm, int n_tors,
                                    unsigned long long seed, cudaStream_t s)
{
    int blocks = (MAX_NUM_OF_RANDOM_MAP + 255) / 256;
    v7_randmaps_kernel<<<blocks, 256, 0, s>>>(d_rm, n_tors, seed);
}

extern "C" void v7_gen_ric_gpu(output_type_cl* d_ric, int thread,
                               float cx, float cy, float cz,
                               float sx, float sy, float sz,
                               int n_tors, unsigned long long seed, cudaStream_t s)
{
    int blocks = (thread + 255) / 256;
    v7_ric_kernel<<<blocks, 256, 0, s>>>(d_ric, thread, cx, cy, cz, sx, sy, sz, n_tors, seed);
}
