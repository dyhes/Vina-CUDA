#ifndef OUTPUT_TYPE_GPU_H_
#define OUTPUT_TYPE_GPU_H_

#include "size.h"
#include "common_gpu.cuh"

typedef double fl;

struct rigid_change_gpu {
    vec_gpu position;
    vec_gpu orientation;
};

template<class Size>
struct ligand_change_gpu {
    rigid_change_gpu rigid;
    fl torsions[Size::torsions_size];
    int torsions_size;
};

template<class Size>
struct change_gpu {
    ligand_change_gpu<Size> ligand;

    __device__ inline void init(int torsions_size);
    __device__ inline void init(change_gpu<Size> & g);
    __device__ inline void init(const change_gpu<Size> & g, int async);
    __device__ inline sz num_floats() const;
    __device__ inline fl operator()(sz index) const;
    __device__ inline fl & operator()(sz index);
};

template<class Size>
__device__ inline void change_gpu<Size>::init(change_gpu<Size> & g) {
    ligand.rigid = g.ligand.rigid;

    ligand.torsions_size = g.ligand.torsions_size;
    for(int i = 0; i < ligand.torsions_size; i++)
        ligand.torsions[i] = g.ligand.torsions[i];
}

template<class Size>
__device__ inline void change_gpu<Size>::init(const change_gpu<Size> & g, int async) {
    if(threadIdx.x == 0) ligand.rigid = g.ligand.rigid;
    if(threadIdx.x == 0) ligand.torsions_size = g.ligand.torsions_size;

    int loop_size = g.ligand.torsions_size > Size::torsions_size ? Size::torsions_size : g.ligand.torsions_size;
    for(int i = threadIdx.x; i < loop_size; i += blockDim.x)
        ligand.torsions[i] = g.ligand.torsions[i];
}

template<class Size>
__device__ inline void change_gpu<Size>::init(int torsions_size) {
    ligand.rigid.position.init(0, 0, 0);
    ligand.rigid.orientation.init(0, 0, 0);

    ligand.torsions_size = torsions_size;
    for(int i = 0; i < torsions_size; i++)
        ligand.torsions[i] = 0;
}

template<class Size>
__device__ inline sz change_gpu<Size>::num_floats() const { return 6 + ligand.torsions_size; }

template<class Size>
__device__ inline fl & change_gpu<Size>::operator()(sz index){
    switch(index) {
        case 0: ;
        case 1: ;
        case 2:
            return ligand.rigid.position[index];
        case 3: ;
        case 4: ;
        case 5: ;
            return ligand.rigid.orientation[index - 3];
        default:
            return ligand.torsions[index - 6];
    }
}

template<class Size>
__device__ inline fl change_gpu<Size>::operator()(sz index) const {
    fl f;
    switch(index) {
        case 0: ;
        case 1: ;
        case 2:
            f = ligand.rigid.position[index];
            break;
        case 3: ;
        case 4: ;
        case 5: ;
            f = ligand.rigid.orientation[index - 3];
            break;
        default:
            f = ligand.torsions[index - 6];
    }

    return f;
}

struct rigid_conf_gpu {
    vec_gpu position;
    qt_gpu orientation;
};

template<class Size>
struct ligand_conf_gpu {
    rigid_conf_gpu rigid;
    fl torsions[Size::torsions_size];
    int torsions_size;
};

template<class Size>
struct conf_gpu {
    ligand_conf_gpu<Size> ligand;

    __device__ inline void increment(
        const change_gpu<Size> & c,
        fl factor
    );

    __device__ inline void randomize(
        const vec_gpu & corner1,
        const vec_gpu & corner2,
        curandState & generator
    );

    __device__ inline void init(conf_gpu<Size> & conf);
    __device__ inline void init(conf_gpu<Size> & conf, int async);
};

__device__ inline void torsions_randomize_gpu(
    fl * torsions,
    int torsions_size,
    curandState & generator
) {
	for(int i = 0; i < torsions_size; i++)
		torsions[i] = random_fl_gpu(-pi_gpu, pi_gpu, generator);
}

__device__ inline void torsions_increment_gpu(
    fl * torsions,
    int torsions_size,
    const fl * c,
    fl factor
) { // new torsions are normalized
	for(int i = 0; i < torsions_size; i++) {
        fl a = factor * c[i];
        normalize_angle_gpu(a);
		torsions[i] += a;
		normalize_angle_gpu(torsions[i]);
	}
}

template<class Size>
__device__ inline void conf_gpu<Size>::init(conf_gpu<Size> & conf) {
    ligand.rigid = conf.ligand.rigid;
    
    ligand.torsions_size = conf.ligand.torsions_size;
    for(int i = 0; i < conf.ligand.torsions_size; i++)
        ligand.torsions[i] = conf.ligand.torsions[i];
}

template<class Size>
__device__ inline void conf_gpu<Size>::init(conf_gpu<Size> & conf, int async) {
    if(threadIdx.x == 0) ligand.rigid = conf.ligand.rigid;
    if(threadIdx.x == 0) ligand.torsions_size = conf.ligand.torsions_size;

    int loop_size = conf.ligand.torsions_size > Size::torsions_size ? Size::torsions_size : conf.ligand.torsions_size;
    for(int i = threadIdx.x; i < conf.ligand.torsions_size; i += blockDim.x)
        ligand.torsions[i] = conf.ligand.torsions[i];
}

template<class Size>
__device__ inline void conf_gpu<Size>::randomize(
    const vec_gpu & corner1,
    const vec_gpu & corner2,
    curandState & generator
) {
	ligand.rigid.position = random_in_box_gpu(corner1, corner2, generator);
    ligand.rigid.orientation = random_orientation_gpu(generator);
    torsions_randomize_gpu(ligand.torsions, ligand.torsions_size, generator);
}

template<class Size>
__device__ inline void conf_gpu<Size>::increment(
    const change_gpu<Size> & c,
    fl factor
) {
    ligand.rigid.position += factor * c.ligand.rigid.position;
	
    vec_gpu rotation;
    rotation = factor * c.ligand.rigid.orientation;
	quaternion_increment_gpu(ligand.rigid.orientation, rotation); // orientation does not get normalized; tests show rounding errors growing very slowly
    torsions_increment_gpu(
        ligand.torsions,
        ligand.torsions_size,
        c.ligand.torsions,
        factor
    );
}

template<class Size>
struct output_type_gpu {
    fl e;
    conf_gpu<Size> c;
    vec_gpu coords[Size::atoms_size];
    int coords_size;

    __device__ inline void init(int conf_torsions_size, fl _e);
};

template<class Size>
__device__ inline void output_type_gpu<Size>::init(
    int conf_torsions_size,
    fl _e
) {
    c.ligand.torsions_size = conf_torsions_size;
    e =_e;
}

#endif