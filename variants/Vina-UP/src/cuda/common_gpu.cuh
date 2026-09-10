#ifndef COMMON_GPU_H_
#define COMMON_GPU_H_

#include <cuda/std/cassert>
#include <curand_kernel.h>
#include <cuda/std/limits>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

typedef double fl;
typedef unsigned long sz;

__device__ const sz XS_TYPE_G0_GPU = 21;
__device__ const sz XS_TYPE_G1_GPU = 24;
__device__ const sz XS_TYPE_G2_GPU = 27;
__device__ const sz XS_TYPE_G3_GPU = 30;
__device__ const sz XS_TYPE_C_H_CG0_GPU = 19;
__device__ const sz XS_TYPE_C_H_CG1_GPU = 22;
__device__ const sz XS_TYPE_C_H_CG2_GPU = 25;
__device__ const sz XS_TYPE_C_H_CG3_GPU = 28;
__device__ const sz XS_TYPE_C_H_GPU = 0;
__device__ const sz XS_TYPE_C_P_CG0_GPU = 20;
__device__ const sz XS_TYPE_C_P_CG1_GPU = 23;
__device__ const sz XS_TYPE_C_P_CG2_GPU = 26;
__device__ const sz XS_TYPE_C_P_CG3_GPU = 29;
__device__ const sz XS_TYPE_C_P_GPU = 1;
__device__ const sz EL_TYPE_SIZE_GPU = 14;
__device__ const sz AD_TYPE_SIZE_GPU = 31;
__device__ const sz XS_TYPE_SIZE_GPU = 32;
__device__ const sz SY_TYPE_SIZE_GPU = 18;
__device__ const sz EL_TYPE_H_GPU = 0;
__device__ const sz max_sz_gpu = (cuda::std::numeric_limits<sz>::max)();
__device__ const fl max_fl_gpu = (cuda::std::numeric_limits<fl>::max)();
__device__ const fl epsilon_fl_gpu = (cuda::std::numeric_limits<fl>::epsilon)();
__device__ const fl pi_gpu = fl(3.1415926535897931);
__device__ const fl fl_tolerance_gpu = fl(0.001);

template<typename T>
__device__ inline T sqr_gpu(T x) { return x * x; }

template<typename T> // T = fl or vec
__device__ inline void curl_gpu(fl & e, T & deriv, fl v) {
	if(e > 0 && v < 0.1 * max_fl_gpu) { // FIXME authentic_v can be gotten rid of everywhere now
		fl tmp = (v < epsilon_fl_gpu) ? 0 : (v / (v + e));
		e *= tmp;
		deriv *= sqr_gpu(tmp);
	}
}

__device__ inline void curl_gpu(fl & e, fl v) {
	if(e > 0 && v < 0.1 * max_fl_gpu) {
		fl tmp = (v < epsilon_fl_gpu) ? 0 : (v / (v + e));
		e *= tmp;
	}
}

__device__ inline int random_int_gpu(int a, int b, curandState & generator) {
	assert(a <= b); // BOOST asserts this as well
	// typedef boost::uniform_int<int> distr;
	// boost::variate_generator<rng&, distr> r(generator, distr(a, b));
	// int tmp = r();
	fl r = curand_uniform(&generator);
	int tmp = (b - a + 1) * r + a;
	assert(tmp >= a);
	assert(tmp <= b);
	return tmp;
}

__device__ inline fl random_fl_gpu(fl a, fl b, curandState & generator) {
	assert(a < b); // BOOST also asserts a < b
	// typedef boost::uniform_real<fl> distr;
	// boost::variate_generator<rng&, distr> r(generator, distr(a, b));
	// fl tmp = r();
	fl r = curand_uniform(&generator);
	fl tmp = (b - a) * r + a;
	assert(tmp >= a);
	assert(tmp <= b);
	return tmp;
}

struct vec_gpu {
    fl data[3];

    __device__ inline void init(fl x, fl y, fl z);
    __device__ inline fl & operator[](sz i);
    __device__ inline const fl & operator[](sz i) const;
    __device__ inline vec_gpu operator+(const vec_gpu & v) const;
    __device__ inline fl norm_sqr() const;
	__device__ inline fl norm() const;
    __device__ inline const vec_gpu & operator*=(fl s);
    __device__ inline const vec_gpu & operator+=(const vec_gpu & v);
    __device__ inline void assign(fl s);
    __device__ inline vec_gpu operator-(const vec_gpu & v) const;
    __device__ inline fl operator*(const vec_gpu & v) const;
    __device__ inline const vec_gpu & operator-=(const vec_gpu & v);
};

__device__ inline void vec_gpu::init(fl x, fl y, fl z) { data[0] = x; data[1] = y, data[2] = z; }

__device__ inline fl & vec_gpu::operator[](sz i) { assert(i < 3); return data[i]; }

__device__ inline const fl & vec_gpu::operator[](sz i) const { assert(i < 3); return data[i]; }

__device__ inline vec_gpu vec_gpu::operator+(const vec_gpu & v) const {
    vec_gpu _v;
    _v.init(data[0] + v[0], data[1] + v[1], data[2] + v[2]);
    return _v;
}

__device__ inline vec_gpu operator*(fl s, const vec_gpu & v) {
    vec_gpu _v;
    _v.data[0] = s * v.data[0]; _v.data[1] = s * v.data[1]; _v.data[2] = s * v.data[2];
    return _v;
}

__device__ inline fl vec_gpu::norm_sqr() const {return sqr_gpu(data[0]) + sqr_gpu(data[1]) + sqr_gpu(data[2]);}

__device__ inline fl vec_gpu::norm() const {return std::sqrt(norm_sqr());}

__device__ inline const vec_gpu & vec_gpu::operator*=(fl s) {
    data[0] *= s; data[1] *= s; data[2] *= s;
    return *this;
}

__device__ inline const vec_gpu & vec_gpu::operator+=(const vec_gpu & v) {
    data[0] += v[0]; data[1] += v[1]; data[2] += v[2];
    return *this;
}

__device__ inline void vec_gpu::assign(fl s) { data[0] = data[1] = data[2] = s; }

__device__ inline vec_gpu vec_gpu::operator-(const vec_gpu & v) const {
    vec_gpu _v;
    _v[0] = data[0] - v[0]; _v[1] = data[1] - v[1]; _v[2] = data[2] - v[2];
    return _v;
}

__device__ inline fl vec_gpu::operator*(const vec_gpu & v) const {return data[0]*v[0] + data[1]*v[1] + data[2]*v[2];}

__device__ inline const vec_gpu & vec_gpu::operator-=(const vec_gpu & v) {
    data[0] -= v[0]; data[1] -= v[1]; data[2] -= v[2];
    return *this;
}


struct mat_gpu {
    fl data[9];

    __device__ inline vec_gpu operator*(const vec_gpu & v) const;
    __device__ inline fl & operator()(sz i, sz j);
};

__device__ inline vec_gpu mat_gpu::operator*(const vec_gpu & v) const {
    vec_gpu _v;
    _v[0] = data[0]*v[0] + data[3]*v[1] + data[6]*v[2];
    _v[1] = data[1]*v[0] + data[4]*v[1] + data[7]*v[2];
    _v[2] = data[2]*v[0] + data[5]*v[1] + data[8]*v[2];
    return _v;
}

__device__ inline fl & mat_gpu::operator()(sz i, sz j) {
    assert(i < 3);
    assert(j < 3); 
    return data[i + 3*j];
}

struct qt_gpu {
    fl a, b, c, d;

    __device__ inline qt_gpu & operator*=(const qt_gpu & q);
    __device__ inline qt_gpu operator*(const qt_gpu & q);
    __device__ inline qt_gpu & operator*=(fl);
};

__device__ inline qt_gpu & qt_gpu::operator*=(const qt_gpu & q) {
    fl ar = q.a, br = q.b, cr = q.c, dr = q.d;
    fl _a = a*ar - b*br - c*cr - d*dr;
    fl _b = a*br + b*ar + c*dr - d*cr;
    fl _c = a*cr - b*dr + c*ar + d*br;
    fl _d = a*dr + b*cr - c*br + d*ar;

    a = _a; b = _b; c = _c; d = _d;
    return *this;
}

__device__ inline qt_gpu & qt_gpu::operator*=(fl r) {
    a *= r; b *= r; c *= r; d *= r;
    return *this;
}

__device__ inline qt_gpu qt_gpu::operator*(const qt_gpu & q) {
    qt_gpu r;
    r = *this;
    r *= q;
    return r;
}

struct atom_gpu {
    enum t {EL, AD, XS, SY};
    sz el, ad, xs, sy;
    vec_gpu coords;

    __device__ inline sz get(t atom_typing_used) const;
};

__device__ inline sz atom_gpu::get(t atom_typing_used) const {
    switch(atom_typing_used) {
        case EL: return el;
        case AD: return ad;
        case XS: return xs;
        case SY: return sy;
        default: assert(false); return max_sz_gpu;
    }
}

__device__ inline sz num_atom_types(atom_gpu::t atom_typing_used) {
	switch(atom_typing_used) {
		case atom_gpu::EL: return EL_TYPE_SIZE_GPU;
		case atom_gpu::AD: return AD_TYPE_SIZE_GPU;
		case atom_gpu::XS: return XS_TYPE_SIZE_GPU;
		case atom_gpu::SY: return SY_TYPE_SIZE_GPU;
		default: assert(false); return max_sz_gpu;
	}
}

__device__ inline vec_gpu random_in_box_gpu(
    const vec_gpu & corner1,
    const vec_gpu & corner2,
    curandState & generator
) { // expects corner1[i] < corner2[i]
	vec_gpu tmp;
	tmp[0] = random_fl_gpu(corner1[0], corner2[0], generator);
    tmp[1] = random_fl_gpu(corner1[1], corner2[1], generator);
    tmp[2] = random_fl_gpu(corner1[2], corner2[2], generator);
	return tmp;
}

__device__ inline vec_gpu random_inside_sphere_gpu(curandState & generator) {
	fl r1 = random_fl_gpu(-1, 1, generator);
	fl r2 = random_fl_gpu(-1, 1, generator);
	fl r3 = random_fl_gpu(-1, 1, generator);
	fl r4 = random_fl_gpu(0, 1, generator);
		
	vec_gpu tmp;
	tmp.init(r1, r2, r3);
	fl v_norm = tmp.norm();
	tmp *= 1/v_norm * r4;
		
	return tmp;
}

__device__ inline vec_gpu cross_product_gpu(const vec_gpu & a, const vec_gpu & b) {
	vec_gpu _v;
	_v[0] = a[1]*b[2] - a[2]*b[1]; _v[1] = a[2]*b[0] - a[0]*b[2]; _v[2] = a[0]*b[1] - a[1]*b[0];
	return _v;
}

__device__ inline vec_gpu elementwise_product_gpu(const vec_gpu & a, const vec_gpu & b) {
	vec_gpu _v;
	_v[0] = a[0]*b[0]; _v[1] = a[1]*b[1]; _v[2] = a[2]*b[2];
	return _v;
}

__device__ inline bool eq_gpu(fl a, fl b) { return std::abs(a - b) < fl_tolerance_gpu; }

__device__ inline fl quaternion_norm_sqr_gpu(const qt_gpu & q) {return sqr_gpu(q.a) + sqr_gpu(q.b) + sqr_gpu(q.c) + sqr_gpu(q.d);}

__device__ inline bool quaternion_is_normalized_gpu(const qt_gpu & q) { return eq_gpu(quaternion_norm_sqr_gpu(q), 1); }

__device__ inline mat_gpu quaternion_to_r3_gpu(const qt_gpu & q) {
    assert(quaternion_is_normalized_gpu(q));

	const fl a = q.a;
	const fl b = q.b;
	const fl c = q.c;
	const fl d = q.d;

	const fl aa = a*a;
	const fl ab = a*b;
	const fl ac = a*c;
	const fl ad = a*d;
	const fl bb = b*b;
	const fl bc = b*c;
	const fl bd = b*d;
	const fl cc = c*c;
	const fl cd = c*d;
	const fl dd = d*d;

	assert(eq_gpu(aa+bb+cc+dd, 1));

	mat_gpu tmp;
	tmp(0, 0) = (aa+bb-cc-dd);
	tmp(0, 1) = 2*(-ad+bc);
	tmp(0, 2) = 2*(ac+bd);

	tmp(1, 0) = 2*(ad+bc);
	tmp(1, 1) = (aa-bb+cc-dd);
	tmp(1, 2) = 2*(-ab+cd);

	tmp(2, 0) = 2*(-ac+bd);
	tmp(2, 1) = 2*(ab+cd);
	tmp(2, 2) = (aa-bb-cc+dd);

	return tmp;
}

__device__ inline void normalize_angle_gpu(fl & x) {
	while(x > 3*pi_gpu || x < -3*pi_gpu) {
		if (x >  3*pi_gpu) {
			fl n = ( x - pi_gpu) / (2*pi_gpu);
			x -= 2*pi_gpu*std::ceil(n);
		}
		else {
			fl n = (-x - pi_gpu) / (2*pi_gpu);
			x += 2*pi_gpu*std::ceil(n);
		}
	}
	
	if(x > pi_gpu) // in (   pi, 3*pi]
		x -= 2*pi_gpu;
	if(x < -pi_gpu) // in [-3*pi,  -pi)
		x += 2*pi_gpu;
	assert(x >= -pi_gpu && x <= pi_gpu);
	// in [-pi, pi]
}

__device__ inline qt_gpu angle_to_quaternion_gpu(const vec_gpu & axis, fl angle) {
	assert(eq_gpu(axis.norm(), 1));
	normalize_angle_gpu(angle); // this is probably only necessary if angles can be very big
	fl c = std::cos(angle/2);
	fl s = std::sin(angle/2);
	
	qt_gpu q;
	q.a = c; q.b = s*axis[0]; q.c = s*axis[1]; q.d = s*axis[2];
	return q;
}

__device__ inline qt_gpu angle_to_quaternion_gpu(const vec_gpu & rotation) {
	fl angle = rotation.norm(); 
	if(angle > epsilon_fl_gpu) {
		vec_gpu axis = (1/angle) * rotation;
		return angle_to_quaternion_gpu(axis, angle);
	}
	qt_gpu qt_identity;
	qt_identity.a = 1;
	qt_identity.b = qt_identity.c = qt_identity.d = 0;
	return qt_identity;
}

__device__ inline void quaternion_normalize_approx_gpu(qt_gpu & q, const fl tolerance = 1e-6) {
	const fl s = quaternion_norm_sqr_gpu(q);
	if(std::abs(s - 1) > tolerance) {
		const fl a = std::sqrt(s);
		assert(a > epsilon_fl_gpu);
		q *= 1/a;
		assert(quaternion_is_normalized_gpu(q));
	}
}

__device__ inline fl random_normal_gpu(fl mean, fl sigma, curandState& generator) { // expects sigma >= 0
	assert(sigma >= 0); // BOOST asserts this as well
	// typedef boost::normal_distribution<fl> distr;
	// boost::variate_generator<rng&, distr> r(generator, distr(mean, sigma));
	// return r();
	fl r = curand_normal(&generator);
	fl tmp = sigma * r + mean;
	return tmp;
}

__device__ inline fl abs(qt_gpu & q) {return std::sqrt(q.a*q.a + q.b*q.b + q.c*q.c + q.d*q.d);}

__device__ inline qt_gpu random_orientation_gpu(curandState& generator) {
	qt_gpu q;
	q.a = random_normal_gpu(0, 1, generator);
	q.b = random_normal_gpu(0, 1, generator);
	q.c = random_normal_gpu(0, 1, generator);
	q.d = random_normal_gpu(0, 1, generator);
	fl nrm = abs(q);
	if(nrm > epsilon_fl_gpu) {
		q *= 1/nrm;
		assert(quaternion_is_normalized_gpu(q));
	}
	return q;
}

__device__ inline void quaternion_increment_gpu(qt_gpu & q, const vec_gpu & rotation) {
	assert(quaternion_is_normalized_gpu(q));
	q = angle_to_quaternion_gpu(rotation) * q;
	quaternion_normalize_approx_gpu(q); // normalization added in 1.1.2
}

template<int tile_size>
__device__ inline fl tile_sum(cg::thread_block_tile<tile_size> & tile, fl * temp_storage_fl, int valid_num) {
	int offset = 1;
	int thread = tile.thread_rank();
	while(offset < valid_num) {
		fl val = temp_storage_fl[thread];
		tile.sync();

		if(thread + offset < valid_num) val += temp_storage_fl[thread + offset];
		
		temp_storage_fl[thread] = val;
		tile.sync();

		offset *= 2;
	}
	
	return temp_storage_fl[0];
}

__device__ inline fl block_sum(fl * temp_storage_fl, int valid_num) {
	int offset = 1;
	int thread = threadIdx.x;
	while(offset < valid_num) {
		fl val = temp_storage_fl[thread];
		__syncthreads();

		if(thread + offset < valid_num) val += temp_storage_fl[thread + offset];
		
		temp_storage_fl[thread] = val;
		__syncthreads();

		offset *= 2;
	}
	
	return temp_storage_fl[0];
}

template<class T>
__device__ inline T block_min(T * temp_storage_int, int valid_num) {
	int offset = 1;
	int thread = threadIdx.x;
	while(offset < valid_num) {
		T val = temp_storage_int[thread];
		__syncthreads();

		if(thread + offset < valid_num)
			val = val < temp_storage_int[thread + offset] ? val : temp_storage_int[thread + offset];

		temp_storage_int[thread] = val;
		__syncthreads();

		offset *= 2;
	}

	return temp_storage_int[0];
}

#endif