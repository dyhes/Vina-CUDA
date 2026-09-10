#ifndef COPY_UTILS_H_
#define COPY_UTILS_H_

#include <boost/ptr_container/ptr_vector.hpp>

struct model;
struct output_type;
struct precalculate_byatom;
struct igrid;
struct vec;
struct monte_carlo;

typedef boost::ptr_vector<output_type> output_container;


void mc_bfgs_kernel_wrapper(
    const monte_carlo & mc,
    const model & m,
    output_container & out,
    const precalculate_byatom & p,
    const igrid & ig,
    const vec& corner1,
    const vec& corner2,
    int num_tasks
);

#endif