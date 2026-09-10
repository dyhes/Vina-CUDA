#ifndef VISITED_H
#define VISITED_H
#include "kernel2.h"

// BUG 3 fix: bitMask must be contiguous across all dimensions
// position[0..2] → bits 0-2, orientation[0..2] → bits 3-5, torsion[0..N] → bits 6+
__device__ inline void ele_init_cl(ele_cl* present, output_type_cl* x, float f_cl, change_cl* d, int lig_torsion_size)
{
	for (int i = 0; i < 3; i++) present->x_cl.position[i] = x->position[i];
	for (int i = 0; i < 4; i++) present->x_cl.orientation[i] = x->orientation[i];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++) present->x_cl.lig_torsion[i] = x->lig_torsion[i];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++) present->x_cl.flex_torsion[i] = x->flex_torsion[i];
	present->energy = f_cl;

	present->d_zero = 0;
	present->d_positive = 0;
	const long ONE = 1;
	long bitMask = 0;

	// position: bits 0-2
	for (int i = 0; i < 3; i++) {
		bitMask = ONE << i;
		if (d->position[i] == 0) present->d_zero |= bitMask;
		else if (d->position[i] > 0) present->d_positive |= bitMask;
	}
	// orientation: bits 3-5
	for (int i = 0; i < 3; i++) {
		bitMask = ONE << (3 + i);
		if (d->orientation[i] == 0) present->d_zero |= bitMask;
		else if (d->orientation[i] > 0) present->d_positive |= bitMask;
	}
	// lig_torsion: bits 6+
	for (int i = 0; i < lig_torsion_size; i++) {
		bitMask = ONE << (6 + i);
		if (d->lig_torsion[i] == 0) present->d_zero |= bitMask;
		else if (d->lig_torsion[i] > 0) present->d_positive |= bitMask;
	}
}

__device__ inline void visited_init(visited_cl* visited)
{
	visited->n_variable = 0;
	visited->p = 0;
	visited->full = false;
	visited->index = 0;
}

// BUG 5 fix: store by VALUE (not pointer) — list_cl is now ele_cl[], not ele_cl*[]
// BUG 6 fix: threshold = 5*n_variable (not 10*n_variable)
__device__ inline void add(visited_cl* visited, output_type_cl* conf_v, float f, change_cl* change_v, int lig_torsion_size)
{
	ele_cl e;
	ele_init_cl(&e, conf_v, f, change_v, lig_torsion_size);

	if (visited->index == 0)
	{
		visited->n_variable = lig_torsion_size + 6;
	}

	if (!visited->full)
	{
		visited->list_cl[visited->index] = e;  // value copy, no dangling pointer
		visited->index++;
		if (visited->index >= 5 * visited->n_variable)
		{
			visited->full = true;
			visited->p = 0;
		}
	}
	else
	{
		visited->list_cl[visited->p] = e;  // circular buffer
		visited->p = (visited->p + 1) % (5 * visited->n_variable);
	}
}

// BUG 2 fix: return true when all dimensions pass (significant), false when any fails
// BUG 3 fix: bitMask contiguous — position bits 0-2, orientation bits 3-5, torsion bits 6+
// BUG 4 fix: torsion loop reads lig_torsion[i] (not position[i])
// BUG 7: formula uses !(nowPositive ^ newXBigger ^ newYBigger) — matches CPU exactly
__device__ bool check(visited_cl* visited, output_type_cl* now_x, float now_f, change_cl* now_d, int neighbor, int lig_torsion_size)
{
	bool newXBigger, newYBigger;
	const long ONE = 1;
	long bitMask = 0;
	newYBigger = (now_f - visited->list_cl[neighbor].energy) > 0;

	// position: bits 0-2
	for (int i = 0; i < 3; i++)
	{
		bitMask = ONE << i;
		if ((visited->list_cl[neighbor].d_zero & bitMask) || !(now_d->position[i])) {
			continue;
		}
		else
		{
			const bool nowPositive = now_d->position[i] > 0;
			const bool dPositive = visited->list_cl[neighbor].d_positive & bitMask;
			if (nowPositive ^ dPositive) {
				continue;
			}
			else {
				newXBigger = (now_x->position[i] - visited->list_cl[neighbor].x_cl.position[i]) > 0;
				if (!(nowPositive ^ newXBigger ^ newYBigger)) {
					continue;
				}
				else {
					return false;
				}
			}
		}
	}
	// orientation: bits 3-5
	for (int i = 0; i < 3; i++)
	{
		bitMask = ONE << (3 + i);
		if ((visited->list_cl[neighbor].d_zero & bitMask) || !(now_d->orientation[i])) {
			continue;
		}
		else
		{
			const bool nowPositive = now_d->orientation[i] > 0;
			const bool dPositive = visited->list_cl[neighbor].d_positive & bitMask;
			if (nowPositive ^ dPositive) {
				continue;
			}
			else {
				newXBigger = (now_x->orientation[i] - visited->list_cl[neighbor].x_cl.orientation[i]) > 0;
				if (!(nowPositive ^ newXBigger ^ newYBigger)) {
					continue;
				}
				else {
					return false;
				}
			}
		}
	}
	// lig_torsion: bits 6+
	for (int i = 0; i < lig_torsion_size; i++)
	{
		bitMask = ONE << (6 + i);
		if ((visited->list_cl[neighbor].d_zero & bitMask) || !(now_d->lig_torsion[i])) {
			continue;
		}
		else
		{
			const bool nowPositive = now_d->lig_torsion[i] > 0;  // BUG 4 fix: lig_torsion, not position
			const bool dPositive = visited->list_cl[neighbor].d_positive & bitMask;
			if (nowPositive ^ dPositive) {
				continue;
			}
			else {
				newXBigger = (now_x->lig_torsion[i] - visited->list_cl[neighbor].x_cl.lig_torsion[i]) > 0;
				if (!(nowPositive ^ newXBigger ^ newYBigger)) {
					continue;
				}
				else {
					return false;
				}
			}
		}
	}
	return true;  // BUG 2 fix: all dimensions passed → significant
}

__device__ inline float dist2_cl(output_type_cl* now, visited_cl* visited, int neighbor, int lig_torsion_size) {
	float out = 0;
	for (int i = 0; i < 3; i++) {
		float d = visited->list_cl[neighbor].x_cl.position[i] - now->position[i];
		out += d * d;
	}
	for (int i = 0; i < 4; i++) {
		float d = visited->list_cl[neighbor].x_cl.orientation[i] - now->orientation[i];
		out += d * d;
	}
	for (int i = 0; i < lig_torsion_size; i++) {
		float d = visited->list_cl[neighbor].x_cl.lig_torsion[i] - now->lig_torsion[i];
		out += d * d;
	}
	return out;
}

// CPU circularvisited::interesting() logic:
//   returns -1 (interesting) if list empty or not full
//   returns -1 (interesting) if any check() passes
//   returns i (not interesting) if all checks fail
// GPU returns bool: true=interesting, false=not interesting
// maxCheck = 4*n_variable (matching CPU), not n_variable
__device__ bool interesting(output_type_cl* conf_v, float f, change_cl* g, visited_cl* visited, int lig_torsion_size)
{
	int len = visited->index;
	if (len == 0) {
		return true;  // empty → interesting
	}
	if (!visited->full) {
		return true;  // not full → interesting
	}

	// full == true: check nearest neighbors
	float dist[SIZE_OF_LIST];
	bool notPicked[SIZE_OF_LIST];

	for (int i = 0; i < len; i++)
	{
		dist[i] = dist2_cl(conf_v, visited, i, lig_torsion_size);
		notPicked[i] = true;
	}

	float min_val = 1e10;
	int p = 0;
	int maxCheck = 4 * visited->n_variable;  // CPU: get_maxCheck() = 4*n_variable
	if (maxCheck > len) maxCheck = len;

	for (int i = 0; i < maxCheck; i++)
	{
		min_val = 1e10;
		for (int j = 0; j < len; j++) {
			if (notPicked[j] && (dist[j] < min_val)) {
				p = j;
				min_val = dist[j];
			}
		}
		notPicked[p] = false;
		if (check(visited, conf_v, f, g, p, lig_torsion_size)) {
			return true;  // interesting
		}
	}
	return false;  // not interesting
}

#endif
