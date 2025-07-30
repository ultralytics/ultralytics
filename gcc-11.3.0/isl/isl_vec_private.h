#ifndef ISL_VEC_PRIVATE_H
#define ISL_VEC_PRIVATE_H

#include <isl_blk.h>
#include <isl/vec.h>

struct isl_vec {
	int ref;

	struct isl_ctx *ctx;

	unsigned size;
	isl_int *el;

	struct isl_blk block;
};

uint32_t isl_vec_get_hash(__isl_keep isl_vec *vec);

__isl_give isl_vec *isl_vec_cow(__isl_take isl_vec *vec);

void isl_vec_lcm(struct isl_vec *vec, isl_int *lcm);
int isl_vec_get_element(__isl_keep isl_vec *vec, int pos, isl_int *v);
__isl_give isl_vec *isl_vec_set(__isl_take isl_vec *vec, isl_int v);

__isl_give isl_vec *isl_vec_expand(__isl_take isl_vec *vec, int pos, int n,
	int *exp, int expanded);

#endif
