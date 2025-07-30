/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_EQUALITIES_H
#define ISL_EQUALITIES_H

#include <isl/set.h>
#include <isl/mat.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_mat *isl_mat_final_variable_compression(__isl_take isl_mat *B,
	int first, __isl_give isl_mat **T2);
__isl_give isl_mat *isl_mat_variable_compression(__isl_take isl_mat *B,
	__isl_give isl_mat **T2);
struct isl_mat *isl_mat_parameter_compression(
			struct isl_mat *B, struct isl_vec *d);
__isl_give isl_mat *isl_mat_parameter_compression_ext(__isl_take isl_mat *B,
	__isl_take isl_mat *A);
struct isl_basic_set *isl_basic_set_remove_equalities(
	struct isl_basic_set *bset, struct isl_mat **T, struct isl_mat **T2);

#if defined(__cplusplus)
}
#endif

#endif
