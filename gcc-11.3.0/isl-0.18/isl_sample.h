/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_SAMPLE_H
#define ISL_SAMPLE_H

#include <isl/set.h>
#include <isl_tab.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_vec *isl_basic_set_sample_vec(__isl_take isl_basic_set *bset);
struct isl_vec *isl_basic_set_sample_bounded(struct isl_basic_set *bset);
__isl_give isl_vec *isl_basic_set_sample_with_cone(
	__isl_take isl_basic_set *bset, __isl_take isl_basic_set *cone);

__isl_give isl_basic_set *isl_basic_set_from_vec(__isl_take isl_vec *vec);

int isl_tab_set_initial_basis_with_cone(struct isl_tab *tab,
	struct isl_tab *tab_cone);
struct isl_vec *isl_tab_sample(struct isl_tab *tab);

#if defined(__cplusplus)
}
#endif

#endif
