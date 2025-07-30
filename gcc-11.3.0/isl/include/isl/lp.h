/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_LP_H
#define ISL_LP_H

#include <isl/aff.h>
#include <isl/val.h>
#include <isl/set_type.h>

enum isl_lp_result {
	isl_lp_error = -1,
	isl_lp_ok = 0,
	isl_lp_unbounded,
	isl_lp_empty
};

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_val *isl_basic_set_min_lp_val(__isl_keep isl_basic_set *bset,
	__isl_keep isl_aff *obj);
__isl_give isl_val *isl_basic_set_max_lp_val(__isl_keep isl_basic_set *bset,
	__isl_keep isl_aff *obj);

#if defined(__cplusplus)
}
#endif

#endif
