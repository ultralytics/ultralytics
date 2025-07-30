/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 */

#ifndef ISL_MORHP_H
#define ISL_MORHP_H

#include <stdio.h>
#include <isl/space.h>
#include <isl/mat.h>
#include <isl/set.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* An isl_morph is a "morphism" on (basic) sets.
 * "map" is an affine mapping from "dom" to "ran"
 * and "inv" is the inverse mapping.
 */
struct isl_morph {
	int ref;

	isl_basic_set *dom;
	isl_basic_set *ran;

	isl_mat *map;
	isl_mat *inv;
};
typedef struct isl_morph isl_morph;

isl_ctx *isl_morph_get_ctx(__isl_keep isl_morph *morph);

__isl_give isl_morph *isl_morph_alloc(
	__isl_take isl_basic_set *dom, __isl_take isl_basic_set *ran,
	__isl_take isl_mat *map, __isl_take isl_mat *inv);
__isl_give isl_morph *isl_morph_copy(__isl_keep isl_morph *morph);
__isl_give isl_morph *isl_morph_identity(__isl_keep isl_basic_set *bset);
void isl_morph_free(__isl_take isl_morph *morph);

__isl_give isl_space *isl_morph_get_dom_space(__isl_keep isl_morph *morph);
__isl_give isl_space *isl_morph_get_ran_space(__isl_keep isl_morph *morph);
__isl_give isl_multi_aff *isl_morph_get_var_multi_aff(
	__isl_keep isl_morph *morph);
unsigned isl_morph_dom_dim(__isl_keep isl_morph *morph, enum isl_dim_type type);
unsigned isl_morph_ran_dim(__isl_keep isl_morph *morph, enum isl_dim_type type);

__isl_give isl_morph *isl_morph_remove_dom_dims(__isl_take isl_morph *morph,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_morph *isl_morph_remove_ran_dims(__isl_take isl_morph *morph,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_morph *isl_morph_dom_params(__isl_take isl_morph *morph);
__isl_give isl_morph *isl_morph_ran_params(__isl_take isl_morph *morph);

__isl_give isl_morph *isl_morph_compose(__isl_take isl_morph *morph1,
	__isl_take isl_morph *morph2);
__isl_give isl_morph *isl_morph_inverse(__isl_take isl_morph *morph);

void isl_morph_print_internal(__isl_take isl_morph *morph, FILE *out);
void isl_morph_dump(__isl_take isl_morph *morph);

__isl_give isl_morph *isl_basic_set_variable_compression(
	__isl_keep isl_basic_set *bset, enum isl_dim_type type);
__isl_give isl_morph *isl_basic_set_parameter_compression(
	__isl_keep isl_basic_set *bset);
__isl_give isl_morph *isl_basic_set_full_compression(
	__isl_keep isl_basic_set *bset);

__isl_give isl_basic_set *isl_morph_basic_set(__isl_take isl_morph *morph,
	__isl_take isl_basic_set *bset);
__isl_give isl_set *isl_morph_set(__isl_take isl_morph *morph,
	__isl_take isl_set *set);
__isl_give isl_vec *isl_morph_vec(__isl_take isl_morph *morph,
	__isl_take isl_vec *vec);

#if defined(__cplusplus)
}
#endif

#endif
