/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_SPACE_H
#define ISL_SPACE_H

#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/printer.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct isl_space;
typedef struct isl_space isl_space;

enum isl_dim_type {
	isl_dim_cst,
	isl_dim_param,
	isl_dim_in,
	isl_dim_out,
	isl_dim_set = isl_dim_out,
	isl_dim_div,
	isl_dim_all
};

isl_ctx *isl_space_get_ctx(__isl_keep isl_space *dim);
__isl_give isl_space *isl_space_alloc(isl_ctx *ctx,
			unsigned nparam, unsigned n_in, unsigned n_out);
__isl_give isl_space *isl_space_set_alloc(isl_ctx *ctx,
			unsigned nparam, unsigned dim);
__isl_give isl_space *isl_space_params_alloc(isl_ctx *ctx, unsigned nparam);
__isl_give isl_space *isl_space_copy(__isl_keep isl_space *dim);
__isl_null isl_space *isl_space_free(__isl_take isl_space *space);

isl_bool isl_space_is_params(__isl_keep isl_space *space);
isl_bool isl_space_is_set(__isl_keep isl_space *space);
isl_bool isl_space_is_map(__isl_keep isl_space *space);

__isl_give isl_space *isl_space_set_tuple_name(__isl_take isl_space *dim,
	enum isl_dim_type type, const char *s);
isl_bool isl_space_has_tuple_name(__isl_keep isl_space *space,
	enum isl_dim_type type);
const char *isl_space_get_tuple_name(__isl_keep isl_space *dim,
				 enum isl_dim_type type);
__isl_give isl_space *isl_space_set_tuple_id(__isl_take isl_space *dim,
	enum isl_dim_type type, __isl_take isl_id *id);
__isl_give isl_space *isl_space_reset_tuple_id(__isl_take isl_space *dim,
	enum isl_dim_type type);
isl_bool isl_space_has_tuple_id(__isl_keep isl_space *dim,
	enum isl_dim_type type);
__isl_give isl_id *isl_space_get_tuple_id(__isl_keep isl_space *dim,
	enum isl_dim_type type);
__isl_give isl_space *isl_space_reset_user(__isl_take isl_space *space);

__isl_give isl_space *isl_space_set_dim_id(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id);
isl_bool isl_space_has_dim_id(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos);
__isl_give isl_id *isl_space_get_dim_id(__isl_keep isl_space *dim,
	enum isl_dim_type type, unsigned pos);

int isl_space_find_dim_by_id(__isl_keep isl_space *dim, enum isl_dim_type type,
	__isl_keep isl_id *id);
int isl_space_find_dim_by_name(__isl_keep isl_space *space,
	enum isl_dim_type type, const char *name);

isl_bool isl_space_has_dim_name(__isl_keep isl_space *space,
	enum isl_dim_type type, unsigned pos);
__isl_give isl_space *isl_space_set_dim_name(__isl_take isl_space *dim,
				 enum isl_dim_type type, unsigned pos,
				 __isl_keep const char *name);
__isl_keep const char *isl_space_get_dim_name(__isl_keep isl_space *dim,
				 enum isl_dim_type type, unsigned pos);

__isl_give isl_space *isl_space_extend(__isl_take isl_space *dim,
			unsigned nparam, unsigned n_in, unsigned n_out);
__isl_give isl_space *isl_space_add_dims(__isl_take isl_space *dim, enum isl_dim_type type,
		unsigned n);
__isl_give isl_space *isl_space_move_dims(__isl_take isl_space *dim,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n);
__isl_give isl_space *isl_space_insert_dims(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos, unsigned n);
__isl_give isl_space *isl_space_join(__isl_take isl_space *left,
	__isl_take isl_space *right);
__isl_give isl_space *isl_space_product(__isl_take isl_space *left,
	__isl_take isl_space *right);
__isl_give isl_space *isl_space_domain_product(__isl_take isl_space *left,
	__isl_take isl_space *right);
__isl_give isl_space *isl_space_range_product(__isl_take isl_space *left,
	__isl_take isl_space *right);
__isl_give isl_space *isl_space_factor_domain(__isl_take isl_space *space);
__isl_give isl_space *isl_space_factor_range(__isl_take isl_space *space);
__isl_give isl_space *isl_space_domain_factor_domain(
	__isl_take isl_space *space);
__isl_give isl_space *isl_space_domain_factor_range(
	__isl_take isl_space *space);
__isl_give isl_space *isl_space_range_factor_domain(
	__isl_take isl_space *space);
__isl_give isl_space *isl_space_range_factor_range(
	__isl_take isl_space *space);
__isl_give isl_space *isl_space_map_from_set(__isl_take isl_space *dim);
__isl_give isl_space *isl_space_map_from_domain_and_range(
	__isl_take isl_space *domain, __isl_take isl_space *range);
__isl_give isl_space *isl_space_reverse(__isl_take isl_space *dim);
__isl_give isl_space *isl_space_drop_dims(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned first, unsigned num);
__isl_give isl_space *isl_space_drop_inputs(__isl_take isl_space *dim,
		unsigned first, unsigned n);
__isl_give isl_space *isl_space_drop_outputs(__isl_take isl_space *dim,
		unsigned first, unsigned n);
__isl_give isl_space *isl_space_domain(__isl_take isl_space *dim);
__isl_give isl_space *isl_space_from_domain(__isl_take isl_space *dim);
__isl_give isl_space *isl_space_range(__isl_take isl_space *dim);
__isl_give isl_space *isl_space_from_range(__isl_take isl_space *dim);
__isl_give isl_space *isl_space_domain_map(__isl_take isl_space *space);
__isl_give isl_space *isl_space_range_map(__isl_take isl_space *space);
__isl_give isl_space *isl_space_params(__isl_take isl_space *space);
__isl_give isl_space *isl_space_set_from_params(__isl_take isl_space *space);

__isl_give isl_space *isl_space_align_params(__isl_take isl_space *dim1,
	__isl_take isl_space *dim2);

isl_bool isl_space_is_wrapping(__isl_keep isl_space *dim);
isl_bool isl_space_domain_is_wrapping(__isl_keep isl_space *space);
isl_bool isl_space_range_is_wrapping(__isl_keep isl_space *space);
__isl_give isl_space *isl_space_wrap(__isl_take isl_space *dim);
__isl_give isl_space *isl_space_unwrap(__isl_take isl_space *dim);

isl_bool isl_space_can_zip(__isl_keep isl_space *dim);
__isl_give isl_space *isl_space_zip(__isl_take isl_space *dim);

isl_bool isl_space_can_curry(__isl_keep isl_space *space);
__isl_give isl_space *isl_space_curry(__isl_take isl_space *space);

isl_bool isl_space_can_range_curry(__isl_keep isl_space *space);
__isl_give isl_space *isl_space_range_curry(__isl_take isl_space *space);

isl_bool isl_space_can_uncurry(__isl_keep isl_space *space);
__isl_give isl_space *isl_space_uncurry(__isl_take isl_space *space);

isl_bool isl_space_is_domain(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_bool isl_space_is_range(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_bool isl_space_is_equal(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_bool isl_space_has_equal_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2);
isl_bool isl_space_tuple_is_equal(__isl_keep isl_space *space1,
	enum isl_dim_type type1, __isl_keep isl_space *space2,
	enum isl_dim_type type2);
isl_bool isl_space_match(__isl_keep isl_space *space1, enum isl_dim_type type1,
	__isl_keep isl_space *space2, enum isl_dim_type type2);
ISL_DEPRECATED
int isl_space_tuple_match(__isl_keep isl_space *space1, enum isl_dim_type type1,
	__isl_keep isl_space *space2, enum isl_dim_type type2);
ISL_DEPRECATED
int isl_space_compatible(__isl_keep isl_space *dim1,
	__isl_keep isl_space *dim2);
unsigned isl_space_dim(__isl_keep isl_space *dim, enum isl_dim_type type);

__isl_give char *isl_space_to_str(__isl_keep isl_space *space);
__isl_give isl_printer *isl_printer_print_space(__isl_take isl_printer *p,
	__isl_keep isl_space *dim);
void isl_space_dump(__isl_keep isl_space *dim);

#if defined(__cplusplus)
}
#endif

#endif
