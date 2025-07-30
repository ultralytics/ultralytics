/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_MAP_H
#define ISL_MAP_H

#include <stdio.h>

#include <isl/ctx.h>
#include <isl/space.h>
#include <isl/vec.h>
#include <isl/mat.h>
#include <isl/printer.h>
#include <isl/local_space.h>
#include <isl/aff_type.h>
#include <isl/list.h>
#include <isl/map_type.h>
#include <isl/val.h>
#include <isl/stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* General notes:
 *
 * All structures are reference counted to allow reuse without duplication.
 * A *_copy operation will increase the reference count, while a *_free
 * operation will decrease the reference count and only actually release
 * the structures when the reference count drops to zero.
 *
 * Functions that return an isa structure will in general _destroy_
 * all argument isa structures (the obvious execption begin the _copy
 * functions).  A pointer passed to such a function may therefore
 * never be used after the function call.  If you want to keep a
 * reference to the old structure(s), use the appropriate _copy function.
 */

unsigned isl_basic_map_n_in(const struct isl_basic_map *bmap);
unsigned isl_basic_map_n_out(const struct isl_basic_map *bmap);
unsigned isl_basic_map_n_param(const struct isl_basic_map *bmap);
unsigned isl_basic_map_n_div(const struct isl_basic_map *bmap);
unsigned isl_basic_map_total_dim(const struct isl_basic_map *bmap);
unsigned isl_basic_map_dim(__isl_keep isl_basic_map *bmap,
				enum isl_dim_type type);

unsigned isl_map_n_in(const struct isl_map *map);
unsigned isl_map_n_out(const struct isl_map *map);
unsigned isl_map_n_param(const struct isl_map *map);
unsigned isl_map_dim(__isl_keep isl_map *map, enum isl_dim_type type);

isl_ctx *isl_basic_map_get_ctx(__isl_keep isl_basic_map *bmap);
isl_ctx *isl_map_get_ctx(__isl_keep isl_map *map);
__isl_give isl_space *isl_basic_map_get_space(__isl_keep isl_basic_map *bmap);
__isl_give isl_space *isl_map_get_space(__isl_keep isl_map *map);

__isl_give isl_aff *isl_basic_map_get_div(__isl_keep isl_basic_map *bmap,
	int pos);

__isl_give isl_local_space *isl_basic_map_get_local_space(
	__isl_keep isl_basic_map *bmap);

__isl_give isl_basic_map *isl_basic_map_set_tuple_name(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type, const char *s);
const char *isl_basic_map_get_tuple_name(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type);
isl_bool isl_map_has_tuple_name(__isl_keep isl_map *map,
	enum isl_dim_type type);
const char *isl_map_get_tuple_name(__isl_keep isl_map *map,
	enum isl_dim_type type);
__isl_give isl_map *isl_map_set_tuple_name(__isl_take isl_map *map,
	enum isl_dim_type type, const char *s);
const char *isl_basic_map_get_dim_name(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos);
isl_bool isl_map_has_dim_name(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos);
const char *isl_map_get_dim_name(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos);
__isl_give isl_basic_map *isl_basic_map_set_dim_name(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, const char *s);
__isl_give isl_map *isl_map_set_dim_name(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, const char *s);

__isl_give isl_basic_map *isl_basic_map_set_tuple_id(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, __isl_take isl_id *id);
__isl_give isl_map *isl_map_set_dim_id(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id);
isl_bool isl_basic_map_has_dim_id(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos);
isl_bool isl_map_has_dim_id(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos);
__isl_give isl_id *isl_map_get_dim_id(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos);
__isl_give isl_map *isl_map_set_tuple_id(__isl_take isl_map *map,
	enum isl_dim_type type, __isl_take isl_id *id);
__isl_give isl_map *isl_map_reset_tuple_id(__isl_take isl_map *map,
	enum isl_dim_type type);
isl_bool isl_map_has_tuple_id(__isl_keep isl_map *map, enum isl_dim_type type);
__isl_give isl_id *isl_map_get_tuple_id(__isl_keep isl_map *map,
	enum isl_dim_type type);
__isl_give isl_map *isl_map_reset_user(__isl_take isl_map *map);

int isl_basic_map_find_dim_by_name(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, const char *name);
int isl_map_find_dim_by_id(__isl_keep isl_map *map, enum isl_dim_type type,
	__isl_keep isl_id *id);
int isl_map_find_dim_by_name(__isl_keep isl_map *map, enum isl_dim_type type,
	const char *name);

int isl_basic_map_is_rational(__isl_keep isl_basic_map *bmap);

__isl_give isl_basic_map *isl_basic_map_identity(__isl_take isl_space *dim);
__isl_null isl_basic_map *isl_basic_map_free(__isl_take isl_basic_map *bmap);
__isl_give isl_basic_map *isl_basic_map_copy(__isl_keep isl_basic_map *bmap);
__isl_give isl_basic_map *isl_basic_map_equal(
	__isl_take isl_space *dim, unsigned n_equal);
__isl_give isl_basic_map *isl_basic_map_less_at(__isl_take isl_space *dim,
	unsigned pos);
__isl_give isl_basic_map *isl_basic_map_more_at(__isl_take isl_space *dim,
	unsigned pos);
__isl_give isl_basic_map *isl_basic_map_empty(__isl_take isl_space *dim);
__isl_give isl_basic_map *isl_basic_map_universe(__isl_take isl_space *dim);
__isl_give isl_basic_map *isl_basic_map_nat_universe(__isl_take isl_space *dim);
__isl_give isl_basic_map *isl_basic_map_remove_redundancies(
	__isl_take isl_basic_map *bmap);
__isl_give isl_map *isl_map_remove_redundancies(__isl_take isl_map *map);
__isl_give isl_basic_map *isl_map_simple_hull(__isl_take isl_map *map);
__isl_export
__isl_give isl_basic_map *isl_map_unshifted_simple_hull(
	__isl_take isl_map *map);
__isl_give isl_basic_map *isl_map_plain_unshifted_simple_hull(
	__isl_take isl_map *map);
__isl_give isl_basic_map *isl_map_unshifted_simple_hull_from_map_list(
	__isl_take isl_map *map, __isl_take isl_map_list *list);

__isl_export
__isl_give isl_basic_map *isl_basic_map_intersect_domain(
		__isl_take isl_basic_map *bmap,
		__isl_take isl_basic_set *bset);
__isl_export
__isl_give isl_basic_map *isl_basic_map_intersect_range(
		__isl_take isl_basic_map *bmap,
		__isl_take isl_basic_set *bset);
__isl_export
__isl_give isl_basic_map *isl_basic_map_intersect(
		__isl_take isl_basic_map *bmap1,
		__isl_take isl_basic_map *bmap2);
__isl_give isl_basic_map *isl_basic_map_list_intersect(
	__isl_take isl_basic_map_list *list);
__isl_export
__isl_give isl_map *isl_basic_map_union(
		__isl_take isl_basic_map *bmap1,
		__isl_take isl_basic_map *bmap2);
__isl_export
__isl_give isl_basic_map *isl_basic_map_apply_domain(
		__isl_take isl_basic_map *bmap1,
		__isl_take isl_basic_map *bmap2);
__isl_export
__isl_give isl_basic_map *isl_basic_map_apply_range(
		__isl_take isl_basic_map *bmap1,
		__isl_take isl_basic_map *bmap2);
__isl_export
__isl_give isl_basic_map *isl_basic_map_affine_hull(
		__isl_take isl_basic_map *bmap);
__isl_give isl_basic_map *isl_basic_map_preimage_domain_multi_aff(
	__isl_take isl_basic_map *bmap, __isl_take isl_multi_aff *ma);
__isl_give isl_basic_map *isl_basic_map_preimage_range_multi_aff(
	__isl_take isl_basic_map *bmap, __isl_take isl_multi_aff *ma);
__isl_export
__isl_give isl_basic_map *isl_basic_map_reverse(__isl_take isl_basic_map *bmap);
__isl_give isl_basic_set *isl_basic_map_domain(__isl_take isl_basic_map *bmap);
__isl_give isl_basic_set *isl_basic_map_range(__isl_take isl_basic_map *bmap);
__isl_give isl_basic_map *isl_basic_map_domain_map(
	__isl_take isl_basic_map *bmap);
__isl_give isl_basic_map *isl_basic_map_range_map(
	__isl_take isl_basic_map *bmap);
__isl_give isl_basic_map *isl_basic_map_remove_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_basic_map *isl_basic_map_eliminate(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n);
ISL_DEPRECATED
__isl_give isl_basic_map *isl_basic_map_from_basic_set(
	__isl_take isl_basic_set *bset, __isl_take isl_space *dim);
__isl_export
__isl_give isl_basic_map *isl_basic_map_sample(__isl_take isl_basic_map *bmap);
__isl_export
__isl_give isl_basic_map *isl_basic_map_detect_equalities(
						__isl_take isl_basic_map *bmap);
__isl_give isl_basic_map *isl_basic_map_read_from_file(isl_ctx *ctx,
	FILE *input);
__isl_constructor
__isl_give isl_basic_map *isl_basic_map_read_from_str(isl_ctx *ctx,
	const char *str);
__isl_give isl_map *isl_map_read_from_file(isl_ctx *ctx, FILE *input);
__isl_constructor
__isl_give isl_map *isl_map_read_from_str(isl_ctx *ctx, const char *str);
void isl_basic_map_dump(__isl_keep isl_basic_map *bmap);
void isl_map_dump(__isl_keep isl_map *map);
__isl_give char *isl_basic_map_to_str(__isl_keep isl_basic_map *bmap);
__isl_give isl_printer *isl_printer_print_basic_map(
	__isl_take isl_printer *printer, __isl_keep isl_basic_map *bmap);
__isl_give char *isl_map_to_str(__isl_keep isl_map *map);
__isl_give isl_printer *isl_printer_print_map(__isl_take isl_printer *printer,
	__isl_keep isl_map *map);
__isl_give isl_basic_map *isl_basic_map_fix_si(__isl_take isl_basic_map *bmap,
		enum isl_dim_type type, unsigned pos, int value);
__isl_give isl_basic_map *isl_basic_map_fix_val(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *v);
__isl_give isl_basic_map *isl_basic_map_lower_bound_si(
		__isl_take isl_basic_map *bmap,
		enum isl_dim_type type, unsigned pos, int value);
__isl_give isl_basic_map *isl_basic_map_upper_bound_si(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, int value);

struct isl_basic_map *isl_basic_map_sum(
		struct isl_basic_map *bmap1, struct isl_basic_map *bmap2);
struct isl_basic_map *isl_basic_map_neg(struct isl_basic_map *bmap);

__isl_give isl_map *isl_map_sum(__isl_take isl_map *map1,
	__isl_take isl_map *map2);
__isl_give isl_map *isl_map_neg(__isl_take isl_map *map);
__isl_give isl_map *isl_map_floordiv_val(__isl_take isl_map *map,
	__isl_take isl_val *d);

__isl_export
isl_bool isl_basic_map_is_equal(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2);
isl_bool isl_basic_map_is_disjoint(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2);

__isl_give isl_map *isl_basic_map_partial_lexmax(
		__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
		__isl_give isl_set **empty);
__isl_give isl_map *isl_basic_map_partial_lexmin(
		__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
		__isl_give isl_set **empty);
__isl_give isl_map *isl_map_partial_lexmax(
		__isl_take isl_map *map, __isl_take isl_set *dom,
		__isl_give isl_set **empty);
__isl_give isl_map *isl_map_partial_lexmin(
		__isl_take isl_map *map, __isl_take isl_set *dom,
		__isl_give isl_set **empty);
__isl_export
__isl_give isl_map *isl_basic_map_lexmin(__isl_take isl_basic_map *bmap);
__isl_export
__isl_give isl_map *isl_basic_map_lexmax(__isl_take isl_basic_map *bmap);
__isl_export
__isl_give isl_map *isl_map_lexmin(__isl_take isl_map *map);
__isl_export
__isl_give isl_map *isl_map_lexmax(__isl_take isl_map *map);
__isl_give isl_pw_multi_aff *isl_basic_map_partial_lexmin_pw_multi_aff(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty);
__isl_give isl_pw_multi_aff *isl_basic_map_partial_lexmax_pw_multi_aff(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty);
__isl_give isl_pw_multi_aff *isl_basic_map_lexmin_pw_multi_aff(
	__isl_take isl_basic_map *bmap);
__isl_give isl_pw_multi_aff *isl_map_lexmin_pw_multi_aff(
	__isl_take isl_map *map);
__isl_give isl_pw_multi_aff *isl_map_lexmax_pw_multi_aff(
	__isl_take isl_map *map);

void isl_basic_map_print_internal(__isl_keep isl_basic_map *bmap,
	FILE *out, int indent);

ISL_DEPRECATED
struct isl_basic_map *isl_map_copy_basic_map(struct isl_map *map);
ISL_DEPRECATED
__isl_give isl_map *isl_map_drop_basic_map(__isl_take isl_map *map,
						__isl_keep isl_basic_map *bmap);

__isl_give isl_val *isl_basic_map_plain_get_val_if_fixed(
	__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos);

int isl_basic_map_image_is_bounded(__isl_keep isl_basic_map *bmap);
isl_bool isl_basic_map_plain_is_universe(__isl_keep isl_basic_map *bmap);
isl_bool isl_basic_map_is_universe(__isl_keep isl_basic_map *bmap);
isl_bool isl_basic_map_plain_is_empty(__isl_keep isl_basic_map *bmap);
__isl_export
isl_bool isl_basic_map_is_empty(__isl_keep isl_basic_map *bmap);
__isl_export
isl_bool isl_basic_map_is_subset(__isl_keep isl_basic_map *bmap1,
		__isl_keep isl_basic_map *bmap2);
isl_bool isl_basic_map_is_strict_subset(__isl_keep isl_basic_map *bmap1,
		__isl_keep isl_basic_map *bmap2);

__isl_give isl_map *isl_map_universe(__isl_take isl_space *dim);
__isl_give isl_map *isl_map_nat_universe(__isl_take isl_space *dim);
__isl_give isl_map *isl_map_empty(__isl_take isl_space *dim);
__isl_give isl_map *isl_map_identity(__isl_take isl_space *dim);
__isl_give isl_map *isl_map_lex_lt_first(__isl_take isl_space *dim, unsigned n);
__isl_give isl_map *isl_map_lex_le_first(__isl_take isl_space *dim, unsigned n);
__isl_give isl_map *isl_map_lex_lt(__isl_take isl_space *set_dim);
__isl_give isl_map *isl_map_lex_le(__isl_take isl_space *set_dim);
__isl_give isl_map *isl_map_lex_gt_first(__isl_take isl_space *dim, unsigned n);
__isl_give isl_map *isl_map_lex_ge_first(__isl_take isl_space *dim, unsigned n);
__isl_give isl_map *isl_map_lex_gt(__isl_take isl_space *set_dim);
__isl_give isl_map *isl_map_lex_ge(__isl_take isl_space *set_dim);
__isl_null isl_map *isl_map_free(__isl_take isl_map *map);
__isl_give isl_map *isl_map_copy(__isl_keep isl_map *map);
__isl_export
__isl_give isl_map *isl_map_reverse(__isl_take isl_map *map);
__isl_export
__isl_give isl_map *isl_map_union(
		__isl_take isl_map *map1,
		__isl_take isl_map *map2);
struct isl_map *isl_map_union_disjoint(
			struct isl_map *map1, struct isl_map *map2);
__isl_export
__isl_give isl_map *isl_map_intersect_domain(
		__isl_take isl_map *map,
		__isl_take isl_set *set);
__isl_export
__isl_give isl_map *isl_map_intersect_range(
		__isl_take isl_map *map,
		__isl_take isl_set *set);
__isl_export
__isl_give isl_map *isl_map_apply_domain(
		__isl_take isl_map *map1,
		__isl_take isl_map *map2);
__isl_export
__isl_give isl_map *isl_map_apply_range(
		__isl_take isl_map *map1,
		__isl_take isl_map *map2);
__isl_give isl_map *isl_map_preimage_domain_multi_aff(__isl_take isl_map *map,
	__isl_take isl_multi_aff *ma);
__isl_give isl_map *isl_map_preimage_range_multi_aff(__isl_take isl_map *map,
	__isl_take isl_multi_aff *ma);
__isl_give isl_map *isl_map_preimage_domain_pw_multi_aff(
	__isl_take isl_map *map, __isl_take isl_pw_multi_aff *pma);
__isl_give isl_map *isl_map_preimage_range_pw_multi_aff(
	__isl_take isl_map *map, __isl_take isl_pw_multi_aff *pma);
__isl_give isl_map *isl_map_preimage_domain_multi_pw_aff(
	__isl_take isl_map *map, __isl_take isl_multi_pw_aff *mpa);
__isl_give isl_basic_map *isl_basic_map_product(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2);
__isl_give isl_map *isl_map_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2);
__isl_give isl_basic_map *isl_basic_map_domain_product(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2);
__isl_give isl_basic_map *isl_basic_map_range_product(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2);
__isl_give isl_map *isl_map_domain_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2);
__isl_give isl_map *isl_map_range_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2);
__isl_give isl_basic_map *isl_basic_map_flat_product(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2);
__isl_give isl_map *isl_map_flat_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2);
__isl_give isl_basic_map *isl_basic_map_flat_range_product(
	__isl_take isl_basic_map *bmap1, __isl_take isl_basic_map *bmap2);
__isl_give isl_map *isl_map_flat_domain_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2);
__isl_give isl_map *isl_map_flat_range_product(__isl_take isl_map *map1,
	__isl_take isl_map *map2);
isl_bool isl_map_domain_is_wrapping(__isl_keep isl_map *map);
isl_bool isl_map_range_is_wrapping(__isl_keep isl_map *map);
__isl_give isl_map *isl_map_factor_domain(__isl_take isl_map *map);
__isl_give isl_map *isl_map_factor_range(__isl_take isl_map *map);
__isl_give isl_map *isl_map_domain_factor_domain(__isl_take isl_map *map);
__isl_give isl_map *isl_map_domain_factor_range(__isl_take isl_map *map);
__isl_give isl_map *isl_map_range_factor_domain(__isl_take isl_map *map);
__isl_give isl_map *isl_map_range_factor_range(__isl_take isl_map *map);
__isl_export
__isl_give isl_map *isl_map_intersect(__isl_take isl_map *map1,
				      __isl_take isl_map *map2);
__isl_export
__isl_give isl_map *isl_map_intersect_params(__isl_take isl_map *map,
		__isl_take isl_set *params);
__isl_export
__isl_give isl_map *isl_map_subtract(
		__isl_take isl_map *map1,
		__isl_take isl_map *map2);
__isl_give isl_map *isl_map_subtract_domain(__isl_take isl_map *map,
	__isl_take isl_set *dom);
__isl_give isl_map *isl_map_subtract_range(__isl_take isl_map *map,
	__isl_take isl_set *dom);
__isl_export
__isl_give isl_map *isl_map_complement(__isl_take isl_map *map);
struct isl_map *isl_map_fix_input_si(struct isl_map *map,
		unsigned input, int value);
__isl_give isl_map *isl_map_fix_si(__isl_take isl_map *map,
		enum isl_dim_type type, unsigned pos, int value);
__isl_give isl_map *isl_map_fix_val(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *v);
__isl_give isl_map *isl_map_lower_bound_si(__isl_take isl_map *map,
		enum isl_dim_type type, unsigned pos, int value);
__isl_give isl_map *isl_map_upper_bound_si(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, int value);
__isl_export
__isl_give isl_basic_set *isl_basic_map_deltas(__isl_take isl_basic_map *bmap);
__isl_export
__isl_give isl_set *isl_map_deltas(__isl_take isl_map *map);
__isl_give isl_basic_map *isl_basic_map_deltas_map(
	__isl_take isl_basic_map *bmap);
__isl_give isl_map *isl_map_deltas_map(__isl_take isl_map *map);
__isl_export
__isl_give isl_map *isl_map_detect_equalities(__isl_take isl_map *map);
__isl_export
__isl_give isl_basic_map *isl_map_affine_hull(__isl_take isl_map *map);
__isl_give isl_basic_map *isl_map_convex_hull(__isl_take isl_map *map);
__isl_export
__isl_give isl_basic_map *isl_map_polyhedral_hull(__isl_take isl_map *map);
__isl_give isl_basic_map *isl_basic_map_add_dims(__isl_take isl_basic_map *bmap,
		enum isl_dim_type type, unsigned n);
__isl_give isl_map *isl_map_add_dims(__isl_take isl_map *map,
		enum isl_dim_type type, unsigned n);
__isl_give isl_basic_map *isl_basic_map_insert_dims(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type,
	unsigned pos, unsigned n);
__isl_give isl_map *isl_map_insert_dims(__isl_take isl_map *map,
		enum isl_dim_type type, unsigned pos, unsigned n);
__isl_give isl_basic_map *isl_basic_map_move_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n);
__isl_give isl_map *isl_map_move_dims(__isl_take isl_map *map,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n);
__isl_give isl_basic_map *isl_basic_map_project_out(
		__isl_take isl_basic_map *bmap,
		enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_map *isl_map_project_out(__isl_take isl_map *map,
		enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_basic_map *isl_basic_map_remove_divs(
	__isl_take isl_basic_map *bmap);
__isl_give isl_map *isl_map_remove_unknown_divs(__isl_take isl_map *map);
__isl_give isl_map *isl_map_remove_divs(__isl_take isl_map *map);
__isl_give isl_map *isl_map_eliminate(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_map *isl_map_remove_dims(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_basic_map *isl_basic_map_remove_divs_involving_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_map *isl_map_remove_divs_involving_dims(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n);
struct isl_map *isl_map_remove_inputs(struct isl_map *map,
	unsigned first, unsigned n);

__isl_give isl_basic_map *isl_basic_map_equate(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2);
__isl_give isl_basic_map *isl_basic_map_order_ge(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2);
__isl_give isl_map *isl_map_order_ge(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2);
__isl_give isl_map *isl_map_order_le(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2);
__isl_give isl_map *isl_map_equate(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2);
__isl_give isl_map *isl_map_oppose(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2);
__isl_give isl_map *isl_map_order_lt(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2);
__isl_give isl_basic_map *isl_basic_map_order_gt(__isl_take isl_basic_map *bmap,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2);
__isl_give isl_map *isl_map_order_gt(__isl_take isl_map *map,
	enum isl_dim_type type1, int pos1, enum isl_dim_type type2, int pos2);

__isl_export
__isl_give isl_map *isl_set_identity(__isl_take isl_set *set);

__isl_export
isl_bool isl_basic_set_is_wrapping(__isl_keep isl_basic_set *bset);
__isl_export
isl_bool isl_set_is_wrapping(__isl_keep isl_set *set);
__isl_give isl_basic_set *isl_basic_map_wrap(__isl_take isl_basic_map *bmap);
__isl_give isl_set *isl_map_wrap(__isl_take isl_map *map);
__isl_give isl_basic_map *isl_basic_set_unwrap(__isl_take isl_basic_set *bset);
__isl_give isl_map *isl_set_unwrap(__isl_take isl_set *set);
__isl_export
__isl_give isl_basic_map *isl_basic_map_flatten(__isl_take isl_basic_map *bmap);
__isl_export
__isl_give isl_map *isl_map_flatten(__isl_take isl_map *map);
__isl_export
__isl_give isl_basic_map *isl_basic_map_flatten_domain(
	__isl_take isl_basic_map *bmap);
__isl_export
__isl_give isl_basic_map *isl_basic_map_flatten_range(
	__isl_take isl_basic_map *bmap);
__isl_export
__isl_give isl_map *isl_map_flatten_domain(__isl_take isl_map *map);
__isl_export
__isl_give isl_map *isl_map_flatten_range(__isl_take isl_map *map);
__isl_export
__isl_give isl_basic_set *isl_basic_set_flatten(__isl_take isl_basic_set *bset);
__isl_export
__isl_give isl_set *isl_set_flatten(__isl_take isl_set *set);
__isl_give isl_map *isl_set_flatten_map(__isl_take isl_set *set);
__isl_give isl_set *isl_map_params(__isl_take isl_map *map);
__isl_give isl_set *isl_map_domain(__isl_take isl_map *bmap);
__isl_give isl_set *isl_map_range(__isl_take isl_map *map);
__isl_give isl_map *isl_map_domain_map(__isl_take isl_map *map);
__isl_give isl_map *isl_map_range_map(__isl_take isl_map *map);
__isl_give isl_map *isl_set_wrapped_domain_map(__isl_take isl_set *set);
__isl_constructor
__isl_give isl_map *isl_map_from_basic_map(__isl_take isl_basic_map *bmap);
__isl_give isl_map *isl_map_from_domain(__isl_take isl_set *set);
__isl_give isl_basic_map *isl_basic_map_from_domain(
	__isl_take isl_basic_set *bset);
__isl_give isl_basic_map *isl_basic_map_from_range(
	__isl_take isl_basic_set *bset);
__isl_give isl_map *isl_map_from_range(__isl_take isl_set *set);
__isl_give isl_basic_map *isl_basic_map_from_domain_and_range(
	__isl_take isl_basic_set *domain, __isl_take isl_basic_set *range);
__isl_give isl_map *isl_map_from_domain_and_range(__isl_take isl_set *domain,
	__isl_take isl_set *range);
ISL_DEPRECATED
__isl_give isl_map *isl_map_from_set(__isl_take isl_set *set,
	__isl_take isl_space *dim);
__isl_export
__isl_give isl_basic_map *isl_map_sample(__isl_take isl_map *map);

isl_bool isl_map_plain_is_empty(__isl_keep isl_map *map);
isl_bool isl_map_plain_is_universe(__isl_keep isl_map *map);
__isl_export
isl_bool isl_map_is_empty(__isl_keep isl_map *map);
__isl_export
isl_bool isl_map_is_subset(__isl_keep isl_map *map1, __isl_keep isl_map *map2);
__isl_export
isl_bool isl_map_is_strict_subset(__isl_keep isl_map *map1,
	__isl_keep isl_map *map2);
__isl_export
isl_bool isl_map_is_equal(__isl_keep isl_map *map1, __isl_keep isl_map *map2);
__isl_export
isl_bool isl_map_is_disjoint(__isl_keep isl_map *map1,
	__isl_keep isl_map *map2);
isl_bool isl_basic_map_is_single_valued(__isl_keep isl_basic_map *bmap);
isl_bool isl_map_plain_is_single_valued(__isl_keep isl_map *map);
__isl_export
isl_bool isl_map_is_single_valued(__isl_keep isl_map *map);
isl_bool isl_map_plain_is_injective(__isl_keep isl_map *map);
__isl_export
isl_bool isl_map_is_injective(__isl_keep isl_map *map);
__isl_export
isl_bool isl_map_is_bijective(__isl_keep isl_map *map);
isl_bool isl_map_is_identity(__isl_keep isl_map *map);
int isl_map_is_translation(__isl_keep isl_map *map);
int isl_map_has_equal_space(__isl_keep isl_map *map1, __isl_keep isl_map *map2);

isl_bool isl_basic_map_can_zip(__isl_keep isl_basic_map *bmap);
isl_bool isl_map_can_zip(__isl_keep isl_map *map);
__isl_give isl_basic_map *isl_basic_map_zip(__isl_take isl_basic_map *bmap);
__isl_give isl_map *isl_map_zip(__isl_take isl_map *map);

isl_bool isl_basic_map_can_curry(__isl_keep isl_basic_map *bmap);
isl_bool isl_map_can_curry(__isl_keep isl_map *map);
__isl_give isl_basic_map *isl_basic_map_curry(__isl_take isl_basic_map *bmap);
__isl_give isl_map *isl_map_curry(__isl_take isl_map *map);

isl_bool isl_map_can_range_curry(__isl_keep isl_map *map);
__isl_give isl_map *isl_map_range_curry(__isl_take isl_map *map);

isl_bool isl_basic_map_can_uncurry(__isl_keep isl_basic_map *bmap);
isl_bool isl_map_can_uncurry(__isl_keep isl_map *map);
__isl_give isl_basic_map *isl_basic_map_uncurry(__isl_take isl_basic_map *bmap);
__isl_give isl_map *isl_map_uncurry(__isl_take isl_map *map);

__isl_give isl_map *isl_map_make_disjoint(__isl_take isl_map *map);
__isl_give isl_map *isl_basic_map_compute_divs(__isl_take isl_basic_map *bmap);
__isl_give isl_map *isl_map_compute_divs(__isl_take isl_map *map);
__isl_give isl_map *isl_map_align_divs(__isl_take isl_map *map);

__isl_give isl_basic_map *isl_basic_map_drop_constraints_involving_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_basic_map *isl_basic_map_drop_constraints_not_involving_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_map *isl_map_drop_constraints_involving_dims(
	__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_map *isl_map_drop_constraints_not_involving_dims(
	__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n);

isl_bool isl_basic_map_involves_dims(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n);
isl_bool isl_map_involves_dims(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n);

void isl_map_print_internal(__isl_keep isl_map *map, FILE *out, int indent);

__isl_give isl_val *isl_map_plain_get_val_if_fixed(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos);

__isl_give isl_basic_map *isl_basic_map_gist_domain(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *context);
__isl_export
__isl_give isl_basic_map *isl_basic_map_gist(__isl_take isl_basic_map *bmap,
	__isl_take isl_basic_map *context);
__isl_export
__isl_give isl_map *isl_map_gist(__isl_take isl_map *map,
	__isl_take isl_map *context);
__isl_export
__isl_give isl_map *isl_map_gist_domain(__isl_take isl_map *map,
	__isl_take isl_set *context);
__isl_give isl_map *isl_map_gist_range(__isl_take isl_map *map,
	__isl_take isl_set *context);
__isl_give isl_map *isl_map_gist_params(__isl_take isl_map *map,
	__isl_take isl_set *context);
__isl_give isl_map *isl_map_gist_basic_map(__isl_take isl_map *map,
	__isl_take isl_basic_map *context);

__isl_export
__isl_give isl_map *isl_map_coalesce(__isl_take isl_map *map);

isl_bool isl_map_plain_is_equal(__isl_keep isl_map *map1,
	__isl_keep isl_map *map2);

uint32_t isl_map_get_hash(__isl_keep isl_map *map);

int isl_map_n_basic_map(__isl_keep isl_map *map);
__isl_export
isl_stat isl_map_foreach_basic_map(__isl_keep isl_map *map,
	isl_stat (*fn)(__isl_take isl_basic_map *bmap, void *user), void *user);

ISL_DEPRECATED
__isl_give isl_map *isl_set_lifting(__isl_take isl_set *set);

__isl_give isl_map *isl_map_fixed_power_val(__isl_take isl_map *map,
	__isl_take isl_val *exp);
__isl_give isl_map *isl_map_power(__isl_take isl_map *map, int *exact);
__isl_give isl_map *isl_map_reaching_path_lengths(__isl_take isl_map *map,
	int *exact);
__isl_give isl_map *isl_map_transitive_closure(__isl_take isl_map *map,
	int *exact);

__isl_give isl_map *isl_map_lex_le_map(__isl_take isl_map *map1,
	__isl_take isl_map *map2);
__isl_give isl_map *isl_map_lex_lt_map(__isl_take isl_map *map1,
	__isl_take isl_map *map2);
__isl_give isl_map *isl_map_lex_ge_map(__isl_take isl_map *map1,
	__isl_take isl_map *map2);
__isl_give isl_map *isl_map_lex_gt_map(__isl_take isl_map *map1,
	__isl_take isl_map *map2);

__isl_give isl_basic_map *isl_basic_map_align_params(
	__isl_take isl_basic_map *bmap, __isl_take isl_space *model);
__isl_give isl_map *isl_map_align_params(__isl_take isl_map *map,
	__isl_take isl_space *model);

__isl_give isl_mat *isl_basic_map_equalities_matrix(
		__isl_keep isl_basic_map *bmap, enum isl_dim_type c1,
		enum isl_dim_type c2, enum isl_dim_type c3,
		enum isl_dim_type c4, enum isl_dim_type c5);
__isl_give isl_mat *isl_basic_map_inequalities_matrix(
		__isl_keep isl_basic_map *bmap, enum isl_dim_type c1,
		enum isl_dim_type c2, enum isl_dim_type c3,
		enum isl_dim_type c4, enum isl_dim_type c5);
__isl_give isl_basic_map *isl_basic_map_from_constraint_matrices(
	__isl_take isl_space *dim,
	__isl_take isl_mat *eq, __isl_take isl_mat *ineq, enum isl_dim_type c1,
	enum isl_dim_type c2, enum isl_dim_type c3,
	enum isl_dim_type c4, enum isl_dim_type c5);

__isl_give isl_basic_map *isl_basic_map_from_aff(__isl_take isl_aff *aff);
__isl_give isl_basic_map *isl_basic_map_from_multi_aff(
	__isl_take isl_multi_aff *maff);
__isl_give isl_basic_map *isl_basic_map_from_aff_list(
	__isl_take isl_space *domain_dim, __isl_take isl_aff_list *list);

__isl_give isl_map *isl_map_from_aff(__isl_take isl_aff *aff);
__isl_give isl_map *isl_map_from_multi_aff(__isl_take isl_multi_aff *maff);

__isl_give isl_pw_aff *isl_map_dim_min(__isl_take isl_map *map, int pos);
__isl_give isl_pw_aff *isl_map_dim_max(__isl_take isl_map *map, int pos);

ISL_DECLARE_LIST_FN(basic_map)
ISL_DECLARE_LIST_FN(map)

#if defined(__cplusplus)
}
#endif

#endif
