#ifndef ISL_UNION_SET_H
#define ISL_UNION_SET_H

#include <isl/point.h>
#include <isl/union_map.h>

#if defined(__cplusplus)
extern "C" {
#endif

unsigned isl_union_set_dim(__isl_keep isl_union_set *uset,
	enum isl_dim_type type);

__isl_constructor
__isl_give isl_union_set *isl_union_set_from_basic_set(
	__isl_take isl_basic_set *bset);
__isl_constructor
__isl_give isl_union_set *isl_union_set_from_set(__isl_take isl_set *set);
__isl_give isl_union_set *isl_union_set_empty(__isl_take isl_space *dim);
__isl_give isl_union_set *isl_union_set_copy(__isl_keep isl_union_set *uset);
__isl_null isl_union_set *isl_union_set_free(__isl_take isl_union_set *uset);

isl_ctx *isl_union_set_get_ctx(__isl_keep isl_union_set *uset);
__isl_give isl_space *isl_union_set_get_space(__isl_keep isl_union_set *uset);

__isl_give isl_union_set *isl_union_set_reset_user(
	__isl_take isl_union_set *uset);

__isl_give isl_union_set *isl_union_set_universe(
	__isl_take isl_union_set *uset);
__isl_give isl_set *isl_union_set_params(__isl_take isl_union_set *uset);

__isl_export
__isl_give isl_union_set *isl_union_set_detect_equalities(
	__isl_take isl_union_set *uset);
__isl_export
__isl_give isl_union_set *isl_union_set_affine_hull(
	__isl_take isl_union_set *uset);
__isl_export
__isl_give isl_union_set *isl_union_set_polyhedral_hull(
	__isl_take isl_union_set *uset);
__isl_give isl_union_set *isl_union_set_remove_redundancies(
	__isl_take isl_union_set *uset);
__isl_give isl_union_set *isl_union_set_simple_hull(
	__isl_take isl_union_set *uset);
__isl_export
__isl_give isl_union_set *isl_union_set_coalesce(
	__isl_take isl_union_set *uset);
__isl_export
__isl_give isl_union_set *isl_union_set_compute_divs(
	__isl_take isl_union_set *uset);
__isl_export
__isl_give isl_union_set *isl_union_set_lexmin(__isl_take isl_union_set *uset);
__isl_export
__isl_give isl_union_set *isl_union_set_lexmax(__isl_take isl_union_set *uset);

__isl_give isl_union_set *isl_union_set_add_set(__isl_take isl_union_set *uset,
	__isl_take isl_set *set);
__isl_export
__isl_give isl_union_set *isl_union_set_union(__isl_take isl_union_set *uset1,
	__isl_take isl_union_set *uset2);
__isl_export
__isl_give isl_union_set *isl_union_set_subtract(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2);
__isl_export
__isl_give isl_union_set *isl_union_set_intersect(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2);
__isl_export
__isl_give isl_union_set *isl_union_set_intersect_params(
	__isl_take isl_union_set *uset, __isl_take isl_set *set);
__isl_give isl_union_set *isl_union_set_product(__isl_take isl_union_set *uset1,
	__isl_take isl_union_set *uset2);
__isl_export
__isl_give isl_union_set *isl_union_set_gist(__isl_take isl_union_set *uset,
	__isl_take isl_union_set *context);
__isl_export
__isl_give isl_union_set *isl_union_set_gist_params(
	__isl_take isl_union_set *uset, __isl_take isl_set *set);

__isl_export
__isl_give isl_union_set *isl_union_set_apply(
	__isl_take isl_union_set *uset, __isl_take isl_union_map *umap);
__isl_give isl_union_set *isl_union_set_preimage_multi_aff(
	__isl_take isl_union_set *uset, __isl_take isl_multi_aff *ma);
__isl_give isl_union_set *isl_union_set_preimage_pw_multi_aff(
	__isl_take isl_union_set *uset, __isl_take isl_pw_multi_aff *pma);
__isl_give isl_union_set *isl_union_set_preimage_union_pw_multi_aff(
	__isl_take isl_union_set *uset,
	__isl_take isl_union_pw_multi_aff *upma);

__isl_give isl_union_set *isl_union_set_project_out(
	__isl_take isl_union_set *uset,
	enum isl_dim_type type, unsigned first, unsigned n);

isl_bool isl_union_set_is_params(__isl_keep isl_union_set *uset);
__isl_export
isl_bool isl_union_set_is_empty(__isl_keep isl_union_set *uset);

__isl_export
isl_bool isl_union_set_is_subset(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2);
__isl_export
isl_bool isl_union_set_is_equal(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2);
isl_bool isl_union_set_is_disjoint(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2);
__isl_export
isl_bool isl_union_set_is_strict_subset(__isl_keep isl_union_set *uset1,
	__isl_keep isl_union_set *uset2);

uint32_t isl_union_set_get_hash(__isl_keep isl_union_set *uset);

int isl_union_set_n_set(__isl_keep isl_union_set *uset);
__isl_export
isl_stat isl_union_set_foreach_set(__isl_keep isl_union_set *uset,
	isl_stat (*fn)(__isl_take isl_set *set, void *user), void *user);
__isl_give int isl_union_set_contains(__isl_keep isl_union_set *uset,
	__isl_keep isl_space *dim);
__isl_give isl_set *isl_union_set_extract_set(__isl_keep isl_union_set *uset,
	__isl_take isl_space *dim);
__isl_give isl_set *isl_set_from_union_set(__isl_take isl_union_set *uset);
__isl_export
isl_stat isl_union_set_foreach_point(__isl_keep isl_union_set *uset,
	isl_stat (*fn)(__isl_take isl_point *pnt, void *user), void *user);

__isl_give isl_basic_set *isl_union_set_sample(__isl_take isl_union_set *uset);
__isl_export
__isl_give isl_point *isl_union_set_sample_point(
	__isl_take isl_union_set *uset);

__isl_constructor
__isl_give isl_union_set *isl_union_set_from_point(__isl_take isl_point *pnt);

__isl_give isl_union_set *isl_union_set_lift(__isl_take isl_union_set *uset);

__isl_give isl_union_map *isl_union_set_lex_lt_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2);
__isl_give isl_union_map *isl_union_set_lex_le_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2);
__isl_give isl_union_map *isl_union_set_lex_gt_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2);
__isl_give isl_union_map *isl_union_set_lex_ge_union_set(
	__isl_take isl_union_set *uset1, __isl_take isl_union_set *uset2);

__isl_give isl_union_set *isl_union_set_coefficients(
	__isl_take isl_union_set *bset);
__isl_give isl_union_set *isl_union_set_solutions(
	__isl_take isl_union_set *bset);

__isl_give isl_union_set *isl_union_set_read_from_file(isl_ctx *ctx,
	FILE *input);
__isl_constructor
__isl_give isl_union_set *isl_union_set_read_from_str(isl_ctx *ctx,
	const char *str);
__isl_give char *isl_union_set_to_str(__isl_keep isl_union_set *uset);
__isl_give isl_printer *isl_printer_print_union_set(__isl_take isl_printer *p,
	__isl_keep isl_union_set *uset);
void isl_union_set_dump(__isl_keep isl_union_set *uset);

ISL_DECLARE_LIST_FN(union_set)

__isl_give isl_union_set *isl_union_set_list_union(
	__isl_take isl_union_set_list *list);

#if defined(__cplusplus)
}
#endif

#endif
