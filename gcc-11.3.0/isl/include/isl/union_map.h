#ifndef ISL_UNION_MAP_H
#define ISL_UNION_MAP_H

#include <isl/space.h>
#include <isl/aff_type.h>
#include <isl/map_type.h>
#include <isl/union_map_type.h>
#include <isl/printer.h>
#include <isl/val.h>

#if defined(__cplusplus)
extern "C" {
#endif

unsigned isl_union_map_dim(__isl_keep isl_union_map *umap,
	enum isl_dim_type type);
isl_bool isl_union_map_involves_dims(__isl_keep isl_union_map *umap,
	enum isl_dim_type type, unsigned first, unsigned n);
__isl_give isl_id *isl_union_map_get_dim_id(__isl_keep isl_union_map *umap,
	enum isl_dim_type type, unsigned pos);

__isl_constructor
__isl_give isl_union_map *isl_union_map_from_basic_map(
	__isl_take isl_basic_map *bmap);
__isl_constructor
__isl_give isl_union_map *isl_union_map_from_map(__isl_take isl_map *map);
__isl_give isl_union_map *isl_union_map_empty(__isl_take isl_space *dim);
__isl_give isl_union_map *isl_union_map_copy(__isl_keep isl_union_map *umap);
__isl_null isl_union_map *isl_union_map_free(__isl_take isl_union_map *umap);

isl_ctx *isl_union_map_get_ctx(__isl_keep isl_union_map *umap);
__isl_give isl_space *isl_union_map_get_space(__isl_keep isl_union_map *umap);

__isl_give isl_union_map *isl_union_map_reset_user(
	__isl_take isl_union_map *umap);

int isl_union_map_find_dim_by_name(__isl_keep isl_union_map *umap,
	enum isl_dim_type type, const char *name);

__isl_give isl_union_map *isl_union_map_universe(
	__isl_take isl_union_map *umap);
__isl_give isl_set *isl_union_map_params(__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_set *isl_union_map_domain(__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_set *isl_union_map_range(__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_domain_map(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_pw_multi_aff *isl_union_map_domain_map_union_pw_multi_aff(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_range_map(
	__isl_take isl_union_map *umap);
__isl_give isl_union_map *isl_union_set_wrapped_domain_map(
	__isl_take isl_union_set *uset);
__isl_give isl_union_map *isl_union_map_from_domain(
	__isl_take isl_union_set *uset);
__isl_give isl_union_map *isl_union_map_from_range(
	__isl_take isl_union_set *uset);

__isl_export
__isl_give isl_union_map *isl_union_map_affine_hull(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_polyhedral_hull(
	__isl_take isl_union_map *umap);
__isl_give isl_union_map *isl_union_map_remove_redundancies(
	__isl_take isl_union_map *umap);
__isl_give isl_union_map *isl_union_map_simple_hull(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_coalesce(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_compute_divs(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_lexmin(__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_lexmax(__isl_take isl_union_map *umap);

__isl_give isl_union_map *isl_union_map_add_map(__isl_take isl_union_map *umap,
	__isl_take isl_map *map);
__isl_export
__isl_give isl_union_map *isl_union_map_union(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2);
__isl_export
__isl_give isl_union_map *isl_union_map_subtract(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_export
__isl_give isl_union_map *isl_union_map_intersect(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_export
__isl_give isl_union_map *isl_union_map_intersect_params(
	__isl_take isl_union_map *umap, __isl_take isl_set *set);
__isl_export
__isl_give isl_union_map *isl_union_map_product(__isl_take isl_union_map *umap1,
	__isl_take isl_union_map *umap2);
__isl_export
__isl_give isl_union_map *isl_union_map_domain_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_give isl_union_map *isl_union_map_flat_domain_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_export
__isl_give isl_union_map *isl_union_map_range_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_give isl_union_map *isl_union_map_flat_range_product(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_export
__isl_give isl_union_map *isl_union_map_domain_factor_domain(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_domain_factor_range(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_range_factor_domain(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_range_factor_range(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_factor_domain(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_factor_range(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_gist(__isl_take isl_union_map *umap,
	__isl_take isl_union_map *context);
__isl_export
__isl_give isl_union_map *isl_union_map_gist_params(
	__isl_take isl_union_map *umap, __isl_take isl_set *set);
__isl_export
__isl_give isl_union_map *isl_union_map_gist_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset);
__isl_export
__isl_give isl_union_map *isl_union_map_gist_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset);

__isl_export
__isl_give isl_union_map *isl_union_map_intersect_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset);
__isl_export
__isl_give isl_union_map *isl_union_map_intersect_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *uset);

__isl_export
__isl_give isl_union_map *isl_union_map_subtract_domain(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *dom);
__isl_export
__isl_give isl_union_map *isl_union_map_subtract_range(
	__isl_take isl_union_map *umap, __isl_take isl_union_set *dom);

__isl_export
__isl_give isl_union_map *isl_union_map_apply_domain(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_export
__isl_give isl_union_map *isl_union_map_apply_range(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_give isl_union_map *isl_union_map_preimage_domain_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_aff *ma);
__isl_give isl_union_map *isl_union_map_preimage_range_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_aff *ma);
__isl_give isl_union_map *isl_union_map_preimage_domain_pw_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_pw_multi_aff *pma);
__isl_give isl_union_map *isl_union_map_preimage_range_pw_multi_aff(
	__isl_take isl_union_map *umap, __isl_take isl_pw_multi_aff *pma);
__isl_give isl_union_map *isl_union_map_preimage_domain_multi_pw_aff(
	__isl_take isl_union_map *umap, __isl_take isl_multi_pw_aff *mpa);
__isl_give isl_union_map *isl_union_map_preimage_domain_union_pw_multi_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_union_pw_multi_aff *upma);
__isl_give isl_union_map *isl_union_map_preimage_range_union_pw_multi_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_union_pw_multi_aff *upma);
__isl_export
__isl_give isl_union_map *isl_union_map_reverse(__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_map_from_domain_and_range(
	__isl_take isl_union_set *domain, __isl_take isl_union_set *range);

__isl_export
__isl_give isl_union_map *isl_union_map_detect_equalities(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_set *isl_union_map_deltas(__isl_take isl_union_map *umap);
__isl_give isl_union_map *isl_union_map_deltas_map(
	__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_set_identity(__isl_take isl_union_set *uset);

__isl_give isl_union_map *isl_union_map_project_out(
	__isl_take isl_union_map *umap,
	enum isl_dim_type type, unsigned first, unsigned n);

__isl_export
isl_bool isl_union_map_is_empty(__isl_keep isl_union_map *umap);
__isl_export
isl_bool isl_union_map_is_single_valued(__isl_keep isl_union_map *umap);
isl_bool isl_union_map_plain_is_injective(__isl_keep isl_union_map *umap);
__isl_export
isl_bool isl_union_map_is_injective(__isl_keep isl_union_map *umap);
__isl_export
isl_bool isl_union_map_is_bijective(__isl_keep isl_union_map *umap);
isl_bool isl_union_map_is_identity(__isl_keep isl_union_map *umap);

__isl_export
isl_bool isl_union_map_is_subset(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2);
__isl_export
isl_bool isl_union_map_is_equal(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2);
isl_bool isl_union_map_is_disjoint(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2);
__isl_export
isl_bool isl_union_map_is_strict_subset(__isl_keep isl_union_map *umap1,
	__isl_keep isl_union_map *umap2);

uint32_t isl_union_map_get_hash(__isl_keep isl_union_map *umap);

int isl_union_map_n_map(__isl_keep isl_union_map *umap);
__isl_export
isl_stat isl_union_map_foreach_map(__isl_keep isl_union_map *umap,
	isl_stat (*fn)(__isl_take isl_map *map, void *user), void *user);
__isl_give int isl_union_map_contains(__isl_keep isl_union_map *umap,
	__isl_keep isl_space *dim);
__isl_give isl_map *isl_union_map_extract_map(__isl_keep isl_union_map *umap,
	__isl_take isl_space *dim);
__isl_give isl_map *isl_map_from_union_map(__isl_take isl_union_map *umap);

__isl_give isl_basic_map *isl_union_map_sample(__isl_take isl_union_map *umap);

__isl_overload
__isl_give isl_union_map *isl_union_map_fixed_power_val(
	__isl_take isl_union_map *umap, __isl_take isl_val *exp);
__isl_give isl_union_map *isl_union_map_power(__isl_take isl_union_map *umap,
	int *exact);
__isl_give isl_union_map *isl_union_map_transitive_closure(
	__isl_take isl_union_map *umap, int *exact);

__isl_give isl_union_map *isl_union_map_lex_lt_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_give isl_union_map *isl_union_map_lex_le_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_give isl_union_map *isl_union_map_lex_gt_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);
__isl_give isl_union_map *isl_union_map_lex_ge_union_map(
	__isl_take isl_union_map *umap1, __isl_take isl_union_map *umap2);

__isl_give isl_union_map *isl_union_map_eq_at_multi_union_pw_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_multi_union_pw_aff *mupa);
__isl_give isl_union_map *isl_union_map_lex_lt_at_multi_union_pw_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_multi_union_pw_aff *mupa);
__isl_give isl_union_map *isl_union_map_lex_gt_at_multi_union_pw_aff(
	__isl_take isl_union_map *umap,
	__isl_take isl_multi_union_pw_aff *mupa);

__isl_give isl_union_map *isl_union_map_read_from_file(isl_ctx *ctx,
	FILE *input);
__isl_constructor
__isl_give isl_union_map *isl_union_map_read_from_str(isl_ctx *ctx,
	const char *str);
__isl_give char *isl_union_map_to_str(__isl_keep isl_union_map *umap);
__isl_give isl_printer *isl_printer_print_union_map(__isl_take isl_printer *p,
	__isl_keep isl_union_map *umap);
void isl_union_map_dump(__isl_keep isl_union_map *umap);

__isl_export
__isl_give isl_union_set *isl_union_map_wrap(__isl_take isl_union_map *umap);
__isl_export
__isl_give isl_union_map *isl_union_set_unwrap(__isl_take isl_union_set *uset);

__isl_export
__isl_give isl_union_map *isl_union_map_zip(__isl_take isl_union_map *umap);
__isl_give isl_union_map *isl_union_map_curry(__isl_take isl_union_map *umap);
__isl_give isl_union_map *isl_union_map_range_curry(
	__isl_take isl_union_map *umap);
__isl_give isl_union_map *isl_union_map_uncurry(__isl_take isl_union_map *umap);

__isl_give isl_union_map *isl_union_map_align_params(
	__isl_take isl_union_map *umap, __isl_take isl_space *model);
__isl_give isl_union_set *isl_union_set_align_params(
	__isl_take isl_union_set *uset, __isl_take isl_space *model);

ISL_DECLARE_LIST_FN(union_map)

#if defined(__cplusplus)
}
#endif

#endif
