#ifndef ISL_SCHEDULE_H
#define ISL_SCHEDULE_H

#include <isl/union_set_type.h>
#include <isl/union_map_type.h>
#include <isl/schedule_type.h>
#include <isl/aff_type.h>
#include <isl/band.h>
#include <isl/space.h>
#include <isl/set_type.h>
#include <isl/list.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct __isl_export isl_schedule_constraints;
typedef struct isl_schedule_constraints isl_schedule_constraints;

isl_stat isl_options_set_schedule_max_coefficient(isl_ctx *ctx, int val);
int isl_options_get_schedule_max_coefficient(isl_ctx *ctx);

isl_stat isl_options_set_schedule_max_constant_term(isl_ctx *ctx, int val);
int isl_options_get_schedule_max_constant_term(isl_ctx *ctx);

isl_stat isl_options_set_schedule_maximize_band_depth(isl_ctx *ctx, int val);
int isl_options_get_schedule_maximize_band_depth(isl_ctx *ctx);

isl_stat isl_options_set_schedule_maximize_coincidence(isl_ctx *ctx, int val);
int isl_options_get_schedule_maximize_coincidence(isl_ctx *ctx);

isl_stat isl_options_set_schedule_outer_coincidence(isl_ctx *ctx, int val);
int isl_options_get_schedule_outer_coincidence(isl_ctx *ctx);

isl_stat isl_options_set_schedule_split_scaled(isl_ctx *ctx, int val);
int isl_options_get_schedule_split_scaled(isl_ctx *ctx);

isl_stat isl_options_set_schedule_treat_coalescing(isl_ctx *ctx, int val);
int isl_options_get_schedule_treat_coalescing(isl_ctx *ctx);

isl_stat isl_options_set_schedule_separate_components(isl_ctx *ctx, int val);
int isl_options_get_schedule_separate_components(isl_ctx *ctx);

isl_stat isl_options_set_schedule_serialize_sccs(isl_ctx *ctx, int val);
int isl_options_get_schedule_serialize_sccs(isl_ctx *ctx);

isl_stat isl_options_set_schedule_whole_component(isl_ctx *ctx, int val);
int isl_options_get_schedule_whole_component(isl_ctx *ctx);

__isl_give isl_schedule_constraints *isl_schedule_constraints_copy(
	__isl_keep isl_schedule_constraints *sc);
__isl_give isl_schedule_constraints *isl_schedule_constraints_on_domain(
	__isl_take isl_union_set *domain);
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_context(
	__isl_take isl_schedule_constraints *sc, __isl_take isl_set *context);
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_validity(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *validity);
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_coincidence(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *coincidence);
__isl_give isl_schedule_constraints *isl_schedule_constraints_set_proximity(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *proximity);
__isl_give isl_schedule_constraints *
isl_schedule_constraints_set_conditional_validity(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *condition,
	__isl_take isl_union_map *validity);
__isl_null isl_schedule_constraints *isl_schedule_constraints_free(
	__isl_take isl_schedule_constraints *sc);

isl_ctx *isl_schedule_constraints_get_ctx(
	__isl_keep isl_schedule_constraints *sc);
__isl_export
__isl_give isl_union_set *isl_schedule_constraints_get_domain(
	__isl_keep isl_schedule_constraints *sc);
__isl_export
__isl_give isl_set *isl_schedule_constraints_get_context(
	__isl_keep isl_schedule_constraints *sc);
__isl_export
__isl_give isl_union_map *isl_schedule_constraints_get_validity(
	__isl_keep isl_schedule_constraints *sc);
__isl_export
__isl_give isl_union_map *isl_schedule_constraints_get_coincidence(
	__isl_keep isl_schedule_constraints *sc);
__isl_export
__isl_give isl_union_map *isl_schedule_constraints_get_proximity(
	__isl_keep isl_schedule_constraints *sc);
__isl_export
__isl_give isl_union_map *isl_schedule_constraints_get_conditional_validity(
	__isl_keep isl_schedule_constraints *sc);
__isl_export
__isl_give isl_union_map *
isl_schedule_constraints_get_conditional_validity_condition(
	__isl_keep isl_schedule_constraints *sc);

__isl_give isl_schedule_constraints *isl_schedule_constraints_apply(
	__isl_take isl_schedule_constraints *sc,
	__isl_take isl_union_map *umap);

__isl_constructor
__isl_give isl_schedule_constraints *isl_schedule_constraints_read_from_str(
	isl_ctx *ctx, const char *str);
__isl_give isl_schedule_constraints *isl_schedule_constraints_read_from_file(
	isl_ctx *ctx, FILE *input);
__isl_give isl_printer *isl_printer_print_schedule_constraints(
	__isl_take isl_printer *p, __isl_keep isl_schedule_constraints *sc);
void isl_schedule_constraints_dump(__isl_keep isl_schedule_constraints *sc);
__isl_give char *isl_schedule_constraints_to_str(
	__isl_keep isl_schedule_constraints *sc);

__isl_give isl_schedule *isl_schedule_constraints_compute_schedule(
	__isl_take isl_schedule_constraints *sc);

__isl_give isl_schedule *isl_union_set_compute_schedule(
	__isl_take isl_union_set *domain,
	__isl_take isl_union_map *validity,
	__isl_take isl_union_map *proximity);

__isl_give isl_schedule *isl_schedule_empty(__isl_take isl_space *space);
__isl_give isl_schedule *isl_schedule_from_domain(
	__isl_take isl_union_set *domain);
__isl_give isl_schedule *isl_schedule_copy(__isl_keep isl_schedule *sched);
__isl_null isl_schedule *isl_schedule_free(__isl_take isl_schedule *sched);
__isl_export
__isl_give isl_union_map *isl_schedule_get_map(__isl_keep isl_schedule *sched);

isl_ctx *isl_schedule_get_ctx(__isl_keep isl_schedule *sched);
isl_bool isl_schedule_plain_is_equal(__isl_keep isl_schedule *schedule1,
	__isl_keep isl_schedule *schedule2);

__isl_export
__isl_give isl_schedule_node *isl_schedule_get_root(
	__isl_keep isl_schedule *schedule);
__isl_give isl_union_set *isl_schedule_get_domain(
	__isl_keep isl_schedule *schedule);

isl_stat isl_schedule_foreach_schedule_node_top_down(
	__isl_keep isl_schedule *sched,
	isl_bool (*fn)(__isl_keep isl_schedule_node *node, void *user),
	void *user);
__isl_give isl_schedule *isl_schedule_map_schedule_node_bottom_up(
	__isl_take isl_schedule *schedule,
	__isl_give isl_schedule_node *(*fn)(
		__isl_take isl_schedule_node *node, void *user), void *user);

__isl_give isl_schedule *isl_schedule_insert_context(
	__isl_take isl_schedule *schedule, __isl_take isl_set *context);
__isl_give isl_schedule *isl_schedule_insert_partial_schedule(
	__isl_take isl_schedule *schedule,
	__isl_take isl_multi_union_pw_aff *partial);
__isl_give isl_schedule *isl_schedule_insert_guard(
	__isl_take isl_schedule *schedule, __isl_take isl_set *guard);
__isl_give isl_schedule *isl_schedule_sequence(
	__isl_take isl_schedule *schedule1, __isl_take isl_schedule *schedule2);
__isl_give isl_schedule *isl_schedule_set(
	__isl_take isl_schedule *schedule1, __isl_take isl_schedule *schedule2);
__isl_give isl_schedule *isl_schedule_intersect_domain(
	__isl_take isl_schedule *schedule, __isl_take isl_union_set *domain);
__isl_give isl_schedule *isl_schedule_gist_domain_params(
	__isl_take isl_schedule *schedule, __isl_take isl_set *context);

__isl_give isl_schedule *isl_schedule_reset_user(
	__isl_take isl_schedule *schedule);
__isl_give isl_schedule *isl_schedule_align_params(
	__isl_take isl_schedule *schedule, __isl_take isl_space *space);
__isl_overload
__isl_give isl_schedule *isl_schedule_pullback_union_pw_multi_aff(
	__isl_take isl_schedule *schedule,
	__isl_take isl_union_pw_multi_aff *upma);
__isl_give isl_schedule *isl_schedule_expand(__isl_take isl_schedule *schedule,
	__isl_take isl_union_pw_multi_aff *contraction,
	__isl_take isl_schedule *expansion);

__isl_give isl_band_list *isl_schedule_get_band_forest(
	__isl_keep isl_schedule *schedule);

__isl_give isl_schedule *isl_schedule_read_from_file(isl_ctx *ctx, FILE *input);
__isl_constructor
__isl_give isl_schedule *isl_schedule_read_from_str(isl_ctx *ctx,
	const char *str);
__isl_give isl_printer *isl_printer_print_schedule(__isl_take isl_printer *p,
	__isl_keep isl_schedule *schedule);
void isl_schedule_dump(__isl_keep isl_schedule *schedule);
__isl_give char *isl_schedule_to_str(__isl_keep isl_schedule *schedule);

int isl_schedule_foreach_band(__isl_keep isl_schedule *sched,
	int (*fn)(__isl_keep isl_band *band, void *user), void *user);

#if defined(__cplusplus)
}
#endif

#endif
