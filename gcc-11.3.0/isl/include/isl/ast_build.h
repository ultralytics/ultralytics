#ifndef ISL_AST_CONTEXT_H
#define ISL_AST_CONTEXT_H

#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/ast.h>
#include <isl/schedule.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct __isl_export isl_ast_build;
typedef struct isl_ast_build isl_ast_build;


isl_stat isl_options_set_ast_build_atomic_upper_bound(isl_ctx *ctx, int val);
int isl_options_get_ast_build_atomic_upper_bound(isl_ctx *ctx);

isl_stat isl_options_set_ast_build_prefer_pdiv(isl_ctx *ctx, int val);
int isl_options_get_ast_build_prefer_pdiv(isl_ctx *ctx);

isl_stat isl_options_set_ast_build_detect_min_max(isl_ctx *ctx, int val);
int isl_options_get_ast_build_detect_min_max(isl_ctx *ctx);

isl_stat isl_options_set_ast_build_exploit_nested_bounds(isl_ctx *ctx, int val);
int isl_options_get_ast_build_exploit_nested_bounds(isl_ctx *ctx);

isl_stat isl_options_set_ast_build_group_coscheduled(isl_ctx *ctx, int val);
int isl_options_get_ast_build_group_coscheduled(isl_ctx *ctx);

#define ISL_AST_BUILD_SEPARATION_BOUNDS_EXPLICIT		0
#define ISL_AST_BUILD_SEPARATION_BOUNDS_IMPLICIT		1
isl_stat isl_options_set_ast_build_separation_bounds(isl_ctx *ctx, int val);
int isl_options_get_ast_build_separation_bounds(isl_ctx *ctx);

isl_stat isl_options_set_ast_build_scale_strides(isl_ctx *ctx, int val);
int isl_options_get_ast_build_scale_strides(isl_ctx *ctx);

isl_stat isl_options_set_ast_build_allow_else(isl_ctx *ctx, int val);
int isl_options_get_ast_build_allow_else(isl_ctx *ctx);

isl_stat isl_options_set_ast_build_allow_or(isl_ctx *ctx, int val);
int isl_options_get_ast_build_allow_or(isl_ctx *ctx);

isl_ctx *isl_ast_build_get_ctx(__isl_keep isl_ast_build *build);

__isl_constructor
__isl_give isl_ast_build *isl_ast_build_alloc(isl_ctx *ctx);
__isl_export
__isl_give isl_ast_build *isl_ast_build_from_context(__isl_take isl_set *set);

__isl_give isl_space *isl_ast_build_get_schedule_space(
	__isl_keep isl_ast_build *build);
__isl_give isl_union_map *isl_ast_build_get_schedule(
	__isl_keep isl_ast_build *build);

__isl_give isl_ast_build *isl_ast_build_restrict(
	__isl_take isl_ast_build *build, __isl_take isl_set *set);

__isl_give isl_ast_build *isl_ast_build_copy(
	__isl_keep isl_ast_build *build);
__isl_null isl_ast_build *isl_ast_build_free(
	__isl_take isl_ast_build *build);

__isl_give isl_ast_build *isl_ast_build_set_options(
	__isl_take isl_ast_build *build,
	__isl_take isl_union_map *options);
__isl_give isl_ast_build *isl_ast_build_set_iterators(
	__isl_take isl_ast_build *build,
	__isl_take isl_id_list *iterators);
__isl_give isl_ast_build *isl_ast_build_set_at_each_domain(
	__isl_take isl_ast_build *build,
	__isl_give isl_ast_node *(*fn)(__isl_take isl_ast_node *node,
		__isl_keep isl_ast_build *build, void *user), void *user);
__isl_give isl_ast_build *isl_ast_build_set_before_each_for(
	__isl_take isl_ast_build *build,
	__isl_give isl_id *(*fn)(__isl_keep isl_ast_build *build,
		void *user), void *user);
__isl_give isl_ast_build *isl_ast_build_set_after_each_for(
	__isl_take isl_ast_build *build,
	__isl_give isl_ast_node *(*fn)(__isl_take isl_ast_node *node,
		__isl_keep isl_ast_build *build, void *user), void *user);
__isl_give isl_ast_build *isl_ast_build_set_before_each_mark(
	__isl_take isl_ast_build *build,
	isl_stat (*fn)(__isl_keep isl_id *mark, __isl_keep isl_ast_build *build,
		void *user), void *user);
__isl_give isl_ast_build *isl_ast_build_set_after_each_mark(
	__isl_take isl_ast_build *build,
	__isl_give isl_ast_node *(*fn)(__isl_take isl_ast_node *node,
		__isl_keep isl_ast_build *build, void *user), void *user);
__isl_give isl_ast_build *isl_ast_build_set_create_leaf(
	__isl_take isl_ast_build *build,
	__isl_give isl_ast_node *(*fn)(__isl_take isl_ast_build *build,
		void *user), void *user);

__isl_overload
__isl_give isl_ast_expr *isl_ast_build_expr_from_set(
	__isl_keep isl_ast_build *build, __isl_take isl_set *set);
__isl_overload
__isl_give isl_ast_expr *isl_ast_build_expr_from_pw_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_pw_aff *pa);
__isl_overload
__isl_give isl_ast_expr *isl_ast_build_access_from_pw_multi_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_pw_multi_aff *pma);
__isl_overload
__isl_give isl_ast_expr *isl_ast_build_access_from_multi_pw_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_multi_pw_aff *mpa);
__isl_overload
__isl_give isl_ast_expr *isl_ast_build_call_from_pw_multi_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_pw_multi_aff *pma);
__isl_overload
__isl_give isl_ast_expr *isl_ast_build_call_from_multi_pw_aff(
	__isl_keep isl_ast_build *build, __isl_take isl_multi_pw_aff *mpa);

__isl_give isl_ast_node *isl_ast_build_node_from_schedule(
	__isl_keep isl_ast_build *build, __isl_take isl_schedule *schedule);
__isl_export
__isl_give isl_ast_node *isl_ast_build_node_from_schedule_map(
	__isl_keep isl_ast_build *build, __isl_take isl_union_map *schedule);
__isl_give isl_ast_node *isl_ast_build_ast_from_schedule(
	__isl_keep isl_ast_build *build, __isl_take isl_union_map *schedule);

#if defined(__cplusplus)
}
#endif

#endif
