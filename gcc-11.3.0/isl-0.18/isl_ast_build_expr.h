#ifndef ISL_AST_BUILD_EXPR_PRIVATE_H
#define ISL_AST_BUILD_EXPR_PRIVATE_H

#include <isl/ast.h>
#include <isl/ast_build.h>

__isl_give isl_ast_expr *isl_ast_build_expr_from_basic_set(
	 __isl_keep isl_ast_build *build, __isl_take isl_basic_set *bset);
__isl_give isl_ast_expr *isl_ast_build_expr_from_set_internal(
	__isl_keep isl_ast_build *build, __isl_take isl_set *set);

__isl_give isl_ast_expr *isl_ast_build_expr_from_pw_aff_internal(
	__isl_keep isl_ast_build *build, __isl_take isl_pw_aff *pa);
__isl_give isl_ast_expr *isl_ast_expr_from_aff(__isl_take isl_aff *aff,
	__isl_keep isl_ast_build *build);
__isl_give isl_ast_expr *isl_ast_expr_set_op_arg(__isl_take isl_ast_expr *expr,
	int pos, __isl_take isl_ast_expr *arg);

__isl_give isl_ast_node *isl_ast_build_call_from_executed(
	__isl_keep isl_ast_build *build, __isl_take isl_map *executed);

#endif
