#ifndef ISL_AST_H
#define ISL_AST_H

#include <isl/ctx.h>
#include <isl/ast_type.h>
#include <isl/id.h>
#include <isl/id_to_ast_expr.h>
#include <isl/val.h>
#include <isl/list.h>
#include <isl/printer.h>

#if defined(__cplusplus)
extern "C" {
#endif

isl_stat isl_options_set_ast_iterator_type(isl_ctx *ctx, const char *val);
const char *isl_options_get_ast_iterator_type(isl_ctx *ctx);

isl_stat isl_options_set_ast_always_print_block(isl_ctx *ctx, int val);
int isl_options_get_ast_always_print_block(isl_ctx *ctx);

__isl_give isl_ast_expr *isl_ast_expr_from_val(__isl_take isl_val *v);
__isl_give isl_ast_expr *isl_ast_expr_from_id(__isl_take isl_id *id);
__isl_give isl_ast_expr *isl_ast_expr_neg(__isl_take isl_ast_expr *expr);
__isl_give isl_ast_expr *isl_ast_expr_add(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_sub(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_mul(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_div(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_pdiv_q(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_pdiv_r(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_and(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_and_then(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_or(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_or_else(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_le(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_lt(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_ge(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_gt(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_eq(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2);
__isl_give isl_ast_expr *isl_ast_expr_access(__isl_take isl_ast_expr *array,
	__isl_take isl_ast_expr_list *indices);
__isl_give isl_ast_expr *isl_ast_expr_call(__isl_take isl_ast_expr *function,
	__isl_take isl_ast_expr_list *arguments);
__isl_give isl_ast_expr *isl_ast_expr_address_of(__isl_take isl_ast_expr *expr);

__isl_give isl_ast_expr *isl_ast_expr_copy(__isl_keep isl_ast_expr *expr);
__isl_null isl_ast_expr *isl_ast_expr_free(__isl_take isl_ast_expr *expr);

isl_ctx *isl_ast_expr_get_ctx(__isl_keep isl_ast_expr *expr);
enum isl_ast_expr_type isl_ast_expr_get_type(__isl_keep isl_ast_expr *expr);
__isl_give isl_val *isl_ast_expr_get_val(__isl_keep isl_ast_expr *expr);
__isl_give isl_id *isl_ast_expr_get_id(__isl_keep isl_ast_expr *expr);

enum isl_ast_op_type isl_ast_expr_get_op_type(__isl_keep isl_ast_expr *expr);
int isl_ast_expr_get_op_n_arg(__isl_keep isl_ast_expr *expr);
__isl_give isl_ast_expr *isl_ast_expr_get_op_arg(__isl_keep isl_ast_expr *expr,
	int pos);
__isl_give isl_ast_expr *isl_ast_expr_set_op_arg(__isl_take isl_ast_expr *expr,
	int pos, __isl_take isl_ast_expr *arg);

isl_bool isl_ast_expr_is_equal(__isl_keep isl_ast_expr *expr1,
	__isl_keep isl_ast_expr *expr2);

__isl_give isl_ast_expr *isl_ast_expr_substitute_ids(
	__isl_take isl_ast_expr *expr, __isl_take isl_id_to_ast_expr *id2expr);

__isl_give isl_printer *isl_printer_print_ast_expr(__isl_take isl_printer *p,
	__isl_keep isl_ast_expr *expr);
void isl_ast_expr_dump(__isl_keep isl_ast_expr *expr);
__isl_give char *isl_ast_expr_to_str(__isl_keep isl_ast_expr *expr);
__isl_export
__isl_give char *isl_ast_expr_to_C_str(__isl_keep isl_ast_expr *expr);

__isl_give isl_ast_node *isl_ast_node_alloc_user(__isl_take isl_ast_expr *expr);
__isl_give isl_ast_node *isl_ast_node_copy(__isl_keep isl_ast_node *node);
__isl_null isl_ast_node *isl_ast_node_free(__isl_take isl_ast_node *node);

isl_ctx *isl_ast_node_get_ctx(__isl_keep isl_ast_node *node);
enum isl_ast_node_type isl_ast_node_get_type(__isl_keep isl_ast_node *node);

__isl_give isl_ast_node *isl_ast_node_set_annotation(
	__isl_take isl_ast_node *node, __isl_take isl_id *annotation);
__isl_give isl_id *isl_ast_node_get_annotation(__isl_keep isl_ast_node *node);

__isl_give isl_ast_expr *isl_ast_node_for_get_iterator(
	__isl_keep isl_ast_node *node);
__isl_give isl_ast_expr *isl_ast_node_for_get_init(
	__isl_keep isl_ast_node *node);
__isl_give isl_ast_expr *isl_ast_node_for_get_cond(
	__isl_keep isl_ast_node *node);
__isl_give isl_ast_expr *isl_ast_node_for_get_inc(
	__isl_keep isl_ast_node *node);
__isl_give isl_ast_node *isl_ast_node_for_get_body(
	__isl_keep isl_ast_node *node);
isl_bool isl_ast_node_for_is_degenerate(__isl_keep isl_ast_node *node);

__isl_give isl_ast_expr *isl_ast_node_if_get_cond(
	__isl_keep isl_ast_node *node);
__isl_give isl_ast_node *isl_ast_node_if_get_then(
	__isl_keep isl_ast_node *node);
isl_bool isl_ast_node_if_has_else(__isl_keep isl_ast_node *node);
__isl_give isl_ast_node *isl_ast_node_if_get_else(
	__isl_keep isl_ast_node *node);

__isl_give isl_ast_node_list *isl_ast_node_block_get_children(
	__isl_keep isl_ast_node *node);

__isl_give isl_id *isl_ast_node_mark_get_id(__isl_keep isl_ast_node *node);
__isl_give isl_ast_node *isl_ast_node_mark_get_node(
	__isl_keep isl_ast_node *node);

__isl_give isl_ast_expr *isl_ast_node_user_get_expr(
	__isl_keep isl_ast_node *node);

isl_stat isl_ast_node_foreach_descendant_top_down(
	__isl_keep isl_ast_node *node,
	isl_bool (*fn)(__isl_keep isl_ast_node *node, void *user), void *user);

__isl_give isl_printer *isl_printer_print_ast_node(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node);
void isl_ast_node_dump(__isl_keep isl_ast_node *node);
__isl_give char *isl_ast_node_to_str(__isl_keep isl_ast_node *node);

__isl_give isl_ast_print_options *isl_ast_print_options_alloc(isl_ctx *ctx);
__isl_give isl_ast_print_options *isl_ast_print_options_copy(
	__isl_keep isl_ast_print_options *options);
__isl_null isl_ast_print_options *isl_ast_print_options_free(
	__isl_take isl_ast_print_options *options);
isl_ctx *isl_ast_print_options_get_ctx(
	__isl_keep isl_ast_print_options *options);

__isl_give isl_ast_print_options *isl_ast_print_options_set_print_user(
	__isl_take isl_ast_print_options *options,
	__isl_give isl_printer *(*print_user)(__isl_take isl_printer *p,
		__isl_take isl_ast_print_options *options,
		__isl_keep isl_ast_node *node, void *user),
	void *user);
__isl_give isl_ast_print_options *isl_ast_print_options_set_print_for(
	__isl_take isl_ast_print_options *options,
	__isl_give isl_printer *(*print_for)(__isl_take isl_printer *p,
		__isl_take isl_ast_print_options *options,
		__isl_keep isl_ast_node *node, void *user),
	void *user);

isl_stat isl_options_set_ast_print_macro_once(isl_ctx *ctx, int val);
int isl_options_get_ast_print_macro_once(isl_ctx *ctx);

isl_stat isl_ast_expr_foreach_ast_op_type(__isl_keep isl_ast_expr *expr,
	isl_stat (*fn)(enum isl_ast_op_type type, void *user), void *user);
isl_stat isl_ast_node_foreach_ast_op_type(__isl_keep isl_ast_node *node,
	isl_stat (*fn)(enum isl_ast_op_type type, void *user), void *user);
__isl_give isl_printer *isl_ast_op_type_set_print_name(
	__isl_take isl_printer *p, enum isl_ast_op_type type,
	__isl_keep const char *name);
__isl_give isl_printer *isl_ast_op_type_print_macro(
	enum isl_ast_op_type type, __isl_take isl_printer *p);
__isl_give isl_printer *isl_ast_expr_print_macros(
	__isl_keep isl_ast_expr *expr, __isl_take isl_printer *p);
__isl_give isl_printer *isl_ast_node_print_macros(
	__isl_keep isl_ast_node *node, __isl_take isl_printer *p);
__isl_give isl_printer *isl_ast_node_print(__isl_keep isl_ast_node *node,
	__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *options);
__isl_give isl_printer *isl_ast_node_for_print(__isl_keep isl_ast_node *node,
	__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *options);
__isl_give isl_printer *isl_ast_node_if_print(__isl_keep isl_ast_node *node,
	__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *options);

__isl_export
__isl_give char *isl_ast_node_to_C_str(__isl_keep isl_ast_node *node);

#if defined(__cplusplus)
}
#endif

#endif
