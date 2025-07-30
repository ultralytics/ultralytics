#ifndef ISL_AST_TYPE_H
#define ISL_AST_TYPE_H

#include <isl/list.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct __isl_export isl_ast_expr;
typedef struct isl_ast_expr isl_ast_expr;

struct __isl_export isl_ast_node;
typedef struct isl_ast_node isl_ast_node;

enum isl_ast_op_type {
	isl_ast_op_error = -1,
	isl_ast_op_and,
	isl_ast_op_and_then,
	isl_ast_op_or,
	isl_ast_op_or_else,
	isl_ast_op_max,
	isl_ast_op_min,
	isl_ast_op_minus,
	isl_ast_op_add,
	isl_ast_op_sub,
	isl_ast_op_mul,
	isl_ast_op_div,
	isl_ast_op_fdiv_q,	/* Round towards -infty */
	isl_ast_op_pdiv_q,	/* Dividend is non-negative */
	isl_ast_op_pdiv_r,	/* Dividend is non-negative */
	isl_ast_op_zdiv_r,	/* Result only compared against zero */
	isl_ast_op_cond,
	isl_ast_op_select,
	isl_ast_op_eq,
	isl_ast_op_le,
	isl_ast_op_lt,
	isl_ast_op_ge,
	isl_ast_op_gt,
	isl_ast_op_call,
	isl_ast_op_access,
	isl_ast_op_member,
	isl_ast_op_address_of
};

enum isl_ast_expr_type {
	isl_ast_expr_error = -1,
	isl_ast_expr_op,
	isl_ast_expr_id,
	isl_ast_expr_int
};

enum isl_ast_node_type {
	isl_ast_node_error = -1,
	isl_ast_node_for = 1,
	isl_ast_node_if,
	isl_ast_node_block,
	isl_ast_node_mark,
	isl_ast_node_user
};

enum isl_ast_loop_type {
	isl_ast_loop_error = -1,
	isl_ast_loop_default = 0,
	isl_ast_loop_atomic,
	isl_ast_loop_unroll,
	isl_ast_loop_separate
};

struct isl_ast_print_options;
typedef struct isl_ast_print_options isl_ast_print_options;

ISL_DECLARE_LIST(ast_expr)
ISL_DECLARE_LIST(ast_node)

#if defined(__cplusplus)
}
#endif

#endif
