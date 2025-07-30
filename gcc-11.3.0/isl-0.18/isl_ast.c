/*
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <string.h>

#include <isl_ast_private.h>

#undef BASE
#define BASE ast_expr

#include <isl_list_templ.c>

#undef BASE
#define BASE ast_node

#include <isl_list_templ.c>

isl_ctx *isl_ast_print_options_get_ctx(
	__isl_keep isl_ast_print_options *options)
{
	return options ? options->ctx : NULL;
}

__isl_give isl_ast_print_options *isl_ast_print_options_alloc(isl_ctx *ctx)
{
	isl_ast_print_options *options;

	options = isl_calloc_type(ctx, isl_ast_print_options);
	if (!options)
		return NULL;

	options->ctx = ctx;
	isl_ctx_ref(ctx);
	options->ref = 1;

	return options;
}

__isl_give isl_ast_print_options *isl_ast_print_options_dup(
	__isl_keep isl_ast_print_options *options)
{
	isl_ctx *ctx;
	isl_ast_print_options *dup;

	if (!options)
		return NULL;

	ctx = isl_ast_print_options_get_ctx(options);
	dup = isl_ast_print_options_alloc(ctx);
	if (!dup)
		return NULL;

	dup->print_for = options->print_for;
	dup->print_for_user = options->print_for_user;
	dup->print_user = options->print_user;
	dup->print_user_user = options->print_user_user;

	return dup;
}

__isl_give isl_ast_print_options *isl_ast_print_options_cow(
	__isl_take isl_ast_print_options *options)
{
	if (!options)
		return NULL;

	if (options->ref == 1)
		return options;
	options->ref--;
	return isl_ast_print_options_dup(options);
}

__isl_give isl_ast_print_options *isl_ast_print_options_copy(
	__isl_keep isl_ast_print_options *options)
{
	if (!options)
		return NULL;

	options->ref++;
	return options;
}

__isl_null isl_ast_print_options *isl_ast_print_options_free(
	__isl_take isl_ast_print_options *options)
{
	if (!options)
		return NULL;

	if (--options->ref > 0)
		return NULL;

	isl_ctx_deref(options->ctx);

	free(options);
	return NULL;
}

/* Set the print_user callback of "options" to "print_user".
 *
 * If this callback is set, then it used to print user nodes in the AST.
 * Otherwise, the expression associated to the user node is printed.
 */
__isl_give isl_ast_print_options *isl_ast_print_options_set_print_user(
	__isl_take isl_ast_print_options *options,
	__isl_give isl_printer *(*print_user)(__isl_take isl_printer *p,
		__isl_take isl_ast_print_options *options,
		__isl_keep isl_ast_node *node, void *user),
	void *user)
{
	options = isl_ast_print_options_cow(options);
	if (!options)
		return NULL;

	options->print_user = print_user;
	options->print_user_user = user;

	return options;
}

/* Set the print_for callback of "options" to "print_for".
 *
 * If this callback is set, then it used to print for nodes in the AST.
 */
__isl_give isl_ast_print_options *isl_ast_print_options_set_print_for(
	__isl_take isl_ast_print_options *options,
	__isl_give isl_printer *(*print_for)(__isl_take isl_printer *p,
		__isl_take isl_ast_print_options *options,
		__isl_keep isl_ast_node *node, void *user),
	void *user)
{
	options = isl_ast_print_options_cow(options);
	if (!options)
		return NULL;

	options->print_for = print_for;
	options->print_for_user = user;

	return options;
}

__isl_give isl_ast_expr *isl_ast_expr_copy(__isl_keep isl_ast_expr *expr)
{
	if (!expr)
		return NULL;

	expr->ref++;
	return expr;
}

__isl_give isl_ast_expr *isl_ast_expr_dup(__isl_keep isl_ast_expr *expr)
{
	int i;
	isl_ctx *ctx;
	isl_ast_expr *dup;

	if (!expr)
		return NULL;

	ctx = isl_ast_expr_get_ctx(expr);
	switch (expr->type) {
	case isl_ast_expr_int:
		dup = isl_ast_expr_from_val(isl_val_copy(expr->u.v));
		break;
	case isl_ast_expr_id:
		dup = isl_ast_expr_from_id(isl_id_copy(expr->u.id));
		break;
	case isl_ast_expr_op:
		dup = isl_ast_expr_alloc_op(ctx,
					    expr->u.op.op, expr->u.op.n_arg);
		if (!dup)
			return NULL;
		for (i = 0; i < expr->u.op.n_arg; ++i)
			dup->u.op.args[i] =
				isl_ast_expr_copy(expr->u.op.args[i]);
		break;
	case isl_ast_expr_error:
		dup = NULL;
	}

	if (!dup)
		return NULL;

	return dup;
}

__isl_give isl_ast_expr *isl_ast_expr_cow(__isl_take isl_ast_expr *expr)
{
	if (!expr)
		return NULL;

	if (expr->ref == 1)
		return expr;
	expr->ref--;
	return isl_ast_expr_dup(expr);
}

__isl_null isl_ast_expr *isl_ast_expr_free(__isl_take isl_ast_expr *expr)
{
	int i;

	if (!expr)
		return NULL;

	if (--expr->ref > 0)
		return NULL;

	isl_ctx_deref(expr->ctx);

	switch (expr->type) {
	case isl_ast_expr_int:
		isl_val_free(expr->u.v);
		break;
	case isl_ast_expr_id:
		isl_id_free(expr->u.id);
		break;
	case isl_ast_expr_op:
		if (expr->u.op.args)
			for (i = 0; i < expr->u.op.n_arg; ++i)
				isl_ast_expr_free(expr->u.op.args[i]);
		free(expr->u.op.args);
		break;
	case isl_ast_expr_error:
		break;
	}

	free(expr);
	return NULL;
}

isl_ctx *isl_ast_expr_get_ctx(__isl_keep isl_ast_expr *expr)
{
	return expr ? expr->ctx : NULL;
}

enum isl_ast_expr_type isl_ast_expr_get_type(__isl_keep isl_ast_expr *expr)
{
	return expr ? expr->type : isl_ast_expr_error;
}

/* Return the integer value represented by "expr".
 */
__isl_give isl_val *isl_ast_expr_get_val(__isl_keep isl_ast_expr *expr)
{
	if (!expr)
		return NULL;
	if (expr->type != isl_ast_expr_int)
		isl_die(isl_ast_expr_get_ctx(expr), isl_error_invalid,
			"expression not an int", return NULL);
	return isl_val_copy(expr->u.v);
}

__isl_give isl_id *isl_ast_expr_get_id(__isl_keep isl_ast_expr *expr)
{
	if (!expr)
		return NULL;
	if (expr->type != isl_ast_expr_id)
		isl_die(isl_ast_expr_get_ctx(expr), isl_error_invalid,
			"expression not an identifier", return NULL);

	return isl_id_copy(expr->u.id);
}

enum isl_ast_op_type isl_ast_expr_get_op_type(__isl_keep isl_ast_expr *expr)
{
	if (!expr)
		return isl_ast_op_error;
	if (expr->type != isl_ast_expr_op)
		isl_die(isl_ast_expr_get_ctx(expr), isl_error_invalid,
			"expression not an operation", return isl_ast_op_error);
	return expr->u.op.op;
}

int isl_ast_expr_get_op_n_arg(__isl_keep isl_ast_expr *expr)
{
	if (!expr)
		return -1;
	if (expr->type != isl_ast_expr_op)
		isl_die(isl_ast_expr_get_ctx(expr), isl_error_invalid,
			"expression not an operation", return -1);
	return expr->u.op.n_arg;
}

__isl_give isl_ast_expr *isl_ast_expr_get_op_arg(__isl_keep isl_ast_expr *expr,
	int pos)
{
	if (!expr)
		return NULL;
	if (expr->type != isl_ast_expr_op)
		isl_die(isl_ast_expr_get_ctx(expr), isl_error_invalid,
			"expression not an operation", return NULL);
	if (pos < 0 || pos >= expr->u.op.n_arg)
		isl_die(isl_ast_expr_get_ctx(expr), isl_error_invalid,
			"index out of bounds", return NULL);

	return isl_ast_expr_copy(expr->u.op.args[pos]);
}

/* Replace the argument at position "pos" of "expr" by "arg".
 */
__isl_give isl_ast_expr *isl_ast_expr_set_op_arg(__isl_take isl_ast_expr *expr,
	int pos, __isl_take isl_ast_expr *arg)
{
	expr = isl_ast_expr_cow(expr);
	if (!expr || !arg)
		goto error;
	if (expr->type != isl_ast_expr_op)
		isl_die(isl_ast_expr_get_ctx(expr), isl_error_invalid,
			"expression not an operation", goto error);
	if (pos < 0 || pos >= expr->u.op.n_arg)
		isl_die(isl_ast_expr_get_ctx(expr), isl_error_invalid,
			"index out of bounds", goto error);

	isl_ast_expr_free(expr->u.op.args[pos]);
	expr->u.op.args[pos] = arg;

	return expr;
error:
	isl_ast_expr_free(arg);
	return isl_ast_expr_free(expr);
}

/* Is "expr1" equal to "expr2"?
 */
isl_bool isl_ast_expr_is_equal(__isl_keep isl_ast_expr *expr1,
	__isl_keep isl_ast_expr *expr2)
{
	int i;

	if (!expr1 || !expr2)
		return isl_bool_error;

	if (expr1 == expr2)
		return isl_bool_true;
	if (expr1->type != expr2->type)
		return isl_bool_false;
	switch (expr1->type) {
	case isl_ast_expr_int:
		return isl_val_eq(expr1->u.v, expr2->u.v);
	case isl_ast_expr_id:
		return expr1->u.id == expr2->u.id;
	case isl_ast_expr_op:
		if (expr1->u.op.op != expr2->u.op.op)
			return isl_bool_false;
		if (expr1->u.op.n_arg != expr2->u.op.n_arg)
			return isl_bool_false;
		for (i = 0; i < expr1->u.op.n_arg; ++i) {
			isl_bool equal;
			equal = isl_ast_expr_is_equal(expr1->u.op.args[i],
							expr2->u.op.args[i]);
			if (equal < 0 || !equal)
				return equal;
		}
		return 1;
	case isl_ast_expr_error:
		return isl_bool_error;
	}

	isl_die(isl_ast_expr_get_ctx(expr1), isl_error_internal,
		"unhandled case", return isl_bool_error);
}

/* Create a new operation expression of operation type "op",
 * with "n_arg" as yet unspecified arguments.
 */
__isl_give isl_ast_expr *isl_ast_expr_alloc_op(isl_ctx *ctx,
	enum isl_ast_op_type op, int n_arg)
{
	isl_ast_expr *expr;

	expr = isl_calloc_type(ctx, isl_ast_expr);
	if (!expr)
		return NULL;

	expr->ctx = ctx;
	isl_ctx_ref(ctx);
	expr->ref = 1;
	expr->type = isl_ast_expr_op;
	expr->u.op.op = op;
	expr->u.op.n_arg = n_arg;
	expr->u.op.args = isl_calloc_array(ctx, isl_ast_expr *, n_arg);

	if (n_arg && !expr->u.op.args)
		return isl_ast_expr_free(expr);

	return expr;
}

/* Create a new id expression representing "id".
 */
__isl_give isl_ast_expr *isl_ast_expr_from_id(__isl_take isl_id *id)
{
	isl_ctx *ctx;
	isl_ast_expr *expr;

	if (!id)
		return NULL;

	ctx = isl_id_get_ctx(id);
	expr = isl_calloc_type(ctx, isl_ast_expr);
	if (!expr)
		goto error;

	expr->ctx = ctx;
	isl_ctx_ref(ctx);
	expr->ref = 1;
	expr->type = isl_ast_expr_id;
	expr->u.id = id;

	return expr;
error:
	isl_id_free(id);
	return NULL;
}

/* Create a new integer expression representing "i".
 */
__isl_give isl_ast_expr *isl_ast_expr_alloc_int_si(isl_ctx *ctx, int i)
{
	isl_ast_expr *expr;

	expr = isl_calloc_type(ctx, isl_ast_expr);
	if (!expr)
		return NULL;

	expr->ctx = ctx;
	isl_ctx_ref(ctx);
	expr->ref = 1;
	expr->type = isl_ast_expr_int;
	expr->u.v = isl_val_int_from_si(ctx, i);
	if (!expr->u.v)
		return isl_ast_expr_free(expr);

	return expr;
}

/* Create a new integer expression representing "v".
 */
__isl_give isl_ast_expr *isl_ast_expr_from_val(__isl_take isl_val *v)
{
	isl_ctx *ctx;
	isl_ast_expr *expr;

	if (!v)
		return NULL;
	if (!isl_val_is_int(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting integer value", goto error);

	ctx = isl_val_get_ctx(v);
	expr = isl_calloc_type(ctx, isl_ast_expr);
	if (!expr)
		goto error;

	expr->ctx = ctx;
	isl_ctx_ref(ctx);
	expr->ref = 1;
	expr->type = isl_ast_expr_int;
	expr->u.v = v;

	return expr;
error:
	isl_val_free(v);
	return NULL;
}

/* Create an expression representing the unary operation "type" applied to
 * "arg".
 */
__isl_give isl_ast_expr *isl_ast_expr_alloc_unary(enum isl_ast_op_type type,
	__isl_take isl_ast_expr *arg)
{
	isl_ctx *ctx;
	isl_ast_expr *expr = NULL;

	if (!arg)
		return NULL;

	ctx = isl_ast_expr_get_ctx(arg);
	expr = isl_ast_expr_alloc_op(ctx, type, 1);
	if (!expr)
		goto error;

	expr->u.op.args[0] = arg;

	return expr;
error:
	isl_ast_expr_free(arg);
	return NULL;
}

/* Create an expression representing the negation of "arg".
 */
__isl_give isl_ast_expr *isl_ast_expr_neg(__isl_take isl_ast_expr *arg)
{
	return isl_ast_expr_alloc_unary(isl_ast_op_minus, arg);
}

/* Create an expression representing the address of "expr".
 */
__isl_give isl_ast_expr *isl_ast_expr_address_of(__isl_take isl_ast_expr *expr)
{
	if (!expr)
		return NULL;

	if (isl_ast_expr_get_type(expr) != isl_ast_expr_op ||
	    isl_ast_expr_get_op_type(expr) != isl_ast_op_access)
		isl_die(isl_ast_expr_get_ctx(expr), isl_error_invalid,
			"can only take address of access expressions",
			return isl_ast_expr_free(expr));

	return isl_ast_expr_alloc_unary(isl_ast_op_address_of, expr);
}

/* Create an expression representing the binary operation "type"
 * applied to "expr1" and "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_alloc_binary(enum isl_ast_op_type type,
	__isl_take isl_ast_expr *expr1, __isl_take isl_ast_expr *expr2)
{
	isl_ctx *ctx;
	isl_ast_expr *expr = NULL;

	if (!expr1 || !expr2)
		goto error;

	ctx = isl_ast_expr_get_ctx(expr1);
	expr = isl_ast_expr_alloc_op(ctx, type, 2);
	if (!expr)
		goto error;

	expr->u.op.args[0] = expr1;
	expr->u.op.args[1] = expr2;

	return expr;
error:
	isl_ast_expr_free(expr1);
	isl_ast_expr_free(expr2);
	return NULL;
}

/* Create an expression representing the sum of "expr1" and "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_add(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_add, expr1, expr2);
}

/* Create an expression representing the difference of "expr1" and "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_sub(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_sub, expr1, expr2);
}

/* Create an expression representing the product of "expr1" and "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_mul(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_mul, expr1, expr2);
}

/* Create an expression representing the quotient of "expr1" and "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_div(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_div, expr1, expr2);
}

/* Create an expression representing the quotient of the integer
 * division of "expr1" by "expr2", where "expr1" is known to be
 * non-negative.
 */
__isl_give isl_ast_expr *isl_ast_expr_pdiv_q(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_pdiv_q, expr1, expr2);
}

/* Create an expression representing the remainder of the integer
 * division of "expr1" by "expr2", where "expr1" is known to be
 * non-negative.
 */
__isl_give isl_ast_expr *isl_ast_expr_pdiv_r(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_pdiv_r, expr1, expr2);
}

/* Create an expression representing the conjunction of "expr1" and "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_and(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_and, expr1, expr2);
}

/* Create an expression representing the conjunction of "expr1" and "expr2",
 * where "expr2" is evaluated only if "expr1" is evaluated to true.
 */
__isl_give isl_ast_expr *isl_ast_expr_and_then(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_and_then, expr1, expr2);
}

/* Create an expression representing the disjunction of "expr1" and "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_or(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_or, expr1, expr2);
}

/* Create an expression representing the disjunction of "expr1" and "expr2",
 * where "expr2" is evaluated only if "expr1" is evaluated to false.
 */
__isl_give isl_ast_expr *isl_ast_expr_or_else(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_or_else, expr1, expr2);
}

/* Create an expression representing "expr1" less than or equal to "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_le(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_le, expr1, expr2);
}

/* Create an expression representing "expr1" less than "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_lt(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_lt, expr1, expr2);
}

/* Create an expression representing "expr1" greater than or equal to "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_ge(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_ge, expr1, expr2);
}

/* Create an expression representing "expr1" greater than "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_gt(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_gt, expr1, expr2);
}

/* Create an expression representing "expr1" equal to "expr2".
 */
__isl_give isl_ast_expr *isl_ast_expr_eq(__isl_take isl_ast_expr *expr1,
	__isl_take isl_ast_expr *expr2)
{
	return isl_ast_expr_alloc_binary(isl_ast_op_eq, expr1, expr2);
}

/* Create an expression of type "type" with as arguments "arg0" followed
 * by "arguments".
 */
static __isl_give isl_ast_expr *ast_expr_with_arguments(
	enum isl_ast_op_type type, __isl_take isl_ast_expr *arg0,
	__isl_take isl_ast_expr_list *arguments)
{
	int i, n;
	isl_ctx *ctx;
	isl_ast_expr *res = NULL;

	if (!arg0 || !arguments)
		goto error;

	ctx = isl_ast_expr_get_ctx(arg0);
	n = isl_ast_expr_list_n_ast_expr(arguments);
	res = isl_ast_expr_alloc_op(ctx, type, 1 + n);
	if (!res)
		goto error;
	for (i = 0; i < n; ++i) {
		isl_ast_expr *arg;
		arg = isl_ast_expr_list_get_ast_expr(arguments, i);
		res->u.op.args[1 + i] = arg;
		if (!arg)
			goto error;
	}
	res->u.op.args[0] = arg0;

	isl_ast_expr_list_free(arguments);
	return res;
error:
	isl_ast_expr_free(arg0);
	isl_ast_expr_list_free(arguments);
	isl_ast_expr_free(res);
	return NULL;
}

/* Create an expression representing an access to "array" with index
 * expressions "indices".
 */
__isl_give isl_ast_expr *isl_ast_expr_access(__isl_take isl_ast_expr *array,
	__isl_take isl_ast_expr_list *indices)
{
	return ast_expr_with_arguments(isl_ast_op_access, array, indices);
}

/* Create an expression representing a call to "function" with argument
 * expressions "arguments".
 */
__isl_give isl_ast_expr *isl_ast_expr_call(__isl_take isl_ast_expr *function,
	__isl_take isl_ast_expr_list *arguments)
{
	return ast_expr_with_arguments(isl_ast_op_call, function, arguments);
}

/* For each subexpression of "expr" of type isl_ast_expr_id,
 * if it appears in "id2expr", then replace it by the corresponding
 * expression.
 */
__isl_give isl_ast_expr *isl_ast_expr_substitute_ids(
	__isl_take isl_ast_expr *expr, __isl_take isl_id_to_ast_expr *id2expr)
{
	int i;
	isl_maybe_isl_ast_expr m;

	if (!expr || !id2expr)
		goto error;

	switch (expr->type) {
	case isl_ast_expr_int:
		break;
	case isl_ast_expr_id:
		m = isl_id_to_ast_expr_try_get(id2expr, expr->u.id);
		if (m.valid < 0)
			goto error;
		if (!m.valid)
			break;
		isl_ast_expr_free(expr);
		expr = m.value;
		break;
	case isl_ast_expr_op:
		for (i = 0; i < expr->u.op.n_arg; ++i) {
			isl_ast_expr *arg;
			arg = isl_ast_expr_copy(expr->u.op.args[i]);
			arg = isl_ast_expr_substitute_ids(arg,
					    isl_id_to_ast_expr_copy(id2expr));
			if (arg == expr->u.op.args[i]) {
				isl_ast_expr_free(arg);
				continue;
			}
			if (!arg)
				expr = isl_ast_expr_free(expr);
			expr = isl_ast_expr_cow(expr);
			if (!expr) {
				isl_ast_expr_free(arg);
				break;
			}
			isl_ast_expr_free(expr->u.op.args[i]);
			expr->u.op.args[i] = arg;
		}
		break;
	case isl_ast_expr_error:
		expr = isl_ast_expr_free(expr);
		break;
	}

	isl_id_to_ast_expr_free(id2expr);
	return expr;
error:
	isl_ast_expr_free(expr);
	isl_id_to_ast_expr_free(id2expr);
	return NULL;
}

isl_ctx *isl_ast_node_get_ctx(__isl_keep isl_ast_node *node)
{
	return node ? node->ctx : NULL;
}

enum isl_ast_node_type isl_ast_node_get_type(__isl_keep isl_ast_node *node)
{
	return node ? node->type : isl_ast_node_error;
}

__isl_give isl_ast_node *isl_ast_node_alloc(isl_ctx *ctx,
	enum isl_ast_node_type type)
{
	isl_ast_node *node;

	node = isl_calloc_type(ctx, isl_ast_node);
	if (!node)
		return NULL;

	node->ctx = ctx;
	isl_ctx_ref(ctx);
	node->ref = 1;
	node->type = type;

	return node;
}

/* Create an if node with the given guard.
 *
 * The then body needs to be filled in later.
 */
__isl_give isl_ast_node *isl_ast_node_alloc_if(__isl_take isl_ast_expr *guard)
{
	isl_ast_node *node;

	if (!guard)
		return NULL;

	node = isl_ast_node_alloc(isl_ast_expr_get_ctx(guard), isl_ast_node_if);
	if (!node)
		goto error;
	node->u.i.guard = guard;

	return node;
error:
	isl_ast_expr_free(guard);
	return NULL;
}

/* Create a for node with the given iterator.
 *
 * The remaining fields need to be filled in later.
 */
__isl_give isl_ast_node *isl_ast_node_alloc_for(__isl_take isl_id *id)
{
	isl_ast_node *node;
	isl_ctx *ctx;

	if (!id)
		return NULL;

	ctx = isl_id_get_ctx(id);
	node = isl_ast_node_alloc(ctx, isl_ast_node_for);
	if (!node)
		goto error;

	node->u.f.iterator = isl_ast_expr_from_id(id);
	if (!node->u.f.iterator)
		return isl_ast_node_free(node);

	return node;
error:
	isl_id_free(id);
	return NULL;
}

/* Create a mark node, marking "node" with "id".
 */
__isl_give isl_ast_node *isl_ast_node_alloc_mark(__isl_take isl_id *id,
	__isl_take isl_ast_node *node)
{
	isl_ctx *ctx;
	isl_ast_node *mark;

	if (!id || !node)
		goto error;

	ctx = isl_id_get_ctx(id);
	mark = isl_ast_node_alloc(ctx, isl_ast_node_mark);
	if (!mark)
		goto error;

	mark->u.m.mark = id;
	mark->u.m.node = node;

	return mark;
error:
	isl_id_free(id);
	isl_ast_node_free(node);
	return NULL;
}

/* Create a user node evaluating "expr".
 */
__isl_give isl_ast_node *isl_ast_node_alloc_user(__isl_take isl_ast_expr *expr)
{
	isl_ctx *ctx;
	isl_ast_node *node;

	if (!expr)
		return NULL;

	ctx = isl_ast_expr_get_ctx(expr);
	node = isl_ast_node_alloc(ctx, isl_ast_node_user);
	if (!node)
		goto error;

	node->u.e.expr = expr;

	return node;
error:
	isl_ast_expr_free(expr);
	return NULL;
}

/* Create a block node with the given children.
 */
__isl_give isl_ast_node *isl_ast_node_alloc_block(
	__isl_take isl_ast_node_list *list)
{
	isl_ast_node *node;
	isl_ctx *ctx;

	if (!list)
		return NULL;

	ctx = isl_ast_node_list_get_ctx(list);
	node = isl_ast_node_alloc(ctx, isl_ast_node_block);
	if (!node)
		goto error;

	node->u.b.children = list;

	return node;
error:
	isl_ast_node_list_free(list);
	return NULL;
}

/* Represent the given list of nodes as a single node, either by
 * extract the node from a single element list or by creating
 * a block node with the list of nodes as children.
 */
__isl_give isl_ast_node *isl_ast_node_from_ast_node_list(
	__isl_take isl_ast_node_list *list)
{
	isl_ast_node *node;

	if (isl_ast_node_list_n_ast_node(list) != 1)
		return isl_ast_node_alloc_block(list);

	node = isl_ast_node_list_get_ast_node(list, 0);
	isl_ast_node_list_free(list);

	return node;
}

__isl_give isl_ast_node *isl_ast_node_copy(__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;

	node->ref++;
	return node;
}

__isl_give isl_ast_node *isl_ast_node_dup(__isl_keep isl_ast_node *node)
{
	isl_ast_node *dup;

	if (!node)
		return NULL;

	dup = isl_ast_node_alloc(isl_ast_node_get_ctx(node), node->type);
	if (!dup)
		return NULL;

	switch (node->type) {
	case isl_ast_node_if:
		dup->u.i.guard = isl_ast_expr_copy(node->u.i.guard);
		dup->u.i.then = isl_ast_node_copy(node->u.i.then);
		dup->u.i.else_node = isl_ast_node_copy(node->u.i.else_node);
		if (!dup->u.i.guard  || !dup->u.i.then ||
		    (node->u.i.else_node && !dup->u.i.else_node))
			return isl_ast_node_free(dup);
		break;
	case isl_ast_node_for:
		dup->u.f.iterator = isl_ast_expr_copy(node->u.f.iterator);
		dup->u.f.init = isl_ast_expr_copy(node->u.f.init);
		dup->u.f.cond = isl_ast_expr_copy(node->u.f.cond);
		dup->u.f.inc = isl_ast_expr_copy(node->u.f.inc);
		dup->u.f.body = isl_ast_node_copy(node->u.f.body);
		if (!dup->u.f.iterator || !dup->u.f.init || !dup->u.f.cond ||
		    !dup->u.f.inc || !dup->u.f.body)
			return isl_ast_node_free(dup);
		break;
	case isl_ast_node_block:
		dup->u.b.children = isl_ast_node_list_copy(node->u.b.children);
		if (!dup->u.b.children)
			return isl_ast_node_free(dup);
		break;
	case isl_ast_node_mark:
		dup->u.m.mark = isl_id_copy(node->u.m.mark);
		dup->u.m.node = isl_ast_node_copy(node->u.m.node);
		if (!dup->u.m.mark || !dup->u.m.node)
			return isl_ast_node_free(dup);
		break;
	case isl_ast_node_user:
		dup->u.e.expr = isl_ast_expr_copy(node->u.e.expr);
		if (!dup->u.e.expr)
			return isl_ast_node_free(dup);
		break;
	case isl_ast_node_error:
		break;
	}

	return dup;
}

__isl_give isl_ast_node *isl_ast_node_cow(__isl_take isl_ast_node *node)
{
	if (!node)
		return NULL;

	if (node->ref == 1)
		return node;
	node->ref--;
	return isl_ast_node_dup(node);
}

__isl_null isl_ast_node *isl_ast_node_free(__isl_take isl_ast_node *node)
{
	if (!node)
		return NULL;

	if (--node->ref > 0)
		return NULL;

	switch (node->type) {
	case isl_ast_node_if:
		isl_ast_expr_free(node->u.i.guard);
		isl_ast_node_free(node->u.i.then);
		isl_ast_node_free(node->u.i.else_node);
		break;
	case isl_ast_node_for:
		isl_ast_expr_free(node->u.f.iterator);
		isl_ast_expr_free(node->u.f.init);
		isl_ast_expr_free(node->u.f.cond);
		isl_ast_expr_free(node->u.f.inc);
		isl_ast_node_free(node->u.f.body);
		break;
	case isl_ast_node_block:
		isl_ast_node_list_free(node->u.b.children);
		break;
	case isl_ast_node_mark:
		isl_id_free(node->u.m.mark);
		isl_ast_node_free(node->u.m.node);
		break;
	case isl_ast_node_user:
		isl_ast_expr_free(node->u.e.expr);
		break;
	case isl_ast_node_error:
		break;
	}

	isl_id_free(node->annotation);
	isl_ctx_deref(node->ctx);
	free(node);

	return NULL;
}

/* Replace the body of the for node "node" by "body".
 */
__isl_give isl_ast_node *isl_ast_node_for_set_body(
	__isl_take isl_ast_node *node, __isl_take isl_ast_node *body)
{
	node = isl_ast_node_cow(node);
	if (!node || !body)
		goto error;
	if (node->type != isl_ast_node_for)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a for node", goto error);

	isl_ast_node_free(node->u.f.body);
	node->u.f.body = body;

	return node;
error:
	isl_ast_node_free(node);
	isl_ast_node_free(body);
	return NULL;
}

__isl_give isl_ast_node *isl_ast_node_for_get_body(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_for)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a for node", return NULL);
	return isl_ast_node_copy(node->u.f.body);
}

/* Mark the given for node as being degenerate.
 */
__isl_give isl_ast_node *isl_ast_node_for_mark_degenerate(
	__isl_take isl_ast_node *node)
{
	node = isl_ast_node_cow(node);
	if (!node)
		return NULL;
	node->u.f.degenerate = 1;
	return node;
}

isl_bool isl_ast_node_for_is_degenerate(__isl_keep isl_ast_node *node)
{
	if (!node)
		return isl_bool_error;
	if (node->type != isl_ast_node_for)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a for node", return isl_bool_error);
	return node->u.f.degenerate;
}

__isl_give isl_ast_expr *isl_ast_node_for_get_iterator(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_for)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a for node", return NULL);
	return isl_ast_expr_copy(node->u.f.iterator);
}

__isl_give isl_ast_expr *isl_ast_node_for_get_init(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_for)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a for node", return NULL);
	return isl_ast_expr_copy(node->u.f.init);
}

/* Return the condition expression of the given for node.
 *
 * If the for node is degenerate, then the condition is not explicitly
 * stored in the node.  Instead, it is constructed as
 *
 *	iterator <= init
 */
__isl_give isl_ast_expr *isl_ast_node_for_get_cond(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_for)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a for node", return NULL);
	if (!node->u.f.degenerate)
		return isl_ast_expr_copy(node->u.f.cond);

	return isl_ast_expr_alloc_binary(isl_ast_op_le,
				isl_ast_expr_copy(node->u.f.iterator),
				isl_ast_expr_copy(node->u.f.init));
}

/* Return the increment of the given for node.
 *
 * If the for node is degenerate, then the increment is not explicitly
 * stored in the node.  We simply return "1".
 */
__isl_give isl_ast_expr *isl_ast_node_for_get_inc(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_for)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a for node", return NULL);
	if (!node->u.f.degenerate)
		return isl_ast_expr_copy(node->u.f.inc);
	return isl_ast_expr_alloc_int_si(isl_ast_node_get_ctx(node), 1);
}

/* Replace the then branch of the if node "node" by "child".
 */
__isl_give isl_ast_node *isl_ast_node_if_set_then(
	__isl_take isl_ast_node *node, __isl_take isl_ast_node *child)
{
	node = isl_ast_node_cow(node);
	if (!node || !child)
		goto error;
	if (node->type != isl_ast_node_if)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not an if node", goto error);

	isl_ast_node_free(node->u.i.then);
	node->u.i.then = child;

	return node;
error:
	isl_ast_node_free(node);
	isl_ast_node_free(child);
	return NULL;
}

__isl_give isl_ast_node *isl_ast_node_if_get_then(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_if)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not an if node", return NULL);
	return isl_ast_node_copy(node->u.i.then);
}

isl_bool isl_ast_node_if_has_else(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return isl_bool_error;
	if (node->type != isl_ast_node_if)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not an if node", return isl_bool_error);
	return node->u.i.else_node != NULL;
}

__isl_give isl_ast_node *isl_ast_node_if_get_else(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_if)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not an if node", return NULL);
	return isl_ast_node_copy(node->u.i.else_node);
}

__isl_give isl_ast_expr *isl_ast_node_if_get_cond(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_if)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a guard node", return NULL);
	return isl_ast_expr_copy(node->u.i.guard);
}

__isl_give isl_ast_node_list *isl_ast_node_block_get_children(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_block)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a block node", return NULL);
	return isl_ast_node_list_copy(node->u.b.children);
}

__isl_give isl_ast_expr *isl_ast_node_user_get_expr(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_user)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a user node", return NULL);

	return isl_ast_expr_copy(node->u.e.expr);
}

/* Return the mark identifier of the mark node "node".
 */
__isl_give isl_id *isl_ast_node_mark_get_id(__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_mark)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a mark node", return NULL);

	return isl_id_copy(node->u.m.mark);
}

/* Return the node marked by mark node "node".
 */
__isl_give isl_ast_node *isl_ast_node_mark_get_node(
	__isl_keep isl_ast_node *node)
{
	if (!node)
		return NULL;
	if (node->type != isl_ast_node_mark)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a mark node", return NULL);

	return isl_ast_node_copy(node->u.m.node);
}

__isl_give isl_id *isl_ast_node_get_annotation(__isl_keep isl_ast_node *node)
{
	return node ? isl_id_copy(node->annotation) : NULL;
}

/* Replace node->annotation by "annotation".
 */
__isl_give isl_ast_node *isl_ast_node_set_annotation(
	__isl_take isl_ast_node *node, __isl_take isl_id *annotation)
{
	node = isl_ast_node_cow(node);
	if (!node || !annotation)
		goto error;

	isl_id_free(node->annotation);
	node->annotation = annotation;

	return node;
error:
	isl_id_free(annotation);
	return isl_ast_node_free(node);
}

/* Traverse the elements of "list" and all their descendants
 * in depth first preorder.
 *
 * Return isl_stat_ok on success and isl_stat_error on failure.
 */
static isl_stat nodelist_foreach(__isl_keep isl_ast_node_list *list,
	isl_bool (*fn)(__isl_keep isl_ast_node *node, void *user), void *user)
{
	int i;

	if (!list)
		return isl_stat_error;

	for (i = 0; i < list->n; ++i) {
		isl_stat ok;
		isl_ast_node *node = list->p[i];

		ok = isl_ast_node_foreach_descendant_top_down(node, fn, user);
		if (ok < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Traverse the descendants of "node" (including the node itself)
 * in depth first preorder.
 *
 * If "fn" returns isl_bool_error on any of the nodes, then the traversal
 * is aborted.
 * If "fn" returns isl_bool_false on any of the nodes, then the subtree rooted
 * at that node is skipped.
 *
 * Return isl_stat_ok on success and isl_stat_error on failure.
 */
isl_stat isl_ast_node_foreach_descendant_top_down(
	__isl_keep isl_ast_node *node,
	isl_bool (*fn)(__isl_keep isl_ast_node *node, void *user), void *user)
{
	isl_bool more;
	isl_stat ok;

	if (!node)
		return isl_stat_error;

	more = fn(node, user);
	if (more < 0)
		return isl_stat_error;
	if (!more)
		return isl_stat_ok;

	switch (node->type) {
	case isl_ast_node_for:
		node = node->u.f.body;
		return isl_ast_node_foreach_descendant_top_down(node, fn, user);
	case isl_ast_node_if:
		ok = isl_ast_node_foreach_descendant_top_down(node->u.i.then,
								fn, user);
		if (ok < 0)
			return isl_stat_error;
		if (!node->u.i.else_node)
			return isl_stat_ok;
		node = node->u.i.else_node;
		return isl_ast_node_foreach_descendant_top_down(node, fn, user);
	case isl_ast_node_block:
		return nodelist_foreach(node->u.b.children, fn, user);
	case isl_ast_node_mark:
		node = node->u.m.node;
		return isl_ast_node_foreach_descendant_top_down(node, fn, user);
	case isl_ast_node_user:
		break;
	case isl_ast_node_error:
		return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Textual C representation of the various operators.
 */
static char *op_str_c[] = {
	[isl_ast_op_and] = "&&",
	[isl_ast_op_and_then] = "&&",
	[isl_ast_op_or] = "||",
	[isl_ast_op_or_else] = "||",
	[isl_ast_op_max] = "max",
	[isl_ast_op_min] = "min",
	[isl_ast_op_minus] = "-",
	[isl_ast_op_add] = "+",
	[isl_ast_op_sub] = "-",
	[isl_ast_op_mul] = "*",
	[isl_ast_op_fdiv_q] = "floord",
	[isl_ast_op_pdiv_q] = "/",
	[isl_ast_op_pdiv_r] = "%",
	[isl_ast_op_zdiv_r] = "%",
	[isl_ast_op_div] = "/",
	[isl_ast_op_eq] = "==",
	[isl_ast_op_le] = "<=",
	[isl_ast_op_ge] = ">=",
	[isl_ast_op_lt] = "<",
	[isl_ast_op_gt] = ">",
	[isl_ast_op_member] = ".",
	[isl_ast_op_address_of] = "&"
};

/* Precedence in C of the various operators.
 * Based on http://en.wikipedia.org/wiki/Operators_in_C_and_C++
 * Lowest value means highest precedence.
 */
static int op_prec[] = {
	[isl_ast_op_and] = 13,
	[isl_ast_op_and_then] = 13,
	[isl_ast_op_or] = 14,
	[isl_ast_op_or_else] = 14,
	[isl_ast_op_max] = 2,
	[isl_ast_op_min] = 2,
	[isl_ast_op_minus] = 3,
	[isl_ast_op_add] = 6,
	[isl_ast_op_sub] = 6,
	[isl_ast_op_mul] = 5,
	[isl_ast_op_div] = 5,
	[isl_ast_op_fdiv_q] = 2,
	[isl_ast_op_pdiv_q] = 5,
	[isl_ast_op_pdiv_r] = 5,
	[isl_ast_op_zdiv_r] = 5,
	[isl_ast_op_cond] = 15,
	[isl_ast_op_select] = 15,
	[isl_ast_op_eq] = 9,
	[isl_ast_op_le] = 8,
	[isl_ast_op_ge] = 8,
	[isl_ast_op_lt] = 8,
	[isl_ast_op_gt] = 8,
	[isl_ast_op_call] = 2,
	[isl_ast_op_access] = 2,
	[isl_ast_op_member] = 2,
	[isl_ast_op_address_of] = 3
};

/* Is the operator left-to-right associative?
 */
static int op_left[] = {
	[isl_ast_op_and] = 1,
	[isl_ast_op_and_then] = 1,
	[isl_ast_op_or] = 1,
	[isl_ast_op_or_else] = 1,
	[isl_ast_op_max] = 1,
	[isl_ast_op_min] = 1,
	[isl_ast_op_minus] = 0,
	[isl_ast_op_add] = 1,
	[isl_ast_op_sub] = 1,
	[isl_ast_op_mul] = 1,
	[isl_ast_op_div] = 1,
	[isl_ast_op_fdiv_q] = 1,
	[isl_ast_op_pdiv_q] = 1,
	[isl_ast_op_pdiv_r] = 1,
	[isl_ast_op_zdiv_r] = 1,
	[isl_ast_op_cond] = 0,
	[isl_ast_op_select] = 0,
	[isl_ast_op_eq] = 1,
	[isl_ast_op_le] = 1,
	[isl_ast_op_ge] = 1,
	[isl_ast_op_lt] = 1,
	[isl_ast_op_gt] = 1,
	[isl_ast_op_call] = 1,
	[isl_ast_op_access] = 1,
	[isl_ast_op_member] = 1,
	[isl_ast_op_address_of] = 0
};

static int is_and(enum isl_ast_op_type op)
{
	return op == isl_ast_op_and || op == isl_ast_op_and_then;
}

static int is_or(enum isl_ast_op_type op)
{
	return op == isl_ast_op_or || op == isl_ast_op_or_else;
}

static int is_add_sub(enum isl_ast_op_type op)
{
	return op == isl_ast_op_add || op == isl_ast_op_sub;
}

static int is_div_mod(enum isl_ast_op_type op)
{
	return op == isl_ast_op_div ||
	       op == isl_ast_op_pdiv_r ||
	       op == isl_ast_op_zdiv_r;
}

static __isl_give isl_printer *print_ast_expr_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_expr *expr);

/* Do we need/want parentheses around "expr" as a subexpression of
 * an "op" operation?  If "left" is set, then "expr" is the left-most
 * operand.
 *
 * We only need parentheses if "expr" represents an operation.
 *
 * If op has a higher precedence than expr->u.op.op, then we need
 * parentheses.
 * If op and expr->u.op.op have the same precedence, but the operations
 * are performed in an order that is different from the associativity,
 * then we need parentheses.
 *
 * An and inside an or technically does not require parentheses,
 * but some compilers complain about that, so we add them anyway.
 *
 * Computations such as "a / b * c" and "a % b + c" can be somewhat
 * difficult to read, so we add parentheses for those as well.
 */
static int sub_expr_need_parens(enum isl_ast_op_type op,
	__isl_keep isl_ast_expr *expr, int left)
{
	if (expr->type != isl_ast_expr_op)
		return 0;

	if (op_prec[expr->u.op.op] > op_prec[op])
		return 1;
	if (op_prec[expr->u.op.op] == op_prec[op] && left != op_left[op])
		return 1;

	if (is_or(op) && is_and(expr->u.op.op))
		return 1;
	if (op == isl_ast_op_mul && expr->u.op.op != isl_ast_op_mul &&
	    op_prec[expr->u.op.op] == op_prec[op])
		return 1;
	if (is_add_sub(op) && is_div_mod(expr->u.op.op))
		return 1;

	return 0;
}

/* Print "expr" as a subexpression of an "op" operation in C format.
 * If "left" is set, then "expr" is the left-most operand.
 */
static __isl_give isl_printer *print_sub_expr_c(__isl_take isl_printer *p,
	enum isl_ast_op_type op, __isl_keep isl_ast_expr *expr, int left)
{
	int need_parens;

	need_parens = sub_expr_need_parens(op, expr, left);

	if (need_parens)
		p = isl_printer_print_str(p, "(");
	p = print_ast_expr_c(p, expr);
	if (need_parens)
		p = isl_printer_print_str(p, ")");
	return p;
}

#define isl_ast_op_last	isl_ast_op_address_of

/* Data structure that holds the user-specified textual
 * representations for the operators in C format.
 * The entries are either NULL or copies of strings.
 * A NULL entry means that the default name should be used.
 */
struct isl_ast_op_names {
	char *op_str[isl_ast_op_last + 1];
};

/* Create an empty struct isl_ast_op_names.
 */
static void *create_names(isl_ctx *ctx)
{
	return isl_calloc_type(ctx, struct isl_ast_op_names);
}

/* Free a struct isl_ast_op_names along with all memory
 * owned by the struct.
 */
static void free_names(void *user)
{
	int i;
	struct isl_ast_op_names *names = user;

	if (!user)
		return;

	for (i = 0; i <= isl_ast_op_last; ++i)
		free(names->op_str[i]);
	free(user);
}

/* Create an identifier that is used to store
 * an isl_ast_op_names note.
 */
static __isl_give isl_id *names_id(isl_ctx *ctx)
{
	return isl_id_alloc(ctx, "isl_ast_op_type_names", NULL);
}

/* Ensure that "p" has a note identified by "id".
 * If there is no such note yet, then it is created by "note_create" and
 * scheduled do be freed by "note_free".
 */
static __isl_give isl_printer *alloc_note(__isl_take isl_printer *p,
	__isl_keep isl_id *id, void *(*note_create)(isl_ctx *),
	void (*note_free)(void *))
{
	isl_ctx *ctx;
	isl_id *note_id;
	isl_bool has_note;
	void *note;

	has_note = isl_printer_has_note(p, id);
	if (has_note < 0)
		return isl_printer_free(p);
	if (has_note)
		return p;

	ctx = isl_printer_get_ctx(p);
	note = note_create(ctx);
	if (!note)
		return isl_printer_free(p);
	note_id = isl_id_alloc(ctx, NULL, note);
	if (!note_id)
		note_free(note);
	else
		note_id = isl_id_set_free_user(note_id, note_free);

	p = isl_printer_set_note(p, isl_id_copy(id), note_id);

	return p;
}

/* Ensure that "p" has an isl_ast_op_names note identified by "id".
 */
static __isl_give isl_printer *alloc_names(__isl_take isl_printer *p,
	__isl_keep isl_id *id)
{
	return alloc_note(p, id, &create_names, &free_names);
}

/* Retrieve the note identified by "id" from "p".
 * The note is assumed to exist.
 */
static void *get_note(__isl_keep isl_printer *p, __isl_keep isl_id *id)
{
	void *note;

	id = isl_printer_get_note(p, isl_id_copy(id));
	note = isl_id_get_user(id);
	isl_id_free(id);

	return note;
}

/* Use "name" to print operations of type "type" to "p".
 *
 * Store the name in an isl_ast_op_names note attached to "p", such that
 * it can be retrieved by get_op_str.
 */
__isl_give isl_printer *isl_ast_op_type_set_print_name(
	__isl_take isl_printer *p, enum isl_ast_op_type type,
	__isl_keep const char *name)
{
	isl_id *id;
	struct isl_ast_op_names *names;

	if (!p)
		return NULL;
	if (type > isl_ast_op_last)
		isl_die(isl_printer_get_ctx(p), isl_error_invalid,
			"invalid type", return isl_printer_free(p));

	id = names_id(isl_printer_get_ctx(p));
	p = alloc_names(p, id);
	names = get_note(p, id);
	isl_id_free(id);
	if (!names)
		return isl_printer_free(p);
	free(names->op_str[type]);
	names->op_str[type] = strdup(name);

	return p;
}

/* Return the textual representation of "type" in C format.
 *
 * If there is a user-specified name in an isl_ast_op_names note
 * associated to "p", then return that.
 * Otherwise, return the default name in op_str.
 */
static const char *get_op_str_c(__isl_keep isl_printer *p,
	enum isl_ast_op_type type)
{
	isl_id *id;
	isl_bool has_names;
	struct isl_ast_op_names *names = NULL;

	id = names_id(isl_printer_get_ctx(p));
	has_names = isl_printer_has_note(p, id);
	if (has_names >= 0 && has_names)
		names = get_note(p, id);
	isl_id_free(id);
	if (names && names->op_str[type])
		return names->op_str[type];
	return op_str_c[type];
}

/* Print a min or max reduction "expr" in C format.
 */
static __isl_give isl_printer *print_min_max_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_expr *expr)
{
	int i = 0;

	for (i = 1; i < expr->u.op.n_arg; ++i) {
		p = isl_printer_print_str(p, get_op_str_c(p, expr->u.op.op));
		p = isl_printer_print_str(p, "(");
	}
	p = isl_printer_print_ast_expr(p, expr->u.op.args[0]);
	for (i = 1; i < expr->u.op.n_arg; ++i) {
		p = isl_printer_print_str(p, ", ");
		p = print_ast_expr_c(p, expr->u.op.args[i]);
		p = isl_printer_print_str(p, ")");
	}

	return p;
}

/* Print a function call "expr" in C format.
 *
 * The first argument represents the function to be called.
 */
static __isl_give isl_printer *print_call_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_expr *expr)
{
	int i = 0;

	p = print_ast_expr_c(p, expr->u.op.args[0]);
	p = isl_printer_print_str(p, "(");
	for (i = 1; i < expr->u.op.n_arg; ++i) {
		if (i != 1)
			p = isl_printer_print_str(p, ", ");
		p = print_ast_expr_c(p, expr->u.op.args[i]);
	}
	p = isl_printer_print_str(p, ")");

	return p;
}

/* Print an array access "expr" in C format.
 *
 * The first argument represents the array being accessed.
 */
static __isl_give isl_printer *print_access_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_expr *expr)
{
	int i = 0;

	p = print_ast_expr_c(p, expr->u.op.args[0]);
	for (i = 1; i < expr->u.op.n_arg; ++i) {
		p = isl_printer_print_str(p, "[");
		p = print_ast_expr_c(p, expr->u.op.args[i]);
		p = isl_printer_print_str(p, "]");
	}

	return p;
}

/* Print "expr" to "p" in C format.
 */
static __isl_give isl_printer *print_ast_expr_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_expr *expr)
{
	if (!p)
		return NULL;
	if (!expr)
		return isl_printer_free(p);

	switch (expr->type) {
	case isl_ast_expr_op:
		if (expr->u.op.op == isl_ast_op_call) {
			p = print_call_c(p, expr);
			break;
		}
		if (expr->u.op.op == isl_ast_op_access) {
			p = print_access_c(p, expr);
			break;
		}
		if (expr->u.op.n_arg == 1) {
			p = isl_printer_print_str(p,
						get_op_str_c(p, expr->u.op.op));
			p = print_sub_expr_c(p, expr->u.op.op,
						expr->u.op.args[0], 0);
			break;
		}
		if (expr->u.op.op == isl_ast_op_fdiv_q) {
			const char *name = get_op_str_c(p, isl_ast_op_fdiv_q);
			p = isl_printer_print_str(p, name);
			p = isl_printer_print_str(p, "(");
			p = print_ast_expr_c(p, expr->u.op.args[0]);
			p = isl_printer_print_str(p, ", ");
			p = print_ast_expr_c(p, expr->u.op.args[1]);
			p = isl_printer_print_str(p, ")");
			break;
		}
		if (expr->u.op.op == isl_ast_op_max ||
		    expr->u.op.op == isl_ast_op_min) {
			p = print_min_max_c(p, expr);
			break;
		}
		if (expr->u.op.op == isl_ast_op_cond ||
		    expr->u.op.op == isl_ast_op_select) {
			p = print_ast_expr_c(p, expr->u.op.args[0]);
			p = isl_printer_print_str(p, " ? ");
			p = print_ast_expr_c(p, expr->u.op.args[1]);
			p = isl_printer_print_str(p, " : ");
			p = print_ast_expr_c(p, expr->u.op.args[2]);
			break;
		}
		if (expr->u.op.n_arg != 2)
			isl_die(isl_printer_get_ctx(p), isl_error_internal,
				"operation should have two arguments",
				return isl_printer_free(p));
		p = print_sub_expr_c(p, expr->u.op.op, expr->u.op.args[0], 1);
		if (expr->u.op.op != isl_ast_op_member)
			p = isl_printer_print_str(p, " ");
		p = isl_printer_print_str(p, get_op_str_c(p, expr->u.op.op));
		if (expr->u.op.op != isl_ast_op_member)
			p = isl_printer_print_str(p, " ");
		p = print_sub_expr_c(p, expr->u.op.op, expr->u.op.args[1], 0);
		break;
	case isl_ast_expr_id:
		p = isl_printer_print_str(p, isl_id_get_name(expr->u.id));
		break;
	case isl_ast_expr_int:
		p = isl_printer_print_val(p, expr->u.v);
		break;
	case isl_ast_expr_error:
		break;
	}

	return p;
}

/* Textual representation of the isl_ast_op_type elements
 * for use in a YAML representation of an isl_ast_expr.
 */
static char *op_str[] = {
	[isl_ast_op_and] = "and",
	[isl_ast_op_and_then] = "and_then",
	[isl_ast_op_or] = "or",
	[isl_ast_op_or_else] = "or_else",
	[isl_ast_op_max] = "max",
	[isl_ast_op_min] = "min",
	[isl_ast_op_minus] = "minus",
	[isl_ast_op_add] = "add",
	[isl_ast_op_sub] = "sub",
	[isl_ast_op_mul] = "mul",
	[isl_ast_op_div] = "div",
	[isl_ast_op_fdiv_q] = "fdiv_q",
	[isl_ast_op_pdiv_q] = "pdiv_q",
	[isl_ast_op_pdiv_r] = "pdiv_r",
	[isl_ast_op_zdiv_r] = "zdiv_r",
	[isl_ast_op_cond] = "cond",
	[isl_ast_op_select] = "select",
	[isl_ast_op_eq] = "eq",
	[isl_ast_op_le] = "le",
	[isl_ast_op_lt] = "lt",
	[isl_ast_op_ge] = "ge",
	[isl_ast_op_gt] = "gt",
	[isl_ast_op_call] = "call",
	[isl_ast_op_access] = "access",
	[isl_ast_op_member] = "member",
	[isl_ast_op_address_of] = "address_of"
};

static __isl_give isl_printer *print_ast_expr_isl(__isl_take isl_printer *p,
	__isl_keep isl_ast_expr *expr);

/* Print the arguments of "expr" to "p" in isl format.
 *
 * If there are no arguments, then nothing needs to be printed.
 * Otherwise add an "args" key to the current mapping with as value
 * the list of arguments of "expr".
 */
static __isl_give isl_printer *print_arguments(__isl_take isl_printer *p,
	__isl_keep isl_ast_expr *expr)
{
	int i, n;

	n = isl_ast_expr_get_op_n_arg(expr);
	if (n < 0)
		return isl_printer_free(p);
	if (n == 0)
		return p;

	p = isl_printer_print_str(p, "args");
	p = isl_printer_yaml_next(p);
	p = isl_printer_yaml_start_sequence(p);
	for (i = 0; i < n; ++i) {
		isl_ast_expr *arg;

		arg = isl_ast_expr_get_op_arg(expr, i);
		p = print_ast_expr_isl(p, arg);
		isl_ast_expr_free(arg);
		p = isl_printer_yaml_next(p);
	}
	p = isl_printer_yaml_end_sequence(p);

	return p;
}

/* Print "expr" to "p" in isl format.
 *
 * In particular, print the isl_ast_expr as a YAML document.
 */
static __isl_give isl_printer *print_ast_expr_isl(__isl_take isl_printer *p,
	__isl_keep isl_ast_expr *expr)
{
	enum isl_ast_expr_type type;
	enum isl_ast_op_type op;
	isl_id *id;
	isl_val *v;

	if (!expr)
		return isl_printer_free(p);

	p = isl_printer_yaml_start_mapping(p);
	type = isl_ast_expr_get_type(expr);
	switch (type) {
	case isl_ast_expr_error:
		return isl_printer_free(p);
	case isl_ast_expr_op:
		op = isl_ast_expr_get_op_type(expr);
		if (op == isl_ast_op_error)
			return isl_printer_free(p);
		p = isl_printer_print_str(p, "op");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, op_str[op]);
		p = isl_printer_yaml_next(p);
		p = print_arguments(p, expr);
		break;
	case isl_ast_expr_id:
		p = isl_printer_print_str(p, "id");
		p = isl_printer_yaml_next(p);
		id = isl_ast_expr_get_id(expr);
		p = isl_printer_print_id(p, id);
		isl_id_free(id);
		break;
	case isl_ast_expr_int:
		p = isl_printer_print_str(p, "val");
		p = isl_printer_yaml_next(p);
		v = isl_ast_expr_get_val(expr);
		p = isl_printer_print_val(p, v);
		isl_val_free(v);
		break;
	}
	p = isl_printer_yaml_end_mapping(p);

	return p;
}

/* Print "expr" to "p".
 *
 * Only an isl and a C format are supported.
 */
__isl_give isl_printer *isl_printer_print_ast_expr(__isl_take isl_printer *p,
	__isl_keep isl_ast_expr *expr)
{
	int format;

	if (!p)
		return NULL;

	format = isl_printer_get_output_format(p);
	switch (format) {
	case ISL_FORMAT_ISL:
		p = print_ast_expr_isl(p, expr);
		break;
	case ISL_FORMAT_C:
		p = print_ast_expr_c(p, expr);
		break;
	default:
		isl_die(isl_printer_get_ctx(p), isl_error_unsupported,
			"output format not supported for ast_expr",
			return isl_printer_free(p));
	}

	return p;
}

static __isl_give isl_printer *print_ast_node_isl(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node);

/* Print a YAML sequence containing the entries in "list" to "p".
 */
static __isl_give isl_printer *print_ast_node_list(__isl_take isl_printer *p,
	__isl_keep isl_ast_node_list *list)
{
	int i, n;

	n = isl_ast_node_list_n_ast_node(list);
	if (n < 0)
		return isl_printer_free(p);

	p = isl_printer_yaml_start_sequence(p);
	for (i = 0; i < n; ++i) {
		isl_ast_node *node;

		node = isl_ast_node_list_get_ast_node(list, i);
		p = print_ast_node_isl(p, node);
		isl_ast_node_free(node);
		p = isl_printer_yaml_next(p);
	}
	p = isl_printer_yaml_end_sequence(p);

	return p;
}

/* Print "node" to "p" in "isl format".
 *
 * In particular, print the isl_ast_node as a YAML document.
 */
static __isl_give isl_printer *print_ast_node_isl(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node)
{
	switch (node->type) {
	case isl_ast_node_for:
		p = isl_printer_yaml_start_mapping(p);
		p = isl_printer_print_str(p, "iterator");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_ast_expr(p, node->u.f.iterator);
		p = isl_printer_yaml_next(p);
		if (node->u.f.degenerate) {
			p = isl_printer_print_str(p, "value");
			p = isl_printer_yaml_next(p);
			p = isl_printer_print_ast_expr(p, node->u.f.init);
			p = isl_printer_yaml_next(p);
		} else {
			p = isl_printer_print_str(p, "init");
			p = isl_printer_yaml_next(p);
			p = isl_printer_print_ast_expr(p, node->u.f.init);
			p = isl_printer_yaml_next(p);
			p = isl_printer_print_str(p, "cond");
			p = isl_printer_yaml_next(p);
			p = isl_printer_print_ast_expr(p, node->u.f.cond);
			p = isl_printer_yaml_next(p);
			p = isl_printer_print_str(p, "inc");
			p = isl_printer_yaml_next(p);
			p = isl_printer_print_ast_expr(p, node->u.f.inc);
			p = isl_printer_yaml_next(p);
		}
		if (node->u.f.body) {
			p = isl_printer_print_str(p, "body");
			p = isl_printer_yaml_next(p);
			p = isl_printer_print_ast_node(p, node->u.f.body);
			p = isl_printer_yaml_next(p);
		}
		p = isl_printer_yaml_end_mapping(p);
		break;
	case isl_ast_node_mark:
		p = isl_printer_yaml_start_mapping(p);
		p = isl_printer_print_str(p, "mark");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_id(p, node->u.m.mark);
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_str(p, "node");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_ast_node(p, node->u.m.node);
		p = isl_printer_yaml_end_mapping(p);
		break;
	case isl_ast_node_user:
		p = isl_printer_yaml_start_mapping(p);
		p = isl_printer_print_str(p, "user");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_ast_expr(p, node->u.e.expr);
		p = isl_printer_yaml_end_mapping(p);
		break;
	case isl_ast_node_if:
		p = isl_printer_yaml_start_mapping(p);
		p = isl_printer_print_str(p, "guard");
		p = isl_printer_yaml_next(p);
		p = isl_printer_print_ast_expr(p, node->u.i.guard);
		p = isl_printer_yaml_next(p);
		if (node->u.i.then) {
			p = isl_printer_print_str(p, "then");
			p = isl_printer_yaml_next(p);
			p = isl_printer_print_ast_node(p, node->u.i.then);
			p = isl_printer_yaml_next(p);
		}
		if (node->u.i.else_node) {
			p = isl_printer_print_str(p, "else");
			p = isl_printer_yaml_next(p);
			p = isl_printer_print_ast_node(p, node->u.i.else_node);
		}
		p = isl_printer_yaml_end_mapping(p);
		break;
	case isl_ast_node_block:
		p = print_ast_node_list(p, node->u.b.children);
		break;
	case isl_ast_node_error:
		break;
	}
	return p;
}

/* Do we need to print a block around the body "node" of a for or if node?
 *
 * If the node is a block, then we need to print a block.
 * Also if the node is a degenerate for then we will print it as
 * an assignment followed by the body of the for loop, so we need a block
 * as well.
 * If the node is an if node with an else, then we print a block
 * to avoid spurious dangling else warnings emitted by some compilers.
 * If the node is a mark, then in principle, we would have to check
 * the child of the mark node.  However, even if the child would not
 * require us to print a block, for readability it is probably best
 * to print a block anyway.
 * If the ast_always_print_block option has been set, then we print a block.
 */
static int need_block(__isl_keep isl_ast_node *node)
{
	isl_ctx *ctx;

	if (node->type == isl_ast_node_block)
		return 1;
	if (node->type == isl_ast_node_for && node->u.f.degenerate)
		return 1;
	if (node->type == isl_ast_node_if && node->u.i.else_node)
		return 1;
	if (node->type == isl_ast_node_mark)
		return 1;

	ctx = isl_ast_node_get_ctx(node);
	return isl_options_get_ast_always_print_block(ctx);
}

static __isl_give isl_printer *print_ast_node_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node,
	__isl_keep isl_ast_print_options *options, int in_block, int in_list);
static __isl_give isl_printer *print_if_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node,
	__isl_keep isl_ast_print_options *options, int new_line,
	int force_block);

/* Print the body "node" of a for or if node.
 * If "else_node" is set, then it is printed as well.
 * If "force_block" is set, then print out the body as a block.
 *
 * We first check if we need to print out a block.
 * We always print out a block if there is an else node to make
 * sure that the else node is matched to the correct if node.
 * For consistency, the corresponding else node is also printed as a block.
 *
 * If the else node is itself an if, then we print it as
 *
 *	} else if (..) {
 *	}
 *
 * Otherwise the else node is printed as
 *
 *	} else {
 *	  node
 *	}
 */
static __isl_give isl_printer *print_body_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node, __isl_keep isl_ast_node *else_node,
	__isl_keep isl_ast_print_options *options, int force_block)
{
	if (!node)
		return isl_printer_free(p);

	if (!force_block && !else_node && !need_block(node)) {
		p = isl_printer_end_line(p);
		p = isl_printer_indent(p, 2);
		p = isl_ast_node_print(node, p,
					isl_ast_print_options_copy(options));
		p = isl_printer_indent(p, -2);
		return p;
	}

	p = isl_printer_print_str(p, " {");
	p = isl_printer_end_line(p);
	p = isl_printer_indent(p, 2);
	p = print_ast_node_c(p, node, options, 1, 0);
	p = isl_printer_indent(p, -2);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "}");
	if (else_node) {
		if (else_node->type == isl_ast_node_if) {
			p = isl_printer_print_str(p, " else ");
			p = print_if_c(p, else_node, options, 0, 1);
		} else {
			p = isl_printer_print_str(p, " else");
			p = print_body_c(p, else_node, NULL, options, 1);
		}
	} else
		p = isl_printer_end_line(p);

	return p;
}

/* Print the start of a compound statement.
 */
static __isl_give isl_printer *start_block(__isl_take isl_printer *p)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "{");
	p = isl_printer_end_line(p);
	p = isl_printer_indent(p, 2);

	return p;
}

/* Print the end of a compound statement.
 */
static __isl_give isl_printer *end_block(__isl_take isl_printer *p)
{
	p = isl_printer_indent(p, -2);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "}");
	p = isl_printer_end_line(p);

	return p;
}

/* Print the for node "node".
 *
 * If the for node is degenerate, it is printed as
 *
 *	type iterator = init;
 *	body
 *
 * Otherwise, it is printed as
 *
 *	for (type iterator = init; cond; iterator += inc)
 *		body
 *
 * "in_block" is set if we are currently inside a block.
 * "in_list" is set if the current node is not alone in the block.
 * If we are not in a block or if the current not is not alone in the block
 * then we print a block around a degenerate for loop such that the variable
 * declaration will not conflict with any potential other declaration
 * of the same variable.
 */
static __isl_give isl_printer *print_for_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node,
	__isl_keep isl_ast_print_options *options, int in_block, int in_list)
{
	isl_id *id;
	const char *name;
	const char *type;

	type = isl_options_get_ast_iterator_type(isl_printer_get_ctx(p));
	if (!node->u.f.degenerate) {
		id = isl_ast_expr_get_id(node->u.f.iterator);
		name = isl_id_get_name(id);
		isl_id_free(id);
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "for (");
		p = isl_printer_print_str(p, type);
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_str(p, name);
		p = isl_printer_print_str(p, " = ");
		p = isl_printer_print_ast_expr(p, node->u.f.init);
		p = isl_printer_print_str(p, "; ");
		p = isl_printer_print_ast_expr(p, node->u.f.cond);
		p = isl_printer_print_str(p, "; ");
		p = isl_printer_print_str(p, name);
		p = isl_printer_print_str(p, " += ");
		p = isl_printer_print_ast_expr(p, node->u.f.inc);
		p = isl_printer_print_str(p, ")");
		p = print_body_c(p, node->u.f.body, NULL, options, 0);
	} else {
		id = isl_ast_expr_get_id(node->u.f.iterator);
		name = isl_id_get_name(id);
		isl_id_free(id);
		if (!in_block || in_list)
			p = start_block(p);
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, type);
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_str(p, name);
		p = isl_printer_print_str(p, " = ");
		p = isl_printer_print_ast_expr(p, node->u.f.init);
		p = isl_printer_print_str(p, ";");
		p = isl_printer_end_line(p);
		p = print_ast_node_c(p, node->u.f.body, options, 1, 0);
		if (!in_block || in_list)
			p = end_block(p);
	}

	return p;
}

/* Print the if node "node".
 * If "new_line" is set then the if node should be printed on a new line.
 * If "force_block" is set, then print out the body as a block.
 */
static __isl_give isl_printer *print_if_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node,
	__isl_keep isl_ast_print_options *options, int new_line,
	int force_block)
{
	if (new_line)
		p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "if (");
	p = isl_printer_print_ast_expr(p, node->u.i.guard);
	p = isl_printer_print_str(p, ")");
	p = print_body_c(p, node->u.i.then, node->u.i.else_node, options,
			force_block);

	return p;
}

/* Print the "node" to "p".
 *
 * "in_block" is set if we are currently inside a block.
 * If so, we do not print a block around the children of a block node.
 * We do this to avoid an extra block around the body of a degenerate
 * for node.
 *
 * "in_list" is set if the current node is not alone in the block.
 */
static __isl_give isl_printer *print_ast_node_c(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node,
	__isl_keep isl_ast_print_options *options, int in_block, int in_list)
{
	switch (node->type) {
	case isl_ast_node_for:
		if (options->print_for)
			return options->print_for(p,
					isl_ast_print_options_copy(options),
					node, options->print_for_user);
		p = print_for_c(p, node, options, in_block, in_list);
		break;
	case isl_ast_node_if:
		p = print_if_c(p, node, options, 1, 0);
		break;
	case isl_ast_node_block:
		if (!in_block)
			p = start_block(p);
		p = isl_ast_node_list_print(node->u.b.children, p, options);
		if (!in_block)
			p = end_block(p);
		break;
	case isl_ast_node_mark:
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "// ");
		p = isl_printer_print_str(p, isl_id_get_name(node->u.m.mark));
		p = isl_printer_end_line(p);
		p = print_ast_node_c(p, node->u.m.node, options, 0, in_list);
		break;
	case isl_ast_node_user:
		if (options->print_user)
			return options->print_user(p,
					isl_ast_print_options_copy(options),
					node, options->print_user_user);
		p = isl_printer_start_line(p);
		p = isl_printer_print_ast_expr(p, node->u.e.expr);
		p = isl_printer_print_str(p, ";");
		p = isl_printer_end_line(p);
		break;
	case isl_ast_node_error:
		break;
	}
	return p;
}

/* Print the for node "node" to "p".
 */
__isl_give isl_printer *isl_ast_node_for_print(__isl_keep isl_ast_node *node,
	__isl_take isl_printer *p, __isl_take isl_ast_print_options *options)
{
	if (!node || !options)
		goto error;
	if (node->type != isl_ast_node_for)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not a for node", goto error);
	p = print_for_c(p, node, options, 0, 0);
	isl_ast_print_options_free(options);
	return p;
error:
	isl_ast_print_options_free(options);
	isl_printer_free(p);
	return NULL;
}

/* Print the if node "node" to "p".
 */
__isl_give isl_printer *isl_ast_node_if_print(__isl_keep isl_ast_node *node,
	__isl_take isl_printer *p, __isl_take isl_ast_print_options *options)
{
	if (!node || !options)
		goto error;
	if (node->type != isl_ast_node_if)
		isl_die(isl_ast_node_get_ctx(node), isl_error_invalid,
			"not an if node", goto error);
	p = print_if_c(p, node, options, 1, 0);
	isl_ast_print_options_free(options);
	return p;
error:
	isl_ast_print_options_free(options);
	isl_printer_free(p);
	return NULL;
}

/* Print "node" to "p".
 */
__isl_give isl_printer *isl_ast_node_print(__isl_keep isl_ast_node *node,
	__isl_take isl_printer *p, __isl_take isl_ast_print_options *options)
{
	if (!options || !node)
		goto error;
	p = print_ast_node_c(p, node, options, 0, 0);
	isl_ast_print_options_free(options);
	return p;
error:
	isl_ast_print_options_free(options);
	isl_printer_free(p);
	return NULL;
}

/* Print "node" to "p".
 */
__isl_give isl_printer *isl_printer_print_ast_node(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node)
{
	int format;
	isl_ast_print_options *options;

	if (!p)
		return NULL;

	format = isl_printer_get_output_format(p);
	switch (format) {
	case ISL_FORMAT_ISL:
		p = print_ast_node_isl(p, node);
		break;
	case ISL_FORMAT_C:
		options = isl_ast_print_options_alloc(isl_printer_get_ctx(p));
		p = isl_ast_node_print(node, p, options);
		break;
	default:
		isl_die(isl_printer_get_ctx(p), isl_error_unsupported,
			"output format not supported for ast_node",
			return isl_printer_free(p));
	}

	return p;
}

/* Print the list of nodes "list" to "p".
 */
__isl_give isl_printer *isl_ast_node_list_print(
	__isl_keep isl_ast_node_list *list, __isl_take isl_printer *p,
	__isl_keep isl_ast_print_options *options)
{
	int i;

	if (!p || !list || !options)
		return isl_printer_free(p);

	for (i = 0; i < list->n; ++i)
		p = print_ast_node_c(p, list->p[i], options, 1, 1);

	return p;
}

#define ISL_AST_MACRO_FLOORD	(1 << 0)
#define ISL_AST_MACRO_MIN	(1 << 1)
#define ISL_AST_MACRO_MAX	(1 << 2)
#define ISL_AST_MACRO_ALL	(ISL_AST_MACRO_FLOORD | \
				 ISL_AST_MACRO_MIN | \
				 ISL_AST_MACRO_MAX)

/* If "expr" contains an isl_ast_op_min, isl_ast_op_max or isl_ast_op_fdiv_q
 * then set the corresponding bit in "macros".
 */
static int ast_expr_required_macros(__isl_keep isl_ast_expr *expr, int macros)
{
	int i;

	if (macros == ISL_AST_MACRO_ALL)
		return macros;

	if (expr->type != isl_ast_expr_op)
		return macros;

	if (expr->u.op.op == isl_ast_op_min)
		macros |= ISL_AST_MACRO_MIN;
	if (expr->u.op.op == isl_ast_op_max)
		macros |= ISL_AST_MACRO_MAX;
	if (expr->u.op.op == isl_ast_op_fdiv_q)
		macros |= ISL_AST_MACRO_FLOORD;

	for (i = 0; i < expr->u.op.n_arg; ++i)
		macros = ast_expr_required_macros(expr->u.op.args[i], macros);

	return macros;
}

static int ast_node_list_required_macros(__isl_keep isl_ast_node_list *list,
	int macros);

/* If "node" contains an isl_ast_op_min, isl_ast_op_max or isl_ast_op_fdiv_q
 * then set the corresponding bit in "macros".
 */
static int ast_node_required_macros(__isl_keep isl_ast_node *node, int macros)
{
	if (macros == ISL_AST_MACRO_ALL)
		return macros;

	switch (node->type) {
	case isl_ast_node_for:
		macros = ast_expr_required_macros(node->u.f.init, macros);
		if (!node->u.f.degenerate) {
			macros = ast_expr_required_macros(node->u.f.cond,
								macros);
			macros = ast_expr_required_macros(node->u.f.inc,
								macros);
		}
		macros = ast_node_required_macros(node->u.f.body, macros);
		break;
	case isl_ast_node_if:
		macros = ast_expr_required_macros(node->u.i.guard, macros);
		macros = ast_node_required_macros(node->u.i.then, macros);
		if (node->u.i.else_node)
			macros = ast_node_required_macros(node->u.i.else_node,
								macros);
		break;
	case isl_ast_node_block:
		macros = ast_node_list_required_macros(node->u.b.children,
							macros);
		break;
	case isl_ast_node_mark:
		macros = ast_node_required_macros(node->u.m.node, macros);
		break;
	case isl_ast_node_user:
		macros = ast_expr_required_macros(node->u.e.expr, macros);
		break;
	case isl_ast_node_error:
		break;
	}

	return macros;
}

/* If "list" contains an isl_ast_op_min, isl_ast_op_max or isl_ast_op_fdiv_q
 * then set the corresponding bit in "macros".
 */
static int ast_node_list_required_macros(__isl_keep isl_ast_node_list *list,
	int macros)
{
	int i;

	for (i = 0; i < list->n; ++i)
		macros = ast_node_required_macros(list->p[i], macros);

	return macros;
}

/* Data structure for keeping track of whether a macro definition
 * for a given type has already been printed.
 * The value is zero if no definition has been printed and non-zero otherwise.
 */
struct isl_ast_op_printed {
	char printed[isl_ast_op_last + 1];
};

/* Create an empty struct isl_ast_op_printed.
 */
static void *create_printed(isl_ctx *ctx)
{
	return isl_calloc_type(ctx, struct isl_ast_op_printed);
}

/* Free a struct isl_ast_op_printed.
 */
static void free_printed(void *user)
{
	free(user);
}

/* Ensure that "p" has an isl_ast_op_printed note identified by "id".
 */
static __isl_give isl_printer *alloc_printed(__isl_take isl_printer *p,
	__isl_keep isl_id *id)
{
	return alloc_note(p, id, &create_printed, &free_printed);
}

/* Create an identifier that is used to store
 * an isl_ast_op_printed note.
 */
static __isl_give isl_id *printed_id(isl_ctx *ctx)
{
	return isl_id_alloc(ctx, "isl_ast_op_type_printed", NULL);
}

/* Did the user specify that a macro definition should only be
 * printed once and has a macro definition for "type" already
 * been printed to "p"?
 * If definitions should only be printed once, but a definition
 * for "p" has not yet been printed, then mark it as having been
 * printed so that it will not printed again.
 * The actual printing is taken care of by the caller.
 */
static isl_bool already_printed_once(__isl_keep isl_printer *p,
	enum isl_ast_op_type type)
{
	isl_ctx *ctx;
	isl_id *id;
	struct isl_ast_op_printed *printed;

	if (!p)
		return isl_bool_error;

	ctx = isl_printer_get_ctx(p);
	if (!isl_options_get_ast_print_macro_once(ctx))
		return isl_bool_false;

	if (type > isl_ast_op_last)
		isl_die(isl_printer_get_ctx(p), isl_error_invalid,
			"invalid type", return isl_bool_error);

	id = printed_id(isl_printer_get_ctx(p));
	p = alloc_printed(p, id);
	printed = get_note(p, id);
	isl_id_free(id);
	if (!printed)
		return isl_bool_error;

	if (printed->printed[type])
		return isl_bool_true;

	printed->printed[type] = 1;
	return isl_bool_false;
}

/* Print a macro definition for the operator "type".
 *
 * If the user has specified that a macro definition should
 * only be printed once to any given printer and if the macro definition
 * has already been printed to "p", then do not print the definition.
 */
__isl_give isl_printer *isl_ast_op_type_print_macro(
	enum isl_ast_op_type type, __isl_take isl_printer *p)
{
	isl_bool skip;

	skip = already_printed_once(p, type);
	if (skip < 0)
		return isl_printer_free(p);
	if (skip)
		return p;

	switch (type) {
	case isl_ast_op_min:
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "#define ");
		p = isl_printer_print_str(p, get_op_str_c(p, type));
		p = isl_printer_print_str(p,
			"(x,y)    ((x) < (y) ? (x) : (y))");
		p = isl_printer_end_line(p);
		break;
	case isl_ast_op_max:
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "#define ");
		p = isl_printer_print_str(p, get_op_str_c(p, type));
		p = isl_printer_print_str(p,
			"(x,y)    ((x) > (y) ? (x) : (y))");
		p = isl_printer_end_line(p);
		break;
	case isl_ast_op_fdiv_q:
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "#define ");
		p = isl_printer_print_str(p, get_op_str_c(p, type));
		p = isl_printer_print_str(p,
			"(n,d) "
			"(((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))");
		p = isl_printer_end_line(p);
		break;
	default:
		break;
	}

	return p;
}

/* Call "fn" for each type of operation represented in the "macros"
 * bit vector.
 */
static isl_stat foreach_ast_op_type(int macros,
	isl_stat (*fn)(enum isl_ast_op_type type, void *user), void *user)
{
	if (macros & ISL_AST_MACRO_MIN && fn(isl_ast_op_min, user) < 0)
		return isl_stat_error;
	if (macros & ISL_AST_MACRO_MAX && fn(isl_ast_op_max, user) < 0)
		return isl_stat_error;
	if (macros & ISL_AST_MACRO_FLOORD && fn(isl_ast_op_fdiv_q, user) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Call "fn" for each type of operation that appears in "expr"
 * and that requires a macro definition.
 */
isl_stat isl_ast_expr_foreach_ast_op_type(__isl_keep isl_ast_expr *expr,
	isl_stat (*fn)(enum isl_ast_op_type type, void *user), void *user)
{
	int macros;

	if (!expr)
		return isl_stat_error;

	macros = ast_expr_required_macros(expr, 0);
	return foreach_ast_op_type(macros, fn, user);
}

/* Call "fn" for each type of operation that appears in "node"
 * and that requires a macro definition.
 */
isl_stat isl_ast_node_foreach_ast_op_type(__isl_keep isl_ast_node *node,
	isl_stat (*fn)(enum isl_ast_op_type type, void *user), void *user)
{
	int macros;

	if (!node)
		return isl_stat_error;

	macros = ast_node_required_macros(node, 0);
	return foreach_ast_op_type(macros, fn, user);
}

static isl_stat ast_op_type_print_macro(enum isl_ast_op_type type, void *user)
{
	isl_printer **p = user;

	*p = isl_ast_op_type_print_macro(type, *p);

	return isl_stat_ok;
}

/* Print macro definitions for all the macros used in the result
 * of printing "expr".
 */
__isl_give isl_printer *isl_ast_expr_print_macros(
	__isl_keep isl_ast_expr *expr, __isl_take isl_printer *p)
{
	if (isl_ast_expr_foreach_ast_op_type(expr,
					    &ast_op_type_print_macro, &p) < 0)
		return isl_printer_free(p);
	return p;
}

/* Print macro definitions for all the macros used in the result
 * of printing "node".
 */
__isl_give isl_printer *isl_ast_node_print_macros(
	__isl_keep isl_ast_node *node, __isl_take isl_printer *p)
{
	if (isl_ast_node_foreach_ast_op_type(node,
					    &ast_op_type_print_macro, &p) < 0)
		return isl_printer_free(p);
	return p;
}

/* Return a string containing C code representing this isl_ast_expr.
 */
__isl_give char *isl_ast_expr_to_C_str(__isl_keep isl_ast_expr *expr)
{
	isl_printer *p;
	char *str;

	if (!expr)
		return NULL;

	p = isl_printer_to_str(isl_ast_expr_get_ctx(expr));
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = isl_printer_print_ast_expr(p, expr);

	str = isl_printer_get_str(p);

	isl_printer_free(p);

	return str;
}

/* Return a string containing C code representing this isl_ast_node.
 */
__isl_give char *isl_ast_node_to_C_str(__isl_keep isl_ast_node *node)
{
	isl_printer *p;
	char *str;

	if (!node)
		return NULL;

	p = isl_printer_to_str(isl_ast_node_get_ctx(node));
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = isl_printer_print_ast_node(p, node);

	str = isl_printer_get_str(p);

	isl_printer_free(p);

	return str;
}
