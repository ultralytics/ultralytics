#include <isl/deprecated/ast_int.h>
#include <isl/deprecated/val_int.h>
#include <isl_ast_private.h>

int isl_ast_expr_get_int(__isl_keep isl_ast_expr *expr, isl_int *v)
{
	if (!expr)
		return -1;
	if (expr->type != isl_ast_expr_int)
		isl_die(isl_ast_expr_get_ctx(expr), isl_error_invalid,
			"expression not an int", return -1);
	return isl_val_get_num_isl_int(expr->u.v, v);
}
