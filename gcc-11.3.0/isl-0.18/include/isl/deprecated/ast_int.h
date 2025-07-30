#ifndef ISL_DEPRECATED_AST_INT_H
#define ISL_DEPRECATED_AST_INT_H

#include <isl/deprecated/int.h>
#include <isl/ast.h>

#if defined(__cplusplus)
extern "C" {
#endif

int isl_ast_expr_get_int(__isl_keep isl_ast_expr *expr, isl_int *v);

#if defined(__cplusplus)
}
#endif

#endif
