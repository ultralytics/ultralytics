#include <isl/id_to_ast_expr.h>
#include <isl/ast.h>

#define isl_id_is_equal(id1,id2)	id1 == id2

#define ISL_KEY		isl_id
#define ISL_VAL		isl_ast_expr
#define ISL_HMAP_SUFFIX	id_to_ast_expr
#define ISL_HMAP	isl_id_to_ast_expr
#define ISL_KEY_IS_EQUAL	isl_id_is_equal
#define ISL_VAL_IS_EQUAL	isl_ast_expr_is_equal
#define ISL_KEY_PRINT		isl_printer_print_id
#define ISL_VAL_PRINT		isl_printer_print_ast_expr

#include <isl/hmap_templ.c>
