#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/space.h>
#include <isl/local_space.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/polynomial.h>
#include <isl/band.h>
#include <isl/constraint.h>
#include <isl/aff.h>
#include <isl/ast.h>
#include <isl/printer.h>
#include <isl/val.h>

#undef BASE
#define BASE id
#include <print_templ.c>
#undef BASE
#define BASE val
#include <print_templ.c>
#undef BASE
#define BASE multi_val
#include <print_templ.c>
#undef BASE
#define BASE space
#include <print_templ.c>
#undef BASE
#define BASE local_space
#include <print_templ.c>
#undef BASE
#define BASE basic_set
#include <print_templ.c>
#undef BASE
#define BASE basic_map
#include <print_templ.c>
#undef BASE
#define BASE set
#include <print_templ.c>
#undef BASE
#define BASE map
#include <print_templ.c>
#undef BASE
#define BASE union_set
#include <print_templ.c>
#undef BASE
#define BASE union_map
#include <print_templ.c>
#undef BASE
#define BASE qpolynomial
#include <print_templ.c>
#undef BASE
#define BASE qpolynomial_fold
#include <print_templ.c>
#undef BASE
#define BASE pw_qpolynomial
#include <print_templ.c>
#undef BASE
#define BASE pw_qpolynomial_fold
#include <print_templ.c>
#undef BASE
#define BASE union_pw_qpolynomial
#include <print_templ.c>
#undef BASE
#define BASE union_pw_qpolynomial_fold
#include <print_templ.c>
#undef BASE
#define BASE band
#include <print_templ.c>
#undef BASE
#define BASE constraint
#include <print_templ.c>
#undef BASE
#define BASE aff
#include <print_templ.c>
#undef BASE
#define BASE pw_aff
#include <print_templ.c>
#undef BASE
#define BASE multi_aff
#include <print_templ.c>
#undef BASE
#define BASE pw_multi_aff
#include <print_templ.c>
#undef BASE
#define BASE union_pw_multi_aff
#include <print_templ.c>
#undef BASE
#define BASE multi_pw_aff
#include <print_templ.c>
#undef BASE
#define BASE union_pw_aff
#include <print_templ.c>
#undef BASE
#define BASE multi_union_pw_aff
#include <print_templ.c>
#undef BASE
#define BASE point
#include <print_templ.c>
#undef BASE
#define BASE ast_expr
#include <print_templ_yaml.c>
#undef BASE
#define BASE ast_node
#include <print_templ_yaml.c>
