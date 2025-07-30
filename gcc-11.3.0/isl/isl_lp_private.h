#ifndef ISL_LP_PRIVATE_H
#define ISL_LP_PRIVATE_H

#include <isl_int.h>
#include <isl/lp.h>
#include <isl/vec.h>

enum isl_lp_result isl_basic_map_solve_lp(__isl_keep isl_basic_map *bmap,
	int max, isl_int *f, isl_int denom, isl_int *opt, isl_int *opt_denom,
	__isl_give isl_vec **sol);
enum isl_lp_result isl_basic_set_solve_lp(__isl_keep isl_basic_set *bset,
	int max, isl_int *f, isl_int denom, isl_int *opt, isl_int *opt_denom,
	__isl_give isl_vec **sol);
enum isl_lp_result isl_map_solve_lp(__isl_keep isl_map *map, int max,
	isl_int *f, isl_int denom, isl_int *opt, isl_int *opt_denom,
	__isl_give isl_vec **sol);
enum isl_lp_result isl_set_solve_lp(__isl_keep isl_set *set, int max,
	isl_int *f, isl_int denom, isl_int *opt, isl_int *opt_denom,
	__isl_give isl_vec **sol);

#endif
