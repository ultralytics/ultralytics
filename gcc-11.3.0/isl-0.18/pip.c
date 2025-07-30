/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <assert.h>
#include <string.h>
#include <isl_map_private.h>
#include <isl/aff.h>
#include <isl/set.h>
#include "isl_tab.h"
#include "isl_sample.h"
#include "isl_scan.h"
#include <isl_seq.h>
#include <isl_ilp_private.h>
#include <isl/printer.h>
#include <isl_point_private.h>
#include <isl_vec_private.h>
#include <isl/options.h>
#include <isl_config.h>

/* The input of this program is the same as that of the "example" program
 * from the PipLib distribution, except that the "big parameter column"
 * should always be -1.
 *
 * Context constraints in PolyLib format
 * -1
 * Problem constraints in PolyLib format
 * Optional list of options
 *
 * The options are
 *	Maximize	compute maximum instead of minimum
 *	Rational	compute rational optimum instead of integer optimum
 *	Urs_parms	don't assume parameters are non-negative
 *	Urs_unknowns	don't assume unknowns are non-negative
 */

struct options {
	struct isl_options	*isl;
	unsigned		 verify;
	unsigned		 format;
};

#define FORMAT_SET	0
#define FORMAT_AFF	1

struct isl_arg_choice pip_format[] = {
	{"set",		FORMAT_SET},
	{"affine",	FORMAT_AFF},
	{0}
};

ISL_ARGS_START(struct options, options_args)
ISL_ARG_CHILD(struct options, isl, "isl", &isl_options_args, "isl options")
ISL_ARG_BOOL(struct options, verify, 'T', "verify", 0, NULL)
ISL_ARG_CHOICE(struct options, format, 0, "format",
	pip_format, FORMAT_SET, "output format")
ISL_ARGS_END

ISL_ARG_DEF(options, struct options, options_args)

static __isl_give isl_basic_set *set_bounds(__isl_take isl_basic_set *bset)
{
	unsigned nparam;
	int i, r;
	isl_point *pt, *pt2;
	isl_basic_set *box;

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	r = nparam >= 8 ? 4 : nparam >= 5 ? 6 : 30;

	pt = isl_basic_set_sample_point(isl_basic_set_copy(bset));
	pt2 = isl_point_copy(pt);

	for (i = 0; i < nparam; ++i) {
		pt = isl_point_add_ui(pt, isl_dim_param, i, r);
		pt2 = isl_point_sub_ui(pt2, isl_dim_param, i, r);
	}

	box = isl_basic_set_box_from_points(pt, pt2);

	return isl_basic_set_intersect(bset, box);
}

static struct isl_basic_set *to_parameter_domain(struct isl_basic_set *context)
{
	context = isl_basic_set_move_dims(context, isl_dim_param, 0,
		    isl_dim_set, 0, isl_basic_set_dim(context, isl_dim_set));
	context = isl_basic_set_params(context);
	return context;
}

/* Plug in the initial values of "params" for the parameters in "bset" and
 * return the result.  The remaining entries in "params", if any,
 * correspond to the existentially quantified variables in the description
 * of the original context and can be ignored.
 */
static __isl_give isl_basic_set *plug_in_parameters(
	__isl_take isl_basic_set *bset, __isl_take isl_vec *params)
{
	int i, n;

	n = isl_basic_set_dim(bset, isl_dim_param);
	for (i = 0; i < n; ++i)
		bset = isl_basic_set_fix(bset,
					 isl_dim_param, i, params->el[1 + i]);

	bset = isl_basic_set_remove_dims(bset, isl_dim_param, 0, n);

	isl_vec_free(params);

	return bset;
}

/* Plug in the initial values of "params" for the parameters in "set" and
 * return the result.  The remaining entries in "params", if any,
 * correspond to the existentially quantified variables in the description
 * of the original context and can be ignored.
 */
static __isl_give isl_set *set_plug_in_parameters(__isl_take isl_set *set,
	__isl_take isl_vec *params)
{
	int i, n;

	n = isl_set_dim(set, isl_dim_param);
	for (i = 0; i < n; ++i)
		set = isl_set_fix(set, isl_dim_param, i, params->el[1 + i]);

	set = isl_set_remove_dims(set, isl_dim_param, 0, n);

	isl_vec_free(params);

	return set;
}

/* Compute the lexicographically minimal (or maximal if max is set)
 * element of bset for the given values of the parameters, by
 * successively solving an ilp problem in each direction.
 */
static __isl_give isl_vec *opt_at(__isl_take isl_basic_set *bset,
	__isl_take isl_vec *params, int max)
{
	unsigned dim;
	isl_ctx *ctx;
	struct isl_vec *opt;
	struct isl_vec *obj;
	int i;

	dim = isl_basic_set_dim(bset, isl_dim_set);

	bset = plug_in_parameters(bset, params);

	ctx = isl_basic_set_get_ctx(bset);
	if (isl_basic_set_plain_is_empty(bset)) {
		opt = isl_vec_alloc(ctx, 0);
		isl_basic_set_free(bset);
		return opt;
	}

	opt = isl_vec_alloc(ctx, 1 + dim);
	assert(opt);

	obj = isl_vec_alloc(ctx, 1 + dim);
	assert(obj);

	isl_int_set_si(opt->el[0], 1);
	isl_int_set_si(obj->el[0], 0);

	for (i = 0; i < dim; ++i) {
		enum isl_lp_result res;

		isl_seq_clr(obj->el + 1, dim);
		isl_int_set_si(obj->el[1 + i], 1);
		res = isl_basic_set_solve_ilp(bset, max, obj->el,
						&opt->el[1 + i], NULL);
		if (res == isl_lp_empty)
			goto empty;
		assert(res == isl_lp_ok);
		bset = isl_basic_set_fix(bset, isl_dim_set, i, opt->el[1 + i]);
	}

	isl_basic_set_free(bset);
	isl_vec_free(obj);

	return opt;
empty:
	isl_vec_free(opt);
	opt = isl_vec_alloc(ctx, 0);
	isl_basic_set_free(bset);
	isl_vec_free(obj);

	return opt;
}

struct isl_scan_pip {
	struct isl_scan_callback callback;
	isl_basic_set *bset;
	isl_set *sol;
	isl_set *empty;
	int stride;
	int n;
	int max;
};

/* Check if the "manually" computed optimum of bset at the "sample"
 * values of the parameters agrees with the solution of pilp problem
 * represented by the pair (sol, empty).
 * In particular, if there is no solution for this value of the parameters,
 * then it should be an element of the parameter domain "empty".
 * Otherwise, the optimal solution, should be equal to the result of
 * plugging in the value of the parameters in "sol".
 */
static isl_stat scan_one(struct isl_scan_callback *callback,
	__isl_take isl_vec *sample)
{
	struct isl_scan_pip *sp = (struct isl_scan_pip *)callback;
	struct isl_vec *opt;

	sp->n--;

	opt = opt_at(isl_basic_set_copy(sp->bset), isl_vec_copy(sample), sp->max);
	assert(opt);

	if (opt->size == 0) {
		isl_point *sample_pnt;
		sample_pnt = isl_point_alloc(isl_set_get_space(sp->empty), sample);
		assert(isl_set_contains_point(sp->empty, sample_pnt));
		isl_point_free(sample_pnt);
		isl_vec_free(opt);
	} else {
		isl_set *sol;
		isl_set *opt_set;
		opt_set = isl_set_from_basic_set(isl_basic_set_from_vec(opt));
		sol = set_plug_in_parameters(isl_set_copy(sp->sol), sample);
		assert(isl_set_is_equal(opt_set, sol));
		isl_set_free(sol);
		isl_set_free(opt_set);
	}

	if (!(sp->n % sp->stride)) {
		printf("o");
		fflush(stdout);
	}

	return sp->n >= 1 ? isl_stat_ok : isl_stat_error;
}

static void check_solution(isl_basic_set *bset, isl_basic_set *context,
	isl_set *sol, isl_set *empty, int max)
{
	struct isl_scan_pip sp;
	isl_int count, count_max;
	int i, n;
	int r;

	context = set_bounds(context);
	context = isl_basic_set_underlying_set(context);

	isl_int_init(count);
	isl_int_init(count_max);

	isl_int_set_si(count_max, 2000);
	r = isl_basic_set_count_upto(context, count_max, &count);
	assert(r >= 0);
	n = isl_int_get_si(count);

	isl_int_clear(count_max);
	isl_int_clear(count);

	sp.callback.add = scan_one;
	sp.bset = bset;
	sp.sol = sol;
	sp.empty = empty;
	sp.n = n;
	sp.stride = n > 70 ? 1 + (n + 1)/70 : 1;
	sp.max = max;

	for (i = 0; i < n; i += sp.stride)
		printf(".");
	printf("\r");
	fflush(stdout);

	isl_basic_set_scan(context, &sp.callback);

	printf("\n");

	isl_basic_set_free(bset);
}

int main(int argc, char **argv)
{
	struct isl_ctx *ctx;
	struct isl_basic_set *context, *bset, *copy, *context_copy;
	struct isl_set *set = NULL;
	struct isl_set *empty;
	isl_pw_multi_aff *pma = NULL;
	int neg_one;
	char s[1024];
	int urs_parms = 0;
	int urs_unknowns = 0;
	int max = 0;
	int rational = 0;
	int n;
	int nparam;
	struct options *options;

	options = options_new_with_defaults();
	assert(options);
	argc = options_parse(options, argc, argv, ISL_ARG_ALL);

	ctx = isl_ctx_alloc_with_options(&options_args, options);

	context = isl_basic_set_read_from_file(ctx, stdin);
	assert(context);
	n = fscanf(stdin, "%d", &neg_one);
	assert(n == 1);
	assert(neg_one == -1);
	bset = isl_basic_set_read_from_file(ctx, stdin);

	while (fgets(s, sizeof(s), stdin)) {
		if (strncasecmp(s, "Maximize", 8) == 0)
			max = 1;
		if (strncasecmp(s, "Rational", 8) == 0) {
			rational = 1;
			bset = isl_basic_set_set_rational(bset);
		}
		if (strncasecmp(s, "Urs_parms", 9) == 0)
			urs_parms = 1;
		if (strncasecmp(s, "Urs_unknowns", 12) == 0)
			urs_unknowns = 1;
	}
	if (!urs_parms)
		context = isl_basic_set_intersect(context,
		isl_basic_set_positive_orthant(isl_basic_set_get_space(context)));
	context = to_parameter_domain(context);
	nparam = isl_basic_set_dim(context, isl_dim_param);
	if (nparam != isl_basic_set_dim(bset, isl_dim_param)) {
		int dim = isl_basic_set_dim(bset, isl_dim_set);
		bset = isl_basic_set_move_dims(bset, isl_dim_param, 0,
					    isl_dim_set, dim - nparam, nparam);
	}
	if (!urs_unknowns)
		bset = isl_basic_set_intersect(bset,
		isl_basic_set_positive_orthant(isl_basic_set_get_space(bset)));

	if (options->verify) {
		copy = isl_basic_set_copy(bset);
		context_copy = isl_basic_set_copy(context);
	}

	if (options->format == FORMAT_AFF) {
		if (max)
			pma = isl_basic_set_partial_lexmax_pw_multi_aff(bset,
								context, &empty);
		else
			pma = isl_basic_set_partial_lexmin_pw_multi_aff(bset,
								context, &empty);
	} else {
		if (max)
			set = isl_basic_set_partial_lexmax(bset,
								context, &empty);
		else
			set = isl_basic_set_partial_lexmin(bset,
								context, &empty);
	}

	if (options->verify) {
		assert(!rational);
		if (options->format == FORMAT_AFF)
			set = isl_set_from_pw_multi_aff(pma);
		check_solution(copy, context_copy, set, empty, max);
		isl_set_free(set);
	} else {
		isl_printer *p;
		p = isl_printer_to_file(ctx, stdout);
		if (options->format == FORMAT_AFF)
			p = isl_printer_print_pw_multi_aff(p, pma);
		else
			p = isl_printer_print_set(p, set);
		p = isl_printer_end_line(p);
		p = isl_printer_print_str(p, "no solution: ");
		p = isl_printer_print_set(p, empty);
		p = isl_printer_end_line(p);
		isl_printer_free(p);
		isl_set_free(set);
		isl_pw_multi_aff_free(pma);
	}

	isl_set_free(empty);
	isl_ctx_free(ctx);

	return 0;
}
