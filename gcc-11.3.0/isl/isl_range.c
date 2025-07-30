#include <isl_ctx_private.h>
#include <isl_constraint_private.h>
#include <isl/set.h>
#include <isl_polynomial_private.h>
#include <isl_morph.h>
#include <isl_range.h>

struct range_data {
	struct isl_bound	*bound;
	int 		    	*signs;
	int			sign;
	int			test_monotonicity;
	int		    	monotonicity;
	int			tight;
	isl_qpolynomial	    	*poly;
	isl_pw_qpolynomial_fold *pwf;
	isl_pw_qpolynomial_fold *pwf_tight;
};

static isl_stat propagate_on_domain(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct range_data *data);

/* Check whether the polynomial "poly" has sign "sign" over "bset",
 * i.e., if sign == 1, check that the lower bound on the polynomial
 * is non-negative and if sign == -1, check that the upper bound on
 * the polynomial is non-positive.
 */
static int has_sign(__isl_keep isl_basic_set *bset,
	__isl_keep isl_qpolynomial *poly, int sign, int *signs)
{
	struct range_data data_m;
	unsigned nparam;
	isl_space *dim;
	isl_val *opt;
	int r;
	enum isl_fold type;

	nparam = isl_basic_set_dim(bset, isl_dim_param);

	bset = isl_basic_set_copy(bset);
	poly = isl_qpolynomial_copy(poly);

	bset = isl_basic_set_move_dims(bset, isl_dim_set, 0,
					isl_dim_param, 0, nparam);
	poly = isl_qpolynomial_move_dims(poly, isl_dim_in, 0,
					isl_dim_param, 0, nparam);

	dim = isl_qpolynomial_get_space(poly);
	dim = isl_space_params(dim);
	dim = isl_space_from_domain(dim);
	dim = isl_space_add_dims(dim, isl_dim_out, 1);

	data_m.test_monotonicity = 0;
	data_m.signs = signs;
	data_m.sign = -sign;
	type = data_m.sign < 0 ? isl_fold_min : isl_fold_max;
	data_m.pwf = isl_pw_qpolynomial_fold_zero(dim, type);
	data_m.tight = 0;
	data_m.pwf_tight = NULL;

	if (propagate_on_domain(bset, poly, &data_m) < 0)
		goto error;

	if (sign > 0)
		opt = isl_pw_qpolynomial_fold_min(data_m.pwf);
	else
		opt = isl_pw_qpolynomial_fold_max(data_m.pwf);

	if (!opt)
		r = -1;
	else if (isl_val_is_nan(opt) ||
		 isl_val_is_infty(opt) ||
		 isl_val_is_neginfty(opt))
		r = 0;
	else
		r = sign * isl_val_sgn(opt) >= 0;

	isl_val_free(opt);

	return r;
error:
	isl_pw_qpolynomial_fold_free(data_m.pwf);
	return -1;
}

/* Return  1 if poly is monotonically increasing in the last set variable,
 *        -1 if poly is monotonically decreasing in the last set variable,
 *	   0 if no conclusion,
 *	  -2 on error.
 *
 * We simply check the sign of p(x+1)-p(x)
 */
static int monotonicity(__isl_keep isl_basic_set *bset,
	__isl_keep isl_qpolynomial *poly, struct range_data *data)
{
	isl_ctx *ctx;
	isl_space *dim;
	isl_qpolynomial *sub = NULL;
	isl_qpolynomial *diff = NULL;
	int result = 0;
	int s;
	unsigned nvar;

	ctx = isl_qpolynomial_get_ctx(poly);
	dim = isl_qpolynomial_get_domain_space(poly);

	nvar = isl_basic_set_dim(bset, isl_dim_set);

	sub = isl_qpolynomial_var_on_domain(isl_space_copy(dim), isl_dim_set, nvar - 1);
	sub = isl_qpolynomial_add(sub,
		isl_qpolynomial_rat_cst_on_domain(dim, ctx->one, ctx->one));

	diff = isl_qpolynomial_substitute(isl_qpolynomial_copy(poly),
			isl_dim_in, nvar - 1, 1, &sub);
	diff = isl_qpolynomial_sub(diff, isl_qpolynomial_copy(poly));

	s = has_sign(bset, diff, 1, data->signs);
	if (s < 0)
		goto error;
	if (s)
		result = 1;
	else {
		s = has_sign(bset, diff, -1, data->signs);
		if (s < 0)
			goto error;
		if (s)
			result = -1;
	}

	isl_qpolynomial_free(diff);
	isl_qpolynomial_free(sub);

	return result;
error:
	isl_qpolynomial_free(diff);
	isl_qpolynomial_free(sub);
	return -2;
}

/* Return a positive ("sign" > 0) or negative ("sign" < 0) infinite polynomial
 * with domain space "space".
 */
static __isl_give isl_qpolynomial *signed_infty(__isl_take isl_space *space,
	int sign)
{
	if (sign > 0)
		return isl_qpolynomial_infty_on_domain(space);
	else
		return isl_qpolynomial_neginfty_on_domain(space);
}

static __isl_give isl_qpolynomial *bound2poly(__isl_take isl_constraint *bound,
	__isl_take isl_space *space, unsigned pos, int sign)
{
	if (!bound)
		return signed_infty(space, sign);
	isl_space_free(space);
	return isl_qpolynomial_from_constraint(bound, isl_dim_set, pos);
}

static int bound_is_integer(__isl_take isl_constraint *bound, unsigned pos)
{
	isl_int c;
	int is_int;

	if (!bound)
		return 1;

	isl_int_init(c);
	isl_constraint_get_coefficient(bound, isl_dim_set, pos, &c);
	is_int = isl_int_is_one(c) || isl_int_is_negone(c);
	isl_int_clear(c);

	return is_int;
}

struct isl_fixed_sign_data {
	int		*signs;
	int		sign;
	isl_qpolynomial	*poly;
};

/* Add term "term" to data->poly if it has sign data->sign.
 * The sign is determined based on the signs of the parameters
 * and variables in data->signs.  The integer divisions, if
 * any, are assumed to be non-negative.
 */
static isl_stat collect_fixed_sign_terms(__isl_take isl_term *term, void *user)
{
	struct isl_fixed_sign_data *data = (struct isl_fixed_sign_data *)user;
	isl_int n;
	int i;
	int sign;
	unsigned nparam;
	unsigned nvar;

	if (!term)
		return isl_stat_error;

	nparam = isl_term_dim(term, isl_dim_param);
	nvar = isl_term_dim(term, isl_dim_set);

	isl_int_init(n);

	isl_term_get_num(term, &n);

	sign = isl_int_sgn(n);
	for (i = 0; i < nparam; ++i) {
		if (data->signs[i] > 0)
			continue;
		if (isl_term_get_exp(term, isl_dim_param, i) % 2)
			sign = -sign;
	}
	for (i = 0; i < nvar; ++i) {
		if (data->signs[nparam + i] > 0)
			continue;
		if (isl_term_get_exp(term, isl_dim_set, i) % 2)
			sign = -sign;
	}

	if (sign == data->sign) {
		isl_qpolynomial *t = isl_qpolynomial_from_term(term);

		data->poly = isl_qpolynomial_add(data->poly, t);
	} else
		isl_term_free(term);

	isl_int_clear(n);

	return isl_stat_ok;
}

/* Construct and return a polynomial that consists of the terms
 * in "poly" that have sign "sign".  The integer divisions, if
 * any, are assumed to be non-negative.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_terms_of_sign(
	__isl_keep isl_qpolynomial *poly, int *signs, int sign)
{
	isl_space *space;
	struct isl_fixed_sign_data data = { signs, sign };

	space = isl_qpolynomial_get_domain_space(poly);
	data.poly = isl_qpolynomial_zero_on_domain(space);

	if (isl_qpolynomial_foreach_term(poly, collect_fixed_sign_terms, &data) < 0)
		goto error;

	return data.poly;
error:
	isl_qpolynomial_free(data.poly);
	return NULL;
}

/* Helper function to add a guarded polynomial to either pwf_tight or pwf,
 * depending on whether the result has been determined to be tight.
 */
static isl_stat add_guarded_poly(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct range_data *data)
{
	enum isl_fold type = data->sign < 0 ? isl_fold_min : isl_fold_max;
	isl_set *set;
	isl_qpolynomial_fold *fold;
	isl_pw_qpolynomial_fold *pwf;

	bset = isl_basic_set_params(bset);
	poly = isl_qpolynomial_project_domain_on_params(poly);

	fold = isl_qpolynomial_fold_alloc(type, poly);
	set = isl_set_from_basic_set(bset);
	pwf = isl_pw_qpolynomial_fold_alloc(type, set, fold);
	if (data->tight)
		data->pwf_tight = isl_pw_qpolynomial_fold_fold(
						data->pwf_tight, pwf);
	else
		data->pwf = isl_pw_qpolynomial_fold_fold(data->pwf, pwf);

	return isl_stat_ok;
}

/* Plug in "sub" for the variable at position "pos" in "poly".
 *
 * If "sub" is an infinite polynomial and if the variable actually
 * appears in "poly", then calling isl_qpolynomial_substitute
 * to perform the substitution may result in a NaN result.
 * In such cases, return positive or negative infinity instead,
 * depending on whether an upper bound or a lower bound is being computed,
 * and mark the result as not being tight.
 */
static __isl_give isl_qpolynomial *plug_in_at_pos(
	__isl_take isl_qpolynomial *poly, int pos,
	__isl_take isl_qpolynomial *sub, struct range_data *data)
{
	isl_bool involves, infty;

	involves = isl_qpolynomial_involves_dims(poly, isl_dim_in, pos, 1);
	if (involves < 0)
		goto error;
	if (!involves) {
		isl_qpolynomial_free(sub);
		return poly;
	}

	infty = isl_qpolynomial_is_infty(sub);
	if (infty >= 0 && !infty)
		infty = isl_qpolynomial_is_neginfty(sub);
	if (infty < 0)
		goto error;
	if (infty) {
		isl_space *space = isl_qpolynomial_get_domain_space(poly);
		data->tight = 0;
		isl_qpolynomial_free(poly);
		isl_qpolynomial_free(sub);
		return signed_infty(space, data->sign);
	}

	poly = isl_qpolynomial_substitute(poly, isl_dim_in, pos, 1, &sub);
	isl_qpolynomial_free(sub);

	return poly;
error:
	isl_qpolynomial_free(poly);
	isl_qpolynomial_free(sub);
	return NULL;
}

/* Given a lower and upper bound on the final variable and constraints
 * on the remaining variables where these bounds are active,
 * eliminate the variable from data->poly based on these bounds.
 * If the polynomial has been determined to be monotonic
 * in the variable, then simply plug in the appropriate bound.
 * If the current polynomial is tight and if this bound is integer,
 * then the result is still tight.  In all other cases, the results
 * may not be tight.
 * Otherwise, plug in the largest bound (in absolute value) in
 * the positive terms (if an upper bound is wanted) or the negative terms
 * (if a lower bounded is wanted) and the other bound in the other terms.
 *
 * If all variables have been eliminated, then record the result.
 * Ohterwise, recurse on the next variable.
 */
static isl_stat propagate_on_bound_pair(__isl_take isl_constraint *lower,
	__isl_take isl_constraint *upper, __isl_take isl_basic_set *bset,
	void *user)
{
	struct range_data *data = (struct range_data *)user;
	int save_tight = data->tight;
	isl_qpolynomial *poly;
	isl_stat r;
	unsigned nvar;

	nvar = isl_basic_set_dim(bset, isl_dim_set);

	if (data->monotonicity) {
		isl_qpolynomial *sub;
		isl_space *dim = isl_qpolynomial_get_domain_space(data->poly);
		if (data->monotonicity * data->sign > 0) {
			if (data->tight)
				data->tight = bound_is_integer(upper, nvar);
			sub = bound2poly(upper, dim, nvar, 1);
			isl_constraint_free(lower);
		} else {
			if (data->tight)
				data->tight = bound_is_integer(lower, nvar);
			sub = bound2poly(lower, dim, nvar, -1);
			isl_constraint_free(upper);
		}
		poly = isl_qpolynomial_copy(data->poly);
		poly = plug_in_at_pos(poly, nvar, sub, data);
		poly = isl_qpolynomial_drop_dims(poly, isl_dim_in, nvar, 1);
	} else {
		isl_qpolynomial *l, *u;
		isl_qpolynomial *pos, *neg;
		isl_space *dim = isl_qpolynomial_get_domain_space(data->poly);
		unsigned nparam = isl_basic_set_dim(bset, isl_dim_param);
		int sign = data->sign * data->signs[nparam + nvar];

		data->tight = 0;

		u = bound2poly(upper, isl_space_copy(dim), nvar, 1);
		l = bound2poly(lower, dim, nvar, -1);

		pos = isl_qpolynomial_terms_of_sign(data->poly, data->signs, sign);
		neg = isl_qpolynomial_terms_of_sign(data->poly, data->signs, -sign);

		pos = plug_in_at_pos(pos, nvar, u, data);
		neg = plug_in_at_pos(neg, nvar, l, data);

		poly = isl_qpolynomial_add(pos, neg);
		poly = isl_qpolynomial_drop_dims(poly, isl_dim_in, nvar, 1);
	}

	if (isl_basic_set_dim(bset, isl_dim_set) == 0)
		r = add_guarded_poly(bset, poly, data);
	else
		r = propagate_on_domain(bset, poly, data);

	data->tight = save_tight;

	return r;
}

/* Recursively perform range propagation on the polynomial "poly"
 * defined over the basic set "bset" and collect the results in "data".
 */
static isl_stat propagate_on_domain(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct range_data *data)
{
	isl_ctx *ctx;
	isl_qpolynomial *save_poly = data->poly;
	int save_monotonicity = data->monotonicity;
	unsigned d;

	if (!bset || !poly)
		goto error;

	ctx = isl_basic_set_get_ctx(bset);
	d = isl_basic_set_dim(bset, isl_dim_set);
	isl_assert(ctx, d >= 1, goto error);

	if (isl_qpolynomial_is_cst(poly, NULL, NULL)) {
		bset = isl_basic_set_project_out(bset, isl_dim_set, 0, d);
		poly = isl_qpolynomial_drop_dims(poly, isl_dim_in, 0, d);
		return add_guarded_poly(bset, poly, data);
	}

	if (data->test_monotonicity)
		data->monotonicity = monotonicity(bset, poly, data);
	else
		data->monotonicity = 0;
	if (data->monotonicity < -1)
		goto error;

	data->poly = poly;
	if (isl_basic_set_foreach_bound_pair(bset, isl_dim_set, d - 1,
					    &propagate_on_bound_pair, data) < 0)
		goto error;

	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);
	data->monotonicity = save_monotonicity;
	data->poly = save_poly;

	return isl_stat_ok;
error:
	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);
	data->monotonicity = save_monotonicity;
	data->poly = save_poly;
	return isl_stat_error;
}

static isl_stat basic_guarded_poly_bound(__isl_take isl_basic_set *bset,
	void *user)
{
	struct range_data *data = (struct range_data *)user;
	isl_ctx *ctx;
	unsigned nparam = isl_basic_set_dim(bset, isl_dim_param);
	unsigned dim = isl_basic_set_dim(bset, isl_dim_set);
	isl_stat r;

	data->signs = NULL;

	ctx = isl_basic_set_get_ctx(bset);
	data->signs = isl_alloc_array(ctx, int,
					isl_basic_set_dim(bset, isl_dim_all));

	if (isl_basic_set_dims_get_sign(bset, isl_dim_set, 0, dim,
					data->signs + nparam) < 0)
		goto error;
	if (isl_basic_set_dims_get_sign(bset, isl_dim_param, 0, nparam,
					data->signs) < 0)
		goto error;

	r = propagate_on_domain(bset, isl_qpolynomial_copy(data->poly), data);

	free(data->signs);

	return r;
error:
	free(data->signs);
	isl_basic_set_free(bset);
	return isl_stat_error;
}

static int qpolynomial_bound_on_domain_range(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct range_data *data)
{
	unsigned nparam = isl_basic_set_dim(bset, isl_dim_param);
	unsigned nvar = isl_basic_set_dim(bset, isl_dim_set);
	isl_set *set = NULL;

	if (!bset)
		goto error;

	if (nvar == 0)
		return add_guarded_poly(bset, poly, data);

	set = isl_set_from_basic_set(bset);
	set = isl_set_split_dims(set, isl_dim_param, 0, nparam);
	set = isl_set_split_dims(set, isl_dim_set, 0, nvar);

	data->poly = poly;

	data->test_monotonicity = 1;
	if (isl_set_foreach_basic_set(set, &basic_guarded_poly_bound, data) < 0)
		goto error;

	isl_set_free(set);
	isl_qpolynomial_free(poly);

	return 0;
error:
	isl_set_free(set);
	isl_qpolynomial_free(poly);
	return -1;
}

int isl_qpolynomial_bound_on_domain_range(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct isl_bound *bound)
{
	struct range_data data;
	int r;

	data.pwf = bound->pwf;
	data.pwf_tight = bound->pwf_tight;
	data.tight = bound->check_tight;
	if (bound->type == isl_fold_min)
		data.sign = -1;
	else
		data.sign = 1;

	r = qpolynomial_bound_on_domain_range(bset, poly, &data);

	bound->pwf = data.pwf;
	bound->pwf_tight = data.pwf_tight;

	return r;
}
