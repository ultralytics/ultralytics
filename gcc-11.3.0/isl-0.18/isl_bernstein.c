/*
 * Copyright 2006-2007 Universiteit Leiden
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, Leiden Institute of Advanced Computer Science,
 * Universiteit Leiden, Niels Bohrweg 1, 2333 CA Leiden, The Netherlands
 * and K.U.Leuven, Departement Computerwetenschappen, Celestijnenlaan 200A,
 * B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 */

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl/set.h>
#include <isl_seq.h>
#include <isl_morph.h>
#include <isl_factorization.h>
#include <isl_vertices_private.h>
#include <isl_polynomial_private.h>
#include <isl_options_private.h>
#include <isl_vec_private.h>
#include <isl_bernstein.h>

struct bernstein_data {
	enum isl_fold type;
	isl_qpolynomial *poly;
	int check_tight;

	isl_cell *cell;

	isl_qpolynomial_fold *fold;
	isl_qpolynomial_fold *fold_tight;
	isl_pw_qpolynomial_fold *pwf;
	isl_pw_qpolynomial_fold *pwf_tight;
};

static int vertex_is_integral(__isl_keep isl_basic_set *vertex)
{
	unsigned nvar;
	unsigned nparam;
	int i;

	nvar = isl_basic_set_dim(vertex, isl_dim_set);
	nparam = isl_basic_set_dim(vertex, isl_dim_param);
	for (i = 0; i < nvar; ++i) {
		int r = nvar - 1 - i;
		if (!isl_int_is_one(vertex->eq[r][1 + nparam + i]) &&
		    !isl_int_is_negone(vertex->eq[r][1 + nparam + i]))
			return 0;
	}

	return 1;
}

static __isl_give isl_qpolynomial *vertex_coordinate(
	__isl_keep isl_basic_set *vertex, int i, __isl_take isl_space *dim)
{
	unsigned nvar;
	unsigned nparam;
	int r;
	isl_int denom;
	isl_qpolynomial *v;

	nvar = isl_basic_set_dim(vertex, isl_dim_set);
	nparam = isl_basic_set_dim(vertex, isl_dim_param);
	r = nvar - 1 - i;

	isl_int_init(denom);
	isl_int_set(denom, vertex->eq[r][1 + nparam + i]);
	isl_assert(vertex->ctx, !isl_int_is_zero(denom), goto error);

	if (isl_int_is_pos(denom))
		isl_seq_neg(vertex->eq[r], vertex->eq[r],
				1 + isl_basic_set_total_dim(vertex));
	else
		isl_int_neg(denom, denom);

	v = isl_qpolynomial_from_affine(dim, vertex->eq[r], denom);
	isl_int_clear(denom);

	return v;
error:
	isl_space_free(dim);
	isl_int_clear(denom);
	return NULL;
}

/* Check whether the bound associated to the selection "k" is tight,
 * which is the case if we select exactly one vertex and if that vertex
 * is integral for all values of the parameters.
 */
static int is_tight(int *k, int n, int d, isl_cell *cell)
{
	int i;

	for (i = 0; i < n; ++i) {
		int v;
		if (k[i] != d) {
			if (k[i])
				return 0;
			continue;
		}
		v = cell->ids[n - 1 - i];
		return vertex_is_integral(cell->vertices->v[v].vertex);
	}

	return 0;
}

static void add_fold(__isl_take isl_qpolynomial *b, __isl_keep isl_set *dom,
	int *k, int n, int d, struct bernstein_data *data)
{
	isl_qpolynomial_fold *fold;

	fold = isl_qpolynomial_fold_alloc(data->type, b);

	if (data->check_tight && is_tight(k, n, d, data->cell))
		data->fold_tight = isl_qpolynomial_fold_fold_on_domain(dom,
							data->fold_tight, fold);
	else
		data->fold = isl_qpolynomial_fold_fold_on_domain(dom,
							data->fold, fold);
}

/* Extract the coefficients of the Bernstein base polynomials and store
 * them in data->fold and data->fold_tight.
 *
 * In particular, the coefficient of each monomial
 * of multi-degree (k[0], k[1], ..., k[n-1]) is divided by the corresponding
 * multinomial coefficient d!/k[0]! k[1]! ... k[n-1]!
 *
 * c[i] contains the coefficient of the selected powers of the first i+1 vars.
 * multinom[i] contains the partial multinomial coefficient.
 */
static void extract_coefficients(isl_qpolynomial *poly,
	__isl_keep isl_set *dom, struct bernstein_data *data)
{
	int i;
	int d;
	int n;
	isl_ctx *ctx;
	isl_qpolynomial **c = NULL;
	int *k = NULL;
	int *left = NULL;
	isl_vec *multinom = NULL;

	if (!poly)
		return;

	ctx = isl_qpolynomial_get_ctx(poly);
	n = isl_qpolynomial_dim(poly, isl_dim_in);
	d = isl_qpolynomial_degree(poly);
	isl_assert(ctx, n >= 2, return);

	c = isl_calloc_array(ctx, isl_qpolynomial *, n);
	k = isl_alloc_array(ctx, int, n);
	left = isl_alloc_array(ctx, int, n);
	multinom = isl_vec_alloc(ctx, n);
	if (!c || !k || !left || !multinom)
		goto error;

	isl_int_set_si(multinom->el[0], 1);
	for (k[0] = d; k[0] >= 0; --k[0]) {
		int i = 1;
		isl_qpolynomial_free(c[0]);
		c[0] = isl_qpolynomial_coeff(poly, isl_dim_in, n - 1, k[0]);
		left[0] = d - k[0];
		k[1] = -1;
		isl_int_set(multinom->el[1], multinom->el[0]);
		while (i > 0) {
			if (i == n - 1) {
				int j;
				isl_space *dim;
				isl_qpolynomial *b;
				isl_qpolynomial *f;
				for (j = 2; j <= left[i - 1]; ++j)
					isl_int_divexact_ui(multinom->el[i],
						multinom->el[i], j);
				b = isl_qpolynomial_coeff(c[i - 1], isl_dim_in,
					n - 1 - i, left[i - 1]);
				b = isl_qpolynomial_project_domain_on_params(b);
				dim = isl_qpolynomial_get_domain_space(b);
				f = isl_qpolynomial_rat_cst_on_domain(dim, ctx->one,
					multinom->el[i]);
				b = isl_qpolynomial_mul(b, f);
				k[n - 1] = left[n - 2];
				add_fold(b, dom, k, n, d, data);
				--i;
				continue;
			}
			if (k[i] >= left[i - 1]) {
				--i;
				continue;
			}
			++k[i];
			if (k[i])
				isl_int_divexact_ui(multinom->el[i],
					multinom->el[i], k[i]);
			isl_qpolynomial_free(c[i]);
			c[i] = isl_qpolynomial_coeff(c[i - 1], isl_dim_in,
					n - 1 - i, k[i]);
			left[i] = left[i - 1] - k[i];
			k[i + 1] = -1;
			isl_int_set(multinom->el[i + 1], multinom->el[i]);
			++i;
		}
		isl_int_mul_ui(multinom->el[0], multinom->el[0], k[0]);
	}

	for (i = 0; i < n; ++i)
		isl_qpolynomial_free(c[i]);

	isl_vec_free(multinom);
	free(left);
	free(k);
	free(c);
	return;
error:
	isl_vec_free(multinom);
	free(left);
	free(k);
	if (c)
		for (i = 0; i < n; ++i)
			isl_qpolynomial_free(c[i]);
	free(c);
	return;
}

/* Perform bernstein expansion on the parametric vertices that are active
 * on "cell".
 *
 * data->poly has been homogenized in the calling function.
 *
 * We plug in the barycentric coordinates for the set variables
 *
 *		\vec x = \sum_i \alpha_i v_i(\vec p)
 *
 * and the constant "1 = \sum_i \alpha_i" for the homogeneous dimension.
 * Next, we extract the coefficients of the Bernstein base polynomials.
 */
static int bernstein_coefficients_cell(__isl_take isl_cell *cell, void *user)
{
	int i, j;
	struct bernstein_data *data = (struct bernstein_data *)user;
	isl_space *dim_param;
	isl_space *dim_dst;
	isl_qpolynomial *poly = data->poly;
	unsigned nvar;
	int n_vertices;
	isl_qpolynomial **subs;
	isl_pw_qpolynomial_fold *pwf;
	isl_set *dom;
	isl_ctx *ctx;

	if (!poly)
		goto error;

	nvar = isl_qpolynomial_dim(poly, isl_dim_in) - 1;
	n_vertices = cell->n_vertices;

	ctx = isl_qpolynomial_get_ctx(poly);
	if (n_vertices > nvar + 1 && ctx->opt->bernstein_triangulate)
		return isl_cell_foreach_simplex(cell,
					    &bernstein_coefficients_cell, user);

	subs = isl_alloc_array(ctx, isl_qpolynomial *, 1 + nvar);
	if (!subs)
		goto error;

	dim_param = isl_basic_set_get_space(cell->dom);
	dim_dst = isl_qpolynomial_get_domain_space(poly);
	dim_dst = isl_space_add_dims(dim_dst, isl_dim_set, n_vertices);

	for (i = 0; i < 1 + nvar; ++i)
		subs[i] = isl_qpolynomial_zero_on_domain(isl_space_copy(dim_dst));

	for (i = 0; i < n_vertices; ++i) {
		isl_qpolynomial *c;
		c = isl_qpolynomial_var_on_domain(isl_space_copy(dim_dst), isl_dim_set,
					1 + nvar + i);
		for (j = 0; j < nvar; ++j) {
			int k = cell->ids[i];
			isl_qpolynomial *v;
			v = vertex_coordinate(cell->vertices->v[k].vertex, j,
						isl_space_copy(dim_param));
			v = isl_qpolynomial_add_dims(v, isl_dim_in,
							1 + nvar + n_vertices);
			v = isl_qpolynomial_mul(v, isl_qpolynomial_copy(c));
			subs[1 + j] = isl_qpolynomial_add(subs[1 + j], v);
		}
		subs[0] = isl_qpolynomial_add(subs[0], c);
	}
	isl_space_free(dim_dst);

	poly = isl_qpolynomial_copy(poly);

	poly = isl_qpolynomial_add_dims(poly, isl_dim_in, n_vertices);
	poly = isl_qpolynomial_substitute(poly, isl_dim_in, 0, 1 + nvar, subs);
	poly = isl_qpolynomial_drop_dims(poly, isl_dim_in, 0, 1 + nvar);

	data->cell = cell;
	dom = isl_set_from_basic_set(isl_basic_set_copy(cell->dom));
	data->fold = isl_qpolynomial_fold_empty(data->type, isl_space_copy(dim_param));
	data->fold_tight = isl_qpolynomial_fold_empty(data->type, dim_param);
	extract_coefficients(poly, dom, data);

	pwf = isl_pw_qpolynomial_fold_alloc(data->type, isl_set_copy(dom),
					    data->fold);
	data->pwf = isl_pw_qpolynomial_fold_fold(data->pwf, pwf);
	pwf = isl_pw_qpolynomial_fold_alloc(data->type, dom, data->fold_tight);
	data->pwf_tight = isl_pw_qpolynomial_fold_fold(data->pwf_tight, pwf);

	isl_qpolynomial_free(poly);
	isl_cell_free(cell);
	for (i = 0; i < 1 + nvar; ++i)
		isl_qpolynomial_free(subs[i]);
	free(subs);
	return 0;
error:
	isl_cell_free(cell);
	return -1;
}

/* Base case of applying bernstein expansion.
 *
 * We compute the chamber decomposition of the parametric polytope "bset"
 * and then perform bernstein expansion on the parametric vertices
 * that are active on each chamber.
 */
static __isl_give isl_pw_qpolynomial_fold *bernstein_coefficients_base(
	__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct bernstein_data *data, int *tight)
{
	unsigned nvar;
	isl_space *dim;
	isl_pw_qpolynomial_fold *pwf;
	isl_vertices *vertices;
	int covers;

	nvar = isl_basic_set_dim(bset, isl_dim_set);
	if (nvar == 0) {
		isl_set *dom;
		isl_qpolynomial_fold *fold;

		fold = isl_qpolynomial_fold_alloc(data->type, poly);
		dom = isl_set_from_basic_set(bset);
		if (tight)
			*tight = 1;
		pwf = isl_pw_qpolynomial_fold_alloc(data->type, dom, fold);
		return isl_pw_qpolynomial_fold_project_domain_on_params(pwf);
	}

	if (isl_qpolynomial_is_zero(poly)) {
		isl_set *dom;
		isl_qpolynomial_fold *fold;
		fold = isl_qpolynomial_fold_alloc(data->type, poly);
		dom = isl_set_from_basic_set(bset);
		pwf = isl_pw_qpolynomial_fold_alloc(data->type, dom, fold);
		if (tight)
			*tight = 1;
		return isl_pw_qpolynomial_fold_project_domain_on_params(pwf);
	}

	dim = isl_basic_set_get_space(bset);
	dim = isl_space_params(dim);
	dim = isl_space_from_domain(dim);
	dim = isl_space_add_dims(dim, isl_dim_set, 1);
	data->pwf = isl_pw_qpolynomial_fold_zero(isl_space_copy(dim), data->type);
	data->pwf_tight = isl_pw_qpolynomial_fold_zero(dim, data->type);
	data->poly = isl_qpolynomial_homogenize(isl_qpolynomial_copy(poly));
	vertices = isl_basic_set_compute_vertices(bset);
	isl_vertices_foreach_disjoint_cell(vertices,
		&bernstein_coefficients_cell, data);
	isl_vertices_free(vertices);
	isl_qpolynomial_free(data->poly);

	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);

	covers = isl_pw_qpolynomial_fold_covers(data->pwf_tight, data->pwf);
	if (covers < 0)
		goto error;

	if (tight)
		*tight = covers;

	if (covers) {
		isl_pw_qpolynomial_fold_free(data->pwf);
		return data->pwf_tight;
	}

	data->pwf = isl_pw_qpolynomial_fold_fold(data->pwf, data->pwf_tight);

	return data->pwf;
error:
	isl_pw_qpolynomial_fold_free(data->pwf_tight);
	isl_pw_qpolynomial_fold_free(data->pwf);
	return NULL;
}

/* Apply bernstein expansion recursively by working in on len[i]
 * set variables at a time, with i ranging from n_group - 1 to 0.
 */
static __isl_give isl_pw_qpolynomial_fold *bernstein_coefficients_recursive(
	__isl_take isl_pw_qpolynomial *pwqp,
	int n_group, int *len, struct bernstein_data *data, int *tight)
{
	int i;
	unsigned nparam;
	unsigned nvar;
	isl_pw_qpolynomial_fold *pwf;

	if (!pwqp)
		return NULL;

	nparam = isl_pw_qpolynomial_dim(pwqp, isl_dim_param);
	nvar = isl_pw_qpolynomial_dim(pwqp, isl_dim_in);

	pwqp = isl_pw_qpolynomial_move_dims(pwqp, isl_dim_param, nparam,
					isl_dim_in, 0, nvar - len[n_group - 1]);
	pwf = isl_pw_qpolynomial_bound(pwqp, data->type, tight);

	for (i = n_group - 2; i >= 0; --i) {
		nparam = isl_pw_qpolynomial_fold_dim(pwf, isl_dim_param);
		pwf = isl_pw_qpolynomial_fold_move_dims(pwf, isl_dim_in, 0,
				isl_dim_param, nparam - len[i], len[i]);
		if (tight && !*tight)
			tight = NULL;
		pwf = isl_pw_qpolynomial_fold_bound(pwf, tight);
	}

	return pwf;
}

static __isl_give isl_pw_qpolynomial_fold *bernstein_coefficients_factors(
	__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct bernstein_data *data, int *tight)
{
	isl_factorizer *f;
	isl_set *set;
	isl_pw_qpolynomial *pwqp;
	isl_pw_qpolynomial_fold *pwf;

	f = isl_basic_set_factorizer(bset);
	if (!f)
		goto error;
	if (f->n_group == 0) {
		isl_factorizer_free(f);
		return  bernstein_coefficients_base(bset, poly, data, tight);
	}

	set = isl_set_from_basic_set(bset);
	pwqp = isl_pw_qpolynomial_alloc(set, poly);
	pwqp = isl_pw_qpolynomial_morph_domain(pwqp, isl_morph_copy(f->morph));

	pwf = bernstein_coefficients_recursive(pwqp, f->n_group, f->len, data,
						tight);

	isl_factorizer_free(f);

	return pwf;
error:
	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);
	return NULL;
}

static __isl_give isl_pw_qpolynomial_fold *bernstein_coefficients_full_recursive(
	__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct bernstein_data *data, int *tight)
{
	int i;
	int *len;
	unsigned nvar;
	isl_pw_qpolynomial_fold *pwf;
	isl_set *set;
	isl_pw_qpolynomial *pwqp;

	if (!bset || !poly)
		goto error;

	nvar = isl_basic_set_dim(bset, isl_dim_set);
	
	len = isl_alloc_array(bset->ctx, int, nvar);
	if (nvar && !len)
		goto error;

	for (i = 0; i < nvar; ++i)
		len[i] = 1;

	set = isl_set_from_basic_set(bset);
	pwqp = isl_pw_qpolynomial_alloc(set, poly);

	pwf = bernstein_coefficients_recursive(pwqp, nvar, len, data, tight);

	free(len);

	return pwf;
error:
	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);
	return NULL;
}

/* Compute a bound on the polynomial defined over the parametric polytope
 * using bernstein expansion and store the result
 * in bound->pwf and bound->pwf_tight.
 *
 * If bernstein_recurse is set to ISL_BERNSTEIN_FACTORS, we check if
 * the polytope can be factorized and apply bernstein expansion recursively
 * on the factors.
 * If bernstein_recurse is set to ISL_BERNSTEIN_INTERVALS, we apply
 * bernstein expansion recursively on each dimension.
 * Otherwise, we apply bernstein expansion on the entire polytope.
 */
int isl_qpolynomial_bound_on_domain_bernstein(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct isl_bound *bound)
{
	struct bernstein_data data;
	isl_pw_qpolynomial_fold *pwf;
	unsigned nvar;
	int tight = 0;
	int *tp = bound->check_tight ? &tight : NULL;

	if (!bset || !poly)
		goto error;

	data.type = bound->type;
	data.check_tight = bound->check_tight;

	nvar = isl_basic_set_dim(bset, isl_dim_set);

	if (bset->ctx->opt->bernstein_recurse & ISL_BERNSTEIN_FACTORS)
		pwf = bernstein_coefficients_factors(bset, poly, &data, tp);
	else if (nvar > 1 &&
	    (bset->ctx->opt->bernstein_recurse & ISL_BERNSTEIN_INTERVALS))
		pwf = bernstein_coefficients_full_recursive(bset, poly, &data, tp);
	else
		pwf = bernstein_coefficients_base(bset, poly, &data, tp);

	if (tight)
		bound->pwf_tight = isl_pw_qpolynomial_fold_fold(bound->pwf_tight, pwf);
	else
		bound->pwf = isl_pw_qpolynomial_fold_fold(bound->pwf, pwf);

	return 0;
error:
	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);
	return -1;
}
