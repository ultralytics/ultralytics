/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 */

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl/map.h>
#include <isl_seq.h>
#include <isl_space_private.h>
#include <isl_lp_private.h>
#include <isl/union_map.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl_options_private.h>
#include <isl_tarjan.h>

int isl_map_is_transitively_closed(__isl_keep isl_map *map)
{
	isl_map *map2;
	int closed;

	map2 = isl_map_apply_range(isl_map_copy(map), isl_map_copy(map));
	closed = isl_map_is_subset(map2, map);
	isl_map_free(map2);

	return closed;
}

int isl_union_map_is_transitively_closed(__isl_keep isl_union_map *umap)
{
	isl_union_map *umap2;
	int closed;

	umap2 = isl_union_map_apply_range(isl_union_map_copy(umap),
					  isl_union_map_copy(umap));
	closed = isl_union_map_is_subset(umap2, umap);
	isl_union_map_free(umap2);

	return closed;
}
 
/* Given a map that represents a path with the length of the path
 * encoded as the difference between the last output coordindate
 * and the last input coordinate, set this length to either
 * exactly "length" (if "exactly" is set) or at least "length"
 * (if "exactly" is not set).
 */
static __isl_give isl_map *set_path_length(__isl_take isl_map *map,
	int exactly, int length)
{
	isl_space *dim;
	struct isl_basic_map *bmap;
	unsigned d;
	unsigned nparam;
	int k;
	isl_int *c;

	if (!map)
		return NULL;

	dim = isl_map_get_space(map);
	d = isl_space_dim(dim, isl_dim_in);
	nparam = isl_space_dim(dim, isl_dim_param);
	bmap = isl_basic_map_alloc_space(dim, 0, 1, 1);
	if (exactly) {
		k = isl_basic_map_alloc_equality(bmap);
		if (k < 0)
			goto error;
		c = bmap->eq[k];
	} else {
		k = isl_basic_map_alloc_inequality(bmap);
		if (k < 0)
			goto error;
		c = bmap->ineq[k];
	}
	isl_seq_clr(c, 1 + isl_basic_map_total_dim(bmap));
	isl_int_set_si(c[0], -length);
	isl_int_set_si(c[1 + nparam + d - 1], -1);
	isl_int_set_si(c[1 + nparam + d + d - 1], 1);

	bmap = isl_basic_map_finalize(bmap);
	map = isl_map_intersect(map, isl_map_from_basic_map(bmap));

	return map;
error:
	isl_basic_map_free(bmap);
	isl_map_free(map);
	return NULL;
}

/* Check whether the overapproximation of the power of "map" is exactly
 * the power of "map".  Let R be "map" and A_k the overapproximation.
 * The approximation is exact if
 *
 *	A_1 = R
 *	A_k = A_{k-1} \circ R			k >= 2
 *
 * Since A_k is known to be an overapproximation, we only need to check
 *
 *	A_1 \subset R
 *	A_k \subset A_{k-1} \circ R		k >= 2
 *
 * In practice, "app" has an extra input and output coordinate
 * to encode the length of the path.  So, we first need to add
 * this coordinate to "map" and set the length of the path to
 * one.
 */
static int check_power_exactness(__isl_take isl_map *map,
	__isl_take isl_map *app)
{
	int exact;
	isl_map *app_1;
	isl_map *app_2;

	map = isl_map_add_dims(map, isl_dim_in, 1);
	map = isl_map_add_dims(map, isl_dim_out, 1);
	map = set_path_length(map, 1, 1);

	app_1 = set_path_length(isl_map_copy(app), 1, 1);

	exact = isl_map_is_subset(app_1, map);
	isl_map_free(app_1);

	if (!exact || exact < 0) {
		isl_map_free(app);
		isl_map_free(map);
		return exact;
	}

	app_1 = set_path_length(isl_map_copy(app), 0, 1);
	app_2 = set_path_length(app, 0, 2);
	app_1 = isl_map_apply_range(map, app_1);

	exact = isl_map_is_subset(app_2, app_1);

	isl_map_free(app_1);
	isl_map_free(app_2);

	return exact;
}

/* Check whether the overapproximation of the power of "map" is exactly
 * the power of "map", possibly after projecting out the power (if "project"
 * is set).
 *
 * If "project" is set and if "steps" can only result in acyclic paths,
 * then we check
 *
 *	A = R \cup (A \circ R)
 *
 * where A is the overapproximation with the power projected out, i.e.,
 * an overapproximation of the transitive closure.
 * More specifically, since A is known to be an overapproximation, we check
 *
 *	A \subset R \cup (A \circ R)
 *
 * Otherwise, we check if the power is exact.
 *
 * Note that "app" has an extra input and output coordinate to encode
 * the length of the part.  If we are only interested in the transitive
 * closure, then we can simply project out these coordinates first.
 */
static int check_exactness(__isl_take isl_map *map, __isl_take isl_map *app,
	int project)
{
	isl_map *test;
	int exact;
	unsigned d;

	if (!project)
		return check_power_exactness(map, app);

	d = isl_map_dim(map, isl_dim_in);
	app = set_path_length(app, 0, 1);
	app = isl_map_project_out(app, isl_dim_in, d, 1);
	app = isl_map_project_out(app, isl_dim_out, d, 1);

	app = isl_map_reset_space(app, isl_map_get_space(map));

	test = isl_map_apply_range(isl_map_copy(map), isl_map_copy(app));
	test = isl_map_union(test, isl_map_copy(map));

	exact = isl_map_is_subset(app, test);

	isl_map_free(app);
	isl_map_free(test);

	isl_map_free(map);

	return exact;
}

/*
 * The transitive closure implementation is based on the paper
 * "Computing the Transitive Closure of a Union of Affine Integer
 * Tuple Relations" by Anna Beletska, Denis Barthou, Wlodzimierz Bielecki and
 * Albert Cohen.
 */

/* Given a set of n offsets v_i (the rows of "steps"), construct a relation
 * of the given dimension specification (Z^{n+1} -> Z^{n+1})
 * that maps an element x to any element that can be reached
 * by taking a non-negative number of steps along any of
 * the extended offsets v'_i = [v_i 1].
 * That is, construct
 *
 * { [x] -> [y] : exists k_i >= 0, y = x + \sum_i k_i v'_i }
 *
 * For any element in this relation, the number of steps taken
 * is equal to the difference in the final coordinates.
 */
static __isl_give isl_map *path_along_steps(__isl_take isl_space *dim,
	__isl_keep isl_mat *steps)
{
	int i, j, k;
	struct isl_basic_map *path = NULL;
	unsigned d;
	unsigned n;
	unsigned nparam;

	if (!dim || !steps)
		goto error;

	d = isl_space_dim(dim, isl_dim_in);
	n = steps->n_row;
	nparam = isl_space_dim(dim, isl_dim_param);

	path = isl_basic_map_alloc_space(isl_space_copy(dim), n, d, n);

	for (i = 0; i < n; ++i) {
		k = isl_basic_map_alloc_div(path);
		if (k < 0)
			goto error;
		isl_assert(steps->ctx, i == k, goto error);
		isl_int_set_si(path->div[k][0], 0);
	}

	for (i = 0; i < d; ++i) {
		k = isl_basic_map_alloc_equality(path);
		if (k < 0)
			goto error;
		isl_seq_clr(path->eq[k], 1 + isl_basic_map_total_dim(path));
		isl_int_set_si(path->eq[k][1 + nparam + i], 1);
		isl_int_set_si(path->eq[k][1 + nparam + d + i], -1);
		if (i == d - 1)
			for (j = 0; j < n; ++j)
				isl_int_set_si(path->eq[k][1 + nparam + 2 * d + j], 1);
		else
			for (j = 0; j < n; ++j)
				isl_int_set(path->eq[k][1 + nparam + 2 * d + j],
					    steps->row[j][i]);
	}

	for (i = 0; i < n; ++i) {
		k = isl_basic_map_alloc_inequality(path);
		if (k < 0)
			goto error;
		isl_seq_clr(path->ineq[k], 1 + isl_basic_map_total_dim(path));
		isl_int_set_si(path->ineq[k][1 + nparam + 2 * d + i], 1);
	}

	isl_space_free(dim);

	path = isl_basic_map_simplify(path);
	path = isl_basic_map_finalize(path);
	return isl_map_from_basic_map(path);
error:
	isl_space_free(dim);
	isl_basic_map_free(path);
	return NULL;
}

#define IMPURE		0
#define PURE_PARAM	1
#define PURE_VAR	2
#define MIXED		3

/* Check whether the parametric constant term of constraint c is never
 * positive in "bset".
 */
static int parametric_constant_never_positive(__isl_keep isl_basic_set *bset,
	isl_int *c, int *div_purity)
{
	unsigned d;
	unsigned n_div;
	unsigned nparam;
	int i;
	int k;
	int empty;

	n_div = isl_basic_set_dim(bset, isl_dim_div);
	d = isl_basic_set_dim(bset, isl_dim_set);
	nparam = isl_basic_set_dim(bset, isl_dim_param);

	bset = isl_basic_set_copy(bset);
	bset = isl_basic_set_cow(bset);
	bset = isl_basic_set_extend_constraints(bset, 0, 1);
	k = isl_basic_set_alloc_inequality(bset);
	if (k < 0)
		goto error;
	isl_seq_clr(bset->ineq[k], 1 + isl_basic_set_total_dim(bset));
	isl_seq_cpy(bset->ineq[k], c, 1 + nparam);
	for (i = 0; i < n_div; ++i) {
		if (div_purity[i] != PURE_PARAM)
			continue;
		isl_int_set(bset->ineq[k][1 + nparam + d + i],
			    c[1 + nparam + d + i]);
	}
	isl_int_sub_ui(bset->ineq[k][0], bset->ineq[k][0], 1);
	empty = isl_basic_set_is_empty(bset);
	isl_basic_set_free(bset);

	return empty;
error:
	isl_basic_set_free(bset);
	return -1;
}

/* Return PURE_PARAM if only the coefficients of the parameters are non-zero.
 * Return PURE_VAR if only the coefficients of the set variables are non-zero.
 * Return MIXED if only the coefficients of the parameters and the set
 * 	variables are non-zero and if moreover the parametric constant
 * 	can never attain positive values.
 * Return IMPURE otherwise.
 */
static int purity(__isl_keep isl_basic_set *bset, isl_int *c, int *div_purity,
	int eq)
{
	unsigned d;
	unsigned n_div;
	unsigned nparam;
	int empty;
	int i;
	int p = 0, v = 0;

	n_div = isl_basic_set_dim(bset, isl_dim_div);
	d = isl_basic_set_dim(bset, isl_dim_set);
	nparam = isl_basic_set_dim(bset, isl_dim_param);

	for (i = 0; i < n_div; ++i) {
		if (isl_int_is_zero(c[1 + nparam + d + i]))
			continue;
		switch (div_purity[i]) {
		case PURE_PARAM: p = 1; break;
		case PURE_VAR: v = 1; break;
		default: return IMPURE;
		}
	}
	if (!p && isl_seq_first_non_zero(c + 1, nparam) == -1)
		return PURE_VAR;
	if (!v && isl_seq_first_non_zero(c + 1 + nparam, d) == -1)
		return PURE_PARAM;

	empty = parametric_constant_never_positive(bset, c, div_purity);
	if (eq && empty >= 0 && !empty) {
		isl_seq_neg(c, c, 1 + nparam + d + n_div);
		empty = parametric_constant_never_positive(bset, c, div_purity);
	}

	return empty < 0 ? -1 : empty ? MIXED : IMPURE;
}

/* Return an array of integers indicating the type of each div in bset.
 * If the div is (recursively) defined in terms of only the parameters,
 * then the type is PURE_PARAM.
 * If the div is (recursively) defined in terms of only the set variables,
 * then the type is PURE_VAR.
 * Otherwise, the type is IMPURE.
 */
static __isl_give int *get_div_purity(__isl_keep isl_basic_set *bset)
{
	int i, j;
	int *div_purity;
	unsigned d;
	unsigned n_div;
	unsigned nparam;

	if (!bset)
		return NULL;

	n_div = isl_basic_set_dim(bset, isl_dim_div);
	d = isl_basic_set_dim(bset, isl_dim_set);
	nparam = isl_basic_set_dim(bset, isl_dim_param);

	div_purity = isl_alloc_array(bset->ctx, int, n_div);
	if (n_div && !div_purity)
		return NULL;

	for (i = 0; i < bset->n_div; ++i) {
		int p = 0, v = 0;
		if (isl_int_is_zero(bset->div[i][0])) {
			div_purity[i] = IMPURE;
			continue;
		}
		if (isl_seq_first_non_zero(bset->div[i] + 2, nparam) != -1)
			p = 1;
		if (isl_seq_first_non_zero(bset->div[i] + 2 + nparam, d) != -1)
			v = 1;
		for (j = 0; j < i; ++j) {
			if (isl_int_is_zero(bset->div[i][2 + nparam + d + j]))
				continue;
			switch (div_purity[j]) {
			case PURE_PARAM: p = 1; break;
			case PURE_VAR: v = 1; break;
			default: p = v = 1; break;
			}
		}
		div_purity[i] = v ? p ? IMPURE : PURE_VAR : PURE_PARAM;
	}

	return div_purity;
}

/* Given a path with the as yet unconstrained length at position "pos",
 * check if setting the length to zero results in only the identity
 * mapping.
 */
static int empty_path_is_identity(__isl_keep isl_basic_map *path, unsigned pos)
{
	isl_basic_map *test = NULL;
	isl_basic_map *id = NULL;
	int k;
	int is_id;

	test = isl_basic_map_copy(path);
	test = isl_basic_map_extend_constraints(test, 1, 0);
	k = isl_basic_map_alloc_equality(test);
	if (k < 0)
		goto error;
	isl_seq_clr(test->eq[k], 1 + isl_basic_map_total_dim(test));
	isl_int_set_si(test->eq[k][pos], 1);
	test = isl_basic_map_gauss(test, NULL);
	id = isl_basic_map_identity(isl_basic_map_get_space(path));
	is_id = isl_basic_map_is_equal(test, id);
	isl_basic_map_free(test);
	isl_basic_map_free(id);
	return is_id;
error:
	isl_basic_map_free(test);
	return -1;
}

/* If any of the constraints is found to be impure then this function
 * sets *impurity to 1.
 *
 * If impurity is NULL then we are dealing with a non-parametric set
 * and so the constraints are obviously PURE_VAR.
 */
static __isl_give isl_basic_map *add_delta_constraints(
	__isl_take isl_basic_map *path,
	__isl_keep isl_basic_set *delta, unsigned off, unsigned nparam,
	unsigned d, int *div_purity, int eq, int *impurity)
{
	int i, k;
	int n = eq ? delta->n_eq : delta->n_ineq;
	isl_int **delta_c = eq ? delta->eq : delta->ineq;
	unsigned n_div;

	n_div = isl_basic_set_dim(delta, isl_dim_div);

	for (i = 0; i < n; ++i) {
		isl_int *path_c;
		int p = PURE_VAR;
		if (impurity)
			p = purity(delta, delta_c[i], div_purity, eq);
		if (p < 0)
			goto error;
		if (p != PURE_VAR && p != PURE_PARAM && !*impurity)
			*impurity = 1;
		if (p == IMPURE)
			continue;
		if (eq && p != MIXED) {
			k = isl_basic_map_alloc_equality(path);
			if (k < 0)
				goto error;
			path_c = path->eq[k];
		} else {
			k = isl_basic_map_alloc_inequality(path);
			if (k < 0)
				goto error;
			path_c = path->ineq[k];
		}
		isl_seq_clr(path_c, 1 + isl_basic_map_total_dim(path));
		if (p == PURE_VAR) {
			isl_seq_cpy(path_c + off,
				    delta_c[i] + 1 + nparam, d);
			isl_int_set(path_c[off + d], delta_c[i][0]);
		} else if (p == PURE_PARAM) {
			isl_seq_cpy(path_c, delta_c[i], 1 + nparam);
		} else {
			isl_seq_cpy(path_c + off,
				    delta_c[i] + 1 + nparam, d);
			isl_seq_cpy(path_c, delta_c[i], 1 + nparam);
		}
		isl_seq_cpy(path_c + off - n_div,
			    delta_c[i] + 1 + nparam + d, n_div);
	}

	return path;
error:
	isl_basic_map_free(path);
	return NULL;
}

/* Given a set of offsets "delta", construct a relation of the
 * given dimension specification (Z^{n+1} -> Z^{n+1}) that
 * is an overapproximation of the relations that
 * maps an element x to any element that can be reached
 * by taking a non-negative number of steps along any of
 * the elements in "delta".
 * That is, construct an approximation of
 *
 *	{ [x] -> [y] : exists f \in \delta, k \in Z :
 *					y = x + k [f, 1] and k >= 0 }
 *
 * For any element in this relation, the number of steps taken
 * is equal to the difference in the final coordinates.
 *
 * In particular, let delta be defined as
 *
 *	\delta = [p] -> { [x] : A x + a >= 0 and B p + b >= 0 and
 *				C x + C'p + c >= 0 and
 *				D x + D'p + d >= 0 }
 *
 * where the constraints C x + C'p + c >= 0 are such that the parametric
 * constant term of each constraint j, "C_j x + C'_j p + c_j",
 * can never attain positive values, then the relation is constructed as
 *
 *	{ [x] -> [y] : exists [f, k] \in Z^{n+1} : y = x + f and
 *			A f + k a >= 0 and B p + b >= 0 and
 *			C f + C'p + c >= 0 and k >= 1 }
 *	union { [x] -> [x] }
 *
 * If the zero-length paths happen to correspond exactly to the identity
 * mapping, then we return
 *
 *	{ [x] -> [y] : exists [f, k] \in Z^{n+1} : y = x + f and
 *			A f + k a >= 0 and B p + b >= 0 and
 *			C f + C'p + c >= 0 and k >= 0 }
 *
 * instead.
 *
 * Existentially quantified variables in \delta are handled by
 * classifying them as independent of the parameters, purely
 * parameter dependent and others.  Constraints containing
 * any of the other existentially quantified variables are removed.
 * This is safe, but leads to an additional overapproximation.
 *
 * If there are any impure constraints, then we also eliminate
 * the parameters from \delta, resulting in a set
 *
 *	\delta' = { [x] : E x + e >= 0 }
 *
 * and add the constraints
 *
 *			E f + k e >= 0
 *
 * to the constructed relation.
 */
static __isl_give isl_map *path_along_delta(__isl_take isl_space *dim,
	__isl_take isl_basic_set *delta)
{
	isl_basic_map *path = NULL;
	unsigned d;
	unsigned n_div;
	unsigned nparam;
	unsigned off;
	int i, k;
	int is_id;
	int *div_purity = NULL;
	int impurity = 0;

	if (!delta)
		goto error;
	n_div = isl_basic_set_dim(delta, isl_dim_div);
	d = isl_basic_set_dim(delta, isl_dim_set);
	nparam = isl_basic_set_dim(delta, isl_dim_param);
	path = isl_basic_map_alloc_space(isl_space_copy(dim), n_div + d + 1,
			d + 1 + delta->n_eq, delta->n_eq + delta->n_ineq + 1);
	off = 1 + nparam + 2 * (d + 1) + n_div;

	for (i = 0; i < n_div + d + 1; ++i) {
		k = isl_basic_map_alloc_div(path);
		if (k < 0)
			goto error;
		isl_int_set_si(path->div[k][0], 0);
	}

	for (i = 0; i < d + 1; ++i) {
		k = isl_basic_map_alloc_equality(path);
		if (k < 0)
			goto error;
		isl_seq_clr(path->eq[k], 1 + isl_basic_map_total_dim(path));
		isl_int_set_si(path->eq[k][1 + nparam + i], 1);
		isl_int_set_si(path->eq[k][1 + nparam + d + 1 + i], -1);
		isl_int_set_si(path->eq[k][off + i], 1);
	}

	div_purity = get_div_purity(delta);
	if (n_div && !div_purity)
		goto error;

	path = add_delta_constraints(path, delta, off, nparam, d,
				     div_purity, 1, &impurity);
	path = add_delta_constraints(path, delta, off, nparam, d,
				     div_purity, 0, &impurity);
	if (impurity) {
		isl_space *dim = isl_basic_set_get_space(delta);
		delta = isl_basic_set_project_out(delta,
						  isl_dim_param, 0, nparam);
		delta = isl_basic_set_add_dims(delta, isl_dim_param, nparam);
		delta = isl_basic_set_reset_space(delta, dim);
		if (!delta)
			goto error;
		path = isl_basic_map_extend_constraints(path, delta->n_eq,
							delta->n_ineq + 1);
		path = add_delta_constraints(path, delta, off, nparam, d,
					     NULL, 1, NULL);
		path = add_delta_constraints(path, delta, off, nparam, d,
					     NULL, 0, NULL);
		path = isl_basic_map_gauss(path, NULL);
	}

	is_id = empty_path_is_identity(path, off + d);
	if (is_id < 0)
		goto error;

	k = isl_basic_map_alloc_inequality(path);
	if (k < 0)
		goto error;
	isl_seq_clr(path->ineq[k], 1 + isl_basic_map_total_dim(path));
	if (!is_id)
		isl_int_set_si(path->ineq[k][0], -1);
	isl_int_set_si(path->ineq[k][off + d], 1);
			
	free(div_purity);
	isl_basic_set_free(delta);
	path = isl_basic_map_finalize(path);
	if (is_id) {
		isl_space_free(dim);
		return isl_map_from_basic_map(path);
	}
	return isl_basic_map_union(path, isl_basic_map_identity(dim));
error:
	free(div_purity);
	isl_space_free(dim);
	isl_basic_set_free(delta);
	isl_basic_map_free(path);
	return NULL;
}

/* Given a dimension specification Z^{n+1} -> Z^{n+1} and a parameter "param",
 * construct a map that equates the parameter to the difference
 * in the final coordinates and imposes that this difference is positive.
 * That is, construct
 *
 *	{ [x,x_s] -> [y,y_s] : k = y_s - x_s > 0 }
 */
static __isl_give isl_map *equate_parameter_to_length(__isl_take isl_space *dim,
	unsigned param)
{
	struct isl_basic_map *bmap;
	unsigned d;
	unsigned nparam;
	int k;

	d = isl_space_dim(dim, isl_dim_in);
	nparam = isl_space_dim(dim, isl_dim_param);
	bmap = isl_basic_map_alloc_space(dim, 0, 1, 1);
	k = isl_basic_map_alloc_equality(bmap);
	if (k < 0)
		goto error;
	isl_seq_clr(bmap->eq[k], 1 + isl_basic_map_total_dim(bmap));
	isl_int_set_si(bmap->eq[k][1 + param], -1);
	isl_int_set_si(bmap->eq[k][1 + nparam + d - 1], -1);
	isl_int_set_si(bmap->eq[k][1 + nparam + d + d - 1], 1);

	k = isl_basic_map_alloc_inequality(bmap);
	if (k < 0)
		goto error;
	isl_seq_clr(bmap->ineq[k], 1 + isl_basic_map_total_dim(bmap));
	isl_int_set_si(bmap->ineq[k][1 + param], 1);
	isl_int_set_si(bmap->ineq[k][0], -1);

	bmap = isl_basic_map_finalize(bmap);
	return isl_map_from_basic_map(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Check whether "path" is acyclic, where the last coordinates of domain
 * and range of path encode the number of steps taken.
 * That is, check whether
 *
 *	{ d | d = y - x and (x,y) in path }
 *
 * does not contain any element with positive last coordinate (positive length)
 * and zero remaining coordinates (cycle).
 */
static int is_acyclic(__isl_take isl_map *path)
{
	int i;
	int acyclic;
	unsigned dim;
	struct isl_set *delta;

	delta = isl_map_deltas(path);
	dim = isl_set_dim(delta, isl_dim_set);
	for (i = 0; i < dim; ++i) {
		if (i == dim -1)
			delta = isl_set_lower_bound_si(delta, isl_dim_set, i, 1);
		else
			delta = isl_set_fix_si(delta, isl_dim_set, i, 0);
	}

	acyclic = isl_set_is_empty(delta);
	isl_set_free(delta);

	return acyclic;
}

/* Given a union of basic maps R = \cup_i R_i \subseteq D \times D
 * and a dimension specification (Z^{n+1} -> Z^{n+1}),
 * construct a map that is an overapproximation of the map
 * that takes an element from the space D \times Z to another
 * element from the same space, such that the first n coordinates of the
 * difference between them is a sum of differences between images
 * and pre-images in one of the R_i and such that the last coordinate
 * is equal to the number of steps taken.
 * That is, let
 *
 *	\Delta_i = { y - x | (x, y) in R_i }
 *
 * then the constructed map is an overapproximation of
 *
 *	{ (x) -> (x + d) | \exists k_i >= 0, \delta_i \in \Delta_i :
 *				d = (\sum_i k_i \delta_i, \sum_i k_i) }
 *
 * The elements of the singleton \Delta_i's are collected as the
 * rows of the steps matrix.  For all these \Delta_i's together,
 * a single path is constructed.
 * For each of the other \Delta_i's, we compute an overapproximation
 * of the paths along elements of \Delta_i.
 * Since each of these paths performs an addition, composition is
 * symmetric and we can simply compose all resulting paths in any order.
 */
static __isl_give isl_map *construct_extended_path(__isl_take isl_space *dim,
	__isl_keep isl_map *map, int *project)
{
	struct isl_mat *steps = NULL;
	struct isl_map *path = NULL;
	unsigned d;
	int i, j, n;

	if (!map)
		goto error;

	d = isl_map_dim(map, isl_dim_in);

	path = isl_map_identity(isl_space_copy(dim));

	steps = isl_mat_alloc(map->ctx, map->n, d);
	if (!steps)
		goto error;

	n = 0;
	for (i = 0; i < map->n; ++i) {
		struct isl_basic_set *delta;

		delta = isl_basic_map_deltas(isl_basic_map_copy(map->p[i]));

		for (j = 0; j < d; ++j) {
			int fixed;

			fixed = isl_basic_set_plain_dim_is_fixed(delta, j,
							    &steps->row[n][j]);
			if (fixed < 0) {
				isl_basic_set_free(delta);
				goto error;
			}
			if (!fixed)
				break;
		}


		if (j < d) {
			path = isl_map_apply_range(path,
				path_along_delta(isl_space_copy(dim), delta));
			path = isl_map_coalesce(path);
		} else {
			isl_basic_set_free(delta);
			++n;
		}
	}

	if (n > 0) {
		steps->n_row = n;
		path = isl_map_apply_range(path,
				path_along_steps(isl_space_copy(dim), steps));
	}

	if (project && *project) {
		*project = is_acyclic(isl_map_copy(path));
		if (*project < 0)
			goto error;
	}

	isl_space_free(dim);
	isl_mat_free(steps);
	return path;
error:
	isl_space_free(dim);
	isl_mat_free(steps);
	isl_map_free(path);
	return NULL;
}

static int isl_set_overlaps(__isl_keep isl_set *set1, __isl_keep isl_set *set2)
{
	isl_set *i;
	int no_overlap;

	if (!set1 || !set2)
		return -1;

	if (!isl_space_tuple_is_equal(set1->dim, isl_dim_set,
					set2->dim, isl_dim_set))
		return 0;

	i = isl_set_intersect(isl_set_copy(set1), isl_set_copy(set2));
	no_overlap = isl_set_is_empty(i);
	isl_set_free(i);

	return no_overlap < 0 ? -1 : !no_overlap;
}

/* Given a union of basic maps R = \cup_i R_i \subseteq D \times D
 * and a dimension specification (Z^{n+1} -> Z^{n+1}),
 * construct a map that is an overapproximation of the map
 * that takes an element from the dom R \times Z to an
 * element from ran R \times Z, such that the first n coordinates of the
 * difference between them is a sum of differences between images
 * and pre-images in one of the R_i and such that the last coordinate
 * is equal to the number of steps taken.
 * That is, let
 *
 *	\Delta_i = { y - x | (x, y) in R_i }
 *
 * then the constructed map is an overapproximation of
 *
 *	{ (x) -> (x + d) | \exists k_i >= 0, \delta_i \in \Delta_i :
 *				d = (\sum_i k_i \delta_i, \sum_i k_i) and
 *				x in dom R and x + d in ran R and
 *				\sum_i k_i >= 1 }
 */
static __isl_give isl_map *construct_component(__isl_take isl_space *dim,
	__isl_keep isl_map *map, int *exact, int project)
{
	struct isl_set *domain = NULL;
	struct isl_set *range = NULL;
	struct isl_map *app = NULL;
	struct isl_map *path = NULL;
	int overlaps;

	domain = isl_map_domain(isl_map_copy(map));
	domain = isl_set_coalesce(domain);
	range = isl_map_range(isl_map_copy(map));
	range = isl_set_coalesce(range);
	overlaps = isl_set_overlaps(domain, range);
	if (overlaps < 0 || !overlaps) {
		isl_set_free(domain);
		isl_set_free(range);
		isl_space_free(dim);

		if (overlaps < 0)
			map = NULL;
		map = isl_map_copy(map);
		map = isl_map_add_dims(map, isl_dim_in, 1);
		map = isl_map_add_dims(map, isl_dim_out, 1);
		map = set_path_length(map, 1, 1);
		return map;
	}
	app = isl_map_from_domain_and_range(domain, range);
	app = isl_map_add_dims(app, isl_dim_in, 1);
	app = isl_map_add_dims(app, isl_dim_out, 1);

	path = construct_extended_path(isl_space_copy(dim), map,
					exact && *exact ? &project : NULL);
	app = isl_map_intersect(app, path);

	if (exact && *exact &&
	    (*exact = check_exactness(isl_map_copy(map), isl_map_copy(app),
				      project)) < 0)
		goto error;

	isl_space_free(dim);
	app = set_path_length(app, 0, 1);
	return app;
error:
	isl_space_free(dim);
	isl_map_free(app);
	return NULL;
}

/* Call construct_component and, if "project" is set, project out
 * the final coordinates.
 */
static __isl_give isl_map *construct_projected_component(
	__isl_take isl_space *dim,
	__isl_keep isl_map *map, int *exact, int project)
{
	isl_map *app;
	unsigned d;

	if (!dim)
		return NULL;
	d = isl_space_dim(dim, isl_dim_in);

	app = construct_component(dim, map, exact, project);
	if (project) {
		app = isl_map_project_out(app, isl_dim_in, d - 1, 1);
		app = isl_map_project_out(app, isl_dim_out, d - 1, 1);
	}
	return app;
}

/* Compute an extended version, i.e., with path lengths, of
 * an overapproximation of the transitive closure of "bmap"
 * with path lengths greater than or equal to zero and with
 * domain and range equal to "dom".
 */
static __isl_give isl_map *q_closure(__isl_take isl_space *dim,
	__isl_take isl_set *dom, __isl_keep isl_basic_map *bmap, int *exact)
{
	int project = 1;
	isl_map *path;
	isl_map *map;
	isl_map *app;

	dom = isl_set_add_dims(dom, isl_dim_set, 1);
	app = isl_map_from_domain_and_range(dom, isl_set_copy(dom));
	map = isl_map_from_basic_map(isl_basic_map_copy(bmap));
	path = construct_extended_path(dim, map, &project);
	app = isl_map_intersect(app, path);

	if ((*exact = check_exactness(map, isl_map_copy(app), project)) < 0)
		goto error;

	return app;
error:
	isl_map_free(app);
	return NULL;
}

/* Check whether qc has any elements of length at least one
 * with domain and/or range outside of dom and ran.
 */
static int has_spurious_elements(__isl_keep isl_map *qc,
	__isl_keep isl_set *dom, __isl_keep isl_set *ran)
{
	isl_set *s;
	int subset;
	unsigned d;

	if (!qc || !dom || !ran)
		return -1;

	d = isl_map_dim(qc, isl_dim_in);

	qc = isl_map_copy(qc);
	qc = set_path_length(qc, 0, 1);
	qc = isl_map_project_out(qc, isl_dim_in, d - 1, 1);
	qc = isl_map_project_out(qc, isl_dim_out, d - 1, 1);

	s = isl_map_domain(isl_map_copy(qc));
	subset = isl_set_is_subset(s, dom);
	isl_set_free(s);
	if (subset < 0)
		goto error;
	if (!subset) {
		isl_map_free(qc);
		return 1;
	}

	s = isl_map_range(qc);
	subset = isl_set_is_subset(s, ran);
	isl_set_free(s);

	return subset < 0 ? -1 : !subset;
error:
	isl_map_free(qc);
	return -1;
}

#define LEFT	2
#define RIGHT	1

/* For each basic map in "map", except i, check whether it combines
 * with the transitive closure that is reflexive on C combines
 * to the left and to the right.
 *
 * In particular, if
 *
 *	dom map_j \subseteq C
 *
 * then right[j] is set to 1.  Otherwise, if
 *
 *	ran map_i \cap dom map_j = \emptyset
 *
 * then right[j] is set to 0.  Otherwise, composing to the right
 * is impossible.
 *
 * Similar, for composing to the left, we have if
 *
 *	ran map_j \subseteq C
 *
 * then left[j] is set to 1.  Otherwise, if
 *
 *	dom map_i \cap ran map_j = \emptyset
 *
 * then left[j] is set to 0.  Otherwise, composing to the left
 * is impossible.
 *
 * The return value is or'd with LEFT if composing to the left
 * is possible and with RIGHT if composing to the right is possible.
 */
static int composability(__isl_keep isl_set *C, int i,
	isl_set **dom, isl_set **ran, int *left, int *right,
	__isl_keep isl_map *map)
{
	int j;
	int ok;

	ok = LEFT | RIGHT;
	for (j = 0; j < map->n && ok; ++j) {
		int overlaps, subset;
		if (j == i)
			continue;

		if (ok & RIGHT) {
			if (!dom[j])
				dom[j] = isl_set_from_basic_set(
					isl_basic_map_domain(
						isl_basic_map_copy(map->p[j])));
			if (!dom[j])
				return -1;
			overlaps = isl_set_overlaps(ran[i], dom[j]);
			if (overlaps < 0)
				return -1;
			if (!overlaps)
				right[j] = 0;
			else {
				subset = isl_set_is_subset(dom[j], C);
				if (subset < 0)
					return -1;
				if (subset)
					right[j] = 1;
				else
					ok &= ~RIGHT;
			}
		}

		if (ok & LEFT) {
			if (!ran[j])
				ran[j] = isl_set_from_basic_set(
					isl_basic_map_range(
						isl_basic_map_copy(map->p[j])));
			if (!ran[j])
				return -1;
			overlaps = isl_set_overlaps(dom[i], ran[j]);
			if (overlaps < 0)
				return -1;
			if (!overlaps)
				left[j] = 0;
			else {
				subset = isl_set_is_subset(ran[j], C);
				if (subset < 0)
					return -1;
				if (subset)
					left[j] = 1;
				else
					ok &= ~LEFT;
			}
		}
	}

	return ok;
}

static __isl_give isl_map *anonymize(__isl_take isl_map *map)
{
	map = isl_map_reset(map, isl_dim_in);
	map = isl_map_reset(map, isl_dim_out);
	return map;
}

/* Return a map that is a union of the basic maps in "map", except i,
 * composed to left and right with qc based on the entries of "left"
 * and "right".
 */
static __isl_give isl_map *compose(__isl_keep isl_map *map, int i,
	__isl_take isl_map *qc, int *left, int *right)
{
	int j;
	isl_map *comp;

	comp = isl_map_empty(isl_map_get_space(map));
	for (j = 0; j < map->n; ++j) {
		isl_map *map_j;

		if (j == i)
			continue;

		map_j = isl_map_from_basic_map(isl_basic_map_copy(map->p[j]));
		map_j = anonymize(map_j);
		if (left && left[j])
			map_j = isl_map_apply_range(map_j, isl_map_copy(qc));
		if (right && right[j])
			map_j = isl_map_apply_range(isl_map_copy(qc), map_j);
		comp = isl_map_union(comp, map_j);
	}

	comp = isl_map_compute_divs(comp);
	comp = isl_map_coalesce(comp);

	isl_map_free(qc);

	return comp;
}

/* Compute the transitive closure of "map" incrementally by
 * computing
 *
 *	map_i^+ \cup qc^+
 *
 * or
 *
 *	map_i^+ \cup ((id \cup map_i^) \circ qc^+)
 *
 * or
 *
 *	map_i^+ \cup (qc^+ \circ (id \cup map_i^))
 *
 * depending on whether left or right are NULL.
 */
static __isl_give isl_map *compute_incremental(
	__isl_take isl_space *dim, __isl_keep isl_map *map,
	int i, __isl_take isl_map *qc, int *left, int *right, int *exact)
{
	isl_map *map_i;
	isl_map *tc;
	isl_map *rtc = NULL;

	if (!map)
		goto error;
	isl_assert(map->ctx, left || right, goto error);

	map_i = isl_map_from_basic_map(isl_basic_map_copy(map->p[i]));
	tc = construct_projected_component(isl_space_copy(dim), map_i,
						exact, 1);
	isl_map_free(map_i);

	if (*exact)
		qc = isl_map_transitive_closure(qc, exact);

	if (!*exact) {
		isl_space_free(dim);
		isl_map_free(tc);
		isl_map_free(qc);
		return isl_map_universe(isl_map_get_space(map));
	}

	if (!left || !right)
		rtc = isl_map_union(isl_map_copy(tc),
				    isl_map_identity(isl_map_get_space(tc)));
	if (!right)
		qc = isl_map_apply_range(rtc, qc);
	if (!left)
		qc = isl_map_apply_range(qc, rtc);
	qc = isl_map_union(tc, qc);

	isl_space_free(dim);

	return qc;
error:
	isl_space_free(dim);
	isl_map_free(qc);
	return NULL;
}

/* Given a map "map", try to find a basic map such that
 * map^+ can be computed as
 *
 * map^+ = map_i^+ \cup
 *    \bigcup_j ((map_i^+ \cup Id_C)^+ \circ map_j \circ (map_i^+ \cup Id_C))^+
 *
 * with C the simple hull of the domain and range of the input map.
 * map_i^ \cup Id_C is computed by allowing the path lengths to be zero
 * and by intersecting domain and range with C.
 * Of course, we need to check that this is actually equal to map_i^ \cup Id_C.
 * Also, we only use the incremental computation if all the transitive
 * closures are exact and if the number of basic maps in the union,
 * after computing the integer divisions, is smaller than the number
 * of basic maps in the input map.
 */
static int incemental_on_entire_domain(__isl_keep isl_space *dim,
	__isl_keep isl_map *map,
	isl_set **dom, isl_set **ran, int *left, int *right,
	__isl_give isl_map **res)
{
	int i;
	isl_set *C;
	unsigned d;

	*res = NULL;

	C = isl_set_union(isl_map_domain(isl_map_copy(map)),
			  isl_map_range(isl_map_copy(map)));
	C = isl_set_from_basic_set(isl_set_simple_hull(C));
	if (!C)
		return -1;
	if (C->n != 1) {
		isl_set_free(C);
		return 0;
	}

	d = isl_map_dim(map, isl_dim_in);

	for (i = 0; i < map->n; ++i) {
		isl_map *qc;
		int exact_i, spurious;
		int j;
		dom[i] = isl_set_from_basic_set(isl_basic_map_domain(
					isl_basic_map_copy(map->p[i])));
		ran[i] = isl_set_from_basic_set(isl_basic_map_range(
					isl_basic_map_copy(map->p[i])));
		qc = q_closure(isl_space_copy(dim), isl_set_copy(C),
				map->p[i], &exact_i);
		if (!qc)
			goto error;
		if (!exact_i) {
			isl_map_free(qc);
			continue;
		}
		spurious = has_spurious_elements(qc, dom[i], ran[i]);
		if (spurious) {
			isl_map_free(qc);
			if (spurious < 0)
				goto error;
			continue;
		}
		qc = isl_map_project_out(qc, isl_dim_in, d, 1);
		qc = isl_map_project_out(qc, isl_dim_out, d, 1);
		qc = isl_map_compute_divs(qc);
		for (j = 0; j < map->n; ++j)
			left[j] = right[j] = 1;
		qc = compose(map, i, qc, left, right);
		if (!qc)
			goto error;
		if (qc->n >= map->n) {
			isl_map_free(qc);
			continue;
		}
		*res = compute_incremental(isl_space_copy(dim), map, i, qc,
				left, right, &exact_i);
		if (!*res)
			goto error;
		if (exact_i)
			break;
		isl_map_free(*res);
		*res = NULL;
	}

	isl_set_free(C);

	return *res != NULL;
error:
	isl_set_free(C);
	return -1;
}

/* Try and compute the transitive closure of "map" as
 *
 * map^+ = map_i^+ \cup
 *    \bigcup_j ((map_i^+ \cup Id_C)^+ \circ map_j \circ (map_i^+ \cup Id_C))^+
 *
 * with C either the simple hull of the domain and range of the entire
 * map or the simple hull of domain and range of map_i.
 */
static __isl_give isl_map *incremental_closure(__isl_take isl_space *dim,
	__isl_keep isl_map *map, int *exact, int project)
{
	int i;
	isl_set **dom = NULL;
	isl_set **ran = NULL;
	int *left = NULL;
	int *right = NULL;
	isl_set *C;
	unsigned d;
	isl_map *res = NULL;

	if (!project)
		return construct_projected_component(dim, map, exact, project);

	if (!map)
		goto error;
	if (map->n <= 1)
		return construct_projected_component(dim, map, exact, project);

	d = isl_map_dim(map, isl_dim_in);

	dom = isl_calloc_array(map->ctx, isl_set *, map->n);
	ran = isl_calloc_array(map->ctx, isl_set *, map->n);
	left = isl_calloc_array(map->ctx, int, map->n);
	right = isl_calloc_array(map->ctx, int, map->n);
	if (!ran || !dom || !left || !right)
		goto error;

	if (incemental_on_entire_domain(dim, map, dom, ran, left, right, &res) < 0)
		goto error;

	for (i = 0; !res && i < map->n; ++i) {
		isl_map *qc;
		int exact_i, spurious, comp;
		if (!dom[i])
			dom[i] = isl_set_from_basic_set(
					isl_basic_map_domain(
						isl_basic_map_copy(map->p[i])));
		if (!dom[i])
			goto error;
		if (!ran[i])
			ran[i] = isl_set_from_basic_set(
					isl_basic_map_range(
						isl_basic_map_copy(map->p[i])));
		if (!ran[i])
			goto error;
		C = isl_set_union(isl_set_copy(dom[i]),
				      isl_set_copy(ran[i]));
		C = isl_set_from_basic_set(isl_set_simple_hull(C));
		if (!C)
			goto error;
		if (C->n != 1) {
			isl_set_free(C);
			continue;
		}
		comp = composability(C, i, dom, ran, left, right, map);
		if (!comp || comp < 0) {
			isl_set_free(C);
			if (comp < 0)
				goto error;
			continue;
		}
		qc = q_closure(isl_space_copy(dim), C, map->p[i], &exact_i);
		if (!qc)
			goto error;
		if (!exact_i) {
			isl_map_free(qc);
			continue;
		}
		spurious = has_spurious_elements(qc, dom[i], ran[i]);
		if (spurious) {
			isl_map_free(qc);
			if (spurious < 0)
				goto error;
			continue;
		}
		qc = isl_map_project_out(qc, isl_dim_in, d, 1);
		qc = isl_map_project_out(qc, isl_dim_out, d, 1);
		qc = isl_map_compute_divs(qc);
		qc = compose(map, i, qc, (comp & LEFT) ? left : NULL,
				(comp & RIGHT) ? right : NULL);
		if (!qc)
			goto error;
		if (qc->n >= map->n) {
			isl_map_free(qc);
			continue;
		}
		res = compute_incremental(isl_space_copy(dim), map, i, qc,
				(comp & LEFT) ? left : NULL,
				(comp & RIGHT) ? right : NULL, &exact_i);
		if (!res)
			goto error;
		if (exact_i)
			break;
		isl_map_free(res);
		res = NULL;
	}

	for (i = 0; i < map->n; ++i) {
		isl_set_free(dom[i]);
		isl_set_free(ran[i]);
	}
	free(dom);
	free(ran);
	free(left);
	free(right);

	if (res) {
		isl_space_free(dim);
		return res;
	}

	return construct_projected_component(dim, map, exact, project);
error:
	if (dom)
		for (i = 0; i < map->n; ++i)
			isl_set_free(dom[i]);
	free(dom);
	if (ran)
		for (i = 0; i < map->n; ++i)
			isl_set_free(ran[i]);
	free(ran);
	free(left);
	free(right);
	isl_space_free(dim);
	return NULL;
}

/* Given an array of sets "set", add "dom" at position "pos"
 * and search for elements at earlier positions that overlap with "dom".
 * If any can be found, then merge all of them, together with "dom", into
 * a single set and assign the union to the first in the array,
 * which becomes the new group leader for all groups involved in the merge.
 * During the search, we only consider group leaders, i.e., those with
 * group[i] = i, as the other sets have already been combined
 * with one of the group leaders.
 */
static int merge(isl_set **set, int *group, __isl_take isl_set *dom, int pos)
{
	int i;

	group[pos] = pos;
	set[pos] = isl_set_copy(dom);

	for (i = pos - 1; i >= 0; --i) {
		int o;

		if (group[i] != i)
			continue;

		o = isl_set_overlaps(set[i], dom);
		if (o < 0)
			goto error;
		if (!o)
			continue;

		set[i] = isl_set_union(set[i], set[group[pos]]);
		set[group[pos]] = NULL;
		if (!set[i])
			goto error;
		group[group[pos]] = i;
		group[pos] = i;
	}

	isl_set_free(dom);
	return 0;
error:
	isl_set_free(dom);
	return -1;
}

/* Replace each entry in the n by n grid of maps by the cross product
 * with the relation { [i] -> [i + 1] }.
 */
static int add_length(__isl_keep isl_map *map, isl_map ***grid, int n)
{
	int i, j, k;
	isl_space *dim;
	isl_basic_map *bstep;
	isl_map *step;
	unsigned nparam;

	if (!map)
		return -1;

	dim = isl_map_get_space(map);
	nparam = isl_space_dim(dim, isl_dim_param);
	dim = isl_space_drop_dims(dim, isl_dim_in, 0, isl_space_dim(dim, isl_dim_in));
	dim = isl_space_drop_dims(dim, isl_dim_out, 0, isl_space_dim(dim, isl_dim_out));
	dim = isl_space_add_dims(dim, isl_dim_in, 1);
	dim = isl_space_add_dims(dim, isl_dim_out, 1);
	bstep = isl_basic_map_alloc_space(dim, 0, 1, 0);
	k = isl_basic_map_alloc_equality(bstep);
	if (k < 0) {
		isl_basic_map_free(bstep);
		return -1;
	}
	isl_seq_clr(bstep->eq[k], 1 + isl_basic_map_total_dim(bstep));
	isl_int_set_si(bstep->eq[k][0], 1);
	isl_int_set_si(bstep->eq[k][1 + nparam], 1);
	isl_int_set_si(bstep->eq[k][1 + nparam + 1], -1);
	bstep = isl_basic_map_finalize(bstep);
	step = isl_map_from_basic_map(bstep);

	for (i = 0; i < n; ++i)
		for (j = 0; j < n; ++j)
			grid[i][j] = isl_map_product(grid[i][j],
						     isl_map_copy(step));

	isl_map_free(step);

	return 0;
}

/* The core of the Floyd-Warshall algorithm.
 * Updates the given n x x matrix of relations in place.
 *
 * The algorithm iterates over all vertices.  In each step, the whole
 * matrix is updated to include all paths that go to the current vertex,
 * possibly stay there a while (including passing through earlier vertices)
 * and then come back.  At the start of each iteration, the diagonal
 * element corresponding to the current vertex is replaced by its
 * transitive closure to account for all indirect paths that stay
 * in the current vertex.
 */
static void floyd_warshall_iterate(isl_map ***grid, int n, int *exact)
{
	int r, p, q;

	for (r = 0; r < n; ++r) {
		int r_exact;
		grid[r][r] = isl_map_transitive_closure(grid[r][r],
				(exact && *exact) ? &r_exact : NULL);
		if (exact && *exact && !r_exact)
			*exact = 0;

		for (p = 0; p < n; ++p)
			for (q = 0; q < n; ++q) {
				isl_map *loop;
				if (p == r && q == r)
					continue;
				loop = isl_map_apply_range(
						isl_map_copy(grid[p][r]),
						isl_map_copy(grid[r][q]));
				grid[p][q] = isl_map_union(grid[p][q], loop);
				loop = isl_map_apply_range(
						isl_map_copy(grid[p][r]),
					isl_map_apply_range(
						isl_map_copy(grid[r][r]),
						isl_map_copy(grid[r][q])));
				grid[p][q] = isl_map_union(grid[p][q], loop);
				grid[p][q] = isl_map_coalesce(grid[p][q]);
			}
	}
}

/* Given a partition of the domains and ranges of the basic maps in "map",
 * apply the Floyd-Warshall algorithm with the elements in the partition
 * as vertices.
 *
 * In particular, there are "n" elements in the partition and "group" is
 * an array of length 2 * map->n with entries in [0,n-1].
 *
 * We first construct a matrix of relations based on the partition information,
 * apply Floyd-Warshall on this matrix of relations and then take the
 * union of all entries in the matrix as the final result.
 *
 * If we are actually computing the power instead of the transitive closure,
 * i.e., when "project" is not set, then the result should have the
 * path lengths encoded as the difference between an extra pair of
 * coordinates.  We therefore apply the nested transitive closures
 * to relations that include these lengths.  In particular, we replace
 * the input relation by the cross product with the unit length relation
 * { [i] -> [i + 1] }.
 */
static __isl_give isl_map *floyd_warshall_with_groups(__isl_take isl_space *dim,
	__isl_keep isl_map *map, int *exact, int project, int *group, int n)
{
	int i, j, k;
	isl_map ***grid = NULL;
	isl_map *app;

	if (!map)
		goto error;

	if (n == 1) {
		free(group);
		return incremental_closure(dim, map, exact, project);
	}

	grid = isl_calloc_array(map->ctx, isl_map **, n);
	if (!grid)
		goto error;
	for (i = 0; i < n; ++i) {
		grid[i] = isl_calloc_array(map->ctx, isl_map *, n);
		if (!grid[i])
			goto error;
		for (j = 0; j < n; ++j)
			grid[i][j] = isl_map_empty(isl_map_get_space(map));
	}

	for (k = 0; k < map->n; ++k) {
		i = group[2 * k];
		j = group[2 * k + 1];
		grid[i][j] = isl_map_union(grid[i][j],
				isl_map_from_basic_map(
					isl_basic_map_copy(map->p[k])));
	}

	if (!project && add_length(map, grid, n) < 0)
		goto error;

	floyd_warshall_iterate(grid, n, exact);

	app = isl_map_empty(isl_map_get_space(grid[0][0]));

	for (i = 0; i < n; ++i) {
		for (j = 0; j < n; ++j)
			app = isl_map_union(app, grid[i][j]);
		free(grid[i]);
	}
	free(grid);

	free(group);
	isl_space_free(dim);

	return app;
error:
	if (grid)
		for (i = 0; i < n; ++i) {
			if (!grid[i])
				continue;
			for (j = 0; j < n; ++j)
				isl_map_free(grid[i][j]);
			free(grid[i]);
		}
	free(grid);
	free(group);
	isl_space_free(dim);
	return NULL;
}

/* Partition the domains and ranges of the n basic relations in list
 * into disjoint cells.
 *
 * To find the partition, we simply consider all of the domains
 * and ranges in turn and combine those that overlap.
 * "set" contains the partition elements and "group" indicates
 * to which partition element a given domain or range belongs.
 * The domain of basic map i corresponds to element 2 * i in these arrays,
 * while the domain corresponds to element 2 * i + 1.
 * During the construction group[k] is either equal to k,
 * in which case set[k] contains the union of all the domains and
 * ranges in the corresponding group, or is equal to some l < k,
 * with l another domain or range in the same group.
 */
static int *setup_groups(isl_ctx *ctx, __isl_keep isl_basic_map **list, int n,
	isl_set ***set, int *n_group)
{
	int i;
	int *group = NULL;
	int g;

	*set = isl_calloc_array(ctx, isl_set *, 2 * n);
	group = isl_alloc_array(ctx, int, 2 * n);

	if (!*set || !group)
		goto error;

	for (i = 0; i < n; ++i) {
		isl_set *dom;
		dom = isl_set_from_basic_set(isl_basic_map_domain(
				isl_basic_map_copy(list[i])));
		if (merge(*set, group, dom, 2 * i) < 0)
			goto error;
		dom = isl_set_from_basic_set(isl_basic_map_range(
				isl_basic_map_copy(list[i])));
		if (merge(*set, group, dom, 2 * i + 1) < 0)
			goto error;
	}

	g = 0;
	for (i = 0; i < 2 * n; ++i)
		if (group[i] == i) {
			if (g != i) {
				(*set)[g] = (*set)[i];
				(*set)[i] = NULL;
			}
			group[i] = g++;
		} else
			group[i] = group[group[i]];

	*n_group = g;

	return group;
error:
	if (*set) {
		for (i = 0; i < 2 * n; ++i)
			isl_set_free((*set)[i]);
		free(*set);
		*set = NULL;
	}
	free(group);
	return NULL;
}

/* Check if the domains and ranges of the basic maps in "map" can
 * be partitioned, and if so, apply Floyd-Warshall on the elements
 * of the partition.  Note that we also apply this algorithm
 * if we want to compute the power, i.e., when "project" is not set.
 * However, the results are unlikely to be exact since the recursive
 * calls inside the Floyd-Warshall algorithm typically result in
 * non-linear path lengths quite quickly.
 */
static __isl_give isl_map *floyd_warshall(__isl_take isl_space *dim,
	__isl_keep isl_map *map, int *exact, int project)
{
	int i;
	isl_set **set = NULL;
	int *group = NULL;
	int n;

	if (!map)
		goto error;
	if (map->n <= 1)
		return incremental_closure(dim, map, exact, project);

	group = setup_groups(map->ctx, map->p, map->n, &set, &n);
	if (!group)
		goto error;

	for (i = 0; i < 2 * map->n; ++i)
		isl_set_free(set[i]);

	free(set);

	return floyd_warshall_with_groups(dim, map, exact, project, group, n);
error:
	isl_space_free(dim);
	return NULL;
}

/* Structure for representing the nodes of the graph of which
 * strongly connected components are being computed.
 *
 * list contains the actual nodes
 * check_closed is set if we may have used the fact that
 * a pair of basic maps can be interchanged
 */
struct isl_tc_follows_data {
	isl_basic_map **list;
	int check_closed;
};

/* Check whether in the computation of the transitive closure
 * "list[i]" (R_1) should follow (or be part of the same component as)
 * "list[j]" (R_2).
 *
 * That is check whether
 *
 *	R_1 \circ R_2
 *
 * is a subset of
 *
 *	R_2 \circ R_1
 *
 * If so, then there is no reason for R_1 to immediately follow R_2
 * in any path.
 *
 * *check_closed is set if the subset relation holds while
 * R_1 \circ R_2 is not empty.
 */
static isl_bool basic_map_follows(int i, int j, void *user)
{
	struct isl_tc_follows_data *data = user;
	struct isl_map *map12 = NULL;
	struct isl_map *map21 = NULL;
	isl_bool subset;

	if (!isl_space_tuple_is_equal(data->list[i]->dim, isl_dim_in,
				    data->list[j]->dim, isl_dim_out))
		return isl_bool_false;

	map21 = isl_map_from_basic_map(
			isl_basic_map_apply_range(
				isl_basic_map_copy(data->list[j]),
				isl_basic_map_copy(data->list[i])));
	subset = isl_map_is_empty(map21);
	if (subset < 0)
		goto error;
	if (subset) {
		isl_map_free(map21);
		return isl_bool_false;
	}

	if (!isl_space_tuple_is_equal(data->list[i]->dim, isl_dim_in,
				    data->list[i]->dim, isl_dim_out) ||
	    !isl_space_tuple_is_equal(data->list[j]->dim, isl_dim_in,
				    data->list[j]->dim, isl_dim_out)) {
		isl_map_free(map21);
		return isl_bool_true;
	}

	map12 = isl_map_from_basic_map(
			isl_basic_map_apply_range(
				isl_basic_map_copy(data->list[i]),
				isl_basic_map_copy(data->list[j])));

	subset = isl_map_is_subset(map21, map12);

	isl_map_free(map12);
	isl_map_free(map21);

	if (subset)
		data->check_closed = 1;

	return subset < 0 ? isl_bool_error : !subset;
error:
	isl_map_free(map21);
	return isl_bool_error;
}

/* Given a union of basic maps R = \cup_i R_i \subseteq D \times D
 * and a dimension specification (Z^{n+1} -> Z^{n+1}),
 * construct a map that is an overapproximation of the map
 * that takes an element from the dom R \times Z to an
 * element from ran R \times Z, such that the first n coordinates of the
 * difference between them is a sum of differences between images
 * and pre-images in one of the R_i and such that the last coordinate
 * is equal to the number of steps taken.
 * If "project" is set, then these final coordinates are not included,
 * i.e., a relation of type Z^n -> Z^n is returned.
 * That is, let
 *
 *	\Delta_i = { y - x | (x, y) in R_i }
 *
 * then the constructed map is an overapproximation of
 *
 *	{ (x) -> (x + d) | \exists k_i >= 0, \delta_i \in \Delta_i :
 *				d = (\sum_i k_i \delta_i, \sum_i k_i) and
 *				x in dom R and x + d in ran R }
 *
 * or
 *
 *	{ (x) -> (x + d) | \exists k_i >= 0, \delta_i \in \Delta_i :
 *				d = (\sum_i k_i \delta_i) and
 *				x in dom R and x + d in ran R }
 *
 * if "project" is set.
 *
 * We first split the map into strongly connected components, perform
 * the above on each component and then join the results in the correct
 * order, at each join also taking in the union of both arguments
 * to allow for paths that do not go through one of the two arguments.
 */
static __isl_give isl_map *construct_power_components(__isl_take isl_space *dim,
	__isl_keep isl_map *map, int *exact, int project)
{
	int i, n, c;
	struct isl_map *path = NULL;
	struct isl_tc_follows_data data;
	struct isl_tarjan_graph *g = NULL;
	int *orig_exact;
	int local_exact;

	if (!map)
		goto error;
	if (map->n <= 1)
		return floyd_warshall(dim, map, exact, project);

	data.list = map->p;
	data.check_closed = 0;
	g = isl_tarjan_graph_init(map->ctx, map->n, &basic_map_follows, &data);
	if (!g)
		goto error;

	orig_exact = exact;
	if (data.check_closed && !exact)
		exact = &local_exact;

	c = 0;
	i = 0;
	n = map->n;
	if (project)
		path = isl_map_empty(isl_map_get_space(map));
	else
		path = isl_map_empty(isl_space_copy(dim));
	path = anonymize(path);
	while (n) {
		struct isl_map *comp;
		isl_map *path_comp, *path_comb;
		comp = isl_map_alloc_space(isl_map_get_space(map), n, 0);
		while (g->order[i] != -1) {
			comp = isl_map_add_basic_map(comp,
				    isl_basic_map_copy(map->p[g->order[i]]));
			--n;
			++i;
		}
		path_comp = floyd_warshall(isl_space_copy(dim),
						comp, exact, project);
		path_comp = anonymize(path_comp);
		path_comb = isl_map_apply_range(isl_map_copy(path),
						isl_map_copy(path_comp));
		path = isl_map_union(path, path_comp);
		path = isl_map_union(path, path_comb);
		isl_map_free(comp);
		++i;
		++c;
	}

	if (c > 1 && data.check_closed && !*exact) {
		int closed;

		closed = isl_map_is_transitively_closed(path);
		if (closed < 0)
			goto error;
		if (!closed) {
			isl_tarjan_graph_free(g);
			isl_map_free(path);
			return floyd_warshall(dim, map, orig_exact, project);
		}
	}

	isl_tarjan_graph_free(g);
	isl_space_free(dim);

	return path;
error:
	isl_tarjan_graph_free(g);
	isl_space_free(dim);
	isl_map_free(path);
	return NULL;
}

/* Given a union of basic maps R = \cup_i R_i \subseteq D \times D,
 * construct a map that is an overapproximation of the map
 * that takes an element from the space D to another
 * element from the same space, such that the difference between
 * them is a strictly positive sum of differences between images
 * and pre-images in one of the R_i.
 * The number of differences in the sum is equated to parameter "param".
 * That is, let
 *
 *	\Delta_i = { y - x | (x, y) in R_i }
 *
 * then the constructed map is an overapproximation of
 *
 *	{ (x) -> (x + d) | \exists k_i >= 0, \delta_i \in \Delta_i :
 *				d = \sum_i k_i \delta_i and k = \sum_i k_i > 0 }
 * or
 *
 *	{ (x) -> (x + d) | \exists k_i >= 0, \delta_i \in \Delta_i :
 *				d = \sum_i k_i \delta_i and \sum_i k_i > 0 }
 *
 * if "project" is set.
 *
 * If "project" is not set, then
 * we construct an extended mapping with an extra coordinate
 * that indicates the number of steps taken.  In particular,
 * the difference in the last coordinate is equal to the number
 * of steps taken to move from a domain element to the corresponding
 * image element(s).
 */
static __isl_give isl_map *construct_power(__isl_keep isl_map *map,
	int *exact, int project)
{
	struct isl_map *app = NULL;
	isl_space *dim = NULL;

	if (!map)
		return NULL;

	dim = isl_map_get_space(map);

	dim = isl_space_add_dims(dim, isl_dim_in, 1);
	dim = isl_space_add_dims(dim, isl_dim_out, 1);

	app = construct_power_components(isl_space_copy(dim), map,
					exact, project);

	isl_space_free(dim);

	return app;
}

/* Compute the positive powers of "map", or an overapproximation.
 * If the result is exact, then *exact is set to 1.
 *
 * If project is set, then we are actually interested in the transitive
 * closure, so we can use a more relaxed exactness check.
 * The lengths of the paths are also projected out instead of being
 * encoded as the difference between an extra pair of final coordinates.
 */
static __isl_give isl_map *map_power(__isl_take isl_map *map,
	int *exact, int project)
{
	struct isl_map *app = NULL;

	if (exact)
		*exact = 1;

	if (!map)
		return NULL;

	isl_assert(map->ctx,
		isl_map_dim(map, isl_dim_in) == isl_map_dim(map, isl_dim_out),
		goto error);

	app = construct_power(map, exact, project);

	isl_map_free(map);
	return app;
error:
	isl_map_free(map);
	isl_map_free(app);
	return NULL;
}

/* Compute the positive powers of "map", or an overapproximation.
 * The result maps the exponent to a nested copy of the corresponding power.
 * If the result is exact, then *exact is set to 1.
 * map_power constructs an extended relation with the path lengths
 * encoded as the difference between the final coordinates.
 * In the final step, this difference is equated to an extra parameter
 * and made positive.  The extra coordinates are subsequently projected out
 * and the parameter is turned into the domain of the result.
 */
__isl_give isl_map *isl_map_power(__isl_take isl_map *map, int *exact)
{
	isl_space *target_dim;
	isl_space *dim;
	isl_map *diff;
	unsigned d;
	unsigned param;

	if (!map)
		return NULL;

	d = isl_map_dim(map, isl_dim_in);
	param = isl_map_dim(map, isl_dim_param);

	map = isl_map_compute_divs(map);
	map = isl_map_coalesce(map);

	if (isl_map_plain_is_empty(map)) {
		map = isl_map_from_range(isl_map_wrap(map));
		map = isl_map_add_dims(map, isl_dim_in, 1);
		map = isl_map_set_dim_name(map, isl_dim_in, 0, "k");
		return map;
	}

	target_dim = isl_map_get_space(map);
	target_dim = isl_space_from_range(isl_space_wrap(target_dim));
	target_dim = isl_space_add_dims(target_dim, isl_dim_in, 1);
	target_dim = isl_space_set_dim_name(target_dim, isl_dim_in, 0, "k");

	map = map_power(map, exact, 0);

	map = isl_map_add_dims(map, isl_dim_param, 1);
	dim = isl_map_get_space(map);
	diff = equate_parameter_to_length(dim, param);
	map = isl_map_intersect(map, diff);
	map = isl_map_project_out(map, isl_dim_in, d, 1);
	map = isl_map_project_out(map, isl_dim_out, d, 1);
	map = isl_map_from_range(isl_map_wrap(map));
	map = isl_map_move_dims(map, isl_dim_in, 0, isl_dim_param, param, 1);

	map = isl_map_reset_space(map, target_dim);

	return map;
}

/* Compute a relation that maps each element in the range of the input
 * relation to the lengths of all paths composed of edges in the input
 * relation that end up in the given range element.
 * The result may be an overapproximation, in which case *exact is set to 0.
 * The resulting relation is very similar to the power relation.
 * The difference are that the domain has been projected out, the
 * range has become the domain and the exponent is the range instead
 * of a parameter.
 */
__isl_give isl_map *isl_map_reaching_path_lengths(__isl_take isl_map *map,
	int *exact)
{
	isl_space *dim;
	isl_map *diff;
	unsigned d;
	unsigned param;

	if (!map)
		return NULL;

	d = isl_map_dim(map, isl_dim_in);
	param = isl_map_dim(map, isl_dim_param);

	map = isl_map_compute_divs(map);
	map = isl_map_coalesce(map);

	if (isl_map_plain_is_empty(map)) {
		if (exact)
			*exact = 1;
		map = isl_map_project_out(map, isl_dim_out, 0, d);
		map = isl_map_add_dims(map, isl_dim_out, 1);
		return map;
	}

	map = map_power(map, exact, 0);

	map = isl_map_add_dims(map, isl_dim_param, 1);
	dim = isl_map_get_space(map);
	diff = equate_parameter_to_length(dim, param);
	map = isl_map_intersect(map, diff);
	map = isl_map_project_out(map, isl_dim_in, 0, d + 1);
	map = isl_map_project_out(map, isl_dim_out, d, 1);
	map = isl_map_reverse(map);
	map = isl_map_move_dims(map, isl_dim_out, 0, isl_dim_param, param, 1);

	return map;
}

/* Check whether equality i of bset is a pure stride constraint
 * on a single dimension, i.e., of the form
 *
 *	v = k e
 *
 * with k a constant and e an existentially quantified variable.
 */
static int is_eq_stride(__isl_keep isl_basic_set *bset, int i)
{
	unsigned nparam;
	unsigned d;
	unsigned n_div;
	int pos1;
	int pos2;

	if (!bset)
		return -1;

	if (!isl_int_is_zero(bset->eq[i][0]))
		return 0;

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	d = isl_basic_set_dim(bset, isl_dim_set);
	n_div = isl_basic_set_dim(bset, isl_dim_div);

	if (isl_seq_first_non_zero(bset->eq[i] + 1, nparam) != -1)
		return 0;
	pos1 = isl_seq_first_non_zero(bset->eq[i] + 1 + nparam, d);
	if (pos1 == -1)
		return 0;
	if (isl_seq_first_non_zero(bset->eq[i] + 1 + nparam + pos1 + 1, 
					d - pos1 - 1) != -1)
		return 0;

	pos2 = isl_seq_first_non_zero(bset->eq[i] + 1 + nparam + d, n_div);
	if (pos2 == -1)
		return 0;
	if (isl_seq_first_non_zero(bset->eq[i] + 1 + nparam + d  + pos2 + 1,
				   n_div - pos2 - 1) != -1)
		return 0;
	if (!isl_int_is_one(bset->eq[i][1 + nparam + pos1]) &&
	    !isl_int_is_negone(bset->eq[i][1 + nparam + pos1]))
		return 0;

	return 1;
}

/* Given a map, compute the smallest superset of this map that is of the form
 *
 *	{ i -> j : L <= j - i <= U and exists a_p: j_p - i_p = M_p a_p }
 *
 * (where p ranges over the (non-parametric) dimensions),
 * compute the transitive closure of this map, i.e.,
 *
 *	{ i -> j : exists k > 0:
 *		k L <= j - i <= k U and exists a: j_p - i_p = M_p a_p }
 *
 * and intersect domain and range of this transitive closure with
 * the given domain and range.
 *
 * If with_id is set, then try to include as much of the identity mapping
 * as possible, by computing
 *
 *	{ i -> j : exists k >= 0:
 *		k L <= j - i <= k U and exists a: j_p - i_p = M_p a_p }
 *
 * instead (i.e., allow k = 0).
 *
 * In practice, we compute the difference set
 *
 *	delta  = { j - i | i -> j in map },
 *
 * look for stride constraint on the individual dimensions and compute
 * (constant) lower and upper bounds for each individual dimension,
 * adding a constraint for each bound not equal to infinity.
 */
static __isl_give isl_map *box_closure_on_domain(__isl_take isl_map *map,
	__isl_take isl_set *dom, __isl_take isl_set *ran, int with_id)
{
	int i;
	int k;
	unsigned d;
	unsigned nparam;
	unsigned total;
	isl_space *dim;
	isl_set *delta;
	isl_map *app = NULL;
	isl_basic_set *aff = NULL;
	isl_basic_map *bmap = NULL;
	isl_vec *obj = NULL;
	isl_int opt;

	isl_int_init(opt);

	delta = isl_map_deltas(isl_map_copy(map));

	aff = isl_set_affine_hull(isl_set_copy(delta));
	if (!aff)
		goto error;
	dim = isl_map_get_space(map);
	d = isl_space_dim(dim, isl_dim_in);
	nparam = isl_space_dim(dim, isl_dim_param);
	total = isl_space_dim(dim, isl_dim_all);
	bmap = isl_basic_map_alloc_space(dim,
					aff->n_div + 1, aff->n_div, 2 * d + 1);
	for (i = 0; i < aff->n_div + 1; ++i) {
		k = isl_basic_map_alloc_div(bmap);
		if (k < 0)
			goto error;
		isl_int_set_si(bmap->div[k][0], 0);
	}
	for (i = 0; i < aff->n_eq; ++i) {
		if (!is_eq_stride(aff, i))
			continue;
		k = isl_basic_map_alloc_equality(bmap);
		if (k < 0)
			goto error;
		isl_seq_clr(bmap->eq[k], 1 + nparam);
		isl_seq_cpy(bmap->eq[k] + 1 + nparam + d,
				aff->eq[i] + 1 + nparam, d);
		isl_seq_neg(bmap->eq[k] + 1 + nparam,
				aff->eq[i] + 1 + nparam, d);
		isl_seq_cpy(bmap->eq[k] + 1 + nparam + 2 * d,
				aff->eq[i] + 1 + nparam + d, aff->n_div);
		isl_int_set_si(bmap->eq[k][1 + total + aff->n_div], 0);
	}
	obj = isl_vec_alloc(map->ctx, 1 + nparam + d);
	if (!obj)
		goto error;
	isl_seq_clr(obj->el, 1 + nparam + d);
	for (i = 0; i < d; ++ i) {
		enum isl_lp_result res;

		isl_int_set_si(obj->el[1 + nparam + i], 1);

		res = isl_set_solve_lp(delta, 0, obj->el, map->ctx->one, &opt,
					NULL, NULL);
		if (res == isl_lp_error)
			goto error;
		if (res == isl_lp_ok) {
			k = isl_basic_map_alloc_inequality(bmap);
			if (k < 0)
				goto error;
			isl_seq_clr(bmap->ineq[k],
					1 + nparam + 2 * d + bmap->n_div);
			isl_int_set_si(bmap->ineq[k][1 + nparam + i], -1);
			isl_int_set_si(bmap->ineq[k][1 + nparam + d + i], 1);
			isl_int_neg(bmap->ineq[k][1 + nparam + 2 * d + aff->n_div], opt);
		}

		res = isl_set_solve_lp(delta, 1, obj->el, map->ctx->one, &opt,
					NULL, NULL);
		if (res == isl_lp_error)
			goto error;
		if (res == isl_lp_ok) {
			k = isl_basic_map_alloc_inequality(bmap);
			if (k < 0)
				goto error;
			isl_seq_clr(bmap->ineq[k],
					1 + nparam + 2 * d + bmap->n_div);
			isl_int_set_si(bmap->ineq[k][1 + nparam + i], 1);
			isl_int_set_si(bmap->ineq[k][1 + nparam + d + i], -1);
			isl_int_set(bmap->ineq[k][1 + nparam + 2 * d + aff->n_div], opt);
		}

		isl_int_set_si(obj->el[1 + nparam + i], 0);
	}
	k = isl_basic_map_alloc_inequality(bmap);
	if (k < 0)
		goto error;
	isl_seq_clr(bmap->ineq[k],
			1 + nparam + 2 * d + bmap->n_div);
	if (!with_id)
		isl_int_set_si(bmap->ineq[k][0], -1);
	isl_int_set_si(bmap->ineq[k][1 + nparam + 2 * d + aff->n_div], 1);

	app = isl_map_from_domain_and_range(dom, ran);

	isl_vec_free(obj);
	isl_basic_set_free(aff);
	isl_map_free(map);
	bmap = isl_basic_map_finalize(bmap);
	isl_set_free(delta);
	isl_int_clear(opt);

	map = isl_map_from_basic_map(bmap);
	map = isl_map_intersect(map, app);

	return map;
error:
	isl_vec_free(obj);
	isl_basic_map_free(bmap);
	isl_basic_set_free(aff);
	isl_set_free(dom);
	isl_set_free(ran);
	isl_map_free(map);
	isl_set_free(delta);
	isl_int_clear(opt);
	return NULL;
}

/* Given a map, compute the smallest superset of this map that is of the form
 *
 *	{ i -> j : L <= j - i <= U and exists a_p: j_p - i_p = M_p a_p }
 *
 * (where p ranges over the (non-parametric) dimensions),
 * compute the transitive closure of this map, i.e.,
 *
 *	{ i -> j : exists k > 0:
 *		k L <= j - i <= k U and exists a: j_p - i_p = M_p a_p }
 *
 * and intersect domain and range of this transitive closure with
 * domain and range of the original map.
 */
static __isl_give isl_map *box_closure(__isl_take isl_map *map)
{
	isl_set *domain;
	isl_set *range;

	domain = isl_map_domain(isl_map_copy(map));
	domain = isl_set_coalesce(domain);
	range = isl_map_range(isl_map_copy(map));
	range = isl_set_coalesce(range);

	return box_closure_on_domain(map, domain, range, 0);
}

/* Given a map, compute the smallest superset of this map that is of the form
 *
 *	{ i -> j : L <= j - i <= U and exists a_p: j_p - i_p = M_p a_p }
 *
 * (where p ranges over the (non-parametric) dimensions),
 * compute the transitive and partially reflexive closure of this map, i.e.,
 *
 *	{ i -> j : exists k >= 0:
 *		k L <= j - i <= k U and exists a: j_p - i_p = M_p a_p }
 *
 * and intersect domain and range of this transitive closure with
 * the given domain.
 */
static __isl_give isl_map *box_closure_with_identity(__isl_take isl_map *map,
	__isl_take isl_set *dom)
{
	return box_closure_on_domain(map, dom, isl_set_copy(dom), 1);
}

/* Check whether app is the transitive closure of map.
 * In particular, check that app is acyclic and, if so,
 * check that
 *
 *	app \subset (map \cup (map \circ app))
 */
static int check_exactness_omega(__isl_keep isl_map *map,
	__isl_keep isl_map *app)
{
	isl_set *delta;
	int i;
	int is_empty, is_exact;
	unsigned d;
	isl_map *test;

	delta = isl_map_deltas(isl_map_copy(app));
	d = isl_set_dim(delta, isl_dim_set);
	for (i = 0; i < d; ++i)
		delta = isl_set_fix_si(delta, isl_dim_set, i, 0);
	is_empty = isl_set_is_empty(delta);
	isl_set_free(delta);
	if (is_empty < 0)
		return -1;
	if (!is_empty)
		return 0;

	test = isl_map_apply_range(isl_map_copy(app), isl_map_copy(map));
	test = isl_map_union(test, isl_map_copy(map));
	is_exact = isl_map_is_subset(app, test);
	isl_map_free(test);

	return is_exact;
}

/* Check if basic map M_i can be combined with all the other
 * basic maps such that
 *
 *	(\cup_j M_j)^+
 *
 * can be computed as
 *
 *	M_i \cup (\cup_{j \ne i} M_i^* \circ M_j \circ M_i^*)^+
 *
 * In particular, check if we can compute a compact representation
 * of
 *
 *		M_i^* \circ M_j \circ M_i^*
 *
 * for each j != i.
 * Let M_i^? be an extension of M_i^+ that allows paths
 * of length zero, i.e., the result of box_closure(., 1).
 * The criterion, as proposed by Kelly et al., is that
 * id = M_i^? - M_i^+ can be represented as a basic map
 * and that
 *
 *	id \circ M_j \circ id = M_j
 *
 * for each j != i.
 *
 * If this function returns 1, then tc and qc are set to
 * M_i^+ and M_i^?, respectively.
 */
static int can_be_split_off(__isl_keep isl_map *map, int i,
	__isl_give isl_map **tc, __isl_give isl_map **qc)
{
	isl_map *map_i, *id = NULL;
	int j = -1;
	isl_set *C;

	*tc = NULL;
	*qc = NULL;

	C = isl_set_union(isl_map_domain(isl_map_copy(map)),
			  isl_map_range(isl_map_copy(map)));
	C = isl_set_from_basic_set(isl_set_simple_hull(C));
	if (!C)
		goto error;

	map_i = isl_map_from_basic_map(isl_basic_map_copy(map->p[i]));
	*tc = box_closure(isl_map_copy(map_i));
	*qc = box_closure_with_identity(map_i, C);
	id = isl_map_subtract(isl_map_copy(*qc), isl_map_copy(*tc));

	if (!id || !*qc)
		goto error;
	if (id->n != 1 || (*qc)->n != 1)
		goto done;

	for (j = 0; j < map->n; ++j) {
		isl_map *map_j, *test;
		int is_ok;

		if (i == j)
			continue;
		map_j = isl_map_from_basic_map(
					isl_basic_map_copy(map->p[j]));
		test = isl_map_apply_range(isl_map_copy(id),
						isl_map_copy(map_j));
		test = isl_map_apply_range(test, isl_map_copy(id));
		is_ok = isl_map_is_equal(test, map_j);
		isl_map_free(map_j);
		isl_map_free(test);
		if (is_ok < 0)
			goto error;
		if (!is_ok)
			break;
	}

done:
	isl_map_free(id);
	if (j == map->n)
		return 1;

	isl_map_free(*qc);
	isl_map_free(*tc);
	*qc = NULL;
	*tc = NULL;

	return 0;
error:
	isl_map_free(id);
	isl_map_free(*qc);
	isl_map_free(*tc);
	*qc = NULL;
	*tc = NULL;
	return -1;
}

static __isl_give isl_map *box_closure_with_check(__isl_take isl_map *map,
	int *exact)
{
	isl_map *app;

	app = box_closure(isl_map_copy(map));
	if (exact)
		*exact = check_exactness_omega(map, app);

	isl_map_free(map);
	return app;
}

/* Compute an overapproximation of the transitive closure of "map"
 * using a variation of the algorithm from
 * "Transitive Closure of Infinite Graphs and its Applications"
 * by Kelly et al.
 *
 * We first check whether we can can split of any basic map M_i and
 * compute
 *
 *	(\cup_j M_j)^+
 *
 * as
 *
 *	M_i \cup (\cup_{j \ne i} M_i^* \circ M_j \circ M_i^*)^+
 *
 * using a recursive call on the remaining map.
 *
 * If not, we simply call box_closure on the whole map.
 */
static __isl_give isl_map *transitive_closure_omega(__isl_take isl_map *map,
	int *exact)
{
	int i, j;
	int exact_i;
	isl_map *app;

	if (!map)
		return NULL;
	if (map->n == 1)
		return box_closure_with_check(map, exact);

	for (i = 0; i < map->n; ++i) {
		int ok;
		isl_map *qc, *tc;
		ok = can_be_split_off(map, i, &tc, &qc);
		if (ok < 0)
			goto error;
		if (!ok)
			continue;

		app = isl_map_alloc_space(isl_map_get_space(map), map->n - 1, 0);

		for (j = 0; j < map->n; ++j) {
			if (j == i)
				continue;
			app = isl_map_add_basic_map(app,
						isl_basic_map_copy(map->p[j]));
		}

		app = isl_map_apply_range(isl_map_copy(qc), app);
		app = isl_map_apply_range(app, qc);

		app = isl_map_union(tc, transitive_closure_omega(app, NULL));
		exact_i = check_exactness_omega(map, app);
		if (exact_i == 1) {
			if (exact)
				*exact = exact_i;
			isl_map_free(map);
			return app;
		}
		isl_map_free(app);
		if (exact_i < 0)
			goto error;
	}

	return box_closure_with_check(map, exact);
error:
	isl_map_free(map);
	return NULL;
}

/* Compute the transitive closure  of "map", or an overapproximation.
 * If the result is exact, then *exact is set to 1.
 * Simply use map_power to compute the powers of map, but tell
 * it to project out the lengths of the paths instead of equating
 * the length to a parameter.
 */
__isl_give isl_map *isl_map_transitive_closure(__isl_take isl_map *map,
	int *exact)
{
	isl_space *target_dim;
	int closed;

	if (!map)
		goto error;

	if (map->ctx->opt->closure == ISL_CLOSURE_BOX)
		return transitive_closure_omega(map, exact);

	map = isl_map_compute_divs(map);
	map = isl_map_coalesce(map);
	closed = isl_map_is_transitively_closed(map);
	if (closed < 0)
		goto error;
	if (closed) {
		if (exact)
			*exact = 1;
		return map;
	}

	target_dim = isl_map_get_space(map);
	map = map_power(map, exact, 1);
	map = isl_map_reset_space(map, target_dim);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

static isl_stat inc_count(__isl_take isl_map *map, void *user)
{
	int *n = user;

	*n += map->n;

	isl_map_free(map);

	return isl_stat_ok;
}

static isl_stat collect_basic_map(__isl_take isl_map *map, void *user)
{
	int i;
	isl_basic_map ***next = user;

	for (i = 0; i < map->n; ++i) {
		**next = isl_basic_map_copy(map->p[i]);
		if (!**next)
			goto error;
		(*next)++;
	}

	isl_map_free(map);
	return isl_stat_ok;
error:
	isl_map_free(map);
	return isl_stat_error;
}

/* Perform Floyd-Warshall on the given list of basic relations.
 * The basic relations may live in different dimensions,
 * but basic relations that get assigned to the diagonal of the
 * grid have domains and ranges of the same dimension and so
 * the standard algorithm can be used because the nested transitive
 * closures are only applied to diagonal elements and because all
 * compositions are peformed on relations with compatible domains and ranges.
 */
static __isl_give isl_union_map *union_floyd_warshall_on_list(isl_ctx *ctx,
	__isl_keep isl_basic_map **list, int n, int *exact)
{
	int i, j, k;
	int n_group;
	int *group = NULL;
	isl_set **set = NULL;
	isl_map ***grid = NULL;
	isl_union_map *app;

	group = setup_groups(ctx, list, n, &set, &n_group);
	if (!group)
		goto error;

	grid = isl_calloc_array(ctx, isl_map **, n_group);
	if (!grid)
		goto error;
	for (i = 0; i < n_group; ++i) {
		grid[i] = isl_calloc_array(ctx, isl_map *, n_group);
		if (!grid[i])
			goto error;
		for (j = 0; j < n_group; ++j) {
			isl_space *dim1, *dim2, *dim;
			dim1 = isl_space_reverse(isl_set_get_space(set[i]));
			dim2 = isl_set_get_space(set[j]);
			dim = isl_space_join(dim1, dim2);
			grid[i][j] = isl_map_empty(dim);
		}
	}

	for (k = 0; k < n; ++k) {
		i = group[2 * k];
		j = group[2 * k + 1];
		grid[i][j] = isl_map_union(grid[i][j],
				isl_map_from_basic_map(
					isl_basic_map_copy(list[k])));
	}
	
	floyd_warshall_iterate(grid, n_group, exact);

	app = isl_union_map_empty(isl_map_get_space(grid[0][0]));

	for (i = 0; i < n_group; ++i) {
		for (j = 0; j < n_group; ++j)
			app = isl_union_map_add_map(app, grid[i][j]);
		free(grid[i]);
	}
	free(grid);

	for (i = 0; i < 2 * n; ++i)
		isl_set_free(set[i]);
	free(set);

	free(group);
	return app;
error:
	if (grid)
		for (i = 0; i < n_group; ++i) {
			if (!grid[i])
				continue;
			for (j = 0; j < n_group; ++j)
				isl_map_free(grid[i][j]);
			free(grid[i]);
		}
	free(grid);
	if (set) {
		for (i = 0; i < 2 * n; ++i)
			isl_set_free(set[i]);
		free(set);
	}
	free(group);
	return NULL;
}

/* Perform Floyd-Warshall on the given union relation.
 * The implementation is very similar to that for non-unions.
 * The main difference is that it is applied unconditionally.
 * We first extract a list of basic maps from the union map
 * and then perform the algorithm on this list.
 */
static __isl_give isl_union_map *union_floyd_warshall(
	__isl_take isl_union_map *umap, int *exact)
{
	int i, n;
	isl_ctx *ctx;
	isl_basic_map **list = NULL;
	isl_basic_map **next;
	isl_union_map *res;

	n = 0;
	if (isl_union_map_foreach_map(umap, inc_count, &n) < 0)
		goto error;

	ctx = isl_union_map_get_ctx(umap);
	list = isl_calloc_array(ctx, isl_basic_map *, n);
	if (!list)
		goto error;

	next = list;
	if (isl_union_map_foreach_map(umap, collect_basic_map, &next) < 0)
		goto error;

	res = union_floyd_warshall_on_list(ctx, list, n, exact);

	if (list) {
		for (i = 0; i < n; ++i)
			isl_basic_map_free(list[i]);
		free(list);
	}

	isl_union_map_free(umap);
	return res;
error:
	if (list) {
		for (i = 0; i < n; ++i)
			isl_basic_map_free(list[i]);
		free(list);
	}
	isl_union_map_free(umap);
	return NULL;
}

/* Decompose the give union relation into strongly connected components.
 * The implementation is essentially the same as that of
 * construct_power_components with the major difference that all
 * operations are performed on union maps.
 */
static __isl_give isl_union_map *union_components(
	__isl_take isl_union_map *umap, int *exact)
{
	int i;
	int n;
	isl_ctx *ctx;
	isl_basic_map **list = NULL;
	isl_basic_map **next;
	isl_union_map *path = NULL;
	struct isl_tc_follows_data data;
	struct isl_tarjan_graph *g = NULL;
	int c, l;
	int recheck = 0;

	n = 0;
	if (isl_union_map_foreach_map(umap, inc_count, &n) < 0)
		goto error;

	if (n == 0)
		return umap;
	if (n <= 1)
		return union_floyd_warshall(umap, exact);

	ctx = isl_union_map_get_ctx(umap);
	list = isl_calloc_array(ctx, isl_basic_map *, n);
	if (!list)
		goto error;

	next = list;
	if (isl_union_map_foreach_map(umap, collect_basic_map, &next) < 0)
		goto error;

	data.list = list;
	data.check_closed = 0;
	g = isl_tarjan_graph_init(ctx, n, &basic_map_follows, &data);
	if (!g)
		goto error;

	c = 0;
	i = 0;
	l = n;
	path = isl_union_map_empty(isl_union_map_get_space(umap));
	while (l) {
		isl_union_map *comp;
		isl_union_map *path_comp, *path_comb;
		comp = isl_union_map_empty(isl_union_map_get_space(umap));
		while (g->order[i] != -1) {
			comp = isl_union_map_add_map(comp,
				    isl_map_from_basic_map(
					isl_basic_map_copy(list[g->order[i]])));
			--l;
			++i;
		}
		path_comp = union_floyd_warshall(comp, exact);
		path_comb = isl_union_map_apply_range(isl_union_map_copy(path),
						isl_union_map_copy(path_comp));
		path = isl_union_map_union(path, path_comp);
		path = isl_union_map_union(path, path_comb);
		++i;
		++c;
	}

	if (c > 1 && data.check_closed && !*exact) {
		int closed;

		closed = isl_union_map_is_transitively_closed(path);
		if (closed < 0)
			goto error;
		recheck = !closed;
	}

	isl_tarjan_graph_free(g);

	for (i = 0; i < n; ++i)
		isl_basic_map_free(list[i]);
	free(list);

	if (recheck) {
		isl_union_map_free(path);
		return union_floyd_warshall(umap, exact);
	}

	isl_union_map_free(umap);

	return path;
error:
	isl_tarjan_graph_free(g);
	if (list) {
		for (i = 0; i < n; ++i)
			isl_basic_map_free(list[i]);
		free(list);
	}
	isl_union_map_free(umap);
	isl_union_map_free(path);
	return NULL;
}

/* Compute the transitive closure  of "umap", or an overapproximation.
 * If the result is exact, then *exact is set to 1.
 */
__isl_give isl_union_map *isl_union_map_transitive_closure(
	__isl_take isl_union_map *umap, int *exact)
{
	int closed;

	if (!umap)
		return NULL;

	if (exact)
		*exact = 1;

	umap = isl_union_map_compute_divs(umap);
	umap = isl_union_map_coalesce(umap);
	closed = isl_union_map_is_transitively_closed(umap);
	if (closed < 0)
		goto error;
	if (closed)
		return umap;
	umap = union_components(umap, exact);
	return umap;
error:
	isl_union_map_free(umap);
	return NULL;
}

struct isl_union_power {
	isl_union_map *pow;
	int *exact;
};

static isl_stat power(__isl_take isl_map *map, void *user)
{
	struct isl_union_power *up = user;

	map = isl_map_power(map, up->exact);
	up->pow = isl_union_map_from_map(map);

	return isl_stat_error;
}

/* Construct a map [x] -> [x+1], with parameters prescribed by "dim".
 */
static __isl_give isl_union_map *increment(__isl_take isl_space *dim)
{
	int k;
	isl_basic_map *bmap;

	dim = isl_space_add_dims(dim, isl_dim_in, 1);
	dim = isl_space_add_dims(dim, isl_dim_out, 1);
	bmap = isl_basic_map_alloc_space(dim, 0, 1, 0);
	k = isl_basic_map_alloc_equality(bmap);
	if (k < 0)
		goto error;
	isl_seq_clr(bmap->eq[k], isl_basic_map_total_dim(bmap));
	isl_int_set_si(bmap->eq[k][0], 1);
	isl_int_set_si(bmap->eq[k][isl_basic_map_offset(bmap, isl_dim_in)], 1);
	isl_int_set_si(bmap->eq[k][isl_basic_map_offset(bmap, isl_dim_out)], -1);
	return isl_union_map_from_map(isl_map_from_basic_map(bmap));
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Construct a map [[x]->[y]] -> [y-x], with parameters prescribed by "dim".
 */
static __isl_give isl_union_map *deltas_map(__isl_take isl_space *dim)
{
	isl_basic_map *bmap;

	dim = isl_space_add_dims(dim, isl_dim_in, 1);
	dim = isl_space_add_dims(dim, isl_dim_out, 1);
	bmap = isl_basic_map_universe(dim);
	bmap = isl_basic_map_deltas_map(bmap);

	return isl_union_map_from_map(isl_map_from_basic_map(bmap));
}

/* Compute the positive powers of "map", or an overapproximation.
 * The result maps the exponent to a nested copy of the corresponding power.
 * If the result is exact, then *exact is set to 1.
 */
__isl_give isl_union_map *isl_union_map_power(__isl_take isl_union_map *umap,
	int *exact)
{
	int n;
	isl_union_map *inc;
	isl_union_map *dm;

	if (!umap)
		return NULL;
	n = isl_union_map_n_map(umap);
	if (n == 0)
		return umap;
	if (n == 1) {
		struct isl_union_power up = { NULL, exact };
		isl_union_map_foreach_map(umap, &power, &up);
		isl_union_map_free(umap);
		return up.pow;
	}
	inc = increment(isl_union_map_get_space(umap));
	umap = isl_union_map_product(inc, umap);
	umap = isl_union_map_transitive_closure(umap, exact);
	umap = isl_union_map_zip(umap);
	dm = deltas_map(isl_union_map_get_space(umap));
	umap = isl_union_map_apply_domain(umap, dm);
	
	return umap;
}

#undef TYPE
#define TYPE isl_map
#include "isl_power_templ.c"

#undef TYPE
#define TYPE isl_union_map
#include "isl_power_templ.c"
