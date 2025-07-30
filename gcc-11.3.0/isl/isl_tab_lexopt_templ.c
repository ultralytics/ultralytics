/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2011      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 */

#define xSF(TYPE,SUFFIX) TYPE ## SUFFIX
#define SF(TYPE,SUFFIX) xSF(TYPE,SUFFIX)

/* Given a basic map with at least two parallel constraints (as found
 * by the function parallel_constraints), first look for more constraints
 * parallel to the two constraint and replace the found list of parallel
 * constraints by a single constraint with as "input" part the minimum
 * of the input parts of the list of constraints.  Then, recursively call
 * basic_map_partial_lexopt (possibly finding more parallel constraints)
 * and plug in the definition of the minimum in the result.
 *
 * As in parallel_constraints, only inequality constraints that only
 * involve input variables that do not occur in any other inequality
 * constraints are considered.
 *
 * More specifically, given a set of constraints
 *
 *	a x + b_i(p) >= 0
 *
 * Replace this set by a single constraint
 *
 *	a x + u >= 0
 *
 * with u a new parameter with constraints
 *
 *	u <= b_i(p)
 *
 * Any solution to the new system is also a solution for the original system
 * since
 *
 *	a x >= -u >= -b_i(p)
 *
 * Moreover, m = min_i(b_i(p)) satisfies the constraints on u and can
 * therefore be plugged into the solution.
 */
static TYPE *SF(basic_map_partial_lexopt_symm,SUFFIX)(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, int max, int first, int second)
{
	int i, n, k;
	int *list = NULL;
	unsigned n_in, n_out, n_div;
	isl_ctx *ctx;
	isl_vec *var = NULL;
	isl_mat *cst = NULL;
	isl_space *map_space, *set_space;

	map_space = isl_basic_map_get_space(bmap);
	set_space = empty ? isl_basic_set_get_space(dom) : NULL;

	n_in = isl_basic_map_dim(bmap, isl_dim_param) +
	       isl_basic_map_dim(bmap, isl_dim_in);
	n_out = isl_basic_map_dim(bmap, isl_dim_all) - n_in;

	ctx = isl_basic_map_get_ctx(bmap);
	list = isl_alloc_array(ctx, int, bmap->n_ineq);
	var = isl_vec_alloc(ctx, n_out);
	if ((bmap->n_ineq && !list) || (n_out && !var))
		goto error;

	list[0] = first;
	list[1] = second;
	isl_seq_cpy(var->el, bmap->ineq[first] + 1 + n_in, n_out);
	for (i = second + 1, n = 2; i < bmap->n_ineq; ++i) {
		if (isl_seq_eq(var->el, bmap->ineq[i] + 1 + n_in, n_out) &&
		    all_single_occurrence(bmap, i, n_in))
			list[n++] = i;
	}

	cst = isl_mat_alloc(ctx, n, 1 + n_in);
	if (!cst)
		goto error;

	for (i = 0; i < n; ++i)
		isl_seq_cpy(cst->row[i], bmap->ineq[list[i]], 1 + n_in);

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		goto error;
	for (i = n - 1; i >= 0; --i)
		if (isl_basic_map_drop_inequality(bmap, list[i]) < 0)
			goto error;

	bmap = isl_basic_map_add_dims(bmap, isl_dim_in, 1);
	bmap = isl_basic_map_extend_constraints(bmap, 0, 1);
	k = isl_basic_map_alloc_inequality(bmap);
	if (k < 0)
		goto error;
	isl_seq_clr(bmap->ineq[k], 1 + n_in);
	isl_int_set_si(bmap->ineq[k][1 + n_in], 1);
	isl_seq_cpy(bmap->ineq[k] + 1 + n_in + 1, var->el, n_out);
	bmap = isl_basic_map_finalize(bmap);

	n_div = isl_basic_set_dim(dom, isl_dim_div);
	dom = isl_basic_set_add_dims(dom, isl_dim_set, 1);
	dom = isl_basic_set_extend_constraints(dom, 0, n);
	for (i = 0; i < n; ++i) {
		k = isl_basic_set_alloc_inequality(dom);
		if (k < 0)
			goto error;
		isl_seq_cpy(dom->ineq[k], cst->row[i], 1 + n_in);
		isl_int_set_si(dom->ineq[k][1 + n_in], -1);
		isl_seq_clr(dom->ineq[k] + 1 + n_in + 1, n_div);
	}

	isl_vec_free(var);
	free(list);

	return SF(basic_map_partial_lexopt_symm_core,SUFFIX)(bmap, dom, empty,
						max, cst, map_space, set_space);
error:
	isl_space_free(map_space);
	isl_space_free(set_space);
	isl_mat_free(cst);
	isl_vec_free(var);
	free(list);
	isl_basic_set_free(dom);
	isl_basic_map_free(bmap);
	return NULL;
}

/* Recursive part of isl_tab_basic_map_partial_lexopt*, after detecting
 * equalities and removing redundant constraints.
 *
 * We first check if there are any parallel constraints (left).
 * If not, we are in the base case.
 * If there are parallel constraints, we replace them by a single
 * constraint in basic_map_partial_lexopt_symm_pma and then call
 * this function recursively to look for more parallel constraints.
 */
static __isl_give TYPE *SF(basic_map_partial_lexopt,SUFFIX)(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, int max)
{
	int par = 0;
	int first, second;

	if (!bmap)
		goto error;

	if (bmap->ctx->opt->pip_symmetry)
		par = parallel_constraints(bmap, &first, &second);
	if (par < 0)
		goto error;
	if (!par)
		return SF(basic_map_partial_lexopt_base,SUFFIX)(bmap, dom,
								empty, max);

	return SF(basic_map_partial_lexopt_symm,SUFFIX)(bmap, dom, empty, max,
							 first, second);
error:
	isl_basic_set_free(dom);
	isl_basic_map_free(bmap);
	return NULL;
}

/* Compute the lexicographic minimum (or maximum if "flags" includes
 * ISL_OPT_MAX) of "bmap" over the domain "dom" and return the result as
 * either a map or a piecewise multi-affine expression depending on TYPE.
 * If "empty" is not NULL, then *empty is assigned a set that
 * contains those parts of the domain where there is no solution.
 * If "flags" includes ISL_OPT_FULL, then "dom" is NULL and the optimum
 * should be computed over the domain of "bmap".  "empty" is also NULL
 * in this case.
 * If "bmap" is marked as rational (ISL_BASIC_MAP_RATIONAL),
 * then we compute the rational optimum.  Otherwise, we compute
 * the integral optimum.
 *
 * We perform some preprocessing.  As the PILP solver does not
 * handle implicit equalities very well, we first make sure all
 * the equalities are explicitly available.
 *
 * We also add context constraints to the basic map and remove
 * redundant constraints.  This is only needed because of the
 * way we handle simple symmetries.  In particular, we currently look
 * for symmetries on the constraints, before we set up the main tableau.
 * It is then no good to look for symmetries on possibly redundant constraints.
 * If the domain was extracted from the basic map, then there is
 * no need to add back those constraints again.
 */
__isl_give TYPE *SF(isl_tab_basic_map_partial_lexopt,SUFFIX)(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, unsigned flags)
{
	int max, full;
	isl_bool compatible;

	if (empty)
		*empty = NULL;

	full = ISL_FL_ISSET(flags, ISL_OPT_FULL);
	if (full)
		dom = extract_domain(bmap, flags);
	compatible = isl_basic_map_compatible_domain(bmap, dom);
	if (compatible < 0)
		goto error;
	if (!compatible)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"domain does not match input", goto error);

	max = ISL_FL_ISSET(flags, ISL_OPT_MAX);
	if (isl_basic_set_dim(dom, isl_dim_all) == 0)
		return SF(basic_map_partial_lexopt,SUFFIX)(bmap, dom, empty,
							    max);

	if (!full)
		bmap = isl_basic_map_intersect_domain(bmap,
						    isl_basic_set_copy(dom));
	bmap = isl_basic_map_detect_equalities(bmap);
	bmap = isl_basic_map_remove_redundancies(bmap);

	return SF(basic_map_partial_lexopt,SUFFIX)(bmap, dom, empty, max);
error:
	isl_basic_set_free(dom);
	isl_basic_map_free(bmap);
	return NULL;
}
