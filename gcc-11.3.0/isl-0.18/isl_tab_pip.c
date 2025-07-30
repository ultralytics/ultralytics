/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France 
 */

#include <isl_ctx_private.h>
#include "isl_map_private.h"
#include <isl_seq.h>
#include "isl_tab.h"
#include "isl_sample.h"
#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl_aff_private.h>
#include <isl_constraint_private.h>
#include <isl_options_private.h>
#include <isl_config.h>

#include <bset_to_bmap.c>

/*
 * The implementation of parametric integer linear programming in this file
 * was inspired by the paper "Parametric Integer Programming" and the
 * report "Solving systems of affine (in)equalities" by Paul Feautrier
 * (and others).
 *
 * The strategy used for obtaining a feasible solution is different
 * from the one used in isl_tab.c.  In particular, in isl_tab.c,
 * upon finding a constraint that is not yet satisfied, we pivot
 * in a row that increases the constant term of the row holding the
 * constraint, making sure the sample solution remains feasible
 * for all the constraints it already satisfied.
 * Here, we always pivot in the row holding the constraint,
 * choosing a column that induces the lexicographically smallest
 * increment to the sample solution.
 *
 * By starting out from a sample value that is lexicographically
 * smaller than any integer point in the problem space, the first
 * feasible integer sample point we find will also be the lexicographically
 * smallest.  If all variables can be assumed to be non-negative,
 * then the initial sample value may be chosen equal to zero.
 * However, we will not make this assumption.  Instead, we apply
 * the "big parameter" trick.  Any variable x is then not directly
 * used in the tableau, but instead it is represented by another
 * variable x' = M + x, where M is an arbitrarily large (positive)
 * value.  x' is therefore always non-negative, whatever the value of x.
 * Taking as initial sample value x' = 0 corresponds to x = -M,
 * which is always smaller than any possible value of x.
 *
 * The big parameter trick is used in the main tableau and
 * also in the context tableau if isl_context_lex is used.
 * In this case, each tableaus has its own big parameter.
 * Before doing any real work, we check if all the parameters
 * happen to be non-negative.  If so, we drop the column corresponding
 * to M from the initial context tableau.
 * If isl_context_gbr is used, then the big parameter trick is only
 * used in the main tableau.
 */

struct isl_context;
struct isl_context_op {
	/* detect nonnegative parameters in context and mark them in tab */
	struct isl_tab *(*detect_nonnegative_parameters)(
			struct isl_context *context, struct isl_tab *tab);
	/* return temporary reference to basic set representation of context */
	struct isl_basic_set *(*peek_basic_set)(struct isl_context *context);
	/* return temporary reference to tableau representation of context */
	struct isl_tab *(*peek_tab)(struct isl_context *context);
	/* add equality; check is 1 if eq may not be valid;
	 * update is 1 if we may want to call ineq_sign on context later.
	 */
	void (*add_eq)(struct isl_context *context, isl_int *eq,
			int check, int update);
	/* add inequality; check is 1 if ineq may not be valid;
	 * update is 1 if we may want to call ineq_sign on context later.
	 */
	void (*add_ineq)(struct isl_context *context, isl_int *ineq,
			int check, int update);
	/* check sign of ineq based on previous information.
	 * strict is 1 if saturation should be treated as a positive sign.
	 */
	enum isl_tab_row_sign (*ineq_sign)(struct isl_context *context,
			isl_int *ineq, int strict);
	/* check if inequality maintains feasibility */
	int (*test_ineq)(struct isl_context *context, isl_int *ineq);
	/* return index of a div that corresponds to "div" */
	int (*get_div)(struct isl_context *context, struct isl_tab *tab,
			struct isl_vec *div);
	/* insert div "div" to context at "pos" and return non-negativity */
	isl_bool (*insert_div)(struct isl_context *context, int pos,
		__isl_keep isl_vec *div);
	int (*detect_equalities)(struct isl_context *context,
			struct isl_tab *tab);
	/* return row index of "best" split */
	int (*best_split)(struct isl_context *context, struct isl_tab *tab);
	/* check if context has already been determined to be empty */
	int (*is_empty)(struct isl_context *context);
	/* check if context is still usable */
	int (*is_ok)(struct isl_context *context);
	/* save a copy/snapshot of context */
	void *(*save)(struct isl_context *context);
	/* restore saved context */
	void (*restore)(struct isl_context *context, void *);
	/* discard saved context */
	void (*discard)(void *);
	/* invalidate context */
	void (*invalidate)(struct isl_context *context);
	/* free context */
	__isl_null struct isl_context *(*free)(struct isl_context *context);
};

/* Shared parts of context representation.
 *
 * "n_unknown" is the number of final unknown integer divisions
 * in the input domain.
 */
struct isl_context {
	struct isl_context_op *op;
	int n_unknown;
};

struct isl_context_lex {
	struct isl_context context;
	struct isl_tab *tab;
};

/* A stack (linked list) of solutions of subtrees of the search space.
 *
 * "M" describes the solution in terms of the dimensions of "dom".
 * The number of columns of "M" is one more than the total number
 * of dimensions of "dom".
 *
 * If "M" is NULL, then there is no solution on "dom".
 */
struct isl_partial_sol {
	int level;
	struct isl_basic_set *dom;
	struct isl_mat *M;

	struct isl_partial_sol *next;
};

struct isl_sol;
struct isl_sol_callback {
	struct isl_tab_callback callback;
	struct isl_sol *sol;
};

/* isl_sol is an interface for constructing a solution to
 * a parametric integer linear programming problem.
 * Every time the algorithm reaches a state where a solution
 * can be read off from the tableau (including cases where the tableau
 * is empty), the function "add" is called on the isl_sol passed
 * to find_solutions_main.
 *
 * The context tableau is owned by isl_sol and is updated incrementally.
 *
 * There are currently two implementations of this interface,
 * isl_sol_map, which simply collects the solutions in an isl_map
 * and (optionally) the parts of the context where there is no solution
 * in an isl_set, and
 * isl_sol_for, which calls a user-defined function for each part of
 * the solution.
 */
struct isl_sol {
	int error;
	int rational;
	int level;
	int max;
	int n_out;
	struct isl_context *context;
	struct isl_partial_sol *partial;
	void (*add)(struct isl_sol *sol,
			    struct isl_basic_set *dom, struct isl_mat *M);
	void (*add_empty)(struct isl_sol *sol, struct isl_basic_set *bset);
	void (*free)(struct isl_sol *sol);
	struct isl_sol_callback	dec_level;
};

static void sol_free(struct isl_sol *sol)
{
	struct isl_partial_sol *partial, *next;
	if (!sol)
		return;
	for (partial = sol->partial; partial; partial = next) {
		next = partial->next;
		isl_basic_set_free(partial->dom);
		isl_mat_free(partial->M);
		free(partial);
	}
	sol->free(sol);
}

/* Push a partial solution represented by a domain and mapping M
 * onto the stack of partial solutions.
 */
static void sol_push_sol(struct isl_sol *sol,
	struct isl_basic_set *dom, struct isl_mat *M)
{
	struct isl_partial_sol *partial;

	if (sol->error || !dom)
		goto error;

	partial = isl_alloc_type(dom->ctx, struct isl_partial_sol);
	if (!partial)
		goto error;

	partial->level = sol->level;
	partial->dom = dom;
	partial->M = M;
	partial->next = sol->partial;

	sol->partial = partial;

	return;
error:
	isl_basic_set_free(dom);
	isl_mat_free(M);
	sol->error = 1;
}

/* Pop one partial solution from the partial solution stack and
 * pass it on to sol->add or sol->add_empty.
 */
static void sol_pop_one(struct isl_sol *sol)
{
	struct isl_partial_sol *partial;

	partial = sol->partial;
	sol->partial = partial->next;

	if (partial->M)
		sol->add(sol, partial->dom, partial->M);
	else
		sol->add_empty(sol, partial->dom);
	free(partial);
}

/* Return a fresh copy of the domain represented by the context tableau.
 */
static struct isl_basic_set *sol_domain(struct isl_sol *sol)
{
	struct isl_basic_set *bset;

	if (sol->error)
		return NULL;

	bset = isl_basic_set_dup(sol->context->op->peek_basic_set(sol->context));
	bset = isl_basic_set_update_from_tab(bset,
			sol->context->op->peek_tab(sol->context));

	return bset;
}

/* Check whether two partial solutions have the same mapping, where n_div
 * is the number of divs that the two partial solutions have in common.
 */
static int same_solution(struct isl_partial_sol *s1, struct isl_partial_sol *s2,
	unsigned n_div)
{
	int i;
	unsigned dim;

	if (!s1->M != !s2->M)
		return 0;
	if (!s1->M)
		return 1;

	dim = isl_basic_set_total_dim(s1->dom) - s1->dom->n_div;

	for (i = 0; i < s1->M->n_row; ++i) {
		if (isl_seq_first_non_zero(s1->M->row[i]+1+dim+n_div,
					    s1->M->n_col-1-dim-n_div) != -1)
			return 0;
		if (isl_seq_first_non_zero(s2->M->row[i]+1+dim+n_div,
					    s2->M->n_col-1-dim-n_div) != -1)
			return 0;
		if (!isl_seq_eq(s1->M->row[i], s2->M->row[i], 1+dim+n_div))
			return 0;
	}
	return 1;
}

/* Pop all solutions from the partial solution stack that were pushed onto
 * the stack at levels that are deeper than the current level.
 * If the two topmost elements on the stack have the same level
 * and represent the same solution, then their domains are combined.
 * This combined domain is the same as the current context domain
 * as sol_pop is called each time we move back to a higher level.
 * If the outer level (0) has been reached, then all partial solutions
 * at the current level are also popped off.
 */
static void sol_pop(struct isl_sol *sol)
{
	struct isl_partial_sol *partial;
	unsigned n_div;

	if (sol->error)
		return;

	partial = sol->partial;
	if (!partial)
		return;

	if (partial->level == 0 && sol->level == 0) {
		for (partial = sol->partial; partial; partial = sol->partial)
			sol_pop_one(sol);
		return;
	}

	if (partial->level <= sol->level)
		return;

	if (partial->next && partial->next->level == partial->level) {
		n_div = isl_basic_set_dim(
				sol->context->op->peek_basic_set(sol->context),
				isl_dim_div);

		if (!same_solution(partial, partial->next, n_div)) {
			sol_pop_one(sol);
			sol_pop_one(sol);
		} else {
			struct isl_basic_set *bset;
			isl_mat *M;
			unsigned n;

			n = isl_basic_set_dim(partial->next->dom, isl_dim_div);
			n -= n_div;
			bset = sol_domain(sol);
			isl_basic_set_free(partial->next->dom);
			partial->next->dom = bset;
			M = partial->next->M;
			if (M) {
				M = isl_mat_drop_cols(M, M->n_col - n, n);
				partial->next->M = M;
				if (!M)
					goto error;
			}
			partial->next->level = sol->level;

			if (!bset)
				goto error;

			sol->partial = partial->next;
			isl_basic_set_free(partial->dom);
			isl_mat_free(partial->M);
			free(partial);
		}
	} else
		sol_pop_one(sol);

	if (sol->level == 0) {
		for (partial = sol->partial; partial; partial = sol->partial)
			sol_pop_one(sol);
		return;
	}

	if (0)
error:		sol->error = 1;
}

static void sol_dec_level(struct isl_sol *sol)
{
	if (sol->error)
		return;

	sol->level--;

	sol_pop(sol);
}

static int sol_dec_level_wrap(struct isl_tab_callback *cb)
{
	struct isl_sol_callback *callback = (struct isl_sol_callback *)cb;

	sol_dec_level(callback->sol);

	return callback->sol->error ? -1 : 0;
}

/* Move down to next level and push callback onto context tableau
 * to decrease the level again when it gets rolled back across
 * the current state.  That is, dec_level will be called with
 * the context tableau in the same state as it is when inc_level
 * is called.
 */
static void sol_inc_level(struct isl_sol *sol)
{
	struct isl_tab *tab;

	if (sol->error)
		return;

	sol->level++;
	tab = sol->context->op->peek_tab(sol->context);
	if (isl_tab_push_callback(tab, &sol->dec_level.callback) < 0)
		sol->error = 1;
}

static void scale_rows(struct isl_mat *mat, isl_int m, int n_row)
{
	int i;

	if (isl_int_is_one(m))
		return;

	for (i = 0; i < n_row; ++i)
		isl_seq_scale(mat->row[i], mat->row[i], m, mat->n_col);
}

/* Add the solution identified by the tableau and the context tableau.
 *
 * The layout of the variables is as follows.
 *	tab->n_var is equal to the total number of variables in the input
 *			map (including divs that were copied from the context)
 *			+ the number of extra divs constructed
 *      Of these, the first tab->n_param and the last tab->n_div variables
 *	correspond to the variables in the context, i.e.,
 *		tab->n_param + tab->n_div = context_tab->n_var
 *	tab->n_param is equal to the number of parameters and input
 *			dimensions in the input map
 *	tab->n_div is equal to the number of divs in the context
 *
 * If there is no solution, then call add_empty with a basic set
 * that corresponds to the context tableau.  (If add_empty is NULL,
 * then do nothing).
 *
 * If there is a solution, then first construct a matrix that maps
 * all dimensions of the context to the output variables, i.e.,
 * the output dimensions in the input map.
 * The divs in the input map (if any) that do not correspond to any
 * div in the context do not appear in the solution.
 * The algorithm will make sure that they have an integer value,
 * but these values themselves are of no interest.
 * We have to be careful not to drop or rearrange any divs in the
 * context because that would change the meaning of the matrix.
 *
 * To extract the value of the output variables, it should be noted
 * that we always use a big parameter M in the main tableau and so
 * the variable stored in this tableau is not an output variable x itself, but
 *	x' = M + x (in case of minimization)
 * or
 *	x' = M - x (in case of maximization)
 * If x' appears in a column, then its optimal value is zero,
 * which means that the optimal value of x is an unbounded number
 * (-M for minimization and M for maximization).
 * We currently assume that the output dimensions in the original map
 * are bounded, so this cannot occur.
 * Similarly, when x' appears in a row, then the coefficient of M in that
 * row is necessarily 1.
 * If the row in the tableau represents
 *	d x' = c + d M + e(y)
 * then, in case of minimization, the corresponding row in the matrix
 * will be
 *	a c + a e(y)
 * with a d = m, the (updated) common denominator of the matrix.
 * In case of maximization, the row will be
 *	-a c - a e(y)
 */
static void sol_add(struct isl_sol *sol, struct isl_tab *tab)
{
	struct isl_basic_set *bset = NULL;
	struct isl_mat *mat = NULL;
	unsigned off;
	int row;
	isl_int m;

	if (sol->error || !tab)
		goto error;

	if (tab->empty && !sol->add_empty)
		return;
	if (sol->context->op->is_empty(sol->context))
		return;

	bset = sol_domain(sol);

	if (tab->empty) {
		sol_push_sol(sol, bset, NULL);
		return;
	}

	off = 2 + tab->M;

	mat = isl_mat_alloc(tab->mat->ctx, 1 + sol->n_out,
					    1 + tab->n_param + tab->n_div);
	if (!mat)
		goto error;

	isl_int_init(m);

	isl_seq_clr(mat->row[0] + 1, mat->n_col - 1);
	isl_int_set_si(mat->row[0][0], 1);
	for (row = 0; row < sol->n_out; ++row) {
		int i = tab->n_param + row;
		int r, j;

		isl_seq_clr(mat->row[1 + row], mat->n_col);
		if (!tab->var[i].is_row) {
			if (tab->M)
				isl_die(mat->ctx, isl_error_invalid,
					"unbounded optimum", goto error2);
			continue;
		}

		r = tab->var[i].index;
		if (tab->M &&
		    isl_int_ne(tab->mat->row[r][2], tab->mat->row[r][0]))
			isl_die(mat->ctx, isl_error_invalid,
				"unbounded optimum", goto error2);
		isl_int_gcd(m, mat->row[0][0], tab->mat->row[r][0]);
		isl_int_divexact(m, tab->mat->row[r][0], m);
		scale_rows(mat, m, 1 + row);
		isl_int_divexact(m, mat->row[0][0], tab->mat->row[r][0]);
		isl_int_mul(mat->row[1 + row][0], m, tab->mat->row[r][1]);
		for (j = 0; j < tab->n_param; ++j) {
			int col;
			if (tab->var[j].is_row)
				continue;
			col = tab->var[j].index;
			isl_int_mul(mat->row[1 + row][1 + j], m,
				    tab->mat->row[r][off + col]);
		}
		for (j = 0; j < tab->n_div; ++j) {
			int col;
			if (tab->var[tab->n_var - tab->n_div+j].is_row)
				continue;
			col = tab->var[tab->n_var - tab->n_div+j].index;
			isl_int_mul(mat->row[1 + row][1 + tab->n_param + j], m,
				    tab->mat->row[r][off + col]);
		}
		if (sol->max)
			isl_seq_neg(mat->row[1 + row], mat->row[1 + row],
				    mat->n_col);
	}

	isl_int_clear(m);

	sol_push_sol(sol, bset, mat);
	return;
error2:
	isl_int_clear(m);
error:
	isl_basic_set_free(bset);
	isl_mat_free(mat);
	sol->error = 1;
}

struct isl_sol_map {
	struct isl_sol	sol;
	struct isl_map	*map;
	struct isl_set	*empty;
};

static void sol_map_free(struct isl_sol_map *sol_map)
{
	if (!sol_map)
		return;
	if (sol_map->sol.context)
		sol_map->sol.context->op->free(sol_map->sol.context);
	isl_map_free(sol_map->map);
	isl_set_free(sol_map->empty);
	free(sol_map);
}

static void sol_map_free_wrap(struct isl_sol *sol)
{
	sol_map_free((struct isl_sol_map *)sol);
}

/* This function is called for parts of the context where there is
 * no solution, with "bset" corresponding to the context tableau.
 * Simply add the basic set to the set "empty".
 */
static void sol_map_add_empty(struct isl_sol_map *sol,
	struct isl_basic_set *bset)
{
	if (!bset || !sol->empty)
		goto error;

	sol->empty = isl_set_grow(sol->empty, 1);
	bset = isl_basic_set_simplify(bset);
	bset = isl_basic_set_finalize(bset);
	sol->empty = isl_set_add_basic_set(sol->empty, isl_basic_set_copy(bset));
	if (!sol->empty)
		goto error;
	isl_basic_set_free(bset);
	return;
error:
	isl_basic_set_free(bset);
	sol->sol.error = 1;
}

static void sol_map_add_empty_wrap(struct isl_sol *sol,
	struct isl_basic_set *bset)
{
	sol_map_add_empty((struct isl_sol_map *)sol, bset);
}

/* Given a basic set "dom" that represents the context and an affine
 * matrix "M" that maps the dimensions of the context to the
 * output variables, construct a basic map with the same parameters
 * and divs as the context, the dimensions of the context as input
 * dimensions and a number of output dimensions that is equal to
 * the number of output dimensions in the input map.
 *
 * The constraints and divs of the context are simply copied
 * from "dom".  For each row
 *	x = c + e(y)
 * an equality
 *	c + e(y) - d x = 0
 * is added, with d the common denominator of M.
 */
static void sol_map_add(struct isl_sol_map *sol,
	struct isl_basic_set *dom, struct isl_mat *M)
{
	int i;
	struct isl_basic_map *bmap = NULL;
	unsigned n_eq;
	unsigned n_ineq;
	unsigned nparam;
	unsigned total;
	unsigned n_div;
	unsigned n_out;

	if (sol->sol.error || !dom || !M)
		goto error;

	n_out = sol->sol.n_out;
	n_eq = dom->n_eq + n_out;
	n_ineq = dom->n_ineq;
	n_div = dom->n_div;
	nparam = isl_basic_set_total_dim(dom) - n_div;
	total = isl_map_dim(sol->map, isl_dim_all);
	bmap = isl_basic_map_alloc_space(isl_map_get_space(sol->map),
					n_div, n_eq, 2 * n_div + n_ineq);
	if (!bmap)
		goto error;
	if (sol->sol.rational)
		ISL_F_SET(bmap, ISL_BASIC_MAP_RATIONAL);
	for (i = 0; i < dom->n_div; ++i) {
		int k = isl_basic_map_alloc_div(bmap);
		if (k < 0)
			goto error;
		isl_seq_cpy(bmap->div[k], dom->div[i], 1 + 1 + nparam);
		isl_seq_clr(bmap->div[k] + 1 + 1 + nparam, total - nparam);
		isl_seq_cpy(bmap->div[k] + 1 + 1 + total,
			    dom->div[i] + 1 + 1 + nparam, i);
	}
	for (i = 0; i < dom->n_eq; ++i) {
		int k = isl_basic_map_alloc_equality(bmap);
		if (k < 0)
			goto error;
		isl_seq_cpy(bmap->eq[k], dom->eq[i], 1 + nparam);
		isl_seq_clr(bmap->eq[k] + 1 + nparam, total - nparam);
		isl_seq_cpy(bmap->eq[k] + 1 + total,
			    dom->eq[i] + 1 + nparam, n_div);
	}
	for (i = 0; i < dom->n_ineq; ++i) {
		int k = isl_basic_map_alloc_inequality(bmap);
		if (k < 0)
			goto error;
		isl_seq_cpy(bmap->ineq[k], dom->ineq[i], 1 + nparam);
		isl_seq_clr(bmap->ineq[k] + 1 + nparam, total - nparam);
		isl_seq_cpy(bmap->ineq[k] + 1 + total,
			dom->ineq[i] + 1 + nparam, n_div);
	}
	for (i = 0; i < M->n_row - 1; ++i) {
		int k = isl_basic_map_alloc_equality(bmap);
		if (k < 0)
			goto error;
		isl_seq_cpy(bmap->eq[k], M->row[1 + i], 1 + nparam);
		isl_seq_clr(bmap->eq[k] + 1 + nparam, n_out);
		isl_int_neg(bmap->eq[k][1 + nparam + i], M->row[0][0]);
		isl_seq_cpy(bmap->eq[k] + 1 + nparam + n_out,
			    M->row[1 + i] + 1 + nparam, n_div);
	}
	bmap = isl_basic_map_simplify(bmap);
	bmap = isl_basic_map_finalize(bmap);
	sol->map = isl_map_grow(sol->map, 1);
	sol->map = isl_map_add_basic_map(sol->map, bmap);
	isl_basic_set_free(dom);
	isl_mat_free(M);
	if (!sol->map)
		sol->sol.error = 1;
	return;
error:
	isl_basic_set_free(dom);
	isl_mat_free(M);
	isl_basic_map_free(bmap);
	sol->sol.error = 1;
}

static void sol_map_add_wrap(struct isl_sol *sol,
	struct isl_basic_set *dom, struct isl_mat *M)
{
	sol_map_add((struct isl_sol_map *)sol, dom, M);
}


/* Store the "parametric constant" of row "row" of tableau "tab" in "line",
 * i.e., the constant term and the coefficients of all variables that
 * appear in the context tableau.
 * Note that the coefficient of the big parameter M is NOT copied.
 * The context tableau may not have a big parameter and even when it
 * does, it is a different big parameter.
 */
static void get_row_parameter_line(struct isl_tab *tab, int row, isl_int *line)
{
	int i;
	unsigned off = 2 + tab->M;

	isl_int_set(line[0], tab->mat->row[row][1]);
	for (i = 0; i < tab->n_param; ++i) {
		if (tab->var[i].is_row)
			isl_int_set_si(line[1 + i], 0);
		else {
			int col = tab->var[i].index;
			isl_int_set(line[1 + i], tab->mat->row[row][off + col]);
		}
	}
	for (i = 0; i < tab->n_div; ++i) {
		if (tab->var[tab->n_var - tab->n_div + i].is_row)
			isl_int_set_si(line[1 + tab->n_param + i], 0);
		else {
			int col = tab->var[tab->n_var - tab->n_div + i].index;
			isl_int_set(line[1 + tab->n_param + i],
				    tab->mat->row[row][off + col]);
		}
	}
}

/* Check if rows "row1" and "row2" have identical "parametric constants",
 * as explained above.
 * In this case, we also insist that the coefficients of the big parameter
 * be the same as the values of the constants will only be the same
 * if these coefficients are also the same.
 */
static int identical_parameter_line(struct isl_tab *tab, int row1, int row2)
{
	int i;
	unsigned off = 2 + tab->M;

	if (isl_int_ne(tab->mat->row[row1][1], tab->mat->row[row2][1]))
		return 0;

	if (tab->M && isl_int_ne(tab->mat->row[row1][2],
				 tab->mat->row[row2][2]))
		return 0;

	for (i = 0; i < tab->n_param + tab->n_div; ++i) {
		int pos = i < tab->n_param ? i :
			tab->n_var - tab->n_div + i - tab->n_param;
		int col;

		if (tab->var[pos].is_row)
			continue;
		col = tab->var[pos].index;
		if (isl_int_ne(tab->mat->row[row1][off + col],
			       tab->mat->row[row2][off + col]))
			return 0;
	}
	return 1;
}

/* Return an inequality that expresses that the "parametric constant"
 * should be non-negative.
 * This function is only called when the coefficient of the big parameter
 * is equal to zero.
 */
static struct isl_vec *get_row_parameter_ineq(struct isl_tab *tab, int row)
{
	struct isl_vec *ineq;

	ineq = isl_vec_alloc(tab->mat->ctx, 1 + tab->n_param + tab->n_div);
	if (!ineq)
		return NULL;

	get_row_parameter_line(tab, row, ineq->el);
	if (ineq)
		ineq = isl_vec_normalize(ineq);

	return ineq;
}

/* Normalize a div expression of the form
 *
 *	[(g*f(x) + c)/(g * m)]
 *
 * with c the constant term and f(x) the remaining coefficients, to
 *
 *	[(f(x) + [c/g])/m]
 */
static void normalize_div(__isl_keep isl_vec *div)
{
	isl_ctx *ctx = isl_vec_get_ctx(div);
	int len = div->size - 2;

	isl_seq_gcd(div->el + 2, len, &ctx->normalize_gcd);
	isl_int_gcd(ctx->normalize_gcd, ctx->normalize_gcd, div->el[0]);

	if (isl_int_is_one(ctx->normalize_gcd))
		return;

	isl_int_divexact(div->el[0], div->el[0], ctx->normalize_gcd);
	isl_int_fdiv_q(div->el[1], div->el[1], ctx->normalize_gcd);
	isl_seq_scale_down(div->el + 2, div->el + 2, ctx->normalize_gcd, len);
}

/* Return an integer division for use in a parametric cut based
 * on the given row.
 * In particular, let the parametric constant of the row be
 *
 *		\sum_i a_i y_i
 *
 * where y_0 = 1, but none of the y_i corresponds to the big parameter M.
 * The div returned is equal to
 *
 *		floor(\sum_i {-a_i} y_i) = floor((\sum_i (-a_i mod d) y_i)/d)
 */
static struct isl_vec *get_row_parameter_div(struct isl_tab *tab, int row)
{
	struct isl_vec *div;

	div = isl_vec_alloc(tab->mat->ctx, 1 + 1 + tab->n_param + tab->n_div);
	if (!div)
		return NULL;

	isl_int_set(div->el[0], tab->mat->row[row][0]);
	get_row_parameter_line(tab, row, div->el + 1);
	isl_seq_neg(div->el + 1, div->el + 1, div->size - 1);
	normalize_div(div);
	isl_seq_fdiv_r(div->el + 1, div->el + 1, div->el[0], div->size - 1);

	return div;
}

/* Return an integer division for use in transferring an integrality constraint
 * to the context.
 * In particular, let the parametric constant of the row be
 *
 *		\sum_i a_i y_i
 *
 * where y_0 = 1, but none of the y_i corresponds to the big parameter M.
 * The the returned div is equal to
 *
 *		floor(\sum_i {a_i} y_i) = floor((\sum_i (a_i mod d) y_i)/d)
 */
static struct isl_vec *get_row_split_div(struct isl_tab *tab, int row)
{
	struct isl_vec *div;

	div = isl_vec_alloc(tab->mat->ctx, 1 + 1 + tab->n_param + tab->n_div);
	if (!div)
		return NULL;

	isl_int_set(div->el[0], tab->mat->row[row][0]);
	get_row_parameter_line(tab, row, div->el + 1);
	normalize_div(div);
	isl_seq_fdiv_r(div->el + 1, div->el + 1, div->el[0], div->size - 1);

	return div;
}

/* Construct and return an inequality that expresses an upper bound
 * on the given div.
 * In particular, if the div is given by
 *
 *	d = floor(e/m)
 *
 * then the inequality expresses
 *
 *	m d <= e
 */
static struct isl_vec *ineq_for_div(struct isl_basic_set *bset, unsigned div)
{
	unsigned total;
	unsigned div_pos;
	struct isl_vec *ineq;

	if (!bset)
		return NULL;

	total = isl_basic_set_total_dim(bset);
	div_pos = 1 + total - bset->n_div + div;

	ineq = isl_vec_alloc(bset->ctx, 1 + total);
	if (!ineq)
		return NULL;

	isl_seq_cpy(ineq->el, bset->div[div] + 1, 1 + total);
	isl_int_neg(ineq->el[div_pos], bset->div[div][0]);
	return ineq;
}

/* Given a row in the tableau and a div that was created
 * using get_row_split_div and that has been constrained to equality, i.e.,
 *
 *		d = floor(\sum_i {a_i} y_i) = \sum_i {a_i} y_i
 *
 * replace the expression "\sum_i {a_i} y_i" in the row by d,
 * i.e., we subtract "\sum_i {a_i} y_i" and add 1 d.
 * The coefficients of the non-parameters in the tableau have been
 * verified to be integral.  We can therefore simply replace coefficient b
 * by floor(b).  For the coefficients of the parameters we have
 * floor(a_i) = a_i - {a_i}, while for the other coefficients, we have
 * floor(b) = b.
 */
static struct isl_tab *set_row_cst_to_div(struct isl_tab *tab, int row, int div)
{
	isl_seq_fdiv_q(tab->mat->row[row] + 1, tab->mat->row[row] + 1,
			tab->mat->row[row][0], 1 + tab->M + tab->n_col);

	isl_int_set_si(tab->mat->row[row][0], 1);

	if (tab->var[tab->n_var - tab->n_div + div].is_row) {
		int drow = tab->var[tab->n_var - tab->n_div + div].index;

		isl_assert(tab->mat->ctx,
			isl_int_is_one(tab->mat->row[drow][0]), goto error);
		isl_seq_combine(tab->mat->row[row] + 1,
			tab->mat->ctx->one, tab->mat->row[row] + 1,
			tab->mat->ctx->one, tab->mat->row[drow] + 1,
			1 + tab->M + tab->n_col);
	} else {
		int dcol = tab->var[tab->n_var - tab->n_div + div].index;

		isl_int_add_ui(tab->mat->row[row][2 + tab->M + dcol],
				tab->mat->row[row][2 + tab->M + dcol], 1);
	}

	return tab;
error:
	isl_tab_free(tab);
	return NULL;
}

/* Check if the (parametric) constant of the given row is obviously
 * negative, meaning that we don't need to consult the context tableau.
 * If there is a big parameter and its coefficient is non-zero,
 * then this coefficient determines the outcome.
 * Otherwise, we check whether the constant is negative and
 * all non-zero coefficients of parameters are negative and
 * belong to non-negative parameters.
 */
static int is_obviously_neg(struct isl_tab *tab, int row)
{
	int i;
	int col;
	unsigned off = 2 + tab->M;

	if (tab->M) {
		if (isl_int_is_pos(tab->mat->row[row][2]))
			return 0;
		if (isl_int_is_neg(tab->mat->row[row][2]))
			return 1;
	}

	if (isl_int_is_nonneg(tab->mat->row[row][1]))
		return 0;
	for (i = 0; i < tab->n_param; ++i) {
		/* Eliminated parameter */
		if (tab->var[i].is_row)
			continue;
		col = tab->var[i].index;
		if (isl_int_is_zero(tab->mat->row[row][off + col]))
			continue;
		if (!tab->var[i].is_nonneg)
			return 0;
		if (isl_int_is_pos(tab->mat->row[row][off + col]))
			return 0;
	}
	for (i = 0; i < tab->n_div; ++i) {
		if (tab->var[tab->n_var - tab->n_div + i].is_row)
			continue;
		col = tab->var[tab->n_var - tab->n_div + i].index;
		if (isl_int_is_zero(tab->mat->row[row][off + col]))
			continue;
		if (!tab->var[tab->n_var - tab->n_div + i].is_nonneg)
			return 0;
		if (isl_int_is_pos(tab->mat->row[row][off + col]))
			return 0;
	}
	return 1;
}

/* Check if the (parametric) constant of the given row is obviously
 * non-negative, meaning that we don't need to consult the context tableau.
 * If there is a big parameter and its coefficient is non-zero,
 * then this coefficient determines the outcome.
 * Otherwise, we check whether the constant is non-negative and
 * all non-zero coefficients of parameters are positive and
 * belong to non-negative parameters.
 */
static int is_obviously_nonneg(struct isl_tab *tab, int row)
{
	int i;
	int col;
	unsigned off = 2 + tab->M;

	if (tab->M) {
		if (isl_int_is_pos(tab->mat->row[row][2]))
			return 1;
		if (isl_int_is_neg(tab->mat->row[row][2]))
			return 0;
	}

	if (isl_int_is_neg(tab->mat->row[row][1]))
		return 0;
	for (i = 0; i < tab->n_param; ++i) {
		/* Eliminated parameter */
		if (tab->var[i].is_row)
			continue;
		col = tab->var[i].index;
		if (isl_int_is_zero(tab->mat->row[row][off + col]))
			continue;
		if (!tab->var[i].is_nonneg)
			return 0;
		if (isl_int_is_neg(tab->mat->row[row][off + col]))
			return 0;
	}
	for (i = 0; i < tab->n_div; ++i) {
		if (tab->var[tab->n_var - tab->n_div + i].is_row)
			continue;
		col = tab->var[tab->n_var - tab->n_div + i].index;
		if (isl_int_is_zero(tab->mat->row[row][off + col]))
			continue;
		if (!tab->var[tab->n_var - tab->n_div + i].is_nonneg)
			return 0;
		if (isl_int_is_neg(tab->mat->row[row][off + col]))
			return 0;
	}
	return 1;
}

/* Given a row r and two columns, return the column that would
 * lead to the lexicographically smallest increment in the sample
 * solution when leaving the basis in favor of the row.
 * Pivoting with column c will increment the sample value by a non-negative
 * constant times a_{V,c}/a_{r,c}, with a_{V,c} the elements of column c
 * corresponding to the non-parametric variables.
 * If variable v appears in a column c_v, the a_{v,c} = 1 iff c = c_v,
 * with all other entries in this virtual row equal to zero.
 * If variable v appears in a row, then a_{v,c} is the element in column c
 * of that row.
 *
 * Let v be the first variable with a_{v,c1}/a_{r,c1} != a_{v,c2}/a_{r,c2}.
 * Then if a_{v,c1}/a_{r,c1} < a_{v,c2}/a_{r,c2}, i.e.,
 * a_{v,c2} a_{r,c1} - a_{v,c1} a_{r,c2} > 0, c1 results in the minimal
 * increment.  Otherwise, it's c2.
 */
static int lexmin_col_pair(struct isl_tab *tab,
	int row, int col1, int col2, isl_int tmp)
{
	int i;
	isl_int *tr;

	tr = tab->mat->row[row] + 2 + tab->M;

	for (i = tab->n_param; i < tab->n_var - tab->n_div; ++i) {
		int s1, s2;
		isl_int *r;

		if (!tab->var[i].is_row) {
			if (tab->var[i].index == col1)
				return col2;
			if (tab->var[i].index == col2)
				return col1;
			continue;
		}

		if (tab->var[i].index == row)
			continue;

		r = tab->mat->row[tab->var[i].index] + 2 + tab->M;
		s1 = isl_int_sgn(r[col1]);
		s2 = isl_int_sgn(r[col2]);
		if (s1 == 0 && s2 == 0)
			continue;
		if (s1 < s2)
			return col1;
		if (s2 < s1)
			return col2;

		isl_int_mul(tmp, r[col2], tr[col1]);
		isl_int_submul(tmp, r[col1], tr[col2]);
		if (isl_int_is_pos(tmp))
			return col1;
		if (isl_int_is_neg(tmp))
			return col2;
	}
	return -1;
}

/* Given a row in the tableau, find and return the column that would
 * result in the lexicographically smallest, but positive, increment
 * in the sample point.
 * If there is no such column, then return tab->n_col.
 * If anything goes wrong, return -1.
 */
static int lexmin_pivot_col(struct isl_tab *tab, int row)
{
	int j;
	int col = tab->n_col;
	isl_int *tr;
	isl_int tmp;

	tr = tab->mat->row[row] + 2 + tab->M;

	isl_int_init(tmp);

	for (j = tab->n_dead; j < tab->n_col; ++j) {
		if (tab->col_var[j] >= 0 &&
		    (tab->col_var[j] < tab->n_param  ||
		    tab->col_var[j] >= tab->n_var - tab->n_div))
			continue;

		if (!isl_int_is_pos(tr[j]))
			continue;

		if (col == tab->n_col)
			col = j;
		else
			col = lexmin_col_pair(tab, row, col, j, tmp);
		isl_assert(tab->mat->ctx, col >= 0, goto error);
	}

	isl_int_clear(tmp);
	return col;
error:
	isl_int_clear(tmp);
	return -1;
}

/* Return the first known violated constraint, i.e., a non-negative
 * constraint that currently has an either obviously negative value
 * or a previously determined to be negative value.
 *
 * If any constraint has a negative coefficient for the big parameter,
 * if any, then we return one of these first.
 */
static int first_neg(struct isl_tab *tab)
{
	int row;

	if (tab->M)
		for (row = tab->n_redundant; row < tab->n_row; ++row) {
			if (!isl_tab_var_from_row(tab, row)->is_nonneg)
				continue;
			if (!isl_int_is_neg(tab->mat->row[row][2]))
				continue;
			if (tab->row_sign)
				tab->row_sign[row] = isl_tab_row_neg;
			return row;
		}
	for (row = tab->n_redundant; row < tab->n_row; ++row) {
		if (!isl_tab_var_from_row(tab, row)->is_nonneg)
			continue;
		if (tab->row_sign) {
			if (tab->row_sign[row] == 0 &&
			    is_obviously_neg(tab, row))
				tab->row_sign[row] = isl_tab_row_neg;
			if (tab->row_sign[row] != isl_tab_row_neg)
				continue;
		} else if (!is_obviously_neg(tab, row))
			continue;
		return row;
	}
	return -1;
}

/* Check whether the invariant that all columns are lexico-positive
 * is satisfied.  This function is not called from the current code
 * but is useful during debugging.
 */
static void check_lexpos(struct isl_tab *tab) __attribute__ ((unused));
static void check_lexpos(struct isl_tab *tab)
{
	unsigned off = 2 + tab->M;
	int col;
	int var;
	int row;

	for (col = tab->n_dead; col < tab->n_col; ++col) {
		if (tab->col_var[col] >= 0 &&
		    (tab->col_var[col] < tab->n_param ||
		     tab->col_var[col] >= tab->n_var - tab->n_div))
			continue;
		for (var = tab->n_param; var < tab->n_var - tab->n_div; ++var) {
			if (!tab->var[var].is_row) {
				if (tab->var[var].index == col)
					break;
				else
					continue;
			}
			row = tab->var[var].index;
			if (isl_int_is_zero(tab->mat->row[row][off + col]))
				continue;
			if (isl_int_is_pos(tab->mat->row[row][off + col]))
				break;
			fprintf(stderr, "lexneg column %d (row %d)\n",
				col, row);
		}
		if (var >= tab->n_var - tab->n_div)
			fprintf(stderr, "zero column %d\n", col);
	}
}

/* Report to the caller that the given constraint is part of an encountered
 * conflict.
 */
static int report_conflicting_constraint(struct isl_tab *tab, int con)
{
	return tab->conflict(con, tab->conflict_user);
}

/* Given a conflicting row in the tableau, report all constraints
 * involved in the row to the caller.  That is, the row itself
 * (if it represents a constraint) and all constraint columns with
 * non-zero (and therefore negative) coefficients.
 */
static int report_conflict(struct isl_tab *tab, int row)
{
	int j;
	isl_int *tr;

	if (!tab->conflict)
		return 0;

	if (tab->row_var[row] < 0 &&
	    report_conflicting_constraint(tab, ~tab->row_var[row]) < 0)
		return -1;

	tr = tab->mat->row[row] + 2 + tab->M;

	for (j = tab->n_dead; j < tab->n_col; ++j) {
		if (tab->col_var[j] >= 0 &&
		    (tab->col_var[j] < tab->n_param  ||
		    tab->col_var[j] >= tab->n_var - tab->n_div))
			continue;

		if (!isl_int_is_neg(tr[j]))
			continue;

		if (tab->col_var[j] < 0 &&
		    report_conflicting_constraint(tab, ~tab->col_var[j]) < 0)
			return -1;
	}

	return 0;
}

/* Resolve all known or obviously violated constraints through pivoting.
 * In particular, as long as we can find any violated constraint, we
 * look for a pivoting column that would result in the lexicographically
 * smallest increment in the sample point.  If there is no such column
 * then the tableau is infeasible.
 */
static int restore_lexmin(struct isl_tab *tab) WARN_UNUSED;
static int restore_lexmin(struct isl_tab *tab)
{
	int row, col;

	if (!tab)
		return -1;
	if (tab->empty)
		return 0;
	while ((row = first_neg(tab)) != -1) {
		col = lexmin_pivot_col(tab, row);
		if (col >= tab->n_col) {
			if (report_conflict(tab, row) < 0)
				return -1;
			if (isl_tab_mark_empty(tab) < 0)
				return -1;
			return 0;
		}
		if (col < 0)
			return -1;
		if (isl_tab_pivot(tab, row, col) < 0)
			return -1;
	}
	return 0;
}

/* Given a row that represents an equality, look for an appropriate
 * pivoting column.
 * In particular, if there are any non-zero coefficients among
 * the non-parameter variables, then we take the last of these
 * variables.  Eliminating this variable in terms of the other
 * variables and/or parameters does not influence the property
 * that all column in the initial tableau are lexicographically
 * positive.  The row corresponding to the eliminated variable
 * will only have non-zero entries below the diagonal of the
 * initial tableau.  That is, we transform
 *
 *		I				I
 *		  1		into		a
 *		    I				  I
 *
 * If there is no such non-parameter variable, then we are dealing with
 * pure parameter equality and we pick any parameter with coefficient 1 or -1
 * for elimination.  This will ensure that the eliminated parameter
 * always has an integer value whenever all the other parameters are integral.
 * If there is no such parameter then we return -1.
 */
static int last_var_col_or_int_par_col(struct isl_tab *tab, int row)
{
	unsigned off = 2 + tab->M;
	int i;

	for (i = tab->n_var - tab->n_div - 1; i >= 0 && i >= tab->n_param; --i) {
		int col;
		if (tab->var[i].is_row)
			continue;
		col = tab->var[i].index;
		if (col <= tab->n_dead)
			continue;
		if (!isl_int_is_zero(tab->mat->row[row][off + col]))
			return col;
	}
	for (i = tab->n_dead; i < tab->n_col; ++i) {
		if (isl_int_is_one(tab->mat->row[row][off + i]))
			return i;
		if (isl_int_is_negone(tab->mat->row[row][off + i]))
			return i;
	}
	return -1;
}

/* Add an equality that is known to be valid to the tableau.
 * We first check if we can eliminate a variable or a parameter.
 * If not, we add the equality as two inequalities.
 * In this case, the equality was a pure parameter equality and there
 * is no need to resolve any constraint violations.
 *
 * This function assumes that at least two more rows and at least
 * two more elements in the constraint array are available in the tableau.
 */
static struct isl_tab *add_lexmin_valid_eq(struct isl_tab *tab, isl_int *eq)
{
	int i;
	int r;

	if (!tab)
		return NULL;
	r = isl_tab_add_row(tab, eq);
	if (r < 0)
		goto error;

	r = tab->con[r].index;
	i = last_var_col_or_int_par_col(tab, r);
	if (i < 0) {
		tab->con[r].is_nonneg = 1;
		if (isl_tab_push_var(tab, isl_tab_undo_nonneg, &tab->con[r]) < 0)
			goto error;
		isl_seq_neg(eq, eq, 1 + tab->n_var);
		r = isl_tab_add_row(tab, eq);
		if (r < 0)
			goto error;
		tab->con[r].is_nonneg = 1;
		if (isl_tab_push_var(tab, isl_tab_undo_nonneg, &tab->con[r]) < 0)
			goto error;
	} else {
		if (isl_tab_pivot(tab, r, i) < 0)
			goto error;
		if (isl_tab_kill_col(tab, i) < 0)
			goto error;
		tab->n_eq++;
	}

	return tab;
error:
	isl_tab_free(tab);
	return NULL;
}

/* Check if the given row is a pure constant.
 */
static int is_constant(struct isl_tab *tab, int row)
{
	unsigned off = 2 + tab->M;

	return isl_seq_first_non_zero(tab->mat->row[row] + off + tab->n_dead,
					tab->n_col - tab->n_dead) == -1;
}

/* Add an equality that may or may not be valid to the tableau.
 * If the resulting row is a pure constant, then it must be zero.
 * Otherwise, the resulting tableau is empty.
 *
 * If the row is not a pure constant, then we add two inequalities,
 * each time checking that they can be satisfied.
 * In the end we try to use one of the two constraints to eliminate
 * a column.
 *
 * This function assumes that at least two more rows and at least
 * two more elements in the constraint array are available in the tableau.
 */
static int add_lexmin_eq(struct isl_tab *tab, isl_int *eq) WARN_UNUSED;
static int add_lexmin_eq(struct isl_tab *tab, isl_int *eq)
{
	int r1, r2;
	int row;
	struct isl_tab_undo *snap;

	if (!tab)
		return -1;
	snap = isl_tab_snap(tab);
	r1 = isl_tab_add_row(tab, eq);
	if (r1 < 0)
		return -1;
	tab->con[r1].is_nonneg = 1;
	if (isl_tab_push_var(tab, isl_tab_undo_nonneg, &tab->con[r1]) < 0)
		return -1;

	row = tab->con[r1].index;
	if (is_constant(tab, row)) {
		if (!isl_int_is_zero(tab->mat->row[row][1]) ||
		    (tab->M && !isl_int_is_zero(tab->mat->row[row][2]))) {
			if (isl_tab_mark_empty(tab) < 0)
				return -1;
			return 0;
		}
		if (isl_tab_rollback(tab, snap) < 0)
			return -1;
		return 0;
	}

	if (restore_lexmin(tab) < 0)
		return -1;
	if (tab->empty)
		return 0;

	isl_seq_neg(eq, eq, 1 + tab->n_var);

	r2 = isl_tab_add_row(tab, eq);
	if (r2 < 0)
		return -1;
	tab->con[r2].is_nonneg = 1;
	if (isl_tab_push_var(tab, isl_tab_undo_nonneg, &tab->con[r2]) < 0)
		return -1;

	if (restore_lexmin(tab) < 0)
		return -1;
	if (tab->empty)
		return 0;

	if (!tab->con[r1].is_row) {
		if (isl_tab_kill_col(tab, tab->con[r1].index) < 0)
			return -1;
	} else if (!tab->con[r2].is_row) {
		if (isl_tab_kill_col(tab, tab->con[r2].index) < 0)
			return -1;
	}

	if (tab->bmap) {
		tab->bmap = isl_basic_map_add_ineq(tab->bmap, eq);
		if (isl_tab_push(tab, isl_tab_undo_bmap_ineq) < 0)
			return -1;
		isl_seq_neg(eq, eq, 1 + tab->n_var);
		tab->bmap = isl_basic_map_add_ineq(tab->bmap, eq);
		isl_seq_neg(eq, eq, 1 + tab->n_var);
		if (isl_tab_push(tab, isl_tab_undo_bmap_ineq) < 0)
			return -1;
		if (!tab->bmap)
			return -1;
	}

	return 0;
}

/* Add an inequality to the tableau, resolving violations using
 * restore_lexmin.
 *
 * This function assumes that at least one more row and at least
 * one more element in the constraint array are available in the tableau.
 */
static struct isl_tab *add_lexmin_ineq(struct isl_tab *tab, isl_int *ineq)
{
	int r;

	if (!tab)
		return NULL;
	if (tab->bmap) {
		tab->bmap = isl_basic_map_add_ineq(tab->bmap, ineq);
		if (isl_tab_push(tab, isl_tab_undo_bmap_ineq) < 0)
			goto error;
		if (!tab->bmap)
			goto error;
	}
	r = isl_tab_add_row(tab, ineq);
	if (r < 0)
		goto error;
	tab->con[r].is_nonneg = 1;
	if (isl_tab_push_var(tab, isl_tab_undo_nonneg, &tab->con[r]) < 0)
		goto error;
	if (isl_tab_row_is_redundant(tab, tab->con[r].index)) {
		if (isl_tab_mark_redundant(tab, tab->con[r].index) < 0)
			goto error;
		return tab;
	}

	if (restore_lexmin(tab) < 0)
		goto error;
	if (!tab->empty && tab->con[r].is_row &&
		 isl_tab_row_is_redundant(tab, tab->con[r].index))
		if (isl_tab_mark_redundant(tab, tab->con[r].index) < 0)
			goto error;
	return tab;
error:
	isl_tab_free(tab);
	return NULL;
}

/* Check if the coefficients of the parameters are all integral.
 */
static int integer_parameter(struct isl_tab *tab, int row)
{
	int i;
	int col;
	unsigned off = 2 + tab->M;

	for (i = 0; i < tab->n_param; ++i) {
		/* Eliminated parameter */
		if (tab->var[i].is_row)
			continue;
		col = tab->var[i].index;
		if (!isl_int_is_divisible_by(tab->mat->row[row][off + col],
						tab->mat->row[row][0]))
			return 0;
	}
	for (i = 0; i < tab->n_div; ++i) {
		if (tab->var[tab->n_var - tab->n_div + i].is_row)
			continue;
		col = tab->var[tab->n_var - tab->n_div + i].index;
		if (!isl_int_is_divisible_by(tab->mat->row[row][off + col],
						tab->mat->row[row][0]))
			return 0;
	}
	return 1;
}

/* Check if the coefficients of the non-parameter variables are all integral.
 */
static int integer_variable(struct isl_tab *tab, int row)
{
	int i;
	unsigned off = 2 + tab->M;

	for (i = tab->n_dead; i < tab->n_col; ++i) {
		if (tab->col_var[i] >= 0 &&
		    (tab->col_var[i] < tab->n_param ||
		     tab->col_var[i] >= tab->n_var - tab->n_div))
			continue;
		if (!isl_int_is_divisible_by(tab->mat->row[row][off + i],
						tab->mat->row[row][0]))
			return 0;
	}
	return 1;
}

/* Check if the constant term is integral.
 */
static int integer_constant(struct isl_tab *tab, int row)
{
	return isl_int_is_divisible_by(tab->mat->row[row][1],
					tab->mat->row[row][0]);
}

#define I_CST	1 << 0
#define I_PAR	1 << 1
#define I_VAR	1 << 2

/* Check for next (non-parameter) variable after "var" (first if var == -1)
 * that is non-integer and therefore requires a cut and return
 * the index of the variable.
 * For parametric tableaus, there are three parts in a row,
 * the constant, the coefficients of the parameters and the rest.
 * For each part, we check whether the coefficients in that part
 * are all integral and if so, set the corresponding flag in *f.
 * If the constant and the parameter part are integral, then the
 * current sample value is integral and no cut is required
 * (irrespective of whether the variable part is integral).
 */
static int next_non_integer_var(struct isl_tab *tab, int var, int *f)
{
	var = var < 0 ? tab->n_param : var + 1;

	for (; var < tab->n_var - tab->n_div; ++var) {
		int flags = 0;
		int row;
		if (!tab->var[var].is_row)
			continue;
		row = tab->var[var].index;
		if (integer_constant(tab, row))
			ISL_FL_SET(flags, I_CST);
		if (integer_parameter(tab, row))
			ISL_FL_SET(flags, I_PAR);
		if (ISL_FL_ISSET(flags, I_CST) && ISL_FL_ISSET(flags, I_PAR))
			continue;
		if (integer_variable(tab, row))
			ISL_FL_SET(flags, I_VAR);
		*f = flags;
		return var;
	}
	return -1;
}

/* Check for first (non-parameter) variable that is non-integer and
 * therefore requires a cut and return the corresponding row.
 * For parametric tableaus, there are three parts in a row,
 * the constant, the coefficients of the parameters and the rest.
 * For each part, we check whether the coefficients in that part
 * are all integral and if so, set the corresponding flag in *f.
 * If the constant and the parameter part are integral, then the
 * current sample value is integral and no cut is required
 * (irrespective of whether the variable part is integral).
 */
static int first_non_integer_row(struct isl_tab *tab, int *f)
{
	int var = next_non_integer_var(tab, -1, f);

	return var < 0 ? -1 : tab->var[var].index;
}

/* Add a (non-parametric) cut to cut away the non-integral sample
 * value of the given row.
 *
 * If the row is given by
 *
 *	m r = f + \sum_i a_i y_i
 *
 * then the cut is
 *
 *	c = - {-f/m} + \sum_i {a_i/m} y_i >= 0
 *
 * The big parameter, if any, is ignored, since it is assumed to be big
 * enough to be divisible by any integer.
 * If the tableau is actually a parametric tableau, then this function
 * is only called when all coefficients of the parameters are integral.
 * The cut therefore has zero coefficients for the parameters.
 *
 * The current value is known to be negative, so row_sign, if it
 * exists, is set accordingly.
 *
 * Return the row of the cut or -1.
 */
static int add_cut(struct isl_tab *tab, int row)
{
	int i;
	int r;
	isl_int *r_row;
	unsigned off = 2 + tab->M;

	if (isl_tab_extend_cons(tab, 1) < 0)
		return -1;
	r = isl_tab_allocate_con(tab);
	if (r < 0)
		return -1;

	r_row = tab->mat->row[tab->con[r].index];
	isl_int_set(r_row[0], tab->mat->row[row][0]);
	isl_int_neg(r_row[1], tab->mat->row[row][1]);
	isl_int_fdiv_r(r_row[1], r_row[1], tab->mat->row[row][0]);
	isl_int_neg(r_row[1], r_row[1]);
	if (tab->M)
		isl_int_set_si(r_row[2], 0);
	for (i = 0; i < tab->n_col; ++i)
		isl_int_fdiv_r(r_row[off + i],
			tab->mat->row[row][off + i], tab->mat->row[row][0]);

	tab->con[r].is_nonneg = 1;
	if (isl_tab_push_var(tab, isl_tab_undo_nonneg, &tab->con[r]) < 0)
		return -1;
	if (tab->row_sign)
		tab->row_sign[tab->con[r].index] = isl_tab_row_neg;

	return tab->con[r].index;
}

#define CUT_ALL 1
#define CUT_ONE 0

/* Given a non-parametric tableau, add cuts until an integer
 * sample point is obtained or until the tableau is determined
 * to be integer infeasible.
 * As long as there is any non-integer value in the sample point,
 * we add appropriate cuts, if possible, for each of these
 * non-integer values and then resolve the violated
 * cut constraints using restore_lexmin.
 * If one of the corresponding rows is equal to an integral
 * combination of variables/constraints plus a non-integral constant,
 * then there is no way to obtain an integer point and we return
 * a tableau that is marked empty.
 * The parameter cutting_strategy controls the strategy used when adding cuts
 * to remove non-integer points. CUT_ALL adds all possible cuts
 * before continuing the search. CUT_ONE adds only one cut at a time.
 */
static struct isl_tab *cut_to_integer_lexmin(struct isl_tab *tab,
	int cutting_strategy)
{
	int var;
	int row;
	int flags;

	if (!tab)
		return NULL;
	if (tab->empty)
		return tab;

	while ((var = next_non_integer_var(tab, -1, &flags)) != -1) {
		do {
			if (ISL_FL_ISSET(flags, I_VAR)) {
				if (isl_tab_mark_empty(tab) < 0)
					goto error;
				return tab;
			}
			row = tab->var[var].index;
			row = add_cut(tab, row);
			if (row < 0)
				goto error;
			if (cutting_strategy == CUT_ONE)
				break;
		} while ((var = next_non_integer_var(tab, var, &flags)) != -1);
		if (restore_lexmin(tab) < 0)
			goto error;
		if (tab->empty)
			break;
	}
	return tab;
error:
	isl_tab_free(tab);
	return NULL;
}

/* Check whether all the currently active samples also satisfy the inequality
 * "ineq" (treated as an equality if eq is set).
 * Remove those samples that do not.
 */
static struct isl_tab *check_samples(struct isl_tab *tab, isl_int *ineq, int eq)
{
	int i;
	isl_int v;

	if (!tab)
		return NULL;

	isl_assert(tab->mat->ctx, tab->bmap, goto error);
	isl_assert(tab->mat->ctx, tab->samples, goto error);
	isl_assert(tab->mat->ctx, tab->samples->n_col == 1 + tab->n_var, goto error);

	isl_int_init(v);
	for (i = tab->n_outside; i < tab->n_sample; ++i) {
		int sgn;
		isl_seq_inner_product(ineq, tab->samples->row[i],
					1 + tab->n_var, &v);
		sgn = isl_int_sgn(v);
		if (eq ? (sgn == 0) : (sgn >= 0))
			continue;
		tab = isl_tab_drop_sample(tab, i);
		if (!tab)
			break;
	}
	isl_int_clear(v);

	return tab;
error:
	isl_tab_free(tab);
	return NULL;
}

/* Check whether the sample value of the tableau is finite,
 * i.e., either the tableau does not use a big parameter, or
 * all values of the variables are equal to the big parameter plus
 * some constant.  This constant is the actual sample value.
 */
static int sample_is_finite(struct isl_tab *tab)
{
	int i;

	if (!tab->M)
		return 1;

	for (i = 0; i < tab->n_var; ++i) {
		int row;
		if (!tab->var[i].is_row)
			return 0;
		row = tab->var[i].index;
		if (isl_int_ne(tab->mat->row[row][0], tab->mat->row[row][2]))
			return 0;
	}
	return 1;
}

/* Check if the context tableau of sol has any integer points.
 * Leave tab in empty state if no integer point can be found.
 * If an integer point can be found and if moreover it is finite,
 * then it is added to the list of sample values.
 *
 * This function is only called when none of the currently active sample
 * values satisfies the most recently added constraint.
 */
static struct isl_tab *check_integer_feasible(struct isl_tab *tab)
{
	struct isl_tab_undo *snap;

	if (!tab)
		return NULL;

	snap = isl_tab_snap(tab);
	if (isl_tab_push_basis(tab) < 0)
		goto error;

	tab = cut_to_integer_lexmin(tab, CUT_ALL);
	if (!tab)
		goto error;

	if (!tab->empty && sample_is_finite(tab)) {
		struct isl_vec *sample;

		sample = isl_tab_get_sample_value(tab);

		if (isl_tab_add_sample(tab, sample) < 0)
			goto error;
	}

	if (!tab->empty && isl_tab_rollback(tab, snap) < 0)
		goto error;

	return tab;
error:
	isl_tab_free(tab);
	return NULL;
}

/* Check if any of the currently active sample values satisfies
 * the inequality "ineq" (an equality if eq is set).
 */
static int tab_has_valid_sample(struct isl_tab *tab, isl_int *ineq, int eq)
{
	int i;
	isl_int v;

	if (!tab)
		return -1;

	isl_assert(tab->mat->ctx, tab->bmap, return -1);
	isl_assert(tab->mat->ctx, tab->samples, return -1);
	isl_assert(tab->mat->ctx, tab->samples->n_col == 1 + tab->n_var, return -1);

	isl_int_init(v);
	for (i = tab->n_outside; i < tab->n_sample; ++i) {
		int sgn;
		isl_seq_inner_product(ineq, tab->samples->row[i],
					1 + tab->n_var, &v);
		sgn = isl_int_sgn(v);
		if (eq ? (sgn == 0) : (sgn >= 0))
			break;
	}
	isl_int_clear(v);

	return i < tab->n_sample;
}

/* Insert a div specified by "div" to the tableau "tab" at position "pos" and
 * return isl_bool_true if the div is obviously non-negative.
 */
static isl_bool context_tab_insert_div(struct isl_tab *tab, int pos,
	__isl_keep isl_vec *div,
	int (*add_ineq)(void *user, isl_int *), void *user)
{
	int i;
	int r;
	struct isl_mat *samples;
	int nonneg;

	r = isl_tab_insert_div(tab, pos, div, add_ineq, user);
	if (r < 0)
		return isl_bool_error;
	nonneg = tab->var[r].is_nonneg;
	tab->var[r].frozen = 1;

	samples = isl_mat_extend(tab->samples,
			tab->n_sample, 1 + tab->n_var);
	tab->samples = samples;
	if (!samples)
		return isl_bool_error;
	for (i = tab->n_outside; i < samples->n_row; ++i) {
		isl_seq_inner_product(div->el + 1, samples->row[i],
			div->size - 1, &samples->row[i][samples->n_col - 1]);
		isl_int_fdiv_q(samples->row[i][samples->n_col - 1],
			       samples->row[i][samples->n_col - 1], div->el[0]);
	}
	tab->samples = isl_mat_move_cols(tab->samples, 1 + pos,
					1 + tab->n_var - 1, 1);
	if (!tab->samples)
		return isl_bool_error;

	return nonneg;
}

/* Add a div specified by "div" to both the main tableau and
 * the context tableau.  In case of the main tableau, we only
 * need to add an extra div.  In the context tableau, we also
 * need to express the meaning of the div.
 * Return the index of the div or -1 if anything went wrong.
 *
 * The new integer division is added before any unknown integer
 * divisions in the context to ensure that it does not get
 * equated to some linear combination involving unknown integer
 * divisions.
 */
static int add_div(struct isl_tab *tab, struct isl_context *context,
	__isl_keep isl_vec *div)
{
	int r;
	int pos;
	isl_bool nonneg;
	struct isl_tab *context_tab = context->op->peek_tab(context);

	if (!tab || !context_tab)
		goto error;

	pos = context_tab->n_var - context->n_unknown;
	if ((nonneg = context->op->insert_div(context, pos, div)) < 0)
		goto error;

	if (!context->op->is_ok(context))
		goto error;

	pos = tab->n_var - context->n_unknown;
	if (isl_tab_extend_vars(tab, 1) < 0)
		goto error;
	r = isl_tab_insert_var(tab, pos);
	if (r < 0)
		goto error;
	if (nonneg)
		tab->var[r].is_nonneg = 1;
	tab->var[r].frozen = 1;
	tab->n_div++;

	return tab->n_div - 1 - context->n_unknown;
error:
	context->op->invalidate(context);
	return -1;
}

static int find_div(struct isl_tab *tab, isl_int *div, isl_int denom)
{
	int i;
	unsigned total = isl_basic_map_total_dim(tab->bmap);

	for (i = 0; i < tab->bmap->n_div; ++i) {
		if (isl_int_ne(tab->bmap->div[i][0], denom))
			continue;
		if (!isl_seq_eq(tab->bmap->div[i] + 1, div, 1 + total))
			continue;
		return i;
	}
	return -1;
}

/* Return the index of a div that corresponds to "div".
 * We first check if we already have such a div and if not, we create one.
 */
static int get_div(struct isl_tab *tab, struct isl_context *context,
	struct isl_vec *div)
{
	int d;
	struct isl_tab *context_tab = context->op->peek_tab(context);

	if (!context_tab)
		return -1;

	d = find_div(context_tab, div->el + 1, div->el[0]);
	if (d != -1)
		return d;

	return add_div(tab, context, div);
}

/* Add a parametric cut to cut away the non-integral sample value
 * of the give row.
 * Let a_i be the coefficients of the constant term and the parameters
 * and let b_i be the coefficients of the variables or constraints
 * in basis of the tableau.
 * Let q be the div q = floor(\sum_i {-a_i} y_i).
 *
 * The cut is expressed as
 *
 *	c = \sum_i -{-a_i} y_i + \sum_i {b_i} x_i + q >= 0
 *
 * If q did not already exist in the context tableau, then it is added first.
 * If q is in a column of the main tableau then the "+ q" can be accomplished
 * by setting the corresponding entry to the denominator of the constraint.
 * If q happens to be in a row of the main tableau, then the corresponding
 * row needs to be added instead (taking care of the denominators).
 * Note that this is very unlikely, but perhaps not entirely impossible.
 *
 * The current value of the cut is known to be negative (or at least
 * non-positive), so row_sign is set accordingly.
 *
 * Return the row of the cut or -1.
 */
static int add_parametric_cut(struct isl_tab *tab, int row,
	struct isl_context *context)
{
	struct isl_vec *div;
	int d;
	int i;
	int r;
	isl_int *r_row;
	int col;
	int n;
	unsigned off = 2 + tab->M;

	if (!context)
		return -1;

	div = get_row_parameter_div(tab, row);
	if (!div)
		return -1;

	n = tab->n_div - context->n_unknown;
	d = context->op->get_div(context, tab, div);
	isl_vec_free(div);
	if (d < 0)
		return -1;

	if (isl_tab_extend_cons(tab, 1) < 0)
		return -1;
	r = isl_tab_allocate_con(tab);
	if (r < 0)
		return -1;

	r_row = tab->mat->row[tab->con[r].index];
	isl_int_set(r_row[0], tab->mat->row[row][0]);
	isl_int_neg(r_row[1], tab->mat->row[row][1]);
	isl_int_fdiv_r(r_row[1], r_row[1], tab->mat->row[row][0]);
	isl_int_neg(r_row[1], r_row[1]);
	if (tab->M)
		isl_int_set_si(r_row[2], 0);
	for (i = 0; i < tab->n_param; ++i) {
		if (tab->var[i].is_row)
			continue;
		col = tab->var[i].index;
		isl_int_neg(r_row[off + col], tab->mat->row[row][off + col]);
		isl_int_fdiv_r(r_row[off + col], r_row[off + col],
				tab->mat->row[row][0]);
		isl_int_neg(r_row[off + col], r_row[off + col]);
	}
	for (i = 0; i < tab->n_div; ++i) {
		if (tab->var[tab->n_var - tab->n_div + i].is_row)
			continue;
		col = tab->var[tab->n_var - tab->n_div + i].index;
		isl_int_neg(r_row[off + col], tab->mat->row[row][off + col]);
		isl_int_fdiv_r(r_row[off + col], r_row[off + col],
				tab->mat->row[row][0]);
		isl_int_neg(r_row[off + col], r_row[off + col]);
	}
	for (i = 0; i < tab->n_col; ++i) {
		if (tab->col_var[i] >= 0 &&
		    (tab->col_var[i] < tab->n_param ||
		     tab->col_var[i] >= tab->n_var - tab->n_div))
			continue;
		isl_int_fdiv_r(r_row[off + i],
			tab->mat->row[row][off + i], tab->mat->row[row][0]);
	}
	if (tab->var[tab->n_var - tab->n_div + d].is_row) {
		isl_int gcd;
		int d_row = tab->var[tab->n_var - tab->n_div + d].index;
		isl_int_init(gcd);
		isl_int_gcd(gcd, tab->mat->row[d_row][0], r_row[0]);
		isl_int_divexact(r_row[0], r_row[0], gcd);
		isl_int_divexact(gcd, tab->mat->row[d_row][0], gcd);
		isl_seq_combine(r_row + 1, gcd, r_row + 1,
				r_row[0], tab->mat->row[d_row] + 1,
				off - 1 + tab->n_col);
		isl_int_mul(r_row[0], r_row[0], tab->mat->row[d_row][0]);
		isl_int_clear(gcd);
	} else {
		col = tab->var[tab->n_var - tab->n_div + d].index;
		isl_int_set(r_row[off + col], tab->mat->row[row][0]);
	}

	tab->con[r].is_nonneg = 1;
	if (isl_tab_push_var(tab, isl_tab_undo_nonneg, &tab->con[r]) < 0)
		return -1;
	if (tab->row_sign)
		tab->row_sign[tab->con[r].index] = isl_tab_row_neg;

	row = tab->con[r].index;

	if (d >= n && context->op->detect_equalities(context, tab) < 0)
		return -1;

	return row;
}

/* Construct a tableau for bmap that can be used for computing
 * the lexicographic minimum (or maximum) of bmap.
 * If not NULL, then dom is the domain where the minimum
 * should be computed.  In this case, we set up a parametric
 * tableau with row signs (initialized to "unknown").
 * If M is set, then the tableau will use a big parameter.
 * If max is set, then a maximum should be computed instead of a minimum.
 * This means that for each variable x, the tableau will contain the variable
 * x' = M - x, rather than x' = M + x.  This in turn means that the coefficient
 * of the variables in all constraints are negated prior to adding them
 * to the tableau.
 */
static struct isl_tab *tab_for_lexmin(struct isl_basic_map *bmap,
	struct isl_basic_set *dom, unsigned M, int max)
{
	int i;
	struct isl_tab *tab;
	unsigned n_var;
	unsigned o_var;

	tab = isl_tab_alloc(bmap->ctx, 2 * bmap->n_eq + bmap->n_ineq + 1,
			    isl_basic_map_total_dim(bmap), M);
	if (!tab)
		return NULL;

	tab->rational = ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL);
	if (dom) {
		tab->n_param = isl_basic_set_total_dim(dom) - dom->n_div;
		tab->n_div = dom->n_div;
		tab->row_sign = isl_calloc_array(bmap->ctx,
					enum isl_tab_row_sign, tab->mat->n_row);
		if (tab->mat->n_row && !tab->row_sign)
			goto error;
	}
	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_EMPTY)) {
		if (isl_tab_mark_empty(tab) < 0)
			goto error;
		return tab;
	}

	for (i = tab->n_param; i < tab->n_var - tab->n_div; ++i) {
		tab->var[i].is_nonneg = 1;
		tab->var[i].frozen = 1;
	}
	o_var = 1 + tab->n_param;
	n_var = tab->n_var - tab->n_param - tab->n_div;
	for (i = 0; i < bmap->n_eq; ++i) {
		if (max)
			isl_seq_neg(bmap->eq[i] + o_var,
				    bmap->eq[i] + o_var, n_var);
		tab = add_lexmin_valid_eq(tab, bmap->eq[i]);
		if (max)
			isl_seq_neg(bmap->eq[i] + o_var,
				    bmap->eq[i] + o_var, n_var);
		if (!tab || tab->empty)
			return tab;
	}
	if (bmap->n_eq && restore_lexmin(tab) < 0)
		goto error;
	for (i = 0; i < bmap->n_ineq; ++i) {
		if (max)
			isl_seq_neg(bmap->ineq[i] + o_var,
				    bmap->ineq[i] + o_var, n_var);
		tab = add_lexmin_ineq(tab, bmap->ineq[i]);
		if (max)
			isl_seq_neg(bmap->ineq[i] + o_var,
				    bmap->ineq[i] + o_var, n_var);
		if (!tab || tab->empty)
			return tab;
	}
	return tab;
error:
	isl_tab_free(tab);
	return NULL;
}

/* Given a main tableau where more than one row requires a split,
 * determine and return the "best" row to split on.
 *
 * Given two rows in the main tableau, if the inequality corresponding
 * to the first row is redundant with respect to that of the second row
 * in the current tableau, then it is better to split on the second row,
 * since in the positive part, both rows will be positive.
 * (In the negative part a pivot will have to be performed and just about
 * anything can happen to the sign of the other row.)
 *
 * As a simple heuristic, we therefore select the row that makes the most
 * of the other rows redundant.
 *
 * Perhaps it would also be useful to look at the number of constraints
 * that conflict with any given constraint.
 *
 * best is the best row so far (-1 when we have not found any row yet).
 * best_r is the number of other rows made redundant by row best.
 * When best is still -1, bset_r is meaningless, but it is initialized
 * to some arbitrary value (0) anyway.  Without this redundant initialization
 * valgrind may warn about uninitialized memory accesses when isl
 * is compiled with some versions of gcc.
 */
static int best_split(struct isl_tab *tab, struct isl_tab *context_tab)
{
	struct isl_tab_undo *snap;
	int split;
	int row;
	int best = -1;
	int best_r = 0;

	if (isl_tab_extend_cons(context_tab, 2) < 0)
		return -1;

	snap = isl_tab_snap(context_tab);

	for (split = tab->n_redundant; split < tab->n_row; ++split) {
		struct isl_tab_undo *snap2;
		struct isl_vec *ineq = NULL;
		int r = 0;
		int ok;

		if (!isl_tab_var_from_row(tab, split)->is_nonneg)
			continue;
		if (tab->row_sign[split] != isl_tab_row_any)
			continue;

		ineq = get_row_parameter_ineq(tab, split);
		if (!ineq)
			return -1;
		ok = isl_tab_add_ineq(context_tab, ineq->el) >= 0;
		isl_vec_free(ineq);
		if (!ok)
			return -1;

		snap2 = isl_tab_snap(context_tab);

		for (row = tab->n_redundant; row < tab->n_row; ++row) {
			struct isl_tab_var *var;

			if (row == split)
				continue;
			if (!isl_tab_var_from_row(tab, row)->is_nonneg)
				continue;
			if (tab->row_sign[row] != isl_tab_row_any)
				continue;

			ineq = get_row_parameter_ineq(tab, row);
			if (!ineq)
				return -1;
			ok = isl_tab_add_ineq(context_tab, ineq->el) >= 0;
			isl_vec_free(ineq);
			if (!ok)
				return -1;
			var = &context_tab->con[context_tab->n_con - 1];
			if (!context_tab->empty &&
			    !isl_tab_min_at_most_neg_one(context_tab, var))
				r++;
			if (isl_tab_rollback(context_tab, snap2) < 0)
				return -1;
		}
		if (best == -1 || r > best_r) {
			best = split;
			best_r = r;
		}
		if (isl_tab_rollback(context_tab, snap) < 0)
			return -1;
	}

	return best;
}

static struct isl_basic_set *context_lex_peek_basic_set(
	struct isl_context *context)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	if (!clex->tab)
		return NULL;
	return isl_tab_peek_bset(clex->tab);
}

static struct isl_tab *context_lex_peek_tab(struct isl_context *context)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	return clex->tab;
}

static void context_lex_add_eq(struct isl_context *context, isl_int *eq,
		int check, int update)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	if (isl_tab_extend_cons(clex->tab, 2) < 0)
		goto error;
	if (add_lexmin_eq(clex->tab, eq) < 0)
		goto error;
	if (check) {
		int v = tab_has_valid_sample(clex->tab, eq, 1);
		if (v < 0)
			goto error;
		if (!v)
			clex->tab = check_integer_feasible(clex->tab);
	}
	if (update)
		clex->tab = check_samples(clex->tab, eq, 1);
	return;
error:
	isl_tab_free(clex->tab);
	clex->tab = NULL;
}

static void context_lex_add_ineq(struct isl_context *context, isl_int *ineq,
		int check, int update)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	if (isl_tab_extend_cons(clex->tab, 1) < 0)
		goto error;
	clex->tab = add_lexmin_ineq(clex->tab, ineq);
	if (check) {
		int v = tab_has_valid_sample(clex->tab, ineq, 0);
		if (v < 0)
			goto error;
		if (!v)
			clex->tab = check_integer_feasible(clex->tab);
	}
	if (update)
		clex->tab = check_samples(clex->tab, ineq, 0);
	return;
error:
	isl_tab_free(clex->tab);
	clex->tab = NULL;
}

static int context_lex_add_ineq_wrap(void *user, isl_int *ineq)
{
	struct isl_context *context = (struct isl_context *)user;
	context_lex_add_ineq(context, ineq, 0, 0);
	return context->op->is_ok(context) ? 0 : -1;
}

/* Check which signs can be obtained by "ineq" on all the currently
 * active sample values.  See row_sign for more information.
 */
static enum isl_tab_row_sign tab_ineq_sign(struct isl_tab *tab, isl_int *ineq,
	int strict)
{
	int i;
	int sgn;
	isl_int tmp;
	enum isl_tab_row_sign res = isl_tab_row_unknown;

	isl_assert(tab->mat->ctx, tab->samples, return isl_tab_row_unknown);
	isl_assert(tab->mat->ctx, tab->samples->n_col == 1 + tab->n_var,
			return isl_tab_row_unknown);

	isl_int_init(tmp);
	for (i = tab->n_outside; i < tab->n_sample; ++i) {
		isl_seq_inner_product(tab->samples->row[i], ineq,
					1 + tab->n_var, &tmp);
		sgn = isl_int_sgn(tmp);
		if (sgn > 0 || (sgn == 0 && strict)) {
			if (res == isl_tab_row_unknown)
				res = isl_tab_row_pos;
			if (res == isl_tab_row_neg)
				res = isl_tab_row_any;
		}
		if (sgn < 0) {
			if (res == isl_tab_row_unknown)
				res = isl_tab_row_neg;
			if (res == isl_tab_row_pos)
				res = isl_tab_row_any;
		}
		if (res == isl_tab_row_any)
			break;
	}
	isl_int_clear(tmp);

	return res;
}

static enum isl_tab_row_sign context_lex_ineq_sign(struct isl_context *context,
			isl_int *ineq, int strict)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	return tab_ineq_sign(clex->tab, ineq, strict);
}

/* Check whether "ineq" can be added to the tableau without rendering
 * it infeasible.
 */
static int context_lex_test_ineq(struct isl_context *context, isl_int *ineq)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	struct isl_tab_undo *snap;
	int feasible;

	if (!clex->tab)
		return -1;

	if (isl_tab_extend_cons(clex->tab, 1) < 0)
		return -1;

	snap = isl_tab_snap(clex->tab);
	if (isl_tab_push_basis(clex->tab) < 0)
		return -1;
	clex->tab = add_lexmin_ineq(clex->tab, ineq);
	clex->tab = check_integer_feasible(clex->tab);
	if (!clex->tab)
		return -1;
	feasible = !clex->tab->empty;
	if (isl_tab_rollback(clex->tab, snap) < 0)
		return -1;

	return feasible;
}

static int context_lex_get_div(struct isl_context *context, struct isl_tab *tab,
		struct isl_vec *div)
{
	return get_div(tab, context, div);
}

/* Insert a div specified by "div" to the context tableau at position "pos" and
 * return isl_bool_true if the div is obviously non-negative.
 * context_tab_add_div will always return isl_bool_true, because all variables
 * in a isl_context_lex tableau are non-negative.
 * However, if we are using a big parameter in the context, then this only
 * reflects the non-negativity of the variable used to _encode_ the
 * div, i.e., div' = M + div, so we can't draw any conclusions.
 */
static isl_bool context_lex_insert_div(struct isl_context *context, int pos,
	__isl_keep isl_vec *div)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	isl_bool nonneg;
	nonneg = context_tab_insert_div(clex->tab, pos, div,
					context_lex_add_ineq_wrap, context);
	if (nonneg < 0)
		return isl_bool_error;
	if (clex->tab->M)
		return isl_bool_false;
	return nonneg;
}

static int context_lex_detect_equalities(struct isl_context *context,
		struct isl_tab *tab)
{
	return 0;
}

static int context_lex_best_split(struct isl_context *context,
		struct isl_tab *tab)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	struct isl_tab_undo *snap;
	int r;

	snap = isl_tab_snap(clex->tab);
	if (isl_tab_push_basis(clex->tab) < 0)
		return -1;
	r = best_split(tab, clex->tab);

	if (r >= 0 && isl_tab_rollback(clex->tab, snap) < 0)
		return -1;

	return r;
}

static int context_lex_is_empty(struct isl_context *context)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	if (!clex->tab)
		return -1;
	return clex->tab->empty;
}

static void *context_lex_save(struct isl_context *context)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	struct isl_tab_undo *snap;

	snap = isl_tab_snap(clex->tab);
	if (isl_tab_push_basis(clex->tab) < 0)
		return NULL;
	if (isl_tab_save_samples(clex->tab) < 0)
		return NULL;

	return snap;
}

static void context_lex_restore(struct isl_context *context, void *save)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	if (isl_tab_rollback(clex->tab, (struct isl_tab_undo *)save) < 0) {
		isl_tab_free(clex->tab);
		clex->tab = NULL;
	}
}

static void context_lex_discard(void *save)
{
}

static int context_lex_is_ok(struct isl_context *context)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	return !!clex->tab;
}

/* For each variable in the context tableau, check if the variable can
 * only attain non-negative values.  If so, mark the parameter as non-negative
 * in the main tableau.  This allows for a more direct identification of some
 * cases of violated constraints.
 */
static struct isl_tab *tab_detect_nonnegative_parameters(struct isl_tab *tab,
	struct isl_tab *context_tab)
{
	int i;
	struct isl_tab_undo *snap;
	struct isl_vec *ineq = NULL;
	struct isl_tab_var *var;
	int n;

	if (context_tab->n_var == 0)
		return tab;

	ineq = isl_vec_alloc(tab->mat->ctx, 1 + context_tab->n_var);
	if (!ineq)
		goto error;

	if (isl_tab_extend_cons(context_tab, 1) < 0)
		goto error;

	snap = isl_tab_snap(context_tab);

	n = 0;
	isl_seq_clr(ineq->el, ineq->size);
	for (i = 0; i < context_tab->n_var; ++i) {
		isl_int_set_si(ineq->el[1 + i], 1);
		if (isl_tab_add_ineq(context_tab, ineq->el) < 0)
			goto error;
		var = &context_tab->con[context_tab->n_con - 1];
		if (!context_tab->empty &&
		    !isl_tab_min_at_most_neg_one(context_tab, var)) {
			int j = i;
			if (i >= tab->n_param)
				j = i - tab->n_param + tab->n_var - tab->n_div;
			tab->var[j].is_nonneg = 1;
			n++;
		}
		isl_int_set_si(ineq->el[1 + i], 0);
		if (isl_tab_rollback(context_tab, snap) < 0)
			goto error;
	}

	if (context_tab->M && n == context_tab->n_var) {
		context_tab->mat = isl_mat_drop_cols(context_tab->mat, 2, 1);
		context_tab->M = 0;
	}

	isl_vec_free(ineq);
	return tab;
error:
	isl_vec_free(ineq);
	isl_tab_free(tab);
	return NULL;
}

static struct isl_tab *context_lex_detect_nonnegative_parameters(
	struct isl_context *context, struct isl_tab *tab)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	struct isl_tab_undo *snap;

	if (!tab)
		return NULL;

	snap = isl_tab_snap(clex->tab);
	if (isl_tab_push_basis(clex->tab) < 0)
		goto error;

	tab = tab_detect_nonnegative_parameters(tab, clex->tab);

	if (isl_tab_rollback(clex->tab, snap) < 0)
		goto error;

	return tab;
error:
	isl_tab_free(tab);
	return NULL;
}

static void context_lex_invalidate(struct isl_context *context)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	isl_tab_free(clex->tab);
	clex->tab = NULL;
}

static __isl_null struct isl_context *context_lex_free(
	struct isl_context *context)
{
	struct isl_context_lex *clex = (struct isl_context_lex *)context;
	isl_tab_free(clex->tab);
	free(clex);

	return NULL;
}

struct isl_context_op isl_context_lex_op = {
	context_lex_detect_nonnegative_parameters,
	context_lex_peek_basic_set,
	context_lex_peek_tab,
	context_lex_add_eq,
	context_lex_add_ineq,
	context_lex_ineq_sign,
	context_lex_test_ineq,
	context_lex_get_div,
	context_lex_insert_div,
	context_lex_detect_equalities,
	context_lex_best_split,
	context_lex_is_empty,
	context_lex_is_ok,
	context_lex_save,
	context_lex_restore,
	context_lex_discard,
	context_lex_invalidate,
	context_lex_free,
};

static struct isl_tab *context_tab_for_lexmin(struct isl_basic_set *bset)
{
	struct isl_tab *tab;

	if (!bset)
		return NULL;
	tab = tab_for_lexmin(bset_to_bmap(bset), NULL, 1, 0);
	if (!tab)
		goto error;
	if (isl_tab_track_bset(tab, bset) < 0)
		goto error;
	tab = isl_tab_init_samples(tab);
	return tab;
error:
	isl_basic_set_free(bset);
	return NULL;
}

static struct isl_context *isl_context_lex_alloc(struct isl_basic_set *dom)
{
	struct isl_context_lex *clex;

	if (!dom)
		return NULL;

	clex = isl_alloc_type(dom->ctx, struct isl_context_lex);
	if (!clex)
		return NULL;

	clex->context.op = &isl_context_lex_op;

	clex->tab = context_tab_for_lexmin(isl_basic_set_copy(dom));
	if (restore_lexmin(clex->tab) < 0)
		goto error;
	clex->tab = check_integer_feasible(clex->tab);
	if (!clex->tab)
		goto error;

	return &clex->context;
error:
	clex->context.op->free(&clex->context);
	return NULL;
}

/* Representation of the context when using generalized basis reduction.
 *
 * "shifted" contains the offsets of the unit hypercubes that lie inside the
 * context.  Any rational point in "shifted" can therefore be rounded
 * up to an integer point in the context.
 * If the context is constrained by any equality, then "shifted" is not used
 * as it would be empty.
 */
struct isl_context_gbr {
	struct isl_context context;
	struct isl_tab *tab;
	struct isl_tab *shifted;
	struct isl_tab *cone;
};

static struct isl_tab *context_gbr_detect_nonnegative_parameters(
	struct isl_context *context, struct isl_tab *tab)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	if (!tab)
		return NULL;
	return tab_detect_nonnegative_parameters(tab, cgbr->tab);
}

static struct isl_basic_set *context_gbr_peek_basic_set(
	struct isl_context *context)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	if (!cgbr->tab)
		return NULL;
	return isl_tab_peek_bset(cgbr->tab);
}

static struct isl_tab *context_gbr_peek_tab(struct isl_context *context)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	return cgbr->tab;
}

/* Initialize the "shifted" tableau of the context, which
 * contains the constraints of the original tableau shifted
 * by the sum of all negative coefficients.  This ensures
 * that any rational point in the shifted tableau can
 * be rounded up to yield an integer point in the original tableau.
 */
static void gbr_init_shifted(struct isl_context_gbr *cgbr)
{
	int i, j;
	struct isl_vec *cst;
	struct isl_basic_set *bset = isl_tab_peek_bset(cgbr->tab);
	unsigned dim = isl_basic_set_total_dim(bset);

	cst = isl_vec_alloc(cgbr->tab->mat->ctx, bset->n_ineq);
	if (!cst)
		return;

	for (i = 0; i < bset->n_ineq; ++i) {
		isl_int_set(cst->el[i], bset->ineq[i][0]);
		for (j = 0; j < dim; ++j) {
			if (!isl_int_is_neg(bset->ineq[i][1 + j]))
				continue;
			isl_int_add(bset->ineq[i][0], bset->ineq[i][0],
				    bset->ineq[i][1 + j]);
		}
	}

	cgbr->shifted = isl_tab_from_basic_set(bset, 0);

	for (i = 0; i < bset->n_ineq; ++i)
		isl_int_set(bset->ineq[i][0], cst->el[i]);

	isl_vec_free(cst);
}

/* Check if the shifted tableau is non-empty, and if so
 * use the sample point to construct an integer point
 * of the context tableau.
 */
static struct isl_vec *gbr_get_shifted_sample(struct isl_context_gbr *cgbr)
{
	struct isl_vec *sample;

	if (!cgbr->shifted)
		gbr_init_shifted(cgbr);
	if (!cgbr->shifted)
		return NULL;
	if (cgbr->shifted->empty)
		return isl_vec_alloc(cgbr->tab->mat->ctx, 0);

	sample = isl_tab_get_sample_value(cgbr->shifted);
	sample = isl_vec_ceil(sample);

	return sample;
}

static struct isl_basic_set *drop_constant_terms(struct isl_basic_set *bset)
{
	int i;

	if (!bset)
		return NULL;

	for (i = 0; i < bset->n_eq; ++i)
		isl_int_set_si(bset->eq[i][0], 0);

	for (i = 0; i < bset->n_ineq; ++i)
		isl_int_set_si(bset->ineq[i][0], 0);

	return bset;
}

static int use_shifted(struct isl_context_gbr *cgbr)
{
	if (!cgbr->tab)
		return 0;
	return cgbr->tab->bmap->n_eq == 0 && cgbr->tab->bmap->n_div == 0;
}

static struct isl_vec *gbr_get_sample(struct isl_context_gbr *cgbr)
{
	struct isl_basic_set *bset;
	struct isl_basic_set *cone;

	if (isl_tab_sample_is_integer(cgbr->tab))
		return isl_tab_get_sample_value(cgbr->tab);

	if (use_shifted(cgbr)) {
		struct isl_vec *sample;

		sample = gbr_get_shifted_sample(cgbr);
		if (!sample || sample->size > 0)
			return sample;

		isl_vec_free(sample);
	}

	if (!cgbr->cone) {
		bset = isl_tab_peek_bset(cgbr->tab);
		cgbr->cone = isl_tab_from_recession_cone(bset, 0);
		if (!cgbr->cone)
			return NULL;
		if (isl_tab_track_bset(cgbr->cone,
					isl_basic_set_copy(bset)) < 0)
			return NULL;
	}
	if (isl_tab_detect_implicit_equalities(cgbr->cone) < 0)
		return NULL;

	if (cgbr->cone->n_dead == cgbr->cone->n_col) {
		struct isl_vec *sample;
		struct isl_tab_undo *snap;

		if (cgbr->tab->basis) {
			if (cgbr->tab->basis->n_col != 1 + cgbr->tab->n_var) {
				isl_mat_free(cgbr->tab->basis);
				cgbr->tab->basis = NULL;
			}
			cgbr->tab->n_zero = 0;
			cgbr->tab->n_unbounded = 0;
		}

		snap = isl_tab_snap(cgbr->tab);

		sample = isl_tab_sample(cgbr->tab);

		if (!sample || isl_tab_rollback(cgbr->tab, snap) < 0) {
			isl_vec_free(sample);
			return NULL;
		}

		return sample;
	}

	cone = isl_basic_set_dup(isl_tab_peek_bset(cgbr->cone));
	cone = drop_constant_terms(cone);
	cone = isl_basic_set_update_from_tab(cone, cgbr->cone);
	cone = isl_basic_set_underlying_set(cone);
	cone = isl_basic_set_gauss(cone, NULL);

	bset = isl_basic_set_dup(isl_tab_peek_bset(cgbr->tab));
	bset = isl_basic_set_update_from_tab(bset, cgbr->tab);
	bset = isl_basic_set_underlying_set(bset);
	bset = isl_basic_set_gauss(bset, NULL);

	return isl_basic_set_sample_with_cone(bset, cone);
}

static void check_gbr_integer_feasible(struct isl_context_gbr *cgbr)
{
	struct isl_vec *sample;

	if (!cgbr->tab)
		return;

	if (cgbr->tab->empty)
		return;

	sample = gbr_get_sample(cgbr);
	if (!sample)
		goto error;

	if (sample->size == 0) {
		isl_vec_free(sample);
		if (isl_tab_mark_empty(cgbr->tab) < 0)
			goto error;
		return;
	}

	if (isl_tab_add_sample(cgbr->tab, sample) < 0)
		goto error;

	return;
error:
	isl_tab_free(cgbr->tab);
	cgbr->tab = NULL;
}

static struct isl_tab *add_gbr_eq(struct isl_tab *tab, isl_int *eq)
{
	if (!tab)
		return NULL;

	if (isl_tab_extend_cons(tab, 2) < 0)
		goto error;

	if (isl_tab_add_eq(tab, eq) < 0)
		goto error;

	return tab;
error:
	isl_tab_free(tab);
	return NULL;
}

/* Add the equality described by "eq" to the context.
 * If "check" is set, then we check if the context is empty after
 * adding the equality.
 * If "update" is set, then we check if the samples are still valid.
 *
 * We do not explicitly add shifted copies of the equality to
 * cgbr->shifted since they would conflict with each other.
 * Instead, we directly mark cgbr->shifted empty.
 */
static void context_gbr_add_eq(struct isl_context *context, isl_int *eq,
		int check, int update)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;

	cgbr->tab = add_gbr_eq(cgbr->tab, eq);

	if (cgbr->shifted && !cgbr->shifted->empty && use_shifted(cgbr)) {
		if (isl_tab_mark_empty(cgbr->shifted) < 0)
			goto error;
	}

	if (cgbr->cone && cgbr->cone->n_col != cgbr->cone->n_dead) {
		if (isl_tab_extend_cons(cgbr->cone, 2) < 0)
			goto error;
		if (isl_tab_add_eq(cgbr->cone, eq) < 0)
			goto error;
	}

	if (check) {
		int v = tab_has_valid_sample(cgbr->tab, eq, 1);
		if (v < 0)
			goto error;
		if (!v)
			check_gbr_integer_feasible(cgbr);
	}
	if (update)
		cgbr->tab = check_samples(cgbr->tab, eq, 1);
	return;
error:
	isl_tab_free(cgbr->tab);
	cgbr->tab = NULL;
}

static void add_gbr_ineq(struct isl_context_gbr *cgbr, isl_int *ineq)
{
	if (!cgbr->tab)
		return;

	if (isl_tab_extend_cons(cgbr->tab, 1) < 0)
		goto error;

	if (isl_tab_add_ineq(cgbr->tab, ineq) < 0)
		goto error;

	if (cgbr->shifted && !cgbr->shifted->empty && use_shifted(cgbr)) {
		int i;
		unsigned dim;
		dim = isl_basic_map_total_dim(cgbr->tab->bmap);

		if (isl_tab_extend_cons(cgbr->shifted, 1) < 0)
			goto error;

		for (i = 0; i < dim; ++i) {
			if (!isl_int_is_neg(ineq[1 + i]))
				continue;
			isl_int_add(ineq[0], ineq[0], ineq[1 + i]);
		}

		if (isl_tab_add_ineq(cgbr->shifted, ineq) < 0)
			goto error;

		for (i = 0; i < dim; ++i) {
			if (!isl_int_is_neg(ineq[1 + i]))
				continue;
			isl_int_sub(ineq[0], ineq[0], ineq[1 + i]);
		}
	}

	if (cgbr->cone && cgbr->cone->n_col != cgbr->cone->n_dead) {
		if (isl_tab_extend_cons(cgbr->cone, 1) < 0)
			goto error;
		if (isl_tab_add_ineq(cgbr->cone, ineq) < 0)
			goto error;
	}

	return;
error:
	isl_tab_free(cgbr->tab);
	cgbr->tab = NULL;
}

static void context_gbr_add_ineq(struct isl_context *context, isl_int *ineq,
		int check, int update)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;

	add_gbr_ineq(cgbr, ineq);
	if (!cgbr->tab)
		return;

	if (check) {
		int v = tab_has_valid_sample(cgbr->tab, ineq, 0);
		if (v < 0)
			goto error;
		if (!v)
			check_gbr_integer_feasible(cgbr);
	}
	if (update)
		cgbr->tab = check_samples(cgbr->tab, ineq, 0);
	return;
error:
	isl_tab_free(cgbr->tab);
	cgbr->tab = NULL;
}

static int context_gbr_add_ineq_wrap(void *user, isl_int *ineq)
{
	struct isl_context *context = (struct isl_context *)user;
	context_gbr_add_ineq(context, ineq, 0, 0);
	return context->op->is_ok(context) ? 0 : -1;
}

static enum isl_tab_row_sign context_gbr_ineq_sign(struct isl_context *context,
			isl_int *ineq, int strict)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	return tab_ineq_sign(cgbr->tab, ineq, strict);
}

/* Check whether "ineq" can be added to the tableau without rendering
 * it infeasible.
 */
static int context_gbr_test_ineq(struct isl_context *context, isl_int *ineq)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	struct isl_tab_undo *snap;
	struct isl_tab_undo *shifted_snap = NULL;
	struct isl_tab_undo *cone_snap = NULL;
	int feasible;

	if (!cgbr->tab)
		return -1;

	if (isl_tab_extend_cons(cgbr->tab, 1) < 0)
		return -1;

	snap = isl_tab_snap(cgbr->tab);
	if (cgbr->shifted)
		shifted_snap = isl_tab_snap(cgbr->shifted);
	if (cgbr->cone)
		cone_snap = isl_tab_snap(cgbr->cone);
	add_gbr_ineq(cgbr, ineq);
	check_gbr_integer_feasible(cgbr);
	if (!cgbr->tab)
		return -1;
	feasible = !cgbr->tab->empty;
	if (isl_tab_rollback(cgbr->tab, snap) < 0)
		return -1;
	if (shifted_snap) {
		if (isl_tab_rollback(cgbr->shifted, shifted_snap))
			return -1;
	} else if (cgbr->shifted) {
		isl_tab_free(cgbr->shifted);
		cgbr->shifted = NULL;
	}
	if (cone_snap) {
		if (isl_tab_rollback(cgbr->cone, cone_snap))
			return -1;
	} else if (cgbr->cone) {
		isl_tab_free(cgbr->cone);
		cgbr->cone = NULL;
	}

	return feasible;
}

/* Return the column of the last of the variables associated to
 * a column that has a non-zero coefficient.
 * This function is called in a context where only coefficients
 * of parameters or divs can be non-zero.
 */
static int last_non_zero_var_col(struct isl_tab *tab, isl_int *p)
{
	int i;
	int col;

	if (tab->n_var == 0)
		return -1;

	for (i = tab->n_var - 1; i >= 0; --i) {
		if (i >= tab->n_param && i < tab->n_var - tab->n_div)
			continue;
		if (tab->var[i].is_row)
			continue;
		col = tab->var[i].index;
		if (!isl_int_is_zero(p[col]))
			return col;
	}

	return -1;
}

/* Look through all the recently added equalities in the context
 * to see if we can propagate any of them to the main tableau.
 *
 * The newly added equalities in the context are encoded as pairs
 * of inequalities starting at inequality "first".
 *
 * We tentatively add each of these equalities to the main tableau
 * and if this happens to result in a row with a final coefficient
 * that is one or negative one, we use it to kill a column
 * in the main tableau.  Otherwise, we discard the tentatively
 * added row.
 * This tentative addition of equality constraints turns
 * on the undo facility of the tableau.  Turn it off again
 * at the end, assuming it was turned off to begin with.
 *
 * Return 0 on success and -1 on failure.
 */
static int propagate_equalities(struct isl_context_gbr *cgbr,
	struct isl_tab *tab, unsigned first)
{
	int i;
	struct isl_vec *eq = NULL;
	isl_bool needs_undo;

	needs_undo = isl_tab_need_undo(tab);
	if (needs_undo < 0)
		goto error;
	eq = isl_vec_alloc(tab->mat->ctx, 1 + tab->n_var);
	if (!eq)
		goto error;

	if (isl_tab_extend_cons(tab, (cgbr->tab->bmap->n_ineq - first)/2) < 0)
		goto error;

	isl_seq_clr(eq->el + 1 + tab->n_param,
		    tab->n_var - tab->n_param - tab->n_div);
	for (i = first; i < cgbr->tab->bmap->n_ineq; i += 2) {
		int j;
		int r;
		struct isl_tab_undo *snap;
		snap = isl_tab_snap(tab);

		isl_seq_cpy(eq->el, cgbr->tab->bmap->ineq[i], 1 + tab->n_param);
		isl_seq_cpy(eq->el + 1 + tab->n_var - tab->n_div,
			    cgbr->tab->bmap->ineq[i] + 1 + tab->n_param,
			    tab->n_div);

		r = isl_tab_add_row(tab, eq->el);
		if (r < 0)
			goto error;
		r = tab->con[r].index;
		j = last_non_zero_var_col(tab, tab->mat->row[r] + 2 + tab->M);
		if (j < 0 || j < tab->n_dead ||
		    !isl_int_is_one(tab->mat->row[r][0]) ||
		    (!isl_int_is_one(tab->mat->row[r][2 + tab->M + j]) &&
		     !isl_int_is_negone(tab->mat->row[r][2 + tab->M + j]))) {
			if (isl_tab_rollback(tab, snap) < 0)
				goto error;
			continue;
		}
		if (isl_tab_pivot(tab, r, j) < 0)
			goto error;
		if (isl_tab_kill_col(tab, j) < 0)
			goto error;

		if (restore_lexmin(tab) < 0)
			goto error;
	}

	if (!needs_undo)
		isl_tab_clear_undo(tab);
	isl_vec_free(eq);

	return 0;
error:
	isl_vec_free(eq);
	isl_tab_free(cgbr->tab);
	cgbr->tab = NULL;
	return -1;
}

static int context_gbr_detect_equalities(struct isl_context *context,
	struct isl_tab *tab)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	unsigned n_ineq;

	if (!cgbr->cone) {
		struct isl_basic_set *bset = isl_tab_peek_bset(cgbr->tab);
		cgbr->cone = isl_tab_from_recession_cone(bset, 0);
		if (!cgbr->cone)
			goto error;
		if (isl_tab_track_bset(cgbr->cone,
					isl_basic_set_copy(bset)) < 0)
			goto error;
	}
	if (isl_tab_detect_implicit_equalities(cgbr->cone) < 0)
		goto error;

	n_ineq = cgbr->tab->bmap->n_ineq;
	cgbr->tab = isl_tab_detect_equalities(cgbr->tab, cgbr->cone);
	if (!cgbr->tab)
		return -1;
	if (cgbr->tab->bmap->n_ineq > n_ineq &&
	    propagate_equalities(cgbr, tab, n_ineq) < 0)
		return -1;

	return 0;
error:
	isl_tab_free(cgbr->tab);
	cgbr->tab = NULL;
	return -1;
}

static int context_gbr_get_div(struct isl_context *context, struct isl_tab *tab,
		struct isl_vec *div)
{
	return get_div(tab, context, div);
}

static isl_bool context_gbr_insert_div(struct isl_context *context, int pos,
	__isl_keep isl_vec *div)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	if (cgbr->cone) {
		int r, n_div, o_div;

		n_div = isl_basic_map_dim(cgbr->cone->bmap, isl_dim_div);
		o_div = cgbr->cone->n_var - n_div;

		if (isl_tab_extend_cons(cgbr->cone, 3) < 0)
			return isl_bool_error;
		if (isl_tab_extend_vars(cgbr->cone, 1) < 0)
			return isl_bool_error;
		if ((r = isl_tab_insert_var(cgbr->cone, pos)) <0)
			return isl_bool_error;

		cgbr->cone->bmap = isl_basic_map_insert_div(cgbr->cone->bmap,
						    r - o_div, div);
		if (!cgbr->cone->bmap)
			return isl_bool_error;
		if (isl_tab_push_var(cgbr->cone, isl_tab_undo_bmap_div,
				    &cgbr->cone->var[r]) < 0)
			return isl_bool_error;
	}
	return context_tab_insert_div(cgbr->tab, pos, div,
					context_gbr_add_ineq_wrap, context);
}

static int context_gbr_best_split(struct isl_context *context,
		struct isl_tab *tab)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	struct isl_tab_undo *snap;
	int r;

	snap = isl_tab_snap(cgbr->tab);
	r = best_split(tab, cgbr->tab);

	if (r >= 0 && isl_tab_rollback(cgbr->tab, snap) < 0)
		return -1;

	return r;
}

static int context_gbr_is_empty(struct isl_context *context)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	if (!cgbr->tab)
		return -1;
	return cgbr->tab->empty;
}

struct isl_gbr_tab_undo {
	struct isl_tab_undo *tab_snap;
	struct isl_tab_undo *shifted_snap;
	struct isl_tab_undo *cone_snap;
};

static void *context_gbr_save(struct isl_context *context)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	struct isl_gbr_tab_undo *snap;

	if (!cgbr->tab)
		return NULL;

	snap = isl_alloc_type(cgbr->tab->mat->ctx, struct isl_gbr_tab_undo);
	if (!snap)
		return NULL;

	snap->tab_snap = isl_tab_snap(cgbr->tab);
	if (isl_tab_save_samples(cgbr->tab) < 0)
		goto error;

	if (cgbr->shifted)
		snap->shifted_snap = isl_tab_snap(cgbr->shifted);
	else
		snap->shifted_snap = NULL;

	if (cgbr->cone)
		snap->cone_snap = isl_tab_snap(cgbr->cone);
	else
		snap->cone_snap = NULL;

	return snap;
error:
	free(snap);
	return NULL;
}

static void context_gbr_restore(struct isl_context *context, void *save)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	struct isl_gbr_tab_undo *snap = (struct isl_gbr_tab_undo *)save;
	if (!snap)
		goto error;
	if (isl_tab_rollback(cgbr->tab, snap->tab_snap) < 0)
		goto error;

	if (snap->shifted_snap) {
		if (isl_tab_rollback(cgbr->shifted, snap->shifted_snap) < 0)
			goto error;
	} else if (cgbr->shifted) {
		isl_tab_free(cgbr->shifted);
		cgbr->shifted = NULL;
	}

	if (snap->cone_snap) {
		if (isl_tab_rollback(cgbr->cone, snap->cone_snap) < 0)
			goto error;
	} else if (cgbr->cone) {
		isl_tab_free(cgbr->cone);
		cgbr->cone = NULL;
	}

	free(snap);

	return;
error:
	free(snap);
	isl_tab_free(cgbr->tab);
	cgbr->tab = NULL;
}

static void context_gbr_discard(void *save)
{
	struct isl_gbr_tab_undo *snap = (struct isl_gbr_tab_undo *)save;
	free(snap);
}

static int context_gbr_is_ok(struct isl_context *context)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	return !!cgbr->tab;
}

static void context_gbr_invalidate(struct isl_context *context)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	isl_tab_free(cgbr->tab);
	cgbr->tab = NULL;
}

static __isl_null struct isl_context *context_gbr_free(
	struct isl_context *context)
{
	struct isl_context_gbr *cgbr = (struct isl_context_gbr *)context;
	isl_tab_free(cgbr->tab);
	isl_tab_free(cgbr->shifted);
	isl_tab_free(cgbr->cone);
	free(cgbr);

	return NULL;
}

struct isl_context_op isl_context_gbr_op = {
	context_gbr_detect_nonnegative_parameters,
	context_gbr_peek_basic_set,
	context_gbr_peek_tab,
	context_gbr_add_eq,
	context_gbr_add_ineq,
	context_gbr_ineq_sign,
	context_gbr_test_ineq,
	context_gbr_get_div,
	context_gbr_insert_div,
	context_gbr_detect_equalities,
	context_gbr_best_split,
	context_gbr_is_empty,
	context_gbr_is_ok,
	context_gbr_save,
	context_gbr_restore,
	context_gbr_discard,
	context_gbr_invalidate,
	context_gbr_free,
};

static struct isl_context *isl_context_gbr_alloc(__isl_keep isl_basic_set *dom)
{
	struct isl_context_gbr *cgbr;

	if (!dom)
		return NULL;

	cgbr = isl_calloc_type(dom->ctx, struct isl_context_gbr);
	if (!cgbr)
		return NULL;

	cgbr->context.op = &isl_context_gbr_op;

	cgbr->shifted = NULL;
	cgbr->cone = NULL;
	cgbr->tab = isl_tab_from_basic_set(dom, 1);
	cgbr->tab = isl_tab_init_samples(cgbr->tab);
	if (!cgbr->tab)
		goto error;
	check_gbr_integer_feasible(cgbr);

	return &cgbr->context;
error:
	cgbr->context.op->free(&cgbr->context);
	return NULL;
}

/* Allocate a context corresponding to "dom".
 * The representation specific fields are initialized by
 * isl_context_lex_alloc or isl_context_gbr_alloc.
 * The shared "n_unknown" field is initialized to the number
 * of final unknown integer divisions in "dom".
 */
static struct isl_context *isl_context_alloc(__isl_keep isl_basic_set *dom)
{
	struct isl_context *context;
	int first;

	if (!dom)
		return NULL;

	if (dom->ctx->opt->context == ISL_CONTEXT_LEXMIN)
		context = isl_context_lex_alloc(dom);
	else
		context = isl_context_gbr_alloc(dom);

	if (!context)
		return NULL;

	first = isl_basic_set_first_unknown_div(dom);
	if (first < 0)
		return context->op->free(context);
	context->n_unknown = isl_basic_set_dim(dom, isl_dim_div) - first;

	return context;
}

/* Construct an isl_sol_map structure for accumulating the solution.
 * If track_empty is set, then we also keep track of the parts
 * of the context where there is no solution.
 * If max is set, then we are solving a maximization, rather than
 * a minimization problem, which means that the variables in the
 * tableau have value "M - x" rather than "M + x".
 */
static struct isl_sol *sol_map_init(struct isl_basic_map *bmap,
	struct isl_basic_set *dom, int track_empty, int max)
{
	struct isl_sol_map *sol_map = NULL;

	if (!bmap)
		goto error;

	sol_map = isl_calloc_type(bmap->ctx, struct isl_sol_map);
	if (!sol_map)
		goto error;

	sol_map->sol.rational = ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL);
	sol_map->sol.dec_level.callback.run = &sol_dec_level_wrap;
	sol_map->sol.dec_level.sol = &sol_map->sol;
	sol_map->sol.max = max;
	sol_map->sol.n_out = isl_basic_map_dim(bmap, isl_dim_out);
	sol_map->sol.add = &sol_map_add_wrap;
	sol_map->sol.add_empty = track_empty ? &sol_map_add_empty_wrap : NULL;
	sol_map->sol.free = &sol_map_free_wrap;
	sol_map->map = isl_map_alloc_space(isl_basic_map_get_space(bmap), 1,
					    ISL_MAP_DISJOINT);
	if (!sol_map->map)
		goto error;

	sol_map->sol.context = isl_context_alloc(dom);
	if (!sol_map->sol.context)
		goto error;

	if (track_empty) {
		sol_map->empty = isl_set_alloc_space(isl_basic_set_get_space(dom),
							1, ISL_SET_DISJOINT);
		if (!sol_map->empty)
			goto error;
	}

	isl_basic_set_free(dom);
	return &sol_map->sol;
error:
	isl_basic_set_free(dom);
	sol_map_free(sol_map);
	return NULL;
}

/* Check whether all coefficients of (non-parameter) variables
 * are non-positive, meaning that no pivots can be performed on the row.
 */
static int is_critical(struct isl_tab *tab, int row)
{
	int j;
	unsigned off = 2 + tab->M;

	for (j = tab->n_dead; j < tab->n_col; ++j) {
		if (tab->col_var[j] >= 0 &&
		    (tab->col_var[j] < tab->n_param  ||
		    tab->col_var[j] >= tab->n_var - tab->n_div))
			continue;

		if (isl_int_is_pos(tab->mat->row[row][off + j]))
			return 0;
	}

	return 1;
}

/* Check whether the inequality represented by vec is strict over the integers,
 * i.e., there are no integer values satisfying the constraint with
 * equality.  This happens if the gcd of the coefficients is not a divisor
 * of the constant term.  If so, scale the constraint down by the gcd
 * of the coefficients.
 */
static int is_strict(struct isl_vec *vec)
{
	isl_int gcd;
	int strict = 0;

	isl_int_init(gcd);
	isl_seq_gcd(vec->el + 1, vec->size - 1, &gcd);
	if (!isl_int_is_one(gcd)) {
		strict = !isl_int_is_divisible_by(vec->el[0], gcd);
		isl_int_fdiv_q(vec->el[0], vec->el[0], gcd);
		isl_seq_scale_down(vec->el + 1, vec->el + 1, gcd, vec->size-1);
	}
	isl_int_clear(gcd);

	return strict;
}

/* Determine the sign of the given row of the main tableau.
 * The result is one of
 *	isl_tab_row_pos: always non-negative; no pivot needed
 *	isl_tab_row_neg: always non-positive; pivot
 *	isl_tab_row_any: can be both positive and negative; split
 *
 * We first handle some simple cases
 *	- the row sign may be known already
 *	- the row may be obviously non-negative
 *	- the parametric constant may be equal to that of another row
 *	  for which we know the sign.  This sign will be either "pos" or
 *	  "any".  If it had been "neg" then we would have pivoted before.
 *
 * If none of these cases hold, we check the value of the row for each
 * of the currently active samples.  Based on the signs of these values
 * we make an initial determination of the sign of the row.
 *
 *	all zero			->	unk(nown)
 *	all non-negative		->	pos
 *	all non-positive		->	neg
 *	both negative and positive	->	all
 *
 * If we end up with "all", we are done.
 * Otherwise, we perform a check for positive and/or negative
 * values as follows.
 *
 *	samples	       neg	       unk	       pos
 *	<0 ?			    Y        N	    Y        N
 *					    pos    any      pos
 *	>0 ?	     Y      N	 Y     N
 *		    any    neg  any   neg
 *
 * There is no special sign for "zero", because we can usually treat zero
 * as either non-negative or non-positive, whatever works out best.
 * However, if the row is "critical", meaning that pivoting is impossible
 * then we don't want to limp zero with the non-positive case, because
 * then we we would lose the solution for those values of the parameters
 * where the value of the row is zero.  Instead, we treat 0 as non-negative
 * ensuring a split if the row can attain both zero and negative values.
 * The same happens when the original constraint was one that could not
 * be satisfied with equality by any integer values of the parameters.
 * In this case, we normalize the constraint, but then a value of zero
 * for the normalized constraint is actually a positive value for the
 * original constraint, so again we need to treat zero as non-negative.
 * In both these cases, we have the following decision tree instead:
 *
 *	all non-negative		->	pos
 *	all negative			->	neg
 *	both negative and non-negative	->	all
 *
 *	samples	       neg	          	       pos
 *	<0 ?			             	    Y        N
 *					           any      pos
 *	>=0 ?	     Y      N
 *		    any    neg
 */
static enum isl_tab_row_sign row_sign(struct isl_tab *tab,
	struct isl_sol *sol, int row)
{
	struct isl_vec *ineq = NULL;
	enum isl_tab_row_sign res = isl_tab_row_unknown;
	int critical;
	int strict;
	int row2;

	if (tab->row_sign[row] != isl_tab_row_unknown)
		return tab->row_sign[row];
	if (is_obviously_nonneg(tab, row))
		return isl_tab_row_pos;
	for (row2 = tab->n_redundant; row2 < tab->n_row; ++row2) {
		if (tab->row_sign[row2] == isl_tab_row_unknown)
			continue;
		if (identical_parameter_line(tab, row, row2))
			return tab->row_sign[row2];
	}

	critical = is_critical(tab, row);

	ineq = get_row_parameter_ineq(tab, row);
	if (!ineq)
		goto error;

	strict = is_strict(ineq);

	res = sol->context->op->ineq_sign(sol->context, ineq->el,
					  critical || strict);

	if (res == isl_tab_row_unknown || res == isl_tab_row_pos) {
		/* test for negative values */
		int feasible;
		isl_seq_neg(ineq->el, ineq->el, ineq->size);
		isl_int_sub_ui(ineq->el[0], ineq->el[0], 1);

		feasible = sol->context->op->test_ineq(sol->context, ineq->el);
		if (feasible < 0)
			goto error;
		if (!feasible)
			res = isl_tab_row_pos;
		else
			res = (res == isl_tab_row_unknown) ? isl_tab_row_neg
							   : isl_tab_row_any;
		if (res == isl_tab_row_neg) {
			isl_seq_neg(ineq->el, ineq->el, ineq->size);
			isl_int_sub_ui(ineq->el[0], ineq->el[0], 1);
		}
	}

	if (res == isl_tab_row_neg) {
		/* test for positive values */
		int feasible;
		if (!critical && !strict)
			isl_int_sub_ui(ineq->el[0], ineq->el[0], 1);

		feasible = sol->context->op->test_ineq(sol->context, ineq->el);
		if (feasible < 0)
			goto error;
		if (feasible)
			res = isl_tab_row_any;
	}

	isl_vec_free(ineq);
	return res;
error:
	isl_vec_free(ineq);
	return isl_tab_row_unknown;
}

static void find_solutions(struct isl_sol *sol, struct isl_tab *tab);

/* Find solutions for values of the parameters that satisfy the given
 * inequality.
 *
 * We currently take a snapshot of the context tableau that is reset
 * when we return from this function, while we make a copy of the main
 * tableau, leaving the original main tableau untouched.
 * These are fairly arbitrary choices.  Making a copy also of the context
 * tableau would obviate the need to undo any changes made to it later,
 * while taking a snapshot of the main tableau could reduce memory usage.
 * If we were to switch to taking a snapshot of the main tableau,
 * we would have to keep in mind that we need to save the row signs
 * and that we need to do this before saving the current basis
 * such that the basis has been restore before we restore the row signs.
 */
static void find_in_pos(struct isl_sol *sol, struct isl_tab *tab, isl_int *ineq)
{
	void *saved;

	if (!sol->context)
		goto error;
	saved = sol->context->op->save(sol->context);

	tab = isl_tab_dup(tab);
	if (!tab)
		goto error;

	sol->context->op->add_ineq(sol->context, ineq, 0, 1);

	find_solutions(sol, tab);

	if (!sol->error)
		sol->context->op->restore(sol->context, saved);
	else
		sol->context->op->discard(saved);
	return;
error:
	sol->error = 1;
}

/* Record the absence of solutions for those values of the parameters
 * that do not satisfy the given inequality with equality.
 */
static void no_sol_in_strict(struct isl_sol *sol,
	struct isl_tab *tab, struct isl_vec *ineq)
{
	int empty;
	void *saved;

	if (!sol->context || sol->error)
		goto error;
	saved = sol->context->op->save(sol->context);

	isl_int_sub_ui(ineq->el[0], ineq->el[0], 1);

	sol->context->op->add_ineq(sol->context, ineq->el, 1, 0);
	if (!sol->context)
		goto error;

	empty = tab->empty;
	tab->empty = 1;
	sol_add(sol, tab);
	tab->empty = empty;

	isl_int_add_ui(ineq->el[0], ineq->el[0], 1);

	sol->context->op->restore(sol->context, saved);
	return;
error:
	sol->error = 1;
}

/* Reset all row variables that are marked to have a sign that may
 * be both positive and negative to have an unknown sign.
 */
static void reset_any_to_unknown(struct isl_tab *tab)
{
	int row;

	for (row = tab->n_redundant; row < tab->n_row; ++row) {
		if (!isl_tab_var_from_row(tab, row)->is_nonneg)
			continue;
		if (tab->row_sign[row] == isl_tab_row_any)
			tab->row_sign[row] = isl_tab_row_unknown;
	}
}

/* Compute the lexicographic minimum of the set represented by the main
 * tableau "tab" within the context "sol->context_tab".
 * On entry the sample value of the main tableau is lexicographically
 * less than or equal to this lexicographic minimum.
 * Pivots are performed until a feasible point is found, which is then
 * necessarily equal to the minimum, or until the tableau is found to
 * be infeasible.  Some pivots may need to be performed for only some
 * feasible values of the context tableau.  If so, the context tableau
 * is split into a part where the pivot is needed and a part where it is not.
 *
 * Whenever we enter the main loop, the main tableau is such that no
 * "obvious" pivots need to be performed on it, where "obvious" means
 * that the given row can be seen to be negative without looking at
 * the context tableau.  In particular, for non-parametric problems,
 * no pivots need to be performed on the main tableau.
 * The caller of find_solutions is responsible for making this property
 * hold prior to the first iteration of the loop, while restore_lexmin
 * is called before every other iteration.
 *
 * Inside the main loop, we first examine the signs of the rows of
 * the main tableau within the context of the context tableau.
 * If we find a row that is always non-positive for all values of
 * the parameters satisfying the context tableau and negative for at
 * least one value of the parameters, we perform the appropriate pivot
 * and start over.  An exception is the case where no pivot can be
 * performed on the row.  In this case, we require that the sign of
 * the row is negative for all values of the parameters (rather than just
 * non-positive).  This special case is handled inside row_sign, which
 * will say that the row can have any sign if it determines that it can
 * attain both negative and zero values.
 *
 * If we can't find a row that always requires a pivot, but we can find
 * one or more rows that require a pivot for some values of the parameters
 * (i.e., the row can attain both positive and negative signs), then we split
 * the context tableau into two parts, one where we force the sign to be
 * non-negative and one where we force is to be negative.
 * The non-negative part is handled by a recursive call (through find_in_pos).
 * Upon returning from this call, we continue with the negative part and
 * perform the required pivot.
 *
 * If no such rows can be found, all rows are non-negative and we have
 * found a (rational) feasible point.  If we only wanted a rational point
 * then we are done.
 * Otherwise, we check if all values of the sample point of the tableau
 * are integral for the variables.  If so, we have found the minimal
 * integral point and we are done.
 * If the sample point is not integral, then we need to make a distinction
 * based on whether the constant term is non-integral or the coefficients
 * of the parameters.  Furthermore, in order to decide how to handle
 * the non-integrality, we also need to know whether the coefficients
 * of the other columns in the tableau are integral.  This leads
 * to the following table.  The first two rows do not correspond
 * to a non-integral sample point and are only mentioned for completeness.
 *
 *	constant	parameters	other
 *
 *	int		int		int	|
 *	int		int		rat	| -> no problem
 *
 *	rat		int		int	  -> fail
 *
 *	rat		int		rat	  -> cut
 *
 *	int		rat		rat	|
 *	rat		rat		rat	| -> parametric cut
 *
 *	int		rat		int	|
 *	rat		rat		int	| -> split context
 *
 * If the parametric constant is completely integral, then there is nothing
 * to be done.  If the constant term is non-integral, but all the other
 * coefficient are integral, then there is nothing that can be done
 * and the tableau has no integral solution.
 * If, on the other hand, one or more of the other columns have rational
 * coefficients, but the parameter coefficients are all integral, then
 * we can perform a regular (non-parametric) cut.
 * Finally, if there is any parameter coefficient that is non-integral,
 * then we need to involve the context tableau.  There are two cases here.
 * If at least one other column has a rational coefficient, then we
 * can perform a parametric cut in the main tableau by adding a new
 * integer division in the context tableau.
 * If all other columns have integral coefficients, then we need to
 * enforce that the rational combination of parameters (c + \sum a_i y_i)/m
 * is always integral.  We do this by introducing an integer division
 * q = floor((c + \sum a_i y_i)/m) and stipulating that its argument should
 * always be integral in the context tableau, i.e., m q = c + \sum a_i y_i.
 * Since q is expressed in the tableau as
 *	c + \sum a_i y_i - m q >= 0
 *	-c - \sum a_i y_i + m q + m - 1 >= 0
 * it is sufficient to add the inequality
 *	-c - \sum a_i y_i + m q >= 0
 * In the part of the context where this inequality does not hold, the
 * main tableau is marked as being empty.
 */
static void find_solutions(struct isl_sol *sol, struct isl_tab *tab)
{
	struct isl_context *context;
	int r;

	if (!tab || sol->error)
		goto error;

	context = sol->context;

	if (tab->empty)
		goto done;
	if (context->op->is_empty(context))
		goto done;

	for (r = 0; r >= 0 && tab && !tab->empty; r = restore_lexmin(tab)) {
		int flags;
		int row;
		enum isl_tab_row_sign sgn;
		int split = -1;
		int n_split = 0;

		for (row = tab->n_redundant; row < tab->n_row; ++row) {
			if (!isl_tab_var_from_row(tab, row)->is_nonneg)
				continue;
			sgn = row_sign(tab, sol, row);
			if (!sgn)
				goto error;
			tab->row_sign[row] = sgn;
			if (sgn == isl_tab_row_any)
				n_split++;
			if (sgn == isl_tab_row_any && split == -1)
				split = row;
			if (sgn == isl_tab_row_neg)
				break;
		}
		if (row < tab->n_row)
			continue;
		if (split != -1) {
			struct isl_vec *ineq;
			if (n_split != 1)
				split = context->op->best_split(context, tab);
			if (split < 0)
				goto error;
			ineq = get_row_parameter_ineq(tab, split);
			if (!ineq)
				goto error;
			is_strict(ineq);
			reset_any_to_unknown(tab);
			tab->row_sign[split] = isl_tab_row_pos;
			sol_inc_level(sol);
			find_in_pos(sol, tab, ineq->el);
			tab->row_sign[split] = isl_tab_row_neg;
			isl_seq_neg(ineq->el, ineq->el, ineq->size);
			isl_int_sub_ui(ineq->el[0], ineq->el[0], 1);
			if (!sol->error)
				context->op->add_ineq(context, ineq->el, 0, 1);
			isl_vec_free(ineq);
			if (sol->error)
				goto error;
			continue;
		}
		if (tab->rational)
			break;
		row = first_non_integer_row(tab, &flags);
		if (row < 0)
			break;
		if (ISL_FL_ISSET(flags, I_PAR)) {
			if (ISL_FL_ISSET(flags, I_VAR)) {
				if (isl_tab_mark_empty(tab) < 0)
					goto error;
				break;
			}
			row = add_cut(tab, row);
		} else if (ISL_FL_ISSET(flags, I_VAR)) {
			struct isl_vec *div;
			struct isl_vec *ineq;
			int d;
			div = get_row_split_div(tab, row);
			if (!div)
				goto error;
			d = context->op->get_div(context, tab, div);
			isl_vec_free(div);
			if (d < 0)
				goto error;
			ineq = ineq_for_div(context->op->peek_basic_set(context), d);
			if (!ineq)
				goto error;
			sol_inc_level(sol);
			no_sol_in_strict(sol, tab, ineq);
			isl_seq_neg(ineq->el, ineq->el, ineq->size);
			context->op->add_ineq(context, ineq->el, 1, 1);
			isl_vec_free(ineq);
			if (sol->error || !context->op->is_ok(context))
				goto error;
			tab = set_row_cst_to_div(tab, row, d);
			if (context->op->is_empty(context))
				break;
		} else
			row = add_parametric_cut(tab, row, context);
		if (row < 0)
			goto error;
	}
	if (r < 0)
		goto error;
done:
	sol_add(sol, tab);
	isl_tab_free(tab);
	return;
error:
	isl_tab_free(tab);
	sol->error = 1;
}

/* Does "sol" contain a pair of partial solutions that could potentially
 * be merged?
 *
 * We currently only check that "sol" is not in an error state
 * and that there are at least two partial solutions of which the final two
 * are defined at the same level.
 */
static int sol_has_mergeable_solutions(struct isl_sol *sol)
{
	if (sol->error)
		return 0;
	if (!sol->partial)
		return 0;
	if (!sol->partial->next)
		return 0;
	return sol->partial->level == sol->partial->next->level;
}

/* Compute the lexicographic minimum of the set represented by the main
 * tableau "tab" within the context "sol->context_tab".
 *
 * As a preprocessing step, we first transfer all the purely parametric
 * equalities from the main tableau to the context tableau, i.e.,
 * parameters that have been pivoted to a row.
 * These equalities are ignored by the main algorithm, because the
 * corresponding rows may not be marked as being non-negative.
 * In parts of the context where the added equality does not hold,
 * the main tableau is marked as being empty.
 *
 * Before we embark on the actual computation, we save a copy
 * of the context.  When we return, we check if there are any
 * partial solutions that can potentially be merged.  If so,
 * we perform a rollback to the initial state of the context.
 * The merging of partial solutions happens inside calls to
 * sol_dec_level that are pushed onto the undo stack of the context.
 * If there are no partial solutions that can potentially be merged
 * then the rollback is skipped as it would just be wasted effort.
 */
static void find_solutions_main(struct isl_sol *sol, struct isl_tab *tab)
{
	int row;
	void *saved;

	if (!tab)
		goto error;

	sol->level = 0;

	for (row = tab->n_redundant; row < tab->n_row; ++row) {
		int p;
		struct isl_vec *eq;

		if (tab->row_var[row] < 0)
			continue;
		if (tab->row_var[row] >= tab->n_param &&
		    tab->row_var[row] < tab->n_var - tab->n_div)
			continue;
		if (tab->row_var[row] < tab->n_param)
			p = tab->row_var[row];
		else
			p = tab->row_var[row]
				+ tab->n_param - (tab->n_var - tab->n_div);

		eq = isl_vec_alloc(tab->mat->ctx, 1+tab->n_param+tab->n_div);
		if (!eq)
			goto error;
		get_row_parameter_line(tab, row, eq->el);
		isl_int_neg(eq->el[1 + p], tab->mat->row[row][0]);
		eq = isl_vec_normalize(eq);

		sol_inc_level(sol);
		no_sol_in_strict(sol, tab, eq);

		isl_seq_neg(eq->el, eq->el, eq->size);
		sol_inc_level(sol);
		no_sol_in_strict(sol, tab, eq);
		isl_seq_neg(eq->el, eq->el, eq->size);

		sol->context->op->add_eq(sol->context, eq->el, 1, 1);

		isl_vec_free(eq);

		if (isl_tab_mark_redundant(tab, row) < 0)
			goto error;

		if (sol->context->op->is_empty(sol->context))
			break;

		row = tab->n_redundant - 1;
	}

	saved = sol->context->op->save(sol->context);

	find_solutions(sol, tab);

	if (sol_has_mergeable_solutions(sol))
		sol->context->op->restore(sol->context, saved);
	else
		sol->context->op->discard(saved);

	sol->level = 0;
	sol_pop(sol);

	return;
error:
	isl_tab_free(tab);
	sol->error = 1;
}

/* Check if integer division "div" of "dom" also occurs in "bmap".
 * If so, return its position within the divs.
 * If not, return -1.
 */
static int find_context_div(struct isl_basic_map *bmap,
	struct isl_basic_set *dom, unsigned div)
{
	int i;
	unsigned b_dim = isl_space_dim(bmap->dim, isl_dim_all);
	unsigned d_dim = isl_space_dim(dom->dim, isl_dim_all);

	if (isl_int_is_zero(dom->div[div][0]))
		return -1;
	if (isl_seq_first_non_zero(dom->div[div] + 2 + d_dim, dom->n_div) != -1)
		return -1;

	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		if (isl_seq_first_non_zero(bmap->div[i] + 2 + d_dim,
					   (b_dim - d_dim) + bmap->n_div) != -1)
			continue;
		if (isl_seq_eq(bmap->div[i], dom->div[div], 2 + d_dim))
			return i;
	}
	return -1;
}

/* The correspondence between the variables in the main tableau,
 * the context tableau, and the input map and domain is as follows.
 * The first n_param and the last n_div variables of the main tableau
 * form the variables of the context tableau.
 * In the basic map, these n_param variables correspond to the
 * parameters and the input dimensions.  In the domain, they correspond
 * to the parameters and the set dimensions.
 * The n_div variables correspond to the integer divisions in the domain.
 * To ensure that everything lines up, we may need to copy some of the
 * integer divisions of the domain to the map.  These have to be placed
 * in the same order as those in the context and they have to be placed
 * after any other integer divisions that the map may have.
 * This function performs the required reordering.
 */
static struct isl_basic_map *align_context_divs(struct isl_basic_map *bmap,
	struct isl_basic_set *dom)
{
	int i;
	int common = 0;
	int other;

	for (i = 0; i < dom->n_div; ++i)
		if (find_context_div(bmap, dom, i) != -1)
			common++;
	other = bmap->n_div - common;
	if (dom->n_div - common > 0) {
		bmap = isl_basic_map_extend_space(bmap, isl_space_copy(bmap->dim),
				dom->n_div - common, 0, 0);
		if (!bmap)
			return NULL;
	}
	for (i = 0; i < dom->n_div; ++i) {
		int pos = find_context_div(bmap, dom, i);
		if (pos < 0) {
			pos = isl_basic_map_alloc_div(bmap);
			if (pos < 0)
				goto error;
			isl_int_set_si(bmap->div[pos][0], 0);
		}
		if (pos != other + i)
			isl_basic_map_swap_div(bmap, pos, other + i);
	}
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Base case of isl_tab_basic_map_partial_lexopt, after removing
 * some obvious symmetries.
 *
 * We make sure the divs in the domain are properly ordered,
 * because they will be added one by one in the given order
 * during the construction of the solution map.
 * Furthermore, make sure that the known integer divisions
 * appear before any unknown integer division because the solution
 * may depend on the known integer divisions, while anything that
 * depends on any variable starting from the first unknown integer
 * division is ignored in sol_pma_add.
 */
static struct isl_sol *basic_map_partial_lexopt_base_sol(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, int max,
	struct isl_sol *(*init)(__isl_keep isl_basic_map *bmap,
		    __isl_take isl_basic_set *dom, int track_empty, int max))
{
	struct isl_tab *tab;
	struct isl_sol *sol = NULL;
	struct isl_context *context;

	if (dom->n_div) {
		dom = isl_basic_set_sort_divs(dom);
		bmap = align_context_divs(bmap, dom);
	}
	sol = init(bmap, dom, !!empty, max);
	if (!sol)
		goto error;

	context = sol->context;
	if (isl_basic_set_plain_is_empty(context->op->peek_basic_set(context)))
		/* nothing */;
	else if (isl_basic_map_plain_is_empty(bmap)) {
		if (sol->add_empty)
			sol->add_empty(sol,
		    isl_basic_set_copy(context->op->peek_basic_set(context)));
	} else {
		tab = tab_for_lexmin(bmap,
				    context->op->peek_basic_set(context), 1, max);
		tab = context->op->detect_nonnegative_parameters(context, tab);
		find_solutions_main(sol, tab);
	}
	if (sol->error)
		goto error;

	isl_basic_map_free(bmap);
	return sol;
error:
	sol_free(sol);
	isl_basic_map_free(bmap);
	return NULL;
}

/* Base case of isl_tab_basic_map_partial_lexopt, after removing
 * some obvious symmetries.
 *
 * We call basic_map_partial_lexopt_base_sol and extract the results.
 */
static __isl_give isl_map *basic_map_partial_lexopt_base(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, int max)
{
	isl_map *result = NULL;
	struct isl_sol *sol;
	struct isl_sol_map *sol_map;

	sol = basic_map_partial_lexopt_base_sol(bmap, dom, empty, max,
						&sol_map_init);
	if (!sol)
		return NULL;
	sol_map = (struct isl_sol_map *) sol;

	result = isl_map_copy(sol_map->map);
	if (empty)
		*empty = isl_set_copy(sol_map->empty);
	sol_free(&sol_map->sol);
	return result;
}

/* Return a count of the number of occurrences of the "n" first
 * variables in the inequality constraints of "bmap".
 */
static __isl_give int *count_occurrences(__isl_keep isl_basic_map *bmap,
	int n)
{
	int i, j;
	isl_ctx *ctx;
	int *occurrences;

	if (!bmap)
		return NULL;
	ctx = isl_basic_map_get_ctx(bmap);
	occurrences = isl_calloc_array(ctx, int, n);
	if (!occurrences)
		return NULL;

	for (i = 0; i < bmap->n_ineq; ++i) {
		for (j = 0; j < n; ++j) {
			if (!isl_int_is_zero(bmap->ineq[i][1 + j]))
				occurrences[j]++;
		}
	}

	return occurrences;
}

/* Do all of the "n" variables with non-zero coefficients in "c"
 * occur in exactly a single constraint.
 * "occurrences" is an array of length "n" containing the number
 * of occurrences of each of the variables in the inequality constraints.
 */
static int single_occurrence(int n, isl_int *c, int *occurrences)
{
	int i;

	for (i = 0; i < n; ++i) {
		if (isl_int_is_zero(c[i]))
			continue;
		if (occurrences[i] != 1)
			return 0;
	}

	return 1;
}

/* Do all of the "n" initial variables that occur in inequality constraint
 * "ineq" of "bmap" only occur in that constraint?
 */
static int all_single_occurrence(__isl_keep isl_basic_map *bmap, int ineq,
	int n)
{
	int i, j;

	for (i = 0; i < n; ++i) {
		if (isl_int_is_zero(bmap->ineq[ineq][1 + i]))
			continue;
		for (j = 0; j < bmap->n_ineq; ++j) {
			if (j == ineq)
				continue;
			if (!isl_int_is_zero(bmap->ineq[j][1 + i]))
				return 0;
		}
	}

	return 1;
}

/* Structure used during detection of parallel constraints.
 * n_in: number of "input" variables: isl_dim_param + isl_dim_in
 * n_out: number of "output" variables: isl_dim_out + isl_dim_div
 * val: the coefficients of the output variables
 */
struct isl_constraint_equal_info {
	isl_basic_map *bmap;
	unsigned n_in;
	unsigned n_out;
	isl_int *val;
};

/* Check whether the coefficients of the output variables
 * of the constraint in "entry" are equal to info->val.
 */
static int constraint_equal(const void *entry, const void *val)
{
	isl_int **row = (isl_int **)entry;
	const struct isl_constraint_equal_info *info = val;

	return isl_seq_eq((*row) + 1 + info->n_in, info->val, info->n_out);
}

/* Check whether "bmap" has a pair of constraints that have
 * the same coefficients for the output variables.
 * Note that the coefficients of the existentially quantified
 * variables need to be zero since the existentially quantified
 * of the result are usually not the same as those of the input.
 * Furthermore, check that each of the input variables that occur
 * in those constraints does not occur in any other constraint.
 * If so, return 1 and return the row indices of the two constraints
 * in *first and *second.
 */
static int parallel_constraints(__isl_keep isl_basic_map *bmap,
	int *first, int *second)
{
	int i;
	isl_ctx *ctx;
	int *occurrences = NULL;
	struct isl_hash_table *table = NULL;
	struct isl_hash_table_entry *entry;
	struct isl_constraint_equal_info info;
	unsigned n_out;
	unsigned n_div;

	ctx = isl_basic_map_get_ctx(bmap);
	table = isl_hash_table_alloc(ctx, bmap->n_ineq);
	if (!table)
		goto error;

	info.n_in = isl_basic_map_dim(bmap, isl_dim_param) +
		    isl_basic_map_dim(bmap, isl_dim_in);
	occurrences = count_occurrences(bmap, info.n_in);
	if (info.n_in && !occurrences)
		goto error;
	info.bmap = bmap;
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	info.n_out = n_out + n_div;
	for (i = 0; i < bmap->n_ineq; ++i) {
		uint32_t hash;

		info.val = bmap->ineq[i] + 1 + info.n_in;
		if (isl_seq_first_non_zero(info.val, n_out) < 0)
			continue;
		if (isl_seq_first_non_zero(info.val + n_out, n_div) >= 0)
			continue;
		if (!single_occurrence(info.n_in, bmap->ineq[i] + 1,
					occurrences))
			continue;
		hash = isl_seq_get_hash(info.val, info.n_out);
		entry = isl_hash_table_find(ctx, table, hash,
					    constraint_equal, &info, 1);
		if (!entry)
			goto error;
		if (entry->data)
			break;
		entry->data = &bmap->ineq[i];
	}

	if (i < bmap->n_ineq) {
		*first = ((isl_int **)entry->data) - bmap->ineq; 
		*second = i;
	}

	isl_hash_table_free(ctx, table);
	free(occurrences);

	return i < bmap->n_ineq;
error:
	isl_hash_table_free(ctx, table);
	free(occurrences);
	return -1;
}

/* Given a set of upper bounds in "var", add constraints to "bset"
 * that make the i-th bound smallest.
 *
 * In particular, if there are n bounds b_i, then add the constraints
 *
 *	b_i <= b_j	for j > i
 *	b_i <  b_j	for j < i
 */
static __isl_give isl_basic_set *select_minimum(__isl_take isl_basic_set *bset,
	__isl_keep isl_mat *var, int i)
{
	isl_ctx *ctx;
	int j, k;

	ctx = isl_mat_get_ctx(var);

	for (j = 0; j < var->n_row; ++j) {
		if (j == i)
			continue;
		k = isl_basic_set_alloc_inequality(bset);
		if (k < 0)
			goto error;
		isl_seq_combine(bset->ineq[k], ctx->one, var->row[j],
				ctx->negone, var->row[i], var->n_col);
		isl_int_set_si(bset->ineq[k][var->n_col], 0);
		if (j < i)
			isl_int_sub_ui(bset->ineq[k][0], bset->ineq[k][0], 1);
	}

	bset = isl_basic_set_finalize(bset);

	return bset;
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Given a set of upper bounds on the last "input" variable m,
 * construct a set that assigns the minimal upper bound to m, i.e.,
 * construct a set that divides the space into cells where one
 * of the upper bounds is smaller than all the others and assign
 * this upper bound to m.
 *
 * In particular, if there are n bounds b_i, then the result
 * consists of n basic sets, each one of the form
 *
 *	m = b_i
 *	b_i <= b_j	for j > i
 *	b_i <  b_j	for j < i
 */
static __isl_give isl_set *set_minimum(__isl_take isl_space *dim,
	__isl_take isl_mat *var)
{
	int i, k;
	isl_basic_set *bset = NULL;
	isl_set *set = NULL;

	if (!dim || !var)
		goto error;

	set = isl_set_alloc_space(isl_space_copy(dim),
				var->n_row, ISL_SET_DISJOINT);

	for (i = 0; i < var->n_row; ++i) {
		bset = isl_basic_set_alloc_space(isl_space_copy(dim), 0,
					       1, var->n_row - 1);
		k = isl_basic_set_alloc_equality(bset);
		if (k < 0)
			goto error;
		isl_seq_cpy(bset->eq[k], var->row[i], var->n_col);
		isl_int_set_si(bset->eq[k][var->n_col], -1);
		bset = select_minimum(bset, var, i);
		set = isl_set_add_basic_set(set, bset);
	}

	isl_space_free(dim);
	isl_mat_free(var);
	return set;
error:
	isl_basic_set_free(bset);
	isl_set_free(set);
	isl_space_free(dim);
	isl_mat_free(var);
	return NULL;
}

/* Given that the last input variable of "bmap" represents the minimum
 * of the bounds in "cst", check whether we need to split the domain
 * based on which bound attains the minimum.
 *
 * A split is needed when the minimum appears in an integer division
 * or in an equality.  Otherwise, it is only needed if it appears in
 * an upper bound that is different from the upper bounds on which it
 * is defined.
 */
static int need_split_basic_map(__isl_keep isl_basic_map *bmap,
	__isl_keep isl_mat *cst)
{
	int i, j;
	unsigned total;
	unsigned pos;

	pos = cst->n_col - 1;
	total = isl_basic_map_dim(bmap, isl_dim_all);

	for (i = 0; i < bmap->n_div; ++i)
		if (!isl_int_is_zero(bmap->div[i][2 + pos]))
			return 1;

	for (i = 0; i < bmap->n_eq; ++i)
		if (!isl_int_is_zero(bmap->eq[i][1 + pos]))
			return 1;

	for (i = 0; i < bmap->n_ineq; ++i) {
		if (isl_int_is_nonneg(bmap->ineq[i][1 + pos]))
			continue;
		if (!isl_int_is_negone(bmap->ineq[i][1 + pos]))
			return 1;
		if (isl_seq_first_non_zero(bmap->ineq[i] + 1 + pos + 1,
					   total - pos - 1) >= 0)
			return 1;

		for (j = 0; j < cst->n_row; ++j)
			if (isl_seq_eq(bmap->ineq[i], cst->row[j], cst->n_col))
				break;
		if (j >= cst->n_row)
			return 1;
	}

	return 0;
}

/* Given that the last set variable of "bset" represents the minimum
 * of the bounds in "cst", check whether we need to split the domain
 * based on which bound attains the minimum.
 *
 * We simply call need_split_basic_map here.  This is safe because
 * the position of the minimum is computed from "cst" and not
 * from "bmap".
 */
static int need_split_basic_set(__isl_keep isl_basic_set *bset,
	__isl_keep isl_mat *cst)
{
	return need_split_basic_map(bset_to_bmap(bset), cst);
}

/* Given that the last set variable of "set" represents the minimum
 * of the bounds in "cst", check whether we need to split the domain
 * based on which bound attains the minimum.
 */
static int need_split_set(__isl_keep isl_set *set, __isl_keep isl_mat *cst)
{
	int i;

	for (i = 0; i < set->n; ++i)
		if (need_split_basic_set(set->p[i], cst))
			return 1;

	return 0;
}

/* Given a set of which the last set variable is the minimum
 * of the bounds in "cst", split each basic set in the set
 * in pieces where one of the bounds is (strictly) smaller than the others.
 * This subdivision is given in "min_expr".
 * The variable is subsequently projected out.
 *
 * We only do the split when it is needed.
 * For example if the last input variable m = min(a,b) and the only
 * constraints in the given basic set are lower bounds on m,
 * i.e., l <= m = min(a,b), then we can simply project out m
 * to obtain l <= a and l <= b, without having to split on whether
 * m is equal to a or b.
 */
static __isl_give isl_set *split(__isl_take isl_set *empty,
	__isl_take isl_set *min_expr, __isl_take isl_mat *cst)
{
	int n_in;
	int i;
	isl_space *dim;
	isl_set *res;

	if (!empty || !min_expr || !cst)
		goto error;

	n_in = isl_set_dim(empty, isl_dim_set);
	dim = isl_set_get_space(empty);
	dim = isl_space_drop_dims(dim, isl_dim_set, n_in - 1, 1);
	res = isl_set_empty(dim);

	for (i = 0; i < empty->n; ++i) {
		isl_set *set;

		set = isl_set_from_basic_set(isl_basic_set_copy(empty->p[i]));
		if (need_split_basic_set(empty->p[i], cst))
			set = isl_set_intersect(set, isl_set_copy(min_expr));
		set = isl_set_remove_dims(set, isl_dim_set, n_in - 1, 1);

		res = isl_set_union_disjoint(res, set);
	}

	isl_set_free(empty);
	isl_set_free(min_expr);
	isl_mat_free(cst);
	return res;
error:
	isl_set_free(empty);
	isl_set_free(min_expr);
	isl_mat_free(cst);
	return NULL;
}

/* Given a map of which the last input variable is the minimum
 * of the bounds in "cst", split each basic set in the set
 * in pieces where one of the bounds is (strictly) smaller than the others.
 * This subdivision is given in "min_expr".
 * The variable is subsequently projected out.
 *
 * The implementation is essentially the same as that of "split".
 */
static __isl_give isl_map *split_domain(__isl_take isl_map *opt,
	__isl_take isl_set *min_expr, __isl_take isl_mat *cst)
{
	int n_in;
	int i;
	isl_space *dim;
	isl_map *res;

	if (!opt || !min_expr || !cst)
		goto error;

	n_in = isl_map_dim(opt, isl_dim_in);
	dim = isl_map_get_space(opt);
	dim = isl_space_drop_dims(dim, isl_dim_in, n_in - 1, 1);
	res = isl_map_empty(dim);

	for (i = 0; i < opt->n; ++i) {
		isl_map *map;

		map = isl_map_from_basic_map(isl_basic_map_copy(opt->p[i]));
		if (need_split_basic_map(opt->p[i], cst))
			map = isl_map_intersect_domain(map,
						       isl_set_copy(min_expr));
		map = isl_map_remove_dims(map, isl_dim_in, n_in - 1, 1);

		res = isl_map_union_disjoint(res, map);
	}

	isl_map_free(opt);
	isl_set_free(min_expr);
	isl_mat_free(cst);
	return res;
error:
	isl_map_free(opt);
	isl_set_free(min_expr);
	isl_mat_free(cst);
	return NULL;
}

static __isl_give isl_map *basic_map_partial_lexopt(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, int max);

/* This function is called from basic_map_partial_lexopt_symm.
 * The last variable of "bmap" and "dom" corresponds to the minimum
 * of the bounds in "cst".  "map_space" is the space of the original
 * input relation (of basic_map_partial_lexopt_symm) and "set_space"
 * is the space of the original domain.
 *
 * We recursively call basic_map_partial_lexopt and then plug in
 * the definition of the minimum in the result.
 */
static __isl_give isl_map *basic_map_partial_lexopt_symm_core(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, int max, __isl_take isl_mat *cst,
	__isl_take isl_space *map_space, __isl_take isl_space *set_space)
{
	isl_map *opt;
	isl_set *min_expr;

	min_expr = set_minimum(isl_basic_set_get_space(dom), isl_mat_copy(cst));

	opt = basic_map_partial_lexopt(bmap, dom, empty, max);

	if (empty) {
		*empty = split(*empty,
			       isl_set_copy(min_expr), isl_mat_copy(cst));
		*empty = isl_set_reset_space(*empty, set_space);
	}

	opt = split_domain(opt, min_expr, cst);
	opt = isl_map_reset_space(opt, map_space);

	return opt;
}

/* Extract a domain from "bmap" for the purpose of computing
 * a lexicographic optimum.
 *
 * This function is only called when the caller wants to compute a full
 * lexicographic optimum, i.e., without specifying a domain.  In this case,
 * the caller is not interested in the part of the domain space where
 * there is no solution and the domain can be initialized to those constraints
 * of "bmap" that only involve the parameters and the input dimensions.
 * This relieves the parametric programming engine from detecting those
 * inequalities and transferring them to the context.  More importantly,
 * it ensures that those inequalities are transferred first and not
 * intermixed with inequalities that actually split the domain.
 *
 * If the caller does not require the absence of existentially quantified
 * variables in the result (i.e., if ISL_OPT_QE is not set in "flags"),
 * then the actual domain of "bmap" can be used.  This ensures that
 * the domain does not need to be split at all just to separate out
 * pieces of the domain that do not have a solution from piece that do.
 * This domain cannot be used in general because it may involve
 * (unknown) existentially quantified variables which will then also
 * appear in the solution.
 */
static __isl_give isl_basic_set *extract_domain(__isl_keep isl_basic_map *bmap,
	unsigned flags)
{
	int n_div;
	int n_out;

	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	bmap = isl_basic_map_copy(bmap);
	if (ISL_FL_ISSET(flags, ISL_OPT_QE)) {
		bmap = isl_basic_map_drop_constraints_involving_dims(bmap,
							isl_dim_div, 0, n_div);
		bmap = isl_basic_map_drop_constraints_involving_dims(bmap,
							isl_dim_out, 0, n_out);
	}
	return isl_basic_map_domain(bmap);
}

#undef TYPE
#define TYPE	isl_map
#undef SUFFIX
#define SUFFIX
#include "isl_tab_lexopt_templ.c"

struct isl_sol_for {
	struct isl_sol	sol;
	int		(*fn)(__isl_take isl_basic_set *dom,
				__isl_take isl_aff_list *list, void *user);
	void		*user;
};

static void sol_for_free(struct isl_sol_for *sol_for)
{
	if (!sol_for)
		return;
	if (sol_for->sol.context)
		sol_for->sol.context->op->free(sol_for->sol.context);
	free(sol_for);
}

static void sol_for_free_wrap(struct isl_sol *sol)
{
	sol_for_free((struct isl_sol_for *)sol);
}

/* Add the solution identified by the tableau and the context tableau.
 *
 * See documentation of sol_add for more details.
 *
 * Instead of constructing a basic map, this function calls a user
 * defined function with the current context as a basic set and
 * a list of affine expressions representing the relation between
 * the input and output.  The space over which the affine expressions
 * are defined is the same as that of the domain.  The number of
 * affine expressions in the list is equal to the number of output variables.
 */
static void sol_for_add(struct isl_sol_for *sol,
	struct isl_basic_set *dom, struct isl_mat *M)
{
	int i;
	isl_ctx *ctx;
	isl_local_space *ls;
	isl_aff *aff;
	isl_aff_list *list;

	if (sol->sol.error || !dom || !M)
		goto error;

	ctx = isl_basic_set_get_ctx(dom);
	ls = isl_basic_set_get_local_space(dom);
	list = isl_aff_list_alloc(ctx, M->n_row - 1);
	for (i = 1; i < M->n_row; ++i) {
		aff = isl_aff_alloc(isl_local_space_copy(ls));
		if (aff) {
			isl_int_set(aff->v->el[0], M->row[0][0]);
			isl_seq_cpy(aff->v->el + 1, M->row[i], M->n_col);
		}
		aff = isl_aff_normalize(aff);
		list = isl_aff_list_add(list, aff);
	}
	isl_local_space_free(ls);

	dom = isl_basic_set_finalize(dom);

	if (sol->fn(isl_basic_set_copy(dom), list, sol->user) < 0)
		goto error;

	isl_basic_set_free(dom);
	isl_mat_free(M);
	return;
error:
	isl_basic_set_free(dom);
	isl_mat_free(M);
	sol->sol.error = 1;
}

static void sol_for_add_wrap(struct isl_sol *sol,
	struct isl_basic_set *dom, struct isl_mat *M)
{
	sol_for_add((struct isl_sol_for *)sol, dom, M);
}

static struct isl_sol_for *sol_for_init(struct isl_basic_map *bmap, int max,
	int (*fn)(__isl_take isl_basic_set *dom, __isl_take isl_aff_list *list,
		  void *user),
	void *user)
{
	struct isl_sol_for *sol_for = NULL;
	isl_space *dom_dim;
	struct isl_basic_set *dom = NULL;

	sol_for = isl_calloc_type(bmap->ctx, struct isl_sol_for);
	if (!sol_for)
		goto error;

	dom_dim = isl_space_domain(isl_space_copy(bmap->dim));
	dom = isl_basic_set_universe(dom_dim);

	sol_for->sol.rational = ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL);
	sol_for->sol.dec_level.callback.run = &sol_dec_level_wrap;
	sol_for->sol.dec_level.sol = &sol_for->sol;
	sol_for->fn = fn;
	sol_for->user = user;
	sol_for->sol.max = max;
	sol_for->sol.n_out = isl_basic_map_dim(bmap, isl_dim_out);
	sol_for->sol.add = &sol_for_add_wrap;
	sol_for->sol.add_empty = NULL;
	sol_for->sol.free = &sol_for_free_wrap;

	sol_for->sol.context = isl_context_alloc(dom);
	if (!sol_for->sol.context)
		goto error;

	isl_basic_set_free(dom);
	return sol_for;
error:
	isl_basic_set_free(dom);
	sol_for_free(sol_for);
	return NULL;
}

static void sol_for_find_solutions(struct isl_sol_for *sol_for,
	struct isl_tab *tab)
{
	find_solutions_main(&sol_for->sol, tab);
}

int isl_basic_map_foreach_lexopt(__isl_keep isl_basic_map *bmap, int max,
	int (*fn)(__isl_take isl_basic_set *dom, __isl_take isl_aff_list *list,
		  void *user),
	void *user)
{
	struct isl_sol_for *sol_for = NULL;

	bmap = isl_basic_map_copy(bmap);
	bmap = isl_basic_map_detect_equalities(bmap);
	if (!bmap)
		return -1;

	sol_for = sol_for_init(bmap, max, fn, user);
	if (!sol_for)
		goto error;

	if (isl_basic_map_plain_is_empty(bmap))
		/* nothing */;
	else {
		struct isl_tab *tab;
		struct isl_context *context = sol_for->sol.context;
		tab = tab_for_lexmin(bmap,
				context->op->peek_basic_set(context), 1, max);
		tab = context->op->detect_nonnegative_parameters(context, tab);
		sol_for_find_solutions(sol_for, tab);
		if (sol_for->sol.error)
			goto error;
	}

	sol_free(&sol_for->sol);
	isl_basic_map_free(bmap);
	return 0;
error:
	sol_free(&sol_for->sol);
	isl_basic_map_free(bmap);
	return -1;
}

int isl_basic_set_foreach_lexopt(__isl_keep isl_basic_set *bset, int max,
	int (*fn)(__isl_take isl_basic_set *dom, __isl_take isl_aff_list *list,
		  void *user),
	void *user)
{
	return isl_basic_map_foreach_lexopt(bset, max, fn, user);
}

/* Check if the given sequence of len variables starting at pos
 * represents a trivial (i.e., zero) solution.
 * The variables are assumed to be non-negative and to come in pairs,
 * with each pair representing a variable of unrestricted sign.
 * The solution is trivial if each such pair in the sequence consists
 * of two identical values, meaning that the variable being represented
 * has value zero.
 */
static int region_is_trivial(struct isl_tab *tab, int pos, int len)
{
	int i;

	if (len == 0)
		return 0;

	for (i = 0; i < len; i +=  2) {
		int neg_row;
		int pos_row;

		neg_row = tab->var[pos + i].is_row ?
				tab->var[pos + i].index : -1;
		pos_row = tab->var[pos + i + 1].is_row ?
				tab->var[pos + i + 1].index : -1;

		if ((neg_row < 0 ||
		     isl_int_is_zero(tab->mat->row[neg_row][1])) &&
		    (pos_row < 0 ||
		     isl_int_is_zero(tab->mat->row[pos_row][1])))
			continue;

		if (neg_row < 0 || pos_row < 0)
			return 0;
		if (isl_int_ne(tab->mat->row[neg_row][1],
			       tab->mat->row[pos_row][1]))
			return 0;
	}

	return 1;
}

/* Return the index of the first trivial region or -1 if all regions
 * are non-trivial.
 */
static int first_trivial_region(struct isl_tab *tab,
	int n_region, struct isl_region *region)
{
	int i;

	for (i = 0; i < n_region; ++i) {
		if (region_is_trivial(tab, region[i].pos, region[i].len))
			return i;
	}

	return -1;
}

/* Check if the solution is optimal, i.e., whether the first
 * n_op entries are zero.
 */
static int is_optimal(__isl_keep isl_vec *sol, int n_op)
{
	int i;

	for (i = 0; i < n_op; ++i)
		if (!isl_int_is_zero(sol->el[1 + i]))
			return 0;
	return 1;
}

/* Add constraints to "tab" that ensure that any solution is significantly
 * better than that represented by "sol".  That is, find the first
 * relevant (within first n_op) non-zero coefficient and force it (along
 * with all previous coefficients) to be zero.
 * If the solution is already optimal (all relevant coefficients are zero),
 * then just mark the table as empty.
 *
 * This function assumes that at least 2 * n_op more rows and at least
 * 2 * n_op more elements in the constraint array are available in the tableau.
 */
static int force_better_solution(struct isl_tab *tab,
	__isl_keep isl_vec *sol, int n_op)
{
	int i;
	isl_ctx *ctx;
	isl_vec *v = NULL;

	if (!sol)
		return -1;

	for (i = 0; i < n_op; ++i)
		if (!isl_int_is_zero(sol->el[1 + i]))
			break;

	if (i == n_op) {
		if (isl_tab_mark_empty(tab) < 0)
			return -1;
		return 0;
	}

	ctx = isl_vec_get_ctx(sol);
	v = isl_vec_alloc(ctx, 1 + tab->n_var);
	if (!v)
		return -1;

	for (; i >= 0; --i) {
		v = isl_vec_clr(v);
		isl_int_set_si(v->el[1 + i], -1);
		if (add_lexmin_eq(tab, v->el) < 0)
			goto error;
	}

	isl_vec_free(v);
	return 0;
error:
	isl_vec_free(v);
	return -1;
}

struct isl_trivial {
	int update;
	int region;
	int side;
	struct isl_tab_undo *snap;
};

/* Return the lexicographically smallest non-trivial solution of the
 * given ILP problem.
 *
 * All variables are assumed to be non-negative.
 *
 * n_op is the number of initial coordinates to optimize.
 * That is, once a solution has been found, we will only continue looking
 * for solution that result in significantly better values for those
 * initial coordinates.  That is, we only continue looking for solutions
 * that increase the number of initial zeros in this sequence.
 *
 * A solution is non-trivial, if it is non-trivial on each of the
 * specified regions.  Each region represents a sequence of pairs
 * of variables.  A solution is non-trivial on such a region if
 * at least one of these pairs consists of different values, i.e.,
 * such that the non-negative variable represented by the pair is non-zero.
 *
 * Whenever a conflict is encountered, all constraints involved are
 * reported to the caller through a call to "conflict".
 *
 * We perform a simple branch-and-bound backtracking search.
 * Each level in the search represents initially trivial region that is forced
 * to be non-trivial.
 * At each level we consider n cases, where n is the length of the region.
 * In terms of the n/2 variables of unrestricted signs being encoded by
 * the region, we consider the cases
 *	x_0 >= 1
 *	x_0 <= -1
 *	x_0 = 0 and x_1 >= 1
 *	x_0 = 0 and x_1 <= -1
 *	x_0 = 0 and x_1 = 0 and x_2 >= 1
 *	x_0 = 0 and x_1 = 0 and x_2 <= -1
 *	...
 * The cases are considered in this order, assuming that each pair
 * x_i_a x_i_b represents the value x_i_b - x_i_a.
 * That is, x_0 >= 1 is enforced by adding the constraint
 *	x_0_b - x_0_a >= 1
 */
__isl_give isl_vec *isl_tab_basic_set_non_trivial_lexmin(
	__isl_take isl_basic_set *bset, int n_op, int n_region,
	struct isl_region *region,
	int (*conflict)(int con, void *user), void *user)
{
	int i, j;
	int r;
	isl_ctx *ctx;
	isl_vec *v = NULL;
	isl_vec *sol = NULL;
	struct isl_tab *tab;
	struct isl_trivial *triv = NULL;
	int level, init;

	if (!bset)
		return NULL;

	ctx = isl_basic_set_get_ctx(bset);
	sol = isl_vec_alloc(ctx, 0);

	tab = tab_for_lexmin(bset, NULL, 0, 0);
	if (!tab)
		goto error;
	tab->conflict = conflict;
	tab->conflict_user = user;

	v = isl_vec_alloc(ctx, 1 + tab->n_var);
	triv = isl_calloc_array(ctx, struct isl_trivial, n_region);
	if (!v || (n_region && !triv))
		goto error;

	level = 0;
	init = 1;

	while (level >= 0) {
		int side, base;

		if (init) {
			tab = cut_to_integer_lexmin(tab, CUT_ONE);
			if (!tab)
				goto error;
			if (tab->empty)
				goto backtrack;
			r = first_trivial_region(tab, n_region, region);
			if (r < 0) {
				for (i = 0; i < level; ++i)
					triv[i].update = 1;
				isl_vec_free(sol);
				sol = isl_tab_get_sample_value(tab);
				if (!sol)
					goto error;
				if (is_optimal(sol, n_op))
					break;
				goto backtrack;
			}
			if (level >= n_region)
				isl_die(ctx, isl_error_internal,
					"nesting level too deep", goto error);
			if (isl_tab_extend_cons(tab,
					    2 * region[r].len + 2 * n_op) < 0)
				goto error;
			triv[level].region = r;
			triv[level].side = 0;
		}

		r = triv[level].region;
		side = triv[level].side;
		base = 2 * (side/2);

		if (side >= region[r].len) {
backtrack:
			level--;
			init = 0;
			if (level >= 0)
				if (isl_tab_rollback(tab, triv[level].snap) < 0)
					goto error;
			continue;
		}

		if (triv[level].update) {
			if (force_better_solution(tab, sol, n_op) < 0)
				goto error;
			triv[level].update = 0;
		}

		if (side == base && base >= 2) {
			for (j = base - 2; j < base; ++j) {
				v = isl_vec_clr(v);
				isl_int_set_si(v->el[1 + region[r].pos + j], 1);
				if (add_lexmin_eq(tab, v->el) < 0)
					goto error;
			}
		}

		triv[level].snap = isl_tab_snap(tab);
		if (isl_tab_push_basis(tab) < 0)
			goto error;

		v = isl_vec_clr(v);
		isl_int_set_si(v->el[0], -1);
		isl_int_set_si(v->el[1 + region[r].pos + side], -1);
		isl_int_set_si(v->el[1 + region[r].pos + (side ^ 1)], 1);
		tab = add_lexmin_ineq(tab, v->el);

		triv[level].side++;
		level++;
		init = 1;
	}

	free(triv);
	isl_vec_free(v);
	isl_tab_free(tab);
	isl_basic_set_free(bset);

	return sol;
error:
	free(triv);
	isl_vec_free(v);
	isl_tab_free(tab);
	isl_basic_set_free(bset);
	isl_vec_free(sol);
	return NULL;
}

/* Wrapper for a tableau that is used for computing
 * the lexicographically smallest rational point of a non-negative set.
 * This point is represented by the sample value of "tab",
 * unless "tab" is empty.
 */
struct isl_tab_lexmin {
	isl_ctx *ctx;
	struct isl_tab *tab;
};

/* Free "tl" and return NULL.
 */
__isl_null isl_tab_lexmin *isl_tab_lexmin_free(__isl_take isl_tab_lexmin *tl)
{
	if (!tl)
		return NULL;
	isl_ctx_deref(tl->ctx);
	isl_tab_free(tl->tab);
	free(tl);

	return NULL;
}

/* Construct an isl_tab_lexmin for computing
 * the lexicographically smallest rational point in "bset",
 * assuming that all variables are non-negative.
 */
__isl_give isl_tab_lexmin *isl_tab_lexmin_from_basic_set(
	__isl_take isl_basic_set *bset)
{
	isl_ctx *ctx;
	isl_tab_lexmin *tl;

	if (!bset)
		return NULL;

	ctx = isl_basic_set_get_ctx(bset);
	tl = isl_calloc_type(ctx, struct isl_tab_lexmin);
	if (!tl)
		goto error;
	tl->ctx = ctx;
	isl_ctx_ref(ctx);
	tl->tab = tab_for_lexmin(bset, NULL, 0, 0);
	isl_basic_set_free(bset);
	if (!tl->tab)
		return isl_tab_lexmin_free(tl);
	return tl;
error:
	isl_basic_set_free(bset);
	isl_tab_lexmin_free(tl);
	return NULL;
}

/* Return the dimension of the set represented by "tl".
 */
int isl_tab_lexmin_dim(__isl_keep isl_tab_lexmin *tl)
{
	return tl ? tl->tab->n_var : -1;
}

/* Add the equality with coefficients "eq" to "tl", updating the optimal
 * solution if needed.
 * The equality is added as two opposite inequality constraints.
 */
__isl_give isl_tab_lexmin *isl_tab_lexmin_add_eq(__isl_take isl_tab_lexmin *tl,
	isl_int *eq)
{
	unsigned n_var;

	if (!tl || !eq)
		return isl_tab_lexmin_free(tl);

	if (isl_tab_extend_cons(tl->tab, 2) < 0)
		return isl_tab_lexmin_free(tl);
	n_var = tl->tab->n_var;
	isl_seq_neg(eq, eq, 1 + n_var);
	tl->tab = add_lexmin_ineq(tl->tab, eq);
	isl_seq_neg(eq, eq, 1 + n_var);
	tl->tab = add_lexmin_ineq(tl->tab, eq);

	if (!tl->tab)
		return isl_tab_lexmin_free(tl);

	return tl;
}

/* Return the lexicographically smallest rational point in the basic set
 * from which "tl" was constructed.
 * If the original input was empty, then return a zero-length vector.
 */
__isl_give isl_vec *isl_tab_lexmin_get_solution(__isl_keep isl_tab_lexmin *tl)
{
	if (!tl)
		return NULL;
	if (tl->tab->empty)
		return isl_vec_alloc(tl->ctx, 0);
	else
		return isl_tab_get_sample_value(tl->tab);
}

/* Return the lexicographically smallest rational point in "bset",
 * assuming that all variables are non-negative.
 * If "bset" is empty, then return a zero-length vector.
 */
__isl_give isl_vec *isl_tab_basic_set_non_neg_lexmin(
	__isl_take isl_basic_set *bset)
{
	isl_tab_lexmin *tl;
	isl_vec *sol;

	tl = isl_tab_lexmin_from_basic_set(bset);
	sol = isl_tab_lexmin_get_solution(tl);
	isl_tab_lexmin_free(tl);
	return sol;
}

struct isl_sol_pma {
	struct isl_sol	sol;
	isl_pw_multi_aff *pma;
	isl_set *empty;
};

static void sol_pma_free(struct isl_sol_pma *sol_pma)
{
	if (!sol_pma)
		return;
	if (sol_pma->sol.context)
		sol_pma->sol.context->op->free(sol_pma->sol.context);
	isl_pw_multi_aff_free(sol_pma->pma);
	isl_set_free(sol_pma->empty);
	free(sol_pma);
}

/* This function is called for parts of the context where there is
 * no solution, with "bset" corresponding to the context tableau.
 * Simply add the basic set to the set "empty".
 */
static void sol_pma_add_empty(struct isl_sol_pma *sol,
	__isl_take isl_basic_set *bset)
{
	if (!bset || !sol->empty)
		goto error;

	sol->empty = isl_set_grow(sol->empty, 1);
	bset = isl_basic_set_simplify(bset);
	bset = isl_basic_set_finalize(bset);
	sol->empty = isl_set_add_basic_set(sol->empty, bset);
	if (!sol->empty)
		sol->sol.error = 1;
	return;
error:
	isl_basic_set_free(bset);
	sol->sol.error = 1;
}

/* Check that the final columns of "M", starting at "first", are zero.
 */
static isl_stat check_final_columns_are_zero(__isl_keep isl_mat *M,
	unsigned first)
{
	int i;
	unsigned rows, cols, n;

	if (!M)
		return isl_stat_error;
	rows = isl_mat_rows(M);
	cols = isl_mat_cols(M);
	n = cols - first;
	for (i = 0; i < rows; ++i)
		if (isl_seq_first_non_zero(M->row[i] + first, n) != -1)
			isl_die(isl_mat_get_ctx(M), isl_error_internal,
				"final columns should be zero",
				return isl_stat_error);
	return isl_stat_ok;
}

/* Set the affine expressions in "ma" according to the rows in "M", which
 * are defined over the local space "ls".
 * The matrix "M" may have extra (zero) columns beyond the number
 * of variables in "ls".
 */
static __isl_give isl_multi_aff *set_from_affine_matrix(
	__isl_take isl_multi_aff *ma, __isl_take isl_local_space *ls,
	__isl_take isl_mat *M)
{
	int i, dim;
	isl_aff *aff;

	if (!ma || !ls || !M)
		goto error;

	dim = isl_local_space_dim(ls, isl_dim_all);
	if (check_final_columns_are_zero(M, 1 + dim) < 0)
		goto error;
	for (i = 1; i < M->n_row; ++i) {
		aff = isl_aff_alloc(isl_local_space_copy(ls));
		if (aff) {
			isl_int_set(aff->v->el[0], M->row[0][0]);
			isl_seq_cpy(aff->v->el + 1, M->row[i], 1 + dim);
		}
		aff = isl_aff_normalize(aff);
		ma = isl_multi_aff_set_aff(ma, i - 1, aff);
	}
	isl_local_space_free(ls);
	isl_mat_free(M);

	return ma;
error:
	isl_local_space_free(ls);
	isl_mat_free(M);
	isl_multi_aff_free(ma);
	return NULL;
}

/* Given a basic set "dom" that represents the context and an affine
 * matrix "M" that maps the dimensions of the context to the
 * output variables, construct an isl_pw_multi_aff with a single
 * cell corresponding to "dom" and affine expressions copied from "M".
 *
 * Note that the description of the initial context may have involved
 * existentially quantified variables, in which case they also appear
 * in "dom".  These need to be removed before creating the affine
 * expression because an affine expression cannot be defined in terms
 * of existentially quantified variables without a known representation.
 * Since newly added integer divisions are inserted before these
 * existentially quantified variables, they are still in the final
 * positions and the corresponding final columns of "M" are zero
 * because align_context_divs adds the existentially quantified
 * variables of the context to the main tableau without any constraints and
 * any equality constraints that are added later on can only serve
 * to eliminate these existentially quantified variables.
 */
static void sol_pma_add(struct isl_sol_pma *sol,
	__isl_take isl_basic_set *dom, __isl_take isl_mat *M)
{
	isl_local_space *ls;
	isl_multi_aff *maff;
	isl_pw_multi_aff *pma;
	int n_div, n_known;

	n_div = isl_basic_set_dim(dom, isl_dim_div);
	n_known = n_div - sol->sol.context->n_unknown;

	maff = isl_multi_aff_alloc(isl_pw_multi_aff_get_space(sol->pma));
	ls = isl_basic_set_get_local_space(dom);
	ls = isl_local_space_drop_dims(ls, isl_dim_div,
					n_known, n_div - n_known);
	maff = set_from_affine_matrix(maff, ls, M);
	dom = isl_basic_set_simplify(dom);
	dom = isl_basic_set_finalize(dom);
	pma = isl_pw_multi_aff_alloc(isl_set_from_basic_set(dom), maff);
	sol->pma = isl_pw_multi_aff_add_disjoint(sol->pma, pma);
	if (!sol->pma)
		sol->sol.error = 1;
}

static void sol_pma_free_wrap(struct isl_sol *sol)
{
	sol_pma_free((struct isl_sol_pma *)sol);
}

static void sol_pma_add_empty_wrap(struct isl_sol *sol,
	__isl_take isl_basic_set *bset)
{
	sol_pma_add_empty((struct isl_sol_pma *)sol, bset);
}

static void sol_pma_add_wrap(struct isl_sol *sol,
	__isl_take isl_basic_set *dom, __isl_take isl_mat *M)
{
	sol_pma_add((struct isl_sol_pma *)sol, dom, M);
}

/* Construct an isl_sol_pma structure for accumulating the solution.
 * If track_empty is set, then we also keep track of the parts
 * of the context where there is no solution.
 * If max is set, then we are solving a maximization, rather than
 * a minimization problem, which means that the variables in the
 * tableau have value "M - x" rather than "M + x".
 */
static struct isl_sol *sol_pma_init(__isl_keep isl_basic_map *bmap,
	__isl_take isl_basic_set *dom, int track_empty, int max)
{
	struct isl_sol_pma *sol_pma = NULL;

	if (!bmap)
		goto error;

	sol_pma = isl_calloc_type(bmap->ctx, struct isl_sol_pma);
	if (!sol_pma)
		goto error;

	sol_pma->sol.rational = ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL);
	sol_pma->sol.dec_level.callback.run = &sol_dec_level_wrap;
	sol_pma->sol.dec_level.sol = &sol_pma->sol;
	sol_pma->sol.max = max;
	sol_pma->sol.n_out = isl_basic_map_dim(bmap, isl_dim_out);
	sol_pma->sol.add = &sol_pma_add_wrap;
	sol_pma->sol.add_empty = track_empty ? &sol_pma_add_empty_wrap : NULL;
	sol_pma->sol.free = &sol_pma_free_wrap;
	sol_pma->pma = isl_pw_multi_aff_empty(isl_basic_map_get_space(bmap));
	if (!sol_pma->pma)
		goto error;

	sol_pma->sol.context = isl_context_alloc(dom);
	if (!sol_pma->sol.context)
		goto error;

	if (track_empty) {
		sol_pma->empty = isl_set_alloc_space(isl_basic_set_get_space(dom),
							1, ISL_SET_DISJOINT);
		if (!sol_pma->empty)
			goto error;
	}

	isl_basic_set_free(dom);
	return &sol_pma->sol;
error:
	isl_basic_set_free(dom);
	sol_pma_free(sol_pma);
	return NULL;
}

/* Base case of isl_tab_basic_map_partial_lexopt, after removing
 * some obvious symmetries.
 *
 * We call basic_map_partial_lexopt_base_sol and extract the results.
 */
static __isl_give isl_pw_multi_aff *basic_map_partial_lexopt_base_pw_multi_aff(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, int max)
{
	isl_pw_multi_aff *result = NULL;
	struct isl_sol *sol;
	struct isl_sol_pma *sol_pma;

	sol = basic_map_partial_lexopt_base_sol(bmap, dom, empty, max,
						&sol_pma_init);
	if (!sol)
		return NULL;
	sol_pma = (struct isl_sol_pma *) sol;

	result = isl_pw_multi_aff_copy(sol_pma->pma);
	if (empty)
		*empty = isl_set_copy(sol_pma->empty);
	sol_free(&sol_pma->sol);
	return result;
}

/* Given that the last input variable of "maff" represents the minimum
 * of some bounds, check whether we need to plug in the expression
 * of the minimum.
 *
 * In particular, check if the last input variable appears in any
 * of the expressions in "maff".
 */
static int need_substitution(__isl_keep isl_multi_aff *maff)
{
	int i;
	unsigned pos;

	pos = isl_multi_aff_dim(maff, isl_dim_in) - 1;

	for (i = 0; i < maff->n; ++i)
		if (isl_aff_involves_dims(maff->p[i], isl_dim_in, pos, 1))
			return 1;

	return 0;
}

/* Given a set of upper bounds on the last "input" variable m,
 * construct a piecewise affine expression that selects
 * the minimal upper bound to m, i.e.,
 * divide the space into cells where one
 * of the upper bounds is smaller than all the others and select
 * this upper bound on that cell.
 *
 * In particular, if there are n bounds b_i, then the result
 * consists of n cell, each one of the form
 *
 *	b_i <= b_j	for j > i
 *	b_i <  b_j	for j < i
 *
 * The affine expression on this cell is
 *
 *	b_i
 */
static __isl_give isl_pw_aff *set_minimum_pa(__isl_take isl_space *space,
	__isl_take isl_mat *var)
{
	int i;
	isl_aff *aff = NULL;
	isl_basic_set *bset = NULL;
	isl_pw_aff *paff = NULL;
	isl_space *pw_space;
	isl_local_space *ls = NULL;

	if (!space || !var)
		goto error;

	ls = isl_local_space_from_space(isl_space_copy(space));
	pw_space = isl_space_copy(space);
	pw_space = isl_space_from_domain(pw_space);
	pw_space = isl_space_add_dims(pw_space, isl_dim_out, 1);
	paff = isl_pw_aff_alloc_size(pw_space, var->n_row);

	for (i = 0; i < var->n_row; ++i) {
		isl_pw_aff *paff_i;

		aff = isl_aff_alloc(isl_local_space_copy(ls));
		bset = isl_basic_set_alloc_space(isl_space_copy(space), 0,
					       0, var->n_row - 1);
		if (!aff || !bset)
			goto error;
		isl_int_set_si(aff->v->el[0], 1);
		isl_seq_cpy(aff->v->el + 1, var->row[i], var->n_col);
		isl_int_set_si(aff->v->el[1 + var->n_col], 0);
		bset = select_minimum(bset, var, i);
		paff_i = isl_pw_aff_alloc(isl_set_from_basic_set(bset), aff);
		paff = isl_pw_aff_add_disjoint(paff, paff_i);
	}

	isl_local_space_free(ls);
	isl_space_free(space);
	isl_mat_free(var);
	return paff;
error:
	isl_aff_free(aff);
	isl_basic_set_free(bset);
	isl_pw_aff_free(paff);
	isl_local_space_free(ls);
	isl_space_free(space);
	isl_mat_free(var);
	return NULL;
}

/* Given a piecewise multi-affine expression of which the last input variable
 * is the minimum of the bounds in "cst", plug in the value of the minimum.
 * This minimum expression is given in "min_expr_pa".
 * The set "min_expr" contains the same information, but in the form of a set.
 * The variable is subsequently projected out.
 *
 * The implementation is similar to those of "split" and "split_domain".
 * If the variable appears in a given expression, then minimum expression
 * is plugged in.  Otherwise, if the variable appears in the constraints
 * and a split is required, then the domain is split.  Otherwise, no split
 * is performed.
 */
static __isl_give isl_pw_multi_aff *split_domain_pma(
	__isl_take isl_pw_multi_aff *opt, __isl_take isl_pw_aff *min_expr_pa,
	__isl_take isl_set *min_expr, __isl_take isl_mat *cst)
{
	int n_in;
	int i;
	isl_space *space;
	isl_pw_multi_aff *res;

	if (!opt || !min_expr || !cst)
		goto error;

	n_in = isl_pw_multi_aff_dim(opt, isl_dim_in);
	space = isl_pw_multi_aff_get_space(opt);
	space = isl_space_drop_dims(space, isl_dim_in, n_in - 1, 1);
	res = isl_pw_multi_aff_empty(space);

	for (i = 0; i < opt->n; ++i) {
		isl_pw_multi_aff *pma;

		pma = isl_pw_multi_aff_alloc(isl_set_copy(opt->p[i].set),
					 isl_multi_aff_copy(opt->p[i].maff));
		if (need_substitution(opt->p[i].maff))
			pma = isl_pw_multi_aff_substitute(pma,
					isl_dim_in, n_in - 1, min_expr_pa);
		else if (need_split_set(opt->p[i].set, cst))
			pma = isl_pw_multi_aff_intersect_domain(pma,
						       isl_set_copy(min_expr));
		pma = isl_pw_multi_aff_project_out(pma,
						    isl_dim_in, n_in - 1, 1);

		res = isl_pw_multi_aff_add_disjoint(res, pma);
	}

	isl_pw_multi_aff_free(opt);
	isl_pw_aff_free(min_expr_pa);
	isl_set_free(min_expr);
	isl_mat_free(cst);
	return res;
error:
	isl_pw_multi_aff_free(opt);
	isl_pw_aff_free(min_expr_pa);
	isl_set_free(min_expr);
	isl_mat_free(cst);
	return NULL;
}

static __isl_give isl_pw_multi_aff *basic_map_partial_lexopt_pw_multi_aff(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, int max);

/* This function is called from basic_map_partial_lexopt_symm.
 * The last variable of "bmap" and "dom" corresponds to the minimum
 * of the bounds in "cst".  "map_space" is the space of the original
 * input relation (of basic_map_partial_lexopt_symm) and "set_space"
 * is the space of the original domain.
 *
 * We recursively call basic_map_partial_lexopt and then plug in
 * the definition of the minimum in the result.
 */
static __isl_give isl_pw_multi_aff *
basic_map_partial_lexopt_symm_core_pw_multi_aff(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *dom,
	__isl_give isl_set **empty, int max, __isl_take isl_mat *cst,
	__isl_take isl_space *map_space, __isl_take isl_space *set_space)
{
	isl_pw_multi_aff *opt;
	isl_pw_aff *min_expr_pa;
	isl_set *min_expr;

	min_expr = set_minimum(isl_basic_set_get_space(dom), isl_mat_copy(cst));
	min_expr_pa = set_minimum_pa(isl_basic_set_get_space(dom),
					isl_mat_copy(cst));

	opt = basic_map_partial_lexopt_pw_multi_aff(bmap, dom, empty, max);

	if (empty) {
		*empty = split(*empty,
			       isl_set_copy(min_expr), isl_mat_copy(cst));
		*empty = isl_set_reset_space(*empty, set_space);
	}

	opt = split_domain_pma(opt, min_expr_pa, min_expr, cst);
	opt = isl_pw_multi_aff_reset_space(opt, map_space);

	return opt;
}

#undef TYPE
#define TYPE	isl_pw_multi_aff
#undef SUFFIX
#define SUFFIX	_pw_multi_aff
#include "isl_tab_lexopt_templ.c"
