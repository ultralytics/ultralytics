/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include "isl_sample.h"
#include <isl/vec.h>
#include <isl/mat.h>
#include <isl_seq.h>
#include "isl_equalities.h"
#include "isl_tab.h"
#include "isl_basis_reduction.h"
#include <isl_factorization.h>
#include <isl_point_private.h>
#include <isl_options_private.h>
#include <isl_vec_private.h>

#include <bset_from_bmap.c>
#include <set_to_map.c>

static struct isl_vec *empty_sample(struct isl_basic_set *bset)
{
	struct isl_vec *vec;

	vec = isl_vec_alloc(bset->ctx, 0);
	isl_basic_set_free(bset);
	return vec;
}

/* Construct a zero sample of the same dimension as bset.
 * As a special case, if bset is zero-dimensional, this
 * function creates a zero-dimensional sample point.
 */
static struct isl_vec *zero_sample(struct isl_basic_set *bset)
{
	unsigned dim;
	struct isl_vec *sample;

	dim = isl_basic_set_total_dim(bset);
	sample = isl_vec_alloc(bset->ctx, 1 + dim);
	if (sample) {
		isl_int_set_si(sample->el[0], 1);
		isl_seq_clr(sample->el + 1, dim);
	}
	isl_basic_set_free(bset);
	return sample;
}

static struct isl_vec *interval_sample(struct isl_basic_set *bset)
{
	int i;
	isl_int t;
	struct isl_vec *sample;

	bset = isl_basic_set_simplify(bset);
	if (!bset)
		return NULL;
	if (isl_basic_set_plain_is_empty(bset))
		return empty_sample(bset);
	if (bset->n_eq == 0 && bset->n_ineq == 0)
		return zero_sample(bset);

	sample = isl_vec_alloc(bset->ctx, 2);
	if (!sample)
		goto error;
	if (!bset)
		return NULL;
	isl_int_set_si(sample->block.data[0], 1);

	if (bset->n_eq > 0) {
		isl_assert(bset->ctx, bset->n_eq == 1, goto error);
		isl_assert(bset->ctx, bset->n_ineq == 0, goto error);
		if (isl_int_is_one(bset->eq[0][1]))
			isl_int_neg(sample->el[1], bset->eq[0][0]);
		else {
			isl_assert(bset->ctx, isl_int_is_negone(bset->eq[0][1]),
				   goto error);
			isl_int_set(sample->el[1], bset->eq[0][0]);
		}
		isl_basic_set_free(bset);
		return sample;
	}

	isl_int_init(t);
	if (isl_int_is_one(bset->ineq[0][1]))
		isl_int_neg(sample->block.data[1], bset->ineq[0][0]);
	else
		isl_int_set(sample->block.data[1], bset->ineq[0][0]);
	for (i = 1; i < bset->n_ineq; ++i) {
		isl_seq_inner_product(sample->block.data,
					bset->ineq[i], 2, &t);
		if (isl_int_is_neg(t))
			break;
	}
	isl_int_clear(t);
	if (i < bset->n_ineq) {
		isl_vec_free(sample);
		return empty_sample(bset);
	}

	isl_basic_set_free(bset);
	return sample;
error:
	isl_basic_set_free(bset);
	isl_vec_free(sample);
	return NULL;
}

/* Find a sample integer point, if any, in bset, which is known
 * to have equalities.  If bset contains no integer points, then
 * return a zero-length vector.
 * We simply remove the known equalities, compute a sample
 * in the resulting bset, using the specified recurse function,
 * and then transform the sample back to the original space.
 */
static struct isl_vec *sample_eq(struct isl_basic_set *bset,
	struct isl_vec *(*recurse)(struct isl_basic_set *))
{
	struct isl_mat *T;
	struct isl_vec *sample;

	if (!bset)
		return NULL;

	bset = isl_basic_set_remove_equalities(bset, &T, NULL);
	sample = recurse(bset);
	if (!sample || sample->size == 0)
		isl_mat_free(T);
	else
		sample = isl_mat_vec_product(T, sample);
	return sample;
}

/* Return a matrix containing the equalities of the tableau
 * in constraint form.  The tableau is assumed to have
 * an associated bset that has been kept up-to-date.
 */
static struct isl_mat *tab_equalities(struct isl_tab *tab)
{
	int i, j;
	int n_eq;
	struct isl_mat *eq;
	struct isl_basic_set *bset;

	if (!tab)
		return NULL;

	bset = isl_tab_peek_bset(tab);
	isl_assert(tab->mat->ctx, bset, return NULL);

	n_eq = tab->n_var - tab->n_col + tab->n_dead;
	if (tab->empty || n_eq == 0)
		return isl_mat_alloc(tab->mat->ctx, 0, tab->n_var);
	if (n_eq == tab->n_var)
		return isl_mat_identity(tab->mat->ctx, tab->n_var);

	eq = isl_mat_alloc(tab->mat->ctx, n_eq, tab->n_var);
	if (!eq)
		return NULL;
	for (i = 0, j = 0; i < tab->n_con; ++i) {
		if (tab->con[i].is_row)
			continue;
		if (tab->con[i].index >= 0 && tab->con[i].index >= tab->n_dead)
			continue;
		if (i < bset->n_eq)
			isl_seq_cpy(eq->row[j], bset->eq[i] + 1, tab->n_var);
		else
			isl_seq_cpy(eq->row[j],
				    bset->ineq[i - bset->n_eq] + 1, tab->n_var);
		++j;
	}
	isl_assert(bset->ctx, j == n_eq, goto error);
	return eq;
error:
	isl_mat_free(eq);
	return NULL;
}

/* Compute and return an initial basis for the bounded tableau "tab".
 *
 * If the tableau is either full-dimensional or zero-dimensional,
 * the we simply return an identity matrix.
 * Otherwise, we construct a basis whose first directions correspond
 * to equalities.
 */
static struct isl_mat *initial_basis(struct isl_tab *tab)
{
	int n_eq;
	struct isl_mat *eq;
	struct isl_mat *Q;

	tab->n_unbounded = 0;
	tab->n_zero = n_eq = tab->n_var - tab->n_col + tab->n_dead;
	if (tab->empty || n_eq == 0 || n_eq == tab->n_var)
		return isl_mat_identity(tab->mat->ctx, 1 + tab->n_var);

	eq = tab_equalities(tab);
	eq = isl_mat_left_hermite(eq, 0, NULL, &Q);
	if (!eq)
		return NULL;
	isl_mat_free(eq);

	Q = isl_mat_lin_to_aff(Q);
	return Q;
}

/* Compute the minimum of the current ("level") basis row over "tab"
 * and store the result in position "level" of "min".
 *
 * This function assumes that at least one more row and at least
 * one more element in the constraint array are available in the tableau.
 */
static enum isl_lp_result compute_min(isl_ctx *ctx, struct isl_tab *tab,
	__isl_keep isl_vec *min, int level)
{
	return isl_tab_min(tab, tab->basis->row[1 + level],
			    ctx->one, &min->el[level], NULL, 0);
}

/* Compute the maximum of the current ("level") basis row over "tab"
 * and store the result in position "level" of "max".
 *
 * This function assumes that at least one more row and at least
 * one more element in the constraint array are available in the tableau.
 */
static enum isl_lp_result compute_max(isl_ctx *ctx, struct isl_tab *tab,
	__isl_keep isl_vec *max, int level)
{
	enum isl_lp_result res;
	unsigned dim = tab->n_var;

	isl_seq_neg(tab->basis->row[1 + level] + 1,
		    tab->basis->row[1 + level] + 1, dim);
	res = isl_tab_min(tab, tab->basis->row[1 + level],
		    ctx->one, &max->el[level], NULL, 0);
	isl_seq_neg(tab->basis->row[1 + level] + 1,
		    tab->basis->row[1 + level] + 1, dim);
	isl_int_neg(max->el[level], max->el[level]);

	return res;
}

/* Perform a greedy search for an integer point in the set represented
 * by "tab", given that the minimal rational value (rounded up to the
 * nearest integer) at "level" is smaller than the maximal rational
 * value (rounded down to the nearest integer).
 *
 * Return 1 if we have found an integer point (if tab->n_unbounded > 0
 * then we may have only found integer values for the bounded dimensions
 * and it is the responsibility of the caller to extend this solution
 * to the unbounded dimensions).
 * Return 0 if greedy search did not result in a solution.
 * Return -1 if some error occurred.
 *
 * We assign a value half-way between the minimum and the maximum
 * to the current dimension and check if the minimal value of the
 * next dimension is still smaller than (or equal) to the maximal value.
 * We continue this process until either
 * - the minimal value (rounded up) is greater than the maximal value
 *	(rounded down).  In this case, greedy search has failed.
 * - we have exhausted all bounded dimensions, meaning that we have
 *	found a solution.
 * - the sample value of the tableau is integral.
 * - some error has occurred.
 */
static int greedy_search(isl_ctx *ctx, struct isl_tab *tab,
	__isl_keep isl_vec *min, __isl_keep isl_vec *max, int level)
{
	struct isl_tab_undo *snap;
	enum isl_lp_result res;

	snap = isl_tab_snap(tab);

	do {
		isl_int_add(tab->basis->row[1 + level][0],
			    min->el[level], max->el[level]);
		isl_int_fdiv_q_ui(tab->basis->row[1 + level][0],
			    tab->basis->row[1 + level][0], 2);
		isl_int_neg(tab->basis->row[1 + level][0],
			    tab->basis->row[1 + level][0]);
		if (isl_tab_add_valid_eq(tab, tab->basis->row[1 + level]) < 0)
			return -1;
		isl_int_set_si(tab->basis->row[1 + level][0], 0);

		if (++level >= tab->n_var - tab->n_unbounded)
			return 1;
		if (isl_tab_sample_is_integer(tab))
			return 1;

		res = compute_min(ctx, tab, min, level);
		if (res == isl_lp_error)
			return -1;
		if (res != isl_lp_ok)
			isl_die(ctx, isl_error_internal,
				"expecting bounded rational solution",
				return -1);
		res = compute_max(ctx, tab, max, level);
		if (res == isl_lp_error)
			return -1;
		if (res != isl_lp_ok)
			isl_die(ctx, isl_error_internal,
				"expecting bounded rational solution",
				return -1);
	} while (isl_int_le(min->el[level], max->el[level]));

	if (isl_tab_rollback(tab, snap) < 0)
		return -1;

	return 0;
}

/* Given a tableau representing a set, find and return
 * an integer point in the set, if there is any.
 *
 * We perform a depth first search
 * for an integer point, by scanning all possible values in the range
 * attained by a basis vector, where an initial basis may have been set
 * by the calling function.  Otherwise an initial basis that exploits
 * the equalities in the tableau is created.
 * tab->n_zero is currently ignored and is clobbered by this function.
 *
 * The tableau is allowed to have unbounded direction, but then
 * the calling function needs to set an initial basis, with the
 * unbounded directions last and with tab->n_unbounded set
 * to the number of unbounded directions.
 * Furthermore, the calling functions needs to add shifted copies
 * of all constraints involving unbounded directions to ensure
 * that any feasible rational value in these directions can be rounded
 * up to yield a feasible integer value.
 * In particular, let B define the given basis x' = B x
 * and let T be the inverse of B, i.e., X = T x'.
 * Let a x + c >= 0 be a constraint of the set represented by the tableau,
 * or a T x' + c >= 0 in terms of the given basis.  Assume that
 * the bounded directions have an integer value, then we can safely
 * round up the values for the unbounded directions if we make sure
 * that x' not only satisfies the original constraint, but also
 * the constraint "a T x' + c + s >= 0" with s the sum of all
 * negative values in the last n_unbounded entries of "a T".
 * The calling function therefore needs to add the constraint
 * a x + c + s >= 0.  The current function then scans the first
 * directions for an integer value and once those have been found,
 * it can compute "T ceil(B x)" to yield an integer point in the set.
 * Note that during the search, the first rows of B may be changed
 * by a basis reduction, but the last n_unbounded rows of B remain
 * unaltered and are also not mixed into the first rows.
 *
 * The search is implemented iteratively.  "level" identifies the current
 * basis vector.  "init" is true if we want the first value at the current
 * level and false if we want the next value.
 *
 * At the start of each level, we first check if we can find a solution
 * using greedy search.  If not, we continue with the exhaustive search.
 *
 * The initial basis is the identity matrix.  If the range in some direction
 * contains more than one integer value, we perform basis reduction based
 * on the value of ctx->opt->gbr
 *	- ISL_GBR_NEVER:	never perform basis reduction
 *	- ISL_GBR_ONCE:		only perform basis reduction the first
 *				time such a range is encountered
 *	- ISL_GBR_ALWAYS:	always perform basis reduction when
 *				such a range is encountered
 *
 * When ctx->opt->gbr is set to ISL_GBR_ALWAYS, then we allow the basis
 * reduction computation to return early.  That is, as soon as it
 * finds a reasonable first direction.
 */ 
struct isl_vec *isl_tab_sample(struct isl_tab *tab)
{
	unsigned dim;
	unsigned gbr;
	struct isl_ctx *ctx;
	struct isl_vec *sample;
	struct isl_vec *min;
	struct isl_vec *max;
	enum isl_lp_result res;
	int level;
	int init;
	int reduced;
	struct isl_tab_undo **snap;

	if (!tab)
		return NULL;
	if (tab->empty)
		return isl_vec_alloc(tab->mat->ctx, 0);

	if (!tab->basis)
		tab->basis = initial_basis(tab);
	if (!tab->basis)
		return NULL;
	isl_assert(tab->mat->ctx, tab->basis->n_row == tab->n_var + 1,
		    return NULL);
	isl_assert(tab->mat->ctx, tab->basis->n_col == tab->n_var + 1,
		    return NULL);

	ctx = tab->mat->ctx;
	dim = tab->n_var;
	gbr = ctx->opt->gbr;

	if (tab->n_unbounded == tab->n_var) {
		sample = isl_tab_get_sample_value(tab);
		sample = isl_mat_vec_product(isl_mat_copy(tab->basis), sample);
		sample = isl_vec_ceil(sample);
		sample = isl_mat_vec_inverse_product(isl_mat_copy(tab->basis),
							sample);
		return sample;
	}

	if (isl_tab_extend_cons(tab, dim + 1) < 0)
		return NULL;

	min = isl_vec_alloc(ctx, dim);
	max = isl_vec_alloc(ctx, dim);
	snap = isl_alloc_array(ctx, struct isl_tab_undo *, dim);

	if (!min || !max || !snap)
		goto error;

	level = 0;
	init = 1;
	reduced = 0;

	while (level >= 0) {
		if (init) {
			int choice;

			res = compute_min(ctx, tab, min, level);
			if (res == isl_lp_error)
				goto error;
			if (res != isl_lp_ok)
				isl_die(ctx, isl_error_internal,
					"expecting bounded rational solution",
					goto error);
			if (isl_tab_sample_is_integer(tab))
				break;
			res = compute_max(ctx, tab, max, level);
			if (res == isl_lp_error)
				goto error;
			if (res != isl_lp_ok)
				isl_die(ctx, isl_error_internal,
					"expecting bounded rational solution",
					goto error);
			if (isl_tab_sample_is_integer(tab))
				break;
			choice = isl_int_lt(min->el[level], max->el[level]);
			if (choice) {
				int g;
				g = greedy_search(ctx, tab, min, max, level);
				if (g < 0)
					goto error;
				if (g)
					break;
			}
			if (!reduced && choice &&
			    ctx->opt->gbr != ISL_GBR_NEVER) {
				unsigned gbr_only_first;
				if (ctx->opt->gbr == ISL_GBR_ONCE)
					ctx->opt->gbr = ISL_GBR_NEVER;
				tab->n_zero = level;
				gbr_only_first = ctx->opt->gbr_only_first;
				ctx->opt->gbr_only_first =
					ctx->opt->gbr == ISL_GBR_ALWAYS;
				tab = isl_tab_compute_reduced_basis(tab);
				ctx->opt->gbr_only_first = gbr_only_first;
				if (!tab || !tab->basis)
					goto error;
				reduced = 1;
				continue;
			}
			reduced = 0;
			snap[level] = isl_tab_snap(tab);
		} else
			isl_int_add_ui(min->el[level], min->el[level], 1);

		if (isl_int_gt(min->el[level], max->el[level])) {
			level--;
			init = 0;
			if (level >= 0)
				if (isl_tab_rollback(tab, snap[level]) < 0)
					goto error;
			continue;
		}
		isl_int_neg(tab->basis->row[1 + level][0], min->el[level]);
		if (isl_tab_add_valid_eq(tab, tab->basis->row[1 + level]) < 0)
			goto error;
		isl_int_set_si(tab->basis->row[1 + level][0], 0);
		if (level + tab->n_unbounded < dim - 1) {
			++level;
			init = 1;
			continue;
		}
		break;
	}

	if (level >= 0) {
		sample = isl_tab_get_sample_value(tab);
		if (!sample)
			goto error;
		if (tab->n_unbounded && !isl_int_is_one(sample->el[0])) {
			sample = isl_mat_vec_product(isl_mat_copy(tab->basis),
						     sample);
			sample = isl_vec_ceil(sample);
			sample = isl_mat_vec_inverse_product(
					isl_mat_copy(tab->basis), sample);
		}
	} else
		sample = isl_vec_alloc(ctx, 0);

	ctx->opt->gbr = gbr;
	isl_vec_free(min);
	isl_vec_free(max);
	free(snap);
	return sample;
error:
	ctx->opt->gbr = gbr;
	isl_vec_free(min);
	isl_vec_free(max);
	free(snap);
	return NULL;
}

static struct isl_vec *sample_bounded(struct isl_basic_set *bset);

/* Compute a sample point of the given basic set, based on the given,
 * non-trivial factorization.
 */
static __isl_give isl_vec *factored_sample(__isl_take isl_basic_set *bset,
	__isl_take isl_factorizer *f)
{
	int i, n;
	isl_vec *sample = NULL;
	isl_ctx *ctx;
	unsigned nparam;
	unsigned nvar;

	ctx = isl_basic_set_get_ctx(bset);
	if (!ctx)
		goto error;

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	nvar = isl_basic_set_dim(bset, isl_dim_set);

	sample = isl_vec_alloc(ctx, 1 + isl_basic_set_total_dim(bset));
	if (!sample)
		goto error;
	isl_int_set_si(sample->el[0], 1);

	bset = isl_morph_basic_set(isl_morph_copy(f->morph), bset);

	for (i = 0, n = 0; i < f->n_group; ++i) {
		isl_basic_set *bset_i;
		isl_vec *sample_i;

		bset_i = isl_basic_set_copy(bset);
		bset_i = isl_basic_set_drop_constraints_involving(bset_i,
			    nparam + n + f->len[i], nvar - n - f->len[i]);
		bset_i = isl_basic_set_drop_constraints_involving(bset_i,
			    nparam, n);
		bset_i = isl_basic_set_drop(bset_i, isl_dim_set,
			    n + f->len[i], nvar - n - f->len[i]);
		bset_i = isl_basic_set_drop(bset_i, isl_dim_set, 0, n);

		sample_i = sample_bounded(bset_i);
		if (!sample_i)
			goto error;
		if (sample_i->size == 0) {
			isl_basic_set_free(bset);
			isl_factorizer_free(f);
			isl_vec_free(sample);
			return sample_i;
		}
		isl_seq_cpy(sample->el + 1 + nparam + n,
			    sample_i->el + 1, f->len[i]);
		isl_vec_free(sample_i);

		n += f->len[i];
	}

	f->morph = isl_morph_inverse(f->morph);
	sample = isl_morph_vec(isl_morph_copy(f->morph), sample);

	isl_basic_set_free(bset);
	isl_factorizer_free(f);
	return sample;
error:
	isl_basic_set_free(bset);
	isl_factorizer_free(f);
	isl_vec_free(sample);
	return NULL;
}

/* Given a basic set that is known to be bounded, find and return
 * an integer point in the basic set, if there is any.
 *
 * After handling some trivial cases, we construct a tableau
 * and then use isl_tab_sample to find a sample, passing it
 * the identity matrix as initial basis.
 */ 
static struct isl_vec *sample_bounded(struct isl_basic_set *bset)
{
	unsigned dim;
	struct isl_vec *sample;
	struct isl_tab *tab = NULL;
	isl_factorizer *f;

	if (!bset)
		return NULL;

	if (isl_basic_set_plain_is_empty(bset))
		return empty_sample(bset);

	dim = isl_basic_set_total_dim(bset);
	if (dim == 0)
		return zero_sample(bset);
	if (dim == 1)
		return interval_sample(bset);
	if (bset->n_eq > 0)
		return sample_eq(bset, sample_bounded);

	f = isl_basic_set_factorizer(bset);
	if (!f)
		goto error;
	if (f->n_group != 0)
		return factored_sample(bset, f);
	isl_factorizer_free(f);

	tab = isl_tab_from_basic_set(bset, 1);
	if (tab && tab->empty) {
		isl_tab_free(tab);
		ISL_F_SET(bset, ISL_BASIC_SET_EMPTY);
		sample = isl_vec_alloc(isl_basic_set_get_ctx(bset), 0);
		isl_basic_set_free(bset);
		return sample;
	}

	if (!ISL_F_ISSET(bset, ISL_BASIC_SET_NO_IMPLICIT))
		if (isl_tab_detect_implicit_equalities(tab) < 0)
			goto error;

	sample = isl_tab_sample(tab);
	if (!sample)
		goto error;

	if (sample->size > 0) {
		isl_vec_free(bset->sample);
		bset->sample = isl_vec_copy(sample);
	}

	isl_basic_set_free(bset);
	isl_tab_free(tab);
	return sample;
error:
	isl_basic_set_free(bset);
	isl_tab_free(tab);
	return NULL;
}

/* Given a basic set "bset" and a value "sample" for the first coordinates
 * of bset, plug in these values and drop the corresponding coordinates.
 *
 * We do this by computing the preimage of the transformation
 *
 *	     [ 1 0 ]
 *	x =  [ s 0 ] x'
 *	     [ 0 I ]
 *
 * where [1 s] is the sample value and I is the identity matrix of the
 * appropriate dimension.
 */
static struct isl_basic_set *plug_in(struct isl_basic_set *bset,
	struct isl_vec *sample)
{
	int i;
	unsigned total;
	struct isl_mat *T;

	if (!bset || !sample)
		goto error;

	total = isl_basic_set_total_dim(bset);
	T = isl_mat_alloc(bset->ctx, 1 + total, 1 + total - (sample->size - 1));
	if (!T)
		goto error;

	for (i = 0; i < sample->size; ++i) {
		isl_int_set(T->row[i][0], sample->el[i]);
		isl_seq_clr(T->row[i] + 1, T->n_col - 1);
	}
	for (i = 0; i < T->n_col - 1; ++i) {
		isl_seq_clr(T->row[sample->size + i], T->n_col);
		isl_int_set_si(T->row[sample->size + i][1 + i], 1);
	}
	isl_vec_free(sample);

	bset = isl_basic_set_preimage(bset, T);
	return bset;
error:
	isl_basic_set_free(bset);
	isl_vec_free(sample);
	return NULL;
}

/* Given a basic set "bset", return any (possibly non-integer) point
 * in the basic set.
 */
static struct isl_vec *rational_sample(struct isl_basic_set *bset)
{
	struct isl_tab *tab;
	struct isl_vec *sample;

	if (!bset)
		return NULL;

	tab = isl_tab_from_basic_set(bset, 0);
	sample = isl_tab_get_sample_value(tab);
	isl_tab_free(tab);

	isl_basic_set_free(bset);

	return sample;
}

/* Given a linear cone "cone" and a rational point "vec",
 * construct a polyhedron with shifted copies of the constraints in "cone",
 * i.e., a polyhedron with "cone" as its recession cone, such that each
 * point x in this polyhedron is such that the unit box positioned at x
 * lies entirely inside the affine cone 'vec + cone'.
 * Any rational point in this polyhedron may therefore be rounded up
 * to yield an integer point that lies inside said affine cone.
 *
 * Denote the constraints of cone by "<a_i, x> >= 0" and the rational
 * point "vec" by v/d.
 * Let b_i = <a_i, v>.  Then the affine cone 'vec + cone' is given
 * by <a_i, x> - b/d >= 0.
 * The polyhedron <a_i, x> - ceil{b/d} >= 0 is a subset of this affine cone.
 * We prefer this polyhedron over the actual affine cone because it doesn't
 * require a scaling of the constraints.
 * If each of the vertices of the unit cube positioned at x lies inside
 * this polyhedron, then the whole unit cube at x lies inside the affine cone.
 * We therefore impose that x' = x + \sum e_i, for any selection of unit
 * vectors lies inside the polyhedron, i.e.,
 *
 *	<a_i, x'> - ceil{b/d} = <a_i, x> + sum a_i - ceil{b/d} >= 0
 *
 * The most stringent of these constraints is the one that selects
 * all negative a_i, so the polyhedron we are looking for has constraints
 *
 *	<a_i, x> + sum_{a_i < 0} a_i - ceil{b/d} >= 0
 *
 * Note that if cone were known to have only non-negative rays
 * (which can be accomplished by a unimodular transformation),
 * then we would only have to check the points x' = x + e_i
 * and we only have to add the smallest negative a_i (if any)
 * instead of the sum of all negative a_i.
 */
static struct isl_basic_set *shift_cone(struct isl_basic_set *cone,
	struct isl_vec *vec)
{
	int i, j, k;
	unsigned total;

	struct isl_basic_set *shift = NULL;

	if (!cone || !vec)
		goto error;

	isl_assert(cone->ctx, cone->n_eq == 0, goto error);

	total = isl_basic_set_total_dim(cone);

	shift = isl_basic_set_alloc_space(isl_basic_set_get_space(cone),
					0, 0, cone->n_ineq);

	for (i = 0; i < cone->n_ineq; ++i) {
		k = isl_basic_set_alloc_inequality(shift);
		if (k < 0)
			goto error;
		isl_seq_cpy(shift->ineq[k] + 1, cone->ineq[i] + 1, total);
		isl_seq_inner_product(shift->ineq[k] + 1, vec->el + 1, total,
				      &shift->ineq[k][0]);
		isl_int_cdiv_q(shift->ineq[k][0],
			       shift->ineq[k][0], vec->el[0]);
		isl_int_neg(shift->ineq[k][0], shift->ineq[k][0]);
		for (j = 0; j < total; ++j) {
			if (isl_int_is_nonneg(shift->ineq[k][1 + j]))
				continue;
			isl_int_add(shift->ineq[k][0],
				    shift->ineq[k][0], shift->ineq[k][1 + j]);
		}
	}

	isl_basic_set_free(cone);
	isl_vec_free(vec);

	return isl_basic_set_finalize(shift);
error:
	isl_basic_set_free(shift);
	isl_basic_set_free(cone);
	isl_vec_free(vec);
	return NULL;
}

/* Given a rational point vec in a (transformed) basic set,
 * such that cone is the recession cone of the original basic set,
 * "round up" the rational point to an integer point.
 *
 * We first check if the rational point just happens to be integer.
 * If not, we transform the cone in the same way as the basic set,
 * pick a point x in this cone shifted to the rational point such that
 * the whole unit cube at x is also inside this affine cone.
 * Then we simply round up the coordinates of x and return the
 * resulting integer point.
 */
static struct isl_vec *round_up_in_cone(struct isl_vec *vec,
	struct isl_basic_set *cone, struct isl_mat *U)
{
	unsigned total;

	if (!vec || !cone || !U)
		goto error;

	isl_assert(vec->ctx, vec->size != 0, goto error);
	if (isl_int_is_one(vec->el[0])) {
		isl_mat_free(U);
		isl_basic_set_free(cone);
		return vec;
	}

	total = isl_basic_set_total_dim(cone);
	cone = isl_basic_set_preimage(cone, U);
	cone = isl_basic_set_remove_dims(cone, isl_dim_set,
					 0, total - (vec->size - 1));

	cone = shift_cone(cone, vec);

	vec = rational_sample(cone);
	vec = isl_vec_ceil(vec);
	return vec;
error:
	isl_mat_free(U);
	isl_vec_free(vec);
	isl_basic_set_free(cone);
	return NULL;
}

/* Concatenate two integer vectors, i.e., two vectors with denominator
 * (stored in element 0) equal to 1.
 */
static struct isl_vec *vec_concat(struct isl_vec *vec1, struct isl_vec *vec2)
{
	struct isl_vec *vec;

	if (!vec1 || !vec2)
		goto error;
	isl_assert(vec1->ctx, vec1->size > 0, goto error);
	isl_assert(vec2->ctx, vec2->size > 0, goto error);
	isl_assert(vec1->ctx, isl_int_is_one(vec1->el[0]), goto error);
	isl_assert(vec2->ctx, isl_int_is_one(vec2->el[0]), goto error);

	vec = isl_vec_alloc(vec1->ctx, vec1->size + vec2->size - 1);
	if (!vec)
		goto error;

	isl_seq_cpy(vec->el, vec1->el, vec1->size);
	isl_seq_cpy(vec->el + vec1->size, vec2->el + 1, vec2->size - 1);

	isl_vec_free(vec1);
	isl_vec_free(vec2);

	return vec;
error:
	isl_vec_free(vec1);
	isl_vec_free(vec2);
	return NULL;
}

/* Give a basic set "bset" with recession cone "cone", compute and
 * return an integer point in bset, if any.
 *
 * If the recession cone is full-dimensional, then we know that
 * bset contains an infinite number of integer points and it is
 * fairly easy to pick one of them.
 * If the recession cone is not full-dimensional, then we first
 * transform bset such that the bounded directions appear as
 * the first dimensions of the transformed basic set.
 * We do this by using a unimodular transformation that transforms
 * the equalities in the recession cone to equalities on the first
 * dimensions.
 *
 * The transformed set is then projected onto its bounded dimensions.
 * Note that to compute this projection, we can simply drop all constraints
 * involving any of the unbounded dimensions since these constraints
 * cannot be combined to produce a constraint on the bounded dimensions.
 * To see this, assume that there is such a combination of constraints
 * that produces a constraint on the bounded dimensions.  This means
 * that some combination of the unbounded dimensions has both an upper
 * bound and a lower bound in terms of the bounded dimensions, but then
 * this combination would be a bounded direction too and would have been
 * transformed into a bounded dimensions.
 *
 * We then compute a sample value in the bounded dimensions.
 * If no such value can be found, then the original set did not contain
 * any integer points and we are done.
 * Otherwise, we plug in the value we found in the bounded dimensions,
 * project out these bounded dimensions and end up with a set with
 * a full-dimensional recession cone.
 * A sample point in this set is computed by "rounding up" any
 * rational point in the set.
 *
 * The sample points in the bounded and unbounded dimensions are
 * then combined into a single sample point and transformed back
 * to the original space.
 */
__isl_give isl_vec *isl_basic_set_sample_with_cone(
	__isl_take isl_basic_set *bset, __isl_take isl_basic_set *cone)
{
	unsigned total;
	unsigned cone_dim;
	struct isl_mat *M, *U;
	struct isl_vec *sample;
	struct isl_vec *cone_sample;
	struct isl_ctx *ctx;
	struct isl_basic_set *bounded;

	if (!bset || !cone)
		goto error;

	ctx = isl_basic_set_get_ctx(bset);
	total = isl_basic_set_total_dim(cone);
	cone_dim = total - cone->n_eq;

	M = isl_mat_sub_alloc6(ctx, cone->eq, 0, cone->n_eq, 1, total);
	M = isl_mat_left_hermite(M, 0, &U, NULL);
	if (!M)
		goto error;
	isl_mat_free(M);

	U = isl_mat_lin_to_aff(U);
	bset = isl_basic_set_preimage(bset, isl_mat_copy(U));

	bounded = isl_basic_set_copy(bset);
	bounded = isl_basic_set_drop_constraints_involving(bounded,
						   total - cone_dim, cone_dim);
	bounded = isl_basic_set_drop_dims(bounded, total - cone_dim, cone_dim);
	sample = sample_bounded(bounded);
	if (!sample || sample->size == 0) {
		isl_basic_set_free(bset);
		isl_basic_set_free(cone);
		isl_mat_free(U);
		return sample;
	}
	bset = plug_in(bset, isl_vec_copy(sample));
	cone_sample = rational_sample(bset);
	cone_sample = round_up_in_cone(cone_sample, cone, isl_mat_copy(U));
	sample = vec_concat(sample, cone_sample);
	sample = isl_mat_vec_product(U, sample);
	return sample;
error:
	isl_basic_set_free(cone);
	isl_basic_set_free(bset);
	return NULL;
}

static void vec_sum_of_neg(struct isl_vec *v, isl_int *s)
{
	int i;

	isl_int_set_si(*s, 0);

	for (i = 0; i < v->size; ++i)
		if (isl_int_is_neg(v->el[i]))
			isl_int_add(*s, *s, v->el[i]);
}

/* Given a tableau "tab", a tableau "tab_cone" that corresponds
 * to the recession cone and the inverse of a new basis U = inv(B),
 * with the unbounded directions in B last,
 * add constraints to "tab" that ensure any rational value
 * in the unbounded directions can be rounded up to an integer value.
 *
 * The new basis is given by x' = B x, i.e., x = U x'.
 * For any rational value of the last tab->n_unbounded coordinates
 * in the update tableau, the value that is obtained by rounding
 * up this value should be contained in the original tableau.
 * For any constraint "a x + c >= 0", we therefore need to add
 * a constraint "a x + c + s >= 0", with s the sum of all negative
 * entries in the last elements of "a U".
 *
 * Since we are not interested in the first entries of any of the "a U",
 * we first drop the columns of U that correpond to bounded directions.
 */
static int tab_shift_cone(struct isl_tab *tab,
	struct isl_tab *tab_cone, struct isl_mat *U)
{
	int i;
	isl_int v;
	struct isl_basic_set *bset = NULL;

	if (tab && tab->n_unbounded == 0) {
		isl_mat_free(U);
		return 0;
	}
	isl_int_init(v);
	if (!tab || !tab_cone || !U)
		goto error;
	bset = isl_tab_peek_bset(tab_cone);
	U = isl_mat_drop_cols(U, 0, tab->n_var - tab->n_unbounded);
	for (i = 0; i < bset->n_ineq; ++i) {
		int ok;
		struct isl_vec *row = NULL;
		if (isl_tab_is_equality(tab_cone, tab_cone->n_eq + i))
			continue;
		row = isl_vec_alloc(bset->ctx, tab_cone->n_var);
		if (!row)
			goto error;
		isl_seq_cpy(row->el, bset->ineq[i] + 1, tab_cone->n_var);
		row = isl_vec_mat_product(row, isl_mat_copy(U));
		if (!row)
			goto error;
		vec_sum_of_neg(row, &v);
		isl_vec_free(row);
		if (isl_int_is_zero(v))
			continue;
		if (isl_tab_extend_cons(tab, 1) < 0)
			goto error;
		isl_int_add(bset->ineq[i][0], bset->ineq[i][0], v);
		ok = isl_tab_add_ineq(tab, bset->ineq[i]) >= 0;
		isl_int_sub(bset->ineq[i][0], bset->ineq[i][0], v);
		if (!ok)
			goto error;
	}

	isl_mat_free(U);
	isl_int_clear(v);
	return 0;
error:
	isl_mat_free(U);
	isl_int_clear(v);
	return -1;
}

/* Compute and return an initial basis for the possibly
 * unbounded tableau "tab".  "tab_cone" is a tableau
 * for the corresponding recession cone.
 * Additionally, add constraints to "tab" that ensure
 * that any rational value for the unbounded directions
 * can be rounded up to an integer value.
 *
 * If the tableau is bounded, i.e., if the recession cone
 * is zero-dimensional, then we just use inital_basis.
 * Otherwise, we construct a basis whose first directions
 * correspond to equalities, followed by bounded directions,
 * i.e., equalities in the recession cone.
 * The remaining directions are then unbounded.
 */
int isl_tab_set_initial_basis_with_cone(struct isl_tab *tab,
	struct isl_tab *tab_cone)
{
	struct isl_mat *eq;
	struct isl_mat *cone_eq;
	struct isl_mat *U, *Q;

	if (!tab || !tab_cone)
		return -1;

	if (tab_cone->n_col == tab_cone->n_dead) {
		tab->basis = initial_basis(tab);
		return tab->basis ? 0 : -1;
	}

	eq = tab_equalities(tab);
	if (!eq)
		return -1;
	tab->n_zero = eq->n_row;
	cone_eq = tab_equalities(tab_cone);
	eq = isl_mat_concat(eq, cone_eq);
	if (!eq)
		return -1;
	tab->n_unbounded = tab->n_var - (eq->n_row - tab->n_zero);
	eq = isl_mat_left_hermite(eq, 0, &U, &Q);
	if (!eq)
		return -1;
	isl_mat_free(eq);
	tab->basis = isl_mat_lin_to_aff(Q);
	if (tab_shift_cone(tab, tab_cone, U) < 0)
		return -1;
	if (!tab->basis)
		return -1;
	return 0;
}

/* Compute and return a sample point in bset using generalized basis
 * reduction.  We first check if the input set has a non-trivial
 * recession cone.  If so, we perform some extra preprocessing in
 * sample_with_cone.  Otherwise, we directly perform generalized basis
 * reduction.
 */
static struct isl_vec *gbr_sample(struct isl_basic_set *bset)
{
	unsigned dim;
	struct isl_basic_set *cone;

	dim = isl_basic_set_total_dim(bset);

	cone = isl_basic_set_recession_cone(isl_basic_set_copy(bset));
	if (!cone)
		goto error;

	if (cone->n_eq < dim)
		return isl_basic_set_sample_with_cone(bset, cone);

	isl_basic_set_free(cone);
	return sample_bounded(bset);
error:
	isl_basic_set_free(bset);
	return NULL;
}

static struct isl_vec *basic_set_sample(struct isl_basic_set *bset, int bounded)
{
	struct isl_ctx *ctx;
	unsigned dim;
	if (!bset)
		return NULL;

	ctx = bset->ctx;
	if (isl_basic_set_plain_is_empty(bset))
		return empty_sample(bset);

	dim = isl_basic_set_n_dim(bset);
	isl_assert(ctx, isl_basic_set_n_param(bset) == 0, goto error);
	isl_assert(ctx, bset->n_div == 0, goto error);

	if (bset->sample && bset->sample->size == 1 + dim) {
		int contains = isl_basic_set_contains(bset, bset->sample);
		if (contains < 0)
			goto error;
		if (contains) {
			struct isl_vec *sample = isl_vec_copy(bset->sample);
			isl_basic_set_free(bset);
			return sample;
		}
	}
	isl_vec_free(bset->sample);
	bset->sample = NULL;

	if (bset->n_eq > 0)
		return sample_eq(bset, bounded ? isl_basic_set_sample_bounded
					       : isl_basic_set_sample_vec);
	if (dim == 0)
		return zero_sample(bset);
	if (dim == 1)
		return interval_sample(bset);

	return bounded ? sample_bounded(bset) : gbr_sample(bset);
error:
	isl_basic_set_free(bset);
	return NULL;
}

__isl_give isl_vec *isl_basic_set_sample_vec(__isl_take isl_basic_set *bset)
{
	return basic_set_sample(bset, 0);
}

/* Compute an integer sample in "bset", where the caller guarantees
 * that "bset" is bounded.
 */
struct isl_vec *isl_basic_set_sample_bounded(struct isl_basic_set *bset)
{
	return basic_set_sample(bset, 1);
}

__isl_give isl_basic_set *isl_basic_set_from_vec(__isl_take isl_vec *vec)
{
	int i;
	int k;
	struct isl_basic_set *bset = NULL;
	struct isl_ctx *ctx;
	unsigned dim;

	if (!vec)
		return NULL;
	ctx = vec->ctx;
	isl_assert(ctx, vec->size != 0, goto error);

	bset = isl_basic_set_alloc(ctx, 0, vec->size - 1, 0, vec->size - 1, 0);
	if (!bset)
		goto error;
	dim = isl_basic_set_n_dim(bset);
	for (i = dim - 1; i >= 0; --i) {
		k = isl_basic_set_alloc_equality(bset);
		if (k < 0)
			goto error;
		isl_seq_clr(bset->eq[k], 1 + dim);
		isl_int_neg(bset->eq[k][0], vec->el[1 + i]);
		isl_int_set(bset->eq[k][1 + i], vec->el[0]);
	}
	bset->sample = vec;

	return bset;
error:
	isl_basic_set_free(bset);
	isl_vec_free(vec);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_sample(__isl_take isl_basic_map *bmap)
{
	struct isl_basic_set *bset;
	struct isl_vec *sample_vec;

	bset = isl_basic_map_underlying_set(isl_basic_map_copy(bmap));
	sample_vec = isl_basic_set_sample_vec(bset);
	if (!sample_vec)
		goto error;
	if (sample_vec->size == 0) {
		isl_vec_free(sample_vec);
		return isl_basic_map_set_to_empty(bmap);
	}
	isl_vec_free(bmap->sample);
	bmap->sample = isl_vec_copy(sample_vec);
	bset = isl_basic_set_from_vec(sample_vec);
	return isl_basic_map_overlying_set(bset, bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_sample(__isl_take isl_basic_set *bset)
{
	return isl_basic_map_sample(bset);
}

__isl_give isl_basic_map *isl_map_sample(__isl_take isl_map *map)
{
	int i;
	isl_basic_map *sample = NULL;

	if (!map)
		goto error;

	for (i = 0; i < map->n; ++i) {
		sample = isl_basic_map_sample(isl_basic_map_copy(map->p[i]));
		if (!sample)
			goto error;
		if (!ISL_F_ISSET(sample, ISL_BASIC_MAP_EMPTY))
			break;
		isl_basic_map_free(sample);
	}
	if (i == map->n)
		sample = isl_basic_map_empty(isl_map_get_space(map));
	isl_map_free(map);
	return sample;
error:
	isl_map_free(map);
	return NULL;
}

__isl_give isl_basic_set *isl_set_sample(__isl_take isl_set *set)
{
	return bset_from_bmap(isl_map_sample(set_to_map(set)));
}

__isl_give isl_point *isl_basic_set_sample_point(__isl_take isl_basic_set *bset)
{
	isl_vec *vec;
	isl_space *dim;

	dim = isl_basic_set_get_space(bset);
	bset = isl_basic_set_underlying_set(bset);
	vec = isl_basic_set_sample_vec(bset);

	return isl_point_alloc(dim, vec);
}

__isl_give isl_point *isl_set_sample_point(__isl_take isl_set *set)
{
	int i;
	isl_point *pnt;

	if (!set)
		return NULL;

	for (i = 0; i < set->n; ++i) {
		pnt = isl_basic_set_sample_point(isl_basic_set_copy(set->p[i]));
		if (!pnt)
			goto error;
		if (!isl_point_is_void(pnt))
			break;
		isl_point_free(pnt);
	}
	if (i == set->n)
		pnt = isl_point_void(isl_set_get_space(set));

	isl_set_free(set);
	return pnt;
error:
	isl_set_free(set);
	return NULL;
}
