/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_seq.h>
#include <isl/set.h>
#include <isl/lp.h>
#include <isl/map.h>
#include "isl_equalities.h"
#include "isl_sample.h"
#include "isl_tab.h"
#include <isl_mat_private.h>
#include <isl_vec_private.h>

#include <bset_to_bmap.c>
#include <bset_from_bmap.c>
#include <set_to_map.c>
#include <set_from_map.c>

struct isl_basic_map *isl_basic_map_implicit_equalities(
						struct isl_basic_map *bmap)
{
	struct isl_tab *tab;

	if (!bmap)
		return bmap;

	bmap = isl_basic_map_gauss(bmap, NULL);
	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_EMPTY))
		return bmap;
	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_NO_IMPLICIT))
		return bmap;
	if (bmap->n_ineq <= 1)
		return bmap;

	tab = isl_tab_from_basic_map(bmap, 0);
	if (isl_tab_detect_implicit_equalities(tab) < 0)
		goto error;
	bmap = isl_basic_map_update_from_tab(bmap, tab);
	isl_tab_free(tab);
	bmap = isl_basic_map_gauss(bmap, NULL);
	ISL_F_SET(bmap, ISL_BASIC_MAP_NO_IMPLICIT);
	return bmap;
error:
	isl_tab_free(tab);
	isl_basic_map_free(bmap);
	return NULL;
}

struct isl_basic_set *isl_basic_set_implicit_equalities(
						struct isl_basic_set *bset)
{
	return bset_from_bmap(
		isl_basic_map_implicit_equalities(bset_to_bmap(bset)));
}

struct isl_map *isl_map_implicit_equalities(struct isl_map *map)
{
	int i;

	if (!map)
		return map;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_implicit_equalities(map->p[i]);
		if (!map->p[i])
			goto error;
	}

	return map;
error:
	isl_map_free(map);
	return NULL;
}

/* Make eq[row][col] of both bmaps equal so we can add the row
 * add the column to the common matrix.
 * Note that because of the echelon form, the columns of row row
 * after column col are zero.
 */
static void set_common_multiple(
	struct isl_basic_set *bset1, struct isl_basic_set *bset2,
	unsigned row, unsigned col)
{
	isl_int m, c;

	if (isl_int_eq(bset1->eq[row][col], bset2->eq[row][col]))
		return;

	isl_int_init(c);
	isl_int_init(m);
	isl_int_lcm(m, bset1->eq[row][col], bset2->eq[row][col]);
	isl_int_divexact(c, m, bset1->eq[row][col]);
	isl_seq_scale(bset1->eq[row], bset1->eq[row], c, col+1);
	isl_int_divexact(c, m, bset2->eq[row][col]);
	isl_seq_scale(bset2->eq[row], bset2->eq[row], c, col+1);
	isl_int_clear(c);
	isl_int_clear(m);
}

/* Delete a given equality, moving all the following equalities one up.
 */
static void delete_row(struct isl_basic_set *bset, unsigned row)
{
	isl_int *t;
	int r;

	t = bset->eq[row];
	bset->n_eq--;
	for (r = row; r < bset->n_eq; ++r)
		bset->eq[r] = bset->eq[r+1];
	bset->eq[bset->n_eq] = t;
}

/* Make first row entries in column col of bset1 identical to
 * those of bset2, using the fact that entry bset1->eq[row][col]=a
 * is non-zero.  Initially, these elements of bset1 are all zero.
 * For each row i < row, we set
 *		A[i] = a * A[i] + B[i][col] * A[row]
 *		B[i] = a * B[i]
 * so that
 *		A[i][col] = B[i][col] = a * old(B[i][col])
 */
static void construct_column(
	struct isl_basic_set *bset1, struct isl_basic_set *bset2,
	unsigned row, unsigned col)
{
	int r;
	isl_int a;
	isl_int b;
	unsigned total;

	isl_int_init(a);
	isl_int_init(b);
	total = 1 + isl_basic_set_n_dim(bset1);
	for (r = 0; r < row; ++r) {
		if (isl_int_is_zero(bset2->eq[r][col]))
			continue;
		isl_int_gcd(b, bset2->eq[r][col], bset1->eq[row][col]);
		isl_int_divexact(a, bset1->eq[row][col], b);
		isl_int_divexact(b, bset2->eq[r][col], b);
		isl_seq_combine(bset1->eq[r], a, bset1->eq[r],
					      b, bset1->eq[row], total);
		isl_seq_scale(bset2->eq[r], bset2->eq[r], a, total);
	}
	isl_int_clear(a);
	isl_int_clear(b);
	delete_row(bset1, row);
}

/* Make first row entries in column col of bset1 identical to
 * those of bset2, using only these entries of the two matrices.
 * Let t be the last row with different entries.
 * For each row i < t, we set
 *	A[i] = (A[t][col]-B[t][col]) * A[i] + (B[i][col]-A[i][col) * A[t]
 *	B[i] = (A[t][col]-B[t][col]) * B[i] + (B[i][col]-A[i][col) * B[t]
 * so that
 *	A[i][col] = B[i][col] = old(A[t][col]*B[i][col]-A[i][col]*B[t][col])
 */
static int transform_column(
	struct isl_basic_set *bset1, struct isl_basic_set *bset2,
	unsigned row, unsigned col)
{
	int i, t;
	isl_int a, b, g;
	unsigned total;

	for (t = row-1; t >= 0; --t)
		if (isl_int_ne(bset1->eq[t][col], bset2->eq[t][col]))
			break;
	if (t < 0)
		return 0;

	total = 1 + isl_basic_set_n_dim(bset1);
	isl_int_init(a);
	isl_int_init(b);
	isl_int_init(g);
	isl_int_sub(b, bset1->eq[t][col], bset2->eq[t][col]);
	for (i = 0; i < t; ++i) {
		isl_int_sub(a, bset2->eq[i][col], bset1->eq[i][col]);
		isl_int_gcd(g, a, b);
		isl_int_divexact(a, a, g);
		isl_int_divexact(g, b, g);
		isl_seq_combine(bset1->eq[i], g, bset1->eq[i], a, bset1->eq[t],
				total);
		isl_seq_combine(bset2->eq[i], g, bset2->eq[i], a, bset2->eq[t],
				total);
	}
	isl_int_clear(a);
	isl_int_clear(b);
	isl_int_clear(g);
	delete_row(bset1, t);
	delete_row(bset2, t);
	return 1;
}

/* The implementation is based on Section 5.2 of Michael Karr,
 * "Affine Relationships Among Variables of a Program",
 * except that the echelon form we use starts from the last column
 * and that we are dealing with integer coefficients.
 */
static struct isl_basic_set *affine_hull(
	struct isl_basic_set *bset1, struct isl_basic_set *bset2)
{
	unsigned total;
	int col;
	int row;

	if (!bset1 || !bset2)
		goto error;

	total = 1 + isl_basic_set_n_dim(bset1);

	row = 0;
	for (col = total-1; col >= 0; --col) {
		int is_zero1 = row >= bset1->n_eq ||
			isl_int_is_zero(bset1->eq[row][col]);
		int is_zero2 = row >= bset2->n_eq ||
			isl_int_is_zero(bset2->eq[row][col]);
		if (!is_zero1 && !is_zero2) {
			set_common_multiple(bset1, bset2, row, col);
			++row;
		} else if (!is_zero1 && is_zero2) {
			construct_column(bset1, bset2, row, col);
		} else if (is_zero1 && !is_zero2) {
			construct_column(bset2, bset1, row, col);
		} else {
			if (transform_column(bset1, bset2, row, col))
				--row;
		}
	}
	isl_assert(bset1->ctx, row == bset1->n_eq, goto error);
	isl_basic_set_free(bset2);
	bset1 = isl_basic_set_normalize_constraints(bset1);
	return bset1;
error:
	isl_basic_set_free(bset1);
	isl_basic_set_free(bset2);
	return NULL;
}

/* Find an integer point in the set represented by "tab"
 * that lies outside of the equality "eq" e(x) = 0.
 * If "up" is true, look for a point satisfying e(x) - 1 >= 0.
 * Otherwise, look for a point satisfying -e(x) - 1 >= 0 (i.e., e(x) <= -1).
 * The point, if found, is returned.
 * If no point can be found, a zero-length vector is returned.
 *
 * Before solving an ILP problem, we first check if simply
 * adding the normal of the constraint to one of the known
 * integer points in the basic set represented by "tab"
 * yields another point inside the basic set.
 *
 * The caller of this function ensures that the tableau is bounded or
 * that tab->basis and tab->n_unbounded have been set appropriately.
 */
static struct isl_vec *outside_point(struct isl_tab *tab, isl_int *eq, int up)
{
	struct isl_ctx *ctx;
	struct isl_vec *sample = NULL;
	struct isl_tab_undo *snap;
	unsigned dim;

	if (!tab)
		return NULL;
	ctx = tab->mat->ctx;

	dim = tab->n_var;
	sample = isl_vec_alloc(ctx, 1 + dim);
	if (!sample)
		return NULL;
	isl_int_set_si(sample->el[0], 1);
	isl_seq_combine(sample->el + 1,
		ctx->one, tab->bmap->sample->el + 1,
		up ? ctx->one : ctx->negone, eq + 1, dim);
	if (isl_basic_map_contains(tab->bmap, sample))
		return sample;
	isl_vec_free(sample);
	sample = NULL;

	snap = isl_tab_snap(tab);

	if (!up)
		isl_seq_neg(eq, eq, 1 + dim);
	isl_int_sub_ui(eq[0], eq[0], 1);

	if (isl_tab_extend_cons(tab, 1) < 0)
		goto error;
	if (isl_tab_add_ineq(tab, eq) < 0)
		goto error;

	sample = isl_tab_sample(tab);

	isl_int_add_ui(eq[0], eq[0], 1);
	if (!up)
		isl_seq_neg(eq, eq, 1 + dim);

	if (sample && isl_tab_rollback(tab, snap) < 0)
		goto error;

	return sample;
error:
	isl_vec_free(sample);
	return NULL;
}

struct isl_basic_set *isl_basic_set_recession_cone(struct isl_basic_set *bset)
{
	int i;

	bset = isl_basic_set_cow(bset);
	if (!bset)
		return NULL;
	isl_assert(bset->ctx, bset->n_div == 0, goto error);

	for (i = 0; i < bset->n_eq; ++i)
		isl_int_set_si(bset->eq[i][0], 0);

	for (i = 0; i < bset->n_ineq; ++i)
		isl_int_set_si(bset->ineq[i][0], 0);

	ISL_F_CLR(bset, ISL_BASIC_SET_NO_IMPLICIT);
	return isl_basic_set_implicit_equalities(bset);
error:
	isl_basic_set_free(bset);
	return NULL;
}

__isl_give isl_set *isl_set_recession_cone(__isl_take isl_set *set)
{
	int i;

	if (!set)
		return NULL;
	if (set->n == 0)
		return set;

	set = isl_set_remove_divs(set);
	set = isl_set_cow(set);
	if (!set)
		return NULL;

	for (i = 0; i < set->n; ++i) {
		set->p[i] = isl_basic_set_recession_cone(set->p[i]);
		if (!set->p[i])
			goto error;
	}

	return set;
error:
	isl_set_free(set);
	return NULL;
}

/* Move "sample" to a point that is one up (or down) from the original
 * point in dimension "pos".
 */
static void adjacent_point(__isl_keep isl_vec *sample, int pos, int up)
{
	if (up)
		isl_int_add_ui(sample->el[1 + pos], sample->el[1 + pos], 1);
	else
		isl_int_sub_ui(sample->el[1 + pos], sample->el[1 + pos], 1);
}

/* Check if any points that are adjacent to "sample" also belong to "bset".
 * If so, add them to "hull" and return the updated hull.
 *
 * Before checking whether and adjacent point belongs to "bset", we first
 * check whether it already belongs to "hull" as this test is typically
 * much cheaper.
 */
static __isl_give isl_basic_set *add_adjacent_points(
	__isl_take isl_basic_set *hull, __isl_take isl_vec *sample,
	__isl_keep isl_basic_set *bset)
{
	int i, up;
	int dim;

	if (!sample)
		goto error;

	dim = isl_basic_set_dim(hull, isl_dim_set);

	for (i = 0; i < dim; ++i) {
		for (up = 0; up <= 1; ++up) {
			int contains;
			isl_basic_set *point;

			adjacent_point(sample, i, up);
			contains = isl_basic_set_contains(hull, sample);
			if (contains < 0)
				goto error;
			if (contains) {
				adjacent_point(sample, i, !up);
				continue;
			}
			contains = isl_basic_set_contains(bset, sample);
			if (contains < 0)
				goto error;
			if (contains) {
				point = isl_basic_set_from_vec(
							isl_vec_copy(sample));
				hull = affine_hull(hull, point);
			}
			adjacent_point(sample, i, !up);
			if (contains)
				break;
		}
	}

	isl_vec_free(sample);

	return hull;
error:
	isl_vec_free(sample);
	isl_basic_set_free(hull);
	return NULL;
}

/* Extend an initial (under-)approximation of the affine hull of basic
 * set represented by the tableau "tab"
 * by looking for points that do not satisfy one of the equalities
 * in the current approximation and adding them to that approximation
 * until no such points can be found any more.
 *
 * The caller of this function ensures that "tab" is bounded or
 * that tab->basis and tab->n_unbounded have been set appropriately.
 *
 * "bset" may be either NULL or the basic set represented by "tab".
 * If "bset" is not NULL, we check for any point we find if any
 * of its adjacent points also belong to "bset".
 */
static __isl_give isl_basic_set *extend_affine_hull(struct isl_tab *tab,
	__isl_take isl_basic_set *hull, __isl_keep isl_basic_set *bset)
{
	int i, j;
	unsigned dim;

	if (!tab || !hull)
		goto error;

	dim = tab->n_var;

	if (isl_tab_extend_cons(tab, 2 * dim + 1) < 0)
		goto error;

	for (i = 0; i < dim; ++i) {
		struct isl_vec *sample;
		struct isl_basic_set *point;
		for (j = 0; j < hull->n_eq; ++j) {
			sample = outside_point(tab, hull->eq[j], 1);
			if (!sample)
				goto error;
			if (sample->size > 0)
				break;
			isl_vec_free(sample);
			sample = outside_point(tab, hull->eq[j], 0);
			if (!sample)
				goto error;
			if (sample->size > 0)
				break;
			isl_vec_free(sample);

			if (isl_tab_add_eq(tab, hull->eq[j]) < 0)
				goto error;
		}
		if (j == hull->n_eq)
			break;
		if (tab->samples &&
		    isl_tab_add_sample(tab, isl_vec_copy(sample)) < 0)
			hull = isl_basic_set_free(hull);
		if (bset)
			hull = add_adjacent_points(hull, isl_vec_copy(sample),
						    bset);
		point = isl_basic_set_from_vec(sample);
		hull = affine_hull(hull, point);
		if (!hull)
			return NULL;
	}

	return hull;
error:
	isl_basic_set_free(hull);
	return NULL;
}

/* Drop all constraints in bmap that involve any of the dimensions
 * first to first+n-1.
 */
static __isl_give isl_basic_map *isl_basic_map_drop_constraints_involving(
	__isl_take isl_basic_map *bmap, unsigned first, unsigned n)
{
	int i;

	if (n == 0)
		return bmap;

	bmap = isl_basic_map_cow(bmap);

	if (!bmap)
		return NULL;

	for (i = bmap->n_eq - 1; i >= 0; --i) {
		if (isl_seq_first_non_zero(bmap->eq[i] + 1 + first, n) == -1)
			continue;
		isl_basic_map_drop_equality(bmap, i);
	}

	for (i = bmap->n_ineq - 1; i >= 0; --i) {
		if (isl_seq_first_non_zero(bmap->ineq[i] + 1 + first, n) == -1)
			continue;
		isl_basic_map_drop_inequality(bmap, i);
	}

	bmap = isl_basic_map_add_known_div_constraints(bmap);
	return bmap;
}

/* Drop all constraints in bset that involve any of the dimensions
 * first to first+n-1.
 */
__isl_give isl_basic_set *isl_basic_set_drop_constraints_involving(
	__isl_take isl_basic_set *bset, unsigned first, unsigned n)
{
	return isl_basic_map_drop_constraints_involving(bset, first, n);
}

/* Drop all constraints in bmap that do not involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_basic_map *isl_basic_map_drop_constraints_not_involving_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	unsigned dim;

	if (n == 0) {
		isl_space *space = isl_basic_map_get_space(bmap);
		isl_basic_map_free(bmap);
		return isl_basic_map_universe(space);
	}
	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;

	dim = isl_basic_map_dim(bmap, type);
	if (first + n > dim || first + n < first)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"index out of bounds", return isl_basic_map_free(bmap));

	first += isl_basic_map_offset(bmap, type) - 1;

	for (i = bmap->n_eq - 1; i >= 0; --i) {
		if (isl_seq_first_non_zero(bmap->eq[i] + 1 + first, n) != -1)
			continue;
		isl_basic_map_drop_equality(bmap, i);
	}

	for (i = bmap->n_ineq - 1; i >= 0; --i) {
		if (isl_seq_first_non_zero(bmap->ineq[i] + 1 + first, n) != -1)
			continue;
		isl_basic_map_drop_inequality(bmap, i);
	}

	bmap = isl_basic_map_add_known_div_constraints(bmap);
	return bmap;
}

/* Drop all constraints in bset that do not involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_basic_set *isl_basic_set_drop_constraints_not_involving_dims(
	__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_basic_map_drop_constraints_not_involving_dims(bset,
							    type, first, n);
}

/* Drop all constraints in bmap that involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_basic_map *isl_basic_map_drop_constraints_involving_dims(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	unsigned dim;

	if (!bmap)
		return NULL;
	if (n == 0)
		return bmap;

	dim = isl_basic_map_dim(bmap, type);
	if (first + n > dim || first + n < first)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"index out of bounds", return isl_basic_map_free(bmap));

	bmap = isl_basic_map_remove_divs_involving_dims(bmap, type, first, n);
	first += isl_basic_map_offset(bmap, type) - 1;
	return isl_basic_map_drop_constraints_involving(bmap, first, n);
}

/* Drop all constraints in bset that involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_basic_set *isl_basic_set_drop_constraints_involving_dims(
	__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_basic_map_drop_constraints_involving_dims(bset,
							    type, first, n);
}

/* Drop constraints from "map" by applying "drop" to each basic map.
 */
static __isl_give isl_map *drop_constraints(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n,
	__isl_give isl_basic_map *(*drop)(__isl_take isl_basic_map *bmap,
		enum isl_dim_type type, unsigned first, unsigned n))
{
	int i;
	unsigned dim;

	if (!map)
		return NULL;

	dim = isl_map_dim(map, type);
	if (first + n > dim || first + n < first)
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"index out of bounds", return isl_map_free(map));

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = drop(map->p[i], type, first, n);
		if (!map->p[i])
			return isl_map_free(map);
	}

	if (map->n > 1)
		ISL_F_CLR(map, ISL_MAP_DISJOINT);

	return map;
}

/* Drop all constraints in map that involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_map *isl_map_drop_constraints_involving_dims(
	__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	if (n == 0)
		return map;
	return drop_constraints(map, type, first, n,
				&isl_basic_map_drop_constraints_involving_dims);
}

/* Drop all constraints in "map" that do not involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_map *isl_map_drop_constraints_not_involving_dims(
	__isl_take isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	if (n == 0) {
		isl_space *space = isl_map_get_space(map);
		isl_map_free(map);
		return isl_map_universe(space);
	}
	return drop_constraints(map, type, first, n,
			    &isl_basic_map_drop_constraints_not_involving_dims);
}

/* Drop all constraints in set that involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_set *isl_set_drop_constraints_involving_dims(
	__isl_take isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_map_drop_constraints_involving_dims(set, type, first, n);
}

/* Drop all constraints in "set" that do not involve any of the dimensions
 * first to first + n - 1 of the given type.
 */
__isl_give isl_set *isl_set_drop_constraints_not_involving_dims(
	__isl_take isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_map_drop_constraints_not_involving_dims(set, type, first, n);
}

/* Construct an initial underapproximation of the hull of "bset"
 * from "sample" and any of its adjacent points that also belong to "bset".
 */
static __isl_give isl_basic_set *initialize_hull(__isl_keep isl_basic_set *bset,
	__isl_take isl_vec *sample)
{
	isl_basic_set *hull;

	hull = isl_basic_set_from_vec(isl_vec_copy(sample));
	hull = add_adjacent_points(hull, sample, bset);

	return hull;
}

/* Look for all equalities satisfied by the integer points in bset,
 * which is assumed to be bounded.
 *
 * The equalities are obtained by successively looking for
 * a point that is affinely independent of the points found so far.
 * In particular, for each equality satisfied by the points so far,
 * we check if there is any point on a hyperplane parallel to the
 * corresponding hyperplane shifted by at least one (in either direction).
 */
static struct isl_basic_set *uset_affine_hull_bounded(struct isl_basic_set *bset)
{
	struct isl_vec *sample = NULL;
	struct isl_basic_set *hull;
	struct isl_tab *tab = NULL;
	unsigned dim;

	if (isl_basic_set_plain_is_empty(bset))
		return bset;

	dim = isl_basic_set_n_dim(bset);

	if (bset->sample && bset->sample->size == 1 + dim) {
		int contains = isl_basic_set_contains(bset, bset->sample);
		if (contains < 0)
			goto error;
		if (contains) {
			if (dim == 0)
				return bset;
			sample = isl_vec_copy(bset->sample);
		} else {
			isl_vec_free(bset->sample);
			bset->sample = NULL;
		}
	}

	tab = isl_tab_from_basic_set(bset, 1);
	if (!tab)
		goto error;
	if (tab->empty) {
		isl_tab_free(tab);
		isl_vec_free(sample);
		return isl_basic_set_set_to_empty(bset);
	}

	if (!sample) {
		struct isl_tab_undo *snap;
		snap = isl_tab_snap(tab);
		sample = isl_tab_sample(tab);
		if (isl_tab_rollback(tab, snap) < 0)
			goto error;
		isl_vec_free(tab->bmap->sample);
		tab->bmap->sample = isl_vec_copy(sample);
	}

	if (!sample)
		goto error;
	if (sample->size == 0) {
		isl_tab_free(tab);
		isl_vec_free(sample);
		return isl_basic_set_set_to_empty(bset);
	}

	hull = initialize_hull(bset, sample);

	hull = extend_affine_hull(tab, hull, bset);
	isl_basic_set_free(bset);
	isl_tab_free(tab);

	return hull;
error:
	isl_vec_free(sample);
	isl_tab_free(tab);
	isl_basic_set_free(bset);
	return NULL;
}

/* Given an unbounded tableau and an integer point satisfying the tableau,
 * construct an initial affine hull containing the recession cone
 * shifted to the given point.
 *
 * The unbounded directions are taken from the last rows of the basis,
 * which is assumed to have been initialized appropriately.
 */
static __isl_give isl_basic_set *initial_hull(struct isl_tab *tab,
	__isl_take isl_vec *vec)
{
	int i;
	int k;
	struct isl_basic_set *bset = NULL;
	struct isl_ctx *ctx;
	unsigned dim;

	if (!vec || !tab)
		return NULL;
	ctx = vec->ctx;
	isl_assert(ctx, vec->size != 0, goto error);

	bset = isl_basic_set_alloc(ctx, 0, vec->size - 1, 0, vec->size - 1, 0);
	if (!bset)
		goto error;
	dim = isl_basic_set_n_dim(bset) - tab->n_unbounded;
	for (i = 0; i < dim; ++i) {
		k = isl_basic_set_alloc_equality(bset);
		if (k < 0)
			goto error;
		isl_seq_cpy(bset->eq[k] + 1, tab->basis->row[1 + i] + 1,
			    vec->size - 1);
		isl_seq_inner_product(bset->eq[k] + 1, vec->el +1,
				      vec->size - 1, &bset->eq[k][0]);
		isl_int_neg(bset->eq[k][0], bset->eq[k][0]);
	}
	bset->sample = vec;
	bset = isl_basic_set_gauss(bset, NULL);

	return bset;
error:
	isl_basic_set_free(bset);
	isl_vec_free(vec);
	return NULL;
}

/* Given a tableau of a set and a tableau of the corresponding
 * recession cone, detect and add all equalities to the tableau.
 * If the tableau is bounded, then we can simply keep the
 * tableau in its state after the return from extend_affine_hull.
 * However, if the tableau is unbounded, then
 * isl_tab_set_initial_basis_with_cone will add some additional
 * constraints to the tableau that have to be removed again.
 * In this case, we therefore rollback to the state before
 * any constraints were added and then add the equalities back in.
 */
struct isl_tab *isl_tab_detect_equalities(struct isl_tab *tab,
	struct isl_tab *tab_cone)
{
	int j;
	struct isl_vec *sample;
	struct isl_basic_set *hull = NULL;
	struct isl_tab_undo *snap;

	if (!tab || !tab_cone)
		goto error;

	snap = isl_tab_snap(tab);

	isl_mat_free(tab->basis);
	tab->basis = NULL;

	isl_assert(tab->mat->ctx, tab->bmap, goto error);
	isl_assert(tab->mat->ctx, tab->samples, goto error);
	isl_assert(tab->mat->ctx, tab->samples->n_col == 1 + tab->n_var, goto error);
	isl_assert(tab->mat->ctx, tab->n_sample > tab->n_outside, goto error);

	if (isl_tab_set_initial_basis_with_cone(tab, tab_cone) < 0)
		goto error;

	sample = isl_vec_alloc(tab->mat->ctx, 1 + tab->n_var);
	if (!sample)
		goto error;

	isl_seq_cpy(sample->el, tab->samples->row[tab->n_outside], sample->size);

	isl_vec_free(tab->bmap->sample);
	tab->bmap->sample = isl_vec_copy(sample);

	if (tab->n_unbounded == 0)
		hull = isl_basic_set_from_vec(isl_vec_copy(sample));
	else
		hull = initial_hull(tab, isl_vec_copy(sample));

	for (j = tab->n_outside + 1; j < tab->n_sample; ++j) {
		isl_seq_cpy(sample->el, tab->samples->row[j], sample->size);
		hull = affine_hull(hull,
				isl_basic_set_from_vec(isl_vec_copy(sample)));
	}

	isl_vec_free(sample);

	hull = extend_affine_hull(tab, hull, NULL);
	if (!hull)
		goto error;

	if (tab->n_unbounded == 0) {
		isl_basic_set_free(hull);
		return tab;
	}

	if (isl_tab_rollback(tab, snap) < 0)
		goto error;

	if (hull->n_eq > tab->n_zero) {
		for (j = 0; j < hull->n_eq; ++j) {
			isl_seq_normalize(tab->mat->ctx, hull->eq[j], 1 + tab->n_var);
			if (isl_tab_add_eq(tab, hull->eq[j]) < 0)
				goto error;
		}
	}

	isl_basic_set_free(hull);

	return tab;
error:
	isl_basic_set_free(hull);
	isl_tab_free(tab);
	return NULL;
}

/* Compute the affine hull of "bset", where "cone" is the recession cone
 * of "bset".
 *
 * We first compute a unimodular transformation that puts the unbounded
 * directions in the last dimensions.  In particular, we take a transformation
 * that maps all equalities to equalities (in HNF) on the first dimensions.
 * Let x be the original dimensions and y the transformed, with y_1 bounded
 * and y_2 unbounded.
 *
 *	       [ y_1 ]			[ y_1 ]   [ Q_1 ]
 *	x = U  [ y_2 ]			[ y_2 ] = [ Q_2 ] x
 *
 * Let's call the input basic set S.  We compute S' = preimage(S, U)
 * and drop the final dimensions including any constraints involving them.
 * This results in set S''.
 * Then we compute the affine hull A'' of S''.
 * Let F y_1 >= g be the constraint system of A''.  In the transformed
 * space the y_2 are unbounded, so we can add them back without any constraints,
 * resulting in
 *
 *		        [ y_1 ]
 *		[ F 0 ] [ y_2 ] >= g
 * or
 *		        [ Q_1 ]
 *		[ F 0 ] [ Q_2 ] x >= g
 * or
 *		F Q_1 x >= g
 *
 * The affine hull in the original space is then obtained as
 * A = preimage(A'', Q_1).
 */
static struct isl_basic_set *affine_hull_with_cone(struct isl_basic_set *bset,
	struct isl_basic_set *cone)
{
	unsigned total;
	unsigned cone_dim;
	struct isl_basic_set *hull;
	struct isl_mat *M, *U, *Q;

	if (!bset || !cone)
		goto error;

	total = isl_basic_set_total_dim(cone);
	cone_dim = total - cone->n_eq;

	M = isl_mat_sub_alloc6(bset->ctx, cone->eq, 0, cone->n_eq, 1, total);
	M = isl_mat_left_hermite(M, 0, &U, &Q);
	if (!M)
		goto error;
	isl_mat_free(M);

	U = isl_mat_lin_to_aff(U);
	bset = isl_basic_set_preimage(bset, isl_mat_copy(U));

	bset = isl_basic_set_drop_constraints_involving(bset, total - cone_dim,
							cone_dim);
	bset = isl_basic_set_drop_dims(bset, total - cone_dim, cone_dim);

	Q = isl_mat_lin_to_aff(Q);
	Q = isl_mat_drop_rows(Q, 1 + total - cone_dim, cone_dim);

	if (bset && bset->sample && bset->sample->size == 1 + total)
		bset->sample = isl_mat_vec_product(isl_mat_copy(Q), bset->sample);

	hull = uset_affine_hull_bounded(bset);

	if (!hull) {
		isl_mat_free(Q);
		isl_mat_free(U);
	} else {
		struct isl_vec *sample = isl_vec_copy(hull->sample);
		U = isl_mat_drop_cols(U, 1 + total - cone_dim, cone_dim);
		if (sample && sample->size > 0)
			sample = isl_mat_vec_product(U, sample);
		else
			isl_mat_free(U);
		hull = isl_basic_set_preimage(hull, Q);
		if (hull) {
			isl_vec_free(hull->sample);
			hull->sample = sample;
		} else
			isl_vec_free(sample);
	}

	isl_basic_set_free(cone);

	return hull;
error:
	isl_basic_set_free(bset);
	isl_basic_set_free(cone);
	return NULL;
}

/* Look for all equalities satisfied by the integer points in bset,
 * which is assumed not to have any explicit equalities.
 *
 * The equalities are obtained by successively looking for
 * a point that is affinely independent of the points found so far.
 * In particular, for each equality satisfied by the points so far,
 * we check if there is any point on a hyperplane parallel to the
 * corresponding hyperplane shifted by at least one (in either direction).
 *
 * Before looking for any outside points, we first compute the recession
 * cone.  The directions of this recession cone will always be part
 * of the affine hull, so there is no need for looking for any points
 * in these directions.
 * In particular, if the recession cone is full-dimensional, then
 * the affine hull is simply the whole universe.
 */
static struct isl_basic_set *uset_affine_hull(struct isl_basic_set *bset)
{
	struct isl_basic_set *cone;

	if (isl_basic_set_plain_is_empty(bset))
		return bset;

	cone = isl_basic_set_recession_cone(isl_basic_set_copy(bset));
	if (!cone)
		goto error;
	if (cone->n_eq == 0) {
		isl_space *space;
		space = isl_basic_set_get_space(bset);
		isl_basic_set_free(cone);
		isl_basic_set_free(bset);
		return isl_basic_set_universe(space);
	}

	if (cone->n_eq < isl_basic_set_total_dim(cone))
		return affine_hull_with_cone(bset, cone);

	isl_basic_set_free(cone);
	return uset_affine_hull_bounded(bset);
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Look for all equalities satisfied by the integer points in bmap
 * that are independent of the equalities already explicitly available
 * in bmap.
 *
 * We first remove all equalities already explicitly available,
 * then look for additional equalities in the reduced space
 * and then transform the result to the original space.
 * The original equalities are _not_ added to this set.  This is
 * the responsibility of the calling function.
 * The resulting basic set has all meaning about the dimensions removed.
 * In particular, dimensions that correspond to existential variables
 * in bmap and that are found to be fixed are not removed.
 */
static struct isl_basic_set *equalities_in_underlying_set(
						struct isl_basic_map *bmap)
{
	struct isl_mat *T1 = NULL;
	struct isl_mat *T2 = NULL;
	struct isl_basic_set *bset = NULL;
	struct isl_basic_set *hull = NULL;

	bset = isl_basic_map_underlying_set(bmap);
	if (!bset)
		return NULL;
	if (bset->n_eq)
		bset = isl_basic_set_remove_equalities(bset, &T1, &T2);
	if (!bset)
		goto error;

	hull = uset_affine_hull(bset);
	if (!T2)
		return hull;

	if (!hull) {
		isl_mat_free(T1);
		isl_mat_free(T2);
	} else {
		struct isl_vec *sample = isl_vec_copy(hull->sample);
		if (sample && sample->size > 0)
			sample = isl_mat_vec_product(T1, sample);
		else
			isl_mat_free(T1);
		hull = isl_basic_set_preimage(hull, T2);
		if (hull) {
			isl_vec_free(hull->sample);
			hull->sample = sample;
		} else
			isl_vec_free(sample);
	}

	return hull;
error:
	isl_mat_free(T1);
	isl_mat_free(T2);
	isl_basic_set_free(bset);
	isl_basic_set_free(hull);
	return NULL;
}

/* Detect and make explicit all equalities satisfied by the (integer)
 * points in bmap.
 */
struct isl_basic_map *isl_basic_map_detect_equalities(
						struct isl_basic_map *bmap)
{
	int i, j;
	struct isl_basic_set *hull = NULL;

	if (!bmap)
		return NULL;
	if (bmap->n_ineq == 0)
		return bmap;
	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_EMPTY))
		return bmap;
	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_ALL_EQUALITIES))
		return bmap;
	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL))
		return isl_basic_map_implicit_equalities(bmap);

	hull = equalities_in_underlying_set(isl_basic_map_copy(bmap));
	if (!hull)
		goto error;
	if (ISL_F_ISSET(hull, ISL_BASIC_SET_EMPTY)) {
		isl_basic_set_free(hull);
		return isl_basic_map_set_to_empty(bmap);
	}
	bmap = isl_basic_map_extend_space(bmap, isl_space_copy(bmap->dim), 0,
					hull->n_eq, 0);
	for (i = 0; i < hull->n_eq; ++i) {
		j = isl_basic_map_alloc_equality(bmap);
		if (j < 0)
			goto error;
		isl_seq_cpy(bmap->eq[j], hull->eq[i],
				1 + isl_basic_set_total_dim(hull));
	}
	isl_vec_free(bmap->sample);
	bmap->sample = isl_vec_copy(hull->sample);
	isl_basic_set_free(hull);
	ISL_F_SET(bmap, ISL_BASIC_MAP_NO_IMPLICIT | ISL_BASIC_MAP_ALL_EQUALITIES);
	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_set_free(hull);
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_detect_equalities(
						__isl_take isl_basic_set *bset)
{
	return bset_from_bmap(
		isl_basic_map_detect_equalities(bset_to_bmap(bset)));
}

__isl_give isl_map *isl_map_detect_equalities(__isl_take isl_map *map)
{
	return isl_map_inline_foreach_basic_map(map,
					    &isl_basic_map_detect_equalities);
}

__isl_give isl_set *isl_set_detect_equalities(__isl_take isl_set *set)
{
	return set_from_map(isl_map_detect_equalities(set_to_map(set)));
}

/* Return the superset of "bmap" described by the equalities
 * satisfied by "bmap" that are already known.
 */
__isl_give isl_basic_map *isl_basic_map_plain_affine_hull(
	__isl_take isl_basic_map *bmap)
{
	bmap = isl_basic_map_cow(bmap);
	if (bmap)
		isl_basic_map_free_inequality(bmap, bmap->n_ineq);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
}

/* Return the superset of "bset" described by the equalities
 * satisfied by "bset" that are already known.
 */
__isl_give isl_basic_set *isl_basic_set_plain_affine_hull(
	__isl_take isl_basic_set *bset)
{
	return isl_basic_map_plain_affine_hull(bset);
}

/* After computing the rational affine hull (by detecting the implicit
 * equalities), we compute the additional equalities satisfied by
 * the integer points (if any) and add the original equalities back in.
 */
struct isl_basic_map *isl_basic_map_affine_hull(struct isl_basic_map *bmap)
{
	bmap = isl_basic_map_detect_equalities(bmap);
	bmap = isl_basic_map_plain_affine_hull(bmap);
	return bmap;
}

struct isl_basic_set *isl_basic_set_affine_hull(struct isl_basic_set *bset)
{
	return bset_from_bmap(isl_basic_map_affine_hull(bset_to_bmap(bset)));
}

/* Given a rational affine matrix "M", add stride constraints to "bmap"
 * that ensure that
 *
 *		M(x)
 *
 * is an integer vector.  The variables x include all the variables
 * of "bmap" except the unknown divs.
 *
 * If d is the common denominator of M, then we need to impose that
 *
 *		d M(x) = 0 	mod d
 *
 * or
 *
 *		exists alpha : d M(x) = d alpha
 *
 * This function is similar to add_strides in isl_morph.c
 */
static __isl_give isl_basic_map *add_strides(__isl_take isl_basic_map *bmap,
	__isl_keep isl_mat *M, int n_known)
{
	int i, div, k;
	isl_int gcd;

	if (isl_int_is_one(M->row[0][0]))
		return bmap;

	bmap = isl_basic_map_extend_space(bmap, isl_space_copy(bmap->dim),
					M->n_row - 1, M->n_row - 1, 0);

	isl_int_init(gcd);
	for (i = 1; i < M->n_row; ++i) {
		isl_seq_gcd(M->row[i], M->n_col, &gcd);
		if (isl_int_is_divisible_by(gcd, M->row[0][0]))
			continue;
		div = isl_basic_map_alloc_div(bmap);
		if (div < 0)
			goto error;
		isl_int_set_si(bmap->div[div][0], 0);
		k = isl_basic_map_alloc_equality(bmap);
		if (k < 0)
			goto error;
		isl_seq_cpy(bmap->eq[k], M->row[i], M->n_col);
		isl_seq_clr(bmap->eq[k] + M->n_col, bmap->n_div - n_known);
		isl_int_set(bmap->eq[k][M->n_col - n_known + div],
			    M->row[0][0]);
	}
	isl_int_clear(gcd);

	return bmap;
error:
	isl_int_clear(gcd);
	isl_basic_map_free(bmap);
	return NULL;
}

/* If there are any equalities that involve (multiple) unknown divs,
 * then extract the stride information encoded by those equalities
 * and make it explicitly available in "bmap".
 *
 * We first sort the divs so that the unknown divs appear last and
 * then we count how many equalities involve these divs.
 *
 * Let these equalities be of the form
 *
 *		A(x) + B y = 0
 *
 * where y represents the unknown divs and x the remaining variables.
 * Let [H 0] be the Hermite Normal Form of B, i.e.,
 *
 *		B = [H 0] Q
 *
 * Then x is a solution of the equalities iff
 *
 *		H^-1 A(x) (= - [I 0] Q y)
 *
 * is an integer vector.  Let d be the common denominator of H^-1.
 * We impose
 *
 *		d H^-1 A(x) = d alpha
 *
 * in add_strides, with alpha fresh existentially quantified variables.
 */
static __isl_give isl_basic_map *isl_basic_map_make_strides_explicit(
	__isl_take isl_basic_map *bmap)
{
	int known;
	int n_known;
	int n, n_col;
	int total;
	isl_ctx *ctx;
	isl_mat *A, *B, *M;

	known = isl_basic_map_divs_known(bmap);
	if (known < 0)
		return isl_basic_map_free(bmap);
	if (known)
		return bmap;
	bmap = isl_basic_map_sort_divs(bmap);
	bmap = isl_basic_map_gauss(bmap, NULL);
	if (!bmap)
		return NULL;

	for (n_known = 0; n_known < bmap->n_div; ++n_known)
		if (isl_int_is_zero(bmap->div[n_known][0]))
			break;
	ctx = isl_basic_map_get_ctx(bmap);
	total = isl_space_dim(bmap->dim, isl_dim_all);
	for (n = 0; n < bmap->n_eq; ++n)
		if (isl_seq_first_non_zero(bmap->eq[n] + 1 + total + n_known,
					    bmap->n_div - n_known) == -1)
			break;
	if (n == 0)
		return bmap;
	B = isl_mat_sub_alloc6(ctx, bmap->eq, 0, n, 0, 1 + total + n_known);
	n_col = bmap->n_div - n_known;
	A = isl_mat_sub_alloc6(ctx, bmap->eq, 0, n, 1 + total + n_known, n_col);
	A = isl_mat_left_hermite(A, 0, NULL, NULL);
	A = isl_mat_drop_cols(A, n, n_col - n);
	A = isl_mat_lin_to_aff(A);
	A = isl_mat_right_inverse(A);
	B = isl_mat_insert_zero_rows(B, 0, 1);
	B = isl_mat_set_element_si(B, 0, 0, 1);
	M = isl_mat_product(A, B);
	if (!M)
		return isl_basic_map_free(bmap);
	bmap = add_strides(bmap, M, n_known);
	bmap = isl_basic_map_gauss(bmap, NULL);
	isl_mat_free(M);

	return bmap;
}

/* Compute the affine hull of each basic map in "map" separately
 * and make all stride information explicit so that we can remove
 * all unknown divs without losing this information.
 * The result is also guaranteed to be gaussed.
 *
 * In simple cases where a div is determined by an equality,
 * calling isl_basic_map_gauss is enough to make the stride information
 * explicit, as it will derive an explicit representation for the div
 * from the equality.  If, however, the stride information
 * is encoded through multiple unknown divs then we need to make
 * some extra effort in isl_basic_map_make_strides_explicit.
 */
static __isl_give isl_map *isl_map_local_affine_hull(__isl_take isl_map *map)
{
	int i;

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_affine_hull(map->p[i]);
		map->p[i] = isl_basic_map_gauss(map->p[i], NULL);
		map->p[i] = isl_basic_map_make_strides_explicit(map->p[i]);
		if (!map->p[i])
			return isl_map_free(map);
	}

	return map;
}

static __isl_give isl_set *isl_set_local_affine_hull(__isl_take isl_set *set)
{
	return isl_map_local_affine_hull(set);
}

/* Return an empty basic map living in the same space as "map".
 */
static __isl_give isl_basic_map *replace_map_by_empty_basic_map(
	__isl_take isl_map *map)
{
	isl_space *space;

	space = isl_map_get_space(map);
	isl_map_free(map);
	return isl_basic_map_empty(space);
}

/* Compute the affine hull of "map".
 *
 * We first compute the affine hull of each basic map separately.
 * Then we align the divs and recompute the affine hulls of the basic
 * maps since some of them may now have extra divs.
 * In order to avoid performing parametric integer programming to
 * compute explicit expressions for the divs, possible leading to
 * an explosion in the number of basic maps, we first drop all unknown
 * divs before aligning the divs.  Note that isl_map_local_affine_hull tries
 * to make sure that all stride information is explicitly available
 * in terms of known divs.  This involves calling isl_basic_set_gauss,
 * which is also needed because affine_hull assumes its input has been gaussed,
 * while isl_map_affine_hull may be called on input that has not been gaussed,
 * in particular from initial_facet_constraint.
 * Similarly, align_divs may reorder some divs so that we need to
 * gauss the result again.
 * Finally, we combine the individual affine hulls into a single
 * affine hull.
 */
__isl_give isl_basic_map *isl_map_affine_hull(__isl_take isl_map *map)
{
	struct isl_basic_map *model = NULL;
	struct isl_basic_map *hull = NULL;
	struct isl_set *set;
	isl_basic_set *bset;

	map = isl_map_detect_equalities(map);
	map = isl_map_local_affine_hull(map);
	map = isl_map_remove_empty_parts(map);
	map = isl_map_remove_unknown_divs(map);
	map = isl_map_align_divs(map);

	if (!map)
		return NULL;

	if (map->n == 0)
		return replace_map_by_empty_basic_map(map);

	model = isl_basic_map_copy(map->p[0]);
	set = isl_map_underlying_set(map);
	set = isl_set_cow(set);
	set = isl_set_local_affine_hull(set);
	if (!set)
		goto error;

	while (set->n > 1)
		set->p[0] = affine_hull(set->p[0], set->p[--set->n]);

	bset = isl_basic_set_copy(set->p[0]);
	hull = isl_basic_map_overlying_set(bset, model);
	isl_set_free(set);
	hull = isl_basic_map_simplify(hull);
	return isl_basic_map_finalize(hull);
error:
	isl_basic_map_free(model);
	isl_set_free(set);
	return NULL;
}

struct isl_basic_set *isl_set_affine_hull(struct isl_set *set)
{
	return bset_from_bmap(isl_map_affine_hull(set_to_map(set)));
}
