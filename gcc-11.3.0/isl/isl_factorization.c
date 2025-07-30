/*
 * Copyright 2005-2007 Universiteit Leiden
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

#include <isl_map_private.h>
#include <isl_factorization.h>
#include <isl_space_private.h>
#include <isl_mat_private.h>

static __isl_give isl_factorizer *isl_factorizer_alloc(
	__isl_take isl_morph *morph, int n_group)
{
	isl_factorizer *f = NULL;
	int *len = NULL;

	if (!morph)
		return NULL;

	if (n_group > 0) {
		len = isl_alloc_array(morph->dom->ctx, int, n_group);
		if (!len)
			goto error;
	}

	f = isl_alloc_type(morph->dom->ctx, struct isl_factorizer);
	if (!f)
		goto error;

	f->morph = morph;
	f->n_group = n_group;
	f->len = len;

	return f;
error:
	free(len);
	isl_morph_free(morph);
	return NULL;
}

void isl_factorizer_free(__isl_take isl_factorizer *f)
{
	if (!f)
		return;

	isl_morph_free(f->morph);
	free(f->len);
	free(f);
}

void isl_factorizer_dump(__isl_take isl_factorizer *f)
{
	int i;

	if (!f)
		return;

	isl_morph_print_internal(f->morph, stderr);
	fprintf(stderr, "[");
	for (i = 0; i < f->n_group; ++i) {
		if (i)
			fprintf(stderr, ", ");
		fprintf(stderr, "%d", f->len[i]);
	}
	fprintf(stderr, "]\n");
}

__isl_give isl_factorizer *isl_factorizer_identity(__isl_keep isl_basic_set *bset)
{
	return isl_factorizer_alloc(isl_morph_identity(bset), 0);
}

__isl_give isl_factorizer *isl_factorizer_groups(__isl_keep isl_basic_set *bset,
	__isl_take isl_mat *Q, __isl_take isl_mat *U, int n, int *len)
{
	int i;
	unsigned nvar;
	unsigned ovar;
	isl_space *dim;
	isl_basic_set *dom;
	isl_basic_set *ran;
	isl_morph *morph;
	isl_factorizer *f;
	isl_mat *id;

	if (!bset || !Q || !U)
		goto error;

	ovar = 1 + isl_space_offset(bset->dim, isl_dim_set);
	id = isl_mat_identity(bset->ctx, ovar);
	Q = isl_mat_diagonal(isl_mat_copy(id), Q);
	U = isl_mat_diagonal(id, U);

	nvar = isl_basic_set_dim(bset, isl_dim_set);
	dim = isl_basic_set_get_space(bset);
	dom = isl_basic_set_universe(isl_space_copy(dim));
	dim = isl_space_drop_dims(dim, isl_dim_set, 0, nvar);
	dim = isl_space_add_dims(dim, isl_dim_set, nvar);
	ran = isl_basic_set_universe(dim);
	morph = isl_morph_alloc(dom, ran, Q, U);
	f = isl_factorizer_alloc(morph, n);
	if (!f)
		return NULL;
	for (i = 0; i < n; ++i)
		f->len[i] = len[i];
	return f;
error:
	isl_mat_free(Q);
	isl_mat_free(U);
	return NULL;
}

struct isl_factor_groups {
	int *pos;		/* for each column: row position of pivot */
	int *group;		/* group to which a column belongs */
	int *cnt;		/* number of columns in the group */
	int *rowgroup;		/* group to which a constraint belongs */
};

/* Initialize isl_factor_groups structure: find pivot row positions,
 * each column initially belongs to its own group and the groups
 * of the constraints are still unknown.
 */
static int init_groups(struct isl_factor_groups *g, __isl_keep isl_mat *H)
{
	int i, j;

	if (!H)
		return -1;

	g->pos = isl_alloc_array(H->ctx, int, H->n_col);
	g->group = isl_alloc_array(H->ctx, int, H->n_col);
	g->cnt = isl_alloc_array(H->ctx, int, H->n_col);
	g->rowgroup = isl_alloc_array(H->ctx, int, H->n_row);

	if (!g->pos || !g->group || !g->cnt || !g->rowgroup)
		return -1;

	for (i = 0; i < H->n_row; ++i)
		g->rowgroup[i] = -1;
	for (i = 0, j = 0; i < H->n_col; ++i) {
		for ( ; j < H->n_row; ++j)
			if (!isl_int_is_zero(H->row[j][i]))
				break;
		g->pos[i] = j;
	}
	for (i = 0; i < H->n_col; ++i) {
		g->group[i] = i;
		g->cnt[i] = 1;
	}

	return 0;
}

/* Update group[k] to the group column k belongs to.
 * When merging two groups, only the group of the current
 * group leader is changed.  Here we change the group of
 * the other members to also point to the group that the
 * old group leader now points to.
 */
static void update_group(struct isl_factor_groups *g, int k)
{
	int p = g->group[k];
	while (g->cnt[p] == 0)
		p = g->group[p];
	g->group[k] = p;
}

/* Merge group i with all groups of the subsequent columns
 * with non-zero coefficients in row j of H.
 * (The previous columns are all zero; otherwise we would have handled
 * the row before.)
 */
static int update_group_i_with_row_j(struct isl_factor_groups *g, int i, int j,
	__isl_keep isl_mat *H)
{
	int k;

	g->rowgroup[j] = g->group[i];
	for (k = i + 1; k < H->n_col && j >= g->pos[k]; ++k) {
		update_group(g, k);
		update_group(g, i);
		if (g->group[k] != g->group[i] &&
		    !isl_int_is_zero(H->row[j][k])) {
			isl_assert(H->ctx, g->cnt[g->group[k]] != 0, return -1);
			isl_assert(H->ctx, g->cnt[g->group[i]] != 0, return -1);
			if (g->group[i] < g->group[k]) {
				g->cnt[g->group[i]] += g->cnt[g->group[k]];
				g->cnt[g->group[k]] = 0;
				g->group[g->group[k]] = g->group[i];
			} else {
				g->cnt[g->group[k]] += g->cnt[g->group[i]];
				g->cnt[g->group[i]] = 0;
				g->group[g->group[i]] = g->group[k];
			}
		}
	}

	return 0;
}

/* Update the group information based on the constraint matrix.
 */
static int update_groups(struct isl_factor_groups *g, __isl_keep isl_mat *H)
{
	int i, j;

	for (i = 0; i < H->n_col && g->cnt[0] < H->n_col; ++i) {
		if (g->pos[i] == H->n_row)
			continue; /* A line direction */
		if (g->rowgroup[g->pos[i]] == -1)
			g->rowgroup[g->pos[i]] = i;
		for (j = g->pos[i] + 1; j < H->n_row; ++j) {
			if (isl_int_is_zero(H->row[j][i]))
				continue;
			if (g->rowgroup[j] != -1)
				continue;
			if (update_group_i_with_row_j(g, i, j, H) < 0)
				return -1;
		}
	}
	for (i = 1; i < H->n_col; ++i)
		update_group(g, i);

	return 0;
}

static void clear_groups(struct isl_factor_groups *g)
{
	if (!g)
		return;
	free(g->pos);
	free(g->group);
	free(g->cnt);
	free(g->rowgroup);
}

/* Determine if the set variables of the basic set can be factorized and
 * return the results in an isl_factorizer.
 *
 * The algorithm works by first computing the Hermite normal form
 * and then grouping columns linked by one or more constraints together,
 * where a constraints "links" two or more columns if the constraint
 * has nonzero coefficients in the columns.
 */
__isl_give isl_factorizer *isl_basic_set_factorizer(
	__isl_keep isl_basic_set *bset)
{
	int i, j, n, done;
	isl_mat *H, *U, *Q;
	unsigned nvar;
	struct isl_factor_groups g = { 0 };
	isl_factorizer *f;

	if (!bset)
		return NULL;

	isl_assert(bset->ctx, isl_basic_set_dim(bset, isl_dim_div) == 0,
		return NULL);

	nvar = isl_basic_set_dim(bset, isl_dim_set);
	if (nvar <= 1)
		return isl_factorizer_identity(bset);

	H = isl_mat_alloc(bset->ctx, bset->n_eq + bset->n_ineq, nvar);
	if (!H)
		return NULL;
	isl_mat_sub_copy(bset->ctx, H->row, bset->eq, bset->n_eq,
		0, 1 + isl_space_offset(bset->dim, isl_dim_set), nvar);
	isl_mat_sub_copy(bset->ctx, H->row + bset->n_eq, bset->ineq, bset->n_ineq,
		0, 1 + isl_space_offset(bset->dim, isl_dim_set), nvar);
	H = isl_mat_left_hermite(H, 0, &U, &Q);

	if (init_groups(&g, H) < 0)
		goto error;
	if (update_groups(&g, H) < 0)
		goto error;

	if (g.cnt[0] == nvar) {
		isl_mat_free(H);
		isl_mat_free(U);
		isl_mat_free(Q);
		clear_groups(&g);

		return isl_factorizer_identity(bset);
	}

	done = 0;
	n = 0;
	while (done != nvar) {
		int group = g.group[done];
		for (i = 1; i < g.cnt[group]; ++i) {
			if (g.group[done + i] == group)
				continue;
			for (j = done + g.cnt[group]; j < nvar; ++j)
				if (g.group[j] == group)
					break;
			if (j == nvar)
				isl_die(bset->ctx, isl_error_internal,
					"internal error", goto error);
			g.group[j] = g.group[done + i];
			Q = isl_mat_swap_rows(Q, done + i, j);
			U = isl_mat_swap_cols(U, done + i, j);
		}
		done += g.cnt[group];
		g.pos[n++] = g.cnt[group];
	}

	f = isl_factorizer_groups(bset, Q, U, n, g.pos);

	isl_mat_free(H);
	clear_groups(&g);

	return f;
error:
	isl_mat_free(H);
	isl_mat_free(U);
	isl_mat_free(Q);
	clear_groups(&g);
	return NULL;
}
