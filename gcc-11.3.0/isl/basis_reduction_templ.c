/*
 * Copyright 2006-2007 Universiteit Leiden
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, Leiden Institute of Advanced Computer Science,
 * Universiteit Leiden, Niels Bohrweg 1, 2333 CA Leiden, The Netherlands
 * and K.U.Leuven, Departement Computerwetenschappen, Celestijnenlaan 200A,
 * B-3001 Leuven, Belgium
 */

#include <stdlib.h>
#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_vec_private.h>
#include <isl_options_private.h>
#include "isl_basis_reduction.h"

static void save_alpha(GBR_LP *lp, int first, int n, GBR_type *alpha)
{
	int i;

	for (i = 0; i < n; ++i)
		GBR_lp_get_alpha(lp, first + i, &alpha[i]);
}

/* Compute a reduced basis for the set represented by the tableau "tab".
 * tab->basis, which must be initialized by the calling function to an affine
 * unimodular basis, is updated to reflect the reduced basis.
 * The first tab->n_zero rows of the basis (ignoring the constant row)
 * are assumed to correspond to equalities and are left untouched.
 * tab->n_zero is updated to reflect any additional equalities that
 * have been detected in the first rows of the new basis.
 * The final tab->n_unbounded rows of the basis are assumed to correspond
 * to unbounded directions and are also left untouched.
 * In particular this means that the remaining rows are assumed to
 * correspond to bounded directions.
 *
 * This function implements the algorithm described in
 * "An Implementation of the Generalized Basis Reduction Algorithm
 *  for Integer Programming" of Cook el al. to compute a reduced basis.
 * We use \epsilon = 1/4.
 *
 * If ctx->opt->gbr_only_first is set, the user is only interested
 * in the first direction.  In this case we stop the basis reduction when
 * the width in the first direction becomes smaller than 2.
 */
struct isl_tab *isl_tab_compute_reduced_basis(struct isl_tab *tab)
{
	unsigned dim;
	struct isl_ctx *ctx;
	struct isl_mat *B;
	int i;
	GBR_LP *lp = NULL;
	GBR_type F_old, alpha, F_new;
	int row;
	isl_int tmp;
	struct isl_vec *b_tmp;
	GBR_type *F = NULL;
	GBR_type *alpha_buffer[2] = { NULL, NULL };
	GBR_type *alpha_saved;
	GBR_type F_saved;
	int use_saved = 0;
	isl_int mu[2];
	GBR_type mu_F[2];
	GBR_type two;
	GBR_type one;
	int empty = 0;
	int fixed = 0;
	int fixed_saved = 0;
	int mu_fixed[2];
	int n_bounded;
	int gbr_only_first;

	if (!tab)
		return NULL;

	if (tab->empty)
		return tab;

	ctx = tab->mat->ctx;
	gbr_only_first = ctx->opt->gbr_only_first;
	dim = tab->n_var;
	B = tab->basis;
	if (!B)
		return tab;

	n_bounded = dim - tab->n_unbounded;
	if (n_bounded <= tab->n_zero + 1)
		return tab;

	isl_int_init(tmp);
	isl_int_init(mu[0]);
	isl_int_init(mu[1]);

	GBR_init(alpha);
	GBR_init(F_old);
	GBR_init(F_new);
	GBR_init(F_saved);
	GBR_init(mu_F[0]);
	GBR_init(mu_F[1]);
	GBR_init(two);
	GBR_init(one);

	b_tmp = isl_vec_alloc(ctx, dim);
	if (!b_tmp)
		goto error;

	F = isl_alloc_array(ctx, GBR_type, n_bounded);
	alpha_buffer[0] = isl_alloc_array(ctx, GBR_type, n_bounded);
	alpha_buffer[1] = isl_alloc_array(ctx, GBR_type, n_bounded);
	alpha_saved = alpha_buffer[0];

	if (!F || !alpha_buffer[0] || !alpha_buffer[1])
		goto error;

	for (i = 0; i < n_bounded; ++i) {
		GBR_init(F[i]);
		GBR_init(alpha_buffer[0][i]);
		GBR_init(alpha_buffer[1][i]);
	}

	GBR_set_ui(two, 2);
	GBR_set_ui(one, 1);

	lp = GBR_lp_init(tab);
	if (!lp)
		goto error;

	i = tab->n_zero;

	GBR_lp_set_obj(lp, B->row[1+i]+1, dim);
	ctx->stats->gbr_solved_lps++;
	if (GBR_lp_solve(lp) < 0)
		goto error;
	GBR_lp_get_obj_val(lp, &F[i]);

	if (GBR_lt(F[i], one)) {
		if (!GBR_is_zero(F[i])) {
			empty = GBR_lp_cut(lp, B->row[1+i]+1);
			if (empty)
				goto done;
			GBR_set_ui(F[i], 0);
		}
		tab->n_zero++;
	}

	do {
		if (i+1 == tab->n_zero) {
			GBR_lp_set_obj(lp, B->row[1+i+1]+1, dim);
			ctx->stats->gbr_solved_lps++;
			if (GBR_lp_solve(lp) < 0)
				goto error;
			GBR_lp_get_obj_val(lp, &F_new);
			fixed = GBR_lp_is_fixed(lp);
			GBR_set_ui(alpha, 0);
		} else
		if (use_saved) {
			row = GBR_lp_next_row(lp);
			GBR_set(F_new, F_saved);
			fixed = fixed_saved;
			GBR_set(alpha, alpha_saved[i]);
		} else {
			row = GBR_lp_add_row(lp, B->row[1+i]+1, dim);
			GBR_lp_set_obj(lp, B->row[1+i+1]+1, dim);
			ctx->stats->gbr_solved_lps++;
			if (GBR_lp_solve(lp) < 0)
				goto error;
			GBR_lp_get_obj_val(lp, &F_new);
			fixed = GBR_lp_is_fixed(lp);

			GBR_lp_get_alpha(lp, row, &alpha);

			if (i > 0)
				save_alpha(lp, row-i, i, alpha_saved);

			if (GBR_lp_del_row(lp) < 0)
				goto error;
		}
		GBR_set(F[i+1], F_new);

		GBR_floor(mu[0], alpha);
		GBR_ceil(mu[1], alpha);

		if (isl_int_eq(mu[0], mu[1]))
			isl_int_set(tmp, mu[0]);
		else {
			int j;

			for (j = 0; j <= 1; ++j) {
				isl_int_set(tmp, mu[j]);
				isl_seq_combine(b_tmp->el,
						ctx->one, B->row[1+i+1]+1,
						tmp, B->row[1+i]+1, dim);
				GBR_lp_set_obj(lp, b_tmp->el, dim);
				ctx->stats->gbr_solved_lps++;
				if (GBR_lp_solve(lp) < 0)
					goto error;
				GBR_lp_get_obj_val(lp, &mu_F[j]);
				mu_fixed[j] = GBR_lp_is_fixed(lp);
				if (i > 0)
					save_alpha(lp, row-i, i, alpha_buffer[j]);
			}

			if (GBR_lt(mu_F[0], mu_F[1]))
				j = 0;
			else
				j = 1;

			isl_int_set(tmp, mu[j]);
			GBR_set(F_new, mu_F[j]);
			fixed = mu_fixed[j];
			alpha_saved = alpha_buffer[j];
		}
		isl_seq_combine(B->row[1+i+1]+1, ctx->one, B->row[1+i+1]+1,
				tmp, B->row[1+i]+1, dim);

		if (i+1 == tab->n_zero && fixed) {
			if (!GBR_is_zero(F[i+1])) {
				empty = GBR_lp_cut(lp, B->row[1+i+1]+1);
				if (empty)
					goto done;
				GBR_set_ui(F[i+1], 0);
			}
			tab->n_zero++;
		}

		GBR_set(F_old, F[i]);

		use_saved = 0;
		/* mu_F[0] = 4 * F_new; mu_F[1] = 3 * F_old */
		GBR_set_ui(mu_F[0], 4);
		GBR_mul(mu_F[0], mu_F[0], F_new);
		GBR_set_ui(mu_F[1], 3);
		GBR_mul(mu_F[1], mu_F[1], F_old);
		if (GBR_lt(mu_F[0], mu_F[1])) {
			B = isl_mat_swap_rows(B, 1 + i, 1 + i + 1);
			if (i > tab->n_zero) {
				use_saved = 1;
				GBR_set(F_saved, F_new);
				fixed_saved = fixed;
				if (GBR_lp_del_row(lp) < 0)
					goto error;
				--i;
			} else {
				GBR_set(F[tab->n_zero], F_new);
				if (gbr_only_first && GBR_lt(F[tab->n_zero], two))
					break;

				if (fixed) {
					if (!GBR_is_zero(F[tab->n_zero])) {
						empty = GBR_lp_cut(lp, B->row[1+tab->n_zero]+1);
						if (empty)
							goto done;
						GBR_set_ui(F[tab->n_zero], 0);
					}
					tab->n_zero++;
				}
			}
		} else {
			GBR_lp_add_row(lp, B->row[1+i]+1, dim);
			++i;
		}
	} while (i < n_bounded - 1);

	if (0) {
done:
		if (empty < 0) {
error:
			isl_mat_free(B);
			B = NULL;
		}
	}

	GBR_lp_delete(lp);

	if (alpha_buffer[1])
		for (i = 0; i < n_bounded; ++i) {
			GBR_clear(F[i]);
			GBR_clear(alpha_buffer[0][i]);
			GBR_clear(alpha_buffer[1][i]);
		}
	free(F);
	free(alpha_buffer[0]);
	free(alpha_buffer[1]);

	isl_vec_free(b_tmp);

	GBR_clear(alpha);
	GBR_clear(F_old);
	GBR_clear(F_new);
	GBR_clear(F_saved);
	GBR_clear(mu_F[0]);
	GBR_clear(mu_F[1]);
	GBR_clear(two);
	GBR_clear(one);

	isl_int_clear(tmp);
	isl_int_clear(mu[0]);
	isl_int_clear(mu[1]);

	tab->basis = B;

	return tab;
}

/* Compute an affine form of a reduced basis of the given basic
 * non-parametric set, which is assumed to be bounded and not
 * include any integer divisions.
 * The first column and the first row correspond to the constant term.
 *
 * If the input contains any equalities, we first create an initial
 * basis with the equalities first.  Otherwise, we start off with
 * the identity matrix.
 */
struct isl_mat *isl_basic_set_reduced_basis(struct isl_basic_set *bset)
{
	struct isl_mat *basis;
	struct isl_tab *tab;

	if (!bset)
		return NULL;

	if (isl_basic_set_dim(bset, isl_dim_div) != 0)
		isl_die(bset->ctx, isl_error_invalid,
			"no integer division allowed", return NULL);
	if (isl_basic_set_dim(bset, isl_dim_param) != 0)
		isl_die(bset->ctx, isl_error_invalid,
			"no parameters allowed", return NULL);

	tab = isl_tab_from_basic_set(bset, 0);
	if (!tab)
		return NULL;

	if (bset->n_eq == 0)
		tab->basis = isl_mat_identity(bset->ctx, 1 + tab->n_var);
	else {
		isl_mat *eq;
		unsigned nvar = isl_basic_set_total_dim(bset);
		eq = isl_mat_sub_alloc6(bset->ctx, bset->eq, 0, bset->n_eq,
					1, nvar);
		eq = isl_mat_left_hermite(eq, 0, NULL, &tab->basis);
		tab->basis = isl_mat_lin_to_aff(tab->basis);
		tab->n_zero = bset->n_eq;
		isl_mat_free(eq);
	}
	tab = isl_tab_compute_reduced_basis(tab);
	if (!tab)
		return NULL;

	basis = isl_mat_copy(tab->basis);

	isl_tab_free(tab);

	return basis;
}
