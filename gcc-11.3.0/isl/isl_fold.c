/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 */

#define ISL_DIM_H
#include <isl_map_private.h>
#include <isl_union_map_private.h>
#include <isl_polynomial_private.h>
#include <isl_point_private.h>
#include <isl_space_private.h>
#include <isl_lp_private.h>
#include <isl_seq.h>
#include <isl_mat_private.h>
#include <isl_val_private.h>
#include <isl_vec_private.h>
#include <isl_config.h>
#include <isl/deprecated/polynomial_int.h>

enum isl_fold isl_fold_type_negate(enum isl_fold type)
{
	switch (type) {
	case isl_fold_min:
		return isl_fold_max;
	case isl_fold_max:
		return isl_fold_min;
	case isl_fold_list:
		return isl_fold_list;
	}

	isl_die(NULL, isl_error_internal, "unhandled isl_fold type", abort());
}

static __isl_give isl_qpolynomial_fold *qpolynomial_fold_alloc(
	enum isl_fold type, __isl_take isl_space *dim, int n)
{
	isl_qpolynomial_fold *fold;

	if (!dim)
		goto error;

	isl_assert(dim->ctx, n >= 0, goto error);
	fold = isl_calloc(dim->ctx, struct isl_qpolynomial_fold,
			sizeof(struct isl_qpolynomial_fold) +
			(n - 1) * sizeof(struct isl_qpolynomial *));
	if (!fold)
		goto error;

	fold->ref = 1;
	fold->size = n;
	fold->n = 0;
	fold->type = type;
	fold->dim = dim;

	return fold;
error:
	isl_space_free(dim);
	return NULL;
}

isl_ctx *isl_qpolynomial_fold_get_ctx(__isl_keep isl_qpolynomial_fold *fold)
{
	return fold ? fold->dim->ctx : NULL;
}

__isl_give isl_space *isl_qpolynomial_fold_get_domain_space(
	__isl_keep isl_qpolynomial_fold *fold)
{
	return fold ? isl_space_copy(fold->dim) : NULL;
}

__isl_give isl_space *isl_qpolynomial_fold_get_space(
	__isl_keep isl_qpolynomial_fold *fold)
{
	isl_space *space;
	if (!fold)
		return NULL;
	space = isl_space_copy(fold->dim);
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, 1);
	return space;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_reset_domain_space(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_space *dim)
{
	int i;

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold || !dim)
		goto error;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_reset_domain_space(fold->qp[i],
							isl_space_copy(dim));
		if (!fold->qp[i])
			goto error;
	}

	isl_space_free(fold->dim);
	fold->dim = dim;

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	isl_space_free(dim);
	return NULL;
}

/* Reset the space of "fold".  This function is called from isl_pw_templ.c
 * and doesn't know if the space of an element object is represented
 * directly or through its domain.  It therefore passes along both.
 */
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_reset_space_and_domain(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_space *space,
	__isl_take isl_space *domain)
{
	isl_space_free(space);
	return isl_qpolynomial_fold_reset_domain_space(fold, domain);
}

int isl_qpolynomial_fold_involves_dims(__isl_keep isl_qpolynomial_fold *fold,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (!fold)
		return -1;
	if (fold->n == 0 || n == 0)
		return 0;

	for (i = 0; i < fold->n; ++i) {
		int involves = isl_qpolynomial_involves_dims(fold->qp[i],
							    type, first, n);
		if (involves < 0 || involves)
			return involves;
	}
	return 0;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_set_dim_name(
	__isl_take isl_qpolynomial_fold *fold,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	int i;

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		return NULL;
	fold->dim = isl_space_set_dim_name(fold->dim, type, pos, s);
	if (!fold->dim)
		goto error;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_set_dim_name(fold->qp[i],
							    type, pos, s);
		if (!fold->qp[i])
			goto error;
	}

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_drop_dims(
	__isl_take isl_qpolynomial_fold *fold,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	enum isl_dim_type set_type;

	if (!fold)
		return NULL;
	if (n == 0)
		return fold;

	set_type = type == isl_dim_in ? isl_dim_set : type;

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		return NULL;
	fold->dim = isl_space_drop_dims(fold->dim, set_type, first, n);
	if (!fold->dim)
		goto error;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_drop_dims(fold->qp[i],
							    type, first, n);
		if (!fold->qp[i])
			goto error;
	}

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_insert_dims(
	__isl_take isl_qpolynomial_fold *fold,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (!fold)
		return NULL;
	if (n == 0 && !isl_space_is_named_or_nested(fold->dim, type))
		return fold;

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		return NULL;
	fold->dim = isl_space_insert_dims(fold->dim, type, first, n);
	if (!fold->dim)
		goto error;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_insert_dims(fold->qp[i],
							    type, first, n);
		if (!fold->qp[i])
			goto error;
	}

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	return NULL;
}

/* Determine the sign of the constant quasipolynomial "qp".
 *
 * Return
 *	-1 if qp <= 0
 *	 1 if qp >= 0
 *	 0 if unknown
 *
 * For qp == 0, we can return either -1 or 1.  In practice, we return 1.
 * For qp == NaN, the sign is undefined, so we return 0.
 */
static int isl_qpolynomial_cst_sign(__isl_keep isl_qpolynomial *qp)
{
	struct isl_upoly_cst *cst;

	if (isl_qpolynomial_is_nan(qp))
		return 0;

	cst = isl_upoly_as_cst(qp->upoly);
	if (!cst)
		return 0;

	return isl_int_sgn(cst->n) < 0 ? -1 : 1;
}

static int isl_qpolynomial_aff_sign(__isl_keep isl_set *set,
	__isl_keep isl_qpolynomial *qp)
{
	enum isl_lp_result res;
	isl_vec *aff;
	isl_int opt;
	int sgn = 0;

	aff = isl_qpolynomial_extract_affine(qp);
	if (!aff)
		return 0;

	isl_int_init(opt);

	res = isl_set_solve_lp(set, 0, aff->el + 1, aff->el[0],
				&opt, NULL, NULL);
	if (res == isl_lp_error)
		goto done;
	if (res == isl_lp_empty ||
	    (res == isl_lp_ok && !isl_int_is_neg(opt))) {
		sgn = 1;
		goto done;
	}

	res = isl_set_solve_lp(set, 1, aff->el + 1, aff->el[0],
				&opt, NULL, NULL);
	if (res == isl_lp_ok && !isl_int_is_pos(opt))
		sgn = -1;

done:
	isl_int_clear(opt);
	isl_vec_free(aff);
	return sgn;
}

/* Determine, if possible, the sign of the quasipolynomial "qp" on
 * the domain "set".
 *
 * If qp is a constant, then the problem is trivial.
 * If qp is linear, then we check if the minimum of the corresponding
 * affine constraint is non-negative or if the maximum is non-positive.
 *
 * Otherwise, we check if the outermost variable "v" has a lower bound "l"
 * in "set".  If so, we write qp(v,v') as
 *
 *	q(v,v') * (v - l) + r(v')
 *
 * if q(v,v') and r(v') have the same known sign, then the original
 * quasipolynomial has the same sign as well.
 *
 * Return
 *	-1 if qp <= 0
 *	 1 if qp >= 0
 *	 0 if unknown
 */
static int isl_qpolynomial_sign(__isl_keep isl_set *set,
	__isl_keep isl_qpolynomial *qp)
{
	int d;
	int i;
	int is;
	struct isl_upoly_rec *rec;
	isl_vec *v;
	isl_int l;
	enum isl_lp_result res;
	int sgn = 0;

	is = isl_qpolynomial_is_cst(qp, NULL, NULL);
	if (is < 0)
		return 0;
	if (is)
		return isl_qpolynomial_cst_sign(qp);

	is = isl_qpolynomial_is_affine(qp);
	if (is < 0)
		return 0;
	if (is)
		return isl_qpolynomial_aff_sign(set, qp);

	if (qp->div->n_row > 0)
		return 0;

	rec = isl_upoly_as_rec(qp->upoly);
	if (!rec)
		return 0;

	d = isl_space_dim(qp->dim, isl_dim_all);
	v = isl_vec_alloc(set->ctx, 2 + d);
	if (!v)
		return 0;

	isl_seq_clr(v->el + 1, 1 + d);
	isl_int_set_si(v->el[0], 1);
	isl_int_set_si(v->el[2 + qp->upoly->var], 1);

	isl_int_init(l);

	res = isl_set_solve_lp(set, 0, v->el + 1, v->el[0], &l, NULL, NULL);
	if (res == isl_lp_ok) {
		isl_qpolynomial *min;
		isl_qpolynomial *base;
		isl_qpolynomial *r, *q;
		isl_qpolynomial *t;

		min = isl_qpolynomial_cst_on_domain(isl_space_copy(qp->dim), l);
		base = isl_qpolynomial_var_pow_on_domain(isl_space_copy(qp->dim),
						qp->upoly->var, 1);

		r = isl_qpolynomial_alloc(isl_space_copy(qp->dim), 0,
					  isl_upoly_copy(rec->p[rec->n - 1]));
		q = isl_qpolynomial_copy(r);

		for (i = rec->n - 2; i >= 0; --i) {
			r = isl_qpolynomial_mul(r, isl_qpolynomial_copy(min));
			t = isl_qpolynomial_alloc(isl_space_copy(qp->dim), 0,
						  isl_upoly_copy(rec->p[i]));
			r = isl_qpolynomial_add(r, t);
			if (i == 0)
				break;
			q = isl_qpolynomial_mul(q, isl_qpolynomial_copy(base));
			q = isl_qpolynomial_add(q, isl_qpolynomial_copy(r));
		}

		if (isl_qpolynomial_is_zero(q))
			sgn = isl_qpolynomial_sign(set, r);
		else if (isl_qpolynomial_is_zero(r))
			sgn = isl_qpolynomial_sign(set, q);
		else {
			int sgn_q, sgn_r;
			sgn_r = isl_qpolynomial_sign(set, r);
			sgn_q = isl_qpolynomial_sign(set, q);
			if (sgn_r == sgn_q)
				sgn = sgn_r;
		}

		isl_qpolynomial_free(min);
		isl_qpolynomial_free(base);
		isl_qpolynomial_free(q);
		isl_qpolynomial_free(r);
	}

	isl_int_clear(l);

	isl_vec_free(v);

	return sgn;
}

/* Combine "fold1" and "fold2" into a single reduction, eliminating
 * those elements of one reduction that are already covered by the other
 * reduction on "set".
 *
 * If "fold1" or "fold2" is an empty reduction, then return
 * the other reduction.
 * If "fold1" or "fold2" is a NaN, then return this NaN.
 */
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_fold_on_domain(
	__isl_keep isl_set *set,
	__isl_take isl_qpolynomial_fold *fold1,
	__isl_take isl_qpolynomial_fold *fold2)
{
	int i, j;
	int n1;
	struct isl_qpolynomial_fold *res = NULL;
	int better;

	if (!fold1 || !fold2)
		goto error;

	isl_assert(fold1->dim->ctx, fold1->type == fold2->type, goto error);
	isl_assert(fold1->dim->ctx, isl_space_is_equal(fold1->dim, fold2->dim),
			goto error);

	better = fold1->type == isl_fold_max ? -1 : 1;

	if (isl_qpolynomial_fold_is_empty(fold1) ||
	    isl_qpolynomial_fold_is_nan(fold2)) {
		isl_qpolynomial_fold_free(fold1);
		return fold2;
	}

	if (isl_qpolynomial_fold_is_empty(fold2) ||
	    isl_qpolynomial_fold_is_nan(fold1)) {
		isl_qpolynomial_fold_free(fold2);
		return fold1;
	}

	res = qpolynomial_fold_alloc(fold1->type, isl_space_copy(fold1->dim),
					fold1->n + fold2->n);
	if (!res)
		goto error;

	for (i = 0; i < fold1->n; ++i) {
		res->qp[res->n] = isl_qpolynomial_copy(fold1->qp[i]);
		if (!res->qp[res->n])
			goto error;
		res->n++;
	}
	n1 = res->n;

	for (i = 0; i < fold2->n; ++i) {
		for (j = n1 - 1; j >= 0; --j) {
			isl_qpolynomial *d;
			int sgn, equal;
			equal = isl_qpolynomial_plain_is_equal(res->qp[j],
								fold2->qp[i]);
			if (equal < 0)
				goto error;
			if (equal)
				break;
			d = isl_qpolynomial_sub(
				isl_qpolynomial_copy(res->qp[j]),
				isl_qpolynomial_copy(fold2->qp[i]));
			sgn = isl_qpolynomial_sign(set, d);
			isl_qpolynomial_free(d);
			if (sgn == 0)
				continue;
			if (sgn != better)
				break;
			isl_qpolynomial_free(res->qp[j]);
			if (j != n1 - 1)
				res->qp[j] = res->qp[n1 - 1];
			n1--;
			if (n1 != res->n - 1)
				res->qp[n1] = res->qp[res->n - 1];
			res->n--;
		}
		if (j >= 0)
			continue;
		res->qp[res->n] = isl_qpolynomial_copy(fold2->qp[i]);
		if (!res->qp[res->n])
			goto error;
		res->n++;
	}

	isl_qpolynomial_fold_free(fold1);
	isl_qpolynomial_fold_free(fold2);

	return res;
error:
	isl_qpolynomial_fold_free(res);
	isl_qpolynomial_fold_free(fold1);
	isl_qpolynomial_fold_free(fold2);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_add_qpolynomial(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_qpolynomial *qp)
{
	int i;

	if (!fold || !qp)
		goto error;

	if (isl_qpolynomial_is_zero(qp)) {
		isl_qpolynomial_free(qp);
		return fold;
	}

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		goto error;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_add(fold->qp[i],
						isl_qpolynomial_copy(qp));
		if (!fold->qp[i])
			goto error;
	}

	isl_qpolynomial_free(qp);
	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_add_on_domain(
	__isl_keep isl_set *dom,
	__isl_take isl_qpolynomial_fold *fold1,
	__isl_take isl_qpolynomial_fold *fold2)
{
	int i;
	isl_qpolynomial_fold *res = NULL;

	if (!fold1 || !fold2)
		goto error;

	if (isl_qpolynomial_fold_is_empty(fold1)) {
		isl_qpolynomial_fold_free(fold1);
		return fold2;
	}

	if (isl_qpolynomial_fold_is_empty(fold2)) {
		isl_qpolynomial_fold_free(fold2);
		return fold1;
	}

	if (fold1->n == 1 && fold2->n != 1)
		return isl_qpolynomial_fold_add_on_domain(dom, fold2, fold1);

	if (fold2->n == 1) {
		res = isl_qpolynomial_fold_add_qpolynomial(fold1,
					isl_qpolynomial_copy(fold2->qp[0]));
		isl_qpolynomial_fold_free(fold2);
		return res;
	}

	res = isl_qpolynomial_fold_add_qpolynomial(
				isl_qpolynomial_fold_copy(fold1),
				isl_qpolynomial_copy(fold2->qp[0]));

	for (i = 1; i < fold2->n; ++i) {
		isl_qpolynomial_fold *res_i;
		res_i = isl_qpolynomial_fold_add_qpolynomial(
					isl_qpolynomial_fold_copy(fold1),
					isl_qpolynomial_copy(fold2->qp[i]));
		res = isl_qpolynomial_fold_fold_on_domain(dom, res, res_i);
	}

	isl_qpolynomial_fold_free(fold1);
	isl_qpolynomial_fold_free(fold2);
	return res;
error:
	isl_qpolynomial_fold_free(res);
	isl_qpolynomial_fold_free(fold1);
	isl_qpolynomial_fold_free(fold2);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_substitute_equalities(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_basic_set *eq)
{
	int i;

	if (!fold || !eq)
		goto error;

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		return NULL;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_substitute_equalities(fold->qp[i],
							isl_basic_set_copy(eq));
		if (!fold->qp[i])
			goto error;
	}

	isl_basic_set_free(eq);
	return fold;
error:
	isl_basic_set_free(eq);
	isl_qpolynomial_fold_free(fold);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_gist(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_set *context)
{
	int i;

	if (!fold || !context)
		goto error;

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		return NULL;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_gist(fold->qp[i],
							isl_set_copy(context));
		if (!fold->qp[i])
			goto error;
	}

	isl_set_free(context);
	return fold;
error:
	isl_set_free(context);
	isl_qpolynomial_fold_free(fold);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_gist_params(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_set *context)
{
	isl_space *space = isl_qpolynomial_fold_get_domain_space(fold);
	isl_set *dom_context = isl_set_universe(space);
	dom_context = isl_set_intersect_params(dom_context, context);
	return isl_qpolynomial_fold_gist(fold, dom_context);
}

#define HAS_TYPE

#undef PW
#define PW isl_pw_qpolynomial_fold
#undef EL
#define EL isl_qpolynomial_fold
#undef EL_IS_ZERO
#define EL_IS_ZERO is_empty
#undef ZERO
#define ZERO zero
#undef IS_ZERO
#define IS_ZERO is_zero
#undef FIELD
#define FIELD fold
#undef DEFAULT_IS_ZERO
#define DEFAULT_IS_ZERO 1

#define NO_NEG
#define NO_SUB
#define NO_PULLBACK

#include <isl_pw_templ.c>

#undef UNION
#define UNION isl_union_pw_qpolynomial_fold
#undef PART
#define PART isl_pw_qpolynomial_fold
#undef PARTS
#define PARTS pw_qpolynomial_fold

#define NO_SUB

#include <isl_union_single.c>
#include <isl_union_eval.c>

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_empty(enum isl_fold type,
	__isl_take isl_space *dim)
{
	return qpolynomial_fold_alloc(type, dim, 0);
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_alloc(
	enum isl_fold type, __isl_take isl_qpolynomial *qp)
{
	isl_qpolynomial_fold *fold;

	if (!qp)
		return NULL;

	fold = qpolynomial_fold_alloc(type, isl_space_copy(qp->dim), 1);
	if (!fold)
		goto error;

	fold->qp[0] = qp;
	fold->n++;

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_copy(
	__isl_keep isl_qpolynomial_fold *fold)
{
	if (!fold)
		return NULL;

	fold->ref++;
	return fold;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_dup(
	__isl_keep isl_qpolynomial_fold *fold)
{
	int i;
	isl_qpolynomial_fold *dup;

	if (!fold)
		return NULL;
	dup = qpolynomial_fold_alloc(fold->type,
					isl_space_copy(fold->dim), fold->n);
	if (!dup)
		return NULL;
	
	dup->n = fold->n;
	for (i = 0; i < fold->n; ++i) {
		dup->qp[i] = isl_qpolynomial_copy(fold->qp[i]);
		if (!dup->qp[i])
			goto error;
	}

	return dup;
error:
	isl_qpolynomial_fold_free(dup);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_cow(
	__isl_take isl_qpolynomial_fold *fold)
{
	if (!fold)
		return NULL;

	if (fold->ref == 1)
		return fold;
	fold->ref--;
	return isl_qpolynomial_fold_dup(fold);
}

void isl_qpolynomial_fold_free(__isl_take isl_qpolynomial_fold *fold)
{
	int i;

	if (!fold)
		return;
	if (--fold->ref > 0)
		return;

	for (i = 0; i < fold->n; ++i)
		isl_qpolynomial_free(fold->qp[i]);
	isl_space_free(fold->dim);
	free(fold);
}

int isl_qpolynomial_fold_is_empty(__isl_keep isl_qpolynomial_fold *fold)
{
	if (!fold)
		return -1;

	return fold->n == 0;
}

/* Does "fold" represent max(NaN) or min(NaN)?
 */
isl_bool isl_qpolynomial_fold_is_nan(__isl_keep isl_qpolynomial_fold *fold)
{
	if (!fold)
		return isl_bool_error;
	if (fold->n != 1)
		return isl_bool_false;
	return isl_qpolynomial_is_nan(fold->qp[0]);
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_fold(
	__isl_take isl_qpolynomial_fold *fold1,
	__isl_take isl_qpolynomial_fold *fold2)
{
	int i;
	struct isl_qpolynomial_fold *res = NULL;

	if (!fold1 || !fold2)
		goto error;

	isl_assert(fold1->dim->ctx, fold1->type == fold2->type, goto error);
	isl_assert(fold1->dim->ctx, isl_space_is_equal(fold1->dim, fold2->dim),
			goto error);

	if (isl_qpolynomial_fold_is_empty(fold1)) {
		isl_qpolynomial_fold_free(fold1);
		return fold2;
	}

	if (isl_qpolynomial_fold_is_empty(fold2)) {
		isl_qpolynomial_fold_free(fold2);
		return fold1;
	}

	res = qpolynomial_fold_alloc(fold1->type, isl_space_copy(fold1->dim),
					fold1->n + fold2->n);
	if (!res)
		goto error;

	for (i = 0; i < fold1->n; ++i) {
		res->qp[res->n] = isl_qpolynomial_copy(fold1->qp[i]);
		if (!res->qp[res->n])
			goto error;
		res->n++;
	}

	for (i = 0; i < fold2->n; ++i) {
		res->qp[res->n] = isl_qpolynomial_copy(fold2->qp[i]);
		if (!res->qp[res->n])
			goto error;
		res->n++;
	}

	isl_qpolynomial_fold_free(fold1);
	isl_qpolynomial_fold_free(fold2);

	return res;
error:
	isl_qpolynomial_fold_free(res);
	isl_qpolynomial_fold_free(fold1);
	isl_qpolynomial_fold_free(fold2);
	return NULL;
}

__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_fold_fold(
	__isl_take isl_pw_qpolynomial_fold *pw1,
	__isl_take isl_pw_qpolynomial_fold *pw2)
{
	int i, j, n;
	struct isl_pw_qpolynomial_fold *res;
	isl_set *set;

	if (!pw1 || !pw2)
		goto error;

	isl_assert(pw1->dim->ctx, isl_space_is_equal(pw1->dim, pw2->dim), goto error);

	if (isl_pw_qpolynomial_fold_is_zero(pw1)) {
		isl_pw_qpolynomial_fold_free(pw1);
		return pw2;
	}

	if (isl_pw_qpolynomial_fold_is_zero(pw2)) {
		isl_pw_qpolynomial_fold_free(pw2);
		return pw1;
	}

	if (pw1->type != pw2->type)
		isl_die(pw1->dim->ctx, isl_error_invalid,
			"fold types don't match", goto error);

	n = (pw1->n + 1) * (pw2->n + 1);
	res = isl_pw_qpolynomial_fold_alloc_size(isl_space_copy(pw1->dim),
						pw1->type, n);

	for (i = 0; i < pw1->n; ++i) {
		set = isl_set_copy(pw1->p[i].set);
		for (j = 0; j < pw2->n; ++j) {
			struct isl_set *common;
			isl_qpolynomial_fold *sum;
			set = isl_set_subtract(set,
					isl_set_copy(pw2->p[j].set));
			common = isl_set_intersect(isl_set_copy(pw1->p[i].set),
						isl_set_copy(pw2->p[j].set));
			if (isl_set_plain_is_empty(common)) {
				isl_set_free(common);
				continue;
			}

			sum = isl_qpolynomial_fold_fold_on_domain(common,
			       isl_qpolynomial_fold_copy(pw1->p[i].fold),
			       isl_qpolynomial_fold_copy(pw2->p[j].fold));

			res = isl_pw_qpolynomial_fold_add_piece(res, common, sum);
		}
		res = isl_pw_qpolynomial_fold_add_piece(res, set,
			isl_qpolynomial_fold_copy(pw1->p[i].fold));
	}

	for (j = 0; j < pw2->n; ++j) {
		set = isl_set_copy(pw2->p[j].set);
		for (i = 0; i < pw1->n; ++i)
			set = isl_set_subtract(set, isl_set_copy(pw1->p[i].set));
		res = isl_pw_qpolynomial_fold_add_piece(res, set,
				    isl_qpolynomial_fold_copy(pw2->p[j].fold));
	}

	isl_pw_qpolynomial_fold_free(pw1);
	isl_pw_qpolynomial_fold_free(pw2);

	return res;
error:
	isl_pw_qpolynomial_fold_free(pw1);
	isl_pw_qpolynomial_fold_free(pw2);
	return NULL;
}

__isl_give isl_union_pw_qpolynomial_fold *isl_union_pw_qpolynomial_fold_fold_pw_qpolynomial_fold(
	__isl_take isl_union_pw_qpolynomial_fold *u,
	__isl_take isl_pw_qpolynomial_fold *part)
{
	struct isl_hash_table_entry *entry;

	u = isl_union_pw_qpolynomial_fold_cow(u);

	if (!part || !u)
		goto error;

	isl_assert(u->space->ctx,
	    isl_space_match(part->dim, isl_dim_param, u->space, isl_dim_param),
	    goto error);

	entry = isl_union_pw_qpolynomial_fold_find_part_entry(u, part->dim, 1);
	if (!entry)
		goto error;

	if (!entry->data)
		entry->data = part;
	else {
		entry->data = isl_pw_qpolynomial_fold_fold(entry->data,
					    isl_pw_qpolynomial_fold_copy(part));
		if (!entry->data)
			goto error;
		isl_pw_qpolynomial_fold_free(part);
	}

	return u;
error:
	isl_pw_qpolynomial_fold_free(part);
	isl_union_pw_qpolynomial_fold_free(u);
	return NULL;
}

static isl_stat fold_part(__isl_take isl_pw_qpolynomial_fold *part, void *user)
{
	isl_union_pw_qpolynomial_fold **u;
	u = (isl_union_pw_qpolynomial_fold **)user;

	*u = isl_union_pw_qpolynomial_fold_fold_pw_qpolynomial_fold(*u, part);

	return isl_stat_ok;
}

__isl_give isl_union_pw_qpolynomial_fold *isl_union_pw_qpolynomial_fold_fold(
	__isl_take isl_union_pw_qpolynomial_fold *u1,
	__isl_take isl_union_pw_qpolynomial_fold *u2)
{
	u1 = isl_union_pw_qpolynomial_fold_cow(u1);

	if (!u1 || !u2)
		goto error;

	if (isl_union_pw_qpolynomial_fold_foreach_pw_qpolynomial_fold(u2,
							&fold_part, &u1) < 0)
		goto error;

	isl_union_pw_qpolynomial_fold_free(u2);

	return u1;
error:
	isl_union_pw_qpolynomial_fold_free(u1);
	isl_union_pw_qpolynomial_fold_free(u2);
	return NULL;
}

__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_fold_from_pw_qpolynomial(
	enum isl_fold type, __isl_take isl_pw_qpolynomial *pwqp)
{
	int i;
	isl_pw_qpolynomial_fold *pwf;

	if (!pwqp)
		return NULL;
	
	pwf = isl_pw_qpolynomial_fold_alloc_size(isl_space_copy(pwqp->dim),
						type, pwqp->n);

	for (i = 0; i < pwqp->n; ++i)
		pwf = isl_pw_qpolynomial_fold_add_piece(pwf,
			isl_set_copy(pwqp->p[i].set),
			isl_qpolynomial_fold_alloc(type,
				isl_qpolynomial_copy(pwqp->p[i].qp)));

	isl_pw_qpolynomial_free(pwqp);

	return pwf;
}

__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_fold_add(
	__isl_take isl_pw_qpolynomial_fold *pwf1,
	__isl_take isl_pw_qpolynomial_fold *pwf2)
{
	return isl_pw_qpolynomial_fold_union_add_(pwf1, pwf2);
}

/* Compare two quasi-polynomial reductions.
 *
 * Return -1 if "fold1" is "smaller" than "fold2", 1 if "fold1" is "greater"
 * than "fold2" and 0 if they are equal.
 */
int isl_qpolynomial_fold_plain_cmp(__isl_keep isl_qpolynomial_fold *fold1,
	__isl_keep isl_qpolynomial_fold *fold2)
{
	int i;

	if (fold1 == fold2)
		return 0;
	if (!fold1)
		return -1;
	if (!fold2)
		return 1;

	if (fold1->n != fold2->n)
		return fold1->n - fold2->n;

	for (i = 0; i < fold1->n; ++i) {
		int cmp;

		cmp = isl_qpolynomial_plain_cmp(fold1->qp[i], fold2->qp[i]);
		if (cmp != 0)
			return cmp;
	}

	return 0;
}

int isl_qpolynomial_fold_plain_is_equal(__isl_keep isl_qpolynomial_fold *fold1,
	__isl_keep isl_qpolynomial_fold *fold2)
{
	int i;

	if (!fold1 || !fold2)
		return -1;

	if (fold1->n != fold2->n)
		return 0;

	/* We probably want to sort the qps first... */
	for (i = 0; i < fold1->n; ++i) {
		int eq = isl_qpolynomial_plain_is_equal(fold1->qp[i], fold2->qp[i]);
		if (eq < 0 || !eq)
			return eq;
	}

	return 1;
}

__isl_give isl_val *isl_qpolynomial_fold_eval(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_point *pnt)
{
	isl_ctx *ctx;
	isl_val *v;

	if (!fold || !pnt)
		goto error;
	ctx = isl_point_get_ctx(pnt);
	isl_assert(pnt->dim->ctx, isl_space_is_equal(pnt->dim, fold->dim), goto error);
	isl_assert(pnt->dim->ctx,
		fold->type == isl_fold_max || fold->type == isl_fold_min,
		goto error);

	if (fold->n == 0)
		v = isl_val_zero(ctx);
	else {
		int i;
		v = isl_qpolynomial_eval(isl_qpolynomial_copy(fold->qp[0]),
						isl_point_copy(pnt));
		for (i = 1; i < fold->n; ++i) {
			isl_val *v_i;
			v_i = isl_qpolynomial_eval(
					    isl_qpolynomial_copy(fold->qp[i]),
					    isl_point_copy(pnt));
			if (fold->type == isl_fold_max)
				v = isl_val_max(v, v_i);
			else
				v = isl_val_min(v, v_i);
		}
	}
	isl_qpolynomial_fold_free(fold);
	isl_point_free(pnt);

	return v;
error:
	isl_qpolynomial_fold_free(fold);
	isl_point_free(pnt);
	return NULL;
}

size_t isl_pw_qpolynomial_fold_size(__isl_keep isl_pw_qpolynomial_fold *pwf)
{
	int i;
	size_t n = 0;

	for (i = 0; i < pwf->n; ++i)
		n += pwf->p[i].fold->n;

	return n;
}

__isl_give isl_val *isl_qpolynomial_fold_opt_on_domain(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_set *set, int max)
{
	int i;
	isl_val *opt;

	if (!set || !fold)
		goto error;

	if (fold->n == 0) {
		opt = isl_val_zero(isl_set_get_ctx(set));
		isl_set_free(set);
		isl_qpolynomial_fold_free(fold);
		return opt;
	}

	opt = isl_qpolynomial_opt_on_domain(isl_qpolynomial_copy(fold->qp[0]),
						isl_set_copy(set), max);
	for (i = 1; i < fold->n; ++i) {
		isl_val *opt_i;
		opt_i = isl_qpolynomial_opt_on_domain(
				isl_qpolynomial_copy(fold->qp[i]),
				isl_set_copy(set), max);
		if (max)
			opt = isl_val_max(opt, opt_i);
		else
			opt = isl_val_min(opt, opt_i);
	}

	isl_set_free(set);
	isl_qpolynomial_fold_free(fold);

	return opt;
error:
	isl_set_free(set);
	isl_qpolynomial_fold_free(fold);
	return NULL;
}

/* Check whether for each quasi-polynomial in "fold2" there is
 * a quasi-polynomial in "fold1" that dominates it on "set".
 */
static int qpolynomial_fold_covers_on_domain(__isl_keep isl_set *set,
	__isl_keep isl_qpolynomial_fold *fold1,
	__isl_keep isl_qpolynomial_fold *fold2)
{
	int i, j;
	int covers;

	if (!set || !fold1 || !fold2)
		return -1;

	covers = fold1->type == isl_fold_max ? 1 : -1;

	for (i = 0; i < fold2->n; ++i) {
		for (j = 0; j < fold1->n; ++j) {
			isl_qpolynomial *d;
			int sgn;

			d = isl_qpolynomial_sub(
				isl_qpolynomial_copy(fold1->qp[j]),
				isl_qpolynomial_copy(fold2->qp[i]));
			sgn = isl_qpolynomial_sign(set, d);
			isl_qpolynomial_free(d);
			if (sgn == covers)
				break;
		}
		if (j >= fold1->n)
			return 0;
	}

	return 1;
}

/* Check whether "pwf1" dominated "pwf2", i.e., the domain of "pwf1" contains
 * that of "pwf2" and on each cell, the corresponding fold from pwf1 dominates
 * that of pwf2.
 */
int isl_pw_qpolynomial_fold_covers(__isl_keep isl_pw_qpolynomial_fold *pwf1,
	__isl_keep isl_pw_qpolynomial_fold *pwf2)
{
	int i, j;
	isl_set *dom1, *dom2;
	int is_subset;

	if (!pwf1 || !pwf2)
		return -1;

	if (pwf2->n == 0)
		return 1;
	if (pwf1->n == 0)
		return 0;

	dom1 = isl_pw_qpolynomial_fold_domain(isl_pw_qpolynomial_fold_copy(pwf1));
	dom2 = isl_pw_qpolynomial_fold_domain(isl_pw_qpolynomial_fold_copy(pwf2));
	is_subset = isl_set_is_subset(dom2, dom1);
	isl_set_free(dom1);
	isl_set_free(dom2);

	if (is_subset < 0 || !is_subset)
		return is_subset;

	for (i = 0; i < pwf2->n; ++i) {
		for (j = 0; j < pwf1->n; ++j) {
			int is_empty;
			isl_set *common;
			int covers;

			common = isl_set_intersect(isl_set_copy(pwf1->p[j].set),
						   isl_set_copy(pwf2->p[i].set));
			is_empty = isl_set_is_empty(common);
			if (is_empty < 0 || is_empty) {
				isl_set_free(common);
				if (is_empty < 0)
					return -1;
				continue;
			}
			covers = qpolynomial_fold_covers_on_domain(common,
					pwf1->p[j].fold, pwf2->p[i].fold);
			isl_set_free(common);
			if (covers < 0 || !covers)
				return covers;
		}
	}

	return 1;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_morph_domain(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_morph *morph)
{
	int i;
	isl_ctx *ctx;

	if (!fold || !morph)
		goto error;

	ctx = fold->dim->ctx;
	isl_assert(ctx, isl_space_is_equal(fold->dim, morph->dom->dim), goto error);

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		goto error;

	isl_space_free(fold->dim);
	fold->dim = isl_space_copy(morph->ran->dim);
	if (!fold->dim)
		goto error;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_morph_domain(fold->qp[i],
						isl_morph_copy(morph));
		if (!fold->qp[i])
			goto error;
	}

	isl_morph_free(morph);

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	isl_morph_free(morph);
	return NULL;
}

enum isl_fold isl_qpolynomial_fold_get_type(__isl_keep isl_qpolynomial_fold *fold)
{
	if (!fold)
		return isl_fold_list;
	return fold->type;
}

enum isl_fold isl_union_pw_qpolynomial_fold_get_type(
	__isl_keep isl_union_pw_qpolynomial_fold *upwf)
{
	if (!upwf)
		return isl_fold_list;
	return upwf->type;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_lift(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_space *dim)
{
	int i;

	if (!fold || !dim)
		goto error;

	if (isl_space_is_equal(fold->dim, dim)) {
		isl_space_free(dim);
		return fold;
	}

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		goto error;

	isl_space_free(fold->dim);
	fold->dim = isl_space_copy(dim);
	if (!fold->dim)
		goto error;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_lift(fold->qp[i],
						isl_space_copy(dim));
		if (!fold->qp[i])
			goto error;
	}

	isl_space_free(dim);

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	isl_space_free(dim);
	return NULL;
}

isl_stat isl_qpolynomial_fold_foreach_qpolynomial(
	__isl_keep isl_qpolynomial_fold *fold,
	isl_stat (*fn)(__isl_take isl_qpolynomial *qp, void *user), void *user)
{
	int i;

	if (!fold)
		return isl_stat_error;

	for (i = 0; i < fold->n; ++i)
		if (fn(isl_qpolynomial_copy(fold->qp[i]), user) < 0)
			return isl_stat_error;

	return isl_stat_ok;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_move_dims(
	__isl_take isl_qpolynomial_fold *fold,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	int i;

	if (n == 0)
		return fold;

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		return NULL;

	fold->dim = isl_space_move_dims(fold->dim, dst_type, dst_pos,
						src_type, src_pos, n);
	if (!fold->dim)
		goto error;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_move_dims(fold->qp[i],
				dst_type, dst_pos, src_type, src_pos, n);
		if (!fold->qp[i])
			goto error;
	}

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	return NULL;
}

/* For each 0 <= i < "n", replace variable "first" + i of type "type"
 * in fold->qp[k] by subs[i].
 */
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_substitute(
	__isl_take isl_qpolynomial_fold *fold,
	enum isl_dim_type type, unsigned first, unsigned n,
	__isl_keep isl_qpolynomial **subs)
{
	int i;

	if (n == 0)
		return fold;

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		return NULL;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_substitute(fold->qp[i],
				type, first, n, subs);
		if (!fold->qp[i])
			goto error;
	}

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	return NULL;
}

static isl_stat add_pwqp(__isl_take isl_pw_qpolynomial *pwqp, void *user)
{
	isl_ctx *ctx;
	isl_pw_qpolynomial_fold *pwf;
	isl_union_pw_qpolynomial_fold **upwf;
	struct isl_hash_table_entry *entry;

	upwf = (isl_union_pw_qpolynomial_fold **)user;

	ctx = pwqp->dim->ctx;
	entry = isl_union_pw_qpolynomial_fold_find_part_entry(*upwf,
			 pwqp->dim, 1);
	if (!entry)
		goto error;

	pwf = isl_pw_qpolynomial_fold_from_pw_qpolynomial((*upwf)->type, pwqp);
	if (!entry->data)
		entry->data = pwf;
	else {
		entry->data = isl_pw_qpolynomial_fold_add(entry->data, pwf);
		if (!entry->data)
			return isl_stat_error;
		if (isl_pw_qpolynomial_fold_is_zero(entry->data))
			*upwf = isl_union_pw_qpolynomial_fold_remove_part_entry(
								*upwf, entry);
	}

	return isl_stat_ok;
error:
	isl_pw_qpolynomial_free(pwqp);
	return isl_stat_error;
}

__isl_give isl_union_pw_qpolynomial_fold *isl_union_pw_qpolynomial_fold_add_union_pw_qpolynomial(
	__isl_take isl_union_pw_qpolynomial_fold *upwf,
	__isl_take isl_union_pw_qpolynomial *upwqp)
{
	upwf = isl_union_pw_qpolynomial_fold_align_params(upwf,
				isl_union_pw_qpolynomial_get_space(upwqp));
	upwqp = isl_union_pw_qpolynomial_align_params(upwqp,
				isl_union_pw_qpolynomial_fold_get_space(upwf));

	upwf = isl_union_pw_qpolynomial_fold_cow(upwf);
	if (!upwf || !upwqp)
		goto error;

	if (isl_union_pw_qpolynomial_foreach_pw_qpolynomial(upwqp, &add_pwqp,
							 &upwf) < 0)
		goto error;

	isl_union_pw_qpolynomial_free(upwqp);

	return upwf;
error:
	isl_union_pw_qpolynomial_fold_free(upwf);
	isl_union_pw_qpolynomial_free(upwqp);
	return NULL;
}

static int join_compatible(__isl_keep isl_space *dim1, __isl_keep isl_space *dim2)
{
	int m;
	m = isl_space_match(dim1, isl_dim_param, dim2, isl_dim_param);
	if (m < 0 || !m)
		return m;
	return isl_space_tuple_is_equal(dim1, isl_dim_out, dim2, isl_dim_in);
}

/* Compute the intersection of the range of the map and the domain
 * of the piecewise quasipolynomial reduction and then compute a bound
 * on the associated quasipolynomial reduction over all elements
 * in this intersection.
 *
 * We first introduce some unconstrained dimensions in the
 * piecewise quasipolynomial, intersect the resulting domain
 * with the wrapped map and the compute the sum.
 */
__isl_give isl_pw_qpolynomial_fold *isl_map_apply_pw_qpolynomial_fold(
	__isl_take isl_map *map, __isl_take isl_pw_qpolynomial_fold *pwf,
	int *tight)
{
	isl_ctx *ctx;
	isl_set *dom;
	isl_space *map_dim;
	isl_space *pwf_dim;
	unsigned n_in;
	int ok;

	ctx = isl_map_get_ctx(map);
	if (!ctx)
		goto error;

	map_dim = isl_map_get_space(map);
	pwf_dim = isl_pw_qpolynomial_fold_get_space(pwf);
	ok = join_compatible(map_dim, pwf_dim);
	isl_space_free(map_dim);
	isl_space_free(pwf_dim);
	if (!ok)
		isl_die(ctx, isl_error_invalid, "incompatible dimensions",
			goto error);

	n_in = isl_map_dim(map, isl_dim_in);
	pwf = isl_pw_qpolynomial_fold_insert_dims(pwf, isl_dim_in, 0, n_in);

	dom = isl_map_wrap(map);
	pwf = isl_pw_qpolynomial_fold_reset_domain_space(pwf,
						isl_set_get_space(dom));

	pwf = isl_pw_qpolynomial_fold_intersect_domain(pwf, dom);
	pwf = isl_pw_qpolynomial_fold_bound(pwf, tight);
	
	return pwf;
error:
	isl_map_free(map);
	isl_pw_qpolynomial_fold_free(pwf);
	return NULL;
}

__isl_give isl_pw_qpolynomial_fold *isl_set_apply_pw_qpolynomial_fold(
	__isl_take isl_set *set, __isl_take isl_pw_qpolynomial_fold *pwf,
	int *tight)
{
	return isl_map_apply_pw_qpolynomial_fold(set, pwf, tight);
}

struct isl_apply_fold_data {
	isl_union_pw_qpolynomial_fold *upwf;
	isl_union_pw_qpolynomial_fold *res;
	isl_map *map;
	int tight;
};

static isl_stat pw_qpolynomial_fold_apply(
	__isl_take isl_pw_qpolynomial_fold *pwf, void *user)
{
	isl_space *map_dim;
	isl_space *pwf_dim;
	struct isl_apply_fold_data *data = user;
	int ok;

	map_dim = isl_map_get_space(data->map);
	pwf_dim = isl_pw_qpolynomial_fold_get_space(pwf);
	ok = join_compatible(map_dim, pwf_dim);
	isl_space_free(map_dim);
	isl_space_free(pwf_dim);

	if (ok) {
		pwf = isl_map_apply_pw_qpolynomial_fold(isl_map_copy(data->map),
				    pwf, data->tight ? &data->tight : NULL);
		data->res = isl_union_pw_qpolynomial_fold_fold_pw_qpolynomial_fold(
							data->res, pwf);
	} else
		isl_pw_qpolynomial_fold_free(pwf);

	return isl_stat_ok;
}

static isl_stat map_apply(__isl_take isl_map *map, void *user)
{
	struct isl_apply_fold_data *data = user;
	isl_stat r;

	data->map = map;
	r = isl_union_pw_qpolynomial_fold_foreach_pw_qpolynomial_fold(
				data->upwf, &pw_qpolynomial_fold_apply, data);

	isl_map_free(map);
	return r;
}

__isl_give isl_union_pw_qpolynomial_fold *isl_union_map_apply_union_pw_qpolynomial_fold(
	__isl_take isl_union_map *umap,
	__isl_take isl_union_pw_qpolynomial_fold *upwf, int *tight)
{
	isl_space *dim;
	enum isl_fold type;
	struct isl_apply_fold_data data;

	upwf = isl_union_pw_qpolynomial_fold_align_params(upwf,
				isl_union_map_get_space(umap));
	umap = isl_union_map_align_params(umap,
				isl_union_pw_qpolynomial_fold_get_space(upwf));

	data.upwf = upwf;
	data.tight = tight ? 1 : 0;
	dim = isl_union_pw_qpolynomial_fold_get_space(upwf);
	type = isl_union_pw_qpolynomial_fold_get_type(upwf);
	data.res = isl_union_pw_qpolynomial_fold_zero(dim, type);
	if (isl_union_map_foreach_map(umap, &map_apply, &data) < 0)
		goto error;

	isl_union_map_free(umap);
	isl_union_pw_qpolynomial_fold_free(upwf);

	if (tight)
		*tight = data.tight;

	return data.res;
error:
	isl_union_map_free(umap);
	isl_union_pw_qpolynomial_fold_free(upwf);
	isl_union_pw_qpolynomial_fold_free(data.res);
	return NULL;
}

__isl_give isl_union_pw_qpolynomial_fold *isl_union_set_apply_union_pw_qpolynomial_fold(
	__isl_take isl_union_set *uset,
	__isl_take isl_union_pw_qpolynomial_fold *upwf, int *tight)
{
	return isl_union_map_apply_union_pw_qpolynomial_fold(uset, upwf, tight);
}

/* Reorder the dimension of "fold" according to the given reordering.
 */
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_realign_domain(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_reordering *r)
{
	int i;

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold || !r)
		goto error;

	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_realign_domain(fold->qp[i],
						    isl_reordering_copy(r));
		if (!fold->qp[i])
			goto error;
	}

	fold = isl_qpolynomial_fold_reset_domain_space(fold,
						    isl_space_copy(r->dim));

	isl_reordering_free(r);

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	isl_reordering_free(r);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_mul_isl_int(
	__isl_take isl_qpolynomial_fold *fold, isl_int v)
{
	int i;

	if (isl_int_is_one(v))
		return fold;
	if (fold && isl_int_is_zero(v)) {
		isl_qpolynomial_fold *zero;
		isl_space *dim = isl_space_copy(fold->dim);
		zero = isl_qpolynomial_fold_empty(fold->type, dim);
		isl_qpolynomial_fold_free(fold);
		return zero;
	}

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		return NULL;

	if (isl_int_is_neg(v))
		fold->type = isl_fold_type_negate(fold->type);
	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_mul_isl_int(fold->qp[i], v);
		if (!fold->qp[i])
			goto error;
	}

	return fold;
error:
	isl_qpolynomial_fold_free(fold);
	return NULL;
}

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_scale(
	__isl_take isl_qpolynomial_fold *fold, isl_int v)
{
	return isl_qpolynomial_fold_mul_isl_int(fold, v);
}

/* Multiply "fold" by "v".
 */
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_scale_val(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_val *v)
{
	int i;

	if (!fold || !v)
		goto error;

	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return fold;
	}
	if (isl_val_is_zero(v)) {
		isl_qpolynomial_fold *zero;
		isl_space *space = isl_qpolynomial_fold_get_domain_space(fold);
		zero = isl_qpolynomial_fold_empty(fold->type, space);
		isl_qpolynomial_fold_free(fold);
		isl_val_free(v);
		return zero;
	}
	if (!isl_val_is_rat(v))
		isl_die(isl_qpolynomial_fold_get_ctx(fold), isl_error_invalid,
			"expecting rational factor", goto error);

	fold = isl_qpolynomial_fold_cow(fold);
	if (!fold)
		goto error;

	if (isl_val_is_neg(v))
		fold->type = isl_fold_type_negate(fold->type);
	for (i = 0; i < fold->n; ++i) {
		fold->qp[i] = isl_qpolynomial_scale_val(fold->qp[i],
							isl_val_copy(v));
		if (!fold->qp[i])
			goto error;
	}

	isl_val_free(v);
	return fold;
error:
	isl_val_free(v);
	isl_qpolynomial_fold_free(fold);
	return NULL;
}

/* Divide "fold" by "v".
 */
__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_scale_down_val(
	__isl_take isl_qpolynomial_fold *fold, __isl_take isl_val *v)
{
	if (!fold || !v)
		goto error;

	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return fold;
	}
	if (!isl_val_is_rat(v))
		isl_die(isl_qpolynomial_fold_get_ctx(fold), isl_error_invalid,
			"expecting rational factor", goto error);
	if (isl_val_is_zero(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"cannot scale down by zero", goto error);

	return isl_qpolynomial_fold_scale_val(fold, isl_val_inv(v));
error:
	isl_val_free(v);
	isl_qpolynomial_fold_free(fold);
	return NULL;
}
