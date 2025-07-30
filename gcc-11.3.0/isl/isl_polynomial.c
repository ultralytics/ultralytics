/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 */

#include <stdlib.h>
#define ISL_DIM_H
#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_factorization.h>
#include <isl_lp_private.h>
#include <isl_seq.h>
#include <isl_union_map_private.h>
#include <isl_constraint_private.h>
#include <isl_polynomial_private.h>
#include <isl_point_private.h>
#include <isl_space_private.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl_range.h>
#include <isl_local.h>
#include <isl_local_space_private.h>
#include <isl_aff_private.h>
#include <isl_val_private.h>
#include <isl_config.h>
#include <isl/deprecated/polynomial_int.h>

static unsigned pos(__isl_keep isl_space *dim, enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_param:	return 0;
	case isl_dim_in:	return dim->nparam;
	case isl_dim_out:	return dim->nparam + dim->n_in;
	default:		return 0;
	}
}

int isl_upoly_is_cst(__isl_keep struct isl_upoly *up)
{
	if (!up)
		return -1;

	return up->var < 0;
}

__isl_keep struct isl_upoly_cst *isl_upoly_as_cst(__isl_keep struct isl_upoly *up)
{
	if (!up)
		return NULL;

	isl_assert(up->ctx, up->var < 0, return NULL);

	return (struct isl_upoly_cst *)up;
}

__isl_keep struct isl_upoly_rec *isl_upoly_as_rec(__isl_keep struct isl_upoly *up)
{
	if (!up)
		return NULL;

	isl_assert(up->ctx, up->var >= 0, return NULL);

	return (struct isl_upoly_rec *)up;
}

/* Compare two polynomials.
 *
 * Return -1 if "up1" is "smaller" than "up2", 1 if "up1" is "greater"
 * than "up2" and 0 if they are equal.
 */
static int isl_upoly_plain_cmp(__isl_keep struct isl_upoly *up1,
	__isl_keep struct isl_upoly *up2)
{
	int i;
	struct isl_upoly_rec *rec1, *rec2;

	if (up1 == up2)
		return 0;
	if (!up1)
		return -1;
	if (!up2)
		return 1;
	if (up1->var != up2->var)
		return up1->var - up2->var;

	if (isl_upoly_is_cst(up1)) {
		struct isl_upoly_cst *cst1, *cst2;
		int cmp;

		cst1 = isl_upoly_as_cst(up1);
		cst2 = isl_upoly_as_cst(up2);
		if (!cst1 || !cst2)
			return 0;
		cmp = isl_int_cmp(cst1->n, cst2->n);
		if (cmp != 0)
			return cmp;
		return isl_int_cmp(cst1->d, cst2->d);
	}

	rec1 = isl_upoly_as_rec(up1);
	rec2 = isl_upoly_as_rec(up2);
	if (!rec1 || !rec2)
		return 0;

	if (rec1->n != rec2->n)
		return rec1->n - rec2->n;

	for (i = 0; i < rec1->n; ++i) {
		int cmp = isl_upoly_plain_cmp(rec1->p[i], rec2->p[i]);
		if (cmp != 0)
			return cmp;
	}

	return 0;
}

isl_bool isl_upoly_is_equal(__isl_keep struct isl_upoly *up1,
	__isl_keep struct isl_upoly *up2)
{
	int i;
	struct isl_upoly_rec *rec1, *rec2;

	if (!up1 || !up2)
		return isl_bool_error;
	if (up1 == up2)
		return isl_bool_true;
	if (up1->var != up2->var)
		return isl_bool_false;
	if (isl_upoly_is_cst(up1)) {
		struct isl_upoly_cst *cst1, *cst2;
		cst1 = isl_upoly_as_cst(up1);
		cst2 = isl_upoly_as_cst(up2);
		if (!cst1 || !cst2)
			return isl_bool_error;
		return isl_int_eq(cst1->n, cst2->n) &&
		       isl_int_eq(cst1->d, cst2->d);
	}

	rec1 = isl_upoly_as_rec(up1);
	rec2 = isl_upoly_as_rec(up2);
	if (!rec1 || !rec2)
		return isl_bool_error;

	if (rec1->n != rec2->n)
		return isl_bool_false;

	for (i = 0; i < rec1->n; ++i) {
		isl_bool eq = isl_upoly_is_equal(rec1->p[i], rec2->p[i]);
		if (eq < 0 || !eq)
			return eq;
	}

	return isl_bool_true;
}

int isl_upoly_is_zero(__isl_keep struct isl_upoly *up)
{
	struct isl_upoly_cst *cst;

	if (!up)
		return -1;
	if (!isl_upoly_is_cst(up))
		return 0;

	cst = isl_upoly_as_cst(up);
	if (!cst)
		return -1;

	return isl_int_is_zero(cst->n) && isl_int_is_pos(cst->d);
}

int isl_upoly_sgn(__isl_keep struct isl_upoly *up)
{
	struct isl_upoly_cst *cst;

	if (!up)
		return 0;
	if (!isl_upoly_is_cst(up))
		return 0;

	cst = isl_upoly_as_cst(up);
	if (!cst)
		return 0;

	return isl_int_sgn(cst->n);
}

int isl_upoly_is_nan(__isl_keep struct isl_upoly *up)
{
	struct isl_upoly_cst *cst;

	if (!up)
		return -1;
	if (!isl_upoly_is_cst(up))
		return 0;

	cst = isl_upoly_as_cst(up);
	if (!cst)
		return -1;

	return isl_int_is_zero(cst->n) && isl_int_is_zero(cst->d);
}

int isl_upoly_is_infty(__isl_keep struct isl_upoly *up)
{
	struct isl_upoly_cst *cst;

	if (!up)
		return -1;
	if (!isl_upoly_is_cst(up))
		return 0;

	cst = isl_upoly_as_cst(up);
	if (!cst)
		return -1;

	return isl_int_is_pos(cst->n) && isl_int_is_zero(cst->d);
}

int isl_upoly_is_neginfty(__isl_keep struct isl_upoly *up)
{
	struct isl_upoly_cst *cst;

	if (!up)
		return -1;
	if (!isl_upoly_is_cst(up))
		return 0;

	cst = isl_upoly_as_cst(up);
	if (!cst)
		return -1;

	return isl_int_is_neg(cst->n) && isl_int_is_zero(cst->d);
}

int isl_upoly_is_one(__isl_keep struct isl_upoly *up)
{
	struct isl_upoly_cst *cst;

	if (!up)
		return -1;
	if (!isl_upoly_is_cst(up))
		return 0;

	cst = isl_upoly_as_cst(up);
	if (!cst)
		return -1;

	return isl_int_eq(cst->n, cst->d) && isl_int_is_pos(cst->d);
}

int isl_upoly_is_negone(__isl_keep struct isl_upoly *up)
{
	struct isl_upoly_cst *cst;

	if (!up)
		return -1;
	if (!isl_upoly_is_cst(up))
		return 0;

	cst = isl_upoly_as_cst(up);
	if (!cst)
		return -1;

	return isl_int_is_negone(cst->n) && isl_int_is_one(cst->d);
}

__isl_give struct isl_upoly_cst *isl_upoly_cst_alloc(struct isl_ctx *ctx)
{
	struct isl_upoly_cst *cst;

	cst = isl_alloc_type(ctx, struct isl_upoly_cst);
	if (!cst)
		return NULL;

	cst->up.ref = 1;
	cst->up.ctx = ctx;
	isl_ctx_ref(ctx);
	cst->up.var = -1;

	isl_int_init(cst->n);
	isl_int_init(cst->d);

	return cst;
}

__isl_give struct isl_upoly *isl_upoly_zero(struct isl_ctx *ctx)
{
	struct isl_upoly_cst *cst;

	cst = isl_upoly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set_si(cst->n, 0);
	isl_int_set_si(cst->d, 1);

	return &cst->up;
}

__isl_give struct isl_upoly *isl_upoly_one(struct isl_ctx *ctx)
{
	struct isl_upoly_cst *cst;

	cst = isl_upoly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set_si(cst->n, 1);
	isl_int_set_si(cst->d, 1);

	return &cst->up;
}

__isl_give struct isl_upoly *isl_upoly_infty(struct isl_ctx *ctx)
{
	struct isl_upoly_cst *cst;

	cst = isl_upoly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set_si(cst->n, 1);
	isl_int_set_si(cst->d, 0);

	return &cst->up;
}

__isl_give struct isl_upoly *isl_upoly_neginfty(struct isl_ctx *ctx)
{
	struct isl_upoly_cst *cst;

	cst = isl_upoly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set_si(cst->n, -1);
	isl_int_set_si(cst->d, 0);

	return &cst->up;
}

__isl_give struct isl_upoly *isl_upoly_nan(struct isl_ctx *ctx)
{
	struct isl_upoly_cst *cst;

	cst = isl_upoly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set_si(cst->n, 0);
	isl_int_set_si(cst->d, 0);

	return &cst->up;
}

__isl_give struct isl_upoly *isl_upoly_rat_cst(struct isl_ctx *ctx,
	isl_int n, isl_int d)
{
	struct isl_upoly_cst *cst;

	cst = isl_upoly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set(cst->n, n);
	isl_int_set(cst->d, d);

	return &cst->up;
}

__isl_give struct isl_upoly_rec *isl_upoly_alloc_rec(struct isl_ctx *ctx,
	int var, int size)
{
	struct isl_upoly_rec *rec;

	isl_assert(ctx, var >= 0, return NULL);
	isl_assert(ctx, size >= 0, return NULL);
	rec = isl_calloc(ctx, struct isl_upoly_rec,
			sizeof(struct isl_upoly_rec) +
			size * sizeof(struct isl_upoly *));
	if (!rec)
		return NULL;

	rec->up.ref = 1;
	rec->up.ctx = ctx;
	isl_ctx_ref(ctx);
	rec->up.var = var;

	rec->n = 0;
	rec->size = size;

	return rec;
}

__isl_give isl_qpolynomial *isl_qpolynomial_reset_domain_space(
	__isl_take isl_qpolynomial *qp, __isl_take isl_space *dim)
{
	qp = isl_qpolynomial_cow(qp);
	if (!qp || !dim)
		goto error;

	isl_space_free(qp->dim);
	qp->dim = dim;

	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_space_free(dim);
	return NULL;
}

/* Reset the space of "qp".  This function is called from isl_pw_templ.c
 * and doesn't know if the space of an element object is represented
 * directly or through its domain.  It therefore passes along both.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_reset_space_and_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_space *space,
	__isl_take isl_space *domain)
{
	isl_space_free(space);
	return isl_qpolynomial_reset_domain_space(qp, domain);
}

isl_ctx *isl_qpolynomial_get_ctx(__isl_keep isl_qpolynomial *qp)
{
	return qp ? qp->dim->ctx : NULL;
}

__isl_give isl_space *isl_qpolynomial_get_domain_space(
	__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_space_copy(qp->dim) : NULL;
}

__isl_give isl_space *isl_qpolynomial_get_space(__isl_keep isl_qpolynomial *qp)
{
	isl_space *space;
	if (!qp)
		return NULL;
	space = isl_space_copy(qp->dim);
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, 1);
	return space;
}

/* Return the number of variables of the given type in the domain of "qp".
 */
unsigned isl_qpolynomial_domain_dim(__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type)
{
	if (!qp)
		return 0;
	if (type == isl_dim_div)
		return qp->div->n_row;
	if (type == isl_dim_all)
		return isl_space_dim(qp->dim, isl_dim_all) +
				    isl_qpolynomial_domain_dim(qp, isl_dim_div);
	return isl_space_dim(qp->dim, type);
}

/* Externally, an isl_qpolynomial has a map space, but internally, the
 * ls field corresponds to the domain of that space.
 */
unsigned isl_qpolynomial_dim(__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type)
{
	if (!qp)
		return 0;
	if (type == isl_dim_out)
		return 1;
	if (type == isl_dim_in)
		type = isl_dim_set;
	return isl_qpolynomial_domain_dim(qp, type);
}

/* Return the offset of the first coefficient of type "type" in
 * the domain of "qp".
 */
unsigned isl_qpolynomial_domain_offset(__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type)
{
	if (!qp)
		return 0;
	switch (type) {
	case isl_dim_cst:
		return 0;
	case isl_dim_param:
	case isl_dim_set:
		return 1 + isl_space_offset(qp->dim, type);
	case isl_dim_div:
		return 1 + isl_space_dim(qp->dim, isl_dim_all);
	default:
		return 0;
	}
}

isl_bool isl_qpolynomial_is_zero(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_upoly_is_zero(qp->upoly) : isl_bool_error;
}

isl_bool isl_qpolynomial_is_one(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_upoly_is_one(qp->upoly) : isl_bool_error;
}

isl_bool isl_qpolynomial_is_nan(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_upoly_is_nan(qp->upoly) : isl_bool_error;
}

isl_bool isl_qpolynomial_is_infty(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_upoly_is_infty(qp->upoly) : isl_bool_error;
}

isl_bool isl_qpolynomial_is_neginfty(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_upoly_is_neginfty(qp->upoly) : isl_bool_error;
}

int isl_qpolynomial_sgn(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_upoly_sgn(qp->upoly) : 0;
}

static void upoly_free_cst(__isl_take struct isl_upoly_cst *cst)
{
	isl_int_clear(cst->n);
	isl_int_clear(cst->d);
}

static void upoly_free_rec(__isl_take struct isl_upoly_rec *rec)
{
	int i;

	for (i = 0; i < rec->n; ++i)
		isl_upoly_free(rec->p[i]);
}

__isl_give struct isl_upoly *isl_upoly_copy(__isl_keep struct isl_upoly *up)
{
	if (!up)
		return NULL;

	up->ref++;
	return up;
}

__isl_give struct isl_upoly *isl_upoly_dup_cst(__isl_keep struct isl_upoly *up)
{
	struct isl_upoly_cst *cst;
	struct isl_upoly_cst *dup;

	cst = isl_upoly_as_cst(up);
	if (!cst)
		return NULL;

	dup = isl_upoly_as_cst(isl_upoly_zero(up->ctx));
	if (!dup)
		return NULL;
	isl_int_set(dup->n, cst->n);
	isl_int_set(dup->d, cst->d);

	return &dup->up;
}

__isl_give struct isl_upoly *isl_upoly_dup_rec(__isl_keep struct isl_upoly *up)
{
	int i;
	struct isl_upoly_rec *rec;
	struct isl_upoly_rec *dup;

	rec = isl_upoly_as_rec(up);
	if (!rec)
		return NULL;

	dup = isl_upoly_alloc_rec(up->ctx, up->var, rec->n);
	if (!dup)
		return NULL;

	for (i = 0; i < rec->n; ++i) {
		dup->p[i] = isl_upoly_copy(rec->p[i]);
		if (!dup->p[i])
			goto error;
		dup->n++;
	}

	return &dup->up;
error:
	isl_upoly_free(&dup->up);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_dup(__isl_keep struct isl_upoly *up)
{
	if (!up)
		return NULL;

	if (isl_upoly_is_cst(up))
		return isl_upoly_dup_cst(up);
	else
		return isl_upoly_dup_rec(up);
}

__isl_give struct isl_upoly *isl_upoly_cow(__isl_take struct isl_upoly *up)
{
	if (!up)
		return NULL;

	if (up->ref == 1)
		return up;
	up->ref--;
	return isl_upoly_dup(up);
}

void isl_upoly_free(__isl_take struct isl_upoly *up)
{
	if (!up)
		return;

	if (--up->ref > 0)
		return;

	if (up->var < 0)
		upoly_free_cst((struct isl_upoly_cst *)up);
	else
		upoly_free_rec((struct isl_upoly_rec *)up);

	isl_ctx_deref(up->ctx);
	free(up);
}

static void isl_upoly_cst_reduce(__isl_keep struct isl_upoly_cst *cst)
{
	isl_int gcd;

	isl_int_init(gcd);
	isl_int_gcd(gcd, cst->n, cst->d);
	if (!isl_int_is_zero(gcd) && !isl_int_is_one(gcd)) {
		isl_int_divexact(cst->n, cst->n, gcd);
		isl_int_divexact(cst->d, cst->d, gcd);
	}
	isl_int_clear(gcd);
}

__isl_give struct isl_upoly *isl_upoly_sum_cst(__isl_take struct isl_upoly *up1,
	__isl_take struct isl_upoly *up2)
{
	struct isl_upoly_cst *cst1;
	struct isl_upoly_cst *cst2;

	up1 = isl_upoly_cow(up1);
	if (!up1 || !up2)
		goto error;

	cst1 = isl_upoly_as_cst(up1);
	cst2 = isl_upoly_as_cst(up2);

	if (isl_int_eq(cst1->d, cst2->d))
		isl_int_add(cst1->n, cst1->n, cst2->n);
	else {
		isl_int_mul(cst1->n, cst1->n, cst2->d);
		isl_int_addmul(cst1->n, cst2->n, cst1->d);
		isl_int_mul(cst1->d, cst1->d, cst2->d);
	}

	isl_upoly_cst_reduce(cst1);

	isl_upoly_free(up2);
	return up1;
error:
	isl_upoly_free(up1);
	isl_upoly_free(up2);
	return NULL;
}

static __isl_give struct isl_upoly *replace_by_zero(
	__isl_take struct isl_upoly *up)
{
	struct isl_ctx *ctx;

	if (!up)
		return NULL;
	ctx = up->ctx;
	isl_upoly_free(up);
	return isl_upoly_zero(ctx);
}

static __isl_give struct isl_upoly *replace_by_constant_term(
	__isl_take struct isl_upoly *up)
{
	struct isl_upoly_rec *rec;
	struct isl_upoly *cst;

	if (!up)
		return NULL;

	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;
	cst = isl_upoly_copy(rec->p[0]);
	isl_upoly_free(up);
	return cst;
error:
	isl_upoly_free(up);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_sum(__isl_take struct isl_upoly *up1,
	__isl_take struct isl_upoly *up2)
{
	int i;
	struct isl_upoly_rec *rec1, *rec2;

	if (!up1 || !up2)
		goto error;

	if (isl_upoly_is_nan(up1)) {
		isl_upoly_free(up2);
		return up1;
	}

	if (isl_upoly_is_nan(up2)) {
		isl_upoly_free(up1);
		return up2;
	}

	if (isl_upoly_is_zero(up1)) {
		isl_upoly_free(up1);
		return up2;
	}

	if (isl_upoly_is_zero(up2)) {
		isl_upoly_free(up2);
		return up1;
	}

	if (up1->var < up2->var)
		return isl_upoly_sum(up2, up1);

	if (up2->var < up1->var) {
		struct isl_upoly_rec *rec;
		if (isl_upoly_is_infty(up2) || isl_upoly_is_neginfty(up2)) {
			isl_upoly_free(up1);
			return up2;
		}
		up1 = isl_upoly_cow(up1);
		rec = isl_upoly_as_rec(up1);
		if (!rec)
			goto error;
		rec->p[0] = isl_upoly_sum(rec->p[0], up2);
		if (rec->n == 1)
			up1 = replace_by_constant_term(up1);
		return up1;
	}

	if (isl_upoly_is_cst(up1))
		return isl_upoly_sum_cst(up1, up2);

	rec1 = isl_upoly_as_rec(up1);
	rec2 = isl_upoly_as_rec(up2);
	if (!rec1 || !rec2)
		goto error;

	if (rec1->n < rec2->n)
		return isl_upoly_sum(up2, up1);

	up1 = isl_upoly_cow(up1);
	rec1 = isl_upoly_as_rec(up1);
	if (!rec1)
		goto error;

	for (i = rec2->n - 1; i >= 0; --i) {
		rec1->p[i] = isl_upoly_sum(rec1->p[i],
					    isl_upoly_copy(rec2->p[i]));
		if (!rec1->p[i])
			goto error;
		if (i == rec1->n - 1 && isl_upoly_is_zero(rec1->p[i])) {
			isl_upoly_free(rec1->p[i]);
			rec1->n--;
		}
	}

	if (rec1->n == 0)
		up1 = replace_by_zero(up1);
	else if (rec1->n == 1)
		up1 = replace_by_constant_term(up1);

	isl_upoly_free(up2);

	return up1;
error:
	isl_upoly_free(up1);
	isl_upoly_free(up2);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_cst_add_isl_int(
	__isl_take struct isl_upoly *up, isl_int v)
{
	struct isl_upoly_cst *cst;

	up = isl_upoly_cow(up);
	if (!up)
		return NULL;

	cst = isl_upoly_as_cst(up);

	isl_int_addmul(cst->n, cst->d, v);

	return up;
}

__isl_give struct isl_upoly *isl_upoly_add_isl_int(
	__isl_take struct isl_upoly *up, isl_int v)
{
	struct isl_upoly_rec *rec;

	if (!up)
		return NULL;

	if (isl_upoly_is_cst(up))
		return isl_upoly_cst_add_isl_int(up, v);

	up = isl_upoly_cow(up);
	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	rec->p[0] = isl_upoly_add_isl_int(rec->p[0], v);
	if (!rec->p[0])
		goto error;

	return up;
error:
	isl_upoly_free(up);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_cst_mul_isl_int(
	__isl_take struct isl_upoly *up, isl_int v)
{
	struct isl_upoly_cst *cst;

	if (isl_upoly_is_zero(up))
		return up;

	up = isl_upoly_cow(up);
	if (!up)
		return NULL;

	cst = isl_upoly_as_cst(up);

	isl_int_mul(cst->n, cst->n, v);

	return up;
}

__isl_give struct isl_upoly *isl_upoly_mul_isl_int(
	__isl_take struct isl_upoly *up, isl_int v)
{
	int i;
	struct isl_upoly_rec *rec;

	if (!up)
		return NULL;

	if (isl_upoly_is_cst(up))
		return isl_upoly_cst_mul_isl_int(up, v);

	up = isl_upoly_cow(up);
	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		rec->p[i] = isl_upoly_mul_isl_int(rec->p[i], v);
		if (!rec->p[i])
			goto error;
	}

	return up;
error:
	isl_upoly_free(up);
	return NULL;
}

/* Multiply the constant polynomial "up" by "v".
 */
static __isl_give struct isl_upoly *isl_upoly_cst_scale_val(
	__isl_take struct isl_upoly *up, __isl_keep isl_val *v)
{
	struct isl_upoly_cst *cst;

	if (isl_upoly_is_zero(up))
		return up;

	up = isl_upoly_cow(up);
	if (!up)
		return NULL;

	cst = isl_upoly_as_cst(up);

	isl_int_mul(cst->n, cst->n, v->n);
	isl_int_mul(cst->d, cst->d, v->d);
	isl_upoly_cst_reduce(cst);

	return up;
}

/* Multiply the polynomial "up" by "v".
 */
static __isl_give struct isl_upoly *isl_upoly_scale_val(
	__isl_take struct isl_upoly *up, __isl_keep isl_val *v)
{
	int i;
	struct isl_upoly_rec *rec;

	if (!up)
		return NULL;

	if (isl_upoly_is_cst(up))
		return isl_upoly_cst_scale_val(up, v);

	up = isl_upoly_cow(up);
	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		rec->p[i] = isl_upoly_scale_val(rec->p[i], v);
		if (!rec->p[i])
			goto error;
	}

	return up;
error:
	isl_upoly_free(up);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_mul_cst(__isl_take struct isl_upoly *up1,
	__isl_take struct isl_upoly *up2)
{
	struct isl_upoly_cst *cst1;
	struct isl_upoly_cst *cst2;

	up1 = isl_upoly_cow(up1);
	if (!up1 || !up2)
		goto error;

	cst1 = isl_upoly_as_cst(up1);
	cst2 = isl_upoly_as_cst(up2);

	isl_int_mul(cst1->n, cst1->n, cst2->n);
	isl_int_mul(cst1->d, cst1->d, cst2->d);

	isl_upoly_cst_reduce(cst1);

	isl_upoly_free(up2);
	return up1;
error:
	isl_upoly_free(up1);
	isl_upoly_free(up2);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_mul_rec(__isl_take struct isl_upoly *up1,
	__isl_take struct isl_upoly *up2)
{
	struct isl_upoly_rec *rec1;
	struct isl_upoly_rec *rec2;
	struct isl_upoly_rec *res = NULL;
	int i, j;
	int size;

	rec1 = isl_upoly_as_rec(up1);
	rec2 = isl_upoly_as_rec(up2);
	if (!rec1 || !rec2)
		goto error;
	size = rec1->n + rec2->n - 1;
	res = isl_upoly_alloc_rec(up1->ctx, up1->var, size);
	if (!res)
		goto error;

	for (i = 0; i < rec1->n; ++i) {
		res->p[i] = isl_upoly_mul(isl_upoly_copy(rec2->p[0]),
					    isl_upoly_copy(rec1->p[i]));
		if (!res->p[i])
			goto error;
		res->n++;
	}
	for (; i < size; ++i) {
		res->p[i] = isl_upoly_zero(up1->ctx);
		if (!res->p[i])
			goto error;
		res->n++;
	}
	for (i = 0; i < rec1->n; ++i) {
		for (j = 1; j < rec2->n; ++j) {
			struct isl_upoly *up;
			up = isl_upoly_mul(isl_upoly_copy(rec2->p[j]),
					    isl_upoly_copy(rec1->p[i]));
			res->p[i + j] = isl_upoly_sum(res->p[i + j], up);
			if (!res->p[i + j])
				goto error;
		}
	}

	isl_upoly_free(up1);
	isl_upoly_free(up2);

	return &res->up;
error:
	isl_upoly_free(up1);
	isl_upoly_free(up2);
	isl_upoly_free(&res->up);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_mul(__isl_take struct isl_upoly *up1,
	__isl_take struct isl_upoly *up2)
{
	if (!up1 || !up2)
		goto error;

	if (isl_upoly_is_nan(up1)) {
		isl_upoly_free(up2);
		return up1;
	}

	if (isl_upoly_is_nan(up2)) {
		isl_upoly_free(up1);
		return up2;
	}

	if (isl_upoly_is_zero(up1)) {
		isl_upoly_free(up2);
		return up1;
	}

	if (isl_upoly_is_zero(up2)) {
		isl_upoly_free(up1);
		return up2;
	}

	if (isl_upoly_is_one(up1)) {
		isl_upoly_free(up1);
		return up2;
	}

	if (isl_upoly_is_one(up2)) {
		isl_upoly_free(up2);
		return up1;
	}

	if (up1->var < up2->var)
		return isl_upoly_mul(up2, up1);

	if (up2->var < up1->var) {
		int i;
		struct isl_upoly_rec *rec;
		if (isl_upoly_is_infty(up2) || isl_upoly_is_neginfty(up2)) {
			isl_ctx *ctx = up1->ctx;
			isl_upoly_free(up1);
			isl_upoly_free(up2);
			return isl_upoly_nan(ctx);
		}
		up1 = isl_upoly_cow(up1);
		rec = isl_upoly_as_rec(up1);
		if (!rec)
			goto error;

		for (i = 0; i < rec->n; ++i) {
			rec->p[i] = isl_upoly_mul(rec->p[i],
						    isl_upoly_copy(up2));
			if (!rec->p[i])
				goto error;
		}
		isl_upoly_free(up2);
		return up1;
	}

	if (isl_upoly_is_cst(up1))
		return isl_upoly_mul_cst(up1, up2);

	return isl_upoly_mul_rec(up1, up2);
error:
	isl_upoly_free(up1);
	isl_upoly_free(up2);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_pow(__isl_take struct isl_upoly *up,
	unsigned power)
{
	struct isl_upoly *res;

	if (!up)
		return NULL;
	if (power == 1)
		return up;

	if (power % 2)
		res = isl_upoly_copy(up);
	else
		res = isl_upoly_one(up->ctx);

	while (power >>= 1) {
		up = isl_upoly_mul(up, isl_upoly_copy(up));
		if (power % 2)
			res = isl_upoly_mul(res, isl_upoly_copy(up));
	}

	isl_upoly_free(up);
	return res;
}

__isl_give isl_qpolynomial *isl_qpolynomial_alloc(__isl_take isl_space *dim,
	unsigned n_div, __isl_take struct isl_upoly *up)
{
	struct isl_qpolynomial *qp = NULL;
	unsigned total;

	if (!dim || !up)
		goto error;

	if (!isl_space_is_set(dim))
		isl_die(isl_space_get_ctx(dim), isl_error_invalid,
			"domain of polynomial should be a set", goto error);

	total = isl_space_dim(dim, isl_dim_all);

	qp = isl_calloc_type(dim->ctx, struct isl_qpolynomial);
	if (!qp)
		goto error;

	qp->ref = 1;
	qp->div = isl_mat_alloc(dim->ctx, n_div, 1 + 1 + total + n_div);
	if (!qp->div)
		goto error;

	qp->dim = dim;
	qp->upoly = up;

	return qp;
error:
	isl_space_free(dim);
	isl_upoly_free(up);
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_copy(__isl_keep isl_qpolynomial *qp)
{
	if (!qp)
		return NULL;

	qp->ref++;
	return qp;
}

__isl_give isl_qpolynomial *isl_qpolynomial_dup(__isl_keep isl_qpolynomial *qp)
{
	struct isl_qpolynomial *dup;

	if (!qp)
		return NULL;

	dup = isl_qpolynomial_alloc(isl_space_copy(qp->dim), qp->div->n_row,
				    isl_upoly_copy(qp->upoly));
	if (!dup)
		return NULL;
	isl_mat_free(dup->div);
	dup->div = isl_mat_copy(qp->div);
	if (!dup->div)
		goto error;

	return dup;
error:
	isl_qpolynomial_free(dup);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_cow(__isl_take isl_qpolynomial *qp)
{
	if (!qp)
		return NULL;

	if (qp->ref == 1)
		return qp;
	qp->ref--;
	return isl_qpolynomial_dup(qp);
}

__isl_null isl_qpolynomial *isl_qpolynomial_free(
	__isl_take isl_qpolynomial *qp)
{
	if (!qp)
		return NULL;

	if (--qp->ref > 0)
		return NULL;

	isl_space_free(qp->dim);
	isl_mat_free(qp->div);
	isl_upoly_free(qp->upoly);

	free(qp);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_var_pow(isl_ctx *ctx, int pos, int power)
{
	int i;
	struct isl_upoly_rec *rec;
	struct isl_upoly_cst *cst;

	rec = isl_upoly_alloc_rec(ctx, pos, 1 + power);
	if (!rec)
		return NULL;
	for (i = 0; i < 1 + power; ++i) {
		rec->p[i] = isl_upoly_zero(ctx);
		if (!rec->p[i])
			goto error;
		rec->n++;
	}
	cst = isl_upoly_as_cst(rec->p[power]);
	isl_int_set_si(cst->n, 1);

	return &rec->up;
error:
	isl_upoly_free(&rec->up);
	return NULL;
}

/* r array maps original positions to new positions.
 */
static __isl_give struct isl_upoly *reorder(__isl_take struct isl_upoly *up,
	int *r)
{
	int i;
	struct isl_upoly_rec *rec;
	struct isl_upoly *base;
	struct isl_upoly *res;

	if (isl_upoly_is_cst(up))
		return up;

	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	isl_assert(up->ctx, rec->n >= 1, goto error);

	base = isl_upoly_var_pow(up->ctx, r[up->var], 1);
	res = reorder(isl_upoly_copy(rec->p[rec->n - 1]), r);

	for (i = rec->n - 2; i >= 0; --i) {
		res = isl_upoly_mul(res, isl_upoly_copy(base));
		res = isl_upoly_sum(res, reorder(isl_upoly_copy(rec->p[i]), r));
	}

	isl_upoly_free(base);
	isl_upoly_free(up);

	return res;
error:
	isl_upoly_free(up);
	return NULL;
}

static int compatible_divs(__isl_keep isl_mat *div1, __isl_keep isl_mat *div2)
{
	int n_row, n_col;
	int equal;

	isl_assert(div1->ctx, div1->n_row >= div2->n_row &&
				div1->n_col >= div2->n_col, return -1);

	if (div1->n_row == div2->n_row)
		return isl_mat_is_equal(div1, div2);

	n_row = div1->n_row;
	n_col = div1->n_col;
	div1->n_row = div2->n_row;
	div1->n_col = div2->n_col;

	equal = isl_mat_is_equal(div1, div2);

	div1->n_row = n_row;
	div1->n_col = n_col;

	return equal;
}

static int cmp_row(__isl_keep isl_mat *div, int i, int j)
{
	int li, lj;

	li = isl_seq_last_non_zero(div->row[i], div->n_col);
	lj = isl_seq_last_non_zero(div->row[j], div->n_col);

	if (li != lj)
		return li - lj;

	return isl_seq_cmp(div->row[i], div->row[j], div->n_col);
}

struct isl_div_sort_info {
	isl_mat	*div;
	int	 row;
};

static int div_sort_cmp(const void *p1, const void *p2)
{
	const struct isl_div_sort_info *i1, *i2;
	i1 = (const struct isl_div_sort_info *) p1;
	i2 = (const struct isl_div_sort_info *) p2;

	return cmp_row(i1->div, i1->row, i2->row);
}

/* Sort divs and remove duplicates.
 */
static __isl_give isl_qpolynomial *sort_divs(__isl_take isl_qpolynomial *qp)
{
	int i;
	int skip;
	int len;
	struct isl_div_sort_info *array = NULL;
	int *pos = NULL, *at = NULL;
	int *reordering = NULL;
	unsigned div_pos;

	if (!qp)
		return NULL;
	if (qp->div->n_row <= 1)
		return qp;

	div_pos = isl_space_dim(qp->dim, isl_dim_all);

	array = isl_alloc_array(qp->div->ctx, struct isl_div_sort_info,
				qp->div->n_row);
	pos = isl_alloc_array(qp->div->ctx, int, qp->div->n_row);
	at = isl_alloc_array(qp->div->ctx, int, qp->div->n_row);
	len = qp->div->n_col - 2;
	reordering = isl_alloc_array(qp->div->ctx, int, len);
	if (!array || !pos || !at || !reordering)
		goto error;

	for (i = 0; i < qp->div->n_row; ++i) {
		array[i].div = qp->div;
		array[i].row = i;
		pos[i] = i;
		at[i] = i;
	}

	qsort(array, qp->div->n_row, sizeof(struct isl_div_sort_info),
		div_sort_cmp);

	for (i = 0; i < div_pos; ++i)
		reordering[i] = i;

	for (i = 0; i < qp->div->n_row; ++i) {
		if (pos[array[i].row] == i)
			continue;
		qp->div = isl_mat_swap_rows(qp->div, i, pos[array[i].row]);
		pos[at[i]] = pos[array[i].row];
		at[pos[array[i].row]] = at[i];
		at[i] = array[i].row;
		pos[array[i].row] = i;
	}

	skip = 0;
	for (i = 0; i < len - div_pos; ++i) {
		if (i > 0 &&
		    isl_seq_eq(qp->div->row[i - skip - 1],
			       qp->div->row[i - skip], qp->div->n_col)) {
			qp->div = isl_mat_drop_rows(qp->div, i - skip, 1);
			isl_mat_col_add(qp->div, 2 + div_pos + i - skip - 1,
						 2 + div_pos + i - skip);
			qp->div = isl_mat_drop_cols(qp->div,
						    2 + div_pos + i - skip, 1);
			skip++;
		}
		reordering[div_pos + array[i].row] = div_pos + i - skip;
	}

	qp->upoly = reorder(qp->upoly, reordering);

	if (!qp->upoly || !qp->div)
		goto error;

	free(at);
	free(pos);
	free(array);
	free(reordering);

	return qp;
error:
	free(at);
	free(pos);
	free(array);
	free(reordering);
	isl_qpolynomial_free(qp);
	return NULL;
}

static __isl_give struct isl_upoly *expand(__isl_take struct isl_upoly *up,
	int *exp, int first)
{
	int i;
	struct isl_upoly_rec *rec;

	if (isl_upoly_is_cst(up))
		return up;

	if (up->var < first)
		return up;

	if (exp[up->var - first] == up->var - first)
		return up;

	up = isl_upoly_cow(up);
	if (!up)
		goto error;

	up->var = exp[up->var - first] + first;

	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		rec->p[i] = expand(rec->p[i], exp, first);
		if (!rec->p[i])
			goto error;
	}

	return up;
error:
	isl_upoly_free(up);
	return NULL;
}

static __isl_give isl_qpolynomial *with_merged_divs(
	__isl_give isl_qpolynomial *(*fn)(__isl_take isl_qpolynomial *qp1,
					  __isl_take isl_qpolynomial *qp2),
	__isl_take isl_qpolynomial *qp1, __isl_take isl_qpolynomial *qp2)
{
	int *exp1 = NULL;
	int *exp2 = NULL;
	isl_mat *div = NULL;
	int n_div1, n_div2;

	qp1 = isl_qpolynomial_cow(qp1);
	qp2 = isl_qpolynomial_cow(qp2);

	if (!qp1 || !qp2)
		goto error;

	isl_assert(qp1->div->ctx, qp1->div->n_row >= qp2->div->n_row &&
				qp1->div->n_col >= qp2->div->n_col, goto error);

	n_div1 = qp1->div->n_row;
	n_div2 = qp2->div->n_row;
	exp1 = isl_alloc_array(qp1->div->ctx, int, n_div1);
	exp2 = isl_alloc_array(qp2->div->ctx, int, n_div2);
	if ((n_div1 && !exp1) || (n_div2 && !exp2))
		goto error;

	div = isl_merge_divs(qp1->div, qp2->div, exp1, exp2);
	if (!div)
		goto error;

	isl_mat_free(qp1->div);
	qp1->div = isl_mat_copy(div);
	isl_mat_free(qp2->div);
	qp2->div = isl_mat_copy(div);

	qp1->upoly = expand(qp1->upoly, exp1, div->n_col - div->n_row - 2);
	qp2->upoly = expand(qp2->upoly, exp2, div->n_col - div->n_row - 2);

	if (!qp1->upoly || !qp2->upoly)
		goto error;

	isl_mat_free(div);
	free(exp1);
	free(exp2);

	return fn(qp1, qp2);
error:
	isl_mat_free(div);
	free(exp1);
	free(exp2);
	isl_qpolynomial_free(qp1);
	isl_qpolynomial_free(qp2);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_add(__isl_take isl_qpolynomial *qp1,
	__isl_take isl_qpolynomial *qp2)
{
	qp1 = isl_qpolynomial_cow(qp1);

	if (!qp1 || !qp2)
		goto error;

	if (qp1->div->n_row < qp2->div->n_row)
		return isl_qpolynomial_add(qp2, qp1);

	isl_assert(qp1->dim->ctx, isl_space_is_equal(qp1->dim, qp2->dim), goto error);
	if (!compatible_divs(qp1->div, qp2->div))
		return with_merged_divs(isl_qpolynomial_add, qp1, qp2);

	qp1->upoly = isl_upoly_sum(qp1->upoly, isl_upoly_copy(qp2->upoly));
	if (!qp1->upoly)
		goto error;

	isl_qpolynomial_free(qp2);

	return qp1;
error:
	isl_qpolynomial_free(qp1);
	isl_qpolynomial_free(qp2);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_add_on_domain(
	__isl_keep isl_set *dom,
	__isl_take isl_qpolynomial *qp1,
	__isl_take isl_qpolynomial *qp2)
{
	qp1 = isl_qpolynomial_add(qp1, qp2);
	qp1 = isl_qpolynomial_gist(qp1, isl_set_copy(dom));
	return qp1;
}

__isl_give isl_qpolynomial *isl_qpolynomial_sub(__isl_take isl_qpolynomial *qp1,
	__isl_take isl_qpolynomial *qp2)
{
	return isl_qpolynomial_add(qp1, isl_qpolynomial_neg(qp2));
}

__isl_give isl_qpolynomial *isl_qpolynomial_add_isl_int(
	__isl_take isl_qpolynomial *qp, isl_int v)
{
	if (isl_int_is_zero(v))
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	qp->upoly = isl_upoly_add_isl_int(qp->upoly, v);
	if (!qp->upoly)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;

}

__isl_give isl_qpolynomial *isl_qpolynomial_neg(__isl_take isl_qpolynomial *qp)
{
	if (!qp)
		return NULL;

	return isl_qpolynomial_mul_isl_int(qp, qp->dim->ctx->negone);
}

__isl_give isl_qpolynomial *isl_qpolynomial_mul_isl_int(
	__isl_take isl_qpolynomial *qp, isl_int v)
{
	if (isl_int_is_one(v))
		return qp;

	if (qp && isl_int_is_zero(v)) {
		isl_qpolynomial *zero;
		zero = isl_qpolynomial_zero_on_domain(isl_space_copy(qp->dim));
		isl_qpolynomial_free(qp);
		return zero;
	}
	
	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	qp->upoly = isl_upoly_mul_isl_int(qp->upoly, v);
	if (!qp->upoly)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_scale(
	__isl_take isl_qpolynomial *qp, isl_int v)
{
	return isl_qpolynomial_mul_isl_int(qp, v);
}

/* Multiply "qp" by "v".
 */
__isl_give isl_qpolynomial *isl_qpolynomial_scale_val(
	__isl_take isl_qpolynomial *qp, __isl_take isl_val *v)
{
	if (!qp || !v)
		goto error;

	if (!isl_val_is_rat(v))
		isl_die(isl_qpolynomial_get_ctx(qp), isl_error_invalid,
			"expecting rational factor", goto error);

	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return qp;
	}

	if (isl_val_is_zero(v)) {
		isl_space *space;

		space = isl_qpolynomial_get_domain_space(qp);
		isl_qpolynomial_free(qp);
		isl_val_free(v);
		return isl_qpolynomial_zero_on_domain(space);
	}

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		goto error;

	qp->upoly = isl_upoly_scale_val(qp->upoly, v);
	if (!qp->upoly)
		qp = isl_qpolynomial_free(qp);

	isl_val_free(v);
	return qp;
error:
	isl_val_free(v);
	isl_qpolynomial_free(qp);
	return NULL;
}

/* Divide "qp" by "v".
 */
__isl_give isl_qpolynomial *isl_qpolynomial_scale_down_val(
	__isl_take isl_qpolynomial *qp, __isl_take isl_val *v)
{
	if (!qp || !v)
		goto error;

	if (!isl_val_is_rat(v))
		isl_die(isl_qpolynomial_get_ctx(qp), isl_error_invalid,
			"expecting rational factor", goto error);
	if (isl_val_is_zero(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"cannot scale down by zero", goto error);

	return isl_qpolynomial_scale_val(qp, isl_val_inv(v));
error:
	isl_val_free(v);
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_mul(__isl_take isl_qpolynomial *qp1,
	__isl_take isl_qpolynomial *qp2)
{
	qp1 = isl_qpolynomial_cow(qp1);

	if (!qp1 || !qp2)
		goto error;

	if (qp1->div->n_row < qp2->div->n_row)
		return isl_qpolynomial_mul(qp2, qp1);

	isl_assert(qp1->dim->ctx, isl_space_is_equal(qp1->dim, qp2->dim), goto error);
	if (!compatible_divs(qp1->div, qp2->div))
		return with_merged_divs(isl_qpolynomial_mul, qp1, qp2);

	qp1->upoly = isl_upoly_mul(qp1->upoly, isl_upoly_copy(qp2->upoly));
	if (!qp1->upoly)
		goto error;

	isl_qpolynomial_free(qp2);

	return qp1;
error:
	isl_qpolynomial_free(qp1);
	isl_qpolynomial_free(qp2);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_pow(__isl_take isl_qpolynomial *qp,
	unsigned power)
{
	qp = isl_qpolynomial_cow(qp);

	if (!qp)
		return NULL;

	qp->upoly = isl_upoly_pow(qp->upoly, power);
	if (!qp->upoly)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_pow(
	__isl_take isl_pw_qpolynomial *pwqp, unsigned power)
{
	int i;

	if (power == 1)
		return pwqp;

	pwqp = isl_pw_qpolynomial_cow(pwqp);
	if (!pwqp)
		return NULL;

	for (i = 0; i < pwqp->n; ++i) {
		pwqp->p[i].qp = isl_qpolynomial_pow(pwqp->p[i].qp, power);
		if (!pwqp->p[i].qp)
			return isl_pw_qpolynomial_free(pwqp);
	}

	return pwqp;
}

__isl_give isl_qpolynomial *isl_qpolynomial_zero_on_domain(
	__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	return isl_qpolynomial_alloc(dim, 0, isl_upoly_zero(dim->ctx));
}

__isl_give isl_qpolynomial *isl_qpolynomial_one_on_domain(
	__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	return isl_qpolynomial_alloc(dim, 0, isl_upoly_one(dim->ctx));
}

__isl_give isl_qpolynomial *isl_qpolynomial_infty_on_domain(
	__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	return isl_qpolynomial_alloc(dim, 0, isl_upoly_infty(dim->ctx));
}

__isl_give isl_qpolynomial *isl_qpolynomial_neginfty_on_domain(
	__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	return isl_qpolynomial_alloc(dim, 0, isl_upoly_neginfty(dim->ctx));
}

__isl_give isl_qpolynomial *isl_qpolynomial_nan_on_domain(
	__isl_take isl_space *dim)
{
	if (!dim)
		return NULL;
	return isl_qpolynomial_alloc(dim, 0, isl_upoly_nan(dim->ctx));
}

__isl_give isl_qpolynomial *isl_qpolynomial_cst_on_domain(
	__isl_take isl_space *dim,
	isl_int v)
{
	struct isl_qpolynomial *qp;
	struct isl_upoly_cst *cst;

	if (!dim)
		return NULL;

	qp = isl_qpolynomial_alloc(dim, 0, isl_upoly_zero(dim->ctx));
	if (!qp)
		return NULL;

	cst = isl_upoly_as_cst(qp->upoly);
	isl_int_set(cst->n, v);

	return qp;
}

int isl_qpolynomial_is_cst(__isl_keep isl_qpolynomial *qp,
	isl_int *n, isl_int *d)
{
	struct isl_upoly_cst *cst;

	if (!qp)
		return -1;

	if (!isl_upoly_is_cst(qp->upoly))
		return 0;

	cst = isl_upoly_as_cst(qp->upoly);
	if (!cst)
		return -1;

	if (n)
		isl_int_set(*n, cst->n);
	if (d)
		isl_int_set(*d, cst->d);

	return 1;
}

/* Return the constant term of "up".
 */
static __isl_give isl_val *isl_upoly_get_constant_val(
	__isl_keep struct isl_upoly *up)
{
	struct isl_upoly_cst *cst;

	if (!up)
		return NULL;

	while (!isl_upoly_is_cst(up)) {
		struct isl_upoly_rec *rec;

		rec = isl_upoly_as_rec(up);
		if (!rec)
			return NULL;
		up = rec->p[0];
	}

	cst = isl_upoly_as_cst(up);
	if (!cst)
		return NULL;
	return isl_val_rat_from_isl_int(cst->up.ctx, cst->n, cst->d);
}

/* Return the constant term of "qp".
 */
__isl_give isl_val *isl_qpolynomial_get_constant_val(
	__isl_keep isl_qpolynomial *qp)
{
	if (!qp)
		return NULL;

	return isl_upoly_get_constant_val(qp->upoly);
}

int isl_upoly_is_affine(__isl_keep struct isl_upoly *up)
{
	int is_cst;
	struct isl_upoly_rec *rec;

	if (!up)
		return -1;

	if (up->var < 0)
		return 1;

	rec = isl_upoly_as_rec(up);
	if (!rec)
		return -1;

	if (rec->n > 2)
		return 0;

	isl_assert(up->ctx, rec->n > 1, return -1);

	is_cst = isl_upoly_is_cst(rec->p[1]);
	if (is_cst < 0)
		return -1;
	if (!is_cst)
		return 0;

	return isl_upoly_is_affine(rec->p[0]);
}

int isl_qpolynomial_is_affine(__isl_keep isl_qpolynomial *qp)
{
	if (!qp)
		return -1;

	if (qp->div->n_row > 0)
		return 0;

	return isl_upoly_is_affine(qp->upoly);
}

static void update_coeff(__isl_keep isl_vec *aff,
	__isl_keep struct isl_upoly_cst *cst, int pos)
{
	isl_int gcd;
	isl_int f;

	if (isl_int_is_zero(cst->n))
		return;

	isl_int_init(gcd);
	isl_int_init(f);
	isl_int_gcd(gcd, cst->d, aff->el[0]);
	isl_int_divexact(f, cst->d, gcd);
	isl_int_divexact(gcd, aff->el[0], gcd);
	isl_seq_scale(aff->el, aff->el, f, aff->size);
	isl_int_mul(aff->el[1 + pos], gcd, cst->n);
	isl_int_clear(gcd);
	isl_int_clear(f);
}

int isl_upoly_update_affine(__isl_keep struct isl_upoly *up,
	__isl_keep isl_vec *aff)
{
	struct isl_upoly_cst *cst;
	struct isl_upoly_rec *rec;

	if (!up || !aff)
		return -1;

	if (up->var < 0) {
		struct isl_upoly_cst *cst;

		cst = isl_upoly_as_cst(up);
		if (!cst)
			return -1;
		update_coeff(aff, cst, 0);
		return 0;
	}

	rec = isl_upoly_as_rec(up);
	if (!rec)
		return -1;
	isl_assert(up->ctx, rec->n == 2, return -1);

	cst = isl_upoly_as_cst(rec->p[1]);
	if (!cst)
		return -1;
	update_coeff(aff, cst, 1 + up->var);

	return isl_upoly_update_affine(rec->p[0], aff);
}

__isl_give isl_vec *isl_qpolynomial_extract_affine(
	__isl_keep isl_qpolynomial *qp)
{
	isl_vec *aff;
	unsigned d;

	if (!qp)
		return NULL;

	d = isl_space_dim(qp->dim, isl_dim_all);
	aff = isl_vec_alloc(qp->div->ctx, 2 + d + qp->div->n_row);
	if (!aff)
		return NULL;

	isl_seq_clr(aff->el + 1, 1 + d + qp->div->n_row);
	isl_int_set_si(aff->el[0], 1);

	if (isl_upoly_update_affine(qp->upoly, aff) < 0)
		goto error;

	return aff;
error:
	isl_vec_free(aff);
	return NULL;
}

/* Compare two quasi-polynomials.
 *
 * Return -1 if "qp1" is "smaller" than "qp2", 1 if "qp1" is "greater"
 * than "qp2" and 0 if they are equal.
 */
int isl_qpolynomial_plain_cmp(__isl_keep isl_qpolynomial *qp1,
	__isl_keep isl_qpolynomial *qp2)
{
	int cmp;

	if (qp1 == qp2)
		return 0;
	if (!qp1)
		return -1;
	if (!qp2)
		return 1;

	cmp = isl_space_cmp(qp1->dim, qp2->dim);
	if (cmp != 0)
		return cmp;

	cmp = isl_local_cmp(qp1->div, qp2->div);
	if (cmp != 0)
		return cmp;

	return isl_upoly_plain_cmp(qp1->upoly, qp2->upoly);
}

/* Is "qp1" obviously equal to "qp2"?
 *
 * NaN is not equal to anything, not even to another NaN.
 */
isl_bool isl_qpolynomial_plain_is_equal(__isl_keep isl_qpolynomial *qp1,
	__isl_keep isl_qpolynomial *qp2)
{
	isl_bool equal;

	if (!qp1 || !qp2)
		return isl_bool_error;

	if (isl_qpolynomial_is_nan(qp1) || isl_qpolynomial_is_nan(qp2))
		return isl_bool_false;

	equal = isl_space_is_equal(qp1->dim, qp2->dim);
	if (equal < 0 || !equal)
		return equal;

	equal = isl_mat_is_equal(qp1->div, qp2->div);
	if (equal < 0 || !equal)
		return equal;

	return isl_upoly_is_equal(qp1->upoly, qp2->upoly);
}

static void upoly_update_den(__isl_keep struct isl_upoly *up, isl_int *d)
{
	int i;
	struct isl_upoly_rec *rec;

	if (isl_upoly_is_cst(up)) {
		struct isl_upoly_cst *cst;
		cst = isl_upoly_as_cst(up);
		if (!cst)
			return;
		isl_int_lcm(*d, *d, cst->d);
		return;
	}

	rec = isl_upoly_as_rec(up);
	if (!rec)
		return;

	for (i = 0; i < rec->n; ++i)
		upoly_update_den(rec->p[i], d);
}

void isl_qpolynomial_get_den(__isl_keep isl_qpolynomial *qp, isl_int *d)
{
	isl_int_set_si(*d, 1);
	if (!qp)
		return;
	upoly_update_den(qp->upoly, d);
}

__isl_give isl_qpolynomial *isl_qpolynomial_var_pow_on_domain(
	__isl_take isl_space *dim, int pos, int power)
{
	struct isl_ctx *ctx;

	if (!dim)
		return NULL;

	ctx = dim->ctx;

	return isl_qpolynomial_alloc(dim, 0, isl_upoly_var_pow(ctx, pos, power));
}

__isl_give isl_qpolynomial *isl_qpolynomial_var_on_domain(__isl_take isl_space *dim,
	enum isl_dim_type type, unsigned pos)
{
	if (!dim)
		return NULL;

	isl_assert(dim->ctx, isl_space_dim(dim, isl_dim_in) == 0, goto error);
	isl_assert(dim->ctx, pos < isl_space_dim(dim, type), goto error);

	if (type == isl_dim_set)
		pos += isl_space_dim(dim, isl_dim_param);

	return isl_qpolynomial_var_pow_on_domain(dim, pos, 1);
error:
	isl_space_free(dim);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_subs(__isl_take struct isl_upoly *up,
	unsigned first, unsigned n, __isl_keep struct isl_upoly **subs)
{
	int i;
	struct isl_upoly_rec *rec;
	struct isl_upoly *base, *res;

	if (!up)
		return NULL;

	if (isl_upoly_is_cst(up))
		return up;

	if (up->var < first)
		return up;

	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	isl_assert(up->ctx, rec->n >= 1, goto error);

	if (up->var >= first + n)
		base = isl_upoly_var_pow(up->ctx, up->var, 1);
	else
		base = isl_upoly_copy(subs[up->var - first]);

	res = isl_upoly_subs(isl_upoly_copy(rec->p[rec->n - 1]), first, n, subs);
	for (i = rec->n - 2; i >= 0; --i) {
		struct isl_upoly *t;
		t = isl_upoly_subs(isl_upoly_copy(rec->p[i]), first, n, subs);
		res = isl_upoly_mul(res, isl_upoly_copy(base));
		res = isl_upoly_sum(res, t);
	}

	isl_upoly_free(base);
	isl_upoly_free(up);
				
	return res;
error:
	isl_upoly_free(up);
	return NULL;
}	

__isl_give struct isl_upoly *isl_upoly_from_affine(isl_ctx *ctx, isl_int *f,
	isl_int denom, unsigned len)
{
	int i;
	struct isl_upoly *up;

	isl_assert(ctx, len >= 1, return NULL);

	up = isl_upoly_rat_cst(ctx, f[0], denom);
	for (i = 0; i < len - 1; ++i) {
		struct isl_upoly *t;
		struct isl_upoly *c;

		if (isl_int_is_zero(f[1 + i]))
			continue;

		c = isl_upoly_rat_cst(ctx, f[1 + i], denom);
		t = isl_upoly_var_pow(ctx, i, 1);
		t = isl_upoly_mul(c, t);
		up = isl_upoly_sum(up, t);
	}

	return up;
}

/* Remove common factor of non-constant terms and denominator.
 */
static void normalize_div(__isl_keep isl_qpolynomial *qp, int div)
{
	isl_ctx *ctx = qp->div->ctx;
	unsigned total = qp->div->n_col - 2;

	isl_seq_gcd(qp->div->row[div] + 2, total, &ctx->normalize_gcd);
	isl_int_gcd(ctx->normalize_gcd,
		    ctx->normalize_gcd, qp->div->row[div][0]);
	if (isl_int_is_one(ctx->normalize_gcd))
		return;

	isl_seq_scale_down(qp->div->row[div] + 2, qp->div->row[div] + 2,
			    ctx->normalize_gcd, total);
	isl_int_divexact(qp->div->row[div][0], qp->div->row[div][0],
			    ctx->normalize_gcd);
	isl_int_fdiv_q(qp->div->row[div][1], qp->div->row[div][1],
			    ctx->normalize_gcd);
}

/* Replace the integer division identified by "div" by the polynomial "s".
 * The integer division is assumed not to appear in the definition
 * of any other integer divisions.
 */
static __isl_give isl_qpolynomial *substitute_div(
	__isl_take isl_qpolynomial *qp,
	int div, __isl_take struct isl_upoly *s)
{
	int i;
	int total;
	int *reordering;

	if (!qp || !s)
		goto error;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		goto error;

	total = isl_space_dim(qp->dim, isl_dim_all);
	qp->upoly = isl_upoly_subs(qp->upoly, total + div, 1, &s);
	if (!qp->upoly)
		goto error;

	reordering = isl_alloc_array(qp->dim->ctx, int, total + qp->div->n_row);
	if (!reordering)
		goto error;
	for (i = 0; i < total + div; ++i)
		reordering[i] = i;
	for (i = total + div + 1; i < total + qp->div->n_row; ++i)
		reordering[i] = i - 1;
	qp->div = isl_mat_drop_rows(qp->div, div, 1);
	qp->div = isl_mat_drop_cols(qp->div, 2 + total + div, 1);
	qp->upoly = reorder(qp->upoly, reordering);
	free(reordering);

	if (!qp->upoly || !qp->div)
		goto error;

	isl_upoly_free(s);
	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_upoly_free(s);
	return NULL;
}

/* Replace all integer divisions [e/d] that turn out to not actually be integer
 * divisions because d is equal to 1 by their definition, i.e., e.
 */
static __isl_give isl_qpolynomial *substitute_non_divs(
	__isl_take isl_qpolynomial *qp)
{
	int i, j;
	int total;
	struct isl_upoly *s;

	if (!qp)
		return NULL;

	total = isl_space_dim(qp->dim, isl_dim_all);
	for (i = 0; qp && i < qp->div->n_row; ++i) {
		if (!isl_int_is_one(qp->div->row[i][0]))
			continue;
		for (j = i + 1; j < qp->div->n_row; ++j) {
			if (isl_int_is_zero(qp->div->row[j][2 + total + i]))
				continue;
			isl_seq_combine(qp->div->row[j] + 1,
				qp->div->ctx->one, qp->div->row[j] + 1,
				qp->div->row[j][2 + total + i],
				qp->div->row[i] + 1, 1 + total + i);
			isl_int_set_si(qp->div->row[j][2 + total + i], 0);
			normalize_div(qp, j);
		}
		s = isl_upoly_from_affine(qp->dim->ctx, qp->div->row[i] + 1,
					qp->div->row[i][0], qp->div->n_col - 1);
		qp = substitute_div(qp, i, s);
		--i;
	}

	return qp;
}

/* Reduce the coefficients of div "div" to lie in the interval [0, d-1],
 * with d the denominator.  When replacing the coefficient e of x by
 * d * frac(e/d) = e - d * floor(e/d), we are subtracting d * floor(e/d) * x
 * inside the division, so we need to add floor(e/d) * x outside.
 * That is, we replace q by q' + floor(e/d) * x and we therefore need
 * to adjust the coefficient of x in each later div that depends on the
 * current div "div" and also in the affine expressions in the rows of "mat"
 * (if they too depend on "div").
 */
static void reduce_div(__isl_keep isl_qpolynomial *qp, int div,
	__isl_keep isl_mat **mat)
{
	int i, j;
	isl_int v;
	unsigned total = qp->div->n_col - qp->div->n_row - 2;

	isl_int_init(v);
	for (i = 0; i < 1 + total + div; ++i) {
		if (isl_int_is_nonneg(qp->div->row[div][1 + i]) &&
		    isl_int_lt(qp->div->row[div][1 + i], qp->div->row[div][0]))
			continue;
		isl_int_fdiv_q(v, qp->div->row[div][1 + i], qp->div->row[div][0]);
		isl_int_fdiv_r(qp->div->row[div][1 + i],
				qp->div->row[div][1 + i], qp->div->row[div][0]);
		*mat = isl_mat_col_addmul(*mat, i, v, 1 + total + div);
		for (j = div + 1; j < qp->div->n_row; ++j) {
			if (isl_int_is_zero(qp->div->row[j][2 + total + div]))
				continue;
			isl_int_addmul(qp->div->row[j][1 + i],
					v, qp->div->row[j][2 + total + div]);
		}
	}
	isl_int_clear(v);
}

/* Check if the last non-zero coefficient is bigger that half of the
 * denominator.  If so, we will invert the div to further reduce the number
 * of distinct divs that may appear.
 * If the last non-zero coefficient is exactly half the denominator,
 * then we continue looking for earlier coefficients that are bigger
 * than half the denominator.
 */
static int needs_invert(__isl_keep isl_mat *div, int row)
{
	int i;
	int cmp;

	for (i = div->n_col - 1; i >= 1; --i) {
		if (isl_int_is_zero(div->row[row][i]))
			continue;
		isl_int_mul_ui(div->row[row][i], div->row[row][i], 2);
		cmp = isl_int_cmp(div->row[row][i], div->row[row][0]);
		isl_int_divexact_ui(div->row[row][i], div->row[row][i], 2);
		if (cmp)
			return cmp > 0;
		if (i == 1)
			return 1;
	}

	return 0;
}

/* Replace div "div" q = [e/d] by -[(-e+(d-1))/d].
 * We only invert the coefficients of e (and the coefficient of q in
 * later divs and in the rows of "mat").  After calling this function, the
 * coefficients of e should be reduced again.
 */
static void invert_div(__isl_keep isl_qpolynomial *qp, int div,
	__isl_keep isl_mat **mat)
{
	unsigned total = qp->div->n_col - qp->div->n_row - 2;

	isl_seq_neg(qp->div->row[div] + 1,
		    qp->div->row[div] + 1, qp->div->n_col - 1);
	isl_int_sub_ui(qp->div->row[div][1], qp->div->row[div][1], 1);
	isl_int_add(qp->div->row[div][1],
		    qp->div->row[div][1], qp->div->row[div][0]);
	*mat = isl_mat_col_neg(*mat, 1 + total + div);
	isl_mat_col_mul(qp->div, 2 + total + div,
			qp->div->ctx->negone, 2 + total + div);
}

/* Reduce all divs of "qp" to have coefficients
 * in the interval [0, d-1], with d the denominator and such that the
 * last non-zero coefficient that is not equal to d/2 is smaller than d/2.
 * The modifications to the integer divisions need to be reflected
 * in the factors of the polynomial that refer to the original
 * integer divisions.  To this end, the modifications are collected
 * as a set of affine expressions and then plugged into the polynomial.
 *
 * After the reduction, some divs may have become redundant or identical,
 * so we call substitute_non_divs and sort_divs.  If these functions
 * eliminate divs or merge two or more divs into one, the coefficients
 * of the enclosing divs may have to be reduced again, so we call
 * ourselves recursively if the number of divs decreases.
 */
static __isl_give isl_qpolynomial *reduce_divs(__isl_take isl_qpolynomial *qp)
{
	int i;
	isl_ctx *ctx;
	isl_mat *mat;
	struct isl_upoly **s;
	unsigned o_div, n_div, total;

	if (!qp)
		return NULL;

	total = isl_qpolynomial_domain_dim(qp, isl_dim_all);
	n_div = isl_qpolynomial_domain_dim(qp, isl_dim_div);
	o_div = isl_qpolynomial_domain_offset(qp, isl_dim_div);
	ctx = isl_qpolynomial_get_ctx(qp);
	mat = isl_mat_zero(ctx, n_div, 1 + total);

	for (i = 0; i < n_div; ++i)
		mat = isl_mat_set_element_si(mat, i, o_div + i, 1);

	for (i = 0; i < qp->div->n_row; ++i) {
		normalize_div(qp, i);
		reduce_div(qp, i, &mat);
		if (needs_invert(qp->div, i)) {
			invert_div(qp, i, &mat);
			reduce_div(qp, i, &mat);
		}
	}
	if (!mat)
		goto error;

	s = isl_alloc_array(ctx, struct isl_upoly *, n_div);
	if (n_div && !s)
		goto error;
	for (i = 0; i < n_div; ++i)
		s[i] = isl_upoly_from_affine(ctx, mat->row[i], ctx->one,
					    1 + total);
	qp->upoly = isl_upoly_subs(qp->upoly, o_div - 1, n_div, s);
	for (i = 0; i < n_div; ++i)
		isl_upoly_free(s[i]);
	free(s);
	if (!qp->upoly)
		goto error;

	isl_mat_free(mat);

	qp = substitute_non_divs(qp);
	qp = sort_divs(qp);
	if (qp && isl_qpolynomial_domain_dim(qp, isl_dim_div) < n_div)
		return reduce_divs(qp);

	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_mat_free(mat);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_rat_cst_on_domain(
	__isl_take isl_space *dim, const isl_int n, const isl_int d)
{
	struct isl_qpolynomial *qp;
	struct isl_upoly_cst *cst;

	if (!dim)
		return NULL;

	qp = isl_qpolynomial_alloc(dim, 0, isl_upoly_zero(dim->ctx));
	if (!qp)
		return NULL;

	cst = isl_upoly_as_cst(qp->upoly);
	isl_int_set(cst->n, n);
	isl_int_set(cst->d, d);

	return qp;
}

/* Return an isl_qpolynomial that is equal to "val" on domain space "domain".
 */
__isl_give isl_qpolynomial *isl_qpolynomial_val_on_domain(
	__isl_take isl_space *domain, __isl_take isl_val *val)
{
	isl_qpolynomial *qp;
	struct isl_upoly_cst *cst;

	if (!domain || !val)
		goto error;

	qp = isl_qpolynomial_alloc(isl_space_copy(domain), 0,
					isl_upoly_zero(domain->ctx));
	if (!qp)
		goto error;

	cst = isl_upoly_as_cst(qp->upoly);
	isl_int_set(cst->n, val->n);
	isl_int_set(cst->d, val->d);

	isl_space_free(domain);
	isl_val_free(val);
	return qp;
error:
	isl_space_free(domain);
	isl_val_free(val);
	return NULL;
}

static int up_set_active(__isl_keep struct isl_upoly *up, int *active, int d)
{
	struct isl_upoly_rec *rec;
	int i;

	if (!up)
		return -1;

	if (isl_upoly_is_cst(up))
		return 0;

	if (up->var < d)
		active[up->var] = 1;

	rec = isl_upoly_as_rec(up);
	for (i = 0; i < rec->n; ++i)
		if (up_set_active(rec->p[i], active, d) < 0)
			return -1;

	return 0;
}

static int set_active(__isl_keep isl_qpolynomial *qp, int *active)
{
	int i, j;
	int d = isl_space_dim(qp->dim, isl_dim_all);

	if (!qp || !active)
		return -1;

	for (i = 0; i < d; ++i)
		for (j = 0; j < qp->div->n_row; ++j) {
			if (isl_int_is_zero(qp->div->row[j][2 + i]))
				continue;
			active[i] = 1;
			break;
		}

	return up_set_active(qp->upoly, active, d);
}

isl_bool isl_qpolynomial_involves_dims(__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	int *active = NULL;
	isl_bool involves = isl_bool_false;

	if (!qp)
		return isl_bool_error;
	if (n == 0)
		return isl_bool_false;

	isl_assert(qp->dim->ctx,
		    first + n <= isl_qpolynomial_dim(qp, type),
		    return isl_bool_error);
	isl_assert(qp->dim->ctx, type == isl_dim_param ||
				 type == isl_dim_in, return isl_bool_error);

	active = isl_calloc_array(qp->dim->ctx, int,
					isl_space_dim(qp->dim, isl_dim_all));
	if (set_active(qp, active) < 0)
		goto error;

	if (type == isl_dim_in)
		first += isl_space_dim(qp->dim, isl_dim_param);
	for (i = 0; i < n; ++i)
		if (active[first + i]) {
			involves = isl_bool_true;
			break;
		}

	free(active);

	return involves;
error:
	free(active);
	return isl_bool_error;
}

/* Remove divs that do not appear in the quasi-polynomial, nor in any
 * of the divs that do appear in the quasi-polynomial.
 */
static __isl_give isl_qpolynomial *remove_redundant_divs(
	__isl_take isl_qpolynomial *qp)
{
	int i, j;
	int d;
	int len;
	int skip;
	int *active = NULL;
	int *reordering = NULL;
	int redundant = 0;
	int n_div;
	isl_ctx *ctx;

	if (!qp)
		return NULL;
	if (qp->div->n_row == 0)
		return qp;

	d = isl_space_dim(qp->dim, isl_dim_all);
	len = qp->div->n_col - 2;
	ctx = isl_qpolynomial_get_ctx(qp);
	active = isl_calloc_array(ctx, int, len);
	if (!active)
		goto error;

	if (up_set_active(qp->upoly, active, len) < 0)
		goto error;

	for (i = qp->div->n_row - 1; i >= 0; --i) {
		if (!active[d + i]) {
			redundant = 1;
			continue;
		}
		for (j = 0; j < i; ++j) {
			if (isl_int_is_zero(qp->div->row[i][2 + d + j]))
				continue;
			active[d + j] = 1;
			break;
		}
	}

	if (!redundant) {
		free(active);
		return qp;
	}

	reordering = isl_alloc_array(qp->div->ctx, int, len);
	if (!reordering)
		goto error;

	for (i = 0; i < d; ++i)
		reordering[i] = i;

	skip = 0;
	n_div = qp->div->n_row;
	for (i = 0; i < n_div; ++i) {
		if (!active[d + i]) {
			qp->div = isl_mat_drop_rows(qp->div, i - skip, 1);
			qp->div = isl_mat_drop_cols(qp->div,
						    2 + d + i - skip, 1);
			skip++;
		}
		reordering[d + i] = d + i - skip;
	}

	qp->upoly = reorder(qp->upoly, reordering);

	if (!qp->upoly || !qp->div)
		goto error;

	free(active);
	free(reordering);

	return qp;
error:
	free(active);
	free(reordering);
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give struct isl_upoly *isl_upoly_drop(__isl_take struct isl_upoly *up,
	unsigned first, unsigned n)
{
	int i;
	struct isl_upoly_rec *rec;

	if (!up)
		return NULL;
	if (n == 0 || up->var < 0 || up->var < first)
		return up;
	if (up->var < first + n) {
		up = replace_by_constant_term(up);
		return isl_upoly_drop(up, first, n);
	}
	up = isl_upoly_cow(up);
	if (!up)
		return NULL;
	up->var -= n;
	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		rec->p[i] = isl_upoly_drop(rec->p[i], first, n);
		if (!rec->p[i])
			goto error;
	}

	return up;
error:
	isl_upoly_free(up);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_set_dim_name(
	__isl_take isl_qpolynomial *qp,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;
	if (type == isl_dim_out)
		isl_die(isl_qpolynomial_get_ctx(qp), isl_error_invalid,
			"cannot set name of output/set dimension",
			return isl_qpolynomial_free(qp));
	if (type == isl_dim_in)
		type = isl_dim_set;
	qp->dim = isl_space_set_dim_name(qp->dim, type, pos, s);
	if (!qp->dim)
		goto error;
	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_drop_dims(
	__isl_take isl_qpolynomial *qp,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	if (!qp)
		return NULL;
	if (type == isl_dim_out)
		isl_die(qp->dim->ctx, isl_error_invalid,
			"cannot drop output/set dimension",
			goto error);
	if (type == isl_dim_in)
		type = isl_dim_set;
	if (n == 0 && !isl_space_is_named_or_nested(qp->dim, type))
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	isl_assert(qp->dim->ctx, first + n <= isl_space_dim(qp->dim, type),
			goto error);
	isl_assert(qp->dim->ctx, type == isl_dim_param ||
				 type == isl_dim_set, goto error);

	qp->dim = isl_space_drop_dims(qp->dim, type, first, n);
	if (!qp->dim)
		goto error;

	if (type == isl_dim_set)
		first += isl_space_dim(qp->dim, isl_dim_param);

	qp->div = isl_mat_drop_cols(qp->div, 2 + first, n);
	if (!qp->div)
		goto error;

	qp->upoly = isl_upoly_drop(qp->upoly, first, n);
	if (!qp->upoly)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

/* Project the domain of the quasi-polynomial onto its parameter space.
 * The quasi-polynomial may not involve any of the domain dimensions.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_project_domain_on_params(
	__isl_take isl_qpolynomial *qp)
{
	isl_space *space;
	unsigned n;
	int involves;

	n = isl_qpolynomial_dim(qp, isl_dim_in);
	involves = isl_qpolynomial_involves_dims(qp, isl_dim_in, 0, n);
	if (involves < 0)
		return isl_qpolynomial_free(qp);
	if (involves)
		isl_die(isl_qpolynomial_get_ctx(qp), isl_error_invalid,
			"polynomial involves some of the domain dimensions",
			return isl_qpolynomial_free(qp));
	qp = isl_qpolynomial_drop_dims(qp, isl_dim_in, 0, n);
	space = isl_qpolynomial_get_domain_space(qp);
	space = isl_space_params(space);
	qp = isl_qpolynomial_reset_domain_space(qp, space);
	return qp;
}

static __isl_give isl_qpolynomial *isl_qpolynomial_substitute_equalities_lifted(
	__isl_take isl_qpolynomial *qp, __isl_take isl_basic_set *eq)
{
	int i, j, k;
	isl_int denom;
	unsigned total;
	unsigned n_div;
	struct isl_upoly *up;

	if (!eq)
		goto error;
	if (eq->n_eq == 0) {
		isl_basic_set_free(eq);
		return qp;
	}

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		goto error;
	qp->div = isl_mat_cow(qp->div);
	if (!qp->div)
		goto error;

	total = 1 + isl_space_dim(eq->dim, isl_dim_all);
	n_div = eq->n_div;
	isl_int_init(denom);
	for (i = 0; i < eq->n_eq; ++i) {
		j = isl_seq_last_non_zero(eq->eq[i], total + n_div);
		if (j < 0 || j == 0 || j >= total)
			continue;

		for (k = 0; k < qp->div->n_row; ++k) {
			if (isl_int_is_zero(qp->div->row[k][1 + j]))
				continue;
			isl_seq_elim(qp->div->row[k] + 1, eq->eq[i], j, total,
					&qp->div->row[k][0]);
			normalize_div(qp, k);
		}

		if (isl_int_is_pos(eq->eq[i][j]))
			isl_seq_neg(eq->eq[i], eq->eq[i], total);
		isl_int_abs(denom, eq->eq[i][j]);
		isl_int_set_si(eq->eq[i][j], 0);

		up = isl_upoly_from_affine(qp->dim->ctx,
						   eq->eq[i], denom, total);
		qp->upoly = isl_upoly_subs(qp->upoly, j - 1, 1, &up);
		isl_upoly_free(up);
	}
	isl_int_clear(denom);

	if (!qp->upoly)
		goto error;

	isl_basic_set_free(eq);

	qp = substitute_non_divs(qp);
	qp = sort_divs(qp);

	return qp;
error:
	isl_basic_set_free(eq);
	isl_qpolynomial_free(qp);
	return NULL;
}

/* Exploit the equalities in "eq" to simplify the quasi-polynomial.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_substitute_equalities(
	__isl_take isl_qpolynomial *qp, __isl_take isl_basic_set *eq)
{
	if (!qp || !eq)
		goto error;
	if (qp->div->n_row > 0)
		eq = isl_basic_set_add_dims(eq, isl_dim_set, qp->div->n_row);
	return isl_qpolynomial_substitute_equalities_lifted(qp, eq);
error:
	isl_basic_set_free(eq);
	isl_qpolynomial_free(qp);
	return NULL;
}

static __isl_give isl_basic_set *add_div_constraints(
	__isl_take isl_basic_set *bset, __isl_take isl_mat *div)
{
	int i;
	unsigned total;

	if (!bset || !div)
		goto error;

	bset = isl_basic_set_extend_constraints(bset, 0, 2 * div->n_row);
	if (!bset)
		goto error;
	total = isl_basic_set_total_dim(bset);
	for (i = 0; i < div->n_row; ++i)
		if (isl_basic_set_add_div_constraints_var(bset,
				    total - div->n_row + i, div->row[i]) < 0)
			goto error;

	isl_mat_free(div);
	return bset;
error:
	isl_mat_free(div);
	isl_basic_set_free(bset);
	return NULL;
}

/* Look for equalities among the variables shared by context and qp
 * and the integer divisions of qp, if any.
 * The equalities are then used to eliminate variables and/or integer
 * divisions from qp.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_gist(
	__isl_take isl_qpolynomial *qp, __isl_take isl_set *context)
{
	isl_basic_set *aff;

	if (!qp)
		goto error;
	if (qp->div->n_row > 0) {
		isl_basic_set *bset;
		context = isl_set_add_dims(context, isl_dim_set,
					    qp->div->n_row);
		bset = isl_basic_set_universe(isl_set_get_space(context));
		bset = add_div_constraints(bset, isl_mat_copy(qp->div));
		context = isl_set_intersect(context,
					    isl_set_from_basic_set(bset));
	}

	aff = isl_set_affine_hull(context);
	return isl_qpolynomial_substitute_equalities_lifted(qp, aff);
error:
	isl_qpolynomial_free(qp);
	isl_set_free(context);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_gist_params(
	__isl_take isl_qpolynomial *qp, __isl_take isl_set *context)
{
	isl_space *space = isl_qpolynomial_get_domain_space(qp);
	isl_set *dom_context = isl_set_universe(space);
	dom_context = isl_set_intersect_params(dom_context, context);
	return isl_qpolynomial_gist(qp, dom_context);
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_from_qpolynomial(
	__isl_take isl_qpolynomial *qp)
{
	isl_set *dom;

	if (!qp)
		return NULL;
	if (isl_qpolynomial_is_zero(qp)) {
		isl_space *dim = isl_qpolynomial_get_space(qp);
		isl_qpolynomial_free(qp);
		return isl_pw_qpolynomial_zero(dim);
	}

	dom = isl_set_universe(isl_qpolynomial_get_domain_space(qp));
	return isl_pw_qpolynomial_alloc(dom, qp);
}

#undef PW
#define PW isl_pw_qpolynomial
#undef EL
#define EL isl_qpolynomial
#undef EL_IS_ZERO
#define EL_IS_ZERO is_zero
#undef ZERO
#define ZERO zero
#undef IS_ZERO
#define IS_ZERO is_zero
#undef FIELD
#define FIELD qp
#undef DEFAULT_IS_ZERO
#define DEFAULT_IS_ZERO 1

#define NO_PULLBACK

#include <isl_pw_templ.c>

#undef UNION
#define UNION isl_union_pw_qpolynomial
#undef PART
#define PART isl_pw_qpolynomial
#undef PARTS
#define PARTS pw_qpolynomial

#include <isl_union_single.c>
#include <isl_union_eval.c>
#include <isl_union_neg.c>

int isl_pw_qpolynomial_is_one(__isl_keep isl_pw_qpolynomial *pwqp)
{
	if (!pwqp)
		return -1;

	if (pwqp->n != -1)
		return 0;

	if (!isl_set_plain_is_universe(pwqp->p[0].set))
		return 0;

	return isl_qpolynomial_is_one(pwqp->p[0].qp);
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_add(
	__isl_take isl_pw_qpolynomial *pwqp1,
	__isl_take isl_pw_qpolynomial *pwqp2)
{
	return isl_pw_qpolynomial_union_add_(pwqp1, pwqp2);
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_mul(
	__isl_take isl_pw_qpolynomial *pwqp1,
	__isl_take isl_pw_qpolynomial *pwqp2)
{
	int i, j, n;
	struct isl_pw_qpolynomial *res;

	if (!pwqp1 || !pwqp2)
		goto error;

	isl_assert(pwqp1->dim->ctx, isl_space_is_equal(pwqp1->dim, pwqp2->dim),
			goto error);

	if (isl_pw_qpolynomial_is_zero(pwqp1)) {
		isl_pw_qpolynomial_free(pwqp2);
		return pwqp1;
	}

	if (isl_pw_qpolynomial_is_zero(pwqp2)) {
		isl_pw_qpolynomial_free(pwqp1);
		return pwqp2;
	}

	if (isl_pw_qpolynomial_is_one(pwqp1)) {
		isl_pw_qpolynomial_free(pwqp1);
		return pwqp2;
	}

	if (isl_pw_qpolynomial_is_one(pwqp2)) {
		isl_pw_qpolynomial_free(pwqp2);
		return pwqp1;
	}

	n = pwqp1->n * pwqp2->n;
	res = isl_pw_qpolynomial_alloc_size(isl_space_copy(pwqp1->dim), n);

	for (i = 0; i < pwqp1->n; ++i) {
		for (j = 0; j < pwqp2->n; ++j) {
			struct isl_set *common;
			struct isl_qpolynomial *prod;
			common = isl_set_intersect(isl_set_copy(pwqp1->p[i].set),
						isl_set_copy(pwqp2->p[j].set));
			if (isl_set_plain_is_empty(common)) {
				isl_set_free(common);
				continue;
			}

			prod = isl_qpolynomial_mul(
				isl_qpolynomial_copy(pwqp1->p[i].qp),
				isl_qpolynomial_copy(pwqp2->p[j].qp));

			res = isl_pw_qpolynomial_add_piece(res, common, prod);
		}
	}

	isl_pw_qpolynomial_free(pwqp1);
	isl_pw_qpolynomial_free(pwqp2);

	return res;
error:
	isl_pw_qpolynomial_free(pwqp1);
	isl_pw_qpolynomial_free(pwqp2);
	return NULL;
}

__isl_give isl_val *isl_upoly_eval(__isl_take struct isl_upoly *up,
	__isl_take isl_vec *vec)
{
	int i;
	struct isl_upoly_rec *rec;
	isl_val *res;
	isl_val *base;

	if (isl_upoly_is_cst(up)) {
		isl_vec_free(vec);
		res = isl_upoly_get_constant_val(up);
		isl_upoly_free(up);
		return res;
	}

	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	isl_assert(up->ctx, rec->n >= 1, goto error);

	base = isl_val_rat_from_isl_int(up->ctx,
					vec->el[1 + up->var], vec->el[0]);

	res = isl_upoly_eval(isl_upoly_copy(rec->p[rec->n - 1]),
				isl_vec_copy(vec));

	for (i = rec->n - 2; i >= 0; --i) {
		res = isl_val_mul(res, isl_val_copy(base));
		res = isl_val_add(res,
			    isl_upoly_eval(isl_upoly_copy(rec->p[i]),
							    isl_vec_copy(vec)));
	}

	isl_val_free(base);
	isl_upoly_free(up);
	isl_vec_free(vec);
	return res;
error:
	isl_upoly_free(up);
	isl_vec_free(vec);
	return NULL;
}

/* Evaluate "qp" in the void point "pnt".
 * In particular, return the value NaN.
 */
static __isl_give isl_val *eval_void(__isl_take isl_qpolynomial *qp,
	__isl_take isl_point *pnt)
{
	isl_ctx *ctx;

	ctx = isl_point_get_ctx(pnt);
	isl_qpolynomial_free(qp);
	isl_point_free(pnt);
	return isl_val_nan(ctx);
}

__isl_give isl_val *isl_qpolynomial_eval(__isl_take isl_qpolynomial *qp,
	__isl_take isl_point *pnt)
{
	isl_bool is_void;
	isl_vec *ext;
	isl_val *v;

	if (!qp || !pnt)
		goto error;
	isl_assert(pnt->dim->ctx, isl_space_is_equal(pnt->dim, qp->dim), goto error);
	is_void = isl_point_is_void(pnt);
	if (is_void < 0)
		goto error;
	if (is_void)
		return eval_void(qp, pnt);

	if (qp->div->n_row == 0)
		ext = isl_vec_copy(pnt->vec);
	else {
		int i;
		unsigned dim = isl_space_dim(qp->dim, isl_dim_all);
		ext = isl_vec_alloc(qp->dim->ctx, 1 + dim + qp->div->n_row);
		if (!ext)
			goto error;

		isl_seq_cpy(ext->el, pnt->vec->el, pnt->vec->size);
		for (i = 0; i < qp->div->n_row; ++i) {
			isl_seq_inner_product(qp->div->row[i] + 1, ext->el,
						1 + dim + i, &ext->el[1+dim+i]);
			isl_int_fdiv_q(ext->el[1+dim+i], ext->el[1+dim+i],
					qp->div->row[i][0]);
		}
	}

	v = isl_upoly_eval(isl_upoly_copy(qp->upoly), ext);

	isl_qpolynomial_free(qp);
	isl_point_free(pnt);

	return v;
error:
	isl_qpolynomial_free(qp);
	isl_point_free(pnt);
	return NULL;
}

int isl_upoly_cmp(__isl_keep struct isl_upoly_cst *cst1,
	__isl_keep struct isl_upoly_cst *cst2)
{
	int cmp;
	isl_int t;
	isl_int_init(t);
	isl_int_mul(t, cst1->n, cst2->d);
	isl_int_submul(t, cst2->n, cst1->d);
	cmp = isl_int_sgn(t);
	isl_int_clear(t);
	return cmp;
}

__isl_give isl_qpolynomial *isl_qpolynomial_insert_dims(
	__isl_take isl_qpolynomial *qp, enum isl_dim_type type,
	unsigned first, unsigned n)
{
	unsigned total;
	unsigned g_pos;
	int *exp;

	if (!qp)
		return NULL;
	if (type == isl_dim_out)
		isl_die(qp->div->ctx, isl_error_invalid,
			"cannot insert output/set dimensions",
			goto error);
	if (type == isl_dim_in)
		type = isl_dim_set;
	if (n == 0 && !isl_space_is_named_or_nested(qp->dim, type))
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	isl_assert(qp->div->ctx, first <= isl_space_dim(qp->dim, type),
		    goto error);

	g_pos = pos(qp->dim, type) + first;

	qp->div = isl_mat_insert_zero_cols(qp->div, 2 + g_pos, n);
	if (!qp->div)
		goto error;

	total = qp->div->n_col - 2;
	if (total > g_pos) {
		int i;
		exp = isl_alloc_array(qp->div->ctx, int, total - g_pos);
		if (!exp)
			goto error;
		for (i = 0; i < total - g_pos; ++i)
			exp[i] = i + n;
		qp->upoly = expand(qp->upoly, exp, g_pos);
		free(exp);
		if (!qp->upoly)
			goto error;
	}

	qp->dim = isl_space_insert_dims(qp->dim, type, first, n);
	if (!qp->dim)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_add_dims(
	__isl_take isl_qpolynomial *qp, enum isl_dim_type type, unsigned n)
{
	unsigned pos;

	pos = isl_qpolynomial_dim(qp, type);

	return isl_qpolynomial_insert_dims(qp, type, pos, n);
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_add_dims(
	__isl_take isl_pw_qpolynomial *pwqp,
	enum isl_dim_type type, unsigned n)
{
	unsigned pos;

	pos = isl_pw_qpolynomial_dim(pwqp, type);

	return isl_pw_qpolynomial_insert_dims(pwqp, type, pos, n);
}

static int *reordering_move(isl_ctx *ctx,
	unsigned len, unsigned dst, unsigned src, unsigned n)
{
	int i;
	int *reordering;

	reordering = isl_alloc_array(ctx, int, len);
	if (!reordering)
		return NULL;

	if (dst <= src) {
		for (i = 0; i < dst; ++i)
			reordering[i] = i;
		for (i = 0; i < n; ++i)
			reordering[src + i] = dst + i;
		for (i = 0; i < src - dst; ++i)
			reordering[dst + i] = dst + n + i;
		for (i = 0; i < len - src - n; ++i)
			reordering[src + n + i] = src + n + i;
	} else {
		for (i = 0; i < src; ++i)
			reordering[i] = i;
		for (i = 0; i < n; ++i)
			reordering[src + i] = dst + i;
		for (i = 0; i < dst - src; ++i)
			reordering[src + n + i] = src + i;
		for (i = 0; i < len - dst - n; ++i)
			reordering[dst + n + i] = dst + n + i;
	}

	return reordering;
}

__isl_give isl_qpolynomial *isl_qpolynomial_move_dims(
	__isl_take isl_qpolynomial *qp,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	unsigned g_dst_pos;
	unsigned g_src_pos;
	int *reordering;

	if (n == 0)
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	if (dst_type == isl_dim_out || src_type == isl_dim_out)
		isl_die(qp->dim->ctx, isl_error_invalid,
			"cannot move output/set dimension",
			goto error);
	if (dst_type == isl_dim_in)
		dst_type = isl_dim_set;
	if (src_type == isl_dim_in)
		src_type = isl_dim_set;

	isl_assert(qp->dim->ctx, src_pos + n <= isl_space_dim(qp->dim, src_type),
		goto error);

	g_dst_pos = pos(qp->dim, dst_type) + dst_pos;
	g_src_pos = pos(qp->dim, src_type) + src_pos;
	if (dst_type > src_type)
		g_dst_pos -= n;

	qp->div = isl_mat_move_cols(qp->div, 2 + g_dst_pos, 2 + g_src_pos, n);
	if (!qp->div)
		goto error;
	qp = sort_divs(qp);
	if (!qp)
		goto error;

	reordering = reordering_move(qp->dim->ctx,
				qp->div->n_col - 2, g_dst_pos, g_src_pos, n);
	if (!reordering)
		goto error;

	qp->upoly = reorder(qp->upoly, reordering);
	free(reordering);
	if (!qp->upoly)
		goto error;

	qp->dim = isl_space_move_dims(qp->dim, dst_type, dst_pos, src_type, src_pos, n);
	if (!qp->dim)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_from_affine(__isl_take isl_space *dim,
	isl_int *f, isl_int denom)
{
	struct isl_upoly *up;

	dim = isl_space_domain(dim);
	if (!dim)
		return NULL;

	up = isl_upoly_from_affine(dim->ctx, f, denom,
					1 + isl_space_dim(dim, isl_dim_all));

	return isl_qpolynomial_alloc(dim, 0, up);
}

__isl_give isl_qpolynomial *isl_qpolynomial_from_aff(__isl_take isl_aff *aff)
{
	isl_ctx *ctx;
	struct isl_upoly *up;
	isl_qpolynomial *qp;

	if (!aff)
		return NULL;

	ctx = isl_aff_get_ctx(aff);
	up = isl_upoly_from_affine(ctx, aff->v->el + 1, aff->v->el[0],
				    aff->v->size - 1);

	qp = isl_qpolynomial_alloc(isl_aff_get_domain_space(aff),
				    aff->ls->div->n_row, up);
	if (!qp)
		goto error;

	isl_mat_free(qp->div);
	qp->div = isl_mat_copy(aff->ls->div);
	qp->div = isl_mat_cow(qp->div);
	if (!qp->div)
		goto error;

	isl_aff_free(aff);
	qp = reduce_divs(qp);
	qp = remove_redundant_divs(qp);
	return qp;
error:
	isl_aff_free(aff);
	return isl_qpolynomial_free(qp);
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_from_pw_aff(
	__isl_take isl_pw_aff *pwaff)
{
	int i;
	isl_pw_qpolynomial *pwqp;

	if (!pwaff)
		return NULL;

	pwqp = isl_pw_qpolynomial_alloc_size(isl_pw_aff_get_space(pwaff),
						pwaff->n);

	for (i = 0; i < pwaff->n; ++i) {
		isl_set *dom;
		isl_qpolynomial *qp;

		dom = isl_set_copy(pwaff->p[i].set);
		qp = isl_qpolynomial_from_aff(isl_aff_copy(pwaff->p[i].aff));
		pwqp = isl_pw_qpolynomial_add_piece(pwqp,  dom, qp);
	}

	isl_pw_aff_free(pwaff);
	return pwqp;
}

__isl_give isl_qpolynomial *isl_qpolynomial_from_constraint(
	__isl_take isl_constraint *c, enum isl_dim_type type, unsigned pos)
{
	isl_aff *aff;

	aff = isl_constraint_get_bound(c, type, pos);
	isl_constraint_free(c);
	return isl_qpolynomial_from_aff(aff);
}

/* For each 0 <= i < "n", replace variable "first" + i of type "type"
 * in "qp" by subs[i].
 */
__isl_give isl_qpolynomial *isl_qpolynomial_substitute(
	__isl_take isl_qpolynomial *qp,
	enum isl_dim_type type, unsigned first, unsigned n,
	__isl_keep isl_qpolynomial **subs)
{
	int i;
	struct isl_upoly **ups;

	if (n == 0)
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	if (type == isl_dim_out)
		isl_die(qp->dim->ctx, isl_error_invalid,
			"cannot substitute output/set dimension",
			goto error);
	if (type == isl_dim_in)
		type = isl_dim_set;

	for (i = 0; i < n; ++i)
		if (!subs[i])
			goto error;

	isl_assert(qp->dim->ctx, first + n <= isl_space_dim(qp->dim, type),
			goto error);

	for (i = 0; i < n; ++i)
		isl_assert(qp->dim->ctx, isl_space_is_equal(qp->dim, subs[i]->dim),
				goto error);

	isl_assert(qp->dim->ctx, qp->div->n_row == 0, goto error);
	for (i = 0; i < n; ++i)
		isl_assert(qp->dim->ctx, subs[i]->div->n_row == 0, goto error);

	first += pos(qp->dim, type);

	ups = isl_alloc_array(qp->dim->ctx, struct isl_upoly *, n);
	if (!ups)
		goto error;
	for (i = 0; i < n; ++i)
		ups[i] = subs[i]->upoly;

	qp->upoly = isl_upoly_subs(qp->upoly, first, n, ups);

	free(ups);

	if (!qp->upoly)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

/* Extend "bset" with extra set dimensions for each integer division
 * in "qp" and then call "fn" with the extended bset and the polynomial
 * that results from replacing each of the integer divisions by the
 * corresponding extra set dimension.
 */
int isl_qpolynomial_as_polynomial_on_domain(__isl_keep isl_qpolynomial *qp,
	__isl_keep isl_basic_set *bset,
	int (*fn)(__isl_take isl_basic_set *bset,
		  __isl_take isl_qpolynomial *poly, void *user), void *user)
{
	isl_space *dim;
	isl_mat *div;
	isl_qpolynomial *poly;

	if (!qp || !bset)
		goto error;
	if (qp->div->n_row == 0)
		return fn(isl_basic_set_copy(bset), isl_qpolynomial_copy(qp),
			  user);

	div = isl_mat_copy(qp->div);
	dim = isl_space_copy(qp->dim);
	dim = isl_space_add_dims(dim, isl_dim_set, qp->div->n_row);
	poly = isl_qpolynomial_alloc(dim, 0, isl_upoly_copy(qp->upoly));
	bset = isl_basic_set_copy(bset);
	bset = isl_basic_set_add_dims(bset, isl_dim_set, qp->div->n_row);
	bset = add_div_constraints(bset, div);

	return fn(bset, poly, user);
error:
	return -1;
}

/* Return total degree in variables first (inclusive) up to last (exclusive).
 */
int isl_upoly_degree(__isl_keep struct isl_upoly *up, int first, int last)
{
	int deg = -1;
	int i;
	struct isl_upoly_rec *rec;

	if (!up)
		return -2;
	if (isl_upoly_is_zero(up))
		return -1;
	if (isl_upoly_is_cst(up) || up->var < first)
		return 0;

	rec = isl_upoly_as_rec(up);
	if (!rec)
		return -2;

	for (i = 0; i < rec->n; ++i) {
		int d;

		if (isl_upoly_is_zero(rec->p[i]))
			continue;
		d = isl_upoly_degree(rec->p[i], first, last);
		if (up->var < last)
			d += i;
		if (d > deg)
			deg = d;
	}

	return deg;
}

/* Return total degree in set variables.
 */
int isl_qpolynomial_degree(__isl_keep isl_qpolynomial *poly)
{
	unsigned ovar;
	unsigned nvar;

	if (!poly)
		return -2;

	ovar = isl_space_offset(poly->dim, isl_dim_set);
	nvar = isl_space_dim(poly->dim, isl_dim_set);
	return isl_upoly_degree(poly->upoly, ovar, ovar + nvar);
}

__isl_give struct isl_upoly *isl_upoly_coeff(__isl_keep struct isl_upoly *up,
	unsigned pos, int deg)
{
	int i;
	struct isl_upoly_rec *rec;

	if (!up)
		return NULL;

	if (isl_upoly_is_cst(up) || up->var < pos) {
		if (deg == 0)
			return isl_upoly_copy(up);
		else
			return isl_upoly_zero(up->ctx);
	}

	rec = isl_upoly_as_rec(up);
	if (!rec)
		return NULL;

	if (up->var == pos) {
		if (deg < rec->n)
			return isl_upoly_copy(rec->p[deg]);
		else
			return isl_upoly_zero(up->ctx);
	}

	up = isl_upoly_copy(up);
	up = isl_upoly_cow(up);
	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		struct isl_upoly *t;
		t = isl_upoly_coeff(rec->p[i], pos, deg);
		if (!t)
			goto error;
		isl_upoly_free(rec->p[i]);
		rec->p[i] = t;
	}

	return up;
error:
	isl_upoly_free(up);
	return NULL;
}

/* Return coefficient of power "deg" of variable "t_pos" of type "type".
 */
__isl_give isl_qpolynomial *isl_qpolynomial_coeff(
	__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type, unsigned t_pos, int deg)
{
	unsigned g_pos;
	struct isl_upoly *up;
	isl_qpolynomial *c;

	if (!qp)
		return NULL;

	if (type == isl_dim_out)
		isl_die(qp->div->ctx, isl_error_invalid,
			"output/set dimension does not have a coefficient",
			return NULL);
	if (type == isl_dim_in)
		type = isl_dim_set;

	isl_assert(qp->div->ctx, t_pos < isl_space_dim(qp->dim, type),
			return NULL);

	g_pos = pos(qp->dim, type) + t_pos;
	up = isl_upoly_coeff(qp->upoly, g_pos, deg);

	c = isl_qpolynomial_alloc(isl_space_copy(qp->dim), qp->div->n_row, up);
	if (!c)
		return NULL;
	isl_mat_free(c->div);
	c->div = isl_mat_copy(qp->div);
	if (!c->div)
		goto error;
	return c;
error:
	isl_qpolynomial_free(c);
	return NULL;
}

/* Homogenize the polynomial in the variables first (inclusive) up to
 * last (exclusive) by inserting powers of variable first.
 * Variable first is assumed not to appear in the input.
 */
__isl_give struct isl_upoly *isl_upoly_homogenize(
	__isl_take struct isl_upoly *up, int deg, int target,
	int first, int last)
{
	int i;
	struct isl_upoly_rec *rec;

	if (!up)
		return NULL;
	if (isl_upoly_is_zero(up))
		return up;
	if (deg == target)
		return up;
	if (isl_upoly_is_cst(up) || up->var < first) {
		struct isl_upoly *hom;

		hom = isl_upoly_var_pow(up->ctx, first, target - deg);
		if (!hom)
			goto error;
		rec = isl_upoly_as_rec(hom);
		rec->p[target - deg] = isl_upoly_mul(rec->p[target - deg], up);

		return hom;
	}

	up = isl_upoly_cow(up);
	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		if (isl_upoly_is_zero(rec->p[i]))
			continue;
		rec->p[i] = isl_upoly_homogenize(rec->p[i],
				up->var < last ? deg + i : i, target,
				first, last);
		if (!rec->p[i])
			goto error;
	}

	return up;
error:
	isl_upoly_free(up);
	return NULL;
}

/* Homogenize the polynomial in the set variables by introducing
 * powers of an extra set variable at position 0.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_homogenize(
	__isl_take isl_qpolynomial *poly)
{
	unsigned ovar;
	unsigned nvar;
	int deg = isl_qpolynomial_degree(poly);

	if (deg < -1)
		goto error;

	poly = isl_qpolynomial_insert_dims(poly, isl_dim_in, 0, 1);
	poly = isl_qpolynomial_cow(poly);
	if (!poly)
		goto error;

	ovar = isl_space_offset(poly->dim, isl_dim_set);
	nvar = isl_space_dim(poly->dim, isl_dim_set);
	poly->upoly = isl_upoly_homogenize(poly->upoly, 0, deg,
						ovar, ovar + nvar);
	if (!poly->upoly)
		goto error;

	return poly;
error:
	isl_qpolynomial_free(poly);
	return NULL;
}

__isl_give isl_term *isl_term_alloc(__isl_take isl_space *dim,
	__isl_take isl_mat *div)
{
	isl_term *term;
	int n;

	if (!dim || !div)
		goto error;

	n = isl_space_dim(dim, isl_dim_all) + div->n_row;

	term = isl_calloc(dim->ctx, struct isl_term,
			sizeof(struct isl_term) + (n - 1) * sizeof(int));
	if (!term)
		goto error;

	term->ref = 1;
	term->dim = dim;
	term->div = div;
	isl_int_init(term->n);
	isl_int_init(term->d);
	
	return term;
error:
	isl_space_free(dim);
	isl_mat_free(div);
	return NULL;
}

__isl_give isl_term *isl_term_copy(__isl_keep isl_term *term)
{
	if (!term)
		return NULL;

	term->ref++;
	return term;
}

__isl_give isl_term *isl_term_dup(__isl_keep isl_term *term)
{
	int i;
	isl_term *dup;
	unsigned total;

	if (!term)
		return NULL;

	total = isl_space_dim(term->dim, isl_dim_all) + term->div->n_row;

	dup = isl_term_alloc(isl_space_copy(term->dim), isl_mat_copy(term->div));
	if (!dup)
		return NULL;

	isl_int_set(dup->n, term->n);
	isl_int_set(dup->d, term->d);

	for (i = 0; i < total; ++i)
		dup->pow[i] = term->pow[i];

	return dup;
}

__isl_give isl_term *isl_term_cow(__isl_take isl_term *term)
{
	if (!term)
		return NULL;

	if (term->ref == 1)
		return term;
	term->ref--;
	return isl_term_dup(term);
}

void isl_term_free(__isl_take isl_term *term)
{
	if (!term)
		return;

	if (--term->ref > 0)
		return;

	isl_space_free(term->dim);
	isl_mat_free(term->div);
	isl_int_clear(term->n);
	isl_int_clear(term->d);
	free(term);
}

unsigned isl_term_dim(__isl_keep isl_term *term, enum isl_dim_type type)
{
	if (!term)
		return 0;

	switch (type) {
	case isl_dim_param:
	case isl_dim_in:
	case isl_dim_out:	return isl_space_dim(term->dim, type);
	case isl_dim_div:	return term->div->n_row;
	case isl_dim_all:	return isl_space_dim(term->dim, isl_dim_all) +
								term->div->n_row;
	default:		return 0;
	}
}

isl_ctx *isl_term_get_ctx(__isl_keep isl_term *term)
{
	return term ? term->dim->ctx : NULL;
}

void isl_term_get_num(__isl_keep isl_term *term, isl_int *n)
{
	if (!term)
		return;
	isl_int_set(*n, term->n);
}

void isl_term_get_den(__isl_keep isl_term *term, isl_int *d)
{
	if (!term)
		return;
	isl_int_set(*d, term->d);
}

/* Return the coefficient of the term "term".
 */
__isl_give isl_val *isl_term_get_coefficient_val(__isl_keep isl_term *term)
{
	if (!term)
		return NULL;

	return isl_val_rat_from_isl_int(isl_term_get_ctx(term),
					term->n, term->d);
}

int isl_term_get_exp(__isl_keep isl_term *term,
	enum isl_dim_type type, unsigned pos)
{
	if (!term)
		return -1;

	isl_assert(term->dim->ctx, pos < isl_term_dim(term, type), return -1);

	if (type >= isl_dim_set)
		pos += isl_space_dim(term->dim, isl_dim_param);
	if (type >= isl_dim_div)
		pos += isl_space_dim(term->dim, isl_dim_set);

	return term->pow[pos];
}

__isl_give isl_aff *isl_term_get_div(__isl_keep isl_term *term, unsigned pos)
{
	isl_local_space *ls;
	isl_aff *aff;

	if (!term)
		return NULL;

	isl_assert(term->dim->ctx, pos < isl_term_dim(term, isl_dim_div),
			return NULL);

	ls = isl_local_space_alloc_div(isl_space_copy(term->dim),
					isl_mat_copy(term->div));
	aff = isl_aff_alloc(ls);
	if (!aff)
		return NULL;

	isl_seq_cpy(aff->v->el, term->div->row[pos], aff->v->size);

	aff = isl_aff_normalize(aff);

	return aff;
}

__isl_give isl_term *isl_upoly_foreach_term(__isl_keep struct isl_upoly *up,
	isl_stat (*fn)(__isl_take isl_term *term, void *user),
	__isl_take isl_term *term, void *user)
{
	int i;
	struct isl_upoly_rec *rec;

	if (!up || !term)
		goto error;

	if (isl_upoly_is_zero(up))
		return term;

	isl_assert(up->ctx, !isl_upoly_is_nan(up), goto error);
	isl_assert(up->ctx, !isl_upoly_is_infty(up), goto error);
	isl_assert(up->ctx, !isl_upoly_is_neginfty(up), goto error);

	if (isl_upoly_is_cst(up)) {
		struct isl_upoly_cst *cst;
		cst = isl_upoly_as_cst(up);
		if (!cst)
			goto error;
		term = isl_term_cow(term);
		if (!term)
			goto error;
		isl_int_set(term->n, cst->n);
		isl_int_set(term->d, cst->d);
		if (fn(isl_term_copy(term), user) < 0)
			goto error;
		return term;
	}

	rec = isl_upoly_as_rec(up);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		term = isl_term_cow(term);
		if (!term)
			goto error;
		term->pow[up->var] = i;
		term = isl_upoly_foreach_term(rec->p[i], fn, term, user);
		if (!term)
			goto error;
	}
	term->pow[up->var] = 0;

	return term;
error:
	isl_term_free(term);
	return NULL;
}

isl_stat isl_qpolynomial_foreach_term(__isl_keep isl_qpolynomial *qp,
	isl_stat (*fn)(__isl_take isl_term *term, void *user), void *user)
{
	isl_term *term;

	if (!qp)
		return isl_stat_error;

	term = isl_term_alloc(isl_space_copy(qp->dim), isl_mat_copy(qp->div));
	if (!term)
		return isl_stat_error;

	term = isl_upoly_foreach_term(qp->upoly, fn, term, user);

	isl_term_free(term);

	return term ? isl_stat_ok : isl_stat_error;
}

__isl_give isl_qpolynomial *isl_qpolynomial_from_term(__isl_take isl_term *term)
{
	struct isl_upoly *up;
	isl_qpolynomial *qp;
	int i, n;

	if (!term)
		return NULL;

	n = isl_space_dim(term->dim, isl_dim_all) + term->div->n_row;

	up = isl_upoly_rat_cst(term->dim->ctx, term->n, term->d);
	for (i = 0; i < n; ++i) {
		if (!term->pow[i])
			continue;
		up = isl_upoly_mul(up,
			isl_upoly_var_pow(term->dim->ctx, i, term->pow[i]));
	}

	qp = isl_qpolynomial_alloc(isl_space_copy(term->dim), term->div->n_row, up);
	if (!qp)
		goto error;
	isl_mat_free(qp->div);
	qp->div = isl_mat_copy(term->div);
	if (!qp->div)
		goto error;

	isl_term_free(term);
	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_term_free(term);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_lift(__isl_take isl_qpolynomial *qp,
	__isl_take isl_space *dim)
{
	int i;
	int extra;
	unsigned total;

	if (!qp || !dim)
		goto error;

	if (isl_space_is_equal(qp->dim, dim)) {
		isl_space_free(dim);
		return qp;
	}

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		goto error;

	extra = isl_space_dim(dim, isl_dim_set) -
			isl_space_dim(qp->dim, isl_dim_set);
	total = isl_space_dim(qp->dim, isl_dim_all);
	if (qp->div->n_row) {
		int *exp;

		exp = isl_alloc_array(qp->div->ctx, int, qp->div->n_row);
		if (!exp)
			goto error;
		for (i = 0; i < qp->div->n_row; ++i)
			exp[i] = extra + i;
		qp->upoly = expand(qp->upoly, exp, total);
		free(exp);
		if (!qp->upoly)
			goto error;
	}
	qp->div = isl_mat_insert_cols(qp->div, 2 + total, extra);
	if (!qp->div)
		goto error;
	for (i = 0; i < qp->div->n_row; ++i)
		isl_seq_clr(qp->div->row[i] + 2 + total, extra);

	isl_space_free(qp->dim);
	qp->dim = dim;

	return qp;
error:
	isl_space_free(dim);
	isl_qpolynomial_free(qp);
	return NULL;
}

/* For each parameter or variable that does not appear in qp,
 * first eliminate the variable from all constraints and then set it to zero.
 */
static __isl_give isl_set *fix_inactive(__isl_take isl_set *set,
	__isl_keep isl_qpolynomial *qp)
{
	int *active = NULL;
	int i;
	int d;
	unsigned nparam;
	unsigned nvar;

	if (!set || !qp)
		goto error;

	d = isl_space_dim(set->dim, isl_dim_all);
	active = isl_calloc_array(set->ctx, int, d);
	if (set_active(qp, active) < 0)
		goto error;

	for (i = 0; i < d; ++i)
		if (!active[i])
			break;

	if (i == d) {
		free(active);
		return set;
	}

	nparam = isl_space_dim(set->dim, isl_dim_param);
	nvar = isl_space_dim(set->dim, isl_dim_set);
	for (i = 0; i < nparam; ++i) {
		if (active[i])
			continue;
		set = isl_set_eliminate(set, isl_dim_param, i, 1);
		set = isl_set_fix_si(set, isl_dim_param, i, 0);
	}
	for (i = 0; i < nvar; ++i) {
		if (active[nparam + i])
			continue;
		set = isl_set_eliminate(set, isl_dim_set, i, 1);
		set = isl_set_fix_si(set, isl_dim_set, i, 0);
	}

	free(active);

	return set;
error:
	free(active);
	isl_set_free(set);
	return NULL;
}

struct isl_opt_data {
	isl_qpolynomial *qp;
	int first;
	isl_val *opt;
	int max;
};

static isl_stat opt_fn(__isl_take isl_point *pnt, void *user)
{
	struct isl_opt_data *data = (struct isl_opt_data *)user;
	isl_val *val;

	val = isl_qpolynomial_eval(isl_qpolynomial_copy(data->qp), pnt);
	if (data->first) {
		data->first = 0;
		data->opt = val;
	} else if (data->max) {
		data->opt = isl_val_max(data->opt, val);
	} else {
		data->opt = isl_val_min(data->opt, val);
	}

	return isl_stat_ok;
}

__isl_give isl_val *isl_qpolynomial_opt_on_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_set *set, int max)
{
	struct isl_opt_data data = { NULL, 1, NULL, max };

	if (!set || !qp)
		goto error;

	if (isl_upoly_is_cst(qp->upoly)) {
		isl_set_free(set);
		data.opt = isl_qpolynomial_get_constant_val(qp);
		isl_qpolynomial_free(qp);
		return data.opt;
	}

	set = fix_inactive(set, qp);

	data.qp = qp;
	if (isl_set_foreach_point(set, opt_fn, &data) < 0)
		goto error;

	if (data.first)
		data.opt = isl_val_zero(isl_set_get_ctx(set));

	isl_set_free(set);
	isl_qpolynomial_free(qp);
	return data.opt;
error:
	isl_set_free(set);
	isl_qpolynomial_free(qp);
	isl_val_free(data.opt);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_morph_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_morph *morph)
{
	int i;
	int n_sub;
	isl_ctx *ctx;
	struct isl_upoly **subs;
	isl_mat *mat, *diag;

	qp = isl_qpolynomial_cow(qp);
	if (!qp || !morph)
		goto error;

	ctx = qp->dim->ctx;
	isl_assert(ctx, isl_space_is_equal(qp->dim, morph->dom->dim), goto error);

	n_sub = morph->inv->n_row - 1;
	if (morph->inv->n_row != morph->inv->n_col)
		n_sub += qp->div->n_row;
	subs = isl_calloc_array(ctx, struct isl_upoly *, n_sub);
	if (n_sub && !subs)
		goto error;

	for (i = 0; 1 + i < morph->inv->n_row; ++i)
		subs[i] = isl_upoly_from_affine(ctx, morph->inv->row[1 + i],
					morph->inv->row[0][0], morph->inv->n_col);
	if (morph->inv->n_row != morph->inv->n_col)
		for (i = 0; i < qp->div->n_row; ++i)
			subs[morph->inv->n_row - 1 + i] =
			    isl_upoly_var_pow(ctx, morph->inv->n_col - 1 + i, 1);

	qp->upoly = isl_upoly_subs(qp->upoly, 0, n_sub, subs);

	for (i = 0; i < n_sub; ++i)
		isl_upoly_free(subs[i]);
	free(subs);

	diag = isl_mat_diag(ctx, 1, morph->inv->row[0][0]);
	mat = isl_mat_diagonal(diag, isl_mat_copy(morph->inv));
	diag = isl_mat_diag(ctx, qp->div->n_row, morph->inv->row[0][0]);
	mat = isl_mat_diagonal(mat, diag);
	qp->div = isl_mat_product(qp->div, mat);
	isl_space_free(qp->dim);
	qp->dim = isl_space_copy(morph->ran->dim);

	if (!qp->upoly || !qp->div || !qp->dim)
		goto error;

	isl_morph_free(morph);

	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_morph_free(morph);
	return NULL;
}

__isl_give isl_union_pw_qpolynomial *isl_union_pw_qpolynomial_mul(
	__isl_take isl_union_pw_qpolynomial *upwqp1,
	__isl_take isl_union_pw_qpolynomial *upwqp2)
{
	return isl_union_pw_qpolynomial_match_bin_op(upwqp1, upwqp2,
						&isl_pw_qpolynomial_mul);
}

/* Reorder the columns of the given div definitions according to the
 * given reordering.
 */
static __isl_give isl_mat *reorder_divs(__isl_take isl_mat *div,
	__isl_take isl_reordering *r)
{
	int i, j;
	isl_mat *mat;
	int extra;

	if (!div || !r)
		goto error;

	extra = isl_space_dim(r->dim, isl_dim_all) + div->n_row - r->len;
	mat = isl_mat_alloc(div->ctx, div->n_row, div->n_col + extra);
	if (!mat)
		goto error;

	for (i = 0; i < div->n_row; ++i) {
		isl_seq_cpy(mat->row[i], div->row[i], 2);
		isl_seq_clr(mat->row[i] + 2, mat->n_col - 2);
		for (j = 0; j < r->len; ++j)
			isl_int_set(mat->row[i][2 + r->pos[j]],
				    div->row[i][2 + j]);
	}

	isl_reordering_free(r);
	isl_mat_free(div);
	return mat;
error:
	isl_reordering_free(r);
	isl_mat_free(div);
	return NULL;
}

/* Reorder the dimension of "qp" according to the given reordering.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_realign_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_reordering *r)
{
	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		goto error;

	r = isl_reordering_extend(r, qp->div->n_row);
	if (!r)
		goto error;

	qp->div = reorder_divs(qp->div, isl_reordering_copy(r));
	if (!qp->div)
		goto error;

	qp->upoly = reorder(qp->upoly, r->pos);
	if (!qp->upoly)
		goto error;

	qp = isl_qpolynomial_reset_domain_space(qp, isl_space_copy(r->dim));

	isl_reordering_free(r);
	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_reordering_free(r);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_align_params(
	__isl_take isl_qpolynomial *qp, __isl_take isl_space *model)
{
	if (!qp || !model)
		goto error;

	if (!isl_space_match(qp->dim, isl_dim_param, model, isl_dim_param)) {
		isl_reordering *exp;

		model = isl_space_drop_dims(model, isl_dim_in,
					0, isl_space_dim(model, isl_dim_in));
		model = isl_space_drop_dims(model, isl_dim_out,
					0, isl_space_dim(model, isl_dim_out));
		exp = isl_parameter_alignment_reordering(qp->dim, model);
		exp = isl_reordering_extend_space(exp,
					isl_qpolynomial_get_domain_space(qp));
		qp = isl_qpolynomial_realign_domain(qp, exp);
	}

	isl_space_free(model);
	return qp;
error:
	isl_space_free(model);
	isl_qpolynomial_free(qp);
	return NULL;
}

struct isl_split_periods_data {
	int max_periods;
	isl_pw_qpolynomial *res;
};

/* Create a slice where the integer division "div" has the fixed value "v".
 * In particular, if "div" refers to floor(f/m), then create a slice
 *
 *	m v <= f <= m v + (m - 1)
 *
 * or
 *
 *	f - m v >= 0
 *	-f + m v + (m - 1) >= 0
 */
static __isl_give isl_set *set_div_slice(__isl_take isl_space *dim,
	__isl_keep isl_qpolynomial *qp, int div, isl_int v)
{
	int total;
	isl_basic_set *bset = NULL;
	int k;

	if (!dim || !qp)
		goto error;

	total = isl_space_dim(dim, isl_dim_all);
	bset = isl_basic_set_alloc_space(isl_space_copy(dim), 0, 0, 2);

	k = isl_basic_set_alloc_inequality(bset);
	if (k < 0)
		goto error;
	isl_seq_cpy(bset->ineq[k], qp->div->row[div] + 1, 1 + total);
	isl_int_submul(bset->ineq[k][0], v, qp->div->row[div][0]);

	k = isl_basic_set_alloc_inequality(bset);
	if (k < 0)
		goto error;
	isl_seq_neg(bset->ineq[k], qp->div->row[div] + 1, 1 + total);
	isl_int_addmul(bset->ineq[k][0], v, qp->div->row[div][0]);
	isl_int_add(bset->ineq[k][0], bset->ineq[k][0], qp->div->row[div][0]);
	isl_int_sub_ui(bset->ineq[k][0], bset->ineq[k][0], 1);

	isl_space_free(dim);
	return isl_set_from_basic_set(bset);
error:
	isl_basic_set_free(bset);
	isl_space_free(dim);
	return NULL;
}

static isl_stat split_periods(__isl_take isl_set *set,
	__isl_take isl_qpolynomial *qp, void *user);

/* Create a slice of the domain "set" such that integer division "div"
 * has the fixed value "v" and add the results to data->res,
 * replacing the integer division by "v" in "qp".
 */
static isl_stat set_div(__isl_take isl_set *set,
	__isl_take isl_qpolynomial *qp, int div, isl_int v,
	struct isl_split_periods_data *data)
{
	int i;
	int total;
	isl_set *slice;
	struct isl_upoly *cst;

	slice = set_div_slice(isl_set_get_space(set), qp, div, v);
	set = isl_set_intersect(set, slice);

	if (!qp)
		goto error;

	total = isl_space_dim(qp->dim, isl_dim_all);

	for (i = div + 1; i < qp->div->n_row; ++i) {
		if (isl_int_is_zero(qp->div->row[i][2 + total + div]))
			continue;
		isl_int_addmul(qp->div->row[i][1],
				qp->div->row[i][2 + total + div], v);
		isl_int_set_si(qp->div->row[i][2 + total + div], 0);
	}

	cst = isl_upoly_rat_cst(qp->dim->ctx, v, qp->dim->ctx->one);
	qp = substitute_div(qp, div, cst);

	return split_periods(set, qp, data);
error:
	isl_set_free(set);
	isl_qpolynomial_free(qp);
	return -1;
}

/* Split the domain "set" such that integer division "div"
 * has a fixed value (ranging from "min" to "max") on each slice
 * and add the results to data->res.
 */
static isl_stat split_div(__isl_take isl_set *set,
	__isl_take isl_qpolynomial *qp, int div, isl_int min, isl_int max,
	struct isl_split_periods_data *data)
{
	for (; isl_int_le(min, max); isl_int_add_ui(min, min, 1)) {
		isl_set *set_i = isl_set_copy(set);
		isl_qpolynomial *qp_i = isl_qpolynomial_copy(qp);

		if (set_div(set_i, qp_i, div, min, data) < 0)
			goto error;
	}
	isl_set_free(set);
	isl_qpolynomial_free(qp);
	return isl_stat_ok;
error:
	isl_set_free(set);
	isl_qpolynomial_free(qp);
	return isl_stat_error;
}

/* If "qp" refers to any integer division
 * that can only attain "max_periods" distinct values on "set"
 * then split the domain along those distinct values.
 * Add the results (or the original if no splitting occurs)
 * to data->res.
 */
static isl_stat split_periods(__isl_take isl_set *set,
	__isl_take isl_qpolynomial *qp, void *user)
{
	int i;
	isl_pw_qpolynomial *pwqp;
	struct isl_split_periods_data *data;
	isl_int min, max;
	int total;
	isl_stat r = isl_stat_ok;

	data = (struct isl_split_periods_data *)user;

	if (!set || !qp)
		goto error;

	if (qp->div->n_row == 0) {
		pwqp = isl_pw_qpolynomial_alloc(set, qp);
		data->res = isl_pw_qpolynomial_add_disjoint(data->res, pwqp);
		return isl_stat_ok;
	}

	isl_int_init(min);
	isl_int_init(max);
	total = isl_space_dim(qp->dim, isl_dim_all);
	for (i = 0; i < qp->div->n_row; ++i) {
		enum isl_lp_result lp_res;

		if (isl_seq_first_non_zero(qp->div->row[i] + 2 + total,
						qp->div->n_row) != -1)
			continue;

		lp_res = isl_set_solve_lp(set, 0, qp->div->row[i] + 1,
					  set->ctx->one, &min, NULL, NULL);
		if (lp_res == isl_lp_error)
			goto error2;
		if (lp_res == isl_lp_unbounded || lp_res == isl_lp_empty)
			continue;
		isl_int_fdiv_q(min, min, qp->div->row[i][0]);

		lp_res = isl_set_solve_lp(set, 1, qp->div->row[i] + 1,
					  set->ctx->one, &max, NULL, NULL);
		if (lp_res == isl_lp_error)
			goto error2;
		if (lp_res == isl_lp_unbounded || lp_res == isl_lp_empty)
			continue;
		isl_int_fdiv_q(max, max, qp->div->row[i][0]);

		isl_int_sub(max, max, min);
		if (isl_int_cmp_si(max, data->max_periods) < 0) {
			isl_int_add(max, max, min);
			break;
		}
	}

	if (i < qp->div->n_row) {
		r = split_div(set, qp, i, min, max, data);
	} else {
		pwqp = isl_pw_qpolynomial_alloc(set, qp);
		data->res = isl_pw_qpolynomial_add_disjoint(data->res, pwqp);
	}

	isl_int_clear(max);
	isl_int_clear(min);

	return r;
error2:
	isl_int_clear(max);
	isl_int_clear(min);
error:
	isl_set_free(set);
	isl_qpolynomial_free(qp);
	return isl_stat_error;
}

/* If any quasi-polynomial in pwqp refers to any integer division
 * that can only attain "max_periods" distinct values on its domain
 * then split the domain along those distinct values.
 */
__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_split_periods(
	__isl_take isl_pw_qpolynomial *pwqp, int max_periods)
{
	struct isl_split_periods_data data;

	data.max_periods = max_periods;
	data.res = isl_pw_qpolynomial_zero(isl_pw_qpolynomial_get_space(pwqp));

	if (isl_pw_qpolynomial_foreach_piece(pwqp, &split_periods, &data) < 0)
		goto error;

	isl_pw_qpolynomial_free(pwqp);

	return data.res;
error:
	isl_pw_qpolynomial_free(data.res);
	isl_pw_qpolynomial_free(pwqp);
	return NULL;
}

/* Construct a piecewise quasipolynomial that is constant on the given
 * domain.  In particular, it is
 *	0	if cst == 0
 *	1	if cst == 1
 *  infinity	if cst == -1
 */
static __isl_give isl_pw_qpolynomial *constant_on_domain(
	__isl_take isl_basic_set *bset, int cst)
{
	isl_space *dim;
	isl_qpolynomial *qp;

	if (!bset)
		return NULL;

	bset = isl_basic_set_params(bset);
	dim = isl_basic_set_get_space(bset);
	if (cst < 0)
		qp = isl_qpolynomial_infty_on_domain(dim);
	else if (cst == 0)
		qp = isl_qpolynomial_zero_on_domain(dim);
	else
		qp = isl_qpolynomial_one_on_domain(dim);
	return isl_pw_qpolynomial_alloc(isl_set_from_basic_set(bset), qp);
}

/* Factor bset, call fn on each of the factors and return the product.
 *
 * If no factors can be found, simply call fn on the input.
 * Otherwise, construct the factors based on the factorizer,
 * call fn on each factor and compute the product.
 */
static __isl_give isl_pw_qpolynomial *compressed_multiplicative_call(
	__isl_take isl_basic_set *bset,
	__isl_give isl_pw_qpolynomial *(*fn)(__isl_take isl_basic_set *bset))
{
	int i, n;
	isl_space *dim;
	isl_set *set;
	isl_factorizer *f;
	isl_qpolynomial *qp;
	isl_pw_qpolynomial *pwqp;
	unsigned nparam;
	unsigned nvar;

	f = isl_basic_set_factorizer(bset);
	if (!f)
		goto error;
	if (f->n_group == 0) {
		isl_factorizer_free(f);
		return fn(bset);
	}

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	nvar = isl_basic_set_dim(bset, isl_dim_set);

	dim = isl_basic_set_get_space(bset);
	dim = isl_space_domain(dim);
	set = isl_set_universe(isl_space_copy(dim));
	qp = isl_qpolynomial_one_on_domain(dim);
	pwqp = isl_pw_qpolynomial_alloc(set, qp);

	bset = isl_morph_basic_set(isl_morph_copy(f->morph), bset);

	for (i = 0, n = 0; i < f->n_group; ++i) {
		isl_basic_set *bset_i;
		isl_pw_qpolynomial *pwqp_i;

		bset_i = isl_basic_set_copy(bset);
		bset_i = isl_basic_set_drop_constraints_involving(bset_i,
			    nparam + n + f->len[i], nvar - n - f->len[i]);
		bset_i = isl_basic_set_drop_constraints_involving(bset_i,
			    nparam, n);
		bset_i = isl_basic_set_drop(bset_i, isl_dim_set,
			    n + f->len[i], nvar - n - f->len[i]);
		bset_i = isl_basic_set_drop(bset_i, isl_dim_set, 0, n);

		pwqp_i = fn(bset_i);
		pwqp = isl_pw_qpolynomial_mul(pwqp, pwqp_i);

		n += f->len[i];
	}

	isl_basic_set_free(bset);
	isl_factorizer_free(f);

	return pwqp;
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Factor bset, call fn on each of the factors and return the product.
 * The function is assumed to evaluate to zero on empty domains,
 * to one on zero-dimensional domains and to infinity on unbounded domains
 * and will not be called explicitly on zero-dimensional or unbounded domains.
 *
 * We first check for some special cases and remove all equalities.
 * Then we hand over control to compressed_multiplicative_call.
 */
__isl_give isl_pw_qpolynomial *isl_basic_set_multiplicative_call(
	__isl_take isl_basic_set *bset,
	__isl_give isl_pw_qpolynomial *(*fn)(__isl_take isl_basic_set *bset))
{
	int bounded;
	isl_morph *morph;
	isl_pw_qpolynomial *pwqp;

	if (!bset)
		return NULL;

	if (isl_basic_set_plain_is_empty(bset))
		return constant_on_domain(bset, 0);

	if (isl_basic_set_dim(bset, isl_dim_set) == 0)
		return constant_on_domain(bset, 1);

	bounded = isl_basic_set_is_bounded(bset);
	if (bounded < 0)
		goto error;
	if (!bounded)
		return constant_on_domain(bset, -1);

	if (bset->n_eq == 0)
		return compressed_multiplicative_call(bset, fn);

	morph = isl_basic_set_full_compression(bset);
	bset = isl_morph_basic_set(isl_morph_copy(morph), bset);

	pwqp = compressed_multiplicative_call(bset, fn);

	morph = isl_morph_dom_params(morph);
	morph = isl_morph_ran_params(morph);
	morph = isl_morph_inverse(morph);

	pwqp = isl_pw_qpolynomial_morph_domain(pwqp, morph);

	return pwqp;
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Drop all floors in "qp", turning each integer division [a/m] into
 * a rational division a/m.  If "down" is set, then the integer division
 * is replaced by (a-(m-1))/m instead.
 */
static __isl_give isl_qpolynomial *qp_drop_floors(
	__isl_take isl_qpolynomial *qp, int down)
{
	int i;
	struct isl_upoly *s;

	if (!qp)
		return NULL;
	if (qp->div->n_row == 0)
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	for (i = qp->div->n_row - 1; i >= 0; --i) {
		if (down) {
			isl_int_sub(qp->div->row[i][1],
				    qp->div->row[i][1], qp->div->row[i][0]);
			isl_int_add_ui(qp->div->row[i][1],
				       qp->div->row[i][1], 1);
		}
		s = isl_upoly_from_affine(qp->dim->ctx, qp->div->row[i] + 1,
					qp->div->row[i][0], qp->div->n_col - 1);
		qp = substitute_div(qp, i, s);
		if (!qp)
			return NULL;
	}

	return qp;
}

/* Drop all floors in "pwqp", turning each integer division [a/m] into
 * a rational division a/m.
 */
static __isl_give isl_pw_qpolynomial *pwqp_drop_floors(
	__isl_take isl_pw_qpolynomial *pwqp)
{
	int i;

	if (!pwqp)
		return NULL;

	if (isl_pw_qpolynomial_is_zero(pwqp))
		return pwqp;

	pwqp = isl_pw_qpolynomial_cow(pwqp);
	if (!pwqp)
		return NULL;

	for (i = 0; i < pwqp->n; ++i) {
		pwqp->p[i].qp = qp_drop_floors(pwqp->p[i].qp, 0);
		if (!pwqp->p[i].qp)
			goto error;
	}

	return pwqp;
error:
	isl_pw_qpolynomial_free(pwqp);
	return NULL;
}

/* Adjust all the integer divisions in "qp" such that they are at least
 * one over the given orthant (identified by "signs").  This ensures
 * that they will still be non-negative even after subtracting (m-1)/m.
 *
 * In particular, f is replaced by f' + v, changing f = [a/m]
 * to f' = [(a - m v)/m].
 * If the constant term k in a is smaller than m,
 * the constant term of v is set to floor(k/m) - 1.
 * For any other term, if the coefficient c and the variable x have
 * the same sign, then no changes are needed.
 * Otherwise, if the variable is positive (and c is negative),
 * then the coefficient of x in v is set to floor(c/m).
 * If the variable is negative (and c is positive),
 * then the coefficient of x in v is set to ceil(c/m).
 */
static __isl_give isl_qpolynomial *make_divs_pos(__isl_take isl_qpolynomial *qp,
	int *signs)
{
	int i, j;
	int total;
	isl_vec *v = NULL;
	struct isl_upoly *s;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;
	qp->div = isl_mat_cow(qp->div);
	if (!qp->div)
		goto error;

	total = isl_space_dim(qp->dim, isl_dim_all);
	v = isl_vec_alloc(qp->div->ctx, qp->div->n_col - 1);

	for (i = 0; i < qp->div->n_row; ++i) {
		isl_int *row = qp->div->row[i];
		v = isl_vec_clr(v);
		if (!v)
			goto error;
		if (isl_int_lt(row[1], row[0])) {
			isl_int_fdiv_q(v->el[0], row[1], row[0]);
			isl_int_sub_ui(v->el[0], v->el[0], 1);
			isl_int_submul(row[1], row[0], v->el[0]);
		}
		for (j = 0; j < total; ++j) {
			if (isl_int_sgn(row[2 + j]) * signs[j] >= 0)
				continue;
			if (signs[j] < 0)
				isl_int_cdiv_q(v->el[1 + j], row[2 + j], row[0]);
			else
				isl_int_fdiv_q(v->el[1 + j], row[2 + j], row[0]);
			isl_int_submul(row[2 + j], row[0], v->el[1 + j]);
		}
		for (j = 0; j < i; ++j) {
			if (isl_int_sgn(row[2 + total + j]) >= 0)
				continue;
			isl_int_fdiv_q(v->el[1 + total + j],
					row[2 + total + j], row[0]);
			isl_int_submul(row[2 + total + j],
					row[0], v->el[1 + total + j]);
		}
		for (j = i + 1; j < qp->div->n_row; ++j) {
			if (isl_int_is_zero(qp->div->row[j][2 + total + i]))
				continue;
			isl_seq_combine(qp->div->row[j] + 1,
				qp->div->ctx->one, qp->div->row[j] + 1,
				qp->div->row[j][2 + total + i], v->el, v->size);
		}
		isl_int_set_si(v->el[1 + total + i], 1);
		s = isl_upoly_from_affine(qp->dim->ctx, v->el,
					qp->div->ctx->one, v->size);
		qp->upoly = isl_upoly_subs(qp->upoly, total + i, 1, &s);
		isl_upoly_free(s);
		if (!qp->upoly)
			goto error;
	}

	isl_vec_free(v);
	return qp;
error:
	isl_vec_free(v);
	isl_qpolynomial_free(qp);
	return NULL;
}

struct isl_to_poly_data {
	int sign;
	isl_pw_qpolynomial *res;
	isl_qpolynomial *qp;
};

/* Appoximate data->qp by a polynomial on the orthant identified by "signs".
 * We first make all integer divisions positive and then split the
 * quasipolynomials into terms with sign data->sign (the direction
 * of the requested approximation) and terms with the opposite sign.
 * In the first set of terms, each integer division [a/m] is
 * overapproximated by a/m, while in the second it is underapproximated
 * by (a-(m-1))/m.
 */
static int to_polynomial_on_orthant(__isl_take isl_set *orthant, int *signs,
	void *user)
{
	struct isl_to_poly_data *data = user;
	isl_pw_qpolynomial *t;
	isl_qpolynomial *qp, *up, *down;

	qp = isl_qpolynomial_copy(data->qp);
	qp = make_divs_pos(qp, signs);

	up = isl_qpolynomial_terms_of_sign(qp, signs, data->sign);
	up = qp_drop_floors(up, 0);
	down = isl_qpolynomial_terms_of_sign(qp, signs, -data->sign);
	down = qp_drop_floors(down, 1);

	isl_qpolynomial_free(qp);
	qp = isl_qpolynomial_add(up, down);

	t = isl_pw_qpolynomial_alloc(orthant, qp);
	data->res = isl_pw_qpolynomial_add_disjoint(data->res, t);

	return 0;
}

/* Approximate each quasipolynomial by a polynomial.  If "sign" is positive,
 * the polynomial will be an overapproximation.  If "sign" is negative,
 * it will be an underapproximation.  If "sign" is zero, the approximation
 * will lie somewhere in between.
 *
 * In particular, is sign == 0, we simply drop the floors, turning
 * the integer divisions into rational divisions.
 * Otherwise, we split the domains into orthants, make all integer divisions
 * positive and then approximate each [a/m] by either a/m or (a-(m-1))/m,
 * depending on the requested sign and the sign of the term in which
 * the integer division appears.
 */
__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_to_polynomial(
	__isl_take isl_pw_qpolynomial *pwqp, int sign)
{
	int i;
	struct isl_to_poly_data data;

	if (sign == 0)
		return pwqp_drop_floors(pwqp);

	if (!pwqp)
		return NULL;

	data.sign = sign;
	data.res = isl_pw_qpolynomial_zero(isl_pw_qpolynomial_get_space(pwqp));

	for (i = 0; i < pwqp->n; ++i) {
		if (pwqp->p[i].qp->div->n_row == 0) {
			isl_pw_qpolynomial *t;
			t = isl_pw_qpolynomial_alloc(
					isl_set_copy(pwqp->p[i].set),
					isl_qpolynomial_copy(pwqp->p[i].qp));
			data.res = isl_pw_qpolynomial_add_disjoint(data.res, t);
			continue;
		}
		data.qp = pwqp->p[i].qp;
		if (isl_set_foreach_orthant(pwqp->p[i].set,
					&to_polynomial_on_orthant, &data) < 0)
			goto error;
	}

	isl_pw_qpolynomial_free(pwqp);

	return data.res;
error:
	isl_pw_qpolynomial_free(pwqp);
	isl_pw_qpolynomial_free(data.res);
	return NULL;
}

static __isl_give isl_pw_qpolynomial *poly_entry(
	__isl_take isl_pw_qpolynomial *pwqp, void *user)
{
	int *sign = user;

	return isl_pw_qpolynomial_to_polynomial(pwqp, *sign);
}

__isl_give isl_union_pw_qpolynomial *isl_union_pw_qpolynomial_to_polynomial(
	__isl_take isl_union_pw_qpolynomial *upwqp, int sign)
{
	return isl_union_pw_qpolynomial_transform_inplace(upwqp,
				   &poly_entry, &sign);
}

__isl_give isl_basic_map *isl_basic_map_from_qpolynomial(
	__isl_take isl_qpolynomial *qp)
{
	int i, k;
	isl_space *dim;
	isl_vec *aff = NULL;
	isl_basic_map *bmap = NULL;
	unsigned pos;
	unsigned n_div;

	if (!qp)
		return NULL;
	if (!isl_upoly_is_affine(qp->upoly))
		isl_die(qp->dim->ctx, isl_error_invalid,
			"input quasi-polynomial not affine", goto error);
	aff = isl_qpolynomial_extract_affine(qp);
	if (!aff)
		goto error;
	dim = isl_qpolynomial_get_space(qp);
	pos = 1 + isl_space_offset(dim, isl_dim_out);
	n_div = qp->div->n_row;
	bmap = isl_basic_map_alloc_space(dim, n_div, 1, 2 * n_div);

	for (i = 0; i < n_div; ++i) {
		k = isl_basic_map_alloc_div(bmap);
		if (k < 0)
			goto error;
		isl_seq_cpy(bmap->div[k], qp->div->row[i], qp->div->n_col);
		isl_int_set_si(bmap->div[k][qp->div->n_col], 0);
		if (isl_basic_map_add_div_constraints(bmap, k) < 0)
			goto error;
	}
	k = isl_basic_map_alloc_equality(bmap);
	if (k < 0)
		goto error;
	isl_int_neg(bmap->eq[k][pos], aff->el[0]);
	isl_seq_cpy(bmap->eq[k], aff->el + 1, pos);
	isl_seq_cpy(bmap->eq[k] + pos + 1, aff->el + 1 + pos, n_div);

	isl_vec_free(aff);
	isl_qpolynomial_free(qp);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
error:
	isl_vec_free(aff);
	isl_qpolynomial_free(qp);
	isl_basic_map_free(bmap);
	return NULL;
}
