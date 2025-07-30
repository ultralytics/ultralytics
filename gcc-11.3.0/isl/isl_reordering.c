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
#include <isl_space_private.h>
#include <isl_reordering.h>

__isl_give isl_reordering *isl_reordering_alloc(isl_ctx *ctx, int len)
{
	isl_reordering *exp;

	exp = isl_alloc(ctx, struct isl_reordering,
			sizeof(struct isl_reordering) + (len - 1) * sizeof(int));
	if (!exp)
		return NULL;

	exp->ref = 1;
	exp->len = len;
	exp->dim = NULL;

	return exp;
}

__isl_give isl_reordering *isl_reordering_copy(__isl_keep isl_reordering *exp)
{
	if (!exp)
		return NULL;

	exp->ref++;
	return exp;
}

__isl_give isl_reordering *isl_reordering_dup(__isl_keep isl_reordering *r)
{
	int i;
	isl_reordering *dup;

	if (!r)
		return NULL;

	dup = isl_reordering_alloc(r->dim->ctx, r->len);
	if (!dup)
		return NULL;

	dup->dim = isl_space_copy(r->dim);
	if (!dup->dim)
		return isl_reordering_free(dup);
	for (i = 0; i < dup->len; ++i)
		dup->pos[i] = r->pos[i];

	return dup;
}

__isl_give isl_reordering *isl_reordering_cow(__isl_take isl_reordering *r)
{
	if (!r)
		return NULL;

	if (r->ref == 1)
		return r;
	r->ref--;
	return isl_reordering_dup(r);
}

void *isl_reordering_free(__isl_take isl_reordering *exp)
{
	if (!exp)
		return NULL;

	if (--exp->ref > 0)
		return NULL;

	isl_space_free(exp->dim);
	free(exp);
	return NULL;
}

/* Construct a reordering that maps the parameters of "alignee"
 * to the corresponding parameters in a new dimension specification
 * that has the parameters of "aligner" first, followed by
 * any remaining parameters of "alignee" that do not occur in "aligner".
 */
__isl_give isl_reordering *isl_parameter_alignment_reordering(
	__isl_keep isl_space *alignee, __isl_keep isl_space *aligner)
{
	int i, j;
	isl_reordering *exp;

	if (!alignee || !aligner)
		return NULL;

	exp = isl_reordering_alloc(alignee->ctx, alignee->nparam);
	if (!exp)
		return NULL;

	exp->dim = isl_space_copy(aligner);

	for (i = 0; i < alignee->nparam; ++i) {
		isl_id *id_i;
		id_i = isl_space_get_dim_id(alignee, isl_dim_param, i);
		if (!id_i)
			isl_die(alignee->ctx, isl_error_invalid,
				"cannot align unnamed parameters", goto error);
		for (j = 0; j < aligner->nparam; ++j) {
			isl_id *id_j;
			id_j = isl_space_get_dim_id(aligner, isl_dim_param, j);
			isl_id_free(id_j);
			if (id_i == id_j)
				break;
		}
		if (j < aligner->nparam) {
			exp->pos[i] = j;
			isl_id_free(id_i);
		} else {
			int pos;
			pos = isl_space_dim(exp->dim, isl_dim_param);
			exp->dim = isl_space_add_dims(exp->dim, isl_dim_param, 1);
			exp->dim = isl_space_set_dim_id(exp->dim,
						isl_dim_param, pos, id_i);
			exp->pos[i] = pos;
		}
	}

	if (!exp->dim)
		return isl_reordering_free(exp);
	return exp;
error:
	isl_reordering_free(exp);
	return NULL;
}

__isl_give isl_reordering *isl_reordering_extend(__isl_take isl_reordering *exp,
	unsigned extra)
{
	int i;
	isl_reordering *res;
	int offset;

	if (!exp)
		return NULL;
	if (extra == 0)
		return exp;

	offset = isl_space_dim(exp->dim, isl_dim_all) - exp->len;
	res = isl_reordering_alloc(exp->dim->ctx, exp->len + extra);
	if (!res)
		goto error;
	res->dim = isl_space_copy(exp->dim);
	for (i = 0; i < exp->len; ++i)
		res->pos[i] = exp->pos[i];
	for (i = exp->len; i < res->len; ++i)
		res->pos[i] = offset + i;

	isl_reordering_free(exp);

	return res;
error:
	isl_reordering_free(exp);
	return NULL;
}

__isl_give isl_reordering *isl_reordering_extend_space(
	__isl_take isl_reordering *exp, __isl_take isl_space *dim)
{
	isl_reordering *res;

	if (!exp || !dim)
		goto error;

	res = isl_reordering_extend(isl_reordering_copy(exp),
				    isl_space_dim(dim, isl_dim_all) - exp->len);
	res = isl_reordering_cow(res);
	if (!res)
		goto error;
	isl_space_free(res->dim);
	res->dim = isl_space_replace(dim, isl_dim_param, exp->dim);

	isl_reordering_free(exp);

	if (!res->dim)
		return isl_reordering_free(res);

	return res;
error:
	isl_reordering_free(exp);
	isl_space_free(dim);
	return NULL;
}

void isl_reordering_dump(__isl_keep isl_reordering *exp)
{
	int i;

	isl_space_dump(exp->dim);
	for (i = 0; i < exp->len; ++i)
		fprintf(stderr, "%d -> %d; ", i, exp->pos[i]);
	fprintf(stderr, "\n");
}
