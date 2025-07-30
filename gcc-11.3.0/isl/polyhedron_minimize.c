/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <assert.h>
#include <isl/set.h>
#include <isl/vec.h>
#include <isl_ilp_private.h>
#include <isl_seq.h>
#include <isl_vec_private.h>

/* The input of this program is the same as that of the "polytope_minimize"
 * program from the barvinok distribution.
 *
 * Constraints of set is PolyLib format.
 * Linear or affine objective function in PolyLib format.
 */

static struct isl_vec *isl_vec_lin_to_aff(struct isl_vec *vec)
{
	struct isl_vec *aff;

	if (!vec)
		return NULL;
	aff = isl_vec_alloc(vec->ctx, 1 + vec->size);
	if (!aff)
		goto error;
	isl_int_set_si(aff->el[0], 0);
	isl_seq_cpy(aff->el + 1, vec->el, vec->size);
	isl_vec_free(vec);
	return aff;
error:
	isl_vec_free(vec);
	return NULL;
}

/* Rotate elements of vector right.
 * In particular, move the constant term from the end of the
 * vector to the start of the vector.
 */
static struct isl_vec *vec_ror(struct isl_vec *vec)
{
	int i;

	if (!vec)
		return NULL;
	for (i = vec->size - 2; i >= 0; --i)
		isl_int_swap(vec->el[i], vec->el[i + 1]);
	return vec;
}

int main(int argc, char **argv)
{
	struct isl_ctx *ctx = isl_ctx_alloc();
	struct isl_basic_set *bset;
	struct isl_vec *obj;
	struct isl_vec *sol;
	isl_int opt;
	unsigned dim;
	enum isl_lp_result res;
	isl_printer *p;

	isl_int_init(opt);
	bset = isl_basic_set_read_from_file(ctx, stdin);
	assert(bset);
	obj = isl_vec_read_from_file(ctx, stdin);
	assert(obj);
	dim = isl_basic_set_total_dim(bset);
	assert(obj->size >= dim && obj->size <= dim + 1);
	if (obj->size != dim + 1)
		obj = isl_vec_lin_to_aff(obj);
	else
		obj = vec_ror(obj);
	res = isl_basic_set_solve_ilp(bset, 0, obj->el, &opt, &sol);
	switch (res) {
	case isl_lp_error:
		fprintf(stderr, "error\n");
		return -1;
	case isl_lp_empty:
		fprintf(stdout, "empty\n");
		break;
	case isl_lp_unbounded:
		fprintf(stdout, "unbounded\n");
		break;
	case isl_lp_ok:
		p = isl_printer_to_file(ctx, stdout);
		p = isl_printer_print_vec(p, sol);
		p = isl_printer_end_line(p);
		p = isl_printer_print_isl_int(p, opt);
		p = isl_printer_end_line(p);
		isl_printer_free(p);
	}
	isl_basic_set_free(bset);
	isl_vec_free(obj);
	isl_vec_free(sol);
	isl_ctx_free(ctx);
	isl_int_clear(opt);

	return 0;
}
