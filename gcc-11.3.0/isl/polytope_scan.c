/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <assert.h>
#include <isl_map_private.h>
#include "isl_equalities.h"
#include <isl_seq.h>
#include "isl_scan.h"
#include <isl_mat_private.h>
#include <isl_vec_private.h>

/* The input of this program is the same as that of the "polytope_scan"
 * program from the barvinok distribution.
 *
 * Constraints of set is PolyLib format.
 *
 * The input set is assumed to be bounded.
 */

struct scan_samples {
	struct isl_scan_callback callback;
	struct isl_mat *samples;
};

static isl_stat scan_samples_add_sample(struct isl_scan_callback *cb,
	__isl_take isl_vec *sample)
{
	struct scan_samples *ss = (struct scan_samples *)cb;

	ss->samples = isl_mat_extend(ss->samples, ss->samples->n_row + 1,
						  ss->samples->n_col);
	if (!ss->samples)
		goto error;

	isl_seq_cpy(ss->samples->row[ss->samples->n_row - 1],
		    sample->el, sample->size);

	isl_vec_free(sample);
	return isl_stat_ok;
error:
	isl_vec_free(sample);
	return isl_stat_error;
}

static struct isl_mat *isl_basic_set_scan_samples(struct isl_basic_set *bset)
{
	isl_ctx *ctx;
	unsigned dim;
	struct scan_samples ss;

	ctx = isl_basic_set_get_ctx(bset);
	dim = isl_basic_set_total_dim(bset);
	ss.callback.add = scan_samples_add_sample;
	ss.samples = isl_mat_alloc(ctx, 0, 1 + dim);
	if (!ss.samples)
		goto error;

	if (isl_basic_set_scan(bset, &ss.callback) < 0) {
		isl_mat_free(ss.samples);
		return NULL;
	}

	return ss.samples;
error:
	isl_basic_set_free(bset);
	return NULL;
}

static struct isl_mat *isl_basic_set_samples(struct isl_basic_set *bset)
{
	struct isl_mat *T;
	struct isl_mat *samples;

	if (!bset)
		return NULL;

	if (bset->n_eq == 0)
		return isl_basic_set_scan_samples(bset);

	bset = isl_basic_set_remove_equalities(bset, &T, NULL);
	samples = isl_basic_set_scan_samples(bset);
	return isl_mat_product(samples, isl_mat_transpose(T));
}

int main(int argc, char **argv)
{
	struct isl_ctx *ctx = isl_ctx_alloc();
	struct isl_basic_set *bset;
	struct isl_mat *samples;

	bset = isl_basic_set_read_from_file(ctx, stdin);
	samples = isl_basic_set_samples(bset);
	isl_mat_print_internal(samples, stdout, 0);
	isl_mat_free(samples);
	isl_ctx_free(ctx);

	return 0;
}
