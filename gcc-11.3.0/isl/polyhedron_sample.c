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
#include "isl_sample.h"
#include <isl/vec.h>

int main(int argc, char **argv)
{
	struct isl_ctx *ctx = isl_ctx_alloc();
	struct isl_basic_set *bset;
	struct isl_vec *sample;
	isl_printer *p;

	bset = isl_basic_set_read_from_file(ctx, stdin);
	sample = isl_basic_set_sample_vec(isl_basic_set_copy(bset));
	p = isl_printer_to_file(ctx, stdout);
	p = isl_printer_print_vec(p, sample);
	p = isl_printer_end_line(p);
	isl_printer_free(p);
	assert(sample);
	if (isl_vec_size(sample) > 0)
		assert(isl_basic_set_contains(bset, sample));
	isl_basic_set_free(bset);
	isl_vec_free(sample);
	isl_ctx_free(ctx);

	return 0;
}
