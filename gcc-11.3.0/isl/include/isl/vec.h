/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_VEC_H
#define ISL_VEC_H

#include <stdio.h>

#include <isl/ctx.h>
#include <isl/val.h>
#include <isl/printer.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct isl_vec;
typedef struct isl_vec isl_vec;

__isl_give isl_vec *isl_vec_alloc(isl_ctx *ctx, unsigned size);
__isl_give isl_vec *isl_vec_copy(__isl_keep isl_vec *vec);
__isl_null isl_vec *isl_vec_free(__isl_take isl_vec *vec);

isl_ctx *isl_vec_get_ctx(__isl_keep isl_vec *vec);

int isl_vec_size(__isl_keep isl_vec *vec);
__isl_give isl_val *isl_vec_get_element_val(__isl_keep isl_vec *vec, int pos);
__isl_give isl_vec *isl_vec_set_element_si(__isl_take isl_vec *vec,
	int pos, int v);
__isl_give isl_vec *isl_vec_set_element_val(__isl_take isl_vec *vec,
	int pos, __isl_take isl_val *v);

isl_bool isl_vec_is_equal(__isl_keep isl_vec *vec1, __isl_keep isl_vec *vec2);
int isl_vec_cmp_element(__isl_keep isl_vec *vec1, __isl_keep isl_vec *vec2,
	int pos);

void isl_vec_dump(__isl_keep isl_vec *vec);
__isl_give isl_printer *isl_printer_print_vec(__isl_take isl_printer *printer,
	__isl_keep isl_vec *vec);

struct isl_vec *isl_vec_ceil(struct isl_vec *vec);
struct isl_vec *isl_vec_normalize(struct isl_vec *vec);
__isl_give isl_vec *isl_vec_set_si(__isl_take isl_vec *vec, int v);
__isl_give isl_vec *isl_vec_set_val(__isl_take isl_vec *vec,
	__isl_take isl_val *v);
__isl_give isl_vec *isl_vec_clr(__isl_take isl_vec *vec);
__isl_give isl_vec *isl_vec_neg(__isl_take isl_vec *vec);
__isl_give isl_vec *isl_vec_add(__isl_take isl_vec *vec1,
	__isl_take isl_vec *vec2);
__isl_give isl_vec *isl_vec_extend(__isl_take isl_vec *vec, unsigned size);
__isl_give isl_vec *isl_vec_zero_extend(__isl_take isl_vec *vec, unsigned size);
__isl_give isl_vec *isl_vec_concat(__isl_take isl_vec *vec1,
	__isl_take isl_vec *vec2);

__isl_give isl_vec *isl_vec_sort(__isl_take isl_vec *vec);

__isl_give isl_vec *isl_vec_read_from_file(isl_ctx *ctx, FILE *input);

__isl_give isl_vec *isl_vec_drop_els(__isl_take isl_vec *vec,
	unsigned pos, unsigned n);
__isl_give isl_vec *isl_vec_insert_els(__isl_take isl_vec *vec,
	unsigned pos, unsigned n);
__isl_give isl_vec *isl_vec_insert_zero_els(__isl_take isl_vec *vec,
	unsigned pos, unsigned n);
__isl_give isl_vec *isl_vec_move_els(__isl_take isl_vec *vec,
	unsigned dst_col, unsigned src_col, unsigned n);

#if defined(__cplusplus)
}
#endif

#endif
