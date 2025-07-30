/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_MAT_H
#define ISL_MAT_H

#include <stdio.h>

#include <isl/ctx.h>
#include <isl/vec.h>
#include <isl/val.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct isl_mat;
typedef struct isl_mat	isl_mat;

isl_ctx *isl_mat_get_ctx(__isl_keep isl_mat *mat);

__isl_give isl_mat *isl_mat_alloc(isl_ctx *ctx,
	unsigned n_row, unsigned n_col);
struct isl_mat *isl_mat_dup(struct isl_mat *mat);
struct isl_mat *isl_mat_extend(struct isl_mat *mat,
	unsigned n_row, unsigned n_col);
struct isl_mat *isl_mat_identity(struct isl_ctx *ctx, unsigned n_row);
__isl_give isl_mat *isl_mat_copy(__isl_keep isl_mat *mat);
struct isl_mat *isl_mat_cow(struct isl_mat *mat);
__isl_null isl_mat *isl_mat_free(__isl_take isl_mat *mat);

int isl_mat_rows(__isl_keep isl_mat *mat);
int isl_mat_cols(__isl_keep isl_mat *mat);
__isl_give isl_val *isl_mat_get_element_val(__isl_keep isl_mat *mat,
	int row, int col);
__isl_give isl_mat *isl_mat_set_element_si(__isl_take isl_mat *mat,
	int row, int col, int v);
__isl_give isl_mat *isl_mat_set_element_val(__isl_take isl_mat *mat,
	int row, int col, __isl_take isl_val *v);

struct isl_mat *isl_mat_swap_cols(struct isl_mat *mat, unsigned i, unsigned j);
struct isl_mat *isl_mat_swap_rows(struct isl_mat *mat, unsigned i, unsigned j);

struct isl_vec *isl_mat_vec_product(struct isl_mat *mat, struct isl_vec *vec);
struct isl_vec *isl_vec_mat_product(struct isl_vec *vec, struct isl_mat *mat);
__isl_give isl_vec *isl_mat_vec_inverse_product(__isl_take isl_mat *mat,
						__isl_take isl_vec *vec);
struct isl_mat *isl_mat_aff_direct_sum(struct isl_mat *left,
					struct isl_mat *right);
__isl_give isl_mat *isl_mat_diagonal(__isl_take isl_mat *mat1,
	__isl_take isl_mat *mat2);
struct isl_mat *isl_mat_left_hermite(struct isl_mat *M,
	int neg, struct isl_mat **U, struct isl_mat **Q);
struct isl_mat *isl_mat_lin_to_aff(struct isl_mat *mat);
struct isl_mat *isl_mat_inverse_product(struct isl_mat *left,
	struct isl_mat *right);
__isl_give isl_mat *isl_mat_product(__isl_take isl_mat *left,
	__isl_take isl_mat *right);
struct isl_mat *isl_mat_transpose(struct isl_mat *mat);
__isl_give isl_mat *isl_mat_right_inverse(__isl_take isl_mat *mat);
__isl_give isl_mat *isl_mat_right_kernel(__isl_take isl_mat *mat);

__isl_give isl_mat *isl_mat_normalize(__isl_take isl_mat *mat);
__isl_give isl_mat *isl_mat_normalize_row(__isl_take isl_mat *mat, int row);

struct isl_mat *isl_mat_drop_cols(struct isl_mat *mat,
				unsigned col, unsigned n);
struct isl_mat *isl_mat_drop_rows(struct isl_mat *mat,
				unsigned row, unsigned n);
__isl_give isl_mat *isl_mat_insert_cols(__isl_take isl_mat *mat,
				unsigned col, unsigned n);
__isl_give isl_mat *isl_mat_insert_rows(__isl_take isl_mat *mat,
				unsigned row, unsigned n);
__isl_give isl_mat *isl_mat_move_cols(__isl_take isl_mat *mat,
	unsigned dst_col, unsigned src_col, unsigned n);
__isl_give isl_mat *isl_mat_add_rows(__isl_take isl_mat *mat, unsigned n);
__isl_give isl_mat *isl_mat_insert_zero_cols(__isl_take isl_mat *mat,
	unsigned first, unsigned n);
__isl_give isl_mat *isl_mat_add_zero_cols(__isl_take isl_mat *mat, unsigned n);
__isl_give isl_mat *isl_mat_insert_zero_rows(__isl_take isl_mat *mat,
	unsigned row, unsigned n);
__isl_give isl_mat *isl_mat_add_zero_rows(__isl_take isl_mat *mat, unsigned n);

void isl_mat_col_add(__isl_keep isl_mat *mat, int dst_col, int src_col);

struct isl_mat *isl_mat_unimodular_complete(struct isl_mat *M, int row);

__isl_give isl_mat *isl_mat_from_row_vec(__isl_take isl_vec *vec);
__isl_give isl_mat *isl_mat_concat(__isl_take isl_mat *top,
	__isl_take isl_mat *bot);
__isl_give isl_mat *isl_mat_vec_concat(__isl_take isl_mat *top,
	__isl_take isl_vec *bot);

int isl_mat_is_equal(__isl_keep isl_mat *mat1, __isl_keep isl_mat *mat2);

int isl_mat_initial_non_zero_cols(__isl_keep isl_mat *mat);

void isl_mat_print_internal(__isl_keep isl_mat *mat, FILE *out, int indent);
void isl_mat_dump(__isl_keep isl_mat *mat);

#if defined(__cplusplus)
}
#endif

#endif
