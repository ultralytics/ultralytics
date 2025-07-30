/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2014      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl/space.h>
#include <isl_seq.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl_space_private.h>
#include <isl_val_private.h>
#include <isl/deprecated/mat_int.h>

isl_ctx *isl_mat_get_ctx(__isl_keep isl_mat *mat)
{
	return mat ? mat->ctx : NULL;
}

/* Return a hash value that digests "mat".
 */
uint32_t isl_mat_get_hash(__isl_keep isl_mat *mat)
{
	int i;
	uint32_t hash;

	if (!mat)
		return 0;

	hash = isl_hash_init();
	isl_hash_byte(hash, mat->n_row & 0xFF);
	isl_hash_byte(hash, mat->n_col & 0xFF);
	for (i = 0; i < mat->n_row; ++i) {
		uint32_t row_hash;

		row_hash = isl_seq_get_hash(mat->row[i], mat->n_col);
		isl_hash_hash(hash, row_hash);
	}

	return hash;
}

struct isl_mat *isl_mat_alloc(struct isl_ctx *ctx,
	unsigned n_row, unsigned n_col)
{
	int i;
	struct isl_mat *mat;

	mat = isl_alloc_type(ctx, struct isl_mat);
	if (!mat)
		return NULL;

	mat->row = NULL;
	mat->block = isl_blk_alloc(ctx, n_row * n_col);
	if (isl_blk_is_error(mat->block))
		goto error;
	mat->row = isl_alloc_array(ctx, isl_int *, n_row);
	if (n_row && !mat->row)
		goto error;

	for (i = 0; i < n_row; ++i)
		mat->row[i] = mat->block.data + i * n_col;

	mat->ctx = ctx;
	isl_ctx_ref(ctx);
	mat->ref = 1;
	mat->n_row = n_row;
	mat->n_col = n_col;
	mat->max_col = n_col;
	mat->flags = 0;

	return mat;
error:
	isl_blk_free(ctx, mat->block);
	free(mat);
	return NULL;
}

struct isl_mat *isl_mat_extend(struct isl_mat *mat,
	unsigned n_row, unsigned n_col)
{
	int i;
	isl_int *old;
	isl_int **row;

	if (!mat)
		return NULL;

	if (mat->max_col >= n_col && mat->n_row >= n_row) {
		if (mat->n_col < n_col)
			mat->n_col = n_col;
		return mat;
	}

	if (mat->max_col < n_col) {
		struct isl_mat *new_mat;

		if (n_row < mat->n_row)
			n_row = mat->n_row;
		new_mat = isl_mat_alloc(mat->ctx, n_row, n_col);
		if (!new_mat)
			goto error;
		for (i = 0; i < mat->n_row; ++i)
			isl_seq_cpy(new_mat->row[i], mat->row[i], mat->n_col);
		isl_mat_free(mat);
		return new_mat;
	}

	mat = isl_mat_cow(mat);
	if (!mat)
		goto error;

	old = mat->block.data;
	mat->block = isl_blk_extend(mat->ctx, mat->block, n_row * mat->max_col);
	if (isl_blk_is_error(mat->block))
		goto error;
	row = isl_realloc_array(mat->ctx, mat->row, isl_int *, n_row);
	if (n_row && !row)
		goto error;
	mat->row = row;

	for (i = 0; i < mat->n_row; ++i)
		mat->row[i] = mat->block.data + (mat->row[i] - old);
	for (i = mat->n_row; i < n_row; ++i)
		mat->row[i] = mat->block.data + i * mat->max_col;
	mat->n_row = n_row;
	if (mat->n_col < n_col)
		mat->n_col = n_col;

	return mat;
error:
	isl_mat_free(mat);
	return NULL;
}

__isl_give isl_mat *isl_mat_sub_alloc6(isl_ctx *ctx, isl_int **row,
	unsigned first_row, unsigned n_row, unsigned first_col, unsigned n_col)
{
	int i;
	struct isl_mat *mat;

	mat = isl_alloc_type(ctx, struct isl_mat);
	if (!mat)
		return NULL;
	mat->row = isl_alloc_array(ctx, isl_int *, n_row);
	if (n_row && !mat->row)
		goto error;
	for (i = 0; i < n_row; ++i)
		mat->row[i] = row[first_row+i] + first_col;
	mat->ctx = ctx;
	isl_ctx_ref(ctx);
	mat->ref = 1;
	mat->n_row = n_row;
	mat->n_col = n_col;
	mat->block = isl_blk_empty();
	mat->flags = ISL_MAT_BORROWED;
	return mat;
error:
	free(mat);
	return NULL;
}

__isl_give isl_mat *isl_mat_sub_alloc(__isl_keep isl_mat *mat,
	unsigned first_row, unsigned n_row, unsigned first_col, unsigned n_col)
{
	if (!mat)
		return NULL;
	return isl_mat_sub_alloc6(mat->ctx, mat->row, first_row, n_row,
				  first_col, n_col);
}

void isl_mat_sub_copy(struct isl_ctx *ctx, isl_int **dst, isl_int **src,
	unsigned n_row, unsigned dst_col, unsigned src_col, unsigned n_col)
{
	int i;

	for (i = 0; i < n_row; ++i)
		isl_seq_cpy(dst[i]+dst_col, src[i]+src_col, n_col);
}

void isl_mat_sub_neg(struct isl_ctx *ctx, isl_int **dst, isl_int **src,
	unsigned n_row, unsigned dst_col, unsigned src_col, unsigned n_col)
{
	int i;

	for (i = 0; i < n_row; ++i)
		isl_seq_neg(dst[i]+dst_col, src[i]+src_col, n_col);
}

struct isl_mat *isl_mat_copy(struct isl_mat *mat)
{
	if (!mat)
		return NULL;

	mat->ref++;
	return mat;
}

struct isl_mat *isl_mat_dup(struct isl_mat *mat)
{
	int i;
	struct isl_mat *mat2;

	if (!mat)
		return NULL;
	mat2 = isl_mat_alloc(mat->ctx, mat->n_row, mat->n_col);
	if (!mat2)
		return NULL;
	for (i = 0; i < mat->n_row; ++i)
		isl_seq_cpy(mat2->row[i], mat->row[i], mat->n_col);
	return mat2;
}

struct isl_mat *isl_mat_cow(struct isl_mat *mat)
{
	struct isl_mat *mat2;
	if (!mat)
		return NULL;

	if (mat->ref == 1 && !ISL_F_ISSET(mat, ISL_MAT_BORROWED))
		return mat;

	mat2 = isl_mat_dup(mat);
	isl_mat_free(mat);
	return mat2;
}

__isl_null isl_mat *isl_mat_free(__isl_take isl_mat *mat)
{
	if (!mat)
		return NULL;

	if (--mat->ref > 0)
		return NULL;

	if (!ISL_F_ISSET(mat, ISL_MAT_BORROWED))
		isl_blk_free(mat->ctx, mat->block);
	isl_ctx_deref(mat->ctx);
	free(mat->row);
	free(mat);

	return NULL;
}

int isl_mat_rows(__isl_keep isl_mat *mat)
{
	return mat ? mat->n_row : -1;
}

int isl_mat_cols(__isl_keep isl_mat *mat)
{
	return mat ? mat->n_col : -1;
}

/* Check that "col" is a valid column position for "mat".
 */
static isl_stat check_col(__isl_keep isl_mat *mat, int col)
{
	if (!mat)
		return isl_stat_error;
	if (col < 0 || col >= mat->n_col)
		isl_die(isl_mat_get_ctx(mat), isl_error_invalid,
			"column out of range", return isl_stat_error);
	return isl_stat_ok;
}

int isl_mat_get_element(__isl_keep isl_mat *mat, int row, int col, isl_int *v)
{
	if (!mat)
		return -1;
	if (row < 0 || row >= mat->n_row)
		isl_die(mat->ctx, isl_error_invalid, "row out of range",
			return -1);
	if (check_col(mat, col) < 0)
		return -1;
	isl_int_set(*v, mat->row[row][col]);
	return 0;
}

/* Extract the element at row "row", oolumn "col" of "mat".
 */
__isl_give isl_val *isl_mat_get_element_val(__isl_keep isl_mat *mat,
	int row, int col)
{
	isl_ctx *ctx;

	if (!mat)
		return NULL;
	ctx = isl_mat_get_ctx(mat);
	if (row < 0 || row >= mat->n_row)
		isl_die(ctx, isl_error_invalid, "row out of range",
			return NULL);
	if (check_col(mat, col) < 0)
		return NULL;
	return isl_val_int_from_isl_int(ctx, mat->row[row][col]);
}

__isl_give isl_mat *isl_mat_set_element(__isl_take isl_mat *mat,
	int row, int col, isl_int v)
{
	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;
	if (row < 0 || row >= mat->n_row)
		isl_die(mat->ctx, isl_error_invalid, "row out of range",
			goto error);
	if (check_col(mat, col) < 0)
		return isl_mat_free(mat);
	isl_int_set(mat->row[row][col], v);
	return mat;
error:
	isl_mat_free(mat);
	return NULL;
}

__isl_give isl_mat *isl_mat_set_element_si(__isl_take isl_mat *mat,
	int row, int col, int v)
{
	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;
	if (row < 0 || row >= mat->n_row)
		isl_die(mat->ctx, isl_error_invalid, "row out of range",
			goto error);
	if (check_col(mat, col) < 0)
		return isl_mat_free(mat);
	isl_int_set_si(mat->row[row][col], v);
	return mat;
error:
	isl_mat_free(mat);
	return NULL;
}

/* Replace the element at row "row", column "col" of "mat" by "v".
 */
__isl_give isl_mat *isl_mat_set_element_val(__isl_take isl_mat *mat,
	int row, int col, __isl_take isl_val *v)
{
	if (!v)
		return isl_mat_free(mat);
	if (!isl_val_is_int(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting integer value", goto error);
	mat = isl_mat_set_element(mat, row, col, v->n);
	isl_val_free(v);
	return mat;
error:
	isl_val_free(v);
	return isl_mat_free(mat);
}

__isl_give isl_mat *isl_mat_diag(isl_ctx *ctx, unsigned n_row, isl_int d)
{
	int i;
	struct isl_mat *mat;

	mat = isl_mat_alloc(ctx, n_row, n_row);
	if (!mat)
		return NULL;
	for (i = 0; i < n_row; ++i) {
		isl_seq_clr(mat->row[i], i);
		isl_int_set(mat->row[i][i], d);
		isl_seq_clr(mat->row[i]+i+1, n_row-(i+1));
	}

	return mat;
}

/* Create an "n_row" by "n_col" matrix with zero elements.
 */
__isl_give isl_mat *isl_mat_zero(isl_ctx *ctx, unsigned n_row, unsigned n_col)
{
	int i;
	isl_mat *mat;

	mat = isl_mat_alloc(ctx, n_row, n_col);
	if (!mat)
		return NULL;
	for (i = 0; i < n_row; ++i)
		isl_seq_clr(mat->row[i], n_col);

	return mat;
}

__isl_give isl_mat *isl_mat_identity(isl_ctx *ctx, unsigned n_row)
{
	if (!ctx)
		return NULL;
	return isl_mat_diag(ctx, n_row, ctx->one);
}

/* Is "mat" a (possibly scaled) identity matrix?
 */
int isl_mat_is_scaled_identity(__isl_keep isl_mat *mat)
{
	int i;

	if (!mat)
		return -1;
	if (mat->n_row != mat->n_col)
		return 0;

	for (i = 0; i < mat->n_row; ++i) {
		if (isl_seq_first_non_zero(mat->row[i], i) != -1)
			return 0;
		if (isl_int_ne(mat->row[0][0], mat->row[i][i]))
			return 0;
		if (isl_seq_first_non_zero(mat->row[i] + i + 1,
					    mat->n_col - (i + 1)) != -1)
			return 0;
	}

	return 1;
}

struct isl_vec *isl_mat_vec_product(struct isl_mat *mat, struct isl_vec *vec)
{
	int i;
	struct isl_vec *prod;

	if (!mat || !vec)
		goto error;

	isl_assert(mat->ctx, mat->n_col == vec->size, goto error);

	prod = isl_vec_alloc(mat->ctx, mat->n_row);
	if (!prod)
		goto error;

	for (i = 0; i < prod->size; ++i)
		isl_seq_inner_product(mat->row[i], vec->el, vec->size,
					&prod->block.data[i]);
	isl_mat_free(mat);
	isl_vec_free(vec);
	return prod;
error:
	isl_mat_free(mat);
	isl_vec_free(vec);
	return NULL;
}

__isl_give isl_vec *isl_mat_vec_inverse_product(__isl_take isl_mat *mat,
	__isl_take isl_vec *vec)
{
	struct isl_mat *vec_mat;
	int i;

	if (!mat || !vec)
		goto error;
	vec_mat = isl_mat_alloc(vec->ctx, vec->size, 1);
	if (!vec_mat)
		goto error;
	for (i = 0; i < vec->size; ++i)
		isl_int_set(vec_mat->row[i][0], vec->el[i]);
	vec_mat = isl_mat_inverse_product(mat, vec_mat);
	isl_vec_free(vec);
	if (!vec_mat)
		return NULL;
	vec = isl_vec_alloc(vec_mat->ctx, vec_mat->n_row);
	if (vec)
		for (i = 0; i < vec->size; ++i)
			isl_int_set(vec->el[i], vec_mat->row[i][0]);
	isl_mat_free(vec_mat);
	return vec;
error:
	isl_mat_free(mat);
	isl_vec_free(vec);
	return NULL;
}

struct isl_vec *isl_vec_mat_product(struct isl_vec *vec, struct isl_mat *mat)
{
	int i, j;
	struct isl_vec *prod;

	if (!mat || !vec)
		goto error;

	isl_assert(mat->ctx, mat->n_row == vec->size, goto error);

	prod = isl_vec_alloc(mat->ctx, mat->n_col);
	if (!prod)
		goto error;

	for (i = 0; i < prod->size; ++i) {
		isl_int_set_si(prod->el[i], 0);
		for (j = 0; j < vec->size; ++j)
			isl_int_addmul(prod->el[i], vec->el[j], mat->row[j][i]);
	}
	isl_mat_free(mat);
	isl_vec_free(vec);
	return prod;
error:
	isl_mat_free(mat);
	isl_vec_free(vec);
	return NULL;
}

struct isl_mat *isl_mat_aff_direct_sum(struct isl_mat *left,
	struct isl_mat *right)
{
	int i;
	struct isl_mat *sum;

	if (!left || !right)
		goto error;

	isl_assert(left->ctx, left->n_row == right->n_row, goto error);
	isl_assert(left->ctx, left->n_row >= 1, goto error);
	isl_assert(left->ctx, left->n_col >= 1, goto error);
	isl_assert(left->ctx, right->n_col >= 1, goto error);
	isl_assert(left->ctx,
	    isl_seq_first_non_zero(left->row[0]+1, left->n_col-1) == -1,
	    goto error);
	isl_assert(left->ctx,
	    isl_seq_first_non_zero(right->row[0]+1, right->n_col-1) == -1,
	    goto error);

	sum = isl_mat_alloc(left->ctx, left->n_row, left->n_col + right->n_col - 1);
	if (!sum)
		goto error;
	isl_int_lcm(sum->row[0][0], left->row[0][0], right->row[0][0]);
	isl_int_divexact(left->row[0][0], sum->row[0][0], left->row[0][0]);
	isl_int_divexact(right->row[0][0], sum->row[0][0], right->row[0][0]);

	isl_seq_clr(sum->row[0]+1, sum->n_col-1);
	for (i = 1; i < sum->n_row; ++i) {
		isl_int_mul(sum->row[i][0], left->row[0][0], left->row[i][0]);
		isl_int_addmul(sum->row[i][0],
				right->row[0][0], right->row[i][0]);
		isl_seq_scale(sum->row[i]+1, left->row[i]+1, left->row[0][0],
				left->n_col-1);
		isl_seq_scale(sum->row[i]+left->n_col,
				right->row[i]+1, right->row[0][0],
				right->n_col-1);
	}

	isl_int_divexact(left->row[0][0], sum->row[0][0], left->row[0][0]);
	isl_int_divexact(right->row[0][0], sum->row[0][0], right->row[0][0]);
	isl_mat_free(left);
	isl_mat_free(right);
	return sum;
error:
	isl_mat_free(left);
	isl_mat_free(right);
	return NULL;
}

static void exchange(struct isl_mat *M, struct isl_mat **U,
	struct isl_mat **Q, unsigned row, unsigned i, unsigned j)
{
	int r;
	for (r = row; r < M->n_row; ++r)
		isl_int_swap(M->row[r][i], M->row[r][j]);
	if (U) {
		for (r = 0; r < (*U)->n_row; ++r)
			isl_int_swap((*U)->row[r][i], (*U)->row[r][j]);
	}
	if (Q)
		isl_mat_swap_rows(*Q, i, j);
}

static void subtract(struct isl_mat *M, struct isl_mat **U,
	struct isl_mat **Q, unsigned row, unsigned i, unsigned j, isl_int m)
{
	int r;
	for (r = row; r < M->n_row; ++r)
		isl_int_submul(M->row[r][j], m, M->row[r][i]);
	if (U) {
		for (r = 0; r < (*U)->n_row; ++r)
			isl_int_submul((*U)->row[r][j], m, (*U)->row[r][i]);
	}
	if (Q) {
		for (r = 0; r < (*Q)->n_col; ++r)
			isl_int_addmul((*Q)->row[i][r], m, (*Q)->row[j][r]);
	}
}

static void oppose(struct isl_mat *M, struct isl_mat **U,
	struct isl_mat **Q, unsigned row, unsigned col)
{
	int r;
	for (r = row; r < M->n_row; ++r)
		isl_int_neg(M->row[r][col], M->row[r][col]);
	if (U) {
		for (r = 0; r < (*U)->n_row; ++r)
			isl_int_neg((*U)->row[r][col], (*U)->row[r][col]);
	}
	if (Q)
		isl_seq_neg((*Q)->row[col], (*Q)->row[col], (*Q)->n_col);
}

/* Given matrix M, compute
 *
 *		M U = H
 *		M   = H Q
 *
 * with U and Q unimodular matrices and H a matrix in column echelon form
 * such that on each echelon row the entries in the non-echelon column
 * are non-negative (if neg == 0) or non-positive (if neg == 1)
 * and strictly smaller (in absolute value) than the entries in the echelon
 * column.
 * If U or Q are NULL, then these matrices are not computed.
 */
struct isl_mat *isl_mat_left_hermite(struct isl_mat *M, int neg,
	struct isl_mat **U, struct isl_mat **Q)
{
	isl_int c;
	int row, col;

	if (U)
		*U = NULL;
	if (Q)
		*Q = NULL;
	if (!M)
		goto error;
	M = isl_mat_cow(M);
	if (!M)
		goto error;
	if (U) {
		*U = isl_mat_identity(M->ctx, M->n_col);
		if (!*U)
			goto error;
	}
	if (Q) {
		*Q = isl_mat_identity(M->ctx, M->n_col);
		if (!*Q)
			goto error;
	}

	col = 0;
	isl_int_init(c);
	for (row = 0; row < M->n_row; ++row) {
		int first, i, off;
		first = isl_seq_abs_min_non_zero(M->row[row]+col, M->n_col-col);
		if (first == -1)
			continue;
		first += col;
		if (first != col)
			exchange(M, U, Q, row, first, col);
		if (isl_int_is_neg(M->row[row][col]))
			oppose(M, U, Q, row, col);
		first = col+1;
		while ((off = isl_seq_first_non_zero(M->row[row]+first,
						       M->n_col-first)) != -1) {
			first += off;
			isl_int_fdiv_q(c, M->row[row][first], M->row[row][col]);
			subtract(M, U, Q, row, col, first, c);
			if (!isl_int_is_zero(M->row[row][first]))
				exchange(M, U, Q, row, first, col);
			else
				++first;
		}
		for (i = 0; i < col; ++i) {
			if (isl_int_is_zero(M->row[row][i]))
				continue;
			if (neg)
				isl_int_cdiv_q(c, M->row[row][i], M->row[row][col]);
			else
				isl_int_fdiv_q(c, M->row[row][i], M->row[row][col]);
			if (isl_int_is_zero(c))
				continue;
			subtract(M, U, Q, row, col, i, c);
		}
		++col;
	}
	isl_int_clear(c);

	return M;
error:
	if (Q) {
		isl_mat_free(*Q);
		*Q = NULL;
	}
	if (U) {
		isl_mat_free(*U);
		*U = NULL;
	}
	isl_mat_free(M);
	return NULL;
}

struct isl_mat *isl_mat_right_kernel(struct isl_mat *mat)
{
	int i, rank;
	struct isl_mat *U = NULL;
	struct isl_mat *K;

	mat = isl_mat_left_hermite(mat, 0, &U, NULL);
	if (!mat || !U)
		goto error;

	for (i = 0, rank = 0; rank < mat->n_col; ++rank) {
		while (i < mat->n_row && isl_int_is_zero(mat->row[i][rank]))
			++i;
		if (i >= mat->n_row)
			break;
	}
	K = isl_mat_alloc(U->ctx, U->n_row, U->n_col - rank);
	if (!K)
		goto error;
	isl_mat_sub_copy(K->ctx, K->row, U->row, U->n_row, 0, rank, U->n_col-rank);
	isl_mat_free(mat);
	isl_mat_free(U);
	return K;
error:
	isl_mat_free(mat);
	isl_mat_free(U);
	return NULL;
}

struct isl_mat *isl_mat_lin_to_aff(struct isl_mat *mat)
{
	int i;
	struct isl_mat *mat2;

	if (!mat)
		return NULL;
	mat2 = isl_mat_alloc(mat->ctx, 1+mat->n_row, 1+mat->n_col);
	if (!mat2)
		goto error;
	isl_int_set_si(mat2->row[0][0], 1);
	isl_seq_clr(mat2->row[0]+1, mat->n_col);
	for (i = 0; i < mat->n_row; ++i) {
		isl_int_set_si(mat2->row[1+i][0], 0);
		isl_seq_cpy(mat2->row[1+i]+1, mat->row[i], mat->n_col);
	}
	isl_mat_free(mat);
	return mat2;
error:
	isl_mat_free(mat);
	return NULL;
}

/* Given two matrices M1 and M2, return the block matrix
 *
 *	[ M1  0  ]
 *	[ 0   M2 ]
 */
__isl_give isl_mat *isl_mat_diagonal(__isl_take isl_mat *mat1,
	__isl_take isl_mat *mat2)
{
	int i;
	isl_mat *mat;

	if (!mat1 || !mat2)
		goto error;

	mat = isl_mat_alloc(mat1->ctx, mat1->n_row + mat2->n_row,
				       mat1->n_col + mat2->n_col);
	if (!mat)
		goto error;
	for (i = 0; i < mat1->n_row; ++i) {
		isl_seq_cpy(mat->row[i], mat1->row[i], mat1->n_col);
		isl_seq_clr(mat->row[i] + mat1->n_col, mat2->n_col);
	}
	for (i = 0; i < mat2->n_row; ++i) {
		isl_seq_clr(mat->row[mat1->n_row + i], mat1->n_col);
		isl_seq_cpy(mat->row[mat1->n_row + i] + mat1->n_col,
						    mat2->row[i], mat2->n_col);
	}
	isl_mat_free(mat1);
	isl_mat_free(mat2);
	return mat;
error:
	isl_mat_free(mat1);
	isl_mat_free(mat2);
	return NULL;
}

static int row_first_non_zero(isl_int **row, unsigned n_row, unsigned col)
{
	int i;

	for (i = 0; i < n_row; ++i)
		if (!isl_int_is_zero(row[i][col]))
			return i;
	return -1;
}

static int row_abs_min_non_zero(isl_int **row, unsigned n_row, unsigned col)
{
	int i, min = row_first_non_zero(row, n_row, col);
	if (min < 0)
		return -1;
	for (i = min + 1; i < n_row; ++i) {
		if (isl_int_is_zero(row[i][col]))
			continue;
		if (isl_int_abs_lt(row[i][col], row[min][col]))
			min = i;
	}
	return min;
}

static void inv_exchange(struct isl_mat *left, struct isl_mat *right,
	unsigned i, unsigned j)
{
	left = isl_mat_swap_rows(left, i, j);
	right = isl_mat_swap_rows(right, i, j);
}

static void inv_oppose(
	struct isl_mat *left, struct isl_mat *right, unsigned row)
{
	isl_seq_neg(left->row[row]+row, left->row[row]+row, left->n_col-row);
	isl_seq_neg(right->row[row], right->row[row], right->n_col);
}

static void inv_subtract(struct isl_mat *left, struct isl_mat *right,
	unsigned row, unsigned i, isl_int m)
{
	isl_int_neg(m, m);
	isl_seq_combine(left->row[i]+row,
			left->ctx->one, left->row[i]+row,
			m, left->row[row]+row,
			left->n_col-row);
	isl_seq_combine(right->row[i], right->ctx->one, right->row[i],
			m, right->row[row], right->n_col);
}

/* Compute inv(left)*right
 */
struct isl_mat *isl_mat_inverse_product(struct isl_mat *left,
	struct isl_mat *right)
{
	int row;
	isl_int a, b;

	if (!left || !right)
		goto error;

	isl_assert(left->ctx, left->n_row == left->n_col, goto error);
	isl_assert(left->ctx, left->n_row == right->n_row, goto error);

	if (left->n_row == 0) {
		isl_mat_free(left);
		return right;
	}

	left = isl_mat_cow(left);
	right = isl_mat_cow(right);
	if (!left || !right)
		goto error;

	isl_int_init(a);
	isl_int_init(b);
	for (row = 0; row < left->n_row; ++row) {
		int pivot, first, i, off;
		pivot = row_abs_min_non_zero(left->row+row, left->n_row-row, row);
		if (pivot < 0) {
			isl_int_clear(a);
			isl_int_clear(b);
			isl_assert(left->ctx, pivot >= 0, goto error);
		}
		pivot += row;
		if (pivot != row)
			inv_exchange(left, right, pivot, row);
		if (isl_int_is_neg(left->row[row][row]))
			inv_oppose(left, right, row);
		first = row+1;
		while ((off = row_first_non_zero(left->row+first,
					left->n_row-first, row)) != -1) {
			first += off;
			isl_int_fdiv_q(a, left->row[first][row],
					left->row[row][row]);
			inv_subtract(left, right, row, first, a);
			if (!isl_int_is_zero(left->row[first][row]))
				inv_exchange(left, right, row, first);
			else
				++first;
		}
		for (i = 0; i < row; ++i) {
			if (isl_int_is_zero(left->row[i][row]))
				continue;
			isl_int_gcd(a, left->row[row][row], left->row[i][row]);
			isl_int_divexact(b, left->row[i][row], a);
			isl_int_divexact(a, left->row[row][row], a);
			isl_int_neg(b, b);
			isl_seq_combine(left->row[i] + i,
					a, left->row[i] + i,
					b, left->row[row] + i,
					left->n_col - i);
			isl_seq_combine(right->row[i], a, right->row[i],
					b, right->row[row], right->n_col);
		}
	}
	isl_int_clear(b);

	isl_int_set(a, left->row[0][0]);
	for (row = 1; row < left->n_row; ++row)
		isl_int_lcm(a, a, left->row[row][row]);
	if (isl_int_is_zero(a)){
		isl_int_clear(a);
		isl_assert(left->ctx, 0, goto error);
	}
	for (row = 0; row < left->n_row; ++row) {
		isl_int_divexact(left->row[row][row], a, left->row[row][row]);
		if (isl_int_is_one(left->row[row][row]))
			continue;
		isl_seq_scale(right->row[row], right->row[row],
				left->row[row][row], right->n_col);
	}
	isl_int_clear(a);

	isl_mat_free(left);
	return right;
error:
	isl_mat_free(left);
	isl_mat_free(right);
	return NULL;
}

void isl_mat_col_scale(struct isl_mat *mat, unsigned col, isl_int m)
{
	int i;

	for (i = 0; i < mat->n_row; ++i)
		isl_int_mul(mat->row[i][col], mat->row[i][col], m);
}

void isl_mat_col_combine(struct isl_mat *mat, unsigned dst,
	isl_int m1, unsigned src1, isl_int m2, unsigned src2)
{
	int i;
	isl_int tmp;

	isl_int_init(tmp);
	for (i = 0; i < mat->n_row; ++i) {
		isl_int_mul(tmp, m1, mat->row[i][src1]);
		isl_int_addmul(tmp, m2, mat->row[i][src2]);
		isl_int_set(mat->row[i][dst], tmp);
	}
	isl_int_clear(tmp);
}

struct isl_mat *isl_mat_right_inverse(struct isl_mat *mat)
{
	struct isl_mat *inv;
	int row;
	isl_int a, b;

	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;

	inv = isl_mat_identity(mat->ctx, mat->n_col);
	inv = isl_mat_cow(inv);
	if (!inv)
		goto error;

	isl_int_init(a);
	isl_int_init(b);
	for (row = 0; row < mat->n_row; ++row) {
		int pivot, first, i, off;
		pivot = isl_seq_abs_min_non_zero(mat->row[row]+row, mat->n_col-row);
		if (pivot < 0) {
			isl_int_clear(a);
			isl_int_clear(b);
			isl_assert(mat->ctx, pivot >= 0, goto error);
		}
		pivot += row;
		if (pivot != row)
			exchange(mat, &inv, NULL, row, pivot, row);
		if (isl_int_is_neg(mat->row[row][row]))
			oppose(mat, &inv, NULL, row, row);
		first = row+1;
		while ((off = isl_seq_first_non_zero(mat->row[row]+first,
						    mat->n_col-first)) != -1) {
			first += off;
			isl_int_fdiv_q(a, mat->row[row][first],
						    mat->row[row][row]);
			subtract(mat, &inv, NULL, row, row, first, a);
			if (!isl_int_is_zero(mat->row[row][first]))
				exchange(mat, &inv, NULL, row, row, first);
			else
				++first;
		}
		for (i = 0; i < row; ++i) {
			if (isl_int_is_zero(mat->row[row][i]))
				continue;
			isl_int_gcd(a, mat->row[row][row], mat->row[row][i]);
			isl_int_divexact(b, mat->row[row][i], a);
			isl_int_divexact(a, mat->row[row][row], a);
			isl_int_neg(a, a);
			isl_mat_col_combine(mat, i, a, i, b, row);
			isl_mat_col_combine(inv, i, a, i, b, row);
		}
	}
	isl_int_clear(b);

	isl_int_set(a, mat->row[0][0]);
	for (row = 1; row < mat->n_row; ++row)
		isl_int_lcm(a, a, mat->row[row][row]);
	if (isl_int_is_zero(a)){
		isl_int_clear(a);
		goto error;
	}
	for (row = 0; row < mat->n_row; ++row) {
		isl_int_divexact(mat->row[row][row], a, mat->row[row][row]);
		if (isl_int_is_one(mat->row[row][row]))
			continue;
		isl_mat_col_scale(inv, row, mat->row[row][row]);
	}
	isl_int_clear(a);

	isl_mat_free(mat);

	return inv;
error:
	isl_mat_free(mat);
	isl_mat_free(inv);
	return NULL;
}

struct isl_mat *isl_mat_transpose(struct isl_mat *mat)
{
	struct isl_mat *transpose = NULL;
	int i, j;

	if (!mat)
		return NULL;

	if (mat->n_col == mat->n_row) {
		mat = isl_mat_cow(mat);
		if (!mat)
			return NULL;
		for (i = 0; i < mat->n_row; ++i)
			for (j = i + 1; j < mat->n_col; ++j)
				isl_int_swap(mat->row[i][j], mat->row[j][i]);
		return mat;
	}
	transpose = isl_mat_alloc(mat->ctx, mat->n_col, mat->n_row);
	if (!transpose)
		goto error;
	for (i = 0; i < mat->n_row; ++i)
		for (j = 0; j < mat->n_col; ++j)
			isl_int_set(transpose->row[j][i], mat->row[i][j]);
	isl_mat_free(mat);
	return transpose;
error:
	isl_mat_free(mat);
	return NULL;
}

struct isl_mat *isl_mat_swap_cols(struct isl_mat *mat, unsigned i, unsigned j)
{
	int r;

	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;
	isl_assert(mat->ctx, i < mat->n_col, goto error);
	isl_assert(mat->ctx, j < mat->n_col, goto error);

	for (r = 0; r < mat->n_row; ++r)
		isl_int_swap(mat->row[r][i], mat->row[r][j]);
	return mat;
error:
	isl_mat_free(mat);
	return NULL;
}

struct isl_mat *isl_mat_swap_rows(struct isl_mat *mat, unsigned i, unsigned j)
{
	isl_int *t;

	if (!mat)
		return NULL;
	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;
	t = mat->row[i];
	mat->row[i] = mat->row[j];
	mat->row[j] = t;
	return mat;
}

/* Calculate the product of two matrices.
 *
 * This function is optimized for operand matrices that contain many zeros and
 * skips multiplications where we know one of the operands is zero.
 */
__isl_give isl_mat *isl_mat_product(__isl_take isl_mat *left,
	__isl_take isl_mat *right)
{
	int i, j, k;
	struct isl_mat *prod;

	if (!left || !right)
		goto error;
	isl_assert(left->ctx, left->n_col == right->n_row, goto error);
	prod = isl_mat_alloc(left->ctx, left->n_row, right->n_col);
	if (!prod)
		goto error;
	if (left->n_col == 0) {
		for (i = 0; i < prod->n_row; ++i)
			isl_seq_clr(prod->row[i], prod->n_col);
		isl_mat_free(left);
		isl_mat_free(right);
		return prod;
	}
	for (i = 0; i < prod->n_row; ++i) {
		for (j = 0; j < prod->n_col; ++j)
			isl_int_mul(prod->row[i][j],
				    left->row[i][0], right->row[0][j]);
		for (k = 1; k < left->n_col; ++k) {
			if (isl_int_is_zero(left->row[i][k]))
				continue;
			for (j = 0; j < prod->n_col; ++j)
				isl_int_addmul(prod->row[i][j],
					    left->row[i][k], right->row[k][j]);
		}
	}
	isl_mat_free(left);
	isl_mat_free(right);
	return prod;
error:
	isl_mat_free(left);
	isl_mat_free(right);
	return NULL;
}

/* Replace the variables x in the rows q by x' given by x = M x',
 * with M the matrix mat.
 *
 * If the number of new variables is greater than the original
 * number of variables, then the rows q have already been
 * preextended.  If the new number is smaller, then the coefficients
 * of the divs, which are not changed, need to be shifted down.
 * The row q may be the equalities, the inequalities or the
 * div expressions.  In the latter case, has_div is true and
 * we need to take into account the extra denominator column.
 */
static int preimage(struct isl_ctx *ctx, isl_int **q, unsigned n,
	unsigned n_div, int has_div, struct isl_mat *mat)
{
	int i;
	struct isl_mat *t;
	int e;

	if (mat->n_col >= mat->n_row)
		e = 0;
	else
		e = mat->n_row - mat->n_col;
	if (has_div)
		for (i = 0; i < n; ++i)
			isl_int_mul(q[i][0], q[i][0], mat->row[0][0]);
	t = isl_mat_sub_alloc6(mat->ctx, q, 0, n, has_div, mat->n_row);
	t = isl_mat_product(t, mat);
	if (!t)
		return -1;
	for (i = 0; i < n; ++i) {
		isl_seq_swp_or_cpy(q[i] + has_div, t->row[i], t->n_col);
		isl_seq_cpy(q[i] + has_div + t->n_col,
			    q[i] + has_div + t->n_col + e, n_div);
		isl_seq_clr(q[i] + has_div + t->n_col + n_div, e);
	}
	isl_mat_free(t);
	return 0;
}

/* Replace the variables x in bset by x' given by x = M x', with
 * M the matrix mat.
 *
 * If there are fewer variables x' then there are x, then we perform
 * the transformation in place, which means that, in principle,
 * this frees up some extra variables as the number
 * of columns remains constant, but we would have to extend
 * the div array too as the number of rows in this array is assumed
 * to be equal to extra.
 */
struct isl_basic_set *isl_basic_set_preimage(struct isl_basic_set *bset,
	struct isl_mat *mat)
{
	struct isl_ctx *ctx;

	if (!bset || !mat)
		goto error;

	ctx = bset->ctx;
	bset = isl_basic_set_cow(bset);
	if (!bset)
		goto error;

	isl_assert(ctx, bset->dim->nparam == 0, goto error);
	isl_assert(ctx, 1+bset->dim->n_out == mat->n_row, goto error);
	isl_assert(ctx, mat->n_col > 0, goto error);

	if (mat->n_col > mat->n_row) {
		bset = isl_basic_set_extend(bset, 0, mat->n_col-1, 0, 0, 0);
		if (!bset)
			goto error;
	} else if (mat->n_col < mat->n_row) {
		bset->dim = isl_space_cow(bset->dim);
		if (!bset->dim)
			goto error;
		bset->dim->n_out -= mat->n_row - mat->n_col;
	}

	if (preimage(ctx, bset->eq, bset->n_eq, bset->n_div, 0,
			isl_mat_copy(mat)) < 0)
		goto error;

	if (preimage(ctx, bset->ineq, bset->n_ineq, bset->n_div, 0,
			isl_mat_copy(mat)) < 0)
		goto error;

	if (preimage(ctx, bset->div, bset->n_div, bset->n_div, 1, mat) < 0)
		goto error2;

	ISL_F_CLR(bset, ISL_BASIC_SET_NO_IMPLICIT);
	ISL_F_CLR(bset, ISL_BASIC_SET_NO_REDUNDANT);
	ISL_F_CLR(bset, ISL_BASIC_SET_NORMALIZED);
	ISL_F_CLR(bset, ISL_BASIC_SET_NORMALIZED_DIVS);
	ISL_F_CLR(bset, ISL_BASIC_SET_ALL_EQUALITIES);

	bset = isl_basic_set_simplify(bset);
	bset = isl_basic_set_finalize(bset);

	return bset;
error:
	isl_mat_free(mat);
error2:
	isl_basic_set_free(bset);
	return NULL;
}

struct isl_set *isl_set_preimage(struct isl_set *set, struct isl_mat *mat)
{
	int i;

	set = isl_set_cow(set);
	if (!set)
		goto error;

	for (i = 0; i < set->n; ++i) {
		set->p[i] = isl_basic_set_preimage(set->p[i],
						    isl_mat_copy(mat));
		if (!set->p[i])
			goto error;
	}
	if (mat->n_col != mat->n_row) {
		set->dim = isl_space_cow(set->dim);
		if (!set->dim)
			goto error;
		set->dim->n_out += mat->n_col;
		set->dim->n_out -= mat->n_row;
	}
	isl_mat_free(mat);
	ISL_F_CLR(set, ISL_SET_NORMALIZED);
	return set;
error:
	isl_set_free(set);
	isl_mat_free(mat);
	return NULL;
}

/* Replace the variables x starting at pos in the rows q
 * by x' with x = M x' with M the matrix mat.
 * That is, replace the corresponding coefficients c by c M.
 */
static int transform(isl_ctx *ctx, isl_int **q, unsigned n,
	unsigned pos, __isl_take isl_mat *mat)
{
	int i;
	isl_mat *t;

	t = isl_mat_sub_alloc6(ctx, q, 0, n, pos, mat->n_row);
	t = isl_mat_product(t, mat);
	if (!t)
		return -1;
	for (i = 0; i < n; ++i)
		isl_seq_swp_or_cpy(q[i] + pos, t->row[i], t->n_col);
	isl_mat_free(t);
	return 0;
}

/* Replace the variables x of type "type" starting at "first" in "bmap"
 * by x' with x = M x' with M the matrix trans.
 * That is, replace the corresponding coefficients c by c M.
 *
 * The transformation matrix should be a square matrix.
 */
__isl_give isl_basic_map *isl_basic_map_transform_dims(
	__isl_take isl_basic_map *bmap, enum isl_dim_type type, unsigned first,
	__isl_take isl_mat *trans)
{
	isl_ctx *ctx;
	unsigned pos;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap || !trans)
		goto error;

	ctx = isl_basic_map_get_ctx(bmap);
	if (trans->n_row != trans->n_col)
		isl_die(trans->ctx, isl_error_invalid,
			"expecting square transformation matrix", goto error);
	if (first + trans->n_row > isl_basic_map_dim(bmap, type))
		isl_die(trans->ctx, isl_error_invalid,
			"oversized transformation matrix", goto error);

	pos = isl_basic_map_offset(bmap, type) + first;

	if (transform(ctx, bmap->eq, bmap->n_eq, pos, isl_mat_copy(trans)) < 0)
		goto error;
	if (transform(ctx, bmap->ineq, bmap->n_ineq, pos,
		      isl_mat_copy(trans)) < 0)
		goto error;
	if (transform(ctx, bmap->div, bmap->n_div, 1 + pos,
		      isl_mat_copy(trans)) < 0)
		goto error;

	ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED);
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED_DIVS);

	isl_mat_free(trans);
	return bmap;
error:
	isl_mat_free(trans);
	isl_basic_map_free(bmap);
	return NULL;
}

/* Replace the variables x of type "type" starting at "first" in "bset"
 * by x' with x = M x' with M the matrix trans.
 * That is, replace the corresponding coefficients c by c M.
 *
 * The transformation matrix should be a square matrix.
 */
__isl_give isl_basic_set *isl_basic_set_transform_dims(
	__isl_take isl_basic_set *bset, enum isl_dim_type type, unsigned first,
	__isl_take isl_mat *trans)
{
	return isl_basic_map_transform_dims(bset, type, first, trans);
}

void isl_mat_print_internal(__isl_keep isl_mat *mat, FILE *out, int indent)
{
	int i, j;

	if (!mat) {
		fprintf(out, "%*snull mat\n", indent, "");
		return;
	}

	if (mat->n_row == 0)
		fprintf(out, "%*s[]\n", indent, "");

	for (i = 0; i < mat->n_row; ++i) {
		if (!i)
			fprintf(out, "%*s[[", indent, "");
		else
			fprintf(out, "%*s[", indent+1, "");
		for (j = 0; j < mat->n_col; ++j) {
			if (j)
			    fprintf(out, ",");
			isl_int_print(out, mat->row[i][j], 0);
		}
		if (i == mat->n_row-1)
			fprintf(out, "]]\n");
		else
			fprintf(out, "]\n");
	}
}

void isl_mat_dump(__isl_keep isl_mat *mat)
{
	isl_mat_print_internal(mat, stderr, 0);
}

struct isl_mat *isl_mat_drop_cols(struct isl_mat *mat, unsigned col, unsigned n)
{
	int r;

	if (n == 0)
		return mat;

	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;

	if (col != mat->n_col-n) {
		for (r = 0; r < mat->n_row; ++r)
			isl_seq_cpy(mat->row[r]+col, mat->row[r]+col+n,
					mat->n_col - col - n);
	}
	mat->n_col -= n;
	return mat;
}

struct isl_mat *isl_mat_drop_rows(struct isl_mat *mat, unsigned row, unsigned n)
{
	int r;

	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;

	for (r = row; r+n < mat->n_row; ++r)
		mat->row[r] = mat->row[r+n];

	mat->n_row -= n;
	return mat;
}

__isl_give isl_mat *isl_mat_insert_cols(__isl_take isl_mat *mat,
				unsigned col, unsigned n)
{
	isl_mat *ext;

	if (!mat)
		return NULL;
	if (n == 0)
		return mat;

	ext = isl_mat_alloc(mat->ctx, mat->n_row, mat->n_col + n);
	if (!ext)
		goto error;

	isl_mat_sub_copy(mat->ctx, ext->row, mat->row, mat->n_row, 0, 0, col);
	isl_mat_sub_copy(mat->ctx, ext->row, mat->row, mat->n_row,
				col + n, col, mat->n_col - col);

	isl_mat_free(mat);
	return ext;
error:
	isl_mat_free(mat);
	return NULL;
}

__isl_give isl_mat *isl_mat_insert_zero_cols(__isl_take isl_mat *mat,
	unsigned first, unsigned n)
{
	int i;

	if (!mat)
		return NULL;
	mat = isl_mat_insert_cols(mat, first, n);
	if (!mat)
		return NULL;

	for (i = 0; i < mat->n_row; ++i)
		isl_seq_clr(mat->row[i] + first, n);

	return mat;
}

__isl_give isl_mat *isl_mat_add_zero_cols(__isl_take isl_mat *mat, unsigned n)
{
	if (!mat)
		return NULL;

	return isl_mat_insert_zero_cols(mat, mat->n_col, n);
}

__isl_give isl_mat *isl_mat_insert_rows(__isl_take isl_mat *mat,
				unsigned row, unsigned n)
{
	isl_mat *ext;

	if (!mat)
		return NULL;
	if (n == 0)
		return mat;

	ext = isl_mat_alloc(mat->ctx, mat->n_row + n, mat->n_col);
	if (!ext)
		goto error;

	isl_mat_sub_copy(mat->ctx, ext->row, mat->row, row, 0, 0, mat->n_col);
	isl_mat_sub_copy(mat->ctx, ext->row + row + n, mat->row + row,
				mat->n_row - row, 0, 0, mat->n_col);

	isl_mat_free(mat);
	return ext;
error:
	isl_mat_free(mat);
	return NULL;
}

__isl_give isl_mat *isl_mat_add_rows(__isl_take isl_mat *mat, unsigned n)
{
	if (!mat)
		return NULL;

	return isl_mat_insert_rows(mat, mat->n_row, n);
}

__isl_give isl_mat *isl_mat_insert_zero_rows(__isl_take isl_mat *mat,
	unsigned row, unsigned n)
{
	int i;

	mat = isl_mat_insert_rows(mat, row, n);
	if (!mat)
		return NULL;
	
	for (i = 0; i < n; ++i)
		isl_seq_clr(mat->row[row + i], mat->n_col);

	return mat;
}

__isl_give isl_mat *isl_mat_add_zero_rows(__isl_take isl_mat *mat, unsigned n)
{
	if (!mat)
		return NULL;

	return isl_mat_insert_zero_rows(mat, mat->n_row, n);
}

void isl_mat_col_submul(struct isl_mat *mat,
			int dst_col, isl_int f, int src_col)
{
	int i;

	for (i = 0; i < mat->n_row; ++i)
		isl_int_submul(mat->row[i][dst_col], f, mat->row[i][src_col]);
}

void isl_mat_col_add(__isl_keep isl_mat *mat, int dst_col, int src_col)
{
	int i;

	if (!mat)
		return;

	for (i = 0; i < mat->n_row; ++i)
		isl_int_add(mat->row[i][dst_col],
			    mat->row[i][dst_col], mat->row[i][src_col]);
}

void isl_mat_col_mul(struct isl_mat *mat, int dst_col, isl_int f, int src_col)
{
	int i;

	for (i = 0; i < mat->n_row; ++i)
		isl_int_mul(mat->row[i][dst_col], f, mat->row[i][src_col]);
}

/* Add "f" times column "src_col" to column "dst_col" of "mat" and
 * return the result.
 */
__isl_give isl_mat *isl_mat_col_addmul(__isl_take isl_mat *mat, int dst_col,
	isl_int f, int src_col)
{
	int i;

	if (check_col(mat, dst_col) < 0 || check_col(mat, src_col) < 0)
		return isl_mat_free(mat);

	for (i = 0; i < mat->n_row; ++i) {
		if (isl_int_is_zero(mat->row[i][src_col]))
			continue;
		mat = isl_mat_cow(mat);
		if (!mat)
			return NULL;
		isl_int_addmul(mat->row[i][dst_col], f, mat->row[i][src_col]);
	}

	return mat;
}

/* Negate column "col" of "mat" and return the result.
 */
__isl_give isl_mat *isl_mat_col_neg(__isl_take isl_mat *mat, int col)
{
	int i;

	if (check_col(mat, col) < 0)
		return isl_mat_free(mat);

	for (i = 0; i < mat->n_row; ++i) {
		if (isl_int_is_zero(mat->row[i][col]))
			continue;
		mat = isl_mat_cow(mat);
		if (!mat)
			return NULL;
		isl_int_neg(mat->row[i][col], mat->row[i][col]);
	}

	return mat;
}

struct isl_mat *isl_mat_unimodular_complete(struct isl_mat *M, int row)
{
	int r;
	struct isl_mat *H = NULL, *Q = NULL;

	if (!M)
		return NULL;

	isl_assert(M->ctx, M->n_row == M->n_col, goto error);
	M->n_row = row;
	H = isl_mat_left_hermite(isl_mat_copy(M), 0, NULL, &Q);
	M->n_row = M->n_col;
	if (!H)
		goto error;
	for (r = 0; r < row; ++r)
		isl_assert(M->ctx, isl_int_is_one(H->row[r][r]), goto error);
	for (r = row; r < M->n_row; ++r)
		isl_seq_cpy(M->row[r], Q->row[r], M->n_col);
	isl_mat_free(H);
	isl_mat_free(Q);
	return M;
error:
	isl_mat_free(H);
	isl_mat_free(Q);
	isl_mat_free(M);
	return NULL;
}

__isl_give isl_mat *isl_mat_concat(__isl_take isl_mat *top,
	__isl_take isl_mat *bot)
{
	struct isl_mat *mat;

	if (!top || !bot)
		goto error;

	isl_assert(top->ctx, top->n_col == bot->n_col, goto error);
	if (top->n_row == 0) {
		isl_mat_free(top);
		return bot;
	}
	if (bot->n_row == 0) {
		isl_mat_free(bot);
		return top;
	}

	mat = isl_mat_alloc(top->ctx, top->n_row + bot->n_row, top->n_col);
	if (!mat)
		goto error;
	isl_mat_sub_copy(mat->ctx, mat->row, top->row, top->n_row,
			 0, 0, mat->n_col);
	isl_mat_sub_copy(mat->ctx, mat->row + top->n_row, bot->row, bot->n_row,
			 0, 0, mat->n_col);
	isl_mat_free(top);
	isl_mat_free(bot);
	return mat;
error:
	isl_mat_free(top);
	isl_mat_free(bot);
	return NULL;
}

int isl_mat_is_equal(__isl_keep isl_mat *mat1, __isl_keep isl_mat *mat2)
{
	int i;

	if (!mat1 || !mat2)
		return -1;

	if (mat1->n_row != mat2->n_row)
		return 0;

	if (mat1->n_col != mat2->n_col)
		return 0;

	for (i = 0; i < mat1->n_row; ++i)
		if (!isl_seq_eq(mat1->row[i], mat2->row[i], mat1->n_col))
			return 0;

	return 1;
}

__isl_give isl_mat *isl_mat_from_row_vec(__isl_take isl_vec *vec)
{
	struct isl_mat *mat;

	if (!vec)
		return NULL;
	mat = isl_mat_alloc(vec->ctx, 1, vec->size);
	if (!mat)
		goto error;

	isl_seq_cpy(mat->row[0], vec->el, vec->size);

	isl_vec_free(vec);
	return mat;
error:
	isl_vec_free(vec);
	return NULL;
}

/* Return a copy of row "row" of "mat" as an isl_vec.
 */
__isl_give isl_vec *isl_mat_get_row(__isl_keep isl_mat *mat, unsigned row)
{
	isl_vec *v;

	if (!mat)
		return NULL;
	if (row >= mat->n_row)
		isl_die(mat->ctx, isl_error_invalid, "row out of range",
			return NULL);

	v = isl_vec_alloc(isl_mat_get_ctx(mat), mat->n_col);
	if (!v)
		return NULL;
	isl_seq_cpy(v->el, mat->row[row], mat->n_col);

	return v;
}

__isl_give isl_mat *isl_mat_vec_concat(__isl_take isl_mat *top,
	__isl_take isl_vec *bot)
{
	return isl_mat_concat(top, isl_mat_from_row_vec(bot));
}

__isl_give isl_mat *isl_mat_move_cols(__isl_take isl_mat *mat,
	unsigned dst_col, unsigned src_col, unsigned n)
{
	isl_mat *res;

	if (!mat)
		return NULL;
	if (n == 0 || dst_col == src_col)
		return mat;

	res = isl_mat_alloc(mat->ctx, mat->n_row, mat->n_col);
	if (!res)
		goto error;

	if (dst_col < src_col) {
		isl_mat_sub_copy(res->ctx, res->row, mat->row, mat->n_row,
				 0, 0, dst_col);
		isl_mat_sub_copy(res->ctx, res->row, mat->row, mat->n_row,
				 dst_col, src_col, n);
		isl_mat_sub_copy(res->ctx, res->row, mat->row, mat->n_row,
				 dst_col + n, dst_col, src_col - dst_col);
		isl_mat_sub_copy(res->ctx, res->row, mat->row, mat->n_row,
				 src_col + n, src_col + n,
				 res->n_col - src_col - n);
	} else {
		isl_mat_sub_copy(res->ctx, res->row, mat->row, mat->n_row,
				 0, 0, src_col);
		isl_mat_sub_copy(res->ctx, res->row, mat->row, mat->n_row,
				 src_col, src_col + n, dst_col - src_col);
		isl_mat_sub_copy(res->ctx, res->row, mat->row, mat->n_row,
				 dst_col, src_col, n);
		isl_mat_sub_copy(res->ctx, res->row, mat->row, mat->n_row,
				 dst_col + n, dst_col + n,
				 res->n_col - dst_col - n);
	}
	isl_mat_free(mat);

	return res;
error:
	isl_mat_free(mat);
	return NULL;
}

/* Return the gcd of the elements in row "row" of "mat" in *gcd.
 * Return isl_stat_ok on success and isl_stat_error on failure.
 */
isl_stat isl_mat_row_gcd(__isl_keep isl_mat *mat, int row, isl_int *gcd)
{
	if (!mat)
		return isl_stat_error;

	if (row < 0 || row >= mat->n_row)
		isl_die(isl_mat_get_ctx(mat), isl_error_invalid,
			"row out of range", return isl_stat_error);
	isl_seq_gcd(mat->row[row], mat->n_col, gcd);

	return isl_stat_ok;
}

void isl_mat_gcd(__isl_keep isl_mat *mat, isl_int *gcd)
{
	int i;
	isl_int g;

	isl_int_set_si(*gcd, 0);
	if (!mat)
		return;

	isl_int_init(g);
	for (i = 0; i < mat->n_row; ++i) {
		isl_seq_gcd(mat->row[i], mat->n_col, &g);
		isl_int_gcd(*gcd, *gcd, g);
	}
	isl_int_clear(g);
}

/* Return the result of scaling "mat" by a factor of "m".
 */
__isl_give isl_mat *isl_mat_scale(__isl_take isl_mat *mat, isl_int m)
{
	int i;

	if (isl_int_is_one(m))
		return mat;

	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;

	for (i = 0; i < mat->n_row; ++i)
		isl_seq_scale(mat->row[i], mat->row[i], m, mat->n_col);

	return mat;
}

__isl_give isl_mat *isl_mat_scale_down(__isl_take isl_mat *mat, isl_int m)
{
	int i;

	if (isl_int_is_one(m))
		return mat;

	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;

	for (i = 0; i < mat->n_row; ++i)
		isl_seq_scale_down(mat->row[i], mat->row[i], m, mat->n_col);

	return mat;
}

__isl_give isl_mat *isl_mat_scale_down_row(__isl_take isl_mat *mat, int row,
	isl_int m)
{
	if (isl_int_is_one(m))
		return mat;

	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;

	isl_seq_scale_down(mat->row[row], mat->row[row], m, mat->n_col);

	return mat;
}

__isl_give isl_mat *isl_mat_normalize(__isl_take isl_mat *mat)
{
	isl_int gcd;

	if (!mat)
		return NULL;

	isl_int_init(gcd);
	isl_mat_gcd(mat, &gcd);
	mat = isl_mat_scale_down(mat, gcd);
	isl_int_clear(gcd);

	return mat;
}

__isl_give isl_mat *isl_mat_normalize_row(__isl_take isl_mat *mat, int row)
{
	mat = isl_mat_cow(mat);
	if (!mat)
		return NULL;

	isl_seq_normalize(mat->ctx, mat->row[row], mat->n_col);

	return mat;
}

/* Number of initial non-zero columns.
 */
int isl_mat_initial_non_zero_cols(__isl_keep isl_mat *mat)
{
	int i;

	if (!mat)
		return -1;

	for (i = 0; i < mat->n_col; ++i)
		if (row_first_non_zero(mat->row, mat->n_row, i) < 0)
			break;

	return i;
}
