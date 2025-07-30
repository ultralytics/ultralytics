/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 */

#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl_seq.h>
#include "isl_map_private.h"
#include "isl_equalities.h"
#include <isl_val_private.h>

/* Given a set of modulo constraints
 *
 *		c + A y = 0 mod d
 *
 * this function computes a particular solution y_0
 *
 * The input is given as a matrix B = [ c A ] and a vector d.
 *
 * The output is matrix containing the solution y_0 or
 * a zero-column matrix if the constraints admit no integer solution.
 *
 * The given set of constrains is equivalent to
 *
 *		c + A y = -D x
 *
 * with D = diag d and x a fresh set of variables.
 * Reducing both c and A modulo d does not change the
 * value of y in the solution and may lead to smaller coefficients.
 * Let M = [ D A ] and [ H 0 ] = M U, the Hermite normal form of M.
 * Then
 *		  [ x ]
 *		M [ y ] = - c
 * and so
 *		               [ x ]
 *		[ H 0 ] U^{-1} [ y ] = - c
 * Let
 *		[ A ]          [ x ]
 *		[ B ] = U^{-1} [ y ]
 * then
 *		H A + 0 B = -c
 *
 * so B may be chosen arbitrarily, e.g., B = 0, and then
 *
 *		       [ x ] = [ -c ]
 *		U^{-1} [ y ] = [  0 ]
 * or
 *		[ x ]     [ -c ]
 *		[ y ] = U [  0 ]
 * specifically,
 *
 *		y = U_{2,1} (-c)
 *
 * If any of the coordinates of this y are non-integer
 * then the constraints admit no integer solution and
 * a zero-column matrix is returned.
 */
static struct isl_mat *particular_solution(struct isl_mat *B, struct isl_vec *d)
{
	int i, j;
	struct isl_mat *M = NULL;
	struct isl_mat *C = NULL;
	struct isl_mat *U = NULL;
	struct isl_mat *H = NULL;
	struct isl_mat *cst = NULL;
	struct isl_mat *T = NULL;

	M = isl_mat_alloc(B->ctx, B->n_row, B->n_row + B->n_col - 1);
	C = isl_mat_alloc(B->ctx, 1 + B->n_row, 1);
	if (!M || !C)
		goto error;
	isl_int_set_si(C->row[0][0], 1);
	for (i = 0; i < B->n_row; ++i) {
		isl_seq_clr(M->row[i], B->n_row);
		isl_int_set(M->row[i][i], d->block.data[i]);
		isl_int_neg(C->row[1 + i][0], B->row[i][0]);
		isl_int_fdiv_r(C->row[1+i][0], C->row[1+i][0], M->row[i][i]);
		for (j = 0; j < B->n_col - 1; ++j)
			isl_int_fdiv_r(M->row[i][B->n_row + j],
					B->row[i][1 + j], M->row[i][i]);
	}
	M = isl_mat_left_hermite(M, 0, &U, NULL);
	if (!M || !U)
		goto error;
	H = isl_mat_sub_alloc(M, 0, B->n_row, 0, B->n_row);
	H = isl_mat_lin_to_aff(H);
	C = isl_mat_inverse_product(H, C);
	if (!C)
		goto error;
	for (i = 0; i < B->n_row; ++i) {
		if (!isl_int_is_divisible_by(C->row[1+i][0], C->row[0][0]))
			break;
		isl_int_divexact(C->row[1+i][0], C->row[1+i][0], C->row[0][0]);
	}
	if (i < B->n_row)
		cst = isl_mat_alloc(B->ctx, B->n_row, 0);
	else
		cst = isl_mat_sub_alloc(C, 1, B->n_row, 0, 1);
	T = isl_mat_sub_alloc(U, B->n_row, B->n_col - 1, 0, B->n_row);
	cst = isl_mat_product(T, cst);
	isl_mat_free(M);
	isl_mat_free(C);
	isl_mat_free(U);
	return cst;
error:
	isl_mat_free(M);
	isl_mat_free(C);
	isl_mat_free(U);
	return NULL;
}

/* Compute and return the matrix
 *
 *		U_1^{-1} diag(d_1, 1, ..., 1)
 *
 * with U_1 the unimodular completion of the first (and only) row of B.
 * The columns of this matrix generate the lattice that satisfies
 * the single (linear) modulo constraint.
 */
static struct isl_mat *parameter_compression_1(
			struct isl_mat *B, struct isl_vec *d)
{
	struct isl_mat *U;

	U = isl_mat_alloc(B->ctx, B->n_col - 1, B->n_col - 1);
	if (!U)
		return NULL;
	isl_seq_cpy(U->row[0], B->row[0] + 1, B->n_col - 1);
	U = isl_mat_unimodular_complete(U, 1);
	U = isl_mat_right_inverse(U);
	if (!U)
		return NULL;
	isl_mat_col_mul(U, 0, d->block.data[0], 0);
	U = isl_mat_lin_to_aff(U);
	return U;
}

/* Compute a common lattice of solutions to the linear modulo
 * constraints specified by B and d.
 * See also the documentation of isl_mat_parameter_compression.
 * We put the matrix
 * 
 *		A = [ L_1^{-T} L_2^{-T} ... L_k^{-T} ]
 *
 * on a common denominator.  This denominator D is the lcm of modulos d.
 * Since L_i = U_i^{-1} diag(d_i, 1, ... 1), we have
 * L_i^{-T} = U_i^T diag(d_i, 1, ... 1)^{-T} = U_i^T diag(1/d_i, 1, ..., 1).
 * Putting this on the common denominator, we have
 * D * L_i^{-T} = U_i^T diag(D/d_i, D, ..., D).
 */
static struct isl_mat *parameter_compression_multi(
			struct isl_mat *B, struct isl_vec *d)
{
	int i, j, k;
	isl_int D;
	struct isl_mat *A = NULL, *U = NULL;
	struct isl_mat *T;
	unsigned size;

	isl_int_init(D);

	isl_vec_lcm(d, &D);

	size = B->n_col - 1;
	A = isl_mat_alloc(B->ctx, size, B->n_row * size);
	U = isl_mat_alloc(B->ctx, size, size);
	if (!U || !A)
		goto error;
	for (i = 0; i < B->n_row; ++i) {
		isl_seq_cpy(U->row[0], B->row[i] + 1, size);
		U = isl_mat_unimodular_complete(U, 1);
		if (!U)
			goto error;
		isl_int_divexact(D, D, d->block.data[i]);
		for (k = 0; k < U->n_col; ++k)
			isl_int_mul(A->row[k][i*size+0], D, U->row[0][k]);
		isl_int_mul(D, D, d->block.data[i]);
		for (j = 1; j < U->n_row; ++j)
			for (k = 0; k < U->n_col; ++k)
				isl_int_mul(A->row[k][i*size+j],
						D, U->row[j][k]);
	}
	A = isl_mat_left_hermite(A, 0, NULL, NULL);
	T = isl_mat_sub_alloc(A, 0, A->n_row, 0, A->n_row);
	T = isl_mat_lin_to_aff(T);
	if (!T)
		goto error;
	isl_int_set(T->row[0][0], D);
	T = isl_mat_right_inverse(T);
	if (!T)
		goto error;
	isl_assert(T->ctx, isl_int_is_one(T->row[0][0]), goto error);
	T = isl_mat_transpose(T);
	isl_mat_free(A);
	isl_mat_free(U);

	isl_int_clear(D);
	return T;
error:
	isl_mat_free(A);
	isl_mat_free(U);
	isl_int_clear(D);
	return NULL;
}

/* Given a set of modulo constraints
 *
 *		c + A y = 0 mod d
 *
 * this function returns an affine transformation T,
 *
 *		y = T y'
 *
 * that bijectively maps the integer vectors y' to integer
 * vectors y that satisfy the modulo constraints.
 *
 * This function is inspired by Section 2.5.3
 * of B. Meister, "Stating and Manipulating Periodicity in the Polytope
 * Model.  Applications to Program Analysis and Optimization".
 * However, the implementation only follows the algorithm of that
 * section for computing a particular solution and not for computing
 * a general homogeneous solution.  The latter is incomplete and
 * may remove some valid solutions.
 * Instead, we use an adaptation of the algorithm in Section 7 of
 * B. Meister, S. Verdoolaege, "Polynomial Approximations in the Polytope
 * Model: Bringing the Power of Quasi-Polynomials to the Masses".
 *
 * The input is given as a matrix B = [ c A ] and a vector d.
 * Each element of the vector d corresponds to a row in B.
 * The output is a lower triangular matrix.
 * If no integer vector y satisfies the given constraints then
 * a matrix with zero columns is returned.
 *
 * We first compute a particular solution y_0 to the given set of
 * modulo constraints in particular_solution.  If no such solution
 * exists, then we return a zero-columned transformation matrix.
 * Otherwise, we compute the generic solution to
 *
 *		A y = 0 mod d
 *
 * That is we want to compute G such that
 *
 *		y = G y''
 *
 * with y'' integer, describes the set of solutions.
 *
 * We first remove the common factors of each row.
 * In particular if gcd(A_i,d_i) != 1, then we divide the whole
 * row i (including d_i) by this common factor.  If afterwards gcd(A_i) != 1,
 * then we divide this row of A by the common factor, unless gcd(A_i) = 0.
 * In the later case, we simply drop the row (in both A and d).
 *
 * If there are no rows left in A, then G is the identity matrix. Otherwise,
 * for each row i, we now determine the lattice of integer vectors
 * that satisfies this row.  Let U_i be the unimodular extension of the
 * row A_i.  This unimodular extension exists because gcd(A_i) = 1.
 * The first component of
 *
 *		y' = U_i y
 *
 * needs to be a multiple of d_i.  Let y' = diag(d_i, 1, ..., 1) y''.
 * Then,
 *
 *		y = U_i^{-1} diag(d_i, 1, ..., 1) y''
 *
 * for arbitrary integer vectors y''.  That is, y belongs to the lattice
 * generated by the columns of L_i = U_i^{-1} diag(d_i, 1, ..., 1).
 * If there is only one row, then G = L_1.
 *
 * If there is more than one row left, we need to compute the intersection
 * of the lattices.  That is, we need to compute an L such that
 *
 *		L = L_i L_i'	for all i
 *
 * with L_i' some integer matrices.  Let A be constructed as follows
 *
 *		A = [ L_1^{-T} L_2^{-T} ... L_k^{-T} ]
 *
 * and computed the Hermite Normal Form of A = [ H 0 ] U
 * Then,
 *
 *		L_i^{-T} = H U_{1,i}
 *
 * or
 *
 *		H^{-T} = L_i U_{1,i}^T
 *
 * In other words G = L = H^{-T}.
 * To ensure that G is lower triangular, we compute and use its Hermite
 * normal form.
 *
 * The affine transformation matrix returned is then
 *
 *		[  1   0  ]
 *		[ y_0  G  ]
 *
 * as any y = y_0 + G y' with y' integer is a solution to the original
 * modulo constraints.
 */
struct isl_mat *isl_mat_parameter_compression(
			struct isl_mat *B, struct isl_vec *d)
{
	int i;
	struct isl_mat *cst = NULL;
	struct isl_mat *T = NULL;
	isl_int D;

	if (!B || !d)
		goto error;
	isl_assert(B->ctx, B->n_row == d->size, goto error);
	cst = particular_solution(B, d);
	if (!cst)
		goto error;
	if (cst->n_col == 0) {
		T = isl_mat_alloc(B->ctx, B->n_col, 0);
		isl_mat_free(cst);
		isl_mat_free(B);
		isl_vec_free(d);
		return T;
	}
	isl_int_init(D);
	/* Replace a*g*row = 0 mod g*m by row = 0 mod m */
	for (i = 0; i < B->n_row; ++i) {
		isl_seq_gcd(B->row[i] + 1, B->n_col - 1, &D);
		if (isl_int_is_one(D))
			continue;
		if (isl_int_is_zero(D)) {
			B = isl_mat_drop_rows(B, i, 1);
			d = isl_vec_cow(d);
			if (!B || !d)
				goto error2;
			isl_seq_cpy(d->block.data+i, d->block.data+i+1,
							d->size - (i+1));
			d->size--;
			i--;
			continue;
		}
		B = isl_mat_cow(B);
		if (!B)
			goto error2;
		isl_seq_scale_down(B->row[i] + 1, B->row[i] + 1, D, B->n_col-1);
		isl_int_gcd(D, D, d->block.data[i]);
		d = isl_vec_cow(d);
		if (!d)
			goto error2;
		isl_int_divexact(d->block.data[i], d->block.data[i], D);
	}
	isl_int_clear(D);
	if (B->n_row == 0)
		T = isl_mat_identity(B->ctx, B->n_col);
	else if (B->n_row == 1)
		T = parameter_compression_1(B, d);
	else
		T = parameter_compression_multi(B, d);
	T = isl_mat_left_hermite(T, 0, NULL, NULL);
	if (!T)
		goto error;
	isl_mat_sub_copy(T->ctx, T->row + 1, cst->row, cst->n_row, 0, 0, 1);
	isl_mat_free(cst);
	isl_mat_free(B);
	isl_vec_free(d);
	return T;
error2:
	isl_int_clear(D);
error:
	isl_mat_free(cst);
	isl_mat_free(B);
	isl_vec_free(d);
	return NULL;
}

/* Given a set of equalities
 *
 *		B(y) + A x = 0						(*)
 *
 * compute and return an affine transformation T,
 *
 *		y = T y'
 *
 * that bijectively maps the integer vectors y' to integer
 * vectors y that satisfy the modulo constraints for some value of x.
 *
 * Let [H 0] be the Hermite Normal Form of A, i.e.,
 *
 *		A = [H 0] Q
 *
 * Then y is a solution of (*) iff
 *
 *		H^-1 B(y) (= - [I 0] Q x)
 *
 * is an integer vector.  Let d be the common denominator of H^-1.
 * We impose
 *
 *		d H^-1 B(y) = 0 mod d
 *
 * and compute the solution using isl_mat_parameter_compression.
 */
__isl_give isl_mat *isl_mat_parameter_compression_ext(__isl_take isl_mat *B,
	__isl_take isl_mat *A)
{
	isl_ctx *ctx;
	isl_vec *d;
	int n_row, n_col;

	if (!A)
		return isl_mat_free(B);

	ctx = isl_mat_get_ctx(A);
	n_row = A->n_row;
	n_col = A->n_col;
	A = isl_mat_left_hermite(A, 0, NULL, NULL);
	A = isl_mat_drop_cols(A, n_row, n_col - n_row);
	A = isl_mat_lin_to_aff(A);
	A = isl_mat_right_inverse(A);
	d = isl_vec_alloc(ctx, n_row);
	if (A)
		d = isl_vec_set(d, A->row[0][0]);
	A = isl_mat_drop_rows(A, 0, 1);
	A = isl_mat_drop_cols(A, 0, 1);
	B = isl_mat_product(A, B);

	return isl_mat_parameter_compression(B, d);
}

/* Return a compression matrix that indicates that there are no solutions
 * to the original constraints.  In particular, return a zero-column
 * matrix with 1 + dim rows.  If "T2" is not NULL, then assign *T2
 * the inverse of this matrix.  *T2 may already have been assigned
 * matrix, so free it first.
 * "free1", "free2" and "free3" are temporary matrices that are
 * not useful when an empty compression is returned.  They are
 * simply freed.
 */
static __isl_give isl_mat *empty_compression(isl_ctx *ctx, unsigned dim,
	__isl_give isl_mat **T2, __isl_take isl_mat *free1,
	__isl_take isl_mat *free2, __isl_take isl_mat *free3)
{
	isl_mat_free(free1);
	isl_mat_free(free2);
	isl_mat_free(free3);
	if (T2) {
		isl_mat_free(*T2);
		*T2 = isl_mat_alloc(ctx, 0, 1 + dim);
	}
	return isl_mat_alloc(ctx, 1 + dim, 0);
}

/* Given a matrix that maps a (possibly) parametric domain to
 * a parametric domain, add in rows that map the "nparam" parameters onto
 * themselves.
 */
static __isl_give isl_mat *insert_parameter_rows(__isl_take isl_mat *mat,
	unsigned nparam)
{
	int i;

	if (nparam == 0)
		return mat;
	if (!mat)
		return NULL;

	mat = isl_mat_insert_rows(mat, 1, nparam);
	if (!mat)
		return NULL;

	for (i = 0; i < nparam; ++i) {
		isl_seq_clr(mat->row[1 + i], mat->n_col);
		isl_int_set(mat->row[1 + i][1 + i], mat->row[0][0]);
	}

	return mat;
}

/* Given a set of equalities
 *
 *		-C(y) + M x = 0
 *
 * this function computes a unimodular transformation from a lower-dimensional
 * space to the original space that bijectively maps the integer points x'
 * in the lower-dimensional space to the integer points x in the original
 * space that satisfy the equalities.
 *
 * The input is given as a matrix B = [ -C M ] and the output is a
 * matrix that maps [1 x'] to [1 x].
 * The number of equality constraints in B is assumed to be smaller than
 * or equal to the number of variables x.
 * "first" is the position of the first x variable.
 * The preceding variables are considered to be y-variables.
 * If T2 is not NULL, then *T2 is set to a matrix mapping [1 x] to [1 x'].
 *
 * First compute the (left) Hermite normal form of M,
 *
 *		M [U1 U2] = M U = H = [H1 0]
 * or
 *		              M = H Q = [H1 0] [Q1]
 *                                             [Q2]
 *
 * with U, Q unimodular, Q = U^{-1} (and H lower triangular).
 * Define the transformed variables as
 *
 *		x = [U1 U2] [ x1' ] = [U1 U2] [Q1] x
 *		            [ x2' ]           [Q2]
 *
 * The equalities then become
 *
 *		-C(y) + H1 x1' = 0   or   x1' = H1^{-1} C(y) = C'(y)
 *
 * If the denominator of the constant term does not divide the
 * the common denominator of the coefficients of y, then every
 * integer point is mapped to a non-integer point and then the original set
 * has no integer solutions (since the x' are a unimodular transformation
 * of the x).  In this case, a zero-column matrix is returned.
 * Otherwise, the transformation is given by
 *
 *		x = U1 H1^{-1} C(y) + U2 x2'
 *
 * The inverse transformation is simply
 *
 *		x2' = Q2 x
 */
__isl_give isl_mat *isl_mat_final_variable_compression(__isl_take isl_mat *B,
	int first, __isl_give isl_mat **T2)
{
	int i, n;
	isl_ctx *ctx;
	isl_mat *H = NULL, *C, *H1, *U = NULL, *U1, *U2;
	unsigned dim;

	if (T2)
		*T2 = NULL;
	if (!B)
		goto error;

	ctx = isl_mat_get_ctx(B);
	dim = B->n_col - 1;
	n = dim - first;
	if (n < B->n_row)
		isl_die(ctx, isl_error_invalid, "too many equality constraints",
			goto error);
	H = isl_mat_sub_alloc(B, 0, B->n_row, 1 + first, n);
	H = isl_mat_left_hermite(H, 0, &U, T2);
	if (!H || !U || (T2 && !*T2))
		goto error;
	if (T2) {
		*T2 = isl_mat_drop_rows(*T2, 0, B->n_row);
		*T2 = isl_mat_diagonal(isl_mat_identity(ctx, 1 + first), *T2);
		if (!*T2)
			goto error;
	}
	C = isl_mat_alloc(ctx, 1 + B->n_row, 1 + first);
	if (!C)
		goto error;
	isl_int_set_si(C->row[0][0], 1);
	isl_seq_clr(C->row[0] + 1, first);
	isl_mat_sub_neg(ctx, C->row + 1, B->row, B->n_row, 0, 0, 1 + first);
	H1 = isl_mat_sub_alloc(H, 0, H->n_row, 0, H->n_row);
	H1 = isl_mat_lin_to_aff(H1);
	C = isl_mat_inverse_product(H1, C);
	if (!C)
		goto error;
	isl_mat_free(H);
	if (!isl_int_is_one(C->row[0][0])) {
		isl_int g;

		isl_int_init(g);
		for (i = 0; i < B->n_row; ++i) {
			isl_seq_gcd(C->row[1 + i] + 1, first, &g);
			isl_int_gcd(g, g, C->row[0][0]);
			if (!isl_int_is_divisible_by(C->row[1 + i][0], g))
				break;
		}
		isl_int_clear(g);

		if (i < B->n_row)
			return empty_compression(ctx, dim, T2, B, C, U);
		C = isl_mat_normalize(C);
	}
	U1 = isl_mat_sub_alloc(U, 0, U->n_row, 0, B->n_row);
	U1 = isl_mat_lin_to_aff(U1);
	U2 = isl_mat_sub_alloc(U, 0, U->n_row, B->n_row, U->n_row - B->n_row);
	U2 = isl_mat_lin_to_aff(U2);
	isl_mat_free(U);
	C = isl_mat_product(U1, C);
	C = isl_mat_aff_direct_sum(C, U2);
	C = insert_parameter_rows(C, first);

	isl_mat_free(B);

	return C;
error:
	isl_mat_free(B);
	isl_mat_free(H);
	isl_mat_free(U);
	if (T2) {
		isl_mat_free(*T2);
		*T2 = NULL;
	}
	return NULL;
}

/* Given a set of equalities
 *
 *		M x - c = 0
 *
 * this function computes a unimodular transformation from a lower-dimensional
 * space to the original space that bijectively maps the integer points x'
 * in the lower-dimensional space to the integer points x in the original
 * space that satisfy the equalities.
 *
 * The input is given as a matrix B = [ -c M ] and the output is a
 * matrix that maps [1 x'] to [1 x].
 * The number of equality constraints in B is assumed to be smaller than
 * or equal to the number of variables x.
 * If T2 is not NULL, then *T2 is set to a matrix mapping [1 x] to [1 x'].
 */
__isl_give isl_mat *isl_mat_variable_compression(__isl_take isl_mat *B,
	__isl_give isl_mat **T2)
{
	return isl_mat_final_variable_compression(B, 0, T2);
}

/* Return "bset" and set *T and *T2 to the identity transformation
 * on "bset" (provided T and T2 are not NULL).
 */
static __isl_give isl_basic_set *return_with_identity(
	__isl_take isl_basic_set *bset, __isl_give isl_mat **T,
	__isl_give isl_mat **T2)
{
	unsigned dim;
	isl_mat *id;

	if (!bset)
		return NULL;
	if (!T && !T2)
		return bset;

	dim = isl_basic_set_dim(bset, isl_dim_set);
	id = isl_mat_identity(isl_basic_map_get_ctx(bset), 1 + dim);
	if (T)
		*T = isl_mat_copy(id);
	if (T2)
		*T2 = isl_mat_copy(id);
	isl_mat_free(id);

	return bset;
}

/* Use the n equalities of bset to unimodularly transform the
 * variables x such that n transformed variables x1' have a constant value
 * and rewrite the constraints of bset in terms of the remaining
 * transformed variables x2'.  The matrix pointed to by T maps
 * the new variables x2' back to the original variables x, while T2
 * maps the original variables to the new variables.
 */
static struct isl_basic_set *compress_variables(
	struct isl_basic_set *bset, struct isl_mat **T, struct isl_mat **T2)
{
	struct isl_mat *B, *TC;
	unsigned dim;

	if (T)
		*T = NULL;
	if (T2)
		*T2 = NULL;
	if (!bset)
		goto error;
	isl_assert(bset->ctx, isl_basic_set_n_param(bset) == 0, goto error);
	isl_assert(bset->ctx, bset->n_div == 0, goto error);
	dim = isl_basic_set_n_dim(bset);
	isl_assert(bset->ctx, bset->n_eq <= dim, goto error);
	if (bset->n_eq == 0)
		return return_with_identity(bset, T, T2);

	B = isl_mat_sub_alloc6(bset->ctx, bset->eq, 0, bset->n_eq, 0, 1 + dim);
	TC = isl_mat_variable_compression(B, T2);
	if (!TC)
		goto error;
	if (TC->n_col == 0) {
		isl_mat_free(TC);
		if (T2) {
			isl_mat_free(*T2);
			*T2 = NULL;
		}
		bset = isl_basic_set_set_to_empty(bset);
		return return_with_identity(bset, T, T2);
	}

	bset = isl_basic_set_preimage(bset, T ? isl_mat_copy(TC) : TC);
	if (T)
		*T = TC;
	return bset;
error:
	isl_basic_set_free(bset);
	return NULL;
}

struct isl_basic_set *isl_basic_set_remove_equalities(
	struct isl_basic_set *bset, struct isl_mat **T, struct isl_mat **T2)
{
	if (T)
		*T = NULL;
	if (T2)
		*T2 = NULL;
	if (!bset)
		return NULL;
	isl_assert(bset->ctx, isl_basic_set_n_param(bset) == 0, goto error);
	bset = isl_basic_set_gauss(bset, NULL);
	if (ISL_F_ISSET(bset, ISL_BASIC_SET_EMPTY))
		return return_with_identity(bset, T, T2);
	bset = compress_variables(bset, T, T2);
	return bset;
error:
	isl_basic_set_free(bset);
	*T = NULL;
	return NULL;
}

/* Check if dimension dim belongs to a residue class
 *		i_dim \equiv r mod m
 * with m != 1 and if so return m in *modulo and r in *residue.
 * As a special case, when i_dim has a fixed value v, then
 * *modulo is set to 0 and *residue to v.
 *
 * If i_dim does not belong to such a residue class, then *modulo
 * is set to 1 and *residue is set to 0.
 */
int isl_basic_set_dim_residue_class(struct isl_basic_set *bset,
	int pos, isl_int *modulo, isl_int *residue)
{
	struct isl_ctx *ctx;
	struct isl_mat *H = NULL, *U = NULL, *C, *H1, *U1;
	unsigned total;
	unsigned nparam;

	if (!bset || !modulo || !residue)
		return -1;

	if (isl_basic_set_plain_dim_is_fixed(bset, pos, residue)) {
		isl_int_set_si(*modulo, 0);
		return 0;
	}

	ctx = isl_basic_set_get_ctx(bset);
	total = isl_basic_set_total_dim(bset);
	nparam = isl_basic_set_n_param(bset);
	H = isl_mat_sub_alloc6(ctx, bset->eq, 0, bset->n_eq, 1, total);
	H = isl_mat_left_hermite(H, 0, &U, NULL);
	if (!H)
		return -1;

	isl_seq_gcd(U->row[nparam + pos]+bset->n_eq,
			total-bset->n_eq, modulo);
	if (isl_int_is_zero(*modulo))
		isl_int_set_si(*modulo, 1);
	if (isl_int_is_one(*modulo)) {
		isl_int_set_si(*residue, 0);
		isl_mat_free(H);
		isl_mat_free(U);
		return 0;
	}

	C = isl_mat_alloc(ctx, 1 + bset->n_eq, 1);
	if (!C)
		goto error;
	isl_int_set_si(C->row[0][0], 1);
	isl_mat_sub_neg(ctx, C->row + 1, bset->eq, bset->n_eq, 0, 0, 1);
	H1 = isl_mat_sub_alloc(H, 0, H->n_row, 0, H->n_row);
	H1 = isl_mat_lin_to_aff(H1);
	C = isl_mat_inverse_product(H1, C);
	isl_mat_free(H);
	U1 = isl_mat_sub_alloc(U, nparam+pos, 1, 0, bset->n_eq);
	U1 = isl_mat_lin_to_aff(U1);
	isl_mat_free(U);
	C = isl_mat_product(U1, C);
	if (!C)
		return -1;
	if (!isl_int_is_divisible_by(C->row[1][0], C->row[0][0])) {
		bset = isl_basic_set_copy(bset);
		bset = isl_basic_set_set_to_empty(bset);
		isl_basic_set_free(bset);
		isl_int_set_si(*modulo, 1);
		isl_int_set_si(*residue, 0);
		return 0;
	}
	isl_int_divexact(*residue, C->row[1][0], C->row[0][0]);
	isl_int_fdiv_r(*residue, *residue, *modulo);
	isl_mat_free(C);
	return 0;
error:
	isl_mat_free(H);
	isl_mat_free(U);
	return -1;
}

/* Check if dimension dim belongs to a residue class
 *		i_dim \equiv r mod m
 * with m != 1 and if so return m in *modulo and r in *residue.
 * As a special case, when i_dim has a fixed value v, then
 * *modulo is set to 0 and *residue to v.
 *
 * If i_dim does not belong to such a residue class, then *modulo
 * is set to 1 and *residue is set to 0.
 */
int isl_set_dim_residue_class(struct isl_set *set,
	int pos, isl_int *modulo, isl_int *residue)
{
	isl_int m;
	isl_int r;
	int i;

	if (!set || !modulo || !residue)
		return -1;

	if (set->n == 0) {
		isl_int_set_si(*modulo, 0);
		isl_int_set_si(*residue, 0);
		return 0;
	}

	if (isl_basic_set_dim_residue_class(set->p[0], pos, modulo, residue)<0)
		return -1;

	if (set->n == 1)
		return 0;

	if (isl_int_is_one(*modulo))
		return 0;

	isl_int_init(m);
	isl_int_init(r);

	for (i = 1; i < set->n; ++i) {
		if (isl_basic_set_dim_residue_class(set->p[i], pos, &m, &r) < 0)
			goto error;
		isl_int_gcd(*modulo, *modulo, m);
		isl_int_sub(m, *residue, r);
		isl_int_gcd(*modulo, *modulo, m);
		if (!isl_int_is_zero(*modulo))
			isl_int_fdiv_r(*residue, *residue, *modulo);
		if (isl_int_is_one(*modulo))
			break;
	}

	isl_int_clear(m);
	isl_int_clear(r);

	return 0;
error:
	isl_int_clear(m);
	isl_int_clear(r);
	return -1;
}

/* Check if dimension "dim" belongs to a residue class
 *		i_dim \equiv r mod m
 * with m != 1 and if so return m in *modulo and r in *residue.
 * As a special case, when i_dim has a fixed value v, then
 * *modulo is set to 0 and *residue to v.
 *
 * If i_dim does not belong to such a residue class, then *modulo
 * is set to 1 and *residue is set to 0.
 */
isl_stat isl_set_dim_residue_class_val(__isl_keep isl_set *set,
	int pos, __isl_give isl_val **modulo, __isl_give isl_val **residue)
{
	*modulo = NULL;
	*residue = NULL;
	if (!set)
		return isl_stat_error;
	*modulo = isl_val_alloc(isl_set_get_ctx(set));
	*residue = isl_val_alloc(isl_set_get_ctx(set));
	if (!*modulo || !*residue)
		goto error;
	if (isl_set_dim_residue_class(set, pos,
					&(*modulo)->n, &(*residue)->n) < 0)
		goto error;
	isl_int_set_si((*modulo)->d, 1);
	isl_int_set_si((*residue)->d, 1);
	return isl_stat_ok;
error:
	isl_val_free(*modulo);
	isl_val_free(*residue);
	return isl_stat_error;
}
