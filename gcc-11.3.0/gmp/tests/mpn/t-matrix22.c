/* Tests matrix22_mul.

Copyright 2008 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

struct matrix {
  mp_size_t alloc;
  mp_size_t n;
  mp_ptr e00, e01, e10, e11;
};

static void
matrix_init (struct matrix *M, mp_size_t n)
{
  mp_ptr p = refmpn_malloc_limbs (4*(n+1));
  M->e00 = p; p += n+1;
  M->e01 = p; p += n+1;
  M->e10 = p; p += n+1;
  M->e11 = p;
  M->alloc = n + 1;
  M->n = 0;
}

static void
matrix_clear (struct matrix *M)
{
  refmpn_free_limbs (M->e00);
}

static void
matrix_copy (struct matrix *R, const struct matrix *M)
{
  R->n = M->n;
  MPN_COPY (R->e00, M->e00, M->n);
  MPN_COPY (R->e01, M->e01, M->n);
  MPN_COPY (R->e10, M->e10, M->n);
  MPN_COPY (R->e11, M->e11, M->n);
}

/* Used with same size, so no need for normalization. */
static int
matrix_equal_p (const struct matrix *A, const struct matrix *B)
{
  return (A->n == B->n
	  && mpn_cmp (A->e00, B->e00, A->n) == 0
	  && mpn_cmp (A->e01, B->e01, A->n) == 0
	  && mpn_cmp (A->e10, B->e10, A->n) == 0
	  && mpn_cmp (A->e11, B->e11, A->n) == 0);
}

static void
matrix_random(struct matrix *M, mp_size_t n, gmp_randstate_ptr rands)
{
  M->n = n;
  mpn_random (M->e00, n);
  mpn_random (M->e01, n);
  mpn_random (M->e10, n);
  mpn_random (M->e11, n);
}

#define MUL(rp, ap, an, bp, bn) do { \
    if (an > bn)		     \
      mpn_mul (rp, ap, an, bp, bn);  \
    else			     \
      mpn_mul (rp, bp, bn, ap, an);  \
  } while(0)

static void
ref_matrix22_mul (struct matrix *R,
		  const struct matrix *A,
		  const struct matrix *B, mp_ptr tp)
{
  mp_size_t an, bn, n;
  mp_ptr r00, r01, r10, r11, a00, a01, a10, a11, b00, b01, b10, b11;

  if (A->n >= B->n)
    {
      r00 = R->e00; a00 = A->e00; b00 = B->e00;
      r01 = R->e01; a01 = A->e01; b01 = B->e01;
      r10 = R->e10; a10 = A->e10; b10 = B->e10;
      r11 = R->e11; a11 = A->e11; b11 = B->e11;
      an = A->n, bn = B->n;
    }
  else
    {
      /* Transpose */
      r00 = R->e00; a00 = B->e00; b00 = A->e00;
      r01 = R->e10; a01 = B->e10; b01 = A->e10;
      r10 = R->e01; a10 = B->e01; b10 = A->e01;
      r11 = R->e11; a11 = B->e11; b11 = A->e11;
      an = B->n, bn = A->n;
    }
  n = an + bn;
  R->n = n + 1;

  mpn_mul (r00, a00, an, b00, bn);
  mpn_mul (tp, a01, an, b10, bn);
  r00[n] = mpn_add_n (r00, r00, tp, n);

  mpn_mul (r01, a00, an, b01, bn);
  mpn_mul (tp, a01, an, b11, bn);
  r01[n] = mpn_add_n (r01, r01, tp, n);

  mpn_mul (r10, a10, an, b00, bn);
  mpn_mul (tp, a11, an, b10, bn);
  r10[n] = mpn_add_n (r10, r10, tp, n);

  mpn_mul (r11, a10, an, b01, bn);
  mpn_mul (tp, a11, an, b11, bn);
  r11[n] = mpn_add_n (r11, r11, tp, n);
}

static void
one_test (const struct matrix *A, const struct matrix *B, int i)
{
  struct matrix R;
  struct matrix P;
  mp_ptr tp;

  matrix_init (&R, A->n + B->n + 1);
  matrix_init (&P, A->n + B->n + 1);

  tp = refmpn_malloc_limbs (mpn_matrix22_mul_itch (A->n, B->n));

  ref_matrix22_mul (&R, A, B, tp);
  matrix_copy (&P, A);
  mpn_matrix22_mul (P.e00, P.e01, P.e10, P.e11, A->n,
		    B->e00, B->e01, B->e10, B->e11, B->n, tp);
  P.n = A->n + B->n + 1;
  if (!matrix_equal_p (&R, &P))
    {
      fprintf (stderr, "ERROR in test %d\n", i);
      gmp_fprintf (stderr, "A = (%Nx, %Nx\n      %Nx, %Nx)\n"
		   "B = (%Nx, %Nx\n      %Nx, %Nx)\n"
		   "R = (%Nx, %Nx (expected)\n      %Nx, %Nx)\n"
		   "P = (%Nx, %Nx (incorrect)\n      %Nx, %Nx)\n",
		   A->e00, A->n, A->e01, A->n, A->e10, A->n, A->e11, A->n,
		   B->e00, B->n, B->e01, B->n, B->e10, B->n, B->e11, B->n,
		   R.e00, R.n, R.e01, R.n, R.e10, R.n, R.e11, R.n,
		   P.e00, P.n, P.e01, P.n, P.e10, P.n, P.e11, P.n);
      abort();
    }
  refmpn_free_limbs (tp);
  matrix_clear (&R);
  matrix_clear (&P);
}

#define MAX_SIZE (2+2*MATRIX22_STRASSEN_THRESHOLD)

int
main (int argc, char **argv)
{
  struct matrix A;
  struct matrix B;

  gmp_randstate_ptr rands;
  mpz_t bs;
  int i;

  tests_start ();
  rands = RANDS;

  matrix_init (&A, MAX_SIZE);
  matrix_init (&B, MAX_SIZE);
  mpz_init (bs);

  for (i = 0; i < 1000; i++)
    {
      mp_size_t an, bn;
      mpz_urandomb (bs, rands, 32);
      an = 1 + mpz_get_ui (bs) % MAX_SIZE;
      mpz_urandomb (bs, rands, 32);
      bn = 1 + mpz_get_ui (bs) % MAX_SIZE;

      matrix_random (&A, an, rands);
      matrix_random (&B, bn, rands);

      one_test (&A, &B, i);
    }
  mpz_clear (bs);
  matrix_clear (&A);
  matrix_clear (&B);

  tests_end ();
  return 0;
}
