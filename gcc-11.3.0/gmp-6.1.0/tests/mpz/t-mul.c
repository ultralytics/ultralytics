/* Test mpz_cmp, mpz_mul.

Copyright 1991, 1993, 1994, 1996, 1997, 2000-2004 Free Software Foundation,
Inc.

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
#include "longlong.h"
#include "tests.h"

void debug_mp (mpz_t);
static void refmpz_mul (mpz_t, const mpz_t, const mpz_t);
void dump_abort (int, const char *, mpz_t, mpz_t, mpz_t, mpz_t);

#define FFT_MIN_BITSIZE 100000

char *extra_fft;

void
one (int i, mpz_t multiplicand, mpz_t multiplier)
{
  mpz_t product, ref_product;

  mpz_init (product);
  mpz_init (ref_product);

  /* Test plain multiplication comparing results against reference code.  */
  mpz_mul (product, multiplier, multiplicand);
  refmpz_mul (ref_product, multiplier, multiplicand);
  if (mpz_cmp (product, ref_product))
    dump_abort (i, "incorrect plain product",
		multiplier, multiplicand, product, ref_product);

  /* Test squaring, comparing results against plain multiplication  */
  mpz_mul (product, multiplier, multiplier);
  mpz_set (multiplicand, multiplier);
  mpz_mul (ref_product, multiplier, multiplicand);
  if (mpz_cmp (product, ref_product))
    dump_abort (i, "incorrect square product",
		multiplier, multiplier, product, ref_product);

  mpz_clear (product);
  mpz_clear (ref_product);
}

int
main (int argc, char **argv)
{
  mpz_t op1, op2;
  int i;
  int fft_max_2exp;

  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long bsi, size_range, fsize_range;

  tests_start ();
  rands = RANDS;

  extra_fft = getenv ("GMP_CHECK_FFT");
  fft_max_2exp = 0;
  if (extra_fft != NULL)
    fft_max_2exp = atoi (extra_fft);

  if (fft_max_2exp <= 1)	/* compat with old use of GMP_CHECK_FFT */
    fft_max_2exp = 22;		/* default limit, good for any machine */

  mpz_init (bs);
  mpz_init (op1);
  mpz_init (op2);

  fsize_range = 4 << 8;		/* a fraction 1/256 of size_range */
  for (i = 0;; i++)
    {
      size_range = fsize_range >> 8;
      fsize_range = fsize_range * 33 / 32;

      if (size_range > fft_max_2exp)
	break;

      mpz_urandomb (bs, rands, size_range);
      mpz_rrandomb (op1, rands, mpz_get_ui (bs));
      if (i & 1)
	mpz_urandomb (bs, rands, size_range);
      mpz_rrandomb (op2, rands, mpz_get_ui (bs));

      mpz_urandomb (bs, rands, 4);
      bsi = mpz_get_ui (bs);
      if ((bsi & 0x3) == 0)
	mpz_neg (op1, op1);
      if ((bsi & 0xC) == 0)
	mpz_neg (op2, op2);

      /* printf ("%d %d\n", SIZ (op1), SIZ (op2)); */
      one (i, op2, op1);
    }

  for (i = -50; i < 0; i++)
    {
      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % fft_max_2exp;

      mpz_urandomb (bs, rands, size_range);
      mpz_rrandomb (op1, rands, mpz_get_ui (bs) + FFT_MIN_BITSIZE);
      mpz_urandomb (bs, rands, size_range);
      mpz_rrandomb (op2, rands, mpz_get_ui (bs) + FFT_MIN_BITSIZE);

      /* printf ("%d: %d %d\n", i, SIZ (op1), SIZ (op2)); */
      fflush (stdout);
      one (-1, op2, op1);
    }

  mpz_clear (bs);
  mpz_clear (op1);
  mpz_clear (op2);

  tests_end ();
  exit (0);
}

static void
refmpz_mul (mpz_t w, const mpz_t u, const mpz_t v)
{
  mp_size_t usize = u->_mp_size;
  mp_size_t vsize = v->_mp_size;
  mp_size_t wsize;
  mp_size_t sign_product;
  mp_ptr up, vp;
  mp_ptr wp;
  mp_size_t talloc;

  sign_product = usize ^ vsize;
  usize = ABS (usize);
  vsize = ABS (vsize);

  if (usize == 0 || vsize == 0)
    {
      SIZ (w) = 0;
      return;
    }

  talloc = usize + vsize;

  up = u->_mp_d;
  vp = v->_mp_d;

  wp = __GMP_ALLOCATE_FUNC_LIMBS (talloc);

  if (usize > vsize)
    refmpn_mul (wp, up, usize, vp, vsize);
  else
    refmpn_mul (wp, vp, vsize, up, usize);
  wsize = usize + vsize;
  wsize -= wp[wsize - 1] == 0;
  MPZ_REALLOC (w, wsize);
  MPN_COPY (PTR(w), wp, wsize);

  SIZ(w) = sign_product < 0 ? -wsize : wsize;
  __GMP_FREE_FUNC_LIMBS (wp, talloc);
}

void
dump_abort (int i, const char *s,
            mpz_t op1, mpz_t op2, mpz_t product, mpz_t ref_product)
{
  mp_size_t b, e;
  fprintf (stderr, "ERROR: %s in test %d\n", s, i);
  fprintf (stderr, "op1          = "); debug_mp (op1);
  fprintf (stderr, "op2          = "); debug_mp (op2);
  fprintf (stderr, "    product  = "); debug_mp (product);
  fprintf (stderr, "ref_product  = "); debug_mp (ref_product);
  for (b = 0; b < ABSIZ(ref_product); b++)
    if (PTR(ref_product)[b] != PTR(product)[b])
      break;
  for (e = ABSIZ(ref_product) - 1; e >= 0; e--)
    if (PTR(ref_product)[e] != PTR(product)[e])
      break;
  printf ("ERRORS in %ld--%ld\n", b, e);
  abort();
}

void
debug_mp (mpz_t x)
{
  size_t siz = mpz_sizeinbase (x, 16);

  if (siz > 65)
    {
      mpz_t q;
      mpz_init (q);
      mpz_tdiv_q_2exp (q, x, 4 * (mpz_sizeinbase (x, 16) - 25));
      gmp_fprintf (stderr, "%ZX...", q);
      mpz_tdiv_r_2exp (q, x, 4 * 25);
      gmp_fprintf (stderr, "%025ZX [%d]\n", q, (int) siz);
      mpz_clear (q);
    }
  else
    {
      gmp_fprintf (stderr, "%ZX\n", x);
    }
}
