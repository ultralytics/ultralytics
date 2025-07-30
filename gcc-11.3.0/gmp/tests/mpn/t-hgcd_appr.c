/* Test mpn_hgcd_appr.

Copyright 1991, 1993, 1994, 1996, 1997, 2000-2004, 2011 Free Software
Foundation, Inc.

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
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

static mp_size_t one_test (mpz_t, mpz_t, int);
static void debug_mp (mpz_t, int);

#define MIN_OPERAND_SIZE 2

struct hgcd_ref
{
  mpz_t m[2][2];
};

static void hgcd_ref_init (struct hgcd_ref *hgcd);
static void hgcd_ref_clear (struct hgcd_ref *hgcd);
static int hgcd_ref (struct hgcd_ref *hgcd, mpz_t a, mpz_t b);
static int hgcd_ref_equal (const struct hgcd_ref *, const struct hgcd_ref *);
static int hgcd_appr_valid_p (mpz_t, mpz_t, mp_size_t, struct hgcd_ref *,
			      mpz_t, mpz_t, mp_size_t, struct hgcd_matrix *);

static int verbose_flag = 0;

int
main (int argc, char **argv)
{
  mpz_t op1, op2, temp1, temp2;
  int i, j, chain_len;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long size_range;

  if (argc > 1)
    {
      if (strcmp (argv[1], "-v") == 0)
	verbose_flag = 1;
      else
	{
	  fprintf (stderr, "Invalid argument.\n");
	  return 1;
	}
    }

  tests_start ();
  rands = RANDS;

  mpz_init (bs);
  mpz_init (op1);
  mpz_init (op2);
  mpz_init (temp1);
  mpz_init (temp2);

  for (i = 0; i < 15; i++)
    {
      /* Generate plain operands with unknown gcd.  These types of operands
	 have proven to trigger certain bugs in development versions of the
	 gcd code. */

      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 13 + 2;

      mpz_urandomb (bs, rands, size_range);
      mpz_urandomb (op1, rands, mpz_get_ui (bs) + MIN_OPERAND_SIZE);
      mpz_urandomb (bs, rands, size_range);
      mpz_urandomb (op2, rands, mpz_get_ui (bs) + MIN_OPERAND_SIZE);

      if (mpz_cmp (op1, op2) < 0)
	mpz_swap (op1, op2);

      if (mpz_size (op1) > 0)
	one_test (op1, op2, i);

      /* Generate a division chain backwards, allowing otherwise
	 unlikely huge quotients.  */

      mpz_set_ui (op1, 0);
      mpz_urandomb (bs, rands, 32);
      mpz_urandomb (bs, rands, mpz_get_ui (bs) % 16 + 1);
      mpz_rrandomb (op2, rands, mpz_get_ui (bs));
      mpz_add_ui (op2, op2, 1);

#if 0
      chain_len = 1000000;
#else
      mpz_urandomb (bs, rands, 32);
      chain_len = mpz_get_ui (bs) % (GMP_NUMB_BITS * GCD_DC_THRESHOLD / 256);
#endif

      for (j = 0; j < chain_len; j++)
	{
	  mpz_urandomb (bs, rands, 32);
	  mpz_urandomb (bs, rands, mpz_get_ui (bs) % 12 + 1);
	  mpz_rrandomb (temp2, rands, mpz_get_ui (bs) + 1);
	  mpz_add_ui (temp2, temp2, 1);
	  mpz_mul (temp1, op2, temp2);
	  mpz_add (op1, op1, temp1);

	  /* Don't generate overly huge operands.  */
	  if (SIZ (op1) > 3 * GCD_DC_THRESHOLD)
	    break;

	  mpz_urandomb (bs, rands, 32);
	  mpz_urandomb (bs, rands, mpz_get_ui (bs) % 12 + 1);
	  mpz_rrandomb (temp2, rands, mpz_get_ui (bs) + 1);
	  mpz_add_ui (temp2, temp2, 1);
	  mpz_mul (temp1, op1, temp2);
	  mpz_add (op2, op2, temp1);

	  /* Don't generate overly huge operands.  */
	  if (SIZ (op2) > 3 * GCD_DC_THRESHOLD)
	    break;
	}
      if (mpz_cmp (op1, op2) < 0)
	mpz_swap (op1, op2);

      if (mpz_size (op1) > 0)
	one_test (op1, op2, i);
    }

  mpz_clear (bs);
  mpz_clear (op1);
  mpz_clear (op2);
  mpz_clear (temp1);
  mpz_clear (temp2);

  tests_end ();
  exit (0);
}

static void
debug_mp (mpz_t x, int base)
{
  mpz_out_str (stderr, base, x); fputc ('\n', stderr);
}

static mp_size_t
one_test (mpz_t a, mpz_t b, int i)
{
  struct hgcd_matrix hgcd;
  struct hgcd_ref ref;

  mpz_t ref_r0;
  mpz_t ref_r1;
  mpz_t hgcd_r0;
  mpz_t hgcd_r1;

  int res[2];
  mp_size_t asize;
  mp_size_t bsize;

  mp_size_t hgcd_init_scratch;
  mp_size_t hgcd_scratch;

  mp_ptr hgcd_init_tp;
  mp_ptr hgcd_tp;
  mp_limb_t marker[4];

  asize = a->_mp_size;
  bsize = b->_mp_size;

  ASSERT (asize >= bsize);

  hgcd_init_scratch = MPN_HGCD_MATRIX_INIT_ITCH (asize);
  hgcd_init_tp = refmpn_malloc_limbs (hgcd_init_scratch + 2) + 1;
  mpn_hgcd_matrix_init (&hgcd, asize, hgcd_init_tp);

  hgcd_scratch = mpn_hgcd_appr_itch (asize);
  hgcd_tp = refmpn_malloc_limbs (hgcd_scratch + 2) + 1;

  mpn_random (marker, 4);

  hgcd_init_tp[-1] = marker[0];
  hgcd_init_tp[hgcd_init_scratch] = marker[1];
  hgcd_tp[-1] = marker[2];
  hgcd_tp[hgcd_scratch] = marker[3];

#if 0
  fprintf (stderr,
	   "one_test: i = %d asize = %d, bsize = %d\n",
	   i, a->_mp_size, b->_mp_size);

  gmp_fprintf (stderr,
	       "one_test: i = %d\n"
	       "  a = %Zx\n"
	       "  b = %Zx\n",
	       i, a, b);
#endif
  hgcd_ref_init (&ref);

  mpz_init_set (ref_r0, a);
  mpz_init_set (ref_r1, b);
  res[0] = hgcd_ref (&ref, ref_r0, ref_r1);

  mpz_init_set (hgcd_r0, a);
  mpz_init_set (hgcd_r1, b);
  if (bsize < asize)
    {
      _mpz_realloc (hgcd_r1, asize);
      MPN_ZERO (hgcd_r1->_mp_d + bsize, asize - bsize);
    }
  res[1] = mpn_hgcd_appr (hgcd_r0->_mp_d,
			  hgcd_r1->_mp_d,
			  asize,
			  &hgcd, hgcd_tp);

  if (hgcd_init_tp[-1] != marker[0]
      || hgcd_init_tp[hgcd_init_scratch] != marker[1]
      || hgcd_tp[-1] != marker[2]
      || hgcd_tp[hgcd_scratch] != marker[3])
    {
      fprintf (stderr, "ERROR in test %d\n", i);
      fprintf (stderr, "scratch space overwritten!\n");

      if (hgcd_init_tp[-1] != marker[0])
	gmp_fprintf (stderr,
		     "before init_tp: %Mx\n"
		     "expected: %Mx\n",
		     hgcd_init_tp[-1], marker[0]);
      if (hgcd_init_tp[hgcd_init_scratch] != marker[1])
	gmp_fprintf (stderr,
		     "after init_tp: %Mx\n"
		     "expected: %Mx\n",
		     hgcd_init_tp[hgcd_init_scratch], marker[1]);
      if (hgcd_tp[-1] != marker[2])
	gmp_fprintf (stderr,
		     "before tp: %Mx\n"
		     "expected: %Mx\n",
		     hgcd_tp[-1], marker[2]);
      if (hgcd_tp[hgcd_scratch] != marker[3])
	gmp_fprintf (stderr,
		     "after tp: %Mx\n"
		     "expected: %Mx\n",
		     hgcd_tp[hgcd_scratch], marker[3]);

      abort ();
    }

  if (!hgcd_appr_valid_p (a, b, res[0], &ref, ref_r0, ref_r1,
			  res[1], &hgcd))
    {
      fprintf (stderr, "ERROR in test %d\n", i);
      fprintf (stderr, "Invalid results for hgcd and hgcd_ref\n");
      fprintf (stderr, "op1=");                 debug_mp (a, -16);
      fprintf (stderr, "op2=");                 debug_mp (b, -16);
      fprintf (stderr, "hgcd_ref: %ld\n", (long) res[0]);
      fprintf (stderr, "mpn_hgcd_appr: %ld\n", (long) res[1]);
      abort ();
    }

  refmpn_free_limbs (hgcd_init_tp - 1);
  refmpn_free_limbs (hgcd_tp - 1);
  hgcd_ref_clear (&ref);
  mpz_clear (ref_r0);
  mpz_clear (ref_r1);
  mpz_clear (hgcd_r0);
  mpz_clear (hgcd_r1);

  return res[0];
}

static void
hgcd_ref_init (struct hgcd_ref *hgcd)
{
  unsigned i;
  for (i = 0; i<2; i++)
    {
      unsigned j;
      for (j = 0; j<2; j++)
	mpz_init (hgcd->m[i][j]);
    }
}

static void
hgcd_ref_clear (struct hgcd_ref *hgcd)
{
  unsigned i;
  for (i = 0; i<2; i++)
    {
      unsigned j;
      for (j = 0; j<2; j++)
	mpz_clear (hgcd->m[i][j]);
    }
}

static int
sdiv_qr (mpz_t q, mpz_t r, mp_size_t s, const mpz_t a, const mpz_t b)
{
  mpz_fdiv_qr (q, r, a, b);
  if (mpz_size (r) <= s)
    {
      mpz_add (r, r, b);
      mpz_sub_ui (q, q, 1);
    }

  return (mpz_sgn (q) > 0);
}

static int
hgcd_ref (struct hgcd_ref *hgcd, mpz_t a, mpz_t b)
{
  mp_size_t n = MAX (mpz_size (a), mpz_size (b));
  mp_size_t s = n/2 + 1;
  mpz_t q;
  int res;

  if (mpz_size (a) <= s || mpz_size (b) <= s)
    return 0;

  res = mpz_cmp (a, b);
  if (res < 0)
    {
      mpz_sub (b, b, a);
      if (mpz_size (b) <= s)
	return 0;

      mpz_set_ui (hgcd->m[0][0], 1); mpz_set_ui (hgcd->m[0][1], 0);
      mpz_set_ui (hgcd->m[1][0], 1); mpz_set_ui (hgcd->m[1][1], 1);
    }
  else if (res > 0)
    {
      mpz_sub (a, a, b);
      if (mpz_size (a) <= s)
	return 0;

      mpz_set_ui (hgcd->m[0][0], 1); mpz_set_ui (hgcd->m[0][1], 1);
      mpz_set_ui (hgcd->m[1][0], 0); mpz_set_ui (hgcd->m[1][1], 1);
    }
  else
    return 0;

  mpz_init (q);

  for (;;)
    {
      ASSERT (mpz_size (a) > s);
      ASSERT (mpz_size (b) > s);

      if (mpz_cmp (a, b) > 0)
	{
	  if (!sdiv_qr (q, a, s, a, b))
	    break;
	  mpz_addmul (hgcd->m[0][1], q, hgcd->m[0][0]);
	  mpz_addmul (hgcd->m[1][1], q, hgcd->m[1][0]);
	}
      else
	{
	  if (!sdiv_qr (q, b, s, b, a))
	    break;
	  mpz_addmul (hgcd->m[0][0], q, hgcd->m[0][1]);
	  mpz_addmul (hgcd->m[1][0], q, hgcd->m[1][1]);
	}
    }

  mpz_clear (q);

  return 1;
}

static int
hgcd_ref_equal (const struct hgcd_ref *A, const struct hgcd_ref *B)
{
  unsigned i;

  for (i = 0; i<2; i++)
    {
      unsigned j;

      for (j = 0; j<2; j++)
	if (mpz_cmp (A->m[i][j], B->m[i][j]) != 0)
	  return 0;
    }

  return 1;
}

static int
hgcd_appr_valid_p (mpz_t a, mpz_t b, mp_size_t res0,
		   struct hgcd_ref *ref, mpz_t ref_r0, mpz_t ref_r1,
		   mp_size_t res1, struct hgcd_matrix *hgcd)
{
  mp_size_t n = MAX (mpz_size (a), mpz_size (b));
  mp_size_t s = n/2 + 1;

  mp_bitcnt_t dbits, abits, margin;
  mpz_t appr_r0, appr_r1, t, q;
  struct hgcd_ref appr;

  if (!res0)
    {
      if (!res1)
	return 1;

      fprintf (stderr, "mpn_hgcd_appr returned 1 when no reduction possible.\n");
      return 0;
    }

  /* NOTE: No *_clear calls on error return, since we're going to
     abort anyway. */
  mpz_init (t);
  mpz_init (q);
  hgcd_ref_init (&appr);
  mpz_init (appr_r0);
  mpz_init (appr_r1);

  if (mpz_size (ref_r0) <= s)
    {
      fprintf (stderr, "ref_r0 too small!!!: "); debug_mp (ref_r0, 16);
      return 0;
    }
  if (mpz_size (ref_r1) <= s)
    {
      fprintf (stderr, "ref_r1 too small!!!: "); debug_mp (ref_r1, 16);
      return 0;
    }

  mpz_sub (t, ref_r0, ref_r1);
  dbits = mpz_sizeinbase (t, 2);
  if (dbits > s*GMP_NUMB_BITS)
    {
      fprintf (stderr, "ref |r0 - r1| too large!!!: "); debug_mp (t, 16);
      return 0;
    }

  if (!res1)
    {
      mpz_set (appr_r0, a);
      mpz_set (appr_r1, b);
    }
  else
    {
      unsigned i;

      for (i = 0; i<2; i++)
	{
	  unsigned j;

	  for (j = 0; j<2; j++)
	    {
	      mp_size_t mn = hgcd->n;
	      MPN_NORMALIZE (hgcd->p[i][j], mn);
	      mpz_realloc (appr.m[i][j], mn);
	      MPN_COPY (PTR (appr.m[i][j]), hgcd->p[i][j], mn);
	      SIZ (appr.m[i][j]) = mn;
	    }
	}
      mpz_mul (appr_r0, appr.m[1][1], a);
      mpz_mul (t, appr.m[0][1], b);
      mpz_sub (appr_r0, appr_r0, t);
      if (mpz_sgn (appr_r0) <= 0
	  || mpz_size (appr_r0) <= s)
	{
	  fprintf (stderr, "appr_r0 too small: "); debug_mp (appr_r0, 16);
	  return 0;
	}

      mpz_mul (appr_r1, appr.m[1][0], a);
      mpz_mul (t, appr.m[0][0], b);
      mpz_sub (appr_r1, t, appr_r1);
      if (mpz_sgn (appr_r1) <= 0
	  || mpz_size (appr_r1) <= s)
	{
	  fprintf (stderr, "appr_r1 too small: "); debug_mp (appr_r1, 16);
	  return 0;
	}
    }

  mpz_sub (t, appr_r0, appr_r1);
  abits = mpz_sizeinbase (t, 2);
  if (abits < dbits)
    {
      fprintf (stderr, "|r0 - r1| too small: "); debug_mp (t, 16);
      return 0;
    }

  /* We lose one bit each time we discard the least significant limbs.
     For the lehmer code, that can happen at most s * (GMP_NUMB_BITS)
     / (GMP_NUMB_BITS - 1) times. For the dc code, we lose an entire
     limb (or more?) for each level of recursion. */

  margin = (n/2+1) * GMP_NUMB_BITS / (GMP_NUMB_BITS - 1);
  {
    mp_size_t rn;
    for (rn = n; ABOVE_THRESHOLD (rn, HGCD_APPR_THRESHOLD); rn = (rn + 1)/2)
      margin += GMP_NUMB_BITS;
  }

  if (verbose_flag && abits > dbits)
    fprintf (stderr, "n = %u: sbits = %u: ref #(r0-r1): %u, appr #(r0-r1): %u excess: %d, margin: %u\n",
	     (unsigned) n, (unsigned) s*GMP_NUMB_BITS,
	     (unsigned) dbits, (unsigned) abits,
	     (int) (abits - s * GMP_NUMB_BITS), (unsigned) margin);

  if (abits > s*GMP_NUMB_BITS + margin)
    {
      fprintf (stderr, "appr |r0 - r1| much larger than minimal (by %u bits, margin = %u bits)\n",
	       (unsigned) (abits - s*GMP_NUMB_BITS), (unsigned) margin);
      return 0;
    }

  while (mpz_cmp (appr_r0, ref_r0) > 0 || mpz_cmp (appr_r1, ref_r1) > 0)
    {
      ASSERT (mpz_size (appr_r0) > s);
      ASSERT (mpz_size (appr_r1) > s);

      if (mpz_cmp (appr_r0, appr_r1) > 0)
	{
	  if (!sdiv_qr (q, appr_r0, s, appr_r0, appr_r1))
	    break;
	  mpz_addmul (appr.m[0][1], q, appr.m[0][0]);
	  mpz_addmul (appr.m[1][1], q, appr.m[1][0]);
	}
      else
	{
	  if (!sdiv_qr (q, appr_r1, s, appr_r1, appr_r0))
	    break;
	  mpz_addmul (appr.m[0][0], q, appr.m[0][1]);
	  mpz_addmul (appr.m[1][0], q, appr.m[1][1]);
	}
    }

  if (mpz_cmp (appr_r0, ref_r0) != 0
      || mpz_cmp (appr_r1, ref_r1) != 0
      || !hgcd_ref_equal (ref, &appr))
    {
      fprintf (stderr, "appr_r0: "); debug_mp (appr_r0, 16);
      fprintf (stderr, "ref_r0: "); debug_mp (ref_r0, 16);

      fprintf (stderr, "appr_r1: "); debug_mp (appr_r1, 16);
      fprintf (stderr, "ref_r1: "); debug_mp (ref_r1, 16);

      return 0;
    }
  mpz_clear (t);
  mpz_clear (q);
  hgcd_ref_clear (&appr);
  mpz_clear (appr_r0);
  mpz_clear (appr_r1);

  return 1;
}
