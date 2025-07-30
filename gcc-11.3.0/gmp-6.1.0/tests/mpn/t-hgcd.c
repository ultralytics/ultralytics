/* Test mpn_hgcd.

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
#include "tests.h"

static mp_size_t one_test (mpz_t, mpz_t, int);
static void debug_mp (mpz_t, int);

#define MIN_OPERAND_SIZE 2

/* Fixed values, for regression testing of mpn_hgcd. */
struct value { int res; const char *a; const char *b; };
static const struct value hgcd_values[] = {
#if GMP_NUMB_BITS == 32
  { 5,
    "0x1bddff867272a9296ac493c251d7f46f09a5591fe",
    "0xb55930a2a68a916450a7de006031068c5ddb0e5c" },
  { 4,
    "0x2f0ece5b1ee9c15e132a01d55768dc13",
    "0x1c6f4fd9873cdb24466e6d03e1cc66e7" },
  { 3, "0x7FFFFC003FFFFFFFFFC5", "0x3FFFFE001FFFFFFFFFE3"},
#endif
  { -1, NULL, NULL }
};

struct hgcd_ref
{
  mpz_t m[2][2];
};

static void hgcd_ref_init (struct hgcd_ref *);
static void hgcd_ref_clear (struct hgcd_ref *);
static int hgcd_ref (struct hgcd_ref *, mpz_t, mpz_t);
static int hgcd_ref_equal (const struct hgcd_matrix *, const struct hgcd_ref *);

int
main (int argc, char **argv)
{
  mpz_t op1, op2, temp1, temp2;
  int i, j, chain_len;
  gmp_randstate_ptr rands;
  mpz_t bs;
  unsigned long size_range;

  tests_start ();
  rands = RANDS;

  mpz_init (bs);
  mpz_init (op1);
  mpz_init (op2);
  mpz_init (temp1);
  mpz_init (temp2);

  for (i = 0; hgcd_values[i].res >= 0; i++)
    {
      mp_size_t res;

      mpz_set_str (op1, hgcd_values[i].a, 0);
      mpz_set_str (op2, hgcd_values[i].b, 0);

      res = one_test (op1, op2, -1-i);
      if (res != hgcd_values[i].res)
	{
	  fprintf (stderr, "ERROR in test %d\n", -1-i);
	  fprintf (stderr, "Bad return code from hgcd\n");
	  fprintf (stderr, "op1=");                 debug_mp (op1, -16);
	  fprintf (stderr, "op2=");                 debug_mp (op2, -16);
	  fprintf (stderr, "expected: %d\n", hgcd_values[i].res);
	  fprintf (stderr, "hgcd:     %d\n", (int) res);
	  abort ();
	}
    }

  for (i = 0; i < 15; i++)
    {
      /* Generate plain operands with unknown gcd.  These types of operands
	 have proven to trigger certain bugs in development versions of the
	 gcd code. */

      mpz_urandomb (bs, rands, 32);
      size_range = mpz_get_ui (bs) % 13 + 2;

      mpz_urandomb (bs, rands, size_range);
      mpz_rrandomb (op1, rands, mpz_get_ui (bs) + MIN_OPERAND_SIZE);
      mpz_urandomb (bs, rands, size_range);
      mpz_rrandomb (op2, rands, mpz_get_ui (bs) + MIN_OPERAND_SIZE);

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

static int
mpz_mpn_equal (const mpz_t a, mp_srcptr bp, mp_size_t bsize);

static mp_size_t
one_test (mpz_t a, mpz_t b, int i)
{
  struct hgcd_matrix hgcd;
  struct hgcd_ref ref;

  mpz_t ref_r0;
  mpz_t ref_r1;
  mpz_t hgcd_r0;
  mpz_t hgcd_r1;

  mp_size_t res[2];
  mp_size_t asize;
  mp_size_t bsize;

  mp_size_t hgcd_init_scratch;
  mp_size_t hgcd_scratch;

  mp_ptr hgcd_init_tp;
  mp_ptr hgcd_tp;

  asize = a->_mp_size;
  bsize = b->_mp_size;

  ASSERT (asize >= bsize);

  hgcd_init_scratch = MPN_HGCD_MATRIX_INIT_ITCH (asize);
  hgcd_init_tp = refmpn_malloc_limbs (hgcd_init_scratch);
  mpn_hgcd_matrix_init (&hgcd, asize, hgcd_init_tp);

  hgcd_scratch = mpn_hgcd_itch (asize);
  hgcd_tp = refmpn_malloc_limbs (hgcd_scratch);

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
  res[1] = mpn_hgcd (hgcd_r0->_mp_d,
		     hgcd_r1->_mp_d,
		     asize,
		     &hgcd, hgcd_tp);

  if (res[0] != res[1])
    {
      fprintf (stderr, "ERROR in test %d\n", i);
      fprintf (stderr, "Different return value from hgcd and hgcd_ref\n");
      fprintf (stderr, "op1=");                 debug_mp (a, -16);
      fprintf (stderr, "op2=");                 debug_mp (b, -16);
      fprintf (stderr, "hgcd_ref: %ld\n", (long) res[0]);
      fprintf (stderr, "mpn_hgcd: %ld\n", (long) res[1]);
      abort ();
    }
  if (res[0] > 0)
    {
      if (!hgcd_ref_equal (&hgcd, &ref)
	  || !mpz_mpn_equal (ref_r0, hgcd_r0->_mp_d, res[1])
	  || !mpz_mpn_equal (ref_r1, hgcd_r1->_mp_d, res[1]))
	{
	  fprintf (stderr, "ERROR in test %d\n", i);
	  fprintf (stderr, "mpn_hgcd and hgcd_ref returned different values\n");
	  fprintf (stderr, "op1=");                 debug_mp (a, -16);
	  fprintf (stderr, "op2=");                 debug_mp (b, -16);
	  abort ();
	}
    }

  refmpn_free_limbs (hgcd_init_tp);
  refmpn_free_limbs (hgcd_tp);
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
  mp_size_t asize;
  mp_size_t bsize;
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

  asize = mpz_size (a);
  bsize = mpz_size (b);
  return MAX (asize, bsize);
}

static int
mpz_mpn_equal (const mpz_t a, mp_srcptr bp, mp_size_t bsize)
{
  mp_srcptr ap = a->_mp_d;
  mp_size_t asize = a->_mp_size;

  MPN_NORMALIZE (bp, bsize);
  return asize == bsize && mpn_cmp (ap, bp, asize) == 0;
}

static int
hgcd_ref_equal (const struct hgcd_matrix *hgcd, const struct hgcd_ref *ref)
{
  unsigned i;

  for (i = 0; i<2; i++)
    {
      unsigned j;

      for (j = 0; j<2; j++)
	if (!mpz_mpn_equal (ref->m[i][j], hgcd->p[i][j], hgcd->n))
	  return 0;
    }

  return 1;
}
