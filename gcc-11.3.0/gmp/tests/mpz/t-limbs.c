/* Test mpz_limbs_* functions

Copyright 2013 Free Software Foundation, Inc.

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

#include <stdlib.h>
#include <stdio.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

#define COUNT 100
#define BITSIZE 500

/* Like mpz_add. For simplicity, support positive inputs only. */
static void
alt_add (mpz_ptr r, mpz_srcptr a, mpz_srcptr b)
{
  mp_size_t an = mpz_size (a);
  mp_size_t bn = mpz_size (b);
  mp_ptr rp;

  ASSERT (an > 0);
  ASSERT (bn > 0);
  if (an < bn)
    {
      MP_SIZE_T_SWAP (an, bn);
      MPZ_SRCPTR_SWAP (a, b);
    }
  rp = mpz_limbs_modify (r, an + 1);
  rp[an] = mpn_add (rp, mpz_limbs_read (a), an, mpz_limbs_read (b), bn);
  mpz_limbs_finish (r, an + 1);
}

static void
check_funcs (const char *name,
	     void (*f)(mpz_ptr, mpz_srcptr, mpz_srcptr),
	     void (*ref_f)(mpz_ptr, mpz_srcptr, mpz_srcptr),
	     mpz_srcptr a, mpz_srcptr b)
{
  mpz_t r, ref;
  mpz_inits (r, ref, NULL);

  ref_f (ref, a, b);
  MPZ_CHECK_FORMAT (ref);
  f (r, a, b);
  MPZ_CHECK_FORMAT (r);

  if (mpz_cmp (r, ref) != 0)
    {
      printf ("%s failed, abits %u, bbits %u\n",
	      name,
	      (unsigned) mpz_sizeinbase (a, 2),
	      (unsigned) mpz_sizeinbase (b, 2));
      gmp_printf ("a = %Zx\n", a);
      gmp_printf ("b = %Zx\n", b);
      gmp_printf ("r = %Zx (bad)\n", r);
      gmp_printf ("ref = %Zx\n", ref);
      abort ();
    }
  mpz_clears (r, ref, NULL);
}

static void
check_add (void)
{
  gmp_randstate_ptr rands = RANDS;
  mpz_t bs, a, b;
  unsigned i;
  mpz_inits (bs, a, b, NULL);
  for (i = 0; i < COUNT; i++)
    {
      mpz_urandomb (bs, rands, 32);
      mpz_rrandomb (a, rands, 1 + mpz_get_ui (bs) % BITSIZE);
      mpz_urandomb (bs, rands, 32);
      mpz_rrandomb (b, rands, 1 + mpz_get_ui (bs) % BITSIZE);

      check_funcs ("add", alt_add, mpz_add, a, b);
    }
  mpz_clears (bs, a, b, NULL);
}

static void
alt_mul (mpz_ptr r, mpz_srcptr a, mpz_srcptr b)
{
  mp_size_t an = mpz_size (a);
  mp_size_t bn = mpz_size (b);
  mp_srcptr ap, bp;
  TMP_DECL;

  TMP_MARK;

  ASSERT (an > 0);
  ASSERT (bn > 0);
  if (an < bn)
    {
      MP_SIZE_T_SWAP (an, bn);
      MPZ_SRCPTR_SWAP (a, b);
    }
  /* NOTE: This copying seems unnecessary; better to allocate new
     result area, and free the old area when done. */
  if (r == a)
    {
      mp_ptr tp =  TMP_ALLOC_LIMBS (an);
      MPN_COPY (tp, mpz_limbs_read (a), an);
      ap = tp;
      bp = (a == b) ? ap : mpz_limbs_read (b);
    }
  else if (r == b)
    {
      mp_ptr tp = TMP_ALLOC_LIMBS (bn);
      MPN_COPY (tp, mpz_limbs_read (b), bn);
      bp = tp;
      ap = mpz_limbs_read (a);
    }
  else
    {
      ap = mpz_limbs_read (a);
      bp = mpz_limbs_read (b);
    }
  mpn_mul (mpz_limbs_write (r, an + bn),
	   ap, an, bp, bn);

  mpz_limbs_finish (r, an + bn);
}

void
check_mul (void)
{
  gmp_randstate_ptr rands = RANDS;
  mpz_t bs, a, b;
  unsigned i;
  mpz_inits (bs, a, b, NULL);
  for (i = 0; i < COUNT; i++)
    {
      mpz_urandomb (bs, rands, 32);
      mpz_rrandomb (a, rands, 1 + mpz_get_ui (bs) % BITSIZE);
      mpz_urandomb (bs, rands, 32);
      mpz_rrandomb (b, rands, 1 + mpz_get_ui (bs) % BITSIZE);

      check_funcs ("mul", alt_mul, mpz_mul, a, b);
    }
  mpz_clears (bs, a, b, NULL);
}

#define MAX_SIZE 100

static void
check_roinit (void)
{
  gmp_randstate_ptr rands = RANDS;
  mpz_t bs, a, b, r, ref;
  unsigned i;

  mpz_inits (bs, a, b, r, ref, NULL);

  for (i = 0; i < COUNT; i++)
    {
      mp_srcptr ap, bp;
      mp_size_t an, bn;
      mpz_urandomb (bs, rands, 32);
      mpz_rrandomb (a, rands, 1 + mpz_get_ui (bs) % BITSIZE);
      mpz_urandomb (bs, rands, 32);
      mpz_rrandomb (b, rands, 1 + mpz_get_ui (bs) % BITSIZE);

      an = mpz_size (a);
      ap = mpz_limbs_read (a);
      bn = mpz_size (b);
      bp = mpz_limbs_read (b);

      mpz_add (ref, a, b);
      {
	mpz_t a1, b1;
#if __STDC_VERSION__ >= 199901
	const mpz_t a2 = MPZ_ROINIT_N ( (mp_ptr) ap, an);
	const mpz_t b2 = MPZ_ROINIT_N ( (mp_ptr) bp, bn);

	mpz_set_ui (r, 0);
	mpz_add (r, a2, b2);
	if (mpz_cmp (r, ref) != 0)
	  {
	    printf ("MPZ_ROINIT_N failed\n");
	    gmp_printf ("a = %Zx\n", a);
	    gmp_printf ("b = %Zx\n", b);
	    gmp_printf ("r = %Zx (bad)\n", r);
	    gmp_printf ("ref = %Zx\n", ref);
	    abort ();
	  }
#endif
	mpz_set_ui (r, 0);
	mpz_add (r, mpz_roinit_n (a1, ap, an), mpz_roinit_n (b1, bp, bn));
	if (mpz_cmp (r, ref) != 0)
	  {
	    printf ("mpz_roinit_n failed\n");
	    gmp_printf ("a = %Zx\n", a);
	    gmp_printf ("b = %Zx\n", b);
	    gmp_printf ("r = %Zx (bad)\n", r);
	    gmp_printf ("ref = %Zx\n", ref);
	    abort ();
	  }
      }
    }
  mpz_clears (bs, a, b, r, ref, NULL);
}

int
main (int argc, char *argv[])
{
  tests_start ();
  tests_end ();

  check_add ();
  check_mul ();
  check_roinit ();

  return 0;

}
