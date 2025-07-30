/* Test mpf_div.

Copyright 2004 Free Software Foundation, Inc.

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


void
check_one (const char *desc, mpf_ptr got, mpf_srcptr u, mpf_srcptr v)
{
  if (! refmpf_validate_division ("mpf_div", got, u, v))
    {
      mp_trace_base = -16;
      mpf_trace ("  u", u);
      mpf_trace ("  v", v);
      printf    ("  %s\n", desc);
      abort ();
    }
}

void
check_rand (void)
{
  unsigned long  min_prec = __GMPF_BITS_TO_PREC (1);
  gmp_randstate_ptr  rands = RANDS;
  unsigned long  prec;
  mpf_t  got, u, v;
  int    i;

  mpf_init (got);
  mpf_init (u);
  mpf_init (v);

  /* separate */
  for (i = 0; i < 100; i++)
    {
      /* got precision */
      prec = min_prec + gmp_urandomm_ui (rands, 15L);
      refmpf_set_prec_limbs (got, prec);

      /* u */
      prec = min_prec + gmp_urandomm_ui (rands, 15L);
      refmpf_set_prec_limbs (u, prec);
      do {
        mpf_random2 (u, PREC(u), (mp_exp_t) 20);
      } while (SIZ(u) == 0);
      if (gmp_urandomb_ui (rands, 1L))
        mpf_neg (u, u);

      /* v */
      prec = min_prec + gmp_urandomm_ui (rands, 15L);
      refmpf_set_prec_limbs (v, prec);
      do {
        mpf_random2 (v, PREC(v), (mp_exp_t) 20);
      } while (SIZ(v) == 0);
      if (gmp_urandomb_ui (rands, 1L))
        mpf_neg (v, v);

      switch (i % 3) {
      case 0:
        mpf_div (got, u, v);
        check_one ("separate", got, u, v);
        break;
      case 1:
        prec = refmpf_set_overlap (got, u);
        mpf_div (got, got, v);
        check_one ("dst == u", got, u, v);
        mpf_set_prec_raw (got, prec);
        break;
      case 2:
        prec = refmpf_set_overlap (got, v);
        mpf_div (got, u, got);
        check_one ("dst == v", got, u, v);
        mpf_set_prec_raw (got, prec);
        break;
      }
    }

  mpf_clear (got);
  mpf_clear (u);
  mpf_clear (v);
}

/* Exercise calls mpf(x,x,x) */
void
check_reuse_three (void)
{
  unsigned long  min_prec = __GMPF_BITS_TO_PREC (1);
  gmp_randstate_ptr  rands = RANDS;
  unsigned long  result_prec, input_prec, set_prec;
  mpf_t  got;
  int    i;

  mpf_init (got);

  for (i = 0; i < 8; i++)
    {
      result_prec = min_prec + gmp_urandomm_ui (rands, 15L);
      input_prec = min_prec + gmp_urandomm_ui (rands, 15L);

      set_prec = MAX (result_prec, input_prec);
      refmpf_set_prec_limbs (got, set_prec);

      /* input, non-zero, possibly negative */
      PREC(got) = input_prec;
      do {
        mpf_random2 (got, input_prec, (mp_exp_t) 20);
      } while (SIZ(got) == 0);
      if (gmp_urandomb_ui (rands, 1L))
        mpf_neg (got, got);

      PREC(got) = result_prec;

      mpf_div (got, got, got);

      /* expect exactly 1.0 always */
      ASSERT_ALWAYS (mpf_cmp_ui (got, 1L) == 0);

      PREC(got) = set_prec;
    }

  mpf_clear (got);
}

void
check_various (void)
{
  mpf_t got, u, v;

  mpf_init (got);
  mpf_init (u);
  mpf_init (v);

  /* 100/4 == 25 */
  mpf_set_prec (got, 20L);
  mpf_set_ui (u, 100L);
  mpf_set_ui (v, 4L);
  mpf_div (got, u, v);
  MPF_CHECK_FORMAT (got);
  ASSERT_ALWAYS (mpf_cmp_ui (got, 25L) == 0);

  /* 1/(2^n+1), a case where truncating the divisor would be wrong */
  mpf_set_prec (got, 500L);
  mpf_set_prec (v, 900L);
  mpf_set_ui (v, 1L);
  mpf_mul_2exp (v, v, 800L);
  mpf_add_ui (v, v, 1L);
  mpf_div (got, u, v);
  check_one ("1/2^n+1, separate", got, u, v);

  mpf_clear (got);
  mpf_clear (u);
  mpf_clear (v);
}

int
main (void)
{
  tests_start ();

  check_various ();
  check_rand ();
  check_reuse_three ();

  tests_end ();
  exit (0);
}
