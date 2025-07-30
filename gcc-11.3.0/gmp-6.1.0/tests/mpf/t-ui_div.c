/* Test mpf_ui_div.

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
check_one (const char *desc, mpf_ptr got, unsigned long u, mpf_srcptr v)
{
  mpf_t      uf;
  mp_limb_t  ulimbs[2];
  mp_size_t  usize;

  ulimbs[0] = u & GMP_NUMB_MASK;
  usize = (u != 0);
#if BITS_PER_ULONG > GMP_NUMB_BITS
  u >>= GMP_NUMB_BITS;
  ulimbs[1] = u;
  usize += (u != 0);
#endif
  PTR(uf) = ulimbs;
  SIZ(uf) = usize;
  EXP(uf) = usize;

  if (! refmpf_validate_division ("mpf_ui_div", got, uf, v))
    {
      mp_trace_base = -16;
      printf    ("  u 0x%lX  (%lu)\n", u, u);
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
  unsigned long  prec, u;
  mpf_t  got, v;
  int    i;

  mpf_init (got);
  mpf_init (v);

  for (i = 0; i < 200; i++)
    {
      /* got precision */
      prec = min_prec + gmp_urandomm_ui (rands, 15L);
      refmpf_set_prec_limbs (got, prec);

      /* u */
      prec = gmp_urandomm_ui (rands, BITS_PER_ULONG+1);
      u = gmp_urandomb_ui (rands, prec);

      /* v precision */
      prec = min_prec + gmp_urandomm_ui (rands, 15L);
      refmpf_set_prec_limbs (v, prec);

      /* v, non-zero */
      do {
        mpf_random2 (v, PREC(v), (mp_exp_t) 20);
      } while (SIZ(v) == 0);

      /* v possibly negative */
      if (gmp_urandomb_ui (rands, 1L))
        mpf_neg (v, v);

      if ((i % 2) == 0)
        {
          /* src != dst */
          mpf_ui_div (got, u, v);
          check_one ("separate", got, u, v);
        }
      else
        {
          /* src == dst */
          prec = refmpf_set_overlap (got, v);
          mpf_ui_div (got, u, got);
          check_one ("overlap src==dst", got, u, v);

          mpf_set_prec_raw (got, prec);
        }
    }

  mpf_clear (got);
  mpf_clear (v);
}

void
check_various (void)
{
  mpf_t got, v;

  mpf_init (got);
  mpf_init (v);

  /* 100/4 == 25 */
  mpf_set_prec (got, 20L);
  mpf_set_ui (v, 4L);
  mpf_ui_div (got, 100L, v);
  MPF_CHECK_FORMAT (got);
  ASSERT_ALWAYS (mpf_cmp_ui (got, 25L) == 0);

  {
    /* 1/(2^n+1), a case where truncating the divisor would be wrong */
    unsigned long  u = 1L;
    mpf_set_prec (got, 500L);
    mpf_set_prec (v, 900L);
    mpf_set_ui (v, 1L);
    mpf_mul_2exp (v, v, 800L);
    mpf_add_ui (v, v, 1L);
    mpf_ui_div (got, u, v);
    check_one ("1/2^n+1, separate", got, u, v);
  }

  mpf_clear (got);
  mpf_clear (v);
}

int
main (void)
{
  tests_start ();

  check_various ();
  check_rand ();

  tests_end ();
  exit (0);
}
