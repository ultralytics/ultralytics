/* Exercise mpf_mul_ui.

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
check_one (const char *desc, mpf_ptr got, mpf_srcptr u, unsigned long v)
{
  mp_size_t  usize, usign;
  mp_ptr     wp;
  mpf_t      want;

  MPF_CHECK_FORMAT (got);

  /* this code not nailified yet */
  ASSERT_ALWAYS (BITS_PER_ULONG <= GMP_NUMB_BITS);
  usign = SIZ (u);
  usize = ABS (usign);
  wp = refmpn_malloc_limbs (usize + 1);
  wp[usize] = mpn_mul_1 (wp, PTR(u), usize, (mp_limb_t) v);

  PTR(want) = wp;
  SIZ(want) = (usign >= 0 ? usize+1 : -(usize+1));
  EXP(want) = EXP(u) + 1;
  refmpf_normalize (want);

  if (! refmpf_validate ("mpf_mul_ui", got, want))
    {
      mp_trace_base = -16;
      printf    ("  %s\n", desc);
      mpf_trace ("  u", u);
      printf    ("  v %ld  0x%lX\n", v, v);
      abort ();
    }

  free (wp);
}

void
check_rand (void)
{
  unsigned long  min_prec = __GMPF_BITS_TO_PREC (1);
  gmp_randstate_ptr  rands = RANDS;
  mpf_t              got, u;
  unsigned long      prec, v;
  int                i;

  /* The nails code in mpf_mul_ui currently isn't exact, so suppress these
     tests for now.  */
  if (BITS_PER_ULONG > GMP_NUMB_BITS)
    return;

  mpf_init (got);
  mpf_init (u);

  for (i = 0; i < 200; i++)
    {
      /* got precision */
      prec = min_prec + gmp_urandomm_ui (rands, 15L);
      refmpf_set_prec_limbs (got, prec);

      /* u precision */
      prec = min_prec + gmp_urandomm_ui (rands, 15L);
      refmpf_set_prec_limbs (u, prec);

      /* u, possibly negative */
      mpf_random2 (u, PREC(u), (mp_exp_t) 20);
      if (gmp_urandomb_ui (rands, 1L))
        mpf_neg (u, u);

      /* v, 0 to BITS_PER_ULONG bits (inclusive) */
      prec = gmp_urandomm_ui (rands, BITS_PER_ULONG+1);
      v = gmp_urandomb_ui (rands, prec);

      if ((i % 2) == 0)
        {
          /* separate */
          mpf_mul_ui (got, u, v);
          check_one ("separate", got, u, v);
        }
      else
        {
          /* overlap */
          prec = refmpf_set_overlap (got, u);
          mpf_mul_ui (got, got, v);
          check_one ("overlap src==dst", got, u, v);

          mpf_set_prec_raw (got, prec);
        }
    }

  mpf_clear (got);
  mpf_clear (u);
}

void
check_various (void)
{
  mpf_t  u, got, want;
  const char   *s;

  mpf_init2 (u,    2*8*sizeof(long));
  mpf_init2 (got,  2*8*sizeof(long));
  mpf_init2 (want, 2*8*sizeof(long));

  s = "0 * ULONG_MAX";
  mpf_set_ui (u, 0L);
  mpf_mul_ui (got, u, ULONG_MAX);
  MPF_CHECK_FORMAT (got);
  mpf_set_ui (want, 0L);
  if (mpf_cmp (got, want) != 0)
    {
    error:
      printf ("Wrong result from %s\n", s);
      mpf_trace ("u   ", u);
      mpf_trace ("got ", got);
      mpf_trace ("want", want);
      abort ();
    }

  s = "1 * ULONG_MAX";
  mpf_set_ui (u, 1L);
  mpf_mul_ui (got, u, ULONG_MAX);
  MPF_CHECK_FORMAT (got);
  mpf_set_ui (want, ULONG_MAX);
  if (mpf_cmp (got, want) != 0)
    goto error;

  mpf_clear (u);
  mpf_clear (got);
  mpf_clear (want);
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
