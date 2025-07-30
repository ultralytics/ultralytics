/* Test mpf_sqrt_ui.

Copyright 2004, 2015 Free Software Foundation, Inc.

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
check_rand (void)
{
  unsigned long      max_prec = 15;
  unsigned long      min_prec = __GMPF_BITS_TO_PREC (1);
  gmp_randstate_ptr  rands = RANDS;
  unsigned long      x, prec;
  mpf_t              r, s;
  int                i;

  mpf_init (r);
  mpf_init (s);
  refmpf_set_prec_limbs (s, 2*max_prec+10);

  for (x = 0; x < 2; x++)
    {
      mpf_sqrt_ui (r, x);
      MPF_CHECK_FORMAT (r);
      if (mpf_cmp_ui (r, x) != 0)
	{
	  printf    ("mpf_sqrt_ui wrong for special case:\n");
          printf    ("  x=%lu\n", x);
          mpf_trace ("  r", r);
	  abort ();
	}
    }

  for (i = 0; i < 50; i++)
    {
      /* input, a random non-zero ulong, exponentially distributed */
      do {
        x = gmp_urandomb_ui (rands,
                             gmp_urandomm_ui (rands, BITS_PER_ULONG) + 1);
      } while (x <= 1);

      /* result precision */
      prec = gmp_urandomm_ui (rands, max_prec-min_prec) + min_prec;
      refmpf_set_prec_limbs (r, prec);

      mpf_sqrt_ui (r, x);
      MPF_CHECK_FORMAT (r);

      /* Expect to prec limbs of result.
         In the current implementation there's no stripping of low zero
         limbs in mpf_sqrt_ui, not even on perfect squares, so size should
         be exactly prec.  */
      if (SIZ(r) != prec)
        {
          printf ("mpf_sqrt_ui result not enough result limbs\n");
          printf    ("  x=%lu\n", x);
          printf    ("  want prec=%lu\n", prec);
          mpf_trace ("  r", r);
          printf    ("  r size %ld\n", (long) SIZ(r));
          printf    ("  r prec %ld\n", (long) PREC(r));
          abort ();
        }

      /* Must have r^2 <= x, since r has been truncated. */
      mpf_mul (s, r, r);
      if (! (mpf_cmp_ui (s, x) <= 0))
        {
          printf    ("mpf_sqrt_ui result too big\n");
          printf    ("  x=%lu\n", x);
          printf    ("  want prec=%lu\n", prec);
          mpf_trace ("  r", r);
          mpf_trace ("  s", s);
          abort ();
        }

      /* Must have (r+ulp)^2 > x.
         No overflow from refmpf_add_ulp since r is only prec limbs. */
      refmpf_add_ulp (r);
      mpf_mul (s, r, r);
      if (! (mpf_cmp_ui (s, x) > 0))
        {
          printf    ("mpf_sqrt_ui result too small\n");
          printf    ("  x=%lu\n", x);
          printf    ("  want prec=%lu\n", prec);
          mpf_trace ("  r+ulp", r);
          mpf_trace ("  s", s);
          abort ();
        }
    }

  mpf_clear (r);
  mpf_clear (s);
}

int
main (int argc, char **argv)
{
  tests_start ();
  mp_trace_base = -16;

  check_rand ();

  tests_end ();
  exit (0);
}
