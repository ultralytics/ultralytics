/* Test mpfr_get_d_2exp.

Copyright 1999-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "mpfr-test.h"


/* Check that hardware rounding doesn't make mpfr_get_d_2exp return a value
   outside its defined range. */
static void
check_round (void)
{
  static const unsigned long data[] = { 1, 32, 53, 54, 64, 128, 256, 512 };
  mpfr_t  f;
  double  got;
  long    got_exp;
  int     i, rnd_mode, neg;

  mpfr_init2 (f, 1024L);

  for (rnd_mode = 0; rnd_mode < MPFR_RND_MAX ; rnd_mode++)
    {
      for (i = 0; i < (int) numberof (data); i++)
        {
          mpfr_set_ui (f, 1L, MPFR_RNDZ);
          mpfr_mul_2exp (f, f, data[i], MPFR_RNDZ);
          mpfr_sub_ui (f, f, 1L, MPFR_RNDZ);

          for (neg = 0; neg <= 1; neg++)
            {
              got = mpfr_get_d_2exp (&got_exp, f, (mpfr_rnd_t) rnd_mode);

              if (neg == 0
                  ? (got < 0.5 || got >= 1.0)
                  : (got <= -1.0 || got > -0.5))
                {
                  printf  ("mpfr_get_d_2exp wrong on 2**%lu-1\n", data[i]);
                  printf  ("result out of range, expect 0.5 <= got < 1.0\n");
                  printf  ("   rnd_mode = %d\n", rnd_mode);
                  printf  ("   data[i]  = %lu\n", data[i]);
                  printf  ("   f    ");
                  mpfr_out_str (stdout, 2, 0, f, MPFR_RNDN);
                  printf  ("\n");
                  d_trace ("   got  ", got);
                  printf  ("   got exp  %ld\n", got_exp);
                  exit(1);
                }

              mpfr_neg (f, f, MPFR_RNDZ);
            }
        }
    }

  mpfr_clear (f);
}


static void
check_inf_nan (void)
{
  /* only if nans and infs are available */
#if _GMP_IEEE_FLOATS
  mpfr_t  x;
  double  d;
  long    exp;

  mpfr_init2 (x, 123);

  mpfr_set_inf (x, 1);
  d = mpfr_get_d_2exp (&exp, x, MPFR_RNDZ);
  MPFR_ASSERTN (d > 0);
  MPFR_ASSERTN (DOUBLE_ISINF (d));

  mpfr_set_inf (x, -1);
  d = mpfr_get_d_2exp (&exp, x, MPFR_RNDZ);
  MPFR_ASSERTN (d < 0);
  MPFR_ASSERTN (DOUBLE_ISINF (d));

  mpfr_set_nan (x);
  d = mpfr_get_d_2exp (&exp, x, MPFR_RNDZ);
  MPFR_ASSERTN (DOUBLE_ISNAN (d));

  mpfr_clear (x);
#endif
}


int
main (void)
{
  tests_start_mpfr ();
  mpfr_test_init ();

  check_round ();
  check_inf_nan ();

  tests_end_mpfr ();
  return 0;
}
