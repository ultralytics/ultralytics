/* Test file for mpfr_sech.

Copyright 2005-2017 Free Software Foundation, Inc.
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

#include "mpfr-test.h"

#define TEST_FUNCTION mpfr_sech
#define TEST_RANDOM_EMIN -64
#define TEST_RANDOM_EMAX 36
#include "tgeneric.c"

static void
check_specials (void)
{
  mpfr_t  x, y;

  mpfr_init2 (x, 123L);
  mpfr_init2 (y, 123L);

  mpfr_set_nan (x);
  mpfr_sech (y, x, MPFR_RNDN);
  if (! mpfr_nan_p (y))
    {
      printf ("Error: sech(NaN) != NaN\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_sech (y, x, MPFR_RNDN);
  if (! (MPFR_IS_ZERO (y) && MPFR_SIGN (y) > 0))
    {
      printf ("Error: sech(+Inf) != +0\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_sech (y, x, MPFR_RNDN);
  if (! (MPFR_IS_ZERO (y) && MPFR_SIGN (y) > 0))
    {
      printf ("Error: sech(-Inf) != +0\n");
      exit (1);
    }

  /* sec(+/-0) = 1 */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_sech (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 1))
    {
      printf ("Error: sech(+0) != 1\n");
      exit (1);
    }
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_sech (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 1))
    {
      printf ("Error: sech(-0) != 1\n");
      exit (1);
    }

  /* check huge x */
  mpfr_set_str (x, "8e8", 10, MPFR_RNDN);
  mpfr_sech (y, x, MPFR_RNDN);
  if (! (mpfr_zero_p (y) && MPFR_SIGN (y) > 0))
    {
      printf ("Error: sech(8e8) != +0\n");
      exit (1);
    }
  mpfr_set_str (x, "-8e8", 10, MPFR_RNDN);
  mpfr_sech (y, x, MPFR_RNDN);
  if (! (mpfr_zero_p (y) && MPFR_SIGN (y) > 0))
    {
      printf ("Error: sech(-8e8) != +0\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
overflowed_sech0 (void)
{
  mpfr_t x, y;
  int emax, i, inex, rnd, err = 0;
  mpfr_exp_t old_emax;

  old_emax = mpfr_get_emax ();

  mpfr_init2 (x, 8);
  mpfr_init2 (y, 8);

  for (emax = -1; emax <= 0; emax++)
    {
      mpfr_set_ui_2exp (y, 1, emax, MPFR_RNDN);
      mpfr_nextbelow (y);
      set_emax (emax);  /* 1 is not representable. */
      /* and if emax < 0, 1 - eps is not representable either. */
      for (i = -1; i <= 1; i++)
        RND_LOOP (rnd)
          {
            mpfr_set_si_2exp (x, i, -512 * ABS (i), MPFR_RNDN);
            mpfr_clear_flags ();
            inex = mpfr_sech (x, x, (mpfr_rnd_t) rnd);
            if ((i == 0 || emax < 0 || rnd == MPFR_RNDN || rnd == MPFR_RNDU) &&
                ! mpfr_overflow_p ())
              {
                printf ("Error in overflowed_sech0 (i = %d, rnd = %s):\n"
                        "  The overflow flag is not set.\n",
                        i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                err = 1;
              }
            if (rnd == MPFR_RNDZ || rnd == MPFR_RNDD)
              {
                if (inex >= 0)
                  {
                    printf ("Error in overflowed_sech0 (i = %d, rnd = %s):\n"
                            "  The inexact value must be negative.\n",
                            i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                    err = 1;
                  }
                if (! mpfr_equal_p (x, y))
                  {
                    printf ("Error in overflowed_sech0 (i = %d, rnd = %s):\n"
                            "  Got ", i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                    mpfr_print_binary (x);
                    printf (" instead of 0.11111111E%d.\n", emax);
                    err = 1;
                  }
              }
            else
              {
                if (inex <= 0)
                  {
                    printf ("Error in overflowed_sech0 (i = %d, rnd = %s):\n"
                            "  The inexact value must be positive.\n",
                            i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                    err = 1;
                  }
                if (! (mpfr_inf_p (x) && MPFR_SIGN (x) > 0))
                  {
                    printf ("Error in overflowed_sech0 (i = %d, rnd = %s):\n"
                            "  Got ", i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                    mpfr_print_binary (x);
                    printf (" instead of +Inf.\n");
                    err = 1;
                  }
              }
          }
      set_emax (old_emax);
    }

  if (err)
    exit (1);
  mpfr_clear (x);
  mpfr_clear (y);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  check_specials ();
  test_generic (2, 200, 10);
  overflowed_sech0 ();

  tests_end_mpfr ();
  return 0;
}
