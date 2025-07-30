/* Test file for mpfr_coth.

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

#define TEST_FUNCTION mpfr_coth
#include "tgeneric.c"

static void
check_specials (void)
{
  mpfr_t  x, y;

  mpfr_init2 (x, 123L);
  mpfr_init2 (y, 123L);

  mpfr_set_nan (x);
  mpfr_coth (y, x, MPFR_RNDN);
  if (! mpfr_nan_p (y))
    {
      printf ("Error: coth(NaN) != NaN\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_coth (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 1))
    {
      printf ("Error: coth(Inf) != 1\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_coth (y, x, MPFR_RNDN);
  if (mpfr_cmp_si (y, -1))
    {
      printf ("Error: coth(-Inf) != -1\n");
      exit (1);
    }

  /* coth(+/-0) = +/-Inf */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_coth (y, x, MPFR_RNDN);
  if (! (mpfr_inf_p (y) && MPFR_SIGN (y) > 0))
    {
      printf ("Error: coth(+0) != +Inf\n");
      exit (1);
    }
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_coth (y, x, MPFR_RNDN);
  if (! (mpfr_inf_p (y) && MPFR_SIGN (y) < 0))
    {
      printf ("Error: coth(-0) != -Inf\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
check_bugs (void)
{
  mpfr_t x, y;

  mpfr_init (x);
  mpfr_init (y);

  /* bug found by Rob (Sisyphus) on 16 Sep 2005 */
  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_set_prec (y, 2);
  mpfr_coth (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 1))
    {
      printf ("Error for coth(2), expected 1, got ");
      mpfr_dump (y);
      exit (1);
    }

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

  mpfr_set_str (x, "18.368400284838550", 10, MPFR_RNDN);
  mpfr_set_str (y, "1.0000000000000002", 10, MPFR_RNDN);
  mpfr_coth (x, x, MPFR_RNDN);
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("Error for coth(18.368400284838550)\n");
      exit (1);
    }

  mpfr_set_str (x, "18.714973875118520", 10, MPFR_RNDN);
  mpfr_coth (x, x, MPFR_RNDN);
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("Error for coth(18.714973875118520)\n");
      exit (1);
    }

  mpfr_set_str (x, "18.714973875118524", 10, MPFR_RNDN);
  mpfr_coth (x, x, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 1) != 0)
    {
      printf ("Error for coth(18.714973875118524)\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
underflowed_cothinf (void)
{
  mpfr_t x, y;
  int i, inex, rnd, err = 0;
  mpfr_exp_t old_emin;

  old_emin = mpfr_get_emin ();

  mpfr_init2 (x, 8);
  mpfr_init2 (y, 8);

  for (i = -1; i <= 1; i += 2)
    RND_LOOP (rnd)
      {
        mpfr_set_inf (x, i);
        mpfr_clear_flags ();
        set_emin (2);  /* 1 is not representable. */
        inex = mpfr_coth (x, x, (mpfr_rnd_t) rnd);
        set_emin (old_emin);
        if (! mpfr_underflow_p ())
          {
            printf ("Error in underflowed_cothinf (i = %d, rnd = %s):\n"
                    "  The underflow flag is not set.\n",
                    i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
            err = 1;
          }
        mpfr_set_si (y, (i < 0 && (rnd == MPFR_RNDD || rnd == MPFR_RNDA)) ||
                        (i > 0 && (rnd == MPFR_RNDU || rnd == MPFR_RNDA))
                     ? 2 : 0, MPFR_RNDN);
        if (i < 0)
          mpfr_neg (y, y, MPFR_RNDN);
        if (! (mpfr_equal_p (x, y) &&
               MPFR_MULT_SIGN (MPFR_SIGN (x), MPFR_SIGN (y)) > 0))
          {
            printf ("Error in underflowed_cothinf (i = %d, rnd = %s):\n"
                    "  Got ", i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
            mpfr_print_binary (x);
            printf (" instead of ");
            mpfr_print_binary (y);
            printf (".\n");
            err = 1;
          }
        if ((rnd == MPFR_RNDD ||
             (i > 0 && (rnd == MPFR_RNDN || rnd == MPFR_RNDZ))) && inex >= 0)
          {
            printf ("Error in underflowed_cothinf (i = %d, rnd = %s):\n"
                    "  The inexact value must be negative.\n",
                    i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
            err = 1;
          }
        if ((rnd == MPFR_RNDU ||
             (i < 0 && (rnd == MPFR_RNDN || rnd == MPFR_RNDZ))) && inex <= 0)
          {
            printf ("Error in underflowed_cothinf (i = %d, rnd = %s):\n"
                    "  The inexact value must be positive.\n",
                    i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
            err = 1;
          }
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
  check_bugs ();
  test_generic (2, 200, 10);
  underflowed_cothinf ();

  tests_end_mpfr ();
  return 0;
}
