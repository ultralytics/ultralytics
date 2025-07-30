/* Test file for mpfr_exp10.

Copyright 2007-2017 Free Software Foundation, Inc.
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
#include <limits.h>

#include "mpfr-test.h"

#define TEST_FUNCTION mpfr_exp10
#define TEST_RANDOM_EMIN -36
#define TEST_RANDOM_EMAX 36
#include "tgeneric.c"

static void
special_overflow (void)
{
  mpfr_t x, y;
  int inex;
  mpfr_exp_t emin, emax;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  set_emin (-125);
  set_emax (128);

  mpfr_init2 (x, 24);
  mpfr_init2 (y, 24);

  mpfr_set_str_binary (x, "0.101100100000000000110100E15");
  inex = mpfr_exp10 (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || inex <= 0)
    {
      printf ("Overflow error.\n");
      mpfr_dump (y);
      printf ("inex = %d\n", inex);
      exit (1);
    }

  mpfr_clear (y);
  mpfr_clear (x);
  set_emin (emin);
  set_emax (emax);
}

static void
emax_m_eps (void)
{
  if (mpfr_get_emax () <= LONG_MAX)
    {
      mpfr_t x, y;
      int inex, ov;

      mpfr_init2 (x, sizeof(mpfr_exp_t) * CHAR_BIT * 4);
      mpfr_init2 (y, 8);
      mpfr_set_si (x, mpfr_get_emax (), MPFR_RNDN);

      mpfr_clear_flags ();
      inex = mpfr_exp10 (y, x, MPFR_RNDN);
      ov = mpfr_overflow_p ();
      if (!ov || !mpfr_inf_p (y) || inex <= 0)
        {
          printf ("Overflow error for x = emax, MPFR_RNDN.\n");
          mpfr_dump (y);
          printf ("inex = %d, %soverflow\n", inex, ov ? "" : "no ");
          exit (1);
        }

      mpfr_clear (x);
      mpfr_clear (y);
    }
}

static void
exp_range (void)
{
  mpfr_t x;
  mpfr_exp_t emin;

  emin = mpfr_get_emin ();
  set_emin (3);
  mpfr_init2 (x, 16);
  mpfr_set_ui (x, 4, MPFR_RNDN);
  mpfr_exp10 (x, x, MPFR_RNDN);
  set_emin (emin);
  if (mpfr_nan_p (x) || mpfr_cmp_ui (x, 10000) != 0)
    {
      printf ("Error in mpfr_exp10 for x = 4, with emin = 3\n");
      printf ("Expected 10000, got ");
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  mpfr_clear (x);
}

static void
overfl_exp10_0 (void)
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
            inex = mpfr_exp10 (x, x, (mpfr_rnd_t) rnd);
            if ((i >= 0 || emax < 0 || rnd == MPFR_RNDN || rnd == MPFR_RNDU) &&
                ! mpfr_overflow_p ())
              {
                printf ("Error in overfl_exp10_0 (i = %d, rnd = %s):\n"
                        "  The overflow flag is not set.\n",
                        i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                err = 1;
              }
            if (rnd == MPFR_RNDZ || rnd == MPFR_RNDD)
              {
                if (inex >= 0)
                  {
                    printf ("Error in overfl_exp10_0 (i = %d, rnd = %s):\n"
                            "  The inexact value must be negative.\n",
                            i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                    err = 1;
                  }
                if (! mpfr_equal_p (x, y))
                  {
                    printf ("Error in overfl_exp10_0 (i = %d, rnd = %s):\n"
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
                    printf ("Error in overfl_exp10_0 (i = %d, rnd = %s):\n"
                            "  The inexact value must be positive.\n",
                            i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                    err = 1;
                  }
                if (! (mpfr_inf_p (x) && MPFR_SIGN (x) > 0))
                  {
                    printf ("Error in overfl_exp10_0 (i = %d, rnd = %s):\n"
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
  mpfr_t x, y;
  mpfr_exp_t emin, emax;
  int inex, ov;

  tests_start_mpfr ();

  special_overflow ();
  emax_m_eps ();
  exp_range ();

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_ui (x, 4, MPFR_RNDN);
  mpfr_exp10 (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 10000) != 0)
    {
      printf ("Error for 10^4, MPFR_RNDN\n");
      exit (1);
    }
  mpfr_exp10 (y, x, MPFR_RNDD);
  if (mpfr_cmp_ui (y, 10000) != 0)
    {
      printf ("Error for 10^4, MPFR_RNDD\n");
      exit (1);
    }
  mpfr_exp10 (y, x, MPFR_RNDU);
  if (mpfr_cmp_ui (y, 10000) != 0)
    {
      printf ("Error for 10^4, MPFR_RNDU\n");
      exit (1);
    }

  mpfr_set_prec (x, 10);
  mpfr_set_prec (y, 10);
  /* save emin */
  emin = mpfr_get_emin ();
  set_emin (-11);
  mpfr_set_si (x, -4, MPFR_RNDN);
  mpfr_exp10 (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || mpfr_sgn (y) < 0)
    {
      printf ("Error for emin = -11, x = -4, RNDN\n");
      printf ("Expected +0\n");
      printf ("Got      "); mpfr_print_binary (y); puts ("");
      exit (1);
    }
  /* restore emin */
  set_emin (emin);

  /* save emax */
  emax = mpfr_get_emax ();
  set_emax (13);
  mpfr_set_ui (x, 4, MPFR_RNDN);
  mpfr_exp10 (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error for emax = 13, x = 4, RNDN\n");
      printf ("Expected +inf\n");
      printf ("Got      "); mpfr_print_binary (y); puts ("");
      exit (1);
    }
  /* restore emax */
  set_emax (emax);

  MPFR_SET_INF (x);
  MPFR_SET_POS (x);
  mpfr_exp10 (y, x, MPFR_RNDN);
  if (!MPFR_IS_INF (y))
    {
      printf ("evaluation of function in INF does not return INF\n");
      exit (1);
    }

  MPFR_CHANGE_SIGN (x);
  mpfr_exp10 (y, x, MPFR_RNDN);
  if (!MPFR_IS_ZERO (y))
    {
      printf ("evaluation of function in -INF does not return 0\n");
      exit (1);
    }

  MPFR_SET_NAN (x);
  mpfr_exp10 (y, x, MPFR_RNDN);
  if (!MPFR_IS_NAN (y))
    {
      printf ("evaluation of function in NaN does not return NaN\n");
      exit (1);
    }

  if ((mpfr_uexp_t) 8 << 31 != 0 ||
      mpfr_get_emax () <= (mpfr_uexp_t) 100000 * 100000)
    {
      /* emax <= 10000000000 */
      mpfr_set_prec (x, 40);
      mpfr_set_prec (y, 40);
      mpfr_set_str (x, "3010299957", 10, MPFR_RNDN);
      mpfr_clear_flags ();
      inex = mpfr_exp10 (y, x, MPFR_RNDN);
      ov = mpfr_overflow_p ();
      if (!(MPFR_IS_INF (y) && MPFR_IS_POS (y) && ov))
        {
          printf ("Overflow error for x = 3010299957, MPFR_RNDN.\n");
          mpfr_dump (y);
          printf ("inex = %d, %soverflow\n", inex, ov ? "" : "no ");
          exit (1);
        }
    }

  test_generic (2, 100, 100);

  mpfr_clear (x);
  mpfr_clear (y);

  overfl_exp10_0 ();

  data_check ("data/exp10", mpfr_exp10, "mpfr_exp10");

  tests_end_mpfr ();
  return 0;
}
