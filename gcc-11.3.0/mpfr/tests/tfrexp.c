/* Test file for mpfr_frexp.

Copyright 2011-2017 Free Software Foundation, Inc.
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

#include <stdlib.h> /* for exit */
#include "mpfr-test.h"

static void
check_special (void)
{
  mpfr_t x, y;
  int inex;
  mpfr_exp_t exp;

  mpfr_init2 (x, 53);
  mpfr_init2 (y, 53);

  mpfr_set_nan (x);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDN);
  if (mpfr_nan_p (y) == 0 || inex != 0)
    {
      printf ("Error for mpfr_frexp(NaN)\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDN);
  if (mpfr_inf_p (y) == 0 || mpfr_sgn (y) <= 0 || inex != 0)
    {
      printf ("Error for mpfr_frexp(+Inf)\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDN);
  if (mpfr_inf_p (y) == 0 || mpfr_sgn (y) >= 0 || inex != 0)
    {
      printf ("Error for mpfr_frexp(-Inf)\n");
      exit (1);
    }

  mpfr_set_zero (x, 1);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDN);
  if (mpfr_zero_p (y) == 0 || mpfr_signbit (y) != 0 || inex != 0 || exp != 0)
    {
      printf ("Error for mpfr_frexp(+0)\n");
      exit (1);
    }

  mpfr_set_zero (x, -1);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDN);
  if (mpfr_zero_p (y) == 0 || mpfr_signbit (y) == 0 || inex != 0 || exp != 0)
    {
      printf ("Error for mpfr_frexp(-0)\n");
      exit (1);
    }

  mpfr_set_ui (x, 17, MPFR_RNDN);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDN);
  /* 17 = 17/32*2^5 */
  if (mpfr_cmp_ui_2exp (y, 17, -5) != 0 || inex != 0 || exp != 5)
    {
      printf ("Error for mpfr_frexp(17)\n");
      exit (1);
    }

  mpfr_set_si (x, -17, MPFR_RNDN);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDN);
  if (mpfr_cmp_si_2exp (y, -17, -5) != 0 || inex != 0 || exp != 5)
    {
      printf ("Error for mpfr_frexp(-17)\n");
      exit (1);
    }

  /* now reduce the precision of y */
  mpfr_set_prec (y, 4);
  mpfr_set_ui (x, 17, MPFR_RNDN);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDN);
  /* 17 -> 16/32*2^5 */
  if (mpfr_cmp_ui_2exp (y, 16, -5) != 0 || inex >= 0 || exp != 5)
    {
      printf ("Error for mpfr_frexp(17) with prec=4, RNDN\n");
      exit (1);
    }

  mpfr_set_ui (x, 17, MPFR_RNDN);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDZ);
  if (mpfr_cmp_ui_2exp (y, 16, -5) != 0 || inex >= 0 || exp != 5)
    {
      printf ("Error for mpfr_frexp(17) with prec=4, RNDZ\n");
      exit (1);
    }

  mpfr_set_ui (x, 17, MPFR_RNDN);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDD);
  if (mpfr_cmp_ui_2exp (y, 16, -5) != 0 || inex >= 0 || exp != 5)
    {
      printf ("Error for mpfr_frexp(17) with prec=4, RNDD\n");
      exit (1);
    }

  mpfr_set_ui (x, 17, MPFR_RNDN);
  inex = mpfr_frexp (&exp, y, x, MPFR_RNDU);
  if (mpfr_cmp_ui_2exp (y, 18, -5) != 0 || inex <= 0 || exp != 5)
    {
      printf ("Error for mpfr_frexp(17) with prec=4, RNDU\n");
      exit (1);
    }

  mpfr_clear (y);
  mpfr_clear (x);
}

static void check1 (void)
{
  mpfr_exp_t emin, emax, e;
  mpfr_t x, y1, y2;
  int r, neg, red;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  set_emin (MPFR_EMIN_MIN);
  set_emax (MPFR_EMAX_MAX);

  mpfr_init2 (x, 7);
  mpfr_inits2 (4, y1, y2, (mpfr_ptr) 0);

  mpfr_set_ui_2exp (x, 1, -2, MPFR_RNDN);
  while (mpfr_regular_p (x))
    {
      /* Test the exponents up to 3 and with the maximum exponent
         (to check potential intermediate overflow). */
      if (MPFR_GET_EXP (x) == 4)
        mpfr_set_exp (x, MPFR_EMAX_MAX);
      e = MPFR_GET_EXP (x);
      for (neg = 0; neg < 2; neg++)
        {
          RND_LOOP (r)
            {
              int inex1, inex2;
              mpfr_exp_t e1, e2;
              unsigned int flags1, flags2;

              for (red = 0; red < 2; red++)
                {
                  if (red)
                    {
                      /* e1: exponent of the rounded value of x. */
                      MPFR_ASSERTN (e1 == e || e1 == e + 1);
                      set_emin (e);
                      set_emax (e);
                      mpfr_clear_flags ();
                      inex1 = e1 < 0 ?
                        mpfr_mul_2ui (y1, x, -e1, (mpfr_rnd_t) r) :
                        mpfr_div_2ui (y1, x, e1, (mpfr_rnd_t) r);
                      flags1 = __gmpfr_flags;
                    }
                  else
                    {
                      inex1 = mpfr_set (y1, x, (mpfr_rnd_t) r);
                      e1 = MPFR_IS_INF (y1) ? e + 1 : MPFR_GET_EXP (y1);
                      flags1 = inex1 != 0 ? MPFR_FLAGS_INEXACT : 0;
                    }
                  mpfr_clear_flags ();
                  inex2 = mpfr_frexp (&e2, y2, x, (mpfr_rnd_t) r);
                  flags2 = __gmpfr_flags;
                  set_emin (MPFR_EMIN_MIN);
                  set_emax (MPFR_EMAX_MAX);
                  if ((!red || e == 0) &&
                      (! mpfr_regular_p (y2) || MPFR_GET_EXP (y2) != 0))
                    {
                      printf ("Error in check1 for %s, red = %d, x = ",
                              mpfr_print_rnd_mode ((mpfr_rnd_t) r), red);
                      mpfr_dump (x);
                      printf ("Expected 1/2 <= |y| < 1, got y = ");
                      mpfr_dump (y2);
                      exit (1);
                    }
                  if (!red)
                    {
                      if (e2 > 0)
                        mpfr_mul_2ui (y2, y2, e2, MPFR_RNDN);
                      else if (e2 < 0)
                        mpfr_div_2ui (y2, y2, -e2, MPFR_RNDN);
                    }
                  if (! (SAME_SIGN (inex1, inex2) &&
                         mpfr_equal_p (y1, y2) &&
                         flags1 == flags2))
                    {
                      printf ("Error in check1 for %s, red = %d, x = ",
                              mpfr_print_rnd_mode ((mpfr_rnd_t) r), red);
                      mpfr_dump (x);
                      printf ("Expected y1 = ");
                      mpfr_dump (y1);
                      printf ("Got      y2 = ");
                      mpfr_dump (y2);
                      printf ("Expected inex ~= %d, got %d\n", inex1, inex2);
                      printf ("Expected flags:");
                      flags_out (flags1);
                      printf ("Got flags:     ");
                      flags_out (flags2);
                      exit (1);
                    }
                }
            }
          mpfr_neg (x, x, MPFR_RNDN);
        }
      mpfr_nextabove (x);
    }

  mpfr_clears (x, y1, y2, (mpfr_ptr) 0);
  set_emin (emin);
  set_emax (emax);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  check_special ();
  check1 ();

  tests_end_mpfr ();
  return 0;
}
