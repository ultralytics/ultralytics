/* Test file for mpfr_hypot.

Copyright 2001-2017 Free Software Foundation, Inc.
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
#include <limits.h>
#include <stdlib.h>

#include "mpfr-test.h"

/* Non-zero when extended exponent range */
static int ext = 0;

static void
special (void)
{
  mpfr_t x, y, z;

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);

  mpfr_set_nan (x);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_hypot (z, x, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (z));
  mpfr_hypot (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (z));
  mpfr_hypot (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (z));

  mpfr_set_inf (x, 1);
  mpfr_set_inf (y, -1);
  mpfr_hypot (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (z) && mpfr_sgn (z) > 0);

  mpfr_set_inf (x, -1);
  mpfr_set_nan (y);
  mpfr_hypot (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (z) && mpfr_sgn (z) > 0);

  mpfr_set_nan (x);
  mpfr_set_inf (y, -1);
  mpfr_hypot (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (z) && mpfr_sgn (z) > 0);

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

static void
test_large (void)
{
  mpfr_t x, y, z, t;

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);
  mpfr_init (t);

  mpfr_set_ui (x, 21, MPFR_RNDN);
  mpfr_set_ui (y, 28, MPFR_RNDN);
  mpfr_set_ui (z, 35, MPFR_RNDN);

  mpfr_mul_2ui (x, x, MPFR_EMAX_DEFAULT-6, MPFR_RNDN);
  mpfr_mul_2ui (y, y, MPFR_EMAX_DEFAULT-6, MPFR_RNDN);
  mpfr_mul_2ui (z, z, MPFR_EMAX_DEFAULT-6, MPFR_RNDN);

  mpfr_hypot (t, x, y, MPFR_RNDN);
  if (mpfr_cmp (z, t))
    {
      printf ("Error in test_large: got\n");
      mpfr_out_str (stdout, 2, 0, t, MPFR_RNDN);
      printf ("\ninstead of\n");
      mpfr_out_str (stdout, 2, 0, z, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_set_prec (x, 53);
  mpfr_set_prec (t, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_str_binary (x, "0.11101100011110000011101000010101010011001101000001100E-1021");
  mpfr_set_str_binary (y, "0.11111001010011000001110110001101011100001000010010100E-1021");
  mpfr_hypot (t, x, y, MPFR_RNDN);
  mpfr_set_str_binary (z, "0.101010111100110111101110111110100110010011001010111E-1020");
  if (mpfr_cmp (z, t))
    {
      printf ("Error in test_large: got\n");
      mpfr_out_str (stdout, 2, 0, t, MPFR_RNDN);
      printf ("\ninstead of\n");
      mpfr_out_str (stdout, 2, 0, z, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_set_prec (x, 240);
  mpfr_set_prec (y, 22);
  mpfr_set_prec (z, 2);
  mpfr_set_prec (t, 2);
  mpfr_set_str_binary (x, "0.100111011010010010110100000100000001100010011100110101101111111101011110111011011101010110100101111000111100010100110000100101011110111011100110100110100101110101101100011000001100000001111101110100100100011011011010110111100110010101000111e-7");
  mpfr_set_str_binary (y, "0.1111000010000011000111E-10");
  mpfr_hypot (t, x, y, MPFR_RNDN);
  mpfr_set_str_binary (z, "0.11E-7");
  if (mpfr_cmp (z, t))
    {
      printf ("Error in test_large: got\n");
      mpfr_out_str (stdout, 2, 0, t, MPFR_RNDN);
      printf ("\ninstead of\n");
      mpfr_out_str (stdout, 2, 0, z, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (t);
}

static void
test_small (void)
{
  mpfr_t x, y, z1, z2;
  int inex1, inex2;
  unsigned int flags;

  /* Test hypot(x,x) with x = 2^(emin-1). Result is x * sqrt(2). */
  mpfr_inits2 (8, x, y, z1, z2, (mpfr_ptr) 0);
  mpfr_set_si_2exp (x, 1, mpfr_get_emin () - 1, MPFR_RNDN);
  mpfr_set_si_2exp (y, 1, mpfr_get_emin () - 1, MPFR_RNDN);
  mpfr_set_ui (z1, 2, MPFR_RNDN);
  inex1 = mpfr_sqrt (z1, z1, MPFR_RNDN);
  inex2 = mpfr_mul (z1, z1, x, MPFR_RNDN);
  MPFR_ASSERTN (inex2 == 0);
  mpfr_clear_flags ();
  inex2 = mpfr_hypot (z2, x, y, MPFR_RNDN);
  flags = __gmpfr_flags;
  if (mpfr_cmp (z1, z2) != 0)
    {
      printf ("Error in test_small%s\nExpected ",
              ext ? ", extended exponent range" : "");
      mpfr_out_str (stdout, 2, 0, z1, MPFR_RNDN);
      printf ("\nGot      ");
      mpfr_out_str (stdout, 2, 0, z2, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  if (! SAME_SIGN (inex1, inex2))
    {
      printf ("Bad ternary value in test_small%s\nExpected %d, got %d\n",
              ext ? ", extended exponent range" : "", inex1, inex2);
      exit (1);
    }
  if (flags != MPFR_FLAGS_INEXACT)
    {
      printf ("Bad flags in test_small%s\nExpected %u, got %u\n",
              ext ? ", extended exponent range" : "",
              (unsigned int) MPFR_FLAGS_INEXACT, flags);
      exit (1);
    }
  mpfr_clears (x, y, z1, z2, (mpfr_ptr) 0);
}

static void
test_large_small (void)
{
  mpfr_t x, y, z;
  int inexact, inex2, r;

  mpfr_init2 (x, 3);
  mpfr_init2 (y, 2);
  mpfr_init2 (z, 2);

  mpfr_set_ui_2exp (x, 1, mpfr_get_emax () / 2, MPFR_RNDN);
  mpfr_set_ui_2exp (y, 1, -1, MPFR_RNDN);
  inexact = mpfr_hypot (z, x, y, MPFR_RNDN);
  if (inexact >= 0 || mpfr_cmp (x, z))
    {
      printf ("Error 1 in test_large_small%s\n",
              ext ? ", extended exponent range" : "");
      exit (1);
    }

  mpfr_mul_ui (x, x, 5, MPFR_RNDN);
  inexact = mpfr_hypot (z, x, y, MPFR_RNDN);
  if (mpfr_cmp (x, z) >= 0)
    {
      printf ("Error 2 in test_large_small%s\n",
              ext ? ", extended exponent range" : "");
      printf ("x = ");
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
      printf ("\n");
      printf ("y = ");
      mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
      printf ("\n");
      printf ("z = ");
      mpfr_out_str (stdout, 2, 0, z, MPFR_RNDN);
      printf (" (in precision 2) instead of\n    ");
      mpfr_out_str (stdout, 2, 2, x, MPFR_RNDU);
      printf ("\n");
      exit (1);
    }

  RND_LOOP(r)
    {
      mpfr_set_ui_2exp (x, 1, mpfr_get_emax () - 1, MPFR_RNDN);
      mpfr_set_ui_2exp (y, 1, mpfr_get_emin (), MPFR_RNDN);
      inexact = mpfr_hypot (z, x, y, (mpfr_rnd_t) r);
      inex2 = mpfr_add_ui (y, x, 1, (mpfr_rnd_t) r);
      if (! mpfr_equal_p (y, z) || ! SAME_SIGN (inexact, inex2))
        {
          printf ("Error 3 in test_large_small, %s%s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r),
                  ext ? ", extended exponent range" : "");
          printf ("Expected ");
          mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
          printf (", inex = %d\n", inex2);
          printf ("Got      ");
          mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
          printf (", inex = %d\n", inexact);
          exit (1);
        }
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

static void
check_overflow (void)
{
  mpfr_t x, y;
  int inex, r;

  mpfr_inits2 (8, x, y, (mpfr_ptr) 0);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_setmax (x, mpfr_get_emax ());

  RND_LOOP(r)
    {
      mpfr_clear_overflow ();
      inex = mpfr_hypot (y, x, x, (mpfr_rnd_t) r);
      if (!mpfr_overflow_p ())
        {
          printf ("No overflow in check_overflow for %s%s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r),
                  ext ? ", extended exponent range" : "");
          exit (1);
        }
      MPFR_ASSERTN (MPFR_IS_POS (y));
      if (r == MPFR_RNDZ || r == MPFR_RNDD)
        {
          MPFR_ASSERTN (inex < 0);
          MPFR_ASSERTN (!mpfr_inf_p (y));
          mpfr_nexttoinf (y);
        }
      else
        {
          MPFR_ASSERTN (inex > 0);
        }
      MPFR_ASSERTN (mpfr_inf_p (y));
    }

  mpfr_clears (x, y, (mpfr_ptr) 0);
}

#define TWO_ARGS
#define TEST_FUNCTION mpfr_hypot
#include "tgeneric.c"

static void
alltst (void)
{
  mpfr_exp_t emin, emax;

  ext = 0;
  test_small ();
  test_large_small ();
  check_overflow ();

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  set_emin (MPFR_EMIN_MIN);
  set_emax (MPFR_EMAX_MAX);
  if (mpfr_get_emin () != emin || mpfr_get_emax () != emax)
    {
      ext = 1;
      test_small ();
      test_large_small ();
      check_overflow ();
      set_emin (emin);
      set_emax (emax);
    }
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  special ();

  test_large ();
  alltst ();

  test_generic (2, 100, 10);

  tests_end_mpfr ();
  return 0;
}
