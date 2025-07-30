/* Test file for exceptions.

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
#include <stdlib.h>

#include "mpfr-test.h"

#define ERROR(s) do { printf(s"\n"); exit(1); } while(0)

/* Test powerof2 */
static void
check_powerof2 (void)
{
  mpfr_t x;

  mpfr_init (x);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_powerof2_raw (x));
  mpfr_set_ui (x, 3, MPFR_RNDN);
  MPFR_ASSERTN (!mpfr_powerof2_raw (x));
  mpfr_clear (x);
}

/* Test default rounding mode */
static void
check_default_rnd (void)
{
  int r;
  mpfr_rnd_t t;
  for(r = 0 ; r < MPFR_RND_MAX ; r++)
    {
      mpfr_set_default_rounding_mode ((mpfr_rnd_t) r);
      t = (mpfr_get_default_rounding_mode) ();
      if ((mpfr_rnd_t) r != t)
        {
          printf ("%s %s\n", mpfr_print_rnd_mode ((mpfr_rnd_t) r),
                  mpfr_print_rnd_mode (t));
          ERROR("ERROR in setting / getting default rounding mode (1)");
        }
    }
  mpfr_set_default_rounding_mode ((mpfr_rnd_t) MPFR_RND_MAX);
  if (mpfr_get_default_rounding_mode() != MPFR_RNDA)
    ERROR("ERROR in setting / getting default rounding mode (2)");
  mpfr_set_default_rounding_mode((mpfr_rnd_t) -1);
  if (mpfr_get_default_rounding_mode() != MPFR_RNDA)
    ERROR("ERROR in setting / getting default rounding mode (3)");
}

static void
check_emin_emax (void)
{
  mpfr_exp_t old_emin, old_emax;

  old_emin = mpfr_get_emin ();
  old_emax = mpfr_get_emax ();

  /* Check the functions not the macros ! */
  if ((mpfr_set_emin)(MPFR_EMIN_MIN) != 0)
    ERROR("set_emin failed!");
  if ((mpfr_get_emin)() != MPFR_EMIN_MIN)
    ERROR("get_emin FAILED!");
  if ((mpfr_set_emin)(MPFR_EMIN_MIN-1) == 0)
    ERROR("set_emin failed! (2)");

  if ((mpfr_set_emax)(MPFR_EMAX_MAX) != 0)
    ERROR("set_emax failed!");
  if ((mpfr_get_emax)() != MPFR_EMAX_MAX)
    ERROR("get_emax FAILED!");
  if ((mpfr_set_emax)(MPFR_EMAX_MAX+1) == 0)
    ERROR("set_emax failed! (2)");

  if ((mpfr_get_emin_min) () != MPFR_EMIN_MIN)
    ERROR ("get_emin_min");
  if ((mpfr_get_emin_max) () != MPFR_EMIN_MAX)
    ERROR ("get_emin_max");
  if ((mpfr_get_emax_min) () != MPFR_EMAX_MIN)
    ERROR ("get_emax_min");
  if ((mpfr_get_emax_max) () != MPFR_EMAX_MAX)
    ERROR ("get_emax_max");

  set_emin (old_emin);
  set_emax (old_emax);
}

static void
check_set_get_prec (void)
{
  mpfr_t x;

  mpfr_init2 (x, 17);
  if (mpfr_get_prec (x) != 17 || (mpfr_get_prec)(x) != 17)
    ERROR ("mpfr_get_prec");
  mpfr_clear (x);
}

static void
mpfr_set_double_range (void)
{
  mpfr_set_default_prec (54);
  if (mpfr_get_default_prec () != 54)
    ERROR ("get_default_prec failed (1)");
  mpfr_set_default_prec (53);
  if ((mpfr_get_default_prec) () != 53)
    ERROR ("get_default_prec failed (2)");

  /* in double precision format, the unbiased exponent is between 0 and
     2047, where 0 is used for subnormal numbers, and 2047 for special
     numbers (infinities, NaN), and the bias is 1023, thus "normal" numbers
     have an exponent between -1022 and 1023, corresponding to numbers
     between 2^(-1022) and previous(2^(1024)).
     (The smallest subnormal number is 0.(0^51)1*2^(-1022)= 2^(-1074).)

     The smallest normal power of two is 1.0*2^(-1022).
     The largest normal power of two is 2^1023.
     (We have to add one for mpfr since mantissa are between 1/2 and 1.)
  */

  set_emin (-1021);
  set_emax (1024);
}

static void
check_flags (void)
{
  mpfr_t x;
  mpfr_exp_t old_emin, old_emax;

  old_emin = mpfr_get_emin ();
  old_emax = mpfr_get_emax ();
  mpfr_init (x);

  /* Check the functions not the macros ! */
  (mpfr_clear_flags)();
  mpfr_set_double_range ();

  mpfr_set_ui (x, 1, MPFR_RNDN);
  (mpfr_clear_overflow)();
  mpfr_mul_2exp (x, x, 1024, MPFR_RNDN);
  if (!(mpfr_overflow_p)())
    ERROR("ERROR: No overflow detected!\n");

  (mpfr_clear_underflow)();
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_div_2exp (x, x, 1025, MPFR_RNDN);
  if (!(mpfr_underflow_p)())
    ERROR("ERROR: No underflow detected!\n");

  (mpfr_clear_nanflag)();
  MPFR_SET_NAN(x);
  mpfr_add (x, x, x, MPFR_RNDN);
  if (!(mpfr_nanflag_p)())
    ERROR("ERROR: No NaN flag!\n");

  (mpfr_clear_inexflag)();
  mpfr_set_ui(x, 2, MPFR_RNDN);
  mpfr_cos(x, x, MPFR_RNDN);
  if (!(mpfr_inexflag_p)())
    ERROR("ERROR: No inexact flag!\n");

  (mpfr_clear_erangeflag) ();
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_mul_2exp (x, x, 1024, MPFR_RNDN);
  mpfr_get_ui (x, MPFR_RNDN);
  if (!(mpfr_erangeflag_p)())
    ERROR ("ERROR: No erange flag!\n");

  mpfr_clear (x);
  set_emin (old_emin);
  set_emax (old_emax);
}

static void
test_set_underflow (void)
{
  mpfr_t x, zero, min;
  mpfr_ptr r[MPFR_RND_MAX];
  int t[MPFR_RND_MAX] = { 1, -1, 1, -1, 1 }; /* RNDN, RNDZ, RNDU, RNDD, RNDA */
  int i;
  int s;

  mpfr_inits (x, zero, min, (mpfr_ptr) 0);
  mpfr_set_ui (zero, 0, MPFR_RNDN);
  mpfr_set_ui (min, 0, MPFR_RNDN);
  mpfr_nextabove (min);
  r[0] = r[2] = r[4] = min; /* RNDN, RNDU, RNDA */
  r[1] = r[3] = zero;       /* RNDZ, RNDD */
  for (s = 1; s > 0; s = -1)
    {
      for (i = 0; i < MPFR_RND_MAX ; i++)
        {
          int j;
          int inex;

          j = s < 0 && i > 1 ? 5 - i : i;
          inex = mpfr_underflow (x, (mpfr_rnd_t) i, s);
          if (mpfr_cmp (x, r[j]) || inex * t[j] <= 0)
            {
              printf ("Error in test_set_underflow, sign = %d,"
                      " rnd_mode = %s\n", s, mpfr_print_rnd_mode ((mpfr_rnd_t) i));
              printf ("Got\n");
              mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
              printf (", inex = %d\ninstead of\n", inex);
              mpfr_out_str (stdout, 2, 0, r[j], MPFR_RNDN);
              printf (", inex = %d\n", t[j]);
              exit (1);
            }
        }
      mpfr_neg (zero, zero, MPFR_RNDN);
      mpfr_neg (min, min, MPFR_RNDN);
    }
  mpfr_clears (x, zero, min, (mpfr_ptr) 0);
}

static void
test_set_overflow (void)
{
  mpfr_t x, inf, max;
  mpfr_ptr r[MPFR_RND_MAX];
  int t[MPFR_RND_MAX] = { 1, -1, 1, -1, 1 }; /* RNDN, RNDZ, RNDU, RNDD, RNDA */
  int i;
  int s;

  mpfr_inits2 (32, x, inf, max, (mpfr_ptr) 0);
  mpfr_set_inf (inf, 1);
  mpfr_set_inf (max, 1);
  mpfr_nextbelow (max);
  r[0] = r[2] = r[4] = inf; /* RNDN, RNDU, RNDA */
  r[1] = r[3] = max;        /* RNDZ, RNDD */
  for (s = 1; s > 0; s = -1)
    {
      for (i = 0; i < MPFR_RND_MAX ; i++)
        {
          int j;
          int inex;

          j = s < 0 && i > 1 ? 5 - i : i;
          inex = mpfr_overflow (x, (mpfr_rnd_t) i, s);
          if (mpfr_cmp (x, r[j]) || inex * t[j] <= 0)
            {
              printf ("Error in test_set_overflow, sign = %d,"
                      " rnd_mode = %s\n", s, mpfr_print_rnd_mode ((mpfr_rnd_t) i));
              printf ("Got\n");
              mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
              printf (", inex = %d\ninstead of\n", inex);
              mpfr_out_str (stdout, 2, 0, r[j], MPFR_RNDN);
              printf (", inex = %d\n", t[j]);
              exit (1);
            }
        }
      mpfr_neg (inf, inf, MPFR_RNDN);
      mpfr_neg (max, max, MPFR_RNDN);
    }
  mpfr_clears (x, inf, max, (mpfr_ptr) 0);
}

static void
check_set (void)
{
  mpfr_clear_flags ();

  mpfr_set_overflow ();
  MPFR_ASSERTN ((mpfr_overflow_p) ());
  mpfr_set_underflow ();
  MPFR_ASSERTN ((mpfr_underflow_p) ());
  mpfr_set_divby0 ();
  MPFR_ASSERTN ((mpfr_divby0_p) ());
  mpfr_set_nanflag ();
  MPFR_ASSERTN ((mpfr_nanflag_p) ());
  mpfr_set_inexflag ();
  MPFR_ASSERTN ((mpfr_inexflag_p) ());
  mpfr_set_erangeflag ();
  MPFR_ASSERTN ((mpfr_erangeflag_p) ());

  MPFR_ASSERTN (__gmpfr_flags == MPFR_FLAGS_ALL);

  mpfr_clear_overflow ();
  MPFR_ASSERTN (! (mpfr_overflow_p) ());
  mpfr_clear_underflow ();
  MPFR_ASSERTN (! (mpfr_underflow_p) ());
  mpfr_clear_divby0 ();
  MPFR_ASSERTN (! (mpfr_divby0_p) ());
  mpfr_clear_nanflag ();
  MPFR_ASSERTN (! (mpfr_nanflag_p) ());
  mpfr_clear_inexflag ();
  MPFR_ASSERTN (! (mpfr_inexflag_p) ());
  mpfr_clear_erangeflag ();
  MPFR_ASSERTN (! (mpfr_erangeflag_p) ());

  MPFR_ASSERTN (__gmpfr_flags == 0);

  (mpfr_set_overflow) ();
  MPFR_ASSERTN (mpfr_overflow_p ());
  (mpfr_set_underflow) ();
  MPFR_ASSERTN (mpfr_underflow_p ());
  (mpfr_set_divby0) ();
  MPFR_ASSERTN (mpfr_divby0_p ());
  (mpfr_set_nanflag) ();
  MPFR_ASSERTN (mpfr_nanflag_p ());
  (mpfr_set_inexflag) ();
  MPFR_ASSERTN (mpfr_inexflag_p ());
  (mpfr_set_erangeflag) ();
  MPFR_ASSERTN (mpfr_erangeflag_p ());

  MPFR_ASSERTN (__gmpfr_flags == MPFR_FLAGS_ALL);

  (mpfr_clear_overflow) ();
  MPFR_ASSERTN (! mpfr_overflow_p ());
  (mpfr_clear_underflow) ();
  MPFR_ASSERTN (! mpfr_underflow_p ());
  (mpfr_clear_divby0) ();
  MPFR_ASSERTN (! mpfr_divby0_p ());
  (mpfr_clear_nanflag) ();
  MPFR_ASSERTN (! mpfr_nanflag_p ());
  (mpfr_clear_inexflag) ();
  MPFR_ASSERTN (! mpfr_inexflag_p ());
  (mpfr_clear_erangeflag) ();
  MPFR_ASSERTN (! mpfr_erangeflag_p ());

  MPFR_ASSERTN (__gmpfr_flags == 0);
}

int
main (int argc, char *argv[])
{
  mpfr_t x, y;
  mpfr_exp_t emin, emax;

  tests_start_mpfr ();

  test_set_underflow ();
  test_set_overflow ();
  check_default_rnd();

  mpfr_init (x);
  mpfr_init (y);

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  if (emin >= emax)
    {
      printf ("Error: emin >= emax\n");
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_mul_2exp (x, x, 1024, MPFR_RNDN);
  mpfr_set_double_range ();
  mpfr_check_range (x, 0, MPFR_RNDN);
  if (!mpfr_inf_p (x) || (mpfr_sgn(x) <= 0))
    {
      printf ("Error: 2^1024 rounded to nearest should give +Inf\n");
      exit (1);
    }

  set_emax (1025);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_mul_2exp (x, x, 1024, MPFR_RNDN);
  mpfr_set_double_range ();
  mpfr_check_range (x, 0, MPFR_RNDD);
  if (!mpfr_number_p (x))
    {
      printf ("Error: 2^1024 rounded down should give a normal number\n");
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_mul_2exp (x, x, 1023, MPFR_RNDN);
  mpfr_add (x, x, x, MPFR_RNDN);
  if (!mpfr_inf_p (x) || (mpfr_sgn(x) <= 0))
    {
      printf ("Error: x+x rounded to nearest for x=2^1023 should give +Inf\n");
      printf ("emax = %ld\n", (long) mpfr_get_emax ());
      printf ("got "); mpfr_print_binary (x); puts ("");
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_mul_2exp (x, x, 1023, MPFR_RNDN);
  mpfr_add (x, x, x, MPFR_RNDD);
  if (!mpfr_number_p (x))
    {
      printf ("Error: x+x rounded down for x=2^1023 should give"
              " a normal number\n");
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_div_2exp (x, x, 1022, MPFR_RNDN);
  mpfr_set_str_binary (y, "1.1e-1022"); /* y = 3/2*x */
  mpfr_sub (y, y, x, MPFR_RNDZ);
  if (mpfr_cmp_ui (y, 0))
    {
      printf ("Error: y-x rounded to zero should give 0"
              " for y=3/2*2^(-1022), x=2^(-1022)\n");
      printf ("y="); mpfr_print_binary (y); puts ("");
      exit (1);
    }

  set_emin (-1026);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_div_2exp (x, x, 1025, MPFR_RNDN);
  mpfr_set_double_range ();
  mpfr_check_range (x, 0, MPFR_RNDN);
  if (!MPFR_IS_ZERO (x) )
    {
      printf ("Error: x rounded to nearest for x=2^-1024 should give Zero\n");
      printf ("emin = %ld\n", (long) mpfr_get_emin ());
      printf ("got "); mpfr_dump (x);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);

  set_emin (emin);
  set_emax (emax);

  check_emin_emax();
  check_flags();
  check_set_get_prec ();
  check_powerof2 ();
  check_set ();

  tests_end_mpfr ();
  return 0;
}
