/* Test file for mpfr_exp.

Copyright 1999, 2001-2017 Free Software Foundation, Inc.
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

#ifdef CHECK_EXTERNAL
static int
test_exp (mpfr_ptr a, mpfr_srcptr b, mpfr_rnd_t rnd_mode)
{
  int res;
  int ok = rnd_mode == MPFR_RNDN && mpfr_number_p (b) && mpfr_get_prec (a)>=53;
  if (ok)
    {
      mpfr_print_raw (b);
    }
  res = mpfr_exp (a, b, rnd_mode);
  if (ok)
    {
      printf (" ");
      mpfr_print_raw (a);
      printf ("\n");
    }
  return res;
}
#else
#define test_exp mpfr_exp
#endif

/* returns the number of ulp of error */
static void
check3 (const char *op, mpfr_rnd_t rnd, const char *res)
{
  mpfr_t x, y;

  mpfr_inits2 (53, x, y, (mpfr_ptr) 0);
  /* y negative. If we forget to set the sign in mpfr_exp, we'll see it. */
  mpfr_set_si (y, -1, MPFR_RNDN);
  mpfr_set_str1 (x, op);
  test_exp (y, x, rnd);
  if (mpfr_cmp_str1 (y, res) )
    {
      printf ("mpfr_exp failed for x=%s, rnd=%s\n",
              op, mpfr_print_rnd_mode (rnd));
      printf ("expected result is %s, got ", res);
      mpfr_out_str (stdout, 10, 0, y, MPFR_RNDN);
      putchar('\n');
      exit (1);
    }
  mpfr_clears (x, y, (mpfr_ptr) 0);
}

/* expx is the value of exp(X) rounded toward -infinity */
static void
check_worst_case (const char *Xs, const char *expxs)
{
  mpfr_t x, y;

  mpfr_inits2 (53, x, y, (mpfr_ptr) 0);
  mpfr_set_str1(x, Xs);
  test_exp(y, x, MPFR_RNDD);
  if (mpfr_cmp_str1 (y, expxs))
    {
      printf ("exp(x) rounded toward -infinity is wrong\n");
      exit(1);
    }
  mpfr_set_str1(x, Xs);
  test_exp(x, x, MPFR_RNDU);
  mpfr_nexttoinf (y);
  if (mpfr_cmp(x,y))
    {
      printf ("exp(x) rounded toward +infinity is wrong\n");
      exit(1);
    }
  mpfr_clears (x, y, (mpfr_ptr) 0);
}

/* worst cases communicated by Jean-Michel Muller and Vincent Lefevre */
static int
check_worst_cases (void)
{
  mpfr_t x; mpfr_t y;

  mpfr_init(x);
  mpfr_set_prec (x, 53);

  check_worst_case("4.44089209850062517562e-16", "1.00000000000000022204");
  check_worst_case("6.39488462184069720009e-14", "1.00000000000006372680");
  check_worst_case("1.84741111297455401935e-12", "1.00000000000184718907");
  check_worst_case("1.76177628026265550074e-10", "1.00000000017617751702");
  check3("1.76177628026265550074e-10", MPFR_RNDN, "1.00000000017617773906");
  check_worst_case("7.54175277499595900852e-10", "1.00000000075417516676");
  check3("7.54175277499595900852e-10", MPFR_RNDN, "1.00000000075417538881");
  /* bug found by Vincent Lefe`vre on December 8, 1999 */
  check3("-5.42410311287441459172e+02", MPFR_RNDN, "2.7176584868845723e-236");
  /* further cases communicated by Vincent Lefe`vre on January 27, 2000 */
  check3("-1.32920285897904911589e-10", MPFR_RNDN, "0.999999999867079769622");
  check3("-1.44037948245738330735e-10", MPFR_RNDN, "0.9999999998559621072757");
  check3("-1.66795910430705305937e-10", MPFR_RNDZ, "0.9999999998332040895832");
  check3("-1.64310953745426656203e-10", MPFR_RNDN, "0.9999999998356891017792");
  check3("-1.38323574826034659172e-10", MPFR_RNDZ, "0.9999999998616764251835");
  check3("-1.23621668465115401498e-10", MPFR_RNDZ, "0.9999999998763783315425");

  mpfr_set_prec (x, 601);
  mpfr_set_str (x, "0.88b6ba510e10450edc258748bc9dfdd466f21b47ed264cdf24aa8f64af1f3fad9ec2301d43c0743f534b5aa20091ff6d352df458ef1ba519811ef6f5b11853534fd8fa32764a0a6d2d0dd20@0", 16, MPFR_RNDZ);
  mpfr_init2 (y, 601);
  mpfr_exp_2 (y, x, MPFR_RNDD);
  mpfr_exp_3 (x, x, MPFR_RNDD);
  if (mpfr_cmp (x, y))
    {
      printf ("mpfr_exp_2 and mpfr_exp_3 differ for prec=601\n");
      printf ("mpfr_exp_2 gives ");
      mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
      printf ("\nmpfr_exp_3 gives ");
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_set_prec (x, 13001);
  mpfr_set_prec (y, 13001);
  mpfr_urandomb (x, RANDS);
  mpfr_exp_3 (y, x, MPFR_RNDN);
  mpfr_exp_2 (x, x, MPFR_RNDN);
  if (mpfr_cmp (x, y))
    {
      printf ("mpfr_exp_2 and mpfr_exp_3 differ for prec=13001\n");
      exit (1);
    }

  mpfr_set_prec (x, 118);
  mpfr_set_str_binary (x, "0.1110010100011101010000111110011000000000000000000000000000000000000000000000000000000000000000000000000000000000000000E-86");
  mpfr_set_prec (y, 118);
  mpfr_exp_2 (y, x, MPFR_RNDU);
  mpfr_exp_3 (x, x, MPFR_RNDU);
  if (mpfr_cmp (x, y))
    {
      printf ("mpfr_exp_2 and mpfr_exp_3 differ for prec=118\n");
      printf ("mpfr_exp_2 gives ");
      mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
      printf ("\nmpfr_exp_3 gives ");
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  return 0;
}

static void
compare_exp2_exp3 (mpfr_prec_t p0, mpfr_prec_t p1)
{
  mpfr_t x, y, z;
  mpfr_prec_t prec;
  mpfr_rnd_t rnd;

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);
  for (prec = p0; prec <= p1; prec ++)
    {
      mpfr_set_prec (x, prec);
      mpfr_set_prec (y, prec);
      mpfr_set_prec (z, prec);
      do
        mpfr_urandomb (x, RANDS);
      while (MPFR_IS_ZERO (x));  /* 0 is handled by mpfr_exp only */
      rnd = RND_RAND ();
      mpfr_exp_2 (y, x, rnd);
      mpfr_exp_3 (z, x, rnd);
      if (mpfr_cmp (y,z))
        {
          printf ("mpfr_exp_2 and mpfr_exp_3 disagree for rnd=%s and\nx=",
                  mpfr_print_rnd_mode (rnd));
          mpfr_print_binary (x);
          puts ("");
          printf ("mpfr_exp_2 gives ");
          mpfr_print_binary (y);
          puts ("");
          printf ("mpfr_exp_3 gives ");
          mpfr_print_binary (z);
          puts ("");
          exit (1);
        }
  }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

static void
check_large (void)
{
  mpfr_t x, z;
  mpfr_prec_t prec;

  /* bug found by Patrick Pe'lissier on 7 Jun 2004 */
  prec = 203780;
  mpfr_init2 (x, prec);
  mpfr_init2 (z, prec);
  mpfr_set_ui (x, 3, MPFR_RNDN);
  mpfr_sqrt (x, x, MPFR_RNDN);
  mpfr_sub_ui (x, x, 1, MPFR_RNDN);
  mpfr_exp_3 (z, x, MPFR_RNDN);
  mpfr_clear (x);
  mpfr_clear (z);
}

#define TEST_FUNCTION test_exp
#define TEST_RANDOM_EMIN -36
#define TEST_RANDOM_EMAX 36
#include "tgeneric.c"

static void
check_special (void)
{
  mpfr_t x, y, z;
  mpfr_exp_t emin, emax;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);

  /* check exp(NaN) = NaN */
  mpfr_set_nan (x);
  test_exp (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error for exp(NaN)\n");
      exit (1);
    }

  /* check exp(+inf) = +inf */
  mpfr_set_inf (x, 1);
  test_exp (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error for exp(+inf)\n");
      exit (1);
    }

  /* check exp(-inf) = +0 */
  mpfr_set_inf (x, -1);
  test_exp (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || mpfr_sgn (y) < 0)
    {
      printf ("Error for exp(-inf)\n");
      exit (1);
    }

  /* Check overflow. Corner case of mpfr_exp_2 */
  mpfr_set_prec (x, 64);
  mpfr_set_emax (MPFR_EMAX_DEFAULT);
  mpfr_set_emin (MPFR_EMIN_DEFAULT);
  mpfr_set_str (x,
    "0.1011000101110010000101111111010100001100000001110001100111001101E30",
                2, MPFR_RNDN);
  mpfr_exp (x, x, MPFR_RNDD);
  if (mpfr_cmp_str (x,
".1111111111111111111111111111111111111111111111111111111111111111E1073741823",
                    2, MPFR_RNDN) != 0)
    {
      printf ("Wrong overflow detection in mpfr_exp\n");
      mpfr_dump (x);
      exit (1);
    }
  /* Check underflow. Corner case of mpfr_exp_2 */
  mpfr_set_str (x,
"-0.1011000101110010000101111111011111010001110011110111100110101100E30",
                2, MPFR_RNDN);
  mpfr_exp (x, x, MPFR_RNDN);
  if (mpfr_cmp_str (x, "0.1E-1073741823", 2, MPFR_RNDN) != 0)
    {
      printf ("Wrong underflow (1) detection in mpfr_exp\n");
      mpfr_dump (x);
      exit (1);
    }
  mpfr_set_str (x,
"-0.1011001101110010000101111111011111010001110011110111100110111101E30",
                2, MPFR_RNDN);
  mpfr_exp (x, x, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 0) != 0)
    {
      printf ("Wrong underflow (2) detection in mpfr_exp\n");
      mpfr_dump (x);
      exit (1);
    }
  /* Check overflow. Corner case of mpfr_exp_3 */
  if (MPFR_PREC_MAX >= MPFR_EXP_THRESHOLD + 10 && MPFR_PREC_MAX >= 64)
    {
      /* this ensures that for small MPFR_EXP_THRESHOLD, the following
         mpfr_set_str conversion is exact */
      mpfr_set_prec (x, (MPFR_EXP_THRESHOLD + 10 > 64)
                       ? MPFR_EXP_THRESHOLD + 10 : 64);
      mpfr_set_str (x,
       "0.1011000101110010000101111111010100001100000001110001100111001101E30",
                    2, MPFR_RNDN);
      mpfr_clear_overflow ();
      mpfr_exp (x, x, MPFR_RNDD);
      if (!mpfr_overflow_p ())
        {
          printf ("Wrong overflow detection in mpfr_exp_3\n");
          mpfr_dump (x);
          exit (1);
        }
      /* Check underflow. Corner case of mpfr_exp_3 */
      mpfr_set_str (x,
      "-0.1011000101110010000101111111011111010001110011110111100110101100E30",
                    2, MPFR_RNDN);
      mpfr_clear_underflow ();
      mpfr_exp (x, x, MPFR_RNDN);
      if (!mpfr_underflow_p ())
        {
          printf ("Wrong underflow detection in mpfr_exp_3\n");
          mpfr_dump (x);
          exit (1);
        }
      mpfr_set_prec (x, 53);
    }

  /* check overflow */
  set_emax (10);
  mpfr_set_ui (x, 7, MPFR_RNDN);
  test_exp (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error for exp(7) for emax=10\n");
      exit (1);
    }
  set_emax (emax);

  /* check underflow */
  set_emin (-10);
  mpfr_set_si (x, -9, MPFR_RNDN);
  test_exp (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || mpfr_sgn (y) < 0)
    {
      printf ("Error for exp(-9) for emin=-10\n");
      printf ("Expected +0\n");
      printf ("Got      "); mpfr_print_binary (y); puts ("");
      exit (1);
    }
  set_emin (emin);

  /* check case EXP(x) < -precy */
  mpfr_set_prec (y, 2);
  mpfr_set_str_binary (x, "-0.1E-3");
  test_exp (y, x, MPFR_RNDD);
  if (mpfr_cmp_ui_2exp (y, 3, -2))
    {
      printf ("Error for exp(-1/16), prec=2, RNDD\n");
      printf ("expected 0.11, got ");
      mpfr_dump (y);
      exit (1);
    }
  test_exp (y, x, MPFR_RNDZ);
  if (mpfr_cmp_ui_2exp (y, 3, -2))
    {
      printf ("Error for exp(-1/16), prec=2, RNDZ\n");
      printf ("expected 0.11, got ");
      mpfr_dump (y);
      exit (1);
    }
  mpfr_set_str_binary (x, "0.1E-3");
  test_exp (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 1))
    {
      printf ("Error for exp(1/16), prec=2, RNDN\n");
      exit (1);
    }
  test_exp (y, x, MPFR_RNDU);
  if (mpfr_cmp_ui_2exp (y, 3, -1))
    {
      printf ("Error for exp(1/16), prec=2, RNDU\n");
      exit (1);
    }

  /* bug reported by Franky Backeljauw, 28 Mar 2003 */
  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_str_binary (x, "1.1101011000111101011110000111010010101001101001110111e28");
  test_exp (y, x, MPFR_RNDN);

  mpfr_set_prec (x, 153);
  mpfr_set_prec (z, 153);
  mpfr_set_str_binary (x, "1.1101011000111101011110000111010010101001101001110111e28");
  test_exp (z, x, MPFR_RNDN);
  mpfr_prec_round (z, 53, MPFR_RNDN);

  if (mpfr_cmp (y, z))
    {
      printf ("Error in mpfr_exp for large argument\n");
      exit (1);
    }

  /* corner cases in mpfr_exp_3 */
  mpfr_set_prec (x, 2);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_prec (y, 2);
  mpfr_exp_3 (y, x, MPFR_RNDN);

  /* Check some little things about overflow detection */
  set_emin (-125);
  set_emax (128);
  mpfr_set_prec (x, 107);
  mpfr_set_prec (y, 107);
  mpfr_set_str_binary (x, "0.11110000000000000000000000000000000000000000000"
                       "0000000000000000000000000000000000000000000000000000"
                       "00000000E4");
  test_exp (y, x, MPFR_RNDN);
  if (mpfr_cmp_str (y, "0.11000111100001100110010101111101011010010101010000"
                    "1101110111100010111001011111111000110111001011001101010"
                    "01E22", 2, MPFR_RNDN))
    {
      printf ("Special overflow error (1)\n");
      mpfr_dump (y);
      exit (1);
    }

  set_emin (emin);
  set_emax (emax);

  /* Check for overflow producing a segfault with HUGE exponent */
  mpfr_set_ui  (x, 3, MPFR_RNDN);
  mpfr_mul_2ui (x, x, 32, MPFR_RNDN);
  test_exp (y, x, MPFR_RNDN); /* Can't test return value: May overflow or not*/

  /* Bug due to wrong approximation of (x)/log2 */
  mpfr_set_prec (x, 163);

  mpfr_set_str (x, "-4.28ac8fceeadcda06bb56359017b1c81b85b392e7", 16,
                MPFR_RNDN);
  mpfr_exp (x, x, MPFR_RNDN);
  if (mpfr_cmp_str (x, "3.fffffffffffffffffffffffffffffffffffffffe8@-2",
                    16, MPFR_RNDN))
    {
      printf ("Error for x= -4.28ac8fceeadcda06bb56359017b1c81b85b392e7");
      printf ("expected  3.fffffffffffffffffffffffffffffffffffffffe8@-2");
      printf ("Got       ");
      mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
    }

  /* bug found by Guillaume Melquiond, 13 Sep 2005 */
  mpfr_set_prec (x, 53);
  mpfr_set_str_binary (x, "-1E-400");
  mpfr_exp (x, x, MPFR_RNDZ);
  if (mpfr_cmp_ui (x, 1) == 0)
    {
      printf ("Error for exp(-2^(-400))\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

/* check sign of inexact flag */
static void
check_inexact (void)
{
  mpfr_t x, y;
  int inexact;

  mpfr_init2 (x, 53);
  mpfr_init2 (y, 53);

  mpfr_set_str_binary (x,
        "1.0000000000001001000110100100101000001101101011100101e2");
  inexact = test_exp (y, x, MPFR_RNDN);
  if (inexact <= 0)
    {
      printf ("Wrong inexact flag (Got %d instead of 1)\n", inexact);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
check_exp10(void)
{
  mpfr_t x;
  int inexact;

  mpfr_init2 (x, 200);
  mpfr_set_ui(x, 4, MPFR_RNDN);

  inexact = mpfr_exp10 (x, x, MPFR_RNDN);
  if (mpfr_cmp_ui(x, 10*10*10*10))
    {
      printf ("exp10: Wrong returned value\n");
      exit (1);
    }
  if (inexact != 0)
    {
      printf ("exp10: Wrong inexact flag\n");
      exit (1);
    }

  mpfr_clear (x);
}

static void
overflowed_exp0 (void)
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
          inex = mpfr_exp (x, x, (mpfr_rnd_t) rnd);
          if ((i >= 0 || emax < 0 || rnd == MPFR_RNDN || rnd == MPFR_RNDU) &&
              ! mpfr_overflow_p ())
            {
              printf ("Error in overflowed_exp0 (i = %d, rnd = %s):\n"
                      "  The overflow flag is not set.\n",
                      i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
              err = 1;
            }
          if (rnd == MPFR_RNDZ || rnd == MPFR_RNDD)
            {
              if (inex >= 0)
                {
                  printf ("Error in overflowed_exp0 (i = %d, rnd = %s):\n"
                          "  The inexact value must be negative.\n",
                          i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  err = 1;
                }
              if (! mpfr_equal_p (x, y))
                {
                  printf ("Error in overflowed_exp0 (i = %d, rnd = %s):\n"
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
                  printf ("Error in overflowed_exp0 (i = %d, rnd = %s):\n"
                          "  The inexact value must be positive.\n",
                          i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  err = 1;
                }
              if (! (mpfr_inf_p (x) && MPFR_SIGN (x) > 0))
                {
                  printf ("Error in overflowed_exp0 (i = %d, rnd = %s):\n"
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

/* This bug occurs in mpfr_exp_2 on a Linux-64 machine, r5475. */
static void
bug20080731 (void)
{
  mpfr_exp_t emin;
  mpfr_t x, y1, y2;
  mpfr_prec_t prec = 64;

  emin = mpfr_get_emin ();
  set_emin (MPFR_EMIN_MIN);

  mpfr_init2 (x, 200);
  mpfr_set_str (x, "-2.c5c85fdf473de6af278ece700fcbdabd03cd0cb9ca62d8b62c@7",
                16, MPFR_RNDN);

  mpfr_init2 (y1, prec);
  mpfr_exp (y1, x, MPFR_RNDU);

  /* Compute the result with a higher internal precision. */
  mpfr_init2 (y2, 300);
  mpfr_exp (y2, x, MPFR_RNDU);
  mpfr_prec_round (y2, prec, MPFR_RNDU);

  if (mpfr_cmp0 (y1, y2) != 0)
    {
      printf ("Error in bug20080731\nExpected ");
      mpfr_out_str (stdout, 16, 0, y2, MPFR_RNDN);
      printf ("\nGot      ");
      mpfr_out_str (stdout, 16, 0, y1, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_clears (x, y1, y2, (mpfr_ptr) 0);
  set_emin (emin);
}

/* Emulate mpfr_exp with mpfr_exp_3 in the general case. */
static int
exp_3 (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  int inexact;

  inexact = mpfr_exp_3 (y, x, rnd_mode);
  return mpfr_check_range (y, inexact, rnd_mode);
}

static void
underflow_up (int extended_emin)
{
  mpfr_t minpos, x, y, t, t2;
  int precx, precy;
  int inex;
  int rnd;
  int e3;
  int i, j;

  mpfr_init2 (minpos, 2);
  mpfr_set_ui (minpos, 0, MPFR_RNDN);
  mpfr_nextabove (minpos);

  /* Let's test values near the underflow boundary.
   *
   * Minimum representable positive number: minpos = 2^(emin - 1).
   * Let's choose an MPFR number x = log(minpos) + eps, with |eps| small
   * (note: eps cannot be 0, and cannot be a rational number either).
   * Then exp(x) = minpos * exp(eps) ~= minpos * (1 + eps + eps^2).
   * We will compute y = rnd(exp(x)) in some rounding mode, precision p.
   *   1. If eps > 0, then in any rounding mode:
   *        rnd(exp(x)) >= minpos and no underflow.
   *      So, let's take x1 = rndu(log(minpos)) in some precision.
   *   2. If eps < 0, then exp(x) < minpos and the result will be either 0
   *      or minpos. An underflow always occurs in MPFR_RNDZ and MPFR_RNDD,
   *      but not necessarily in MPFR_RNDN and MPFR_RNDU (this is underflow
   *      after rounding in an unbounded exponent range). If -a < eps < -b,
   *        minpos * (1 - a) < exp(x) < minpos * (1 - b + b^2).
   *      - If eps > -2^(-p), no underflow in MPFR_RNDU.
   *      - If eps > -2^(-p-1), no underflow in MPFR_RNDN.
   *      - If eps < - (2^(-p-1) + 2^(-2p-1)), underflow in MPFR_RNDN.
   *      - If eps < - (2^(-p) + 2^(-2p+1)), underflow in MPFR_RNDU.
   *      - In MPFR_RNDN, result is minpos iff exp(eps) > 1/2, i.e.
   *        - log(2) < eps < ...
   *
   * Moreover, since precy < MPFR_EXP_THRESHOLD (to avoid tests that take
   * too much time), mpfr_exp() always selects mpfr_exp_2(); so, we need
   * to test mpfr_exp_3() too. This will be done via the e3 variable:
   *   e3 = 0: mpfr_exp(), thus mpfr_exp_2().
   *   e3 = 1: mpfr_exp_3(), via the exp_3() wrapper.
   * i.e.: inex = e3 ? exp_3 (y, x, rnd) : mpfr_exp (y, x, rnd);
   */

  /* Case eps > 0. In revision 5461 (trunk) on a 64-bit Linux machine:
   *   Incorrect flags in underflow_up, eps > 0, MPFR_RNDN and extended emin
   *   for precx = 96, precy = 16, mpfr_exp_3
   *   Got 9 instead of 8.
   * Note: testing this case in several precisions for x and y introduces
   * some useful random. Indeed, the bug is not always triggered.
   * Fixed in r5469.
   */
  for (precx = 16; precx <= 128; precx += 16)
    {
      mpfr_init2 (x, precx);
      mpfr_log (x, minpos, MPFR_RNDU);
      for (precy = 16; precy <= 128; precy += 16)
        {
          mpfr_init2 (y, precy);

          for (e3 = 0; e3 <= 1; e3++)
            {
              RND_LOOP (rnd)
                {
                  int err = 0;

                  mpfr_clear_flags ();
                  inex = e3 ? exp_3 (y, x, (mpfr_rnd_t) rnd)
                    : mpfr_exp (y, x, (mpfr_rnd_t) rnd);
                  if (__gmpfr_flags != MPFR_FLAGS_INEXACT)
                    {
                      printf ("Incorrect flags in underflow_up, eps > 0, %s",
                              mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                      if (extended_emin)
                        printf (" and extended emin");
                      printf ("\nfor precx = %d, precy = %d, %s\n",
                              precx, precy, e3 ? "mpfr_exp_3" : "mpfr_exp");
                      printf ("Got %u instead of %u.\n", __gmpfr_flags,
                              (unsigned int) MPFR_FLAGS_INEXACT);
                      err = 1;
                    }
                  if (mpfr_cmp0 (y, minpos) < 0)
                    {
                      printf ("Incorrect result in underflow_up, eps > 0, %s",
                              mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                      if (extended_emin)
                        printf (" and extended emin");
                      printf ("\nfor precx = %d, precy = %d, %s\n",
                              precx, precy, e3 ? "mpfr_exp_3" : "mpfr_exp");
                      mpfr_dump (y);
                      err = 1;
                    }
                  MPFR_ASSERTN (inex != 0);
                  if (rnd == MPFR_RNDD || rnd == MPFR_RNDZ)
                    MPFR_ASSERTN (inex < 0);
                  if (rnd == MPFR_RNDU)
                    MPFR_ASSERTN (inex > 0);
                  if (err)
                    exit (1);
                }
            }

          mpfr_clear (y);
        }
      mpfr_clear (x);
    }

  /* Case - log(2) < eps < 0 in MPFR_RNDN, starting with small-precision x;
   * only check the result and the ternary value.
   * Previous to r5453 (trunk), on 32-bit and 64-bit machines, this fails
   * for precx = 65 and precy = 16, e.g.:
   *   exp_2.c:264:  assertion failed: ...
   * because mpfr_sub (r, x, r, MPFR_RNDU); yields a null value. This is
   * fixed in r5453 by going to next Ziv's iteration.
   */
  for (precx = sizeof(mpfr_exp_t) * CHAR_BIT + 1; precx <= 81; precx += 8)
    {
      mpfr_init2 (x, precx);
      mpfr_log (x, minpos, MPFR_RNDD);  /* |ulp| <= 1/2 */
      for (precy = 16; precy <= 128; precy += 16)
        {
          mpfr_init2 (y, precy);
          inex = mpfr_exp (y, x, MPFR_RNDN);
          if (inex <= 0 || mpfr_cmp0 (y, minpos) != 0)
            {
              printf ("Error in underflow_up, - log(2) < eps < 0");
              if (extended_emin)
                printf (" and extended emin");
              printf (" for prec = %d\nExpected ", precy);
              mpfr_out_str (stdout, 16, 0, minpos, MPFR_RNDN);
              printf (" (minimum positive MPFR number) and inex > 0\nGot ");
              mpfr_out_str (stdout, 16, 0, y, MPFR_RNDN);
              printf ("\nwith inex = %d\n", inex);
              exit (1);
            }
          mpfr_clear (y);
        }
      mpfr_clear (x);
    }

  /* Cases eps ~ -2^(-p) and eps ~ -2^(-p-1). More precisely,
   *   _ for j = 0, eps > -2^(-(p+i)),
   *   _ for j = 1, eps < - (2^(-(p+i)) + 2^(1-2(p+i))),
   * where i = 0 or 1.
   */
  mpfr_inits2 (2, t, t2, (mpfr_ptr) 0);
  for (precy = 16; precy <= 128; precy += 16)
    {
      mpfr_set_ui_2exp (t, 1, - precy, MPFR_RNDN);         /* 2^(-p) */
      mpfr_set_ui_2exp (t2, 1, 1 - 2 * precy, MPFR_RNDN);  /* 2^(-2p+1) */
      precx = sizeof(mpfr_exp_t) * CHAR_BIT + 2 * precy + 8;
      mpfr_init2 (x, precx);
      mpfr_init2 (y, precy);
      for (i = 0; i <= 1; i++)
        {
          for (j = 0; j <= 1; j++)
            {
              if (j == 0)
                {
                  /* Case eps > -2^(-(p+i)). */
                  mpfr_log (x, minpos, MPFR_RNDU);
                }
              else  /* j == 1 */
                {
                  /* Case eps < - (2^(-(p+i)) + 2^(1-2(p+i))). */
                  mpfr_log (x, minpos, MPFR_RNDD);
                  inex = mpfr_sub (x, x, t2, MPFR_RNDN);
                  MPFR_ASSERTN (inex == 0);
                }
              inex = mpfr_sub (x, x, t, MPFR_RNDN);
              MPFR_ASSERTN (inex == 0);

              RND_LOOP (rnd)
                for (e3 = 0; e3 <= 1; e3++)
                  {
                    int err = 0;
                    unsigned int flags;

                    flags = MPFR_FLAGS_INEXACT |
                      (((rnd == MPFR_RNDU || rnd == MPFR_RNDA)
                             && (i == 1 || j == 0)) ||
                       (rnd == MPFR_RNDN && (i == 1 && j == 0)) ?
                       0 : MPFR_FLAGS_UNDERFLOW);
                    mpfr_clear_flags ();
                    inex = e3 ? exp_3 (y, x, (mpfr_rnd_t) rnd)
                      : mpfr_exp (y, x, (mpfr_rnd_t) rnd);
                    if (__gmpfr_flags != flags)
                      {
                        printf ("Incorrect flags in underflow_up, %s",
                                mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                        if (extended_emin)
                          printf (" and extended emin");
                        printf ("\nfor precx = %d, precy = %d, ",
                                precx, precy);
                        if (j == 0)
                          printf ("eps >~ -2^(-%d)", precy + i);
                        else
                          printf ("eps <~ - (2^(-%d) + 2^(%d))",
                                  precy + i, 1 - 2 * (precy + i));
                        printf (", %s\n", e3 ? "mpfr_exp_3" : "mpfr_exp");
                        printf ("Got %u instead of %u.\n",
                                __gmpfr_flags, flags);
                        err = 1;
                      }
                    if (rnd == MPFR_RNDU || rnd == MPFR_RNDA || rnd == MPFR_RNDN ?
                        mpfr_cmp0 (y, minpos) != 0 : MPFR_NOTZERO (y))
                      {
                        printf ("Incorrect result in underflow_up, %s",
                                mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                        if (extended_emin)
                          printf (" and extended emin");
                        printf ("\nfor precx = %d, precy = %d, ",
                                precx, precy);
                        if (j == 0)
                          printf ("eps >~ -2^(-%d)", precy + i);
                        else
                          printf ("eps <~ - (2^(-%d) + 2^(%d))",
                                  precy + i, 1 - 2 * (precy + i));
                        printf (", %s\n", e3 ? "mpfr_exp_3" : "mpfr_exp");
                        mpfr_dump (y);
                        err = 1;
                      }
                    if (err)
                      exit (1);
                  }  /* for (e3 ...) */
            }  /* for (j ...) */
          mpfr_div_2si (t, t, 1, MPFR_RNDN);
          mpfr_div_2si (t2, t2, 2, MPFR_RNDN);
        }  /* for (i ...) */
      mpfr_clears (x, y, (mpfr_ptr) 0);
    }  /* for (precy ...) */
  mpfr_clears (t, t2, (mpfr_ptr) 0);

  /* Case exp(eps) ~= 1/2, i.e. eps ~= - log(2).
   * We choose x0 and x1 with high enough precision such that:
   *   x0 = rndd(rndd(log(minpos)) - rndu(log(2)))
   *   x1 = rndu(rndu(log(minpos)) - rndd(log(2)))
   * In revision 5507 (trunk) on a 64-bit Linux machine, this fails:
   *   Error in underflow_up, eps >~ - log(2) and extended emin
   *   for precy = 16, mpfr_exp
   *   Expected 1.0@-1152921504606846976 (minimum positive MPFR number),
   *   inex > 0 and flags = 9
   *   Got 0
   *   with inex = -1 and flags = 9
   * due to a double-rounding problem in mpfr_mul_2si when rescaling
   * the result.
   */
  mpfr_inits2 (sizeof(mpfr_exp_t) * CHAR_BIT + 64, x, t, (mpfr_ptr) 0);
  for (i = 0; i <= 1; i++)
    {
      mpfr_log (x, minpos, i ? MPFR_RNDU : MPFR_RNDD);
      mpfr_const_log2 (t, i ? MPFR_RNDD : MPFR_RNDU);
      mpfr_sub (x, x, t, i ? MPFR_RNDU : MPFR_RNDD);
      for (precy = 16; precy <= 128; precy += 16)
        {
          mpfr_init2 (y, precy);
          for (e3 = 0; e3 <= 1; e3++)
            {
              unsigned int flags, uflags =
                MPFR_FLAGS_INEXACT | MPFR_FLAGS_UNDERFLOW;

              mpfr_clear_flags ();
              inex = e3 ? exp_3 (y, x, MPFR_RNDN) : mpfr_exp (y, x, MPFR_RNDN);
              flags = __gmpfr_flags;
              if (flags != uflags ||
                  (i ? (inex <= 0 || mpfr_cmp0 (y, minpos) != 0)
                     : (inex >= 0 || MPFR_NOTZERO (y))))
                {
                  printf ("Error in underflow_up, eps %c~ - log(2)",
                          i ? '>' : '<');
                  if (extended_emin)
                    printf (" and extended emin");
                  printf ("\nfor precy = %d, %s\nExpected ", precy,
                          e3 ? "mpfr_exp_3" : "mpfr_exp");
                  if (i)
                    {
                      mpfr_out_str (stdout, 16, 0, minpos, MPFR_RNDN);
                      printf (" (minimum positive MPFR number),\ninex >");
                    }
                  else
                    {
                      printf ("+0, inex <");
                    }
                  printf (" 0 and flags = %u\nGot ", uflags);
                  mpfr_out_str (stdout, 16, 0, y, MPFR_RNDN);
                  printf ("\nwith inex = %d and flags = %u\n", inex, flags);
                  exit (1);
                }
            }
          mpfr_clear (y);
        }
    }
  mpfr_clears (x, t, (mpfr_ptr) 0);

  mpfr_clear (minpos);
}

static void
underflow (void)
{
  mpfr_exp_t emin;

  underflow_up (0);

  emin = mpfr_get_emin ();
  set_emin (MPFR_EMIN_MIN);
  if (mpfr_get_emin () != emin)
    {
      underflow_up (1);
      set_emin (emin);
    }
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  if (argc > 1)
    check_large ();

  check_inexact ();
  check_special ();

  test_generic (2, 100, 100);

  compare_exp2_exp3 (20, 1000);
  check_worst_cases();
  check3("0.0", MPFR_RNDU, "1.0");
  check3("-1e-170", MPFR_RNDU, "1.0");
  check3("-1e-170", MPFR_RNDN, "1.0");
  check3("-8.88024741073346941839e-17", MPFR_RNDU, "1.0");
  check3("8.70772839244701057915e-01", MPFR_RNDN, "2.38875626491680437269");
  check3("1.0", MPFR_RNDN, "2.71828182845904509080");
  check3("-3.42135637628104173534e-07", MPFR_RNDZ, "0.999999657864420798958");
  /* worst case for argument reduction, very near from 5*log(2),
     thanks to Jean-Michel Muller  */
  check3("3.4657359027997265421", MPFR_RNDN, "32.0");
  check3("3.4657359027997265421", MPFR_RNDU, "32.0");
  check3("3.4657359027997265421", MPFR_RNDD, "31.999999999999996447");
  check3("2.26523754332090625496e+01", MPFR_RNDD, "6.8833785261699581146e9");
  check3("1.31478962104089092122e+01", MPFR_RNDZ, "5.12930793917860137299e+05");
  check3("4.25637507920002378103e-01", MPFR_RNDU, "1.53056585656161181497e+00");
  check3("6.26551618962329307459e-16", MPFR_RNDU, "1.00000000000000066613e+00");
  check3("-3.35589513871216568383e-03",MPFR_RNDD, "9.96649729583626853291e-01");
  check3("1.95151388850007272424e+01", MPFR_RNDU, "2.98756340674767792225e+08");
  check3("2.45045953503350730784e+01", MPFR_RNDN, "4.38743344916128387451e+10");
  check3("2.58165606081678085104e+01", MPFR_RNDD, "1.62925781879432281494e+11");
  check3("-2.36539020084338638128e+01",MPFR_RNDZ, "5.33630792749924762447e-11");
  check3("2.39211946135858077866e+01", MPFR_RNDU, "2.44817704330214385986e+10");
  check3("-2.78190533055889162029e+01",MPFR_RNDZ, "8.2858803483596879512e-13");
  check3("2.64028186174889789584e+01", MPFR_RNDD, "2.9281844652878973388e11");
  check3("2.92086338843268329413e+01", MPFR_RNDZ, "4.8433797301907177734e12");
  check3("-2.46355324071459982349e+01",MPFR_RNDZ, "1.9995129297760994791e-11");
  check3("-2.23509444608605427618e+01",MPFR_RNDZ, "1.9638492867489702307e-10");
  check3("-2.41175390197331687148e+01",MPFR_RNDD, "3.3564940885530624592e-11");
  check3("2.46363885231578088053e+01", MPFR_RNDU, "5.0055014282693267822e10");
  check3("111.1263531080090984914932050742208957672119140625", MPFR_RNDN, "1.8262572323517295459e48");
  check3("-3.56196340354684821250e+02",MPFR_RNDN, "2.0225297096141478156e-155");
  check3("6.59678273772710895173e+02", MPFR_RNDU, "3.1234469273830195529e286");
  check3("5.13772529701934331570e+02", MPFR_RNDD, "1.3445427121297197752e223");
  check3("3.57430211008718345056e+02", MPFR_RNDD, "1.6981197246857298443e155");
  check3("3.82001814471465536371e+02", MPFR_RNDU, "7.9667300591087367805e165");
  check3("5.92396038219384422518e+02", MPFR_RNDD, "1.880747529554661989e257");
  check3("-5.02678550462488090034e+02",MPFR_RNDU, "4.8919201895446217839e-219");
  check3("5.30015757134837031117e+02", MPFR_RNDD, "1.5237672861171573939e230");
  check3("5.16239362447650933063e+02", MPFR_RNDZ, "1.5845518406744492105e224");
  check3("6.00812634798592370977e-01", MPFR_RNDN, "1.823600119339019443");
  check_exp10 ();

  bug20080731 ();

  overflowed_exp0 ();
  underflow ();

  data_check ("data/exp", mpfr_exp, "mpfr_exp");
  bad_cases (mpfr_exp, mpfr_log, "mpfr_exp", 0, -256, 255, 4, 128, 800, 50);

  tests_end_mpfr ();
  return 0;
}
