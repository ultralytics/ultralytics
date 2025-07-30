/* Test file for mpfr_pow, mpfr_pow_ui and mpfr_pow_si.

Copyright 2000-2017 Free Software Foundation, Inc.
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
#include <math.h>
#include <limits.h>

#include "mpfr-test.h"

#ifdef CHECK_EXTERNAL
static int
test_pow (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mpfr_rnd_t rnd_mode)
{
  int res;
  int ok = rnd_mode == MPFR_RNDN && mpfr_number_p (b) && mpfr_number_p (c)
    && mpfr_get_prec (a) >= 53;
  if (ok)
    {
      mpfr_print_raw (b);
      printf (" ");
      mpfr_print_raw (c);
    }
  res = mpfr_pow (a, b, c, rnd_mode);
  if (ok)
    {
      printf (" ");
      mpfr_print_raw (a);
      printf ("\n");
    }
  return res;
}
#else
#define test_pow mpfr_pow
#endif

#define TEST_FUNCTION test_pow
#define TWO_ARGS
#define TEST_RANDOM_POS 16
#define TGENERIC_NOWARNING 1
#include "tgeneric.c"

#define TEST_FUNCTION mpfr_pow_ui
#define INTEGER_TYPE  unsigned long
#define RAND_FUNCTION(x) mpfr_random2(x, MPFR_LIMB_SIZE (x), 1, RANDS)
#include "tgeneric_ui.c"

#define TEST_FUNCTION mpfr_pow_si
#define INTEGER_TYPE  long
#define RAND_FUNCTION(x) mpfr_random2(x, MPFR_LIMB_SIZE (x), 1, RANDS)
#define test_generic_ui test_generic_si
#include "tgeneric_ui.c"

static void
check_pow_ui (void)
{
  mpfr_t a, b;
  unsigned long n;
  int res;

  mpfr_init2 (a, 53);
  mpfr_init2 (b, 53);

  /* check in-place operations */
  mpfr_set_str (b, "0.6926773", 10, MPFR_RNDN);
  mpfr_pow_ui (a, b, 10, MPFR_RNDN);
  mpfr_pow_ui (b, b, 10, MPFR_RNDN);
  if (mpfr_cmp (a, b))
    {
      printf ("Error for mpfr_pow_ui (b, b, ...)\n");
      exit (1);
    }

  /* check large exponents */
  mpfr_set_ui (b, 1, MPFR_RNDN);
  mpfr_pow_ui (a, b, 4294967295UL, MPFR_RNDN);

  mpfr_set_inf (a, -1);
  mpfr_pow_ui (a, a, 4049053855UL, MPFR_RNDN);
  if (!mpfr_inf_p (a) || (mpfr_sgn (a) >= 0))
    {
      printf ("Error for (-Inf)^4049053855\n");
      exit (1);
    }

  mpfr_set_inf (a, -1);
  mpfr_pow_ui (a, a, (unsigned long) 30002752, MPFR_RNDN);
  if (!mpfr_inf_p (a) || (mpfr_sgn (a) <= 0))
    {
      printf ("Error for (-Inf)^30002752\n");
      exit (1);
    }

  /* Check underflow */
  mpfr_set_str_binary (a, "1E-1");
  res = mpfr_pow_ui (a, a, -mpfr_get_emin (), MPFR_RNDN);
  if (MPFR_GET_EXP (a) != mpfr_get_emin () + 1)
    {
      printf ("Error for (1e-1)^MPFR_EMAX_MAX\n");
      mpfr_dump (a);
      exit (1);
    }

  mpfr_set_str_binary (a, "1E-10");
  res = mpfr_pow_ui (a, a, -mpfr_get_emin (), MPFR_RNDZ);
  if (!MPFR_IS_ZERO (a))
    {
      printf ("Error for (1e-10)^MPFR_EMAX_MAX\n");
      mpfr_dump (a);
      exit (1);
    }

  /* Check overflow */
  mpfr_set_str_binary (a, "1E10");
  res = mpfr_pow_ui (a, a, ULONG_MAX, MPFR_RNDN);
  if (!MPFR_IS_INF (a) || MPFR_SIGN (a) < 0)
    {
      printf ("Error for (1e10)^ULONG_MAX\n");
      exit (1);
    }

  /* Bug in pow_ui.c from r3214 to r5107: if x = y (same mpfr_t argument),
     the input argument is negative, n is odd, an overflow or underflow
     occurs, and the temporary result res is positive, then the result
     gets a wrong sign (positive instead of negative). */
  mpfr_set_str_binary (a, "-1E10");
  n = (ULONG_MAX ^ (ULONG_MAX >> 1)) + 1;
  res = mpfr_pow_ui (a, a, n, MPFR_RNDN);
  if (!MPFR_IS_INF (a) || MPFR_SIGN (a) > 0)
    {
      printf ("Error for (-1e10)^%lu, expected -Inf,\ngot ", n);
      mpfr_dump (a);
      exit (1);
    }

  /* Check 0 */
  MPFR_SET_ZERO (a);
  MPFR_SET_POS (a);
  mpfr_set_si (b, -1, MPFR_RNDN);
  res = mpfr_pow_ui (b, a, 1, MPFR_RNDN);
  if (res != 0 || MPFR_IS_NEG (b))
    {
      printf ("Error for (0+)^1\n");
      exit (1);
    }
  MPFR_SET_ZERO (a);
  MPFR_SET_NEG (a);
  mpfr_set_ui (b, 1, MPFR_RNDN);
  res = mpfr_pow_ui (b, a, 5, MPFR_RNDN);
  if (res != 0 || MPFR_IS_POS (b))
    {
      printf ("Error for (0-)^5\n");
      exit (1);
    }
  MPFR_SET_ZERO (a);
  MPFR_SET_NEG (a);
  mpfr_set_si (b, -1, MPFR_RNDN);
  res = mpfr_pow_ui (b, a, 6, MPFR_RNDN);
  if (res != 0 || MPFR_IS_NEG (b))
    {
      printf ("Error for (0-)^6\n");
      exit (1);
    }

  mpfr_set_prec (a, 122);
  mpfr_set_prec (b, 122);
  mpfr_set_str_binary (a, "0.10000010010000111101001110100101101010011110011100001111000001001101000110011001001001001011001011010110110110101000111011E1");
  mpfr_set_str_binary (b, "0.11111111100101001001000001000001100011100000001110111111100011111000111011100111111111110100011000111011000100100011001011E51290375");
  mpfr_pow_ui (a, a, 2026876995UL, MPFR_RNDU);
  if (mpfr_cmp (a, b) != 0)
    {
      printf ("Error for x^2026876995\n");
      exit (1);
    }

  mpfr_set_prec (a, 29);
  mpfr_set_prec (b, 29);
  mpfr_set_str_binary (a, "1.0000000000000000000000001111");
  mpfr_set_str_binary (b, "1.1001101111001100111001010111e165");
  mpfr_pow_ui (a, a, 2055225053, MPFR_RNDZ);
  if (mpfr_cmp (a, b) != 0)
    {
      printf ("Error for x^2055225053\n");
      printf ("Expected ");
      mpfr_out_str (stdout, 2, 0, b, MPFR_RNDN);
      printf ("\nGot      ");
      mpfr_out_str (stdout, 2, 0, a, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  /* worst case found by Vincent Lefevre, 25 Nov 2006 */
  mpfr_set_prec (a, 53);
  mpfr_set_prec (b, 53);
  mpfr_set_str_binary (a, "1.0000010110000100001000101101101001011101101011010111");
  mpfr_set_str_binary (b, "1.0000110111101111011010110100001100010000001010110100E1");
  mpfr_pow_ui (a, a, 35, MPFR_RNDN);
  if (mpfr_cmp (a, b) != 0)
    {
      printf ("Error in mpfr_pow_ui for worst case (1)\n");
      printf ("Expected ");
      mpfr_out_str (stdout, 2, 0, b, MPFR_RNDN);
      printf ("\nGot      ");
      mpfr_out_str (stdout, 2, 0, a, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  /* worst cases found on 2006-11-26 */
  mpfr_set_str_binary (a, "1.0110100111010001101001010111001110010100111111000011");
  mpfr_set_str_binary (b, "1.1111010011101110001111010110000101110000110110101100E17");
  mpfr_pow_ui (a, a, 36, MPFR_RNDD);
  if (mpfr_cmp (a, b) != 0)
    {
      printf ("Error in mpfr_pow_ui for worst case (2)\n");
      printf ("Expected ");
      mpfr_out_str (stdout, 2, 0, b, MPFR_RNDN);
      printf ("\nGot      ");
      mpfr_out_str (stdout, 2, 0, a, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  mpfr_set_str_binary (a, "1.1001010100001110000110111111100011011101110011000100");
  mpfr_set_str_binary (b, "1.1100011101101101100010110001000001110001111110010001E23");
  mpfr_pow_ui (a, a, 36, MPFR_RNDU);
  if (mpfr_cmp (a, b) != 0)
    {
      printf ("Error in mpfr_pow_ui for worst case (3)\n");
      printf ("Expected ");
      mpfr_out_str (stdout, 2, 0, b, MPFR_RNDN);
      printf ("\nGot      ");
      mpfr_out_str (stdout, 2, 0, a, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_clear (a);
  mpfr_clear (b);
}

static void
check_pow_si (void)
{
  mpfr_t x;

  mpfr_init (x);

  mpfr_set_nan (x);
  mpfr_pow_si (x, x, -1, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (x));

  mpfr_set_inf (x, 1);
  mpfr_pow_si (x, x, -1, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_POS(x));

  mpfr_set_inf (x, -1);
  mpfr_pow_si (x, x, -1, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_NEG(x));

  mpfr_set_inf (x, -1);
  mpfr_pow_si (x, x, -2, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_POS(x));

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_pow_si (x, x, -1, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_pow_si (x, x, -1, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) < 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_pow_si (x, x, -2, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);

  mpfr_set_si (x, 2, MPFR_RNDN);
  mpfr_pow_si (x, x, LONG_MAX, MPFR_RNDN);  /* 2^LONG_MAX */
  if (LONG_MAX > mpfr_get_emax () - 1)  /* LONG_MAX + 1 > emax */
    {
      MPFR_ASSERTN (mpfr_inf_p (x));
    }
  else
    {
      MPFR_ASSERTN (mpfr_cmp_si_2exp (x, 1, (mpfr_exp_t) LONG_MAX));
    }

  mpfr_set_si (x, 2, MPFR_RNDN);
  mpfr_pow_si (x, x, LONG_MIN, MPFR_RNDN);  /* 2^LONG_MIN */
  if (LONG_MIN + 1 < mpfr_get_emin ())
    {
      MPFR_ASSERTN (mpfr_zero_p (x));
    }
  else
    {
      MPFR_ASSERTN (mpfr_cmp_si_2exp (x, 1, (mpfr_exp_t) LONG_MIN));
    }

  mpfr_set_si (x, 2, MPFR_RNDN);
  mpfr_pow_si (x, x, LONG_MIN + 1, MPFR_RNDN);  /* 2^(LONG_MIN+1) */
  if (mpfr_nan_p (x))
    {
      printf ("Error in pow_si(2, LONG_MIN+1): got NaN\n");
      exit (1);
    }
  if (LONG_MIN + 2 < mpfr_get_emin ())
    {
      MPFR_ASSERTN (mpfr_zero_p (x));
    }
  else
    {
      MPFR_ASSERTN (mpfr_cmp_si_2exp (x, 1, (mpfr_exp_t) (LONG_MIN + 1)));
    }

  mpfr_set_si_2exp (x, 1, -1, MPFR_RNDN);  /* 0.5 */
  mpfr_pow_si (x, x, LONG_MIN, MPFR_RNDN);  /* 2^(-LONG_MIN) */
  if (LONG_MIN < 1 - mpfr_get_emax ())  /* 1 - LONG_MIN > emax */
    {
      MPFR_ASSERTN (mpfr_inf_p (x));
    }
  else
    {
      MPFR_ASSERTN (mpfr_cmp_si_2exp (x, 2, (mpfr_exp_t) - (LONG_MIN + 1)));
    }

  mpfr_clear (x);
}

static void
check_special_pow_si (void)
{
  mpfr_t a, b;
  mpfr_exp_t emin;

  mpfr_init (a);
  mpfr_init (b);
  mpfr_set_str (a, "2E100000000", 10, MPFR_RNDN);
  mpfr_set_si (b, -10, MPFR_RNDN);
  test_pow (b, a, b, MPFR_RNDN);
  if (!MPFR_IS_ZERO(b))
    {
      printf("Pow(2E10000000, -10) failed\n");
      mpfr_dump (a);
      mpfr_dump (b);
      exit(1);
    }

  emin = mpfr_get_emin ();
  mpfr_set_emin (-10);
  mpfr_set_si (a, -2, MPFR_RNDN);
  mpfr_pow_si (b, a, -10000, MPFR_RNDN);
  if (!MPFR_IS_ZERO (b))
    {
      printf ("Pow_so (1, -10000) doesn't underflow if emin=-10.\n");
      mpfr_dump (a);
      mpfr_dump (b);
      exit (1);
    }
  mpfr_set_emin (emin);
  mpfr_clear (a);
  mpfr_clear (b);
}

static void
pow_si_long_min (void)
{
  mpfr_t x, y, z;
  int inex;

  mpfr_inits2 (sizeof(long) * CHAR_BIT + 32, x, y, z, (mpfr_ptr) 0);
  mpfr_set_si_2exp (x, 3, -1, MPFR_RNDN);  /* 1.5 */

  inex = mpfr_set_si (y, LONG_MIN, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);
  mpfr_nextbelow (y);
  mpfr_pow (y, x, y, MPFR_RNDD);

  inex = mpfr_set_si (z, LONG_MIN, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);
  mpfr_nextabove (z);
  mpfr_pow (z, x, z, MPFR_RNDU);

  mpfr_pow_si (x, x, LONG_MIN, MPFR_RNDN);  /* 1.5^LONG_MIN */
  if (mpfr_cmp (x, y) < 0 || mpfr_cmp (x, z) > 0)
    {
      printf ("Error in pow_si_long_min\n");
      exit (1);
    }

  mpfr_clears (x, y, z, (mpfr_ptr) 0);
}

static void
check_inexact (mpfr_prec_t p)
{
  mpfr_t x, y, z, t;
  unsigned long u;
  mpfr_prec_t q;
  int inexact, cmp;
  int rnd;

  mpfr_init2 (x, p);
  mpfr_init (y);
  mpfr_init (z);
  mpfr_init (t);
  mpfr_urandomb (x, RANDS);
  u = randlimb () % 2;
  for (q = 2; q <= p; q++)
    for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
      {
        mpfr_set_prec (y, q);
        mpfr_set_prec (z, q + 10);
        mpfr_set_prec (t, q);
        inexact = mpfr_pow_ui (y, x, u, (mpfr_rnd_t) rnd);
        cmp = mpfr_pow_ui (z, x, u, (mpfr_rnd_t) rnd);
        if (mpfr_can_round (z, q + 10, (mpfr_rnd_t) rnd, (mpfr_rnd_t) rnd, q))
          {
            cmp = mpfr_set (t, z, (mpfr_rnd_t) rnd) || cmp;
            if (mpfr_cmp (y, t))
              {
                printf ("results differ for u=%lu rnd=%s\n",
                        u, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                printf ("x="); mpfr_print_binary (x); puts ("");
                printf ("y="); mpfr_print_binary (y); puts ("");
                printf ("t="); mpfr_print_binary (t); puts ("");
                printf ("z="); mpfr_print_binary (z); puts ("");
                exit (1);
              }
            if (((inexact == 0) && (cmp != 0)) ||
                ((inexact != 0) && (cmp == 0)))
              {
                printf ("Wrong inexact flag for p=%u, q=%u, rnd=%s\n",
                        (unsigned int) p, (unsigned int) q,
                        mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                printf ("expected %d, got %d\n", cmp, inexact);
                printf ("u=%lu x=", u); mpfr_print_binary (x); puts ("");
                printf ("y="); mpfr_print_binary (y); puts ("");
                exit (1);
              }
          }
      }

  /* check exact power */
  mpfr_set_prec (x, p);
  mpfr_set_prec (y, p);
  mpfr_set_prec (z, p);
  mpfr_set_ui (x, 4, MPFR_RNDN);
  mpfr_set_str (y, "0.5", 10, MPFR_RNDN);
  test_pow (z, x, y, MPFR_RNDZ);

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (t);
}

static void
special (void)
{
  mpfr_t x, y, z, t;
  mpfr_exp_t emin, emax;
  int inex;

  mpfr_init2 (x, 53);
  mpfr_init2 (y, 53);
  mpfr_init2 (z, 53);
  mpfr_init2 (t, 2);

  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_pow_si (x, x, -2, MPFR_RNDN);
  if (mpfr_cmp_ui_2exp (x, 1, -2))
    {
      printf ("Error in pow_si(x,x,-2) for x=2\n");
      exit (1);
    }
  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_set_si (y, -2, MPFR_RNDN);
  test_pow (x, x, y, MPFR_RNDN);
  if (mpfr_cmp_ui_2exp (x, 1, -2))
    {
      printf ("Error in pow(x,x,y) for x=2, y=-2\n");
      exit (1);
    }

  mpfr_set_prec (x, 2);
  mpfr_set_str_binary (x, "1.0e-1");
  mpfr_set_prec (y, 53);
  mpfr_set_str_binary (y, "0.11010110011100101010110011001010100111000001000101110E-1");
  mpfr_set_prec (z, 2);
  test_pow (z, x, y, MPFR_RNDZ);
  mpfr_set_str_binary (x, "1.0e-1");
  if (mpfr_cmp (x, z))
    {
      printf ("Error in mpfr_pow (1)\n");
      exit (1);
    }

  mpfr_set_prec (x, 64);
  mpfr_set_prec (y, 64);
  mpfr_set_prec (z, 64);
  mpfr_set_prec (t, 64);
  mpfr_set_str_binary (x, "0.111011000111100000111010000101010100110011010000011");
  mpfr_set_str_binary (y, "0.111110010100110000011101100011010111000010000100101");
  mpfr_set_str_binary (t, "0.1110110011110110001000110100100001001111010011111000010000011001");

  test_pow (z, x, y, MPFR_RNDN);
  if (mpfr_cmp (z, t))
    {
      printf ("Error in mpfr_pow for prec=64, rnd=MPFR_RNDN\n");
      exit (1);
    }

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_prec (z, 53);
  mpfr_set_str (x, "5.68824667828621954868e-01", 10, MPFR_RNDN);
  mpfr_set_str (y, "9.03327850535952658895e-01", 10, MPFR_RNDN);
  test_pow (z, x, y, MPFR_RNDZ);
  if (mpfr_cmp_str1 (z, "0.60071044650456473235"))
    {
      printf ("Error in mpfr_pow for prec=53, rnd=MPFR_RNDZ\n");
      exit (1);
    }

  mpfr_set_prec (t, 2);
  mpfr_set_prec (x, 30);
  mpfr_set_prec (y, 30);
  mpfr_set_prec (z, 30);
  mpfr_set_str (x, "1.00000000001010111110001111011e1", 2, MPFR_RNDN);
  mpfr_set_str (t, "-0.5", 10, MPFR_RNDN);
  test_pow (z, x, t, MPFR_RNDN);
  mpfr_set_str (y, "1.01101001111010101110000101111e-1", 2, MPFR_RNDN);
  if (mpfr_cmp (z, y))
    {
      printf ("Error in mpfr_pow for prec=30, rnd=MPFR_RNDN\n");
      exit (1);
    }

  mpfr_set_prec (x, 21);
  mpfr_set_prec (y, 21);
  mpfr_set_prec (z, 21);
  mpfr_set_str (x, "1.11111100100001100101", 2, MPFR_RNDN);
  test_pow (z, x, t, MPFR_RNDZ);
  mpfr_set_str (y, "1.01101011010001100000e-1", 2, MPFR_RNDN);
  if (mpfr_cmp (z, y))
    {
      printf ("Error in mpfr_pow for prec=21, rnd=MPFR_RNDZ\n");
      printf ("Expected ");
      mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
      printf ("\nGot      ");
      mpfr_out_str (stdout, 2, 0, z, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  /* From http://www.terra.es/personal9/ismaeljc/hall.htm */
  mpfr_set_prec (x, 113);
  mpfr_set_prec (y, 2);
  mpfr_set_prec (z, 169);
  mpfr_set_str1 (x, "6078673043126084065007902175846955");
  mpfr_set_ui_2exp (y, 3, -1, MPFR_RNDN);
  test_pow (z, x, y, MPFR_RNDZ);
  if (mpfr_cmp_str1 (z, "473928882491000966028828671876527456070714790264144"))
    {
      printf ("Error in mpfr_pow for 6078673043126084065007902175846955");
      printf ("^(3/2), MPFR_RNDZ\nExpected ");
      printf ("4.73928882491000966028828671876527456070714790264144e50");
      printf ("\nGot      ");
      mpfr_out_str (stdout, 10, 0, z, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  test_pow (z, x, y, MPFR_RNDU);
  if (mpfr_cmp_str1 (z, "473928882491000966028828671876527456070714790264145"))
    {
      printf ("Error in mpfr_pow for 6078673043126084065007902175846955");
      printf ("^(3/2), MPFR_RNDU\nExpected ");
      printf ("4.73928882491000966028828671876527456070714790264145e50");
      printf ("\nGot      ");
      mpfr_out_str (stdout, 10, 0, z, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_set_prec (y, 2);
  mpfr_set_str_binary (y, "1E10");
  test_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (z) && MPFR_IS_POS(z));
  mpfr_set_inf (x, -1);
  test_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (z) && MPFR_IS_POS(z));
  mpfr_set_prec (y, 10);
  mpfr_set_str_binary (y, "1.000000001E9");
  test_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (z) && MPFR_IS_NEG(z));
  mpfr_set_str_binary (y, "1.000000001E8");
  test_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (z) && MPFR_IS_POS(z));

  mpfr_set_inf (x, -1);
  mpfr_set_prec (y, 2 * mp_bits_per_limb);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_mul_2exp (y, y, mp_bits_per_limb - 1, MPFR_RNDN);
  /* y = 2^(mp_bits_per_limb - 1) */
  test_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (z) && MPFR_IS_POS(z));
  mpfr_nextabove (y);
  test_pow (z, x, y, MPFR_RNDN);
  /* y = 2^(mp_bits_per_limb - 1) + epsilon */
  MPFR_ASSERTN(mpfr_inf_p (z) && MPFR_IS_POS(z));
  mpfr_nextbelow (y);
  mpfr_div_2exp (y, y, 1, MPFR_RNDN);
  mpfr_nextabove (y);
  test_pow (z, x, y, MPFR_RNDN);
  /* y = 2^(mp_bits_per_limb - 2) + epsilon */
  MPFR_ASSERTN(mpfr_inf_p (z) && MPFR_IS_POS(z));

  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_set_prec (y, 2);
  mpfr_set_str_binary (y, "1E10");
  test_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (z, 1) == 0);

  /* Check (-0)^(17.0001) */
  mpfr_set_prec (x, 6);
  mpfr_set_prec (y, 640);
  MPFR_SET_ZERO (x); MPFR_SET_NEG (x);
  mpfr_set_ui (y, 17, MPFR_RNDN); mpfr_nextabove (y);
  test_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO (z) && MPFR_IS_POS (z));

  /* Bugs reported by Kevin Rauch on 29 Oct 2007 */
  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  mpfr_set_emin (-1000000);
  mpfr_set_emax ( 1000000);
  mpfr_set_prec (x, 64);
  mpfr_set_prec (y, 64);
  mpfr_set_prec (z, 64);
  mpfr_set_str (x, "-0.5", 10, MPFR_RNDN);
  mpfr_set_str (y, "-0.ffffffffffffffff", 16, MPFR_RNDN);
  mpfr_set_exp (y, mpfr_get_emax ());
  inex = mpfr_pow (z, x, y, MPFR_RNDN);
  /* (-0.5)^(-n) = 1/2^n for n even */
  MPFR_ASSERTN(mpfr_inf_p (z) && MPFR_IS_POS (z) && inex > 0);

  /* (-1)^(-n) = 1 for n even */
  mpfr_set_str (x, "-1", 10, MPFR_RNDN);
  inex = mpfr_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (z, 1) == 0 && inex == 0);

  /* (-1)^n = 1 for n even */
  mpfr_set_str (x, "-1", 10, MPFR_RNDN);
  mpfr_neg (y, y, MPFR_RNDN);
  inex = mpfr_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (z, 1) == 0 && inex == 0);

  /* (-1.5)^n = +Inf for n even */
  mpfr_set_str (x, "-1.5", 10, MPFR_RNDN);
  inex = mpfr_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (z) && MPFR_IS_POS (z) && inex > 0);

  /* (-n)^1.5 = NaN for n even */
  mpfr_neg (y, y, MPFR_RNDN);
  mpfr_set_str (x, "1.5", 10, MPFR_RNDN);
  inex = mpfr_pow (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (z));

  /* x^(-1.5) = NaN for x small < 0 */
  mpfr_set_str (x, "-0.8", 16, MPFR_RNDN);
  mpfr_set_exp (x, mpfr_get_emin ());
  mpfr_set_str (y, "-1.5", 10, MPFR_RNDN);
  inex = mpfr_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (z));

  mpfr_set_emin (emin);
  mpfr_set_emax (emax);
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (t);
}

static void
particular_cases (void)
{
  mpfr_t t[11], r, r2;
  mpz_t z;
  long si;

  static const char *name[11] = {
    "NaN", "+inf", "-inf", "+0", "-0", "+1", "-1", "+2", "-2", "+0.5", "-0.5"};
  int i, j;
  int error = 0;

  mpz_init (z);

  for (i = 0; i < 11; i++)
    mpfr_init2 (t[i], 2);
  mpfr_init2 (r, 6);
  mpfr_init2 (r2, 6);

  mpfr_set_nan (t[0]);
  mpfr_set_inf (t[1], 1);
  mpfr_set_ui (t[3], 0, MPFR_RNDN);
  mpfr_set_ui (t[5], 1, MPFR_RNDN);
  mpfr_set_ui (t[7], 2, MPFR_RNDN);
  mpfr_div_2ui (t[9], t[5], 1, MPFR_RNDN);
  for (i = 1; i < 11; i += 2)
    mpfr_neg (t[i+1], t[i], MPFR_RNDN);

  for (i = 0; i < 11; i++)
    for (j = 0; j < 11; j++)
      {
        double d;
        int p;
        static const int q[11][11] = {
          /*          NaN +inf -inf  +0   -0   +1   -1   +2   -2  +0.5 -0.5 */
          /*  NaN */ { 0,   0,   0,  128, 128,  0,   0,   0,   0,   0,   0  },
          /* +inf */ { 0,   1,   2,  128, 128,  1,   2,   1,   2,   1,   2  },
          /* -inf */ { 0,   1,   2,  128, 128, -1,  -2,   1,   2,   1,   2  },
          /*  +0  */ { 0,   2,   1,  128, 128,  2,   1,   2,   1,   2,   1  },
          /*  -0  */ { 0,   2,   1,  128, 128, -2,  -1,   2,   1,   2,   1  },
          /*  +1  */ {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 },
          /*  -1  */ { 0,  128, 128, 128, 128,-128,-128, 128, 128,  0,   0  },
          /*  +2  */ { 0,   1,   2,  128, 128, 256,  64, 512,  32, 180,  90 },
          /*  -2  */ { 0,   1,   2,  128, 128,-256, -64, 512,  32,  0,   0  },
          /* +0.5 */ { 0,   2,   1,  128, 128,  64, 256,  32, 512,  90, 180 },
          /* -0.5 */ { 0,   2,   1,  128, 128, -64,-256,  32, 512,  0,   0  }
        };
        /* This define is used to make the following table readable */
#define N MPFR_FLAGS_NAN
#define I MPFR_FLAGS_INEXACT
#define D MPFR_FLAGS_DIVBY0
        static const unsigned int f[11][11] = {
          /*          NaN +inf -inf  +0 -0 +1 -1 +2 -2 +0.5 -0.5 */
          /*  NaN */ { N,   N,   N,  0,  0, N, N, N, N,  N,   N  },
          /* +inf */ { N,   0,   0,  0,  0, 0, 0, 0, 0,  0,   0  },
          /* -inf */ { N,   0,   0,  0,  0, 0, 0, 0, 0,  0,   0  },
          /*  +0  */ { N,   0,   0,  0,  0, 0, D, 0, D,  0,   D  },
          /*  -0  */ { N,   0,   0,  0,  0, 0, D, 0, D,  0,   D  },
          /*  +1  */ { 0,   0,   0,  0,  0, 0, 0, 0, 0,  0,   0  },
          /*  -1  */ { N,   0,   0,  0,  0, 0, 0, 0, 0,  N,   N  },
          /*  +2  */ { N,   0,   0,  0,  0, 0, 0, 0, 0,  I,   I  },
          /*  -2  */ { N,   0,   0,  0,  0, 0, 0, 0, 0,  N,   N  },
          /* +0.5 */ { N,   0,   0,  0,  0, 0, 0, 0, 0,  I,   I  },
          /* -0.5 */ { N,   0,   0,  0,  0, 0, 0, 0, 0,  N,   N  }
        };
#undef N
#undef I
#undef D
        mpfr_clear_flags ();
        test_pow (r, t[i], t[j], MPFR_RNDN);
        p = mpfr_nan_p (r) ? 0 : mpfr_inf_p (r) ? 1 :
          mpfr_cmp_ui (r, 0) == 0 ? 2 :
          (d = mpfr_get_d (r, MPFR_RNDN), (int) (ABS(d) * 128.0));
        if (p != 0 && MPFR_IS_NEG (r))
          p = -p;
        if (p != q[i][j])
          {
            printf ("Error in mpfr_pow for (%s)^(%s) (%d,%d):\n"
                    "got %d instead of %d\n",
                    name[i], name[j], i, j, p, q[i][j]);
            mpfr_dump (r);
            error = 1;
          }
        if (__gmpfr_flags != f[i][j])
          {
            printf ("Error in mpfr_pow for (%s)^(%s) (%d,%d):\n"
                    "Flags = %u instead of expected %u\n",
                    name[i], name[j], i, j, __gmpfr_flags, f[i][j]);
            mpfr_dump (r);
            error = 1;
          }
        /* Perform the same tests with pow_z & pow_si & pow_ui
           if t[j] is an integer */
        if (mpfr_integer_p (t[j]))
          {
            /* mpfr_pow_z */
            mpfr_clear_flags ();
            mpfr_get_z (z, t[j], MPFR_RNDN);
            mpfr_pow_z (r, t[i], z, MPFR_RNDN);
            p = mpfr_nan_p (r) ? 0 : mpfr_inf_p (r) ? 1 :
              mpfr_cmp_ui (r, 0) == 0 ? 2 :
              (d = mpfr_get_d (r, MPFR_RNDN), (int) (ABS(d) * 128.0));
            if (p != 0 && MPFR_IS_NEG (r))
              p = -p;
            if (p != q[i][j])
              {
                printf ("Error in mpfr_pow_z for (%s)^(%s) (%d,%d):\n"
                        "got %d instead of %d\n",
                        name[i], name[j], i, j, p, q[i][j]);
                mpfr_dump (r);
                error = 1;
              }
            if (__gmpfr_flags != f[i][j])
              {
                printf ("Error in mpfr_pow_z for (%s)^(%s) (%d,%d):\n"
                        "Flags = %u instead of expected %u\n",
                        name[i], name[j], i, j, __gmpfr_flags, f[i][j]);
                mpfr_dump (r);
                error = 1;
              }
            /* mpfr_pow_si */
            mpfr_clear_flags ();
            si = mpfr_get_si (t[j], MPFR_RNDN);
            mpfr_pow_si (r, t[i], si, MPFR_RNDN);
            p = mpfr_nan_p (r) ? 0 : mpfr_inf_p (r) ? 1 :
              mpfr_cmp_ui (r, 0) == 0 ? 2 :
              (d = mpfr_get_d (r, MPFR_RNDN), (int) (ABS(d) * 128.0));
            if (p != 0 && MPFR_IS_NEG (r))
              p = -p;
            if (p != q[i][j])
              {
                printf ("Error in mpfr_pow_si for (%s)^(%s) (%d,%d):\n"
                        "got %d instead of %d\n",
                        name[i], name[j], i, j, p, q[i][j]);
                mpfr_dump (r);
                error = 1;
              }
            if (__gmpfr_flags != f[i][j])
              {
                printf ("Error in mpfr_pow_si for (%s)^(%s) (%d,%d):\n"
                        "Flags = %u instead of expected %u\n",
                        name[i], name[j], i, j, __gmpfr_flags, f[i][j]);
                mpfr_dump (r);
                error = 1;
              }
            /* if si >= 0, test mpfr_pow_ui */
            if (si >= 0)
              {
                mpfr_clear_flags ();
                mpfr_pow_ui (r, t[i], si, MPFR_RNDN);
                p = mpfr_nan_p (r) ? 0 : mpfr_inf_p (r) ? 1 :
                  mpfr_cmp_ui (r, 0) == 0 ? 2 :
                  (d = mpfr_get_d (r, MPFR_RNDN), (int) (ABS(d) * 128.0));
                if (p != 0 && MPFR_IS_NEG (r))
                  p = -p;
                if (p != q[i][j])
                  {
                    printf ("Error in mpfr_pow_ui for (%s)^(%s) (%d,%d):\n"
                            "got %d instead of %d\n",
                            name[i], name[j], i, j, p, q[i][j]);
                    mpfr_dump (r);
                    error = 1;
                  }
                if (__gmpfr_flags != f[i][j])
                  {
                    printf ("Error in mpfr_pow_ui for (%s)^(%s) (%d,%d):\n"
                            "Flags = %u instead of expected %u\n",
                            name[i], name[j], i, j, __gmpfr_flags, f[i][j]);
                    mpfr_dump (r);
                    error = 1;
                  }
              }
          } /* integer_p */
        /* Perform the same tests with mpfr_ui_pow */
        if (mpfr_integer_p (t[i]) && MPFR_IS_POS (t[i]))
          {
            /* mpfr_ui_pow */
            mpfr_clear_flags ();
            si = mpfr_get_si (t[i], MPFR_RNDN);
            mpfr_ui_pow (r, si, t[j], MPFR_RNDN);
            p = mpfr_nan_p (r) ? 0 : mpfr_inf_p (r) ? 1 :
              mpfr_cmp_ui (r, 0) == 0 ? 2 :
              (d = mpfr_get_d (r, MPFR_RNDN), (int) (ABS(d) * 128.0));
            if (p != 0 && MPFR_IS_NEG (r))
              p = -p;
            if (p != q[i][j])
              {
                printf ("Error in mpfr_ui_pow for (%s)^(%s) (%d,%d):\n"
                        "got %d instead of %d\n",
                        name[i], name[j], i, j, p, q[i][j]);
                mpfr_dump (r);
                error = 1;
              }
            if (__gmpfr_flags != f[i][j])
              {
                printf ("Error in mpfr_ui_pow for (%s)^(%s) (%d,%d):\n"
                        "Flags = %u instead of expected %u\n",
                        name[i], name[j], i, j, __gmpfr_flags, f[i][j]);
                mpfr_dump (r);
                error = 1;
              }
          }
      }

  for (i = 0; i < 11; i++)
    mpfr_clear (t[i]);
  mpfr_clear (r);
  mpfr_clear (r2);
  mpz_clear (z);

  if (error)
    exit (1);
}

static void
underflows (void)
{
  mpfr_t x, y, z;
  int err = 0;
  int inexact;
  int i;
  mpfr_exp_t emin;

  mpfr_init2 (x, 64);
  mpfr_init2 (y, 64);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_exp (x, mpfr_get_emin());

  for (i = 3; i < 10; i++)
    {
      mpfr_set_ui (y, i, MPFR_RNDN);
      mpfr_div_2ui (y, y, 1, MPFR_RNDN);
      test_pow (y, x, y, MPFR_RNDN);
      if (!MPFR_IS_FP(y) || mpfr_cmp_ui (y, 0))
        {
          printf ("Error in mpfr_pow for ");
          mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
          printf (" ^ (%d/2)\nGot ", i);
          mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
          printf (" instead of 0.\n");
          exit (1);
        }
    }

  mpfr_init2 (z, 55);
  mpfr_set_str (x, "0.110011010011101001110001110100010000110111101E0",
                2, MPFR_RNDN);
  mpfr_set_str (y, "0.101110010011111001011010100011011100111110011E40",
                2, MPFR_RNDN);
  mpfr_clear_flags ();
  inexact = mpfr_pow (z, x, y, MPFR_RNDU);
  if (!mpfr_underflow_p ())
    {
      printf ("Underflow flag is not set for special underflow test.\n");
      err = 1;
    }
  if (inexact <= 0)
    {
      printf ("Ternary value is wrong for special underflow test.\n");
      err = 1;
    }
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_nextabove (x);
  if (mpfr_cmp (x, z) != 0)
    {
      printf ("Wrong value for special underflow test.\nGot ");
      mpfr_out_str (stdout, 2, 0, z, MPFR_RNDN);
      printf ("\ninstead of ");
      mpfr_out_str (stdout, 2, 2, x, MPFR_RNDN);
      printf ("\n");
      err = 1;
    }
  if (err)
    exit (1);

  /* MPFR currently (2006-08-19) segfaults on the following code (and
     possibly makes other programs crash due to the lack of memory),
     because y is converted into an mpz_t, and the required precision
     is too high. */
  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 2);
  mpfr_set_prec (z, 12);
  mpfr_set_ui_2exp (x, 3, -2, MPFR_RNDN);
  mpfr_set_ui_2exp (y, 1, mpfr_get_emax () - 1, MPFR_RNDN);
  mpfr_clear_flags ();
  mpfr_pow (z, x, y, MPFR_RNDN);
  if (!mpfr_underflow_p () || MPFR_NOTZERO (z))
    {
      printf ("Underflow test with large y fails.\n");
      exit (1);
    }

  emin = mpfr_get_emin ();
  mpfr_set_emin (-256);
  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 2);
  mpfr_set_prec (z, 12);
  mpfr_set_ui_2exp (x, 3, -2, MPFR_RNDN);
  mpfr_set_ui_2exp (y, 1, 38, MPFR_RNDN);
  mpfr_clear_flags ();
  inexact = mpfr_pow (z, x, y, MPFR_RNDN);
  if (!mpfr_underflow_p () || MPFR_NOTZERO (z) || inexact >= 0)
    {
      printf ("Bad underflow detection for 0.75^(2^38). Obtained:\n"
              "Underflow flag... %-3s (should be 'yes')\n"
              "Zero result...... %-3s (should be 'yes')\n"
              "Inexact value.... %-3d (should be negative)\n",
              mpfr_underflow_p () ? "yes" : "no",
              MPFR_IS_ZERO (z) ? "yes" : "no", inexact);
      exit (1);
    }
  mpfr_set_emin (emin);

  emin = mpfr_get_emin ();
  mpfr_set_emin (-256);
  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 40);
  mpfr_set_prec (z, 12);
  mpfr_set_ui_2exp (x, 3, -1, MPFR_RNDN);
  mpfr_set_si_2exp (y, -1, 38, MPFR_RNDN);
  for (i = 0; i < 4; i++)
    {
      if (i == 2)
        mpfr_neg (x, x, MPFR_RNDN);
      mpfr_clear_flags ();
      inexact = mpfr_pow (z, x, y, MPFR_RNDN);
      if (!mpfr_underflow_p () || MPFR_NOTZERO (z) ||
          (i == 3 ? (inexact <= 0) : (inexact >= 0)))
        {
          printf ("Bad underflow detection for (");
          mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
          printf (")^(-2^38-%d). Obtained:\n"
                  "Overflow flag.... %-3s (should be 'no')\n"
                  "Underflow flag... %-3s (should be 'yes')\n"
                  "Zero result...... %-3s (should be 'yes')\n"
                  "Inexact value.... %-3d (should be %s)\n", i,
                  mpfr_overflow_p () ? "yes" : "no",
                  mpfr_underflow_p () ? "yes" : "no",
                  MPFR_IS_ZERO (z) ? "yes" : "no", inexact,
                  i == 3 ? "positive" : "negative");
          exit (1);
        }
      inexact = mpfr_sub_ui (y, y, 1, MPFR_RNDN);
      MPFR_ASSERTN (inexact == 0);
    }
  mpfr_set_emin (emin);

  mpfr_clears (x, y, z, (mpfr_ptr) 0);
}

static void
overflows (void)
{
  mpfr_t a, b;

  /* bug found by Ming J. Tsai <mingjt@delvron.us>, 4 Oct 2003 */

  mpfr_init_set_str (a, "5.1e32", 10, MPFR_RNDN);
  mpfr_init (b);

  test_pow (b, a, a, MPFR_RNDN);
  if (!(mpfr_inf_p (b) && mpfr_sgn (b) > 0))
    {
      printf ("Error for a^a for a=5.1e32\n");
      printf ("Expected +Inf, got ");
      mpfr_out_str (stdout, 10, 0, b, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_clear(a);
  mpfr_clear(b);
}

static void
overflows2 (void)
{
  mpfr_t x, y, z;
  mpfr_exp_t emin, emax;
  int e;

  /* x^y in reduced exponent range, where x = 2^b and y is not an integer
     (so that mpfr_pow_z is not used). */

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  set_emin (-128);

  mpfr_inits2 (16, x, y, z, (mpfr_ptr) 0);

  mpfr_set_si_2exp (x, 1, -64, MPFR_RNDN);  /* 2^(-64) */
  mpfr_set_si_2exp (y, -1, -1, MPFR_RNDN);  /* -0.5 */
  for (e = 2; e <= 32; e += 17)
    {
      set_emax (e);
      mpfr_clear_flags ();
      mpfr_pow (z, x, y, MPFR_RNDN);
      if (MPFR_IS_NEG (z) || ! mpfr_inf_p (z))
        {
          printf ("Error in overflows2 (e = %d): expected +Inf, got ", e);
          mpfr_dump (z);
          exit (1);
        }
      if (__gmpfr_flags != (MPFR_FLAGS_OVERFLOW | MPFR_FLAGS_INEXACT))
        {
          printf ("Error in overflows2 (e = %d): bad flags (%u)\n",
                  e, __gmpfr_flags);
          exit (1);
        }
    }

  mpfr_clears (x, y, z, (mpfr_ptr) 0);

  set_emin (emin);
  set_emax (emax);
}

static void
overflows3 (void)
{
  /* x^y where x = 2^b, y is not an integer (so that mpfr_pow_z is not used)
     and b * y = emax in the extended exponent range. If emax is divisible
     by 3, we choose x = 2^(-2*emax/3) and y = -3/2.
     Test also with nextbelow(x). */

  if (MPFR_EMAX_MAX % 3 == 0)
    {
      mpfr_t x, y, z, t;
      mpfr_exp_t emin, emax;
      unsigned int flags;
      int i;

      emin = mpfr_get_emin ();
      emax = mpfr_get_emax ();
      set_emin (MPFR_EMIN_MIN);
      set_emax (MPFR_EMAX_MAX);

      mpfr_inits2 (16, x, y, z, t, (mpfr_ptr) 0);

      mpfr_set_si_2exp (x, 1, -2 * (MPFR_EMAX_MAX / 3), MPFR_RNDN);
      for (i = 0; i <= 1; i++)
        {
          mpfr_set_si_2exp (y, -3, -1, MPFR_RNDN);
          mpfr_clear_flags ();
          mpfr_pow (z, x, y, MPFR_RNDN);
          if (MPFR_IS_NEG (z) || ! mpfr_inf_p (z))
            {
              printf ("Error in overflows3 (RNDN, i = %d): expected +Inf,"
                      " got ", i);
              mpfr_dump (z);
              exit (1);
            }
          if (__gmpfr_flags != (MPFR_FLAGS_OVERFLOW | MPFR_FLAGS_INEXACT))
            {
              printf ("Error in overflows3 (RNDN, i = %d): bad flags (%u)\n",
                      i, __gmpfr_flags);
              exit (1);
            }

          mpfr_clear_flags ();
          mpfr_pow (z, x, y, MPFR_RNDZ);
          flags = __gmpfr_flags;
          mpfr_set (t, z, MPFR_RNDN);
          mpfr_nextabove (t);
          if (MPFR_IS_NEG (z) || mpfr_inf_p (z) || ! mpfr_inf_p (t))
            {
              printf ("Error in overflows3 (RNDZ, i = %d):\nexpected ", i);
              mpfr_set_inf (t, 1);
              mpfr_nextbelow (t);
              mpfr_dump (t);
              printf ("got      ");
              mpfr_dump (z);
              exit (1);
            }
          if (flags != (MPFR_FLAGS_OVERFLOW | MPFR_FLAGS_INEXACT))
            {
              printf ("Error in overflows3 (RNDZ, i = %d): bad flags (%u)\n",
                      i, flags);
              exit (1);
            }
          mpfr_nextbelow (x);
        }

      mpfr_clears (x, y, z, t, (mpfr_ptr) 0);

      set_emin (emin);
      set_emax (emax);
    }
}

static void
x_near_one (void)
{
  mpfr_t x, y, z;
  int inex;

  mpfr_init2 (x, 32);
  mpfr_init2 (y, 4);
  mpfr_init2 (z, 33);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_nextbelow (x);
  mpfr_set_ui_2exp (y, 11, -2, MPFR_RNDN);
  inex = mpfr_pow (z, x, y, MPFR_RNDN);
  if (mpfr_cmp_str (z, "0.111111111111111111111111111111011E0", 2, MPFR_RNDN)
      || inex <= 0)
    {
      printf ("Failure in x_near_one, got inex = %d and\nz = ", inex);
      mpfr_dump (z);
    }

  mpfr_clears (x, y, z, (mpfr_ptr) 0);
}

static int
mpfr_pow275 (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t r)
{
  mpfr_t z;
  int inex;

  mpfr_init2 (z, 4);
  mpfr_set_ui_2exp (z, 11, -2, MPFR_RNDN);
  inex = mpfr_pow (y, x, z, MPFR_RNDN);
  mpfr_clear (z);
  return inex;
}

/* Bug found by Kevin P. Rauch */
static void
bug20071103 (void)
{
  mpfr_t x, y, z;
  mpfr_exp_t emin, emax;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  mpfr_set_emin (-1000000);
  mpfr_set_emax ( 1000000);

  mpfr_inits2 (64, x, y, z, (mpfr_ptr) 0);
  mpfr_set_si_2exp (x, -3, -1, MPFR_RNDN);  /* x = -1.5 */
  mpfr_set_str (y, "-0.ffffffffffffffff", 16, MPFR_RNDN);
  mpfr_set_exp (y, mpfr_get_emax ());
  mpfr_clear_flags ();
  mpfr_pow (z, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_zero_p (z) && MPFR_SIGN (z) > 0 &&
                __gmpfr_flags == (MPFR_FLAGS_UNDERFLOW | MPFR_FLAGS_INEXACT));
  mpfr_clears (x, y, z, (mpfr_ptr) 0);

  set_emin (emin);
  set_emax (emax);
}

/* Bug found by Kevin P. Rauch */
static void
bug20071104 (void)
{
  mpfr_t x, y, z;
  mpfr_exp_t emin, emax;
  int inex;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  mpfr_set_emin (-1000000);
  mpfr_set_emax ( 1000000);

  mpfr_inits2 (20, x, y, z, (mpfr_ptr) 0);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_nextbelow (x);             /* x = -2^(emin-1) */
  mpfr_set_si (y, -2, MPFR_RNDN);  /* y = -2 */
  mpfr_clear_flags ();
  inex = mpfr_pow (z, x, y, MPFR_RNDN);
  if (! mpfr_inf_p (z) || MPFR_SIGN (z) < 0)
    {
      printf ("Error in bug20071104: expected +Inf, got ");
      mpfr_dump (z);
      exit (1);
    }
  if (inex <= 0)
    {
      printf ("Error in bug20071104: bad ternary value (%d)\n", inex);
      exit (1);
    }
  if (__gmpfr_flags != (MPFR_FLAGS_OVERFLOW | MPFR_FLAGS_INEXACT))
    {
      printf ("Error in bug20071104: bad flags (%u)\n", __gmpfr_flags);
      exit (1);
    }
  mpfr_clears (x, y, z, (mpfr_ptr) 0);

  set_emin (emin);
  set_emax (emax);
}

/* Bug found by Kevin P. Rauch */
static void
bug20071127 (void)
{
  mpfr_t x, y, z;
  int i, tern;
  mpfr_exp_t emin, emax;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  mpfr_set_emin (-1000000);
  mpfr_set_emax ( 1000000);

  mpfr_init2 (x, 128);
  mpfr_init2 (y, 128);
  mpfr_init2 (z, 128);

  mpfr_set_str (x, "0.80000000000000000000000000000001", 16, MPFR_RNDN);

  for (i = 1; i < 9; i *= 2)
    {
      mpfr_set_str (y, "8000000000000000", 16, MPFR_RNDN);
      mpfr_add_si (y, y, i, MPFR_RNDN);
      tern = mpfr_pow (z, x, y, MPFR_RNDN);
      MPFR_ASSERTN (mpfr_zero_p (z) && MPFR_IS_POS (z) && tern < 0);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);

  mpfr_set_emin (emin);
  mpfr_set_emax (emax);
}

/* Bug found by Kevin P. Rauch */
static void
bug20071128 (void)
{
  mpfr_t max_val, x, y, z;
  int i, tern;
  mpfr_exp_t emin, emax;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  mpfr_set_emin (-1000000);
  mpfr_set_emax ( 1000000);

  mpfr_init2 (max_val, 64);
  mpfr_init2 (x, 64);
  mpfr_init2 (y, 64);
  mpfr_init2 (z, 64);

  mpfr_set_str (max_val, "0.ffffffffffffffff", 16, MPFR_RNDN);
  mpfr_set_exp (max_val, mpfr_get_emax ());

  mpfr_neg (x, max_val, MPFR_RNDN);

  /* on 64-bit machines */
  for (i = 41; i < 45; i++)
    {
      mpfr_set_si_2exp (y, -1, i, MPFR_RNDN);
      mpfr_add_si (y, y, 1, MPFR_RNDN);
      tern = mpfr_pow (z, x, y, MPFR_RNDN);
      MPFR_ASSERTN (mpfr_zero_p (z) && MPFR_IS_NEG (z) && tern > 0);
    }

  /* on 32-bit machines */
  for (i = 9; i < 13; i++)
    {
      mpfr_set_si_2exp (y, -1, i, MPFR_RNDN);
      mpfr_add_si (y, y, 1, MPFR_RNDN);
      tern = mpfr_pow (z, x, y, MPFR_RNDN);
      MPFR_ASSERTN(mpfr_zero_p (z) && MPFR_SIGN(z) < 0);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (max_val);

  mpfr_set_emin (emin);
  mpfr_set_emax (emax);
}

/* Bug found by Kevin P. Rauch */
static void
bug20071218 (void)
{
  mpfr_t x, y, z, t;
  int tern;

  mpfr_inits2 (64, x, y, z, t, (mpfr_ptr) 0);
  mpfr_set_str (x, "0x.80000000000002P-1023", 0, MPFR_RNDN);
  mpfr_set_str (y, "100000.000000002", 16, MPFR_RNDN);
  mpfr_set_ui (t, 0, MPFR_RNDN);
  mpfr_nextabove (t);
  tern = mpfr_pow (z, x, y, MPFR_RNDN);
  if (mpfr_cmp0 (z, t) != 0)
    {
      printf ("Error in bug20071218 (1): Expected\n");
      mpfr_dump (t);
      printf ("Got\n");
      mpfr_dump (z);
      exit (1);
    }
  if (tern <= 0)
    {
      printf ("Error in bug20071218 (1): bad ternary value"
              " (%d instead of positive)\n", tern);
      exit (1);
    }
  mpfr_mul_2ui (y, y, 32, MPFR_RNDN);
  tern = mpfr_pow (z, x, y, MPFR_RNDN);
  if (MPFR_NOTZERO (z) || MPFR_IS_NEG (z))
    {
      printf ("Error in bug20071218 (2): expected 0, got\n");
      mpfr_dump (z);
      exit (1);
    }
  if (tern >= 0)
    {
      printf ("Error in bug20071218 (2): bad ternary value"
              " (%d instead of negative)\n", tern);
      exit (1);
    }
  mpfr_clears (x, y, z, t, (mpfr_ptr) 0);
}

/* With revision 5429, this gives:
 *   pow.c:43:  assertion failed: !mpfr_integer_p (y)
 * This is fixed in revision 5432.
 */
static void
bug20080721 (void)
{
  mpfr_t x, y, z, t[2];
  int inex;
  int rnd;
  int err = 0;

  /* Note: input values have been chosen in a way to select the
   * general case. If mpfr_pow is modified, in particular line
   *     if (y_is_integer && (MPFR_GET_EXP (y) <= 256))
   * make sure that this test still does what we want.
   */
  mpfr_inits2 (4913, x, y, (mpfr_ptr) 0);
  mpfr_inits2 (8, z, t[0], t[1], (mpfr_ptr) 0);
  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_nextbelow (x);
  mpfr_set_ui_2exp (y, 1, mpfr_get_prec (y) - 1, MPFR_RNDN);
  inex = mpfr_add_ui (y, y, 1, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);
  mpfr_set_str_binary (t[0], "-0.10101101e2");
  mpfr_set_str_binary (t[1], "-0.10101110e2");
  RND_LOOP (rnd)
    {
      int i, inex0;

      i = (rnd == MPFR_RNDN || rnd == MPFR_RNDD || rnd == MPFR_RNDA);
      inex0 = i ? -1 : 1;
      mpfr_clear_flags ();
      inex = mpfr_pow (z, x, y, (mpfr_rnd_t) rnd);
      if (__gmpfr_flags != MPFR_FLAGS_INEXACT || ! SAME_SIGN (inex, inex0)
          || MPFR_IS_NAN (z) || mpfr_cmp (z, t[i]) != 0)
        {
          unsigned int flags = __gmpfr_flags;

          printf ("Error in bug20080721 with %s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
          printf ("expected ");
          mpfr_out_str (stdout, 2, 0, t[i], MPFR_RNDN);
          printf (", inex = %d, flags = %u\n", inex0,
                  (unsigned int) MPFR_FLAGS_INEXACT);
          printf ("got      ");
          mpfr_out_str (stdout, 2, 0, z, MPFR_RNDN);
          printf (", inex = %d, flags = %u\n", inex, flags);
          err = 1;
        }
    }
  mpfr_clears (x, y, z, t[0], t[1], (mpfr_ptr) 0);
  if (err)
    exit (1);
}

/* The following test fails in r5552 (32-bit and 64-bit). This is due to:
 *   mpfr_log (t, absx, MPFR_RNDU);
 *   mpfr_mul (t, y, t, MPFR_RNDU);
 * in pow.c, that is supposed to compute an upper bound on exp(y*ln|x|),
 * but this is incorrect if y is negative.
 */
static void
bug20080820 (void)
{
  mpfr_exp_t emin;
  mpfr_t x, y, z1, z2;

  emin = mpfr_get_emin ();
  mpfr_set_emin (MPFR_EMIN_MIN);
  mpfr_init2 (x, 80);
  mpfr_init2 (y, sizeof (mpfr_exp_t) * CHAR_BIT + 32);
  mpfr_init2 (z1, 2);
  mpfr_init2 (z2, 80);
  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_nextbelow (x);
  mpfr_set_exp_t (y, mpfr_get_emin () - 2, MPFR_RNDN);
  mpfr_nextabove (y);
  mpfr_pow (z1, x, y, MPFR_RNDN);
  mpfr_pow (z2, x, y, MPFR_RNDN);
  /* As x > 0, the rounded value of x^y to nearest in precision p is equal
     to 0 iff x^y <= 2^(emin - 2). In particular, this does not depend on
     the precision p. Hence the following test. */
  if (MPFR_IS_ZERO (z1) && MPFR_NOTZERO (z2))
    {
      printf ("Error in bug20080820\n");
      exit (1);
    }
  mpfr_clears (x, y, z1, z2, (mpfr_ptr) 0);
  set_emin (emin);
}

static void
bug20110320 (void)
{
  mpfr_exp_t emin;
  mpfr_t x, y, z1, z2;
  int inex;
  unsigned int flags;

  emin = mpfr_get_emin ();
  mpfr_set_emin (11);
  mpfr_inits2 (2, x, y, z1, z2, (mpfr_ptr) 0);
  mpfr_set_ui_2exp (x, 1, 215, MPFR_RNDN);
  mpfr_set_ui (y, 1024, MPFR_RNDN);
  mpfr_clear_flags ();
  inex = mpfr_pow (z1, x, y, MPFR_RNDN);
  flags = __gmpfr_flags;
  mpfr_set_ui_2exp (z2, 1, 215*1024, MPFR_RNDN);
  if (inex != 0 || flags != 0 || ! mpfr_equal_p (z1, z2))
    {
      printf ("Error in bug20110320\n");
      printf ("Expected inex = 0, flags = 0, z = ");
      mpfr_dump (z2);
      printf ("Got      inex = %d, flags = %u, z = ", inex, flags);
      mpfr_dump (z1);
      exit (1);
    }
  mpfr_clears (x, y, z1, z2, (mpfr_ptr) 0);
  set_emin (emin);
}

int
main (int argc, char **argv)
{
  mpfr_prec_t p;

  tests_start_mpfr ();

  bug20071127 ();
  special ();
  particular_cases ();
  check_pow_ui ();
  check_pow_si ();
  check_special_pow_si ();
  pow_si_long_min ();
  for (p = 2; p < 100; p++)
    check_inexact (p);
  underflows ();
  overflows ();
  overflows2 ();
  overflows3 ();
  x_near_one ();
  bug20071103 ();
  bug20071104 ();
  bug20071128 ();
  bug20071218 ();
  bug20080721 ();
  bug20080820 ();
  bug20110320 ();

  test_generic (2, 100, 100);
  test_generic_ui (2, 100, 100);
  test_generic_si (2, 100, 100);

  data_check ("data/pow275", mpfr_pow275, "mpfr_pow275");

  tests_end_mpfr ();
  return 0;
}
