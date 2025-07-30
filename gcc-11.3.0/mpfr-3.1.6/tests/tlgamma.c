/* mpfr_tlgamma -- test file for lgamma function

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

static int
mpfr_lgamma_nosign (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  int inex, sign;

  inex = mpfr_lgamma (y, &sign, x, rnd);
  if (!MPFR_IS_SINGULAR (y))
    {
      MPFR_ASSERTN (sign == 1 || sign == -1);
      if (sign == -1 && (rnd == MPFR_RNDN || rnd == MPFR_RNDZ))
        {
          mpfr_neg (y, y, MPFR_RNDN);
          inex = -inex;
          /* This is a way to check with the generic tests, that the value
             returned in the sign variable is consistent, but warning! The
             tested function depends on the rounding mode: it is
               * lgamma(x) = log(|Gamma(x)|) in MPFR_RNDD and MPFR_RNDU;
               * lgamma(x) * sign(Gamma(x)) in MPFR_RNDN and MPFR_RNDZ. */
        }
    }

  return inex;
}

#define TEST_FUNCTION mpfr_lgamma_nosign
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y;
  int inex;
  int sign;
  mpfr_exp_t emin, emax;

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_nan (x);
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error for lgamma(NaN)\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error for lgamma(-Inf)\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0 || sign != 1)
    {
      printf ("Error for lgamma(+Inf)\n");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0 || sign != 1)
    {
      printf ("Error for lgamma(+0)\n");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0 || sign != -1)
    {
      printf ("Error for lgamma(-0)\n");
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  if (MPFR_IS_NAN (y) || mpfr_cmp_ui (y, 0) || MPFR_IS_NEG (y) || sign != 1)
    {
      printf ("Error for lgamma(1)\n");
      exit (1);
    }

  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error for lgamma(-1)\n");
      exit (1);
    }

  mpfr_set_ui (x, 2, MPFR_RNDN);
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  if (MPFR_IS_NAN (y) || mpfr_cmp_ui (y, 0) || MPFR_IS_NEG (y) || sign != 1)
    {
      printf ("Error for lgamma(2)\n");
      exit (1);
    }

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

#define CHECK_X1 "1.0762904832837976166"
#define CHECK_Y1 "-0.039418362817587634939"

  mpfr_set_str (x, CHECK_X1, 10, MPFR_RNDN);
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str (x, CHECK_Y1, 10, MPFR_RNDN);
  if (mpfr_equal_p (y, x) == 0 || sign != 1)
    {
      printf ("mpfr_lgamma("CHECK_X1") is wrong:\n"
              "expected ");
      mpfr_print_binary (x); putchar ('\n');
      printf ("got      ");
      mpfr_print_binary (y); putchar ('\n');
      exit (1);
    }

#define CHECK_X2 "9.23709516716202383435e-01"
#define CHECK_Y2 "0.049010669407893718563"
  mpfr_set_str (x, CHECK_X2, 10, MPFR_RNDN);
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str (x, CHECK_Y2, 10, MPFR_RNDN);
  if (mpfr_equal_p (y, x) == 0 || sign != 1)
    {
      printf ("mpfr_lgamma("CHECK_X2") is wrong:\n"
              "expected ");
      mpfr_print_binary (x); putchar ('\n');
      printf ("got      ");
      mpfr_print_binary (y); putchar ('\n');
      exit (1);
    }

  mpfr_set_prec (x, 8);
  mpfr_set_prec (y, 175);
  mpfr_set_ui (x, 33, MPFR_RNDN);
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDU);
  mpfr_set_prec (x, 175);
  mpfr_set_str_binary (x, "0.1010001100011101101011001101110010100001000001000001110011000001101100001111001001000101011011100100010101011110100111110101010100010011010010000101010111001100011000101111E7");
  if (mpfr_equal_p (x, y) == 0 || sign != 1)
    {
      printf ("Error in mpfr_lgamma (1)\n");
      exit (1);
    }

  mpfr_set_prec (x, 21);
  mpfr_set_prec (y, 8);
  mpfr_set_ui (y, 120, MPFR_RNDN);
  sign = -17;
  mpfr_lgamma (x, &sign, y, MPFR_RNDZ);
  mpfr_set_prec (y, 21);
  mpfr_set_str_binary (y, "0.111000101000001100101E9");
  if (mpfr_equal_p (x, y) == 0 || sign != 1)
    {
      printf ("Error in mpfr_lgamma (120)\n");
      printf ("Expected "); mpfr_print_binary (y); puts ("");
      printf ("Got      "); mpfr_print_binary (x); puts ("");
      exit (1);
    }

  mpfr_set_prec (x, 3);
  mpfr_set_prec (y, 206);
  mpfr_set_str_binary (x, "0.110e10");
  sign = -17;
  inex = mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_prec (x, 206);
  mpfr_set_str_binary (x, "0.10000111011000000011100010101001100110001110000111100011000100100110110010001011011110101001111011110110000001010100111011010000000011100110110101100111000111010011110010000100010111101010001101000110101001E13");
  if (mpfr_equal_p (x, y) == 0 || sign != 1)
    {
      printf ("Error in mpfr_lgamma (768)\n");
      exit (1);
    }
  if (inex >= 0)
    {
      printf ("Wrong flag for mpfr_lgamma (768)\n");
      exit (1);
    }

  mpfr_set_prec (x, 4);
  mpfr_set_prec (y, 4);
  mpfr_set_str_binary (x, "0.1100E-66");
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.1100E6");
  if (mpfr_equal_p (x, y) == 0 || sign != 1)
    {
      printf ("Error for lgamma(0.1100E-66)\n");
      printf ("Expected ");
      mpfr_dump (x);
      printf ("Got      ");
      mpfr_dump (y);
      exit (1);
    }

  mpfr_set_prec (x, 256);
  mpfr_set_prec (y, 32);
  mpfr_set_si_2exp (x, -1, 200, MPFR_RNDN);
  mpfr_add_ui (x, x, 1, MPFR_RNDN);
  mpfr_div_2ui (x, x, 1, MPFR_RNDN);
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_prec (x, 32);
  mpfr_set_str_binary (x, "-0.10001000111011111011000010100010E207");
  if (mpfr_equal_p (x, y) == 0 || sign != 1)
    {
      printf ("Error for lgamma(-2^199+0.5)\n");
      printf ("Got        ");
      mpfr_dump (y);
      printf ("instead of ");
      mpfr_dump (x);
      exit (1);
    }

  mpfr_set_prec (x, 256);
  mpfr_set_prec (y, 32);
  mpfr_set_si_2exp (x, -1, 200, MPFR_RNDN);
  mpfr_sub_ui (x, x, 1, MPFR_RNDN);
  mpfr_div_2ui (x, x, 1, MPFR_RNDN);
  sign = -17;
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_prec (x, 32);
  mpfr_set_str_binary (x, "-0.10001000111011111011000010100010E207");
  if (mpfr_equal_p (x, y) == 0 || sign != -1)
    {
      printf ("Error for lgamma(-2^199-0.5)\n");
      printf ("Got        ");
      mpfr_dump (y);
      printf ("with sign %d instead of ", sign);
      mpfr_dump (x);
      printf ("with sign -1.\n");
      exit (1);
    }

  mpfr_set_prec (x, 10);
  mpfr_set_prec (y, 10);
  mpfr_set_str_binary (x, "-0.1101111000E-3");
  inex = mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "10.01001011");
  if (mpfr_equal_p (x, y) == 0 || sign != -1 || inex >= 0)
    {
      printf ("Error for lgamma(-0.1101111000E-3)\n");
      printf ("Got        ");
      mpfr_dump (y);
      printf ("instead of ");
      mpfr_dump (x);
      printf ("with sign %d instead of -1 (inex=%d).\n", sign, inex);
      exit (1);
    }

  mpfr_set_prec (x, 18);
  mpfr_set_prec (y, 28);
  mpfr_set_str_binary (x, "-1.10001101010001101e-196");
  inex = mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_prec (x, 28);
  mpfr_set_str_binary (x, "0.100001110110101011011010011E8");
  MPFR_ASSERTN (mpfr_equal_p (x, y) && inex < 0);

  /* values reported by Kaveh Ghazi on 14 Jul 2007, where mpfr_lgamma()
     takes forever */
#define VAL1 "-0.11100001001010110111001010001001001011110100110000110E-55"
#define OUT1 "100110.01000000010111001110110101110101001001100110111"
#define VAL2 "-0.11100001001010110111001010001001001011110011111111100E-55"
#define OUT2 "100110.0100000001011100111011010111010100100110011111"
#define VAL3 "-0.11100001001010110111001010001001001001110101101010100E-55"
#define OUT3 "100110.01000000010111001110110101110101001011110111011"
#define VAL4 "-0.10001111110110110100100100000000001111110001001001011E-57"
#define OUT4 "101000.0001010111110011101101000101111111010001100011"
#define VAL5 "-0.10001111110110110100100100000000001111011111100001000E-57"
#define OUT5 "101000.00010101111100111011010001011111110100111000001"
#define VAL6 "-0.10001111110110110100100100000000001111011101100011001E-57"
#define OUT6 "101000.0001010111110011101101000101111111010011101111"

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

  mpfr_set_str_binary (x, VAL1);
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str_binary (x, OUT1);
  MPFR_ASSERTN(sign == -1 && mpfr_equal_p(x, y));

  mpfr_set_str_binary (x, VAL2);
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str_binary (x, OUT2);
  MPFR_ASSERTN(sign == -1 && mpfr_equal_p (x, y));

  mpfr_set_str_binary (x, VAL3);
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str_binary (x, OUT3);
  MPFR_ASSERTN(sign == -1 && mpfr_equal_p (x, y));

  mpfr_set_str_binary (x, VAL4);
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str_binary (x, OUT4);
  MPFR_ASSERTN(sign == -1 && mpfr_equal_p (x, y));

  mpfr_set_str_binary (x, VAL5);
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str_binary (x, OUT5);
  MPFR_ASSERTN(sign == -1 && mpfr_equal_p (x, y));

  mpfr_set_str_binary (x, VAL6);
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str_binary (x, OUT6);
  MPFR_ASSERTN(sign == -1 && mpfr_equal_p (x, y));

  /* further test from Kaveh Ghazi */
  mpfr_set_str_binary (x, "-0.10011010101001010010001110010111010111011101010111001E-53");
  mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "100101.00111101101010000000101010111010001111001101111");
  MPFR_ASSERTN(sign == -1 && mpfr_equal_p (x, y));

  /* bug found by Kevin Rauch on 26 Oct 2007 */
  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  mpfr_set_emin (-1000000000);
  mpfr_set_emax (1000000000);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_lgamma (x, &sign, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_get_emin () == -1000000000);
  MPFR_ASSERTN(mpfr_get_emax () == 1000000000);
  mpfr_set_emin (emin);
  mpfr_set_emax (emax);

  /* two other bugs reported by Kevin Rauch on 27 Oct 2007 */
  mpfr_set_prec (x, 128);
  mpfr_set_prec (y, 128);
  mpfr_set_str_binary (x, "0.11000110011110111111110010100110000000000000000000000000000000000000000000000000000000000000000001000011000110100100110111101010E-765689");
  inex = mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "10000001100100101111011011010000111010001001110000111010011000101001011111011111110011011010110100101111110111001001010100011101E-108");
  MPFR_ASSERTN(inex < 0 && mpfr_cmp (y, x) == 0 && sign > 0);

  mpfr_set_prec (x, 128);
  mpfr_set_prec (y, 256);
  mpfr_set_str_binary (x, "0.1011111111111111100000111011111E-31871");
  inex = mpfr_lgamma (y, &sign, x, MPFR_RNDN);
  mpfr_set_prec (x, 256);
  mpfr_set_str (x, "AC9729B83707E6797612D0D76DAF42B1240A677FF1B6E3783FD4E53037143B1P-237", 16, MPFR_RNDN);
  MPFR_ASSERTN(inex < 0 && mpfr_cmp (y, x) == 0 && sign > 0);

  mpfr_clear (x);
  mpfr_clear (y);
}

static int
mpfr_lgamma1 (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t r)
{
  int sign;

  return mpfr_lgamma (y, &sign, x, r);
}

int
main (void)
{
  tests_start_mpfr ();

  special ();
  test_generic (2, 100, 2);

  data_check ("data/lgamma", mpfr_lgamma1, "mpfr_lgamma");

  tests_end_mpfr ();
  return 0;
}
