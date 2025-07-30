/* mpfr_tgamma -- test file for gamma function

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

/* Note: there could be an incorrect test about suspicious overflow
   (MPFR_SUSPICIOUS_OVERFLOW) for x = 2^(-emax) = 0.5 * 2^(emin+1) in
   RNDZ or RNDD, but this case is never tested in the generic tests. */
#define TEST_FUNCTION mpfr_gamma
#define RAND_FUNCTION(x) mpfr_random2(x, MPFR_LIMB_SIZE (x), 1, RANDS)
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y;
  int inex;

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_nan (x);
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error for gamma(NaN)\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error for gamma(-Inf)\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error for gamma(+Inf)\n");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error for gamma(+0)\n");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) > 0)
    {
      printf ("Error for gamma(-0)\n");
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 1))
    {
      printf ("Error for gamma(1)\n");
      exit (1);
    }

  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error for gamma(-1)\n");
      exit (1);
    }

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

#define CHECK_X1 "1.0762904832837976166"
#define CHECK_Y1 "0.96134843256452096050"

  mpfr_set_str (x, CHECK_X1, 10, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  mpfr_set_str (x, CHECK_Y1, 10, MPFR_RNDN);
  if (mpfr_cmp (y, x))
    {
      printf ("mpfr_lngamma("CHECK_X1") is wrong:\n"
              "expected ");
      mpfr_print_binary (x); putchar ('\n');
      printf ("got      ");
      mpfr_print_binary (y); putchar ('\n');
      exit (1);
    }

#define CHECK_X2 "9.23709516716202383435e-01"
#define CHECK_Y2 "1.0502315560291053398"
  mpfr_set_str (x, CHECK_X2, 10, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  mpfr_set_str (x, CHECK_Y2, 10, MPFR_RNDN);
  if (mpfr_cmp (y, x))
    {
      printf ("mpfr_lngamma("CHECK_X2") is wrong:\n"
              "expected ");
      mpfr_print_binary (x); putchar ('\n');
      printf ("got      ");
      mpfr_print_binary (y); putchar ('\n');
      exit (1);
    }

  mpfr_set_prec (x, 8);
  mpfr_set_prec (y, 175);
  mpfr_set_ui (x, 33, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDU);
  mpfr_set_prec (x, 175);
  mpfr_set_str_binary (x, "0.110010101011010101101000010101010111000110011101001000101011000001100010111001101001011E118");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_gamma (1)\n");
      exit (1);
    }

  mpfr_set_prec (x, 21);
  mpfr_set_prec (y, 8);
  mpfr_set_ui (y, 120, MPFR_RNDN);
  mpfr_gamma (x, y, MPFR_RNDZ);
  mpfr_set_prec (y, 21);
  mpfr_set_str_binary (y, "0.101111101110100110110E654");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_gamma (120)\n");
      printf ("Expected "); mpfr_print_binary (y); puts ("");
      printf ("Got      "); mpfr_print_binary (x); puts ("");
      exit (1);
    }

  mpfr_set_prec (x, 3);
  mpfr_set_prec (y, 206);
  mpfr_set_str_binary (x, "0.110e10");
  inex = mpfr_gamma (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 206);
  mpfr_set_str_binary (x, "0.110111100001000001101010010001000111000100000100111000010011100011011111001100011110101000111101101100110001001100110100001001111110000101010000100100011100010011101110000001000010001100010000101001111E6250");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_gamma (768)\n");
      exit (1);
    }
  if (inex <= 0)
    {
      printf ("Wrong flag for mpfr_gamma (768)\n");
      exit (1);
    }

  /* worst case to exercise retry */
  mpfr_set_prec (x, 1000);
  mpfr_set_prec (y, 869);
  mpfr_const_pi (x, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);

  mpfr_set_prec (x, 4);
  mpfr_set_prec (y, 4);
  mpfr_set_str_binary (x, "-0.1100E-66");
  mpfr_gamma (y, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "-0.1011E67");
  if (mpfr_cmp (x, y))
    {
      printf ("Error for gamma(-0.1100E-66)\n");
      exit (1);
    }

  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 2);
  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_clear_inexflag ();
  mpfr_gamma (y, x, MPFR_RNDN);
  if (mpfr_inexflag_p ())
    {
      printf ("Wrong inexact flag for gamma(2)\n");
      printf ("expected 0, got 1\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
special_overflow (void)
{
  mpfr_t x, y;
  mpfr_exp_t emin, emax;
  int inex;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  set_emin (-125);
  set_emax (128);

  mpfr_init2 (x, 24);
  mpfr_init2 (y, 24);
  mpfr_set_str_binary (x, "0.101100100000000000110100E7");
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y))
    {
      printf ("Overflow error.\n");
      mpfr_dump (y);
      exit (1);
    }

  /* problem mentioned by Kenneth Wilder, 18 Aug 2005 */
  mpfr_set_prec (x, 29);
  mpfr_set_prec (y, 29);
  mpfr_set_str (x, "-200000000.5", 10, MPFR_RNDN); /* exact */
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!(mpfr_zero_p (y) && MPFR_SIGN (y) < 0))
    {
      printf ("Error for gamma(-200000000.5)\n");
      printf ("expected -0");
      printf ("got      ");
      mpfr_dump (y);
      exit (1);
    }

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_str (x, "-200000000.1", 10, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!(mpfr_zero_p (y) && MPFR_SIGN (y) < 0))
    {
      printf ("Error for gamma(-200000000.1), prec=53\n");
      printf ("expected -0");
      printf ("got      ");
      mpfr_dump (y);
      exit (1);
    }

  /* another problem mentioned by Kenneth Wilder, 29 Aug 2005 */
  mpfr_set_prec (x, 333);
  mpfr_set_prec (y, 14);
  mpfr_set_str (x, "-2.0000000000000000000000005", 10, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 14);
  mpfr_set_str_binary (x, "-11010011110001E66");
  if (mpfr_cmp (x, y))
    {
      printf ("Error for gamma(-2.0000000000000000000000005)\n");
      printf ("expected "); mpfr_dump (x);
      printf ("got      "); mpfr_dump (y);
      exit (1);
    }

  /* another tests from Kenneth Wilder, 31 Aug 2005 */
  set_emax (200);
  set_emin (-200);
  mpfr_set_prec (x, 38);
  mpfr_set_prec (y, 54);
  mpfr_set_str_binary (x, "0.11101111011100111101001001010110101001E-166");
  mpfr_gamma (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 54);
  mpfr_set_str_binary (x, "0.100010001101100001110110001010111111010000100101011E167");
  if (mpfr_cmp (x, y))
    {
      printf ("Error for gamma (test 1)\n");
      printf ("expected "); mpfr_dump (x);
      printf ("got      "); mpfr_dump (y);
      exit (1);
    }

  set_emax (1000);
  set_emin (-2000);
  mpfr_set_prec (x, 38);
  mpfr_set_prec (y, 71);
  mpfr_set_str_binary (x, "10101011011100001111111000010111110010E-1034");
  /* 184083777010*2^(-1034) */
  mpfr_gamma (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 71);
  mpfr_set_str_binary (x, "10111111001000011110010001000000000000110011110000000011101011111111100E926");
  /* 1762885132679550982140*2^926 */
  if (mpfr_cmp (x, y))
    {
      printf ("Error for gamma (test 2)\n");
      printf ("expected "); mpfr_dump (x);
      printf ("got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_prec (x, 38);
  mpfr_set_prec (y, 88);
  mpfr_set_str_binary (x, "10111100111001010000100001100100100101E-104");
  /* 202824096037*2^(-104) */
  mpfr_gamma (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 88);
  mpfr_set_str_binary (x, "1010110101111000111010111100010110101010100110111000001011000111000011101100001101110010E-21");
  /* 209715199999500283894743922*2^(-21) */
  if (mpfr_cmp (x, y))
    {
      printf ("Error for gamma (test 3)\n");
      printf ("expected "); mpfr_dump (x);
      printf ("got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_prec (x, 171);
  mpfr_set_prec (y, 38);
  mpfr_set_str (x, "-2993155353253689176481146537402947624254601559176535", 10,
                MPFR_RNDN);
  mpfr_div_2exp (x, x, 170, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 38);
  mpfr_set_str (x, "201948391737", 10, MPFR_RNDN);
  mpfr_mul_2exp (x, x, 92, MPFR_RNDN);
  if (mpfr_cmp (x, y))
    {
      printf ("Error for gamma (test 5)\n");
      printf ("expected "); mpfr_dump (x);
      printf ("got      "); mpfr_dump (y);
      exit (1);
    }

  set_emin (-500000);
  mpfr_set_prec (x, 337);
  mpfr_set_prec (y, 38);
  mpfr_set_str (x, "-30000.000000000000000000000000000000000000000000001", 10,
                MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 38);
  mpfr_set_str (x, "-3.623795987425E-121243", 10, MPFR_RNDN);
  if (mpfr_cmp (x, y))
    {
      printf ("Error for gamma (test 7)\n");
      printf ("expected "); mpfr_dump (x);
      printf ("got      "); mpfr_dump (y);
      exit (1);
    }

  /* was producing infinite loop */
  set_emin (emin);
  mpfr_set_prec (x, 71);
  mpfr_set_prec (y, 71);
  mpfr_set_str (x, "-200000000.1", 10, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!(mpfr_zero_p (y) && MPFR_SIGN (y) < 0))
    {
      printf ("Error for gamma (test 8)\n");
      printf ("expected "); mpfr_dump (x);
      printf ("got      "); mpfr_dump (y);
      exit (1);
    }

  set_emax (1073741823);
  mpfr_set_prec (x, 29);
  mpfr_set_prec (y, 29);
  mpfr_set_str (x, "423786866", 10, MPFR_RNDN);
  mpfr_gamma (y, x, MPFR_RNDN);
  if (!mpfr_inf_p (y) || mpfr_sgn (y) < 0)
    {
      printf ("Error for gamma(423786866)\n");
      exit (1);
    }

  /* check exact result */
  mpfr_set_prec (x, 2);
  mpfr_set_ui (x, 3, MPFR_RNDN);
  inex = mpfr_gamma (x, x, MPFR_RNDN);
  if (inex != 0 || mpfr_cmp_ui (x, 2) != 0)
    {
      printf ("Error for gamma(3)\n");
      exit (1);
    }

  mpfr_set_emax (1024);
  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_str_binary (x, "101010110100110011111010000110001000111100000110101E-43");
  mpfr_gamma (x, x, MPFR_RNDU);
  mpfr_set_str_binary (y, "110000011110001000111110110101011110000100001111111E971");
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("Error for gamma(4)\n");
      printf ("expected "); mpfr_dump (y);
      printf ("got      "); mpfr_dump (x);
      exit (1);
    }

  set_emin (emin);
  set_emax (emax);

  /* bug found by Kevin Rauch, 26 Oct 2007 */
  mpfr_set_str (x, "1e19", 10, MPFR_RNDN);
  inex = mpfr_gamma (x, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (x) && inex > 0);

  mpfr_clear (y);
  mpfr_clear (x);
}

/* test gamma on some integral values (from Christopher Creutzig). */
static void
gamma_integer (void)
{
  mpz_t n;
  mpfr_t x, y;
  unsigned int i;

  mpz_init (n);
  mpfr_init2 (x, 149);
  mpfr_init2 (y, 149);

  for (i = 0; i < 100; i++)
    {
      mpz_fac_ui (n, i);
      mpfr_set_ui (x, i+1, MPFR_RNDN);
      mpfr_gamma (y, x, MPFR_RNDN);
      mpfr_set_z (x, n, MPFR_RNDN);
      if (!mpfr_equal_p (x, y))
        {
          printf ("Error for gamma(%u)\n", i+1);
          printf ("expected "); mpfr_dump (x);
          printf ("got      "); mpfr_dump (y);
          exit (1);
        }
    }
  mpfr_clear (y);
  mpfr_clear (x);
  mpz_clear (n);
}

/* bug found by Kevin Rauch */
static void
test20071231 (void)
{
  mpfr_t x;
  int inex;
  mpfr_exp_t emin;

  emin = mpfr_get_emin ();
  mpfr_set_emin (-1000000);

  mpfr_init2 (x, 21);
  mpfr_set_str (x, "-1000001.5", 10, MPFR_RNDN);
  inex = mpfr_gamma (x, x, MPFR_RNDN);
  MPFR_ASSERTN(MPFR_IS_ZERO(x) && MPFR_IS_POS(x) && inex < 0);
  mpfr_clear (x);

  mpfr_set_emin (emin);

  mpfr_init2 (x, 53);
  mpfr_set_str (x, "-1000000001.5", 10, MPFR_RNDN);
  inex = mpfr_gamma (x, x, MPFR_RNDN);
  MPFR_ASSERTN(MPFR_IS_ZERO(x) && MPFR_IS_POS(x) && inex < 0);
  mpfr_clear (x);
}

/* bug found by Stathis in mpfr_gamma, only occurs on 32-bit machines;
   the second test is for 64-bit machines. This bug reappeared due to
   r8159. */
static void
test20100709 (void)
{
  mpfr_t x, y, z;
  int sign;
  int inex;
  mpfr_exp_t emin;

  mpfr_init2 (x, 100);
  mpfr_init2 (y, 32);
  mpfr_init2 (z, 32);
  mpfr_set_str (x, "-4.6308260837372266e+07", 10, MPFR_RNDN);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_nextabove (y);
  mpfr_log (y, y, MPFR_RNDD);
  mpfr_const_log2 (z, MPFR_RNDU);
  mpfr_sub (y, y, z, MPFR_RNDD); /* log(MIN/2) = log(MIN) - log(2) */
  mpfr_lgamma (z, &sign, x, MPFR_RNDU);
  MPFR_ASSERTN (sign == -1);
  MPFR_ASSERTN (mpfr_less_p (z, y)); /* thus underflow */
  inex = mpfr_gamma (x, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO(x) && MPFR_IS_NEG(x) && inex > 0);
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);

  /* Similar test for 64-bit machines (also valid with a 32-bit exponent,
     but will not trigger the bug). */
  emin = mpfr_get_emin ();
  mpfr_set_emin (MPFR_EMIN_MIN);
  mpfr_init2 (x, 100);
  mpfr_init2 (y, 32);
  mpfr_init2 (z, 32);
  mpfr_set_str (x, "-90.6308260837372266e+15", 10, MPFR_RNDN);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_nextabove (y);
  mpfr_log (y, y, MPFR_RNDD);
  mpfr_const_log2 (z, MPFR_RNDU);
  mpfr_sub (y, y, z, MPFR_RNDD); /* log(MIN/2) = log(MIN) - log(2) */
  mpfr_lgamma (z, &sign, x, MPFR_RNDU);
  MPFR_ASSERTN (sign == -1);
  MPFR_ASSERTN (mpfr_less_p (z, y)); /* thus underflow */
  inex = mpfr_gamma (x, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO(x) && MPFR_IS_NEG(x) && inex > 0);
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_set_emin (emin);
}

/* bug found by Giridhar Tammana */
static void
test20120426 (void)
{
  mpfr_t xa, xb;
  int i;
  mpfr_exp_t emin;

  mpfr_init2 (xa, 53);
  mpfr_init2 (xb, 53);
  mpfr_set_d (xb, -168.5, MPFR_RNDN);
  emin = mpfr_get_emin ();
  mpfr_set_emin (-1073);
  i = mpfr_gamma (xa, xb, MPFR_RNDN);
  i = mpfr_subnormalize (xa, i, MPFR_RNDN); /* new ternary value */
  mpfr_set_str (xb, "-9.5737343987585366746184749943e-304", 10, MPFR_RNDN);
  if (!((i > 0) && (mpfr_cmp (xa, xb) == 0)))
    {
      printf ("Error in test20120426, i=%d\n", i);
      printf ("expected ");
      mpfr_print_binary (xb); putchar ('\n');
      printf ("got      ");
      mpfr_print_binary (xa); putchar ('\n');
      exit (1);
    }
  mpfr_set_emin (emin);
  mpfr_clear (xa);
  mpfr_clear (xb);
}

static void
exprange (void)
{
  mpfr_exp_t emin, emax;
  mpfr_t x, y, z;
  int inex1, inex2;
  unsigned int flags1, flags2;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  mpfr_init2 (x, 16);
  mpfr_inits2 (8, y, z, (mpfr_ptr) 0);

  mpfr_set_ui_2exp (x, 5, -1, MPFR_RNDN);
  mpfr_clear_flags ();
  inex1 = mpfr_gamma (y, x, MPFR_RNDN);
  flags1 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  mpfr_set_emin (0);
  mpfr_clear_flags ();
  inex2 = mpfr_gamma (z, x, MPFR_RNDN);
  flags2 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  mpfr_set_emin (emin);
  if (inex1 != inex2 || flags1 != flags2 || ! mpfr_equal_p (y, z))
    {
      printf ("Error in exprange (test1)\n");
      printf ("x = ");
      mpfr_dump (x);
      printf ("Expected inex1 = %d, flags1 = %u, ", SIGN (inex1), flags1);
      mpfr_dump (y);
      printf ("Got      inex2 = %d, flags2 = %u, ", SIGN (inex2), flags2);
      mpfr_dump (z);
      exit (1);
    }

  mpfr_set_ui_2exp (x, 32769, -60, MPFR_RNDN);
  mpfr_clear_flags ();
  inex1 = mpfr_gamma (y, x, MPFR_RNDD);
  flags1 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  mpfr_set_emax (45);
  mpfr_clear_flags ();
  inex2 = mpfr_gamma (z, x, MPFR_RNDD);
  flags2 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  mpfr_set_emax (emax);
  if (inex1 != inex2 || flags1 != flags2 || ! mpfr_equal_p (y, z))
    {
      printf ("Error in exprange (test2)\n");
      printf ("x = ");
      mpfr_dump (x);
      printf ("Expected inex1 = %d, flags1 = %u, ", SIGN (inex1), flags1);
      mpfr_dump (y);
      printf ("Got      inex2 = %d, flags2 = %u, ", SIGN (inex2), flags2);
      mpfr_dump (z);
      exit (1);
    }

  mpfr_set_emax (44);
  mpfr_clear_flags ();
  inex1 = mpfr_check_range (y, inex1, MPFR_RNDD);
  flags1 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  mpfr_clear_flags ();
  inex2 = mpfr_gamma (z, x, MPFR_RNDD);
  flags2 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  mpfr_set_emax (emax);
  if (inex1 != inex2 || flags1 != flags2 || ! mpfr_equal_p (y, z))
    {
      printf ("Error in exprange (test3)\n");
      printf ("x = ");
      mpfr_dump (x);
      printf ("Expected inex1 = %d, flags1 = %u, ", SIGN (inex1), flags1);
      mpfr_dump (y);
      printf ("Got      inex2 = %d, flags2 = %u, ", SIGN (inex2), flags2);
      mpfr_dump (z);
      exit (1);
    }

  mpfr_set_ui_2exp (x, 1, -60, MPFR_RNDN);
  mpfr_clear_flags ();
  inex1 = mpfr_gamma (y, x, MPFR_RNDD);
  flags1 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  mpfr_set_emax (60);
  mpfr_clear_flags ();
  inex2 = mpfr_gamma (z, x, MPFR_RNDD);
  flags2 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  mpfr_set_emax (emax);
  if (inex1 != inex2 || flags1 != flags2 || ! mpfr_equal_p (y, z))
    {
      printf ("Error in exprange (test4)\n");
      printf ("x = ");
      mpfr_dump (x);
      printf ("Expected inex1 = %d, flags1 = %u, ", SIGN (inex1), flags1);
      mpfr_dump (y);
      printf ("Got      inex2 = %d, flags2 = %u, ", SIGN (inex2), flags2);
      mpfr_dump (z);
      exit (1);
    }

  MPFR_ASSERTN (MPFR_EMIN_MIN == - MPFR_EMAX_MAX);
  mpfr_set_emin (MPFR_EMIN_MIN);
  mpfr_set_emax (MPFR_EMAX_MAX);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_nextabove (x);  /* x = 2^(emin - 1) */
  mpfr_set_inf (y, 1);
  inex1 = 1;
  flags1 = MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW;
  mpfr_clear_flags ();
  /* MPFR_RNDU: overflow, infinity since 1/x = 2^(emax + 1) */
  inex2 = mpfr_gamma (z, x, MPFR_RNDU);
  flags2 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  if (inex1 != inex2 || flags1 != flags2 || ! mpfr_equal_p (y, z))
    {
      printf ("Error in exprange (test5)\n");
      printf ("x = ");
      mpfr_dump (x);
      printf ("Expected inex1 = %d, flags1 = %u, ", SIGN (inex1), flags1);
      mpfr_dump (y);
      printf ("Got      inex2 = %d, flags2 = %u, ", SIGN (inex2), flags2);
      mpfr_dump (z);
      exit (1);
    }
  mpfr_clear_flags ();
  /* MPFR_RNDN: overflow, infinity since 1/x = 2^(emax + 1) */
  inex2 = mpfr_gamma (z, x, MPFR_RNDN);
  flags2 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  if (inex1 != inex2 || flags1 != flags2 || ! mpfr_equal_p (y, z))
    {
      printf ("Error in exprange (test6)\n");
      printf ("x = ");
      mpfr_dump (x);
      printf ("Expected inex1 = %d, flags1 = %u, ", SIGN (inex1), flags1);
      mpfr_dump (y);
      printf ("Got      inex2 = %d, flags2 = %u, ", SIGN (inex2), flags2);
      mpfr_dump (z);
      exit (1);
    }
  mpfr_nextbelow (y);
  inex1 = -1;
  mpfr_clear_flags ();
  /* MPFR_RNDD: overflow, maxnum since 1/x = 2^(emax + 1) */
  inex2 = mpfr_gamma (z, x, MPFR_RNDD);
  flags2 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  if (inex1 != inex2 || flags1 != flags2 || ! mpfr_equal_p (y, z))
    {
      printf ("Error in exprange (test7)\n");
      printf ("x = ");
      mpfr_dump (x);
      printf ("Expected inex1 = %d, flags1 = %u, ", SIGN (inex1), flags1);
      mpfr_dump (y);
      printf ("Got      inex2 = %d, flags2 = %u, ", SIGN (inex2), flags2);
      mpfr_dump (z);
      exit (1);
    }
  mpfr_mul_2ui (x, x, 1, MPFR_RNDN);  /* x = 2^emin */
  mpfr_set_inf (y, 1);
  inex1 = 1;
  flags1 = MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW;
  mpfr_clear_flags ();
  /* MPFR_RNDU: overflow, infinity since 1/x = 2^emax */
  inex2 = mpfr_gamma (z, x, MPFR_RNDU);
  flags2 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  if (inex1 != inex2 || flags1 != flags2 || ! mpfr_equal_p (y, z))
    {
      printf ("Error in exprange (test8)\n");
      printf ("x = ");
      mpfr_dump (x);
      printf ("Expected inex1 = %d, flags1 = %u, ", SIGN (inex1), flags1);
      mpfr_dump (y);
      printf ("Got      inex2 = %d, flags2 = %u, ", SIGN (inex2), flags2);
      mpfr_dump (z);
      exit (1);
    }
  mpfr_clear_flags ();
  /* MPFR_RNDN: overflow, infinity since 1/x = 2^emax */
  inex2 = mpfr_gamma (z, x, MPFR_RNDN);
  flags2 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  if (inex1 != inex2 || flags1 != flags2 || ! mpfr_equal_p (y, z))
    {
      printf ("Error in exprange (test9)\n");
      printf ("x = ");
      mpfr_dump (x);
      printf ("Expected inex1 = %d, flags1 = %u, ", SIGN (inex1), flags1);
      mpfr_dump (y);
      printf ("Got      inex2 = %d, flags2 = %u, ", SIGN (inex2), flags2);
      mpfr_dump (z);
      exit (1);
    }
  mpfr_nextbelow (y);
  inex1 = -1;
  flags1 = MPFR_FLAGS_INEXACT;
  mpfr_clear_flags ();
  /* MPFR_RNDD: no overflow, maxnum since 1/x = 2^emax and euler > 0 */
  inex2 = mpfr_gamma (z, x, MPFR_RNDD);
  flags2 = __gmpfr_flags;
  MPFR_ASSERTN (mpfr_inexflag_p ());
  if (inex1 != inex2 || flags1 != flags2 || ! mpfr_equal_p (y, z))
    {
      printf ("Error in exprange (test10)\n");
      printf ("x = ");
      mpfr_dump (x);
      printf ("Expected inex1 = %d, flags1 = %u, ", SIGN (inex1), flags1);
      mpfr_dump (y);
      printf ("Got      inex2 = %d, flags2 = %u, ", SIGN (inex2), flags2);
      mpfr_dump (z);
      exit (1);
    }
  mpfr_set_emin (emin);
  mpfr_set_emax (emax);

  mpfr_clears (x, y, z, (mpfr_ptr) 0);
}

static int
tiny_aux (int stop, mpfr_exp_t e)
{
  mpfr_t x, y, z;
  int r, s, spm, inex, err = 0;
  int expected_dir[2][5] = { { 1, -1, 1, -1, 1 }, { 1, 1, 1, -1, -1 } };
  mpfr_exp_t saved_emax;

  saved_emax = mpfr_get_emax ();

  mpfr_init2 (x, 32);
  mpfr_inits2 (8, y, z, (mpfr_ptr) 0);

  mpfr_set_ui_2exp (x, 1, e, MPFR_RNDN);
  spm = 1;
  for (s = 0; s < 2; s++)
    {
      RND_LOOP(r)
        {
          mpfr_rnd_t rr = (mpfr_rnd_t) r;
          mpfr_exp_t exponent, emax;

          /* Exponent of the rounded value in unbounded exponent range. */
          exponent = expected_dir[s][r] < 0 && s == 0 ? - e : 1 - e;

          for (emax = exponent - 1; emax <= exponent; emax++)
            {
              unsigned int flags, expected_flags = MPFR_FLAGS_INEXACT;
              int overflow, expected_inex = expected_dir[s][r];

              if (emax > MPFR_EMAX_MAX)
                break;
              mpfr_set_emax (emax);

              mpfr_clear_flags ();
              inex = mpfr_gamma (y, x, rr);
              flags = __gmpfr_flags;
              mpfr_clear_flags ();
              mpfr_set_si_2exp (z, spm, - e, MPFR_RNDU);
              overflow = mpfr_overflow_p ();
              /* z is 1/x - euler rounded toward +inf */

              if (overflow && rr == MPFR_RNDN && s == 1)
                expected_inex = -1;

              if (expected_inex < 0)
                mpfr_nextbelow (z); /* 1/x - euler rounded toward -inf */

              if (exponent > emax)
                expected_flags |= MPFR_FLAGS_OVERFLOW;

              if (!(mpfr_equal_p (y, z) && flags == expected_flags
                    && SAME_SIGN (inex, expected_inex)))
                {
                  printf ("Error in tiny for s = %d, r = %s, emax = %"
                          MPFR_EXP_FSPEC "d%s\n  on ",
                          s, mpfr_print_rnd_mode (rr), emax,
                          exponent > emax ? " (overflow)" : "");
                  mpfr_dump (x);
                  printf ("  expected inex = %2d, ", expected_inex);
                  mpfr_dump (z);
                  printf ("  got      inex = %2d, ", SIGN (inex));
                  mpfr_dump (y);
                  printf ("  expected flags = %u, got %u\n",
                          expected_flags, flags);
                  if (stop)
                    exit (1);
                  err = 1;
                }
            }
        }
      mpfr_neg (x, x, MPFR_RNDN);
      spm = - spm;
    }

  mpfr_clears (x, y, z, (mpfr_ptr) 0);
  mpfr_set_emax (saved_emax);
  return err;
}

static void
tiny (int stop)
{
  mpfr_exp_t emin;
  int err = 0;

  emin = mpfr_get_emin ();

  /* Note: in r7499, exponent -17 will select the generic code (in
     tiny_aux, x has precision 32), while the other exponent values
     will select special code for tiny values. */
  err |= tiny_aux (stop, -17);
  err |= tiny_aux (stop, -999);
  err |= tiny_aux (stop, mpfr_get_emin ());

  if (emin != MPFR_EMIN_MIN)
    {
      mpfr_set_emin (MPFR_EMIN_MIN);
      err |= tiny_aux (stop, MPFR_EMIN_MIN);
      mpfr_set_emin (emin);
    }

  if (err)
    exit (1);
}

/* Test mpfr_gamma in precision p1 by comparing it with exp(lgamma(x))
   computing with a working precision p2. Assume that x is not an
   integer <= 2. */
static void
exp_lgamma (mpfr_t x, mpfr_prec_t p1, mpfr_prec_t p2)
{
  mpfr_t yd, yu, zd, zu;
  int inexd, inexu, sign;
  int underflow = -1, overflow = -1;  /* -1: we don't know */
  int got_underflow, got_overflow;

  if (mpfr_integer_p (x) && mpfr_cmp_si (x, 2) <= 0)
    {
      printf ("Warning! x is an integer <= 2 in exp_lgamma: ");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      return;
    }
  mpfr_inits2 (p2, yd, yu, (mpfr_ptr) 0);
  inexd = mpfr_lgamma (yd, &sign, x, MPFR_RNDD);
  mpfr_set (yu, yd, MPFR_RNDN);  /* exact */
  if (inexd)
    mpfr_nextabove (yu);
  mpfr_clear_flags ();
  mpfr_exp (yd, yd, MPFR_RNDD);
  if (! mpfr_underflow_p ())
    underflow = 0;
  if (mpfr_overflow_p ())
    overflow = 1;
  mpfr_clear_flags ();
  mpfr_exp (yu, yu, MPFR_RNDU);
  if (mpfr_underflow_p ())
    underflow = 1;
  if (! mpfr_overflow_p ())
    overflow = 0;
  if (sign < 0)
    {
      mpfr_neg (yd, yd, MPFR_RNDN);  /* exact */
      mpfr_neg (yu, yu, MPFR_RNDN);  /* exact */
      mpfr_swap (yd, yu);
    }
  /* yd < Gamma(x) < yu (strict inequalities since x != 1 and x != 2) */
  mpfr_inits2 (p1, zd, zu, (mpfr_ptr) 0);
  mpfr_clear_flags ();
  inexd = mpfr_gamma (zd, x, MPFR_RNDD);  /* zd <= Gamma(x) < yu */
  got_underflow = underflow == -1 ? -1 : !! mpfr_underflow_p ();
  got_overflow = overflow == -1 ? -1 : !! mpfr_overflow_p ();
  if (! mpfr_less_p (zd, yu) || inexd > 0 ||
      got_underflow != underflow ||
      got_overflow != overflow)
    {
      printf ("Error in exp_lgamma on x = ");
      mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
      printf ("yu = ");
      mpfr_dump (yu);
      printf ("zd = ");
      mpfr_dump (zd);
      printf ("got inexd = %d, expected <= 0\n", inexd);
      printf ("got underflow = %d, expected %d\n", got_underflow, underflow);
      printf ("got overflow = %d, expected %d\n", got_overflow, overflow);
      exit (1);
    }
  mpfr_clear_flags ();
  inexu = mpfr_gamma (zu, x, MPFR_RNDU);  /* zu >= Gamma(x) > yd */
  got_underflow = underflow == -1 ? -1 : !! mpfr_underflow_p ();
  got_overflow = overflow == -1 ? -1 : !! mpfr_overflow_p ();
  if (! mpfr_greater_p (zu, yd) || inexu < 0 ||
      got_underflow != underflow ||
      got_overflow != overflow)
    {
      printf ("Error in exp_lgamma on x = ");
      mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
      printf ("yd = ");
      mpfr_dump (yd);
      printf ("zu = ");
      mpfr_dump (zu);
      printf ("got inexu = %d, expected >= 0\n", inexu);
      printf ("got underflow = %d, expected %d\n", got_underflow, underflow);
      printf ("got overflow = %d, expected %d\n", got_overflow, overflow);
      exit (1);
    }
  if (mpfr_equal_p (zd, zu))
    {
      if (inexd != 0 || inexu != 0)
        {
          printf ("Error in exp_lgamma on x = ");
          mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
          printf ("zd = zu, thus exact, but inexd = %d and inexu = %d\n",
                  inexd, inexu);
          exit (1);
        }
      MPFR_ASSERTN (got_underflow == 0);
      MPFR_ASSERTN (got_overflow == 0);
    }
  else if (inexd == 0 || inexu == 0)
    {
      printf ("Error in exp_lgamma on x = ");
          mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
          printf ("zd != zu, thus inexact, but inexd = %d and inexu = %d\n",
                  inexd, inexu);
          exit (1);
    }
  mpfr_clears (yd, yu, zd, zu, (mpfr_ptr) 0);
}

static void
exp_lgamma_tests (void)
{
  mpfr_t x;
  mpfr_exp_t emin, emax;
  int i;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  set_emin (MPFR_EMIN_MIN);
  set_emax (MPFR_EMAX_MAX);

  mpfr_init2 (x, 96);
  for (i = 3; i <= 8; i++)
    {
      mpfr_set_ui (x, i, MPFR_RNDN);
      exp_lgamma (x, 53, 64);
      mpfr_nextbelow (x);
      exp_lgamma (x, 53, 64);
      mpfr_nextabove (x);
      mpfr_nextabove (x);
      exp_lgamma (x, 53, 64);
    }
  mpfr_set_str (x, "1.7", 10, MPFR_RNDN);
  exp_lgamma (x, 53, 64);
  mpfr_set_str (x, "-4.6308260837372266e+07", 10, MPFR_RNDN);
  exp_lgamma (x, 53, 64);
  mpfr_set_str (x, "-90.6308260837372266e+15", 10, MPFR_RNDN);
  exp_lgamma (x, 53, 64);
  /* The following test gives a large positive result < +Inf */
  mpfr_set_str (x, "1.2b13fc45a92dea1@14", 16, MPFR_RNDN);
  exp_lgamma (x, 53, 64);
  /* Idem for a large negative result > -Inf */
  mpfr_set_str (x, "-1.2b13fc45a92de81@14", 16, MPFR_RNDN);
  exp_lgamma (x, 53, 64);
  /* The following two tests trigger an endless loop in r8186
     on 64-bit machines (64-bit exponent). The second one (due
     to undetected overflow) is a direct consequence of the
     first one, due to the call of Gamma(2-x) if x < 1. */
  mpfr_set_str (x, "1.2b13fc45a92dec8@14", 16, MPFR_RNDN);
  exp_lgamma (x, 53, 64);
  mpfr_set_str (x, "-1.2b13fc45a92dea8@14", 16, MPFR_RNDN);
  exp_lgamma (x, 53, 64);
  /* Similar tests (overflow threshold) for 32-bit machines. */
  mpfr_set_str (x, "2ab68d8.657542f855111c61", 16, MPFR_RNDN);
  exp_lgamma (x, 12, 64);
  mpfr_set_str (x, "-2ab68d6.657542f855111c61", 16, MPFR_RNDN);
  exp_lgamma (x, 12, 64);
  /* The following test is an overflow on 32-bit and 64-bit machines.
     Revision r8189 fails on 64-bit machines as the flag is unset. */
  mpfr_set_str (x, "1.2b13fc45a92ded8@14", 16, MPFR_RNDN);
  exp_lgamma (x, 53, 64);
  /* On the following tests, with r8196, one gets an underflow on
     32-bit machines, while a normal result is expected (see FIXME
     in gamma.c:382). */
  mpfr_set_str (x, "-2ab68d6.657542f855111c6104", 16, MPFR_RNDN);
  exp_lgamma (x, 12, 64);  /* failure on 32-bit machines */
  mpfr_set_str (x, "-12b13fc45a92deb.1c6c5bc964", 16, MPFR_RNDN);
  exp_lgamma (x, 12, 64);  /* failure on 64-bit machines */
  mpfr_clear (x);

  set_emin (emin);
  set_emax (emax);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  special ();
  special_overflow ();
  exprange ();
  tiny (argc == 1);
  test_generic (2, 100, 2);
  gamma_integer ();
  test20071231 ();
  test20100709 ();
  test20120426 ();
  exp_lgamma_tests ();

  data_check ("data/gamma", mpfr_gamma, "mpfr_gamma");

  tests_end_mpfr ();
  return 0;
}
