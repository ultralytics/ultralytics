/* Test file for mpfr_atan.

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

static void
special (void)
{
  mpfr_t x, y, z;
  int r;
  int i;

  mpfr_init2 (x, 53);
  mpfr_init2 (y, 53);
  mpfr_init2 (z, 53);

  mpfr_set_str_binary (x, "1.0000100110000001100111100011001110101110100111011101");
  mpfr_set_str_binary (y, "1.1001101101110100101100110011011101101000011010111110e-1");
  mpfr_atan (z, x, MPFR_RNDN);
  if (mpfr_cmp (y, z))
    {
      printf ("Error in mpfr_atan for prec=53, rnd=MPFR_RNDN\n");
      printf ("x=");
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
      printf ("\nexpected ");
      mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
      printf ("\ngot      ");
      mpfr_out_str (stdout, 2, 0, z, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  /* atan(+Inf) = Pi/2 */
  for (r = 0; r < MPFR_RND_MAX ; r++)
    {
      mpfr_set_inf (x, 1);
      mpfr_atan (y, x, (mpfr_rnd_t) r);
      mpfr_const_pi (x, (mpfr_rnd_t) r);
      mpfr_div_2exp (x, x, 1, (mpfr_rnd_t) r);
      if (mpfr_cmp (x, y))
        {
          printf ("Error: mpfr_atan(+Inf), rnd=%s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit (1);
        }
    }

  /* atan(-Inf) = - Pi/2 */
  for (r = 0; r < MPFR_RND_MAX ; r++)
    {
      mpfr_set_inf (x, -1);
      mpfr_atan (y, x, (mpfr_rnd_t) r);
      mpfr_const_pi (x, MPFR_INVERT_RND((mpfr_rnd_t) r));
      mpfr_neg (x, x, (mpfr_rnd_t) r);
      mpfr_div_2exp (x, x, 1, (mpfr_rnd_t) r);
      if (mpfr_cmp (x, y))
        {
          printf ("Error: mpfr_atan(-Inf), rnd=%s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit (1);
        }
    }

  /* atan(NaN) = NaN */
  mpfr_set_nan (x);
  mpfr_atan (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error: mpfr_atan(NaN) <> NaN\n");
      exit (1);
    }

  /* atan(+/-0) = +/-0 */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  MPFR_SET_NEG (y);
  mpfr_atan (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || MPFR_IS_NEG (y))
    {
      printf ("Error: mpfr_atan (+0) <> +0\n");
      exit (1);
    }
  mpfr_atan (x, x, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 0) || MPFR_IS_NEG (x))
    {
      printf ("Error: mpfr_atan (+0) <> +0 (in place)\n");
      exit (1);
    }
  mpfr_neg (x, x, MPFR_RNDN);
  MPFR_SET_POS (y);
  mpfr_atan (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || MPFR_IS_POS (y))
    {
      printf ("Error: mpfr_atan (-0) <> -0\n");
      exit (1);
    }
  mpfr_atan (x, x, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 0) || MPFR_IS_POS (x))
    {
      printf ("Error: mpfr_atan (-0) <> -0 (in place)\n");
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_set_prec (y, 32);

  /* test one random positive argument */
  mpfr_set_str_binary (x, "0.10000100001100101001001001011001");
  mpfr_atan (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "0.1111010000001111001111000000011E-1");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_atan (1)\n");
      exit (1);
    }

  /* test one random negative argument */
  mpfr_set_str_binary (x, "-0.1100001110110000010101011001011");
  mpfr_atan (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "-0.101001110001010010110001110001");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_atan (2)\n");
      mpfr_print_binary (x); printf ("\n");
      mpfr_print_binary (y); printf ("\n");
      exit (1);
    }

  mpfr_set_prec (x, 3);
  mpfr_set_prec (y, 192);
  mpfr_set_prec (z, 192);
  mpfr_set_str_binary (x, "-0.100e1");
  mpfr_atan (z, x, MPFR_RNDD);
  mpfr_set_str_binary (y, "-0.110010010000111111011010101000100010000101101000110000100011010011000100110001100110001010001011100000001101110000011100110100010010100100000010010011100000100010001010011001111100110001110101");
  if (mpfr_cmp (z, y))
    {
      printf ("Error in mpfr_atan (3)\n");
      printf ("Expected "); mpfr_print_binary (y); printf ("\n");
      printf ("Got      "); mpfr_print_binary (z); printf ("\n");
      exit (1);
    }

  /* Test regression */
  mpfr_set_prec (x, 51);
  mpfr_set_prec (y, 51);
  mpfr_set_str_binary (x,
           "0.101100100000101111111010001111111000001000000000000E-11");
  i = mpfr_atan (y, x, MPFR_RNDN);
  if (mpfr_cmp_str (y,
   "1.01100100000101111111001110011001010110100100000000e-12", 2, MPFR_RNDN)
      || i >= 0)
    {
      printf ("Wrong Regression test (%d)\n", i);
      mpfr_dump (y);
      exit (1);
    }

  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_atan (x, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_NEG (x));

  /* Test regression */
  mpfr_set_prec (x, 48);
  mpfr_set_prec (y, 48);
  mpfr_set_str_binary (x, "1.11001110010000011111100000010000000000000000000e-19");
  mpfr_atan (y, x, MPFR_RNDD);
  if (mpfr_cmp_str (y, "0.111001110010000011111100000001111111110000010011E-18", 2, MPFR_RNDN))
    {
      printf ("Error in mpfr_atan (4)\n");
      printf ("Input    1.11001110010000011111100000010000000000000000000e-19 [prec=48]\n");
      printf ("Expected 0.111001110010000011111100000001111111110000010011E-18\n");
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

#define TEST_FUNCTION mpfr_atan
#define test_generic test_generic_atan
#define RAND_FUNCTION(x) (mpfr_urandomb (x, RANDS), mpfr_mul_2si (x, x, (randlimb () %1000-500), MPFR_RNDN))
#include "tgeneric.c"

#define TEST_FUNCTION mpfr_atan2
#define TWO_ARGS
#define test_generic test_generic_atan2
#include "tgeneric.c"

#define TEST_FUNCTION mpfr_atan2
#define TWO_ARGS
#define RAND_FUNCTION(x) (mpfr_urandomb (x, RANDS), MPFR_SET_NEG (x))
#define test_generic test_generic_atan2_neg
#include "tgeneric.c"

static void
special_overflow (void)
{
  mpfr_t x, y;
  mpfr_exp_t emin, emax;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  set_emin (-125);
  set_emax (128);
  mpfr_init2 (x, 24);
  mpfr_init2 (y, 48);
  mpfr_set_str_binary (x, "0.101101010001001101111010E0");
  mpfr_atan (y, x, MPFR_RNDN);
  if (mpfr_cmp_str (y, "0.100111011001100111000010111101000111010101011110E0",
                    2, MPFR_RNDN))
    {
      printf("Special Overflow error.\n");
      mpfr_dump (y);
      exit (1);
    }

  /* intermediate Pi overflows while atan(+Inf) = Pi/2 is representable */
  set_emax (1);
  mpfr_set_inf (x, +1);
  mpfr_clear_flags ();
  mpfr_atan (y, x, MPFR_RNDN);
  if (mpfr_cmp_str (y, "C90FDAA22169p-47", 16, MPFR_RNDN)
      || mpfr_overflow_p ())
    {
      printf("atan(+Inf) = Pi/2 should not overflow when emax = %ld\n",
             (long int) mpfr_get_emax ());
      mpfr_dump (y);
      exit (1);
    }

  /* atan(+Inf) = Pi/2 underflows */
  set_emax (128);
  set_emin (3);
  mpfr_clear_flags ();
  mpfr_atan (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || !mpfr_underflow_p ())
    {
      printf("atan(+Inf) = Pi/2 should underflow when emin = %ld\n",
             (long int) mpfr_get_emin ());
      mpfr_dump (y);
      exit (1);
    }

  /* intermediate Pi overflows while atan(+1) = Pi/4 is representable */
  set_emax (1);
  set_emin (-128);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_clear_flags ();
  mpfr_atan (y, x, MPFR_RNDN);
  if (mpfr_cmp_str (y, "C90FDAA22169p-48", 16, MPFR_RNDN)
      || mpfr_overflow_p ())
    {
      printf("atan(+1) = Pi/4 should not overflow when emax = %ld\n",
             (long int) mpfr_get_emax ());
      mpfr_dump (y);
      exit (1);
    }

  /* atan(+1) = Pi/4 underflows and is rounded up to 1 */
  set_emax (128);
  set_emin (1);
  mpfr_set_prec (y, 2);
  mpfr_clear_flags ();
  mpfr_atan (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 1) || !mpfr_underflow_p ())
    {
      printf("atan(+1) = Pi/4 should underflow when emin = %+ld\n",
             (long int) mpfr_get_emin ());
      mpfr_dump (y);
      exit (1);
    }

  /* atan(+1) = Pi/4 underflows and is rounded down to 0 */
  mpfr_clear_flags ();
  mpfr_atan (y, x, MPFR_RNDD);
  if (mpfr_cmp_ui (y, 0) || !mpfr_underflow_p ())
    {
      printf("atan(+1) = Pi/4 should underflow when emin = %+ld\n",
             (long int) mpfr_get_emin ());
      mpfr_dump (y);
      exit (1);
    }

  mpfr_clear (y);
  mpfr_clear (x);
  set_emin (emin);
  set_emax (emax);
}

static void
special_atan2 (void)
{
  mpfr_t x, y, z;

  mpfr_inits2 (4, x, y, z, (mpfr_ptr) 0);

  /* Anything with NAN should be set to NAN */
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_set_nan (x);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_NAN (z));
  mpfr_swap (x, y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_NAN (z));

  /* 0+ 0+ --> 0+ */
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO (z) && MPFR_IS_POS (z));
  /* 0- 0+ --> 0- */
  MPFR_CHANGE_SIGN (y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO (z) && MPFR_IS_NEG (z));
  /* 0- 0- --> -PI */
  MPFR_CHANGE_SIGN (x);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "-3.1415", 10, MPFR_RNDN) == 0);
  /* 0+ 0- --> +PI */
  MPFR_CHANGE_SIGN (y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "3.1415", 10, MPFR_RNDN) == 0);
  /* 0+ -1 --> PI */
  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "3.1415", 10, MPFR_RNDN) == 0);
  /* 0- -1 --> -PI */
  MPFR_CHANGE_SIGN (y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "-3.1415", 10, MPFR_RNDN) == 0);
  /* 0- +1 --> 0- */
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO (z) && MPFR_IS_NEG (z));
  /* 0+ +1 --> 0+ */
  MPFR_CHANGE_SIGN (y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO (z) && MPFR_IS_POS (z));
  /* +1 0+ --> PI/2 */
  mpfr_swap (x, y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "1.57075", 10, MPFR_RNDN) == 0);
  /* +1 0- --> PI/2 */
  MPFR_CHANGE_SIGN (x);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "1.57075", 10, MPFR_RNDN) == 0);
  /* -1 0- --> -PI/2 */
  MPFR_CHANGE_SIGN (y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "-1.57075", 10, MPFR_RNDN) == 0);
  /* -1 0+ --> -PI/2 */
  MPFR_CHANGE_SIGN (x);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "-1.57075", 10, MPFR_RNDN) == 0);

  /* -1 +INF --> -0 */
  MPFR_SET_INF (x);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO (z) && MPFR_IS_NEG (z));
  /* +1 +INF --> +0 */
  MPFR_CHANGE_SIGN (y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO (z) && MPFR_IS_POS (z));
  /* +1 -INF --> +PI */
  MPFR_CHANGE_SIGN (x);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "3.1415", 10, MPFR_RNDN) == 0);
  /* -1 -INF --> -PI */
  MPFR_CHANGE_SIGN (y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "-3.1415", 10, MPFR_RNDN) == 0);
  /* -INF -1 --> -PI/2 */
  mpfr_swap (x, y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "-1.57075", 10, MPFR_RNDN) == 0);
  /* +INF -1  --> PI/2 */
  MPFR_CHANGE_SIGN (y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "1.57075", 10, MPFR_RNDN) == 0);
  /* +INF -INF --> 3*PI/4 */
  MPFR_SET_INF (x);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "2.356194490192344928", 10, MPFR_RNDN) == 0);
  /* +INF +INF --> PI/4 */
  MPFR_CHANGE_SIGN (x);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "0.785375", 10, MPFR_RNDN) == 0);
  /* -INF +INF --> -PI/4 */
  MPFR_CHANGE_SIGN (y);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "-0.785375", 10, MPFR_RNDN) == 0);
  /* -INF -INF --> -3*PI/4 */
  MPFR_CHANGE_SIGN (x);
  mpfr_atan2 (z, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_str (z, "-2.356194490192344928", 10, MPFR_RNDN) == 0);
  mpfr_set_prec (z, 905); /* exercises Ziv's loop */
  mpfr_atan2 (z, y, x, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp_str (z, "-2.35619449019234492884698253745962716314787704953132936573120844423086230471465674897102611900658780098661106488496172998532038345716293667379401955609636083808771307702645389082916973346721171619778647332160823174945008459635673617534008737395340143185923642519259526145784", 10, MPFR_RNDN) == 0);

  mpfr_clears (x, y, z, (mpfr_ptr) 0);
}

/* from Christopher Creutzig, 18 Jul 2007 */
static void
smallvals_atan2 (void)
{
  mpfr_t a, x, y;
  mpfr_exp_t old_emin;

  mpfr_inits (a, x, y, (mpfr_ptr) 0);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_nextbelow (y);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  /* y=-2^(-emin-1), x=1 */

  mpfr_atan2 (a, y, x, MPFR_RNDD);
  MPFR_ASSERTN (mpfr_equal_p (a, y));

  mpfr_atan2 (a, y, x, MPFR_RNDU);
  MPFR_ASSERTN (mpfr_zero_p (a) && MPFR_IS_NEG(a));

  mpfr_set_prec (x, 8);
  mpfr_set_prec (y, 8);
  mpfr_set_prec (a, 8);
  old_emin = mpfr_get_emin ();
  mpfr_set_emin (MPFR_EMIN_MIN);

  mpfr_set_si (y, 3, MPFR_RNDN);
  mpfr_set_exp (y, mpfr_get_emin ());
  mpfr_set_str_binary (x, "1.1");
  mpfr_atan2 (a, y, x, MPFR_RNDU);
  mpfr_set_si (y, 1, MPFR_RNDN);
  mpfr_set_exp (y, mpfr_get_emin ());
  MPFR_ASSERTN (mpfr_equal_p (a, y));

  /* From a bug reported by Christopher Creutzig on 2007-08-28.
     Added test in each rounding mode.
     Segmentation fault or assertion failure due to an infinite Ziv loop. */
  mpfr_set_si (y, 1, MPFR_RNDN);
  mpfr_set_exp (y, mpfr_get_emin ());
  mpfr_set_str_binary (x, "1.01");
  mpfr_atan2 (a, y, x, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_zero_p (a));
  mpfr_atan2 (a, y, x, MPFR_RNDD);
  MPFR_ASSERTN (mpfr_zero_p (a));
  mpfr_atan2 (a, y, x, MPFR_RNDU);
  MPFR_ASSERTN (mpfr_equal_p (a, y));
  mpfr_atan2 (a, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_equal_p (a, y));

  /* trigger underflow with rounding to nearest */
  mpfr_set_ui (x, 4, MPFR_RNDN);
  mpfr_atan2 (a, y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_zero_p (a));

  mpfr_set_emin (old_emin);

  mpfr_clears (a, x, y, (mpfr_ptr) 0);
}

/* Bug found by Robert Bajema (regression in MPFR 2.3.0).
   The cause is the underflow flag set before the mpfr_atan2 call. */
static void
atan2_bug_20071003 (void)
{
  mpfr_t a, x, y, z;

  mpfr_inits (a, x, y, z, (mpfr_ptr) 0);

  mpfr_set_underflow ();
  mpfr_set_str_binary (y,
    "-0.10100110110100110111010110111111100110100010001110110E2");
  mpfr_set_str_binary (x,
    "0.10100101010110010100010010111000110110011110001011110E3");
  mpfr_set_str_binary (z,
    "-0.11101111001101101100111011001101000010010111101110110E-1");
  mpfr_atan2 (a, y, x, MPFR_RNDN);
  if (! mpfr_equal_p (a, z))
    {
      printf ("mpfr_atan2 fails on:\n");
      printf ("  y = ");
      mpfr_dump (y);
      printf ("  x = ");
      mpfr_dump (x);
      printf ("Expected ");
      mpfr_dump (z);
      printf ("Got      ");
      mpfr_dump (a);
      exit (1);
    }

  mpfr_clears (a, x, y, z, (mpfr_ptr) 0);
}

/* Bug found on 2009-04-29 by Christopher Creutzig.
 * With r6179: atan.c:62: MPFR assertion failed: r > n
 */
static void
atan2_different_prec (void)
{
  mpfr_t a, x, y;

  mpfr_init2 (a, 59);
  mpfr_init2 (x, 59);
  mpfr_init2 (y, 86);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_nextbelow (y);
  mpfr_atan2 (a, y, x, MPFR_RNDN);

  mpfr_clears (a, x, y, (mpfr_ptr) 0);
}

static void
atan2_pow_of_2 (void)
{
  mpfr_t x, y, r, g;
  int i;
  int d[] = { 0, -1, 1 };
  int ntests = sizeof (d) / sizeof (int);

  mpfr_init2 (x, 53);
  mpfr_init2 (y, 53);
  mpfr_init2 (r, 53);
  mpfr_init2 (g, 53);

  /* atan(42) */
  mpfr_set_str_binary (g, "1100011000000011110011111001100110101000011010010011E-51");

  for (i = 0; i < ntests; ++i)
    {
      mpfr_set_ui (y, 42, MPFR_RNDN);
      mpfr_mul_2si (y, y, d[i], MPFR_RNDN);
      mpfr_set_ui_2exp (x, 1, d[i], MPFR_RNDN);
      mpfr_atan2 (r, y, x, MPFR_RNDN);
      if (mpfr_equal_p (r, g) == 0)
        {
          printf ("Error in mpfr_atan2 (5)\n");
          printf ("Expected "); mpfr_print_binary (g); printf ("\n");
          printf ("Got      "); mpfr_print_binary (r); printf ("\n");
          exit (1);
        }
    }
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (r);
  mpfr_clear (g);
}

/* https://sympa.inria.fr/sympa/arc/mpfr/2011-05/msg00008.html
 * Incorrect flags (in debug mode on a 32-bit machine, assertion failure).
 */
static void
reduced_expo_range (void)
{
  mpfr_exp_t emin, emax;
  mpfr_t x, y, ex_y;
  int inex, ex_inex;
  unsigned int flags, ex_flags;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  mpfr_inits2 (12, x, y, ex_y, (mpfr_ptr) 0);
  mpfr_set_str (x, "0.1e-5", 2, MPFR_RNDN);

  mpfr_set_emin (-5);
  mpfr_set_emax (-5);
  mpfr_clear_flags ();
  inex = mpfr_atan (y, x, MPFR_RNDN);
  flags = __gmpfr_flags;
  mpfr_set_emin (emin);
  mpfr_set_emax (emax);

  mpfr_set_str (ex_y, "0.1e-5", 2, MPFR_RNDN);
  ex_inex = 1;
  ex_flags = MPFR_FLAGS_INEXACT;

  if (SIGN (inex) != ex_inex || flags != ex_flags ||
      ! mpfr_equal_p (y, ex_y))
    {
      printf ("Error in reduced_expo_range\non x = ");
      mpfr_dump (x);
      printf ("Expected y = ");
      mpfr_out_str (stdout, 2, 0, ex_y, MPFR_RNDN);
      printf ("\n         inex = %d, flags = %u\n", ex_inex, ex_flags);
      printf ("Got      y = ");
      mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
      printf ("\n         inex = %d, flags = %u\n", SIGN (inex), flags);
      exit (1);
    }

  mpfr_clears (x, y, ex_y, (mpfr_ptr) 0);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  special_overflow ();
  special ();
  special_atan2 ();
  smallvals_atan2 ();
  atan2_bug_20071003 ();
  atan2_different_prec ();
  reduced_expo_range ();

  test_generic_atan  (2, 200, 17);
  test_generic_atan2 (2, 200, 17);
  test_generic_atan2_neg (2, 200, 17);

  data_check ("data/atan", mpfr_atan, "mpfr_atan");
  bad_cases (mpfr_atan, mpfr_tan, "mpfr_atan", 256, -40, 1, 4, 128, 800, 40);
  atan2_pow_of_2 ();

  tests_end_mpfr ();
  return 0;
}
