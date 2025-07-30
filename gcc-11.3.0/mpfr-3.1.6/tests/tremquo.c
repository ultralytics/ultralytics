/* tremquo -- test file for mpfr_remquo and mpfr_remainder

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

#include "mpfr-test.h"

static void
bug20090227 (void)
{
  mpfr_t x, y, r1, r2;

  mpfr_init2 (x, 118);
  mpfr_init2 (y, 181);
  mpfr_init2 (r1, 140);
  mpfr_init2 (r2, 140);
  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_set_str_binary (y, "1.100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000000011011100000111001101000100101001000000100100111000001000100010100110011111010");
  mpfr_remainder (r1, x, y, MPFR_RNDU);
  /* since the quotient is -1, r1 is the rounding of x+y */
  mpfr_add (r2, x, y, MPFR_RNDU);
  if (mpfr_cmp (r1, r2))
    {
      printf ("Error in mpfr_remainder (bug20090227)\n");
      printf ("Expected ");
      mpfr_dump (r2);
      printf ("Got      ");
      mpfr_dump (r1);
      exit (1);
    }
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (r1);
  mpfr_clear (r2);
}

int
main (int argc, char *argv[])
{
  mpfr_t x, y, r;
  long q[1];
  int inex;

  if (argc == 3) /* usage: tremquo x y (rnd=MPFR_RNDN implicit) */
    {
      mpfr_init2 (x, GMP_NUMB_BITS);
      mpfr_init2 (y, GMP_NUMB_BITS);
      mpfr_init2 (r, GMP_NUMB_BITS);
      mpfr_set_str (x, argv[1], 10, MPFR_RNDN);
      mpfr_set_str (y, argv[2], 10, MPFR_RNDN);
      mpfr_remquo (r, q, x, y, MPFR_RNDN);
      printf ("r=");
      mpfr_out_str (stdout, 10, 0, r, MPFR_RNDN);
      printf (" q=%ld\n", q[0]);
      mpfr_clear (x);
      mpfr_clear (y);
      mpfr_clear (r);
      return 0;
    }

  tests_start_mpfr ();

  bug20090227 ();

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (r);

  /* special values */
  mpfr_set_nan (x);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (r));

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_nan (y);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (r));

  mpfr_set_inf (x, 1); /* +Inf */
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (r));

  mpfr_set_inf (x, 1); /* +Inf */
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (r));

  mpfr_set_inf (x, 1); /* +Inf */
  mpfr_set_inf (y, 1);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (r));

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_set_inf (y, 1);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (r, 0) == 0 && MPFR_IS_POS (r));
  MPFR_ASSERTN (q[0] == (long) 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN); /* -0 */
  mpfr_set_inf (y, 1);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (r, 0) == 0 && MPFR_IS_NEG (r));
  MPFR_ASSERTN (q[0] == (long) 0);

  mpfr_set_ui (x, 17, MPFR_RNDN);
  mpfr_set_inf (y, 1);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp (r, x) == 0);
  MPFR_ASSERTN (q[0] == (long) 0);

  mpfr_set_ui (x, 17, MPFR_RNDN);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (r));

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_set_ui (y, 17, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (r, 0) == 0 && MPFR_IS_POS (r));
  MPFR_ASSERTN (q[0] == (long) 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_set_ui (y, 17, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (r, 0) == 0 && MPFR_IS_NEG (r));
  MPFR_ASSERTN (q[0] == (long) 0);

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

  /* check four possible sign combinations */
  mpfr_set_ui (x, 42, MPFR_RNDN);
  mpfr_set_ui (y, 17, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (r, 8) == 0);
  MPFR_ASSERTN (q[0] == (long) 2);
  mpfr_set_si (x, -42, MPFR_RNDN);
  mpfr_set_ui (y, 17, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_si (r, -8) == 0);
  MPFR_ASSERTN (q[0] == (long) -2);
  mpfr_set_si (x, -42, MPFR_RNDN);
  mpfr_set_si (y, -17, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_si (r, -8) == 0);
  MPFR_ASSERTN (q[0] == (long) 2);
  mpfr_set_ui (x, 42, MPFR_RNDN);
  mpfr_set_si (y, -17, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (r, 8) == 0);
  MPFR_ASSERTN (q[0] == (long) -2);

  mpfr_set_prec (x, 100);
  mpfr_set_prec (y, 50);
  mpfr_set_ui (x, 42, MPFR_RNDN);
  mpfr_nextabove (x); /* 42 + 2^(-94) */
  mpfr_set_ui (y, 21, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui_2exp (r, 1, -94) == 0);
  MPFR_ASSERTN (q[0] == (long) 2);

  mpfr_set_prec (x, 50);
  mpfr_set_prec (y, 100);
  mpfr_set_ui (x, 42, MPFR_RNDN);
  mpfr_nextabove (x); /* 42 + 2^(-44) */
  mpfr_set_ui (y, 21, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui_2exp (r, 1, -44) == 0);
  MPFR_ASSERTN (q[0] == (long) 2);

  mpfr_set_prec (x, 100);
  mpfr_set_prec (y, 50);
  mpfr_set_ui (x, 42, MPFR_RNDN);
  mpfr_set_ui (y, 21, MPFR_RNDN);
  mpfr_nextabove (y); /* 21 + 2^(-45) */
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  /* r should be 42 - 2*(21 + 2^(-45)) = -2^(-44) */
  MPFR_ASSERTN (mpfr_cmp_si_2exp (r, -1, -44) == 0);
  MPFR_ASSERTN (q[0] == (long) 2);

  mpfr_set_prec (x, 50);
  mpfr_set_prec (y, 100);
  mpfr_set_ui (x, 42, MPFR_RNDN);
  mpfr_set_ui (y, 21, MPFR_RNDN);
  mpfr_nextabove (y); /* 21 + 2^(-95) */
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  /* r should be 42 - 2*(21 + 2^(-95)) = -2^(-94) */
  MPFR_ASSERTN (mpfr_cmp_si_2exp (r, -1, -94) == 0);
  MPFR_ASSERTN (q[0] == (long) 2);

  /* exercise large quotient */
  mpfr_set_ui_2exp (x, 1, 65, MPFR_RNDN);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  /* quotient is 2^65 */
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_si (r, 0) == 0);
  MPFR_ASSERTN (q[0] % 1073741824L == 0L);

  /* another large quotient */
  mpfr_set_prec (x, 65);
  mpfr_set_prec (y, 65);
  mpfr_const_pi (x, MPFR_RNDN);
  mpfr_mul_2exp (x, x, 63, MPFR_RNDN);
  mpfr_const_log2 (y, MPFR_RNDN);
  mpfr_set_prec (r, 10);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  /* q should be 41803643793084085130, r should be 605/2048 */
  MPFR_ASSERTN (mpfr_cmp_ui_2exp (r, 605, -11) == 0);
  MPFR_ASSERTN ((q[0] > 0) && ((q[0] % 1073741824L) == 733836170L));

  /* check cases where quotient is 1.5 +/- eps */
  mpfr_set_prec (x, 65);
  mpfr_set_prec (y, 65);
  mpfr_set_prec (r, 63);
  mpfr_set_ui (x, 3, MPFR_RNDN);
  mpfr_set_ui (y, 2, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  /* x/y = 1.5, quotient should be 2 (even rule), remainder should be -1 */
  MPFR_ASSERTN (mpfr_cmp_si (r, -1) == 0);
  MPFR_ASSERTN (q[0] == 2L);
  mpfr_set_ui (x, 3, MPFR_RNDN);
  mpfr_nextabove (x); /* 3 + 2^(-63) */
  mpfr_set_ui (y, 2, MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  /* x/y = 1.5 + 2^(-64), quo should be 2, r should be -1 + 2^(-63) */
  MPFR_ASSERTN (mpfr_add_ui (r, r, 1, MPFR_RNDN) == 0);
  MPFR_ASSERTN (mpfr_cmp_ui_2exp (r, 1, -63) == 0);
  MPFR_ASSERTN (q[0] == 2L);
  mpfr_set_ui (x, 3, MPFR_RNDN);
  mpfr_set_ui (y, 2, MPFR_RNDN);
  mpfr_nextabove (y); /* 2 + 2^(-63) */
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  /* x/y = 1.5 - eps, quo should be 1, r should be 1 - 2^(-63) */
  MPFR_ASSERTN (mpfr_sub_ui (r, r, 1, MPFR_RNDN) == 0);
  MPFR_ASSERTN (mpfr_cmp_si_2exp (r, -1, -63) == 0);
  MPFR_ASSERTN (q[0] == 1L);

  /* bug founds by Kaveh Ghazi, 3 May 2007 */
  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_set_ui (y, 3, MPFR_RNDN);
  mpfr_remainder (r, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_si (r, -1) == 0);

  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_remainder (r, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_si (r, 0) == 0 && MPFR_SIGN (r) < 0);

  /* check argument reuse */
  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_remainder (x, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_si (x, 0) == 0 && MPFR_SIGN (x) < 0);

  mpfr_set_ui_2exp (x, 1, mpfr_get_emax () - 1, MPFR_RNDN);
  mpfr_set_ui_2exp (y, 1, mpfr_get_emin (), MPFR_RNDN);
  mpfr_remquo (r, q, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_zero_p (r) && MPFR_SIGN (r) > 0);
  MPFR_ASSERTN (q[0] == 0);

  mpfr_set_prec (x, 380);
  mpfr_set_prec (y, 385);
  mpfr_set_str_binary (x, "0.11011010010110011101011000100100101100101011010001011100110001100101111001010100001011111110111100101110101010110011010101000100000100011101101100001011101110100111101111111010001001000010000110010110011100111000001110111010000100101001010111100100010001101001110100011110010000000001110001111001101100111011001000110110011100100011111110010100011001000001001011010111010000000000E-2");
  mpfr_set_str_binary (y, "0.1100011000011101011010001100010111001110110111001101010010111100111100011010010011011101111101111001010111111110001001100001111101001000000010100101111001001110010110000111001000101010111001001000100101011111000010100110001111000110011011010101111101100110010101011010011101100001011101001000101111110110110110000001001101110111110110111110111111001001011110001110011111100000000000000E-1");
  mpfr_set_prec (r, 2);
  inex = mpfr_remainder (r, x, y, MPFR_RNDA);
  MPFR_ASSERTN(mpfr_cmp_si_2exp (r, -3, -4) == 0);
  MPFR_ASSERTN(inex < 0);

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (r);

  tests_end_mpfr ();

  return 0;
}
