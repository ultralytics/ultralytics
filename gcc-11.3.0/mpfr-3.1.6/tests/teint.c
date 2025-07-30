/* Test file for mpfr_eint.

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

#define TEST_FUNCTION mpfr_eint
#define TEST_RANDOM_POS 8
#define TEST_RANDOM_EMAX 40
#include "tgeneric.c"

static void
check_specials (void)
{
  mpfr_t  x, y;

  mpfr_init2 (x, 123L);
  mpfr_init2 (y, 123L);

  mpfr_set_nan (x);
  mpfr_eint (y, x, MPFR_RNDN);
  if (! mpfr_nan_p (y))
    {
      printf ("Error: eint(NaN) != NaN\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_eint (y, x, MPFR_RNDN);
  if (! (mpfr_inf_p (y) && mpfr_sgn (y) > 0))
    {
      printf ("Error: eint(+Inf) != +Inf\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_eint (y, x, MPFR_RNDN);
  if (! mpfr_nan_p (y))
    {
      printf ("Error: eint(-Inf) != NaN\n");
      exit (1);
    }

  /* eint(+/-0) = -Inf */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_eint (y, x, MPFR_RNDN);
  if (! (mpfr_inf_p (y) && mpfr_sgn (y) < 0))
    {
      printf ("Error: eint(+0) != -Inf\n");
      exit (1);
    }
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_eint (y, x, MPFR_RNDN);
  if (! (mpfr_inf_p (y) && mpfr_sgn (y) < 0))
    {
      printf ("Error: eint(-0) != -Inf\n");
      exit (1);
    }

  /* eint(x) = NaN for x < 0 */
  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_eint (y, x, MPFR_RNDN);
  if (! mpfr_nan_p (y))
    {
      printf ("Error: eint(-1) != NaN\n");
      exit (1);
    }

  mpfr_set_prec (x, 17);
  mpfr_set_prec (y, 17);
  mpfr_set_str_binary (x, "1.0111110100100110e-2");
  mpfr_set_str_binary (y, "-1.0010101001110100e-10");
  mpfr_eint (x, x, MPFR_RNDZ);
  if (mpfr_cmp (x, y))
    {
      printf ("Error for x=1.0111110100100110e-2, MPFR_RNDZ\n");
      printf ("expected "); mpfr_dump (y);
      printf ("got      "); mpfr_dump (x);
      exit (1);
    }

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_str_binary (x, "0.10E4");
  mpfr_eint (x, x, MPFR_RNDN);
  mpfr_set_str (y, "440.37989953483827", 10, MPFR_RNDN);
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("Error for x=0.10E4, MPFR_RNDZ\n");
      printf ("expected "); mpfr_dump (y);
      printf ("got      "); mpfr_dump (x);
      exit (1);
    }

  mpfr_set_prec (x, 63);
  mpfr_set_prec (y, 63);
  mpfr_set_str_binary (x, "1.01111101011100111000011010001000101101011000011001111101011010e-2");
  mpfr_eint (x, x, MPFR_RNDZ);
  mpfr_set_str_binary (y, "1.11010110001101000001010010000100001111001000100100000001011100e-17");
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("Error (1) for MPFR_RNDZ\n");
      printf ("expected "); mpfr_dump (y);
      printf ("got      "); mpfr_dump (x);
      exit (1);
    }

  /* check large x */
  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_str_binary (x, "1E6");
  mpfr_eint (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "10100011110001101001110000110010111000100111010001E37");
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("Error for x=2^6, MPFR_RNDN\n");
      printf ("expected "); mpfr_dump (y);
      printf ("got      "); mpfr_dump (x);
      exit (1);
    }
  mpfr_set_str_binary (x, "1E7");
  mpfr_eint (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "11001100100011110000101001011010110111111011110011E128");
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("Error for x=2^7, MPFR_RNDN\n");
      printf ("expected "); mpfr_dump (y);
      printf ("got      "); mpfr_dump (x);
      exit (1);
    }
  mpfr_set_str_binary (x, "1E8");
  mpfr_eint (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "1010000110000101111111011011000101001000101011101001E310");
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("Error for x=2^8, MPFR_RNDN\n");
      printf ("expected "); mpfr_dump (y);
      printf ("got      "); mpfr_dump (x);
      exit (1);
    }
  mpfr_set_str_binary (x, "1E9");
  mpfr_eint (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "11001010101000001010101101110000010110011101110010101E677");
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("Error for x=2^9, MPFR_RNDN\n");
      printf ("expected "); mpfr_dump (y);
      printf ("got      "); mpfr_dump (x);
      exit (1);
    }
  mpfr_set_str_binary (x, "1E10");
  mpfr_eint (x, x, MPFR_RNDN);
  mpfr_set_str_binary (y, "10011111111010010110110101101000101100101010101101101E1415");
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("Error for x=2^10, MPFR_RNDN\n");
      printf ("expected "); mpfr_dump (y);
      printf ("got      "); mpfr_dump (x);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  if (argc != 1) /* teint x [prec] */
    {
      mpfr_t x;
      mpfr_prec_t p;
      p = (argc < 3) ? 53 : atoi (argv[2]);
      mpfr_init2 (x, p);
      mpfr_set_str (x, argv[1], 10, MPFR_RNDN);
      printf ("eint(");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      printf (")=");
      mpfr_eint (x, x, MPFR_RNDN);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      printf ("\n");
      mpfr_clear (x);
    }
  else
    {
      check_specials ();

      test_generic (2, 100, 100);
    }

  tests_end_mpfr ();
  return 0;
}
