/* tyn -- test file for the Bessel function of second kind

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

int
main (int argc, char *argv[])
{
  mpfr_t x, y;
  long n;
  mpfr_prec_t prec = 53;

  tests_start_mpfr ();

  mpfr_init (x);
  mpfr_init (y);

  if (argc != 1)
    {
      if (argc != 4)
        {
          printf ("Usage: tyn n x prec\n");
          exit (1);
        }
      n = atoi (argv[1]);
      prec = atoi (argv[3]);
      mpfr_set_prec (x, prec);
      mpfr_set_prec (y, prec);
      mpfr_set_str (x, argv[2], 10, MPFR_RNDN);
      mpfr_yn (y, n, x, MPFR_RNDN);
      printf ("Y(%ld,", n);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      printf (")=");
      mpfr_out_str (stdout, 10, 0, y, MPFR_RNDN);
      printf ("\n");
      goto end;
    }

  /* special values */
  mpfr_set_nan (x);
  mpfr_yn (y, 17, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (y));

  mpfr_set_inf (x, 1); /* +Inf */
  mpfr_yn (y, 17, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 0) == 0 && MPFR_IS_POS (y));

  mpfr_set_inf (x, -1); /* -Inf */
  mpfr_yn (y, 17, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (y));

  mpfr_set_ui (x, 0, MPFR_RNDN); /* +0 */
  mpfr_yn (y, 0, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && MPFR_IS_NEG (y)); /* y0(+0)=-Inf */
  mpfr_yn (y, 17, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && MPFR_IS_NEG (y)); /* y17(+0)=-Inf */
  mpfr_yn (y, -17, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && MPFR_IS_POS (y)); /* y(-17,+0)=+Inf */
  mpfr_yn (y, -42, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && MPFR_IS_NEG (y)); /* y(-42,+0)=-Inf */

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN); /* -0 */
  mpfr_yn (y, 0, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && MPFR_IS_NEG (y)); /* y0(-0)=-Inf */
  mpfr_yn (y, 17, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && MPFR_IS_NEG (y)); /* y17(-0)=-Inf */
  mpfr_yn (y, -17, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && MPFR_IS_POS (y)); /* y(-17,-0)=+Inf */
  mpfr_yn (y, -42, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && MPFR_IS_NEG (y)); /* y(-42,-0)=-Inf */

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_yn (y, 0, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.00010110100110000000001000100110111100110101100011011111");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_yn for n=0, x=1, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_yn (y, 1, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "-0.110001111111110110010000001111101011001101011100101");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_yn for n=1, x=1, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_yn (y, -1, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.110001111111110110010000001111101011001101011100101");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_yn for n=-1, x=1, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_yn (y, 2, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "-1.101001101001001100100010101001000101101000010010001");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_yn for n=2, x=1, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_yn (y, -2, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "-1.101001101001001100100010101001000101101000010010001");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_yn for n=-2, x=1, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_yn (y, 17, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "-0.11000100111000100010101101011000110011001101100001011E60");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_yn for n=17, x=1, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_yn (y, -17, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.11000100111000100010101101011000110011001101100001011E60");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_yn for n=-17, x=1, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

  mpfr_set_ui (x, 17, MPFR_RNDN);
  mpfr_yn (y, 1, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.00101010110011011111001100000001101011011001111111");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_yn for n=1, x=17, rnd=MPFR_RNDN\n");
      printf ("Expected "); mpfr_dump (x);
      printf ("Got      "); mpfr_dump (y);
      exit (1);
    }

 end:
  mpfr_clear (x);
  mpfr_clear (y);

  tests_end_mpfr ();
  return 0;
}
