/* mpfr_tli2 -- test file for dilogarithm function

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

#if MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0)

#define TEST_FUNCTION mpfr_li2
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y;
  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_nan (x);
  mpfr_li2 (y, x, MPFR_RNDN);
  if (!mpfr_nan_p (y))
    {
      printf ("Error for li2(NaN)\n");
      exit (1);
    }

  mpfr_set_inf (x, -1);
  mpfr_li2 (y, x, MPFR_RNDN);
  if (!MPFR_IS_INF (y) || MPFR_IS_POS (y))
    {
      printf ("Error for li2(-Inf)\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  mpfr_li2 (y, x, MPFR_RNDN);
  if (!MPFR_IS_INF (y) || MPFR_IS_POS (y))
    {
      printf ("Error for li2(+Inf)\n");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_li2 (y, x, MPFR_RNDN);
  if (!MPFR_IS_ZERO (y) || MPFR_IS_NEG (y))
    {
      printf ("Error for li2(+0)\n");
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_li2 (y, x, MPFR_RNDN);
  if (!MPFR_IS_ZERO (y) || MPFR_IS_POS (y))
    {
      printf ("Error for li2(-0)\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
normal (void)
{
  int inexact;
  mpfr_t x, y;
  mpfr_init (x);
  mpfr_init (y);

  /* x1 = 2^-3 */
  mpfr_set_str (x, "1p-3", 2, MPFR_RNDD);
  mpfr_li2 (x, x, MPFR_RNDN);
  if (mpfr_cmp_str (x, "0x1087a7a9e42141p-55", 16, MPFR_RNDN) != 0)
    {
      printf ("Error for li2(x1)\n");
      exit (1);
    }

  /* check MPFR_FAST_COMPUTE_IF_SMALL_INPUT */
  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 20);
  mpfr_set_ui_2exp (x, 1, -21, MPFR_RNDN);
  mpfr_li2 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp (y, x) == 0);

  mpfr_set_si_2exp (x, -1, -21, MPFR_RNDN);
  mpfr_li2 (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp (y, x) == 0);

  /* worst case */
  /* x2 = 0x7F18EA6537E00E983196CDDC6EFAC57Fp-129
     Li2(x2) = 2^-2 + 2^-6 + 2^-120 */
  mpfr_set_prec (x, 128);
  mpfr_set_str (x, "7F18EA6537E00E983196CDDC6EFAC57Fp-129", 16, MPFR_RNDN);

  /* round to nearest mode and 4 bits of precision,
     it should be rounded to 2^-2 + 2^-5 and */
  mpfr_set_prec (y, 4);
  inexact = mpfr_li2 (y, x, MPFR_RNDN);
  if (inexact != 1 || mpfr_cmp_str (y, "0.1001p-1", 2, MPFR_RNDN) != 0)
    {
      printf ("Error for li2(x2, RNDN)\n");
      exit (1);
    }

  /* round toward zero mode and 5 bits of precision,
     it should be rounded to 2^-2 + 2^-6 */
  mpfr_set_prec (y, 5);
  inexact = mpfr_li2 (y, x, MPFR_RNDZ);
  if (inexact != -1 || mpfr_cmp_str (y, "0.10001p-1", 2, MPFR_RNDN) != 0)
    {
      printf ("Error for li2(x2, RNDZ)\n");
      exit (1);
    }

  /* round away from zero mode and 5 bits of precision,
     it should be rounded to 2^-2 + 2^-5 */
  inexact = mpfr_li2 (y, x, MPFR_RNDU);
  if (inexact != 1 || mpfr_cmp_str (y, "0.10010p-1", 2, MPFR_RNDN) != 0)
    {
      printf ("Error for li2(x2, RNDU)\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
bug20091013 (void)
{
  mpfr_t x, y;
  int inex;

  mpfr_init2 (x, 17);
  mpfr_init2 (y, 2);
  mpfr_set_str_binary (x, "0.10000000000000000E-16");
  inex = mpfr_li2 (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui_2exp (y, 1, -17) != 0)
    {
      printf ("Error in bug20091013()\n");
      printf ("expected 2^(-17)\n");
      printf ("got      ");
      mpfr_dump (y);
      exit (1);
    }
  if (inex >= 0)
    {
      printf ("Error in bug20091013()\n");
      printf ("expected negative ternary value, got %d\n", inex);
      exit (1);
    }
  mpfr_clear (x);
  mpfr_clear (y);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  bug20091013 ();

  special ();

  normal ();

  test_generic (2, 100, 2);

  data_check ("data/li2", mpfr_li2, "mpfr_li2");

  tests_end_mpfr ();
  return 0;
}

#else

int
main (void)
{
  printf ("Warning! Test disabled for this MPFR version.\n");
  return 0;
}

#endif
