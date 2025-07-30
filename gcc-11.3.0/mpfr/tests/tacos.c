/* Test file for mpfr_acos.

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

#define TEST_FUNCTION mpfr_acos
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y;
  int inex1, inex2;

  mpfr_init2 (x, 32);
  mpfr_init2 (y, 32);

  mpfr_set_str_binary (x, "0.10001000001001011000100001E-6");
  mpfr_acos (y, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "1.10001111111111110001110110001");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_acos (1)\n");
      exit (1);
    }

  mpfr_set_str_binary (x, "-0.01101011110111100111010011001011");
  mpfr_acos (y, x, MPFR_RNDZ);
  mpfr_set_str_binary (x, "10.0000000101111000011101000101");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_acos (2)\n");
      mpfr_print_binary (y); printf ("\n");
      exit (1);
    }

  mpfr_set_prec (x, 2);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  inex1 = mpfr_acos (x, x, MPFR_RNDN); /* Pi/2 */
  inex2 = mpfr_const_pi (x, MPFR_RNDN);
  if (inex1 != inex2)
    {
      printf ("Error in mpfr_acos (3) for prec=2\n");
      exit (1);
    }

  mpfr_clear (y);
  mpfr_clear (x);
}

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
  mpfr_set_str_binary (x, "0.101100100000000000110100E0");
  mpfr_acos (y, x, MPFR_RNDN);
  if (mpfr_cmp_str (y, "0.110011010100101111000100111010111011010000001001E0",
                    2, MPFR_RNDN))
    {
      printf("Special Overflow error.\n");
      mpfr_dump (y);
      exit (1);
    }
  mpfr_clear (y);
  mpfr_clear (x);
  set_emin (emin);
  set_emax (emax);
}

int
main (void)
{
  mpfr_t x, y;
  int r;

  tests_start_mpfr ();

  special_overflow ();
  special ();

  mpfr_init (x);
  mpfr_init (y);

  MPFR_SET_NAN(x);
  mpfr_acos (y, x, MPFR_RNDN);
  if (mpfr_nan_p(y) == 0)
    {
      printf ("Error: acos(NaN) != NaN\n");
      exit (1);
    }

  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_acos (y, x, MPFR_RNDN);
  if (mpfr_nan_p(y) == 0)
    {
      printf ("Error: acos(2) != NaN\n");
      exit (1);
    }

  mpfr_set_si (x, -2, MPFR_RNDN);
  mpfr_acos (y, x, MPFR_RNDN);
  if (mpfr_nan_p(y) == 0)
    {
      printf ("Error: acos(-2) != NaN\n");
      exit (1);
    }

  /* acos (1) = 0 */
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_acos (y, x, MPFR_RNDN);
  if (mpfr_cmp_ui (y, 0) || mpfr_sgn (y) < 0)
    {
      printf ("Error: acos(1) != +0.0\n");
      exit (1);
    }

  /* acos (0) = Pi/2 */
  for (r = 0; r < MPFR_RND_MAX; r++)
    {
      mpfr_set_ui (x, 0, MPFR_RNDN); /* exact */
      mpfr_acos (y, x, (mpfr_rnd_t) r);
      mpfr_const_pi (x, (mpfr_rnd_t) r);
      mpfr_div_2exp (x, x, 1, MPFR_RNDN); /* exact */
      if (mpfr_cmp (x, y))
        {
          printf ("Error: acos(0) != Pi/2 for rnd=%s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit (1);
        }
    }

  /* acos (-1) = Pi */
  for (r = 0; r < MPFR_RND_MAX; r++)
    {
      mpfr_set_si (x, -1, MPFR_RNDN); /* exact */
      mpfr_acos (y, x, (mpfr_rnd_t) r);
      mpfr_const_pi (x, (mpfr_rnd_t) r);
      if (mpfr_cmp (x, y))
        {
          printf ("Error: acos(1) != Pi for rnd=%s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit (1);
        }
    }

  test_generic (2, 100, 7);

  mpfr_clear (x);
  mpfr_clear (y);

  data_check ("data/acos", mpfr_acos, "mpfr_acos");
  bad_cases (mpfr_acos, mpfr_cos, "mpfr_acos", 0, -40, 2, 4, 128, 800, 30);

  tests_end_mpfr ();
  return 0;
}
