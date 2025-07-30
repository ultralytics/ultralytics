/* Test file for mpfr_greater_p, mpfr_greaterequal_p, mpfr_less_p,
   mpfr_lessequal_p, mpfr_lessgreater_p, mpfr_equal_p, mpfr_unordered_p
   functions.

Copyright 2003, 2006-2017 Free Software Foundation, Inc.
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
cmp_tests (void)
{
  mpfr_t x, y;
  long i;

  mpfr_inits (x, y, (mpfr_ptr) 0);
  for (i = 0; i < 80000; i++)
    {
      mpfr_prec_t precx, precy;
      int signx, signy, cmp;
      unsigned int cmpbool = 0;

      precx = (randlimb () % 17) * 11 + MPFR_PREC_MIN;
      precy = (randlimb () % 17) * 11 + MPFR_PREC_MIN;
      mpfr_set_prec (x, precx);
      mpfr_set_prec (y, precy);
      mpfr_urandomb (x, RANDS);
      mpfr_urandomb (y, RANDS);
      signx = randlimb () & 1;
      signy = randlimb () % 256 ? signx : 1 - signx;
      /* signy = signx most of the time (most interesting case) */
      if (signx)
        mpfr_neg (x, x, MPFR_RNDN);
      if (signy)
        mpfr_neg (y, y, MPFR_RNDN);
      if (i <= 1)
        mpfr_set_nan (x);
      if (i == 0 || i == 2)
        mpfr_set_nan (y);
      if (mpfr_greater_p (x, y))
        cmpbool |= 0x01;
      if (mpfr_greaterequal_p (x, y))
        cmpbool |= 0x02;
      if (mpfr_less_p (x, y))
        cmpbool |= 0x04;
      if (mpfr_lessequal_p (x, y))
        cmpbool |= 0x08;
      if (mpfr_lessgreater_p (x, y))
        cmpbool |= 0x10;
      if (mpfr_equal_p (x, y))
        cmpbool |= 0x20;
      if (mpfr_unordered_p (x, y))
        cmpbool |= 0x40;
      if ((i <= 2 && cmpbool != 0x40) ||
          (i > 2 && (cmp = mpfr_cmp (x, y),
                     (cmp == 0 && cmpbool != 0x2a) ||
                     (cmp < 0 && cmpbool != 0x1c) ||
                     (cmp > 0 && cmpbool != 0x13))))
        {
          printf ("Error in cmp_tests for\nx = ");
          mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
          printf (" and\ny = ");
          mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
          printf ("\n");
          exit (1);
        }
    }
  mpfr_clears (x, y, (mpfr_ptr) 0);
}

static void
eq_tests (void)
{
  mpfr_t x, y;
  long i;

  mpfr_inits (x, y, (mpfr_ptr) 0);
  for (i = 0; i < 20000; i++)
    {
      mpfr_prec_t precx;

      precx = (randlimb () % 17) * 11 + MPFR_PREC_MIN;
      mpfr_set_prec (x, precx);
      mpfr_set_prec (y, precx + (randlimb () % 64));
      mpfr_urandomb (x, RANDS);
      if (randlimb () & 1)
        mpfr_neg (x, x, MPFR_RNDN);
      mpfr_set (y, x, MPFR_RNDN);  /* exact -> x = y */
      if (mpfr_greater_p (x, y) || !mpfr_greaterequal_p (x, y) ||
          mpfr_less_p (x, y) || !mpfr_lessequal_p (x, y) ||
          mpfr_lessgreater_p (x, y) || !mpfr_equal_p (x, y) ||
          mpfr_unordered_p (x, y))
        {
          printf ("Error in eq_tests for x = ");
          mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
          printf ("\n");
          exit (1);
        }
    }
  mpfr_clears (x, y, (mpfr_ptr) 0);
}

int
main (void)
{
  tests_start_mpfr ();
  cmp_tests ();
  eq_tests ();
  tests_end_mpfr ();
  return 0;
}
