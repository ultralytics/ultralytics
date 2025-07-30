/* Test file for mpfr_trunc, mpfr_ceil, mpfr_floor.

Copyright 1999-2004, 2006-2017 Free Software Foundation, Inc.
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

#define SIZEX 100

int
main (void)
{
  int j, k;
  mpfr_t x, y, z, t, y2, z2, t2;

  tests_start_mpfr ();

  mpfr_inits2 (SIZEX, x, y, z, t, y2, z2, t2, (mpfr_ptr) 0);

  mpfr_set_str1 (x, "0.5");
  mpfr_ceil(y, x);
  if (mpfr_cmp_ui (y, 1))
    {
      printf ("Error in mpfr_ceil for x=0.5: expected 1.0, got ");
      mpfr_print_binary(y);
      putchar('\n');
      exit (1);
    }

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_ceil(y, x);
  if (mpfr_cmp_ui(y,0))
    {
      printf ("Error in mpfr_ceil for x=0.0: expected 0.0, got ");
      mpfr_print_binary(y);
      putchar('\n');
      exit (1);
    }

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_ceil(y, x);
  if (mpfr_cmp_ui(y,1))
    {
      printf ("Error in mpfr_ceil for x=1.0: expected 1.0, got ");
      mpfr_print_binary(y);
      putchar('\n');
      exit (1);
    }

  for (j=0;j<1000;j++)
    {
      mpfr_urandomb (x, RANDS);
      MPFR_EXP (x) = 2;

      for (k = 2; k <= SIZEX; k++)
        {
          mpfr_set_prec(y, k);
          mpfr_set_prec(y2, k);
          mpfr_set_prec(z, k);
          mpfr_set_prec(z2, k);
          mpfr_set_prec(t, k);
          mpfr_set_prec(t2, k);

          mpfr_floor(y, x);
          mpfr_set(y2, x, MPFR_RNDD);

          mpfr_trunc(z, x);
          mpfr_set(z2, x, MPFR_RNDZ);

          mpfr_ceil(t, x);
          mpfr_set(t2, x, MPFR_RNDU);

          if (!mpfr_eq(y, y2, k))
            {
              printf("Error in floor, x = "); mpfr_print_binary(x);
              printf("\n");
              printf("floor(x) = "); mpfr_print_binary(y);
              printf("\n");
              printf("round(x, RNDD) = "); mpfr_print_binary(y2);
              printf("\n");
              exit(1);
            }

          if (!mpfr_eq(z, z2, k))
            {
              printf("Error in trunc, x = "); mpfr_print_binary(x);
              printf("\n");
              printf("trunc(x) = "); mpfr_print_binary(z);
              printf("\n");
              printf("round(x, RNDZ) = "); mpfr_print_binary(z2);
              printf("\n");
              exit(1);
            }

          if (!mpfr_eq(y, y2, k))
            {
              printf("Error in ceil, x = "); mpfr_print_binary(x);
              printf("\n");
              printf("ceil(x) = "); mpfr_print_binary(t);
              printf("\n");
              printf("round(x, RNDU) = "); mpfr_print_binary(t2);
              printf("\n");
              exit(1);
            }
          MPFR_EXP(x)++;
        }
    }

  mpfr_clears (x, y, z, t, y2, z2, t2, (mpfr_ptr) 0);

  tests_end_mpfr ();
  return 0;
}
