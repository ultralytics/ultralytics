/* Exercise mpz_mfac_uiui.

Copyright 2000-2002, 2012 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include <stdio.h>
#include <stdlib.h>
#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


/* Usage: t-mfac_uiui [x|num]

   With no arguments testing goes up to the initial value of "limit" below.
   With a number argument tests are carried that far, or with a literal "x"
   tests are continued without limit (this being meant only for development
   purposes).  */

#define MULTIFAC_WHEEL (2*3*11)
#define MULTIFAC_WHEEL2 (5*13)

int
main (int argc, char *argv[])
{
  mpz_t ref[MULTIFAC_WHEEL], ref2[MULTIFAC_WHEEL2], res;
  unsigned long n, j, m, m2;
  unsigned long limit = 2222, step = 1;

  tests_start ();

  if (argc > 1 && argv[1][0] == 'x')
    limit = ULONG_MAX;
  else if (argc > 1)
    limit = atoi (argv[1]);

  /* for small limb testing */
  limit = MIN (limit, MP_LIMB_T_MAX);

  for (m = 0; m < MULTIFAC_WHEEL; m++)
    mpz_init_set_ui(ref [m],1);
  for (m2 = 0; m2 < MULTIFAC_WHEEL2; m2++)
    mpz_init_set_ui(ref2 [m2],1);

  mpz_init (res);

  m = 0;
  m2 = 0;
  for (n = 0; n <= limit;)
    {
      mpz_mfac_uiui (res, n, MULTIFAC_WHEEL);
      MPZ_CHECK_FORMAT (res);
      if (mpz_cmp (ref[m], res) != 0)
        {
          printf ("mpz_mfac_uiui(%lu,%d) wrong\n", n, MULTIFAC_WHEEL);
          printf ("  got  "); mpz_out_str (stdout, 10, res); printf("\n");
          printf ("  want "); mpz_out_str (stdout, 10, ref[m]); printf("\n");
          abort ();
        }
      mpz_mfac_uiui (res, n, MULTIFAC_WHEEL2);
      MPZ_CHECK_FORMAT (res);
      if (mpz_cmp (ref2[m2], res) != 0)
        {
          printf ("mpz_mfac_uiui(%lu,%d) wrong\n", n, MULTIFAC_WHEEL2);
          printf ("  got  "); mpz_out_str (stdout, 10, res); printf("\n");
          printf ("  want "); mpz_out_str (stdout, 10, ref2[m2]); printf("\n");
          abort ();
        }
      if (n + step <= limit)
	for (j = 0; j < step; j++) {
	  n++; m++; m2++;
	  if (m >= MULTIFAC_WHEEL) m -= MULTIFAC_WHEEL;
	  if (m2 >= MULTIFAC_WHEEL2) m2 -= MULTIFAC_WHEEL2;
	  mpz_mul_ui (ref[m], ref[m], n); /* Compute a reference, with current library */
	  mpz_mul_ui (ref2[m2], ref2[m2], n); /* Compute a reference, with current library */
	}
      else n += step;
    }
  mpz_fac_ui (ref[0], n);
  mpz_mfac_uiui (res, n, 1);
  MPZ_CHECK_FORMAT (res);
  if (mpz_cmp (ref[0], res) != 0)
    {
      printf ("mpz_mfac_uiui(%lu,1) wrong\n", n);
      printf ("  got  "); mpz_out_str (stdout, 10, res); printf("\n");
      printf ("  want "); mpz_out_str (stdout, 10, ref[0]); printf("\n");
      abort ();
    }

  mpz_2fac_ui (ref[0], n);
  mpz_mfac_uiui (res, n, 2);
  MPZ_CHECK_FORMAT (res);
  if (mpz_cmp (ref[0], res) != 0)
    {
      printf ("mpz_mfac_uiui(%lu,1) wrong\n", n);
      printf ("  got  "); mpz_out_str (stdout, 10, res); printf("\n");
      printf ("  want "); mpz_out_str (stdout, 10, ref[0]); printf("\n");
      abort ();
    }

  n++;
  mpz_2fac_ui (ref[0], n);
  mpz_mfac_uiui (res, n, 2);
  MPZ_CHECK_FORMAT (res);
  if (mpz_cmp (ref[0], res) != 0)
    {
      printf ("mpz_mfac_uiui(%lu,2) wrong\n", n);
      printf ("  got  "); mpz_out_str (stdout, 10, res); printf("\n");
      printf ("  want "); mpz_out_str (stdout, 10, ref[0]); printf("\n");
      abort ();
    }

  for (m = 0; m < MULTIFAC_WHEEL; m++)
    mpz_clear (ref[m]);
  for (m2 = 0; m2 < MULTIFAC_WHEEL2; m2++)
    mpz_clear (ref2[m2]);
  mpz_clear (res);

  tests_end ();

  exit (0);
}
