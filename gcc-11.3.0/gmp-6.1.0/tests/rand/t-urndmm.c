/* Test mpz_urandomm.

Copyright 2002 Free Software Foundation, Inc.

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
#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

#ifndef TRUE
#define TRUE (1)
#endif
#ifndef FALSE
#define FALSE (0)
#endif

int
check_params (void)
{
  gmp_randstate_t r1, r2;
  mpz_t a, b, m;
  int i;
  int result;

  result = TRUE;

  mpz_init (a);
  mpz_init (b);
  mpz_init (m);

  if (result)
    {
      /* Test the consistency between urandomm and urandomb. */
      gmp_randinit_default (r1);
      gmp_randinit_default (r2);
      gmp_randseed_ui (r1, 85L);
      gmp_randseed_ui (r2, 85L);
      mpz_set_ui (m, 0L);
      mpz_setbit (m, 80L);
      for (i = 0; i < 100; i++)
	{
	  mpz_urandomm (a, r1, m);
	  mpz_urandomb (b, r2, 80L);
	  if (mpz_cmp (a, b) != 0)
	    {
	      result = FALSE;
	      printf ("mpz_urandomm != mpz_urandomb\n");
	      break;
	    }
	}
      gmp_randclear (r1);
      gmp_randclear (r2);
    }

  if (result)
    {
      /* Test that mpz_urandomm returns the correct result with a
	 broken LC.  */
      mpz_set_ui (a, 0L);
      gmp_randinit_lc_2exp (r1, a, 0xffL, 8L);
      mpz_set_ui (m, 5L);
      /* Warning: This code hangs in gmp 4.1 and below */
      for (i = 0; i < 100; i++)
	{
	  mpz_urandomm (a, r1, m);
	  if (mpz_cmp_ui (a, 2L) != 0)
	    {
	      result = FALSE;
	      gmp_printf ("mpz_urandomm returns %Zd instead of 2\n", a);
	      break;
	    }
	}
      gmp_randclear (r1);
    }

  if (result)
    {
      /* Test that the results are always in range for either
         positive or negative values of m.  */
      gmp_randinit_default (r1);
      mpz_set_ui (m, 5L);
      mpz_set_si (b, -5L);
      for (i = 0; i < 100; i++)
	{
	  mpz_urandomm (a, r1, m);
	  if (mpz_cmp_ui (a, 5L) >= 0 || mpz_sgn (a) < 0)
	    {
	      result = FALSE;
	      gmp_printf ("Out-of-range or non-positive value: %Zd\n", a);
	      break;
	    }
	  mpz_urandomm (a, r1, b);
	  if (mpz_cmp_ui (a, 5L) >= 0 || mpz_sgn (a) < 0)
	    {
	      result = FALSE;
	      gmp_printf ("Out-of-range or non-positive value (from negative modulus): %Zd\n", a);
	      break;
	    }
	}
      gmp_randclear (r1);
    }

  if (result)
    {
      /* Test that m=1 forces always result=0.  */
      gmp_randinit_default (r1);
      mpz_set_ui (m, 1L);
      for (i = 0; i < 100; i++)
	{
	  mpz_urandomm (a, r1, m);
	  if (mpz_sgn (a) != 0)
	    {
	      result = FALSE;
	      gmp_printf ("mpz_urandomm fails with m=1 (result=%Zd)\n", a);
	      break;
	    }
	}
      gmp_randclear (r1);
    }

  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (m);
  return result;
}

int
main (int argc, char *argv[])
{
  int result = TRUE;

  tests_start ();

  if (result)
    if (!check_params ())
      result = FALSE;

  tests_end ();

  if (result)
    return 0; /* pass */
  else
    return 1; /* fail */
}
