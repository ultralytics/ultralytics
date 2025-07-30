/* Test mpz_lcm and mpz_lcm_ui.

Copyright 2001 Free Software Foundation, Inc.

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
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


void
check_all (mpz_ptr want, mpz_srcptr x_orig, mpz_srcptr y_orig)
{
  mpz_t  got, x, y;
  int    negx, negy, swap, inplace;

  mpz_init (got);
  mpz_init_set (x, x_orig);
  mpz_init_set (y, y_orig);

  for (swap = 0; swap < 2; swap++)
    {
      mpz_swap (x, y);

      for (negx = 0; negx < 2; negx++)
	{
	  mpz_neg (x, x);

	  for (negy = 0; negy < 2; negy++)
	    {
	      mpz_neg (y, y);

	      for (inplace = 0; inplace <= 1; inplace++)
		{
		  if (inplace)
		    { mpz_set (got, x); mpz_lcm (got, got, y); }
		  else
		    mpz_lcm (got, x, y);
		  MPZ_CHECK_FORMAT (got);

		  if (mpz_cmp (got, want) != 0)
		    {
		      printf ("mpz_lcm wrong, inplace=%d\n", inplace);
		    fail:
		      mpz_trace ("x", x);
		      mpz_trace ("y", y);
		      mpz_trace ("got", got);
		      mpz_trace ("want", want);
		      abort ();
		    }

		  if (mpz_fits_ulong_p (y))
		    {
		      unsigned long  yu = mpz_get_ui (y);
		      if (inplace)
			{ mpz_set (got, x); mpz_lcm_ui (got, got, yu); }
		      else
			mpz_lcm_ui (got, x, yu);

		      if (mpz_cmp (got, want) != 0)
			{
			  printf ("mpz_lcm_ui wrong, inplace=%d\n", inplace);
			  printf    ("yu=%lu\n", yu);
			  goto fail;
			}
		    }
		}
	    }
	}
    }

  mpz_clear (got);
  mpz_clear (x);
  mpz_clear (y);
}


void
check_primes (void)
{
  static unsigned long  prime[] = {
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,
    101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,
    191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,
    281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,
    389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,
  };
  mpz_t  want, x, y;
  int    i;

  mpz_init (want);
  mpz_init (x);
  mpz_init (y);

  /* Check zeros. */
  mpz_set_ui (want, 0);
  mpz_set_ui (x, 1);
  check_all (want, want, want);
  check_all (want, want, x);
  check_all (want, x, want);

  /* New prime each time. */
  mpz_set_ui (want, 1L);
  for (i = 0; i < numberof (prime); i++)
    {
      mpz_set (x, want);
      mpz_set_ui (y, prime[i]);
      mpz_mul_ui (want, want, prime[i]);
      check_all (want, x, y);
    }

  /* Old prime each time. */
  mpz_set (x, want);
  for (i = 0; i < numberof (prime); i++)
    {
      mpz_set_ui (y, prime[i]);
      check_all (want, x, y);
    }

  /* One old, one new each time. */
  mpz_set_ui (want, prime[0]);
  for (i = 1; i < numberof (prime); i++)
    {
      mpz_set (x, want);
      mpz_set_ui (y, prime[i] * prime[i-1]);
      mpz_mul_ui (want, want, prime[i]);
      check_all (want, x, y);
    }

  /* Triplets with A,B in x and B,C in y. */
  mpz_set_ui (want, 1L);
  mpz_set_ui (x, 1L);
  mpz_set_ui (y, 1L);
  for (i = 0; i+2 < numberof (prime); i += 3)
    {
      mpz_mul_ui (want, want, prime[i]);
      mpz_mul_ui (want, want, prime[i+1]);
      mpz_mul_ui (want, want, prime[i+2]);

      mpz_mul_ui (x, x, prime[i]);
      mpz_mul_ui (x, x, prime[i+1]);

      mpz_mul_ui (y, y, prime[i+1]);
      mpz_mul_ui (y, y, prime[i+2]);

      check_all (want, x, y);
    }


  mpz_clear (want);
  mpz_clear (x);
  mpz_clear (y);
}



int
main (int argc, char *argv[])
{
  tests_start ();

  check_primes ();

  tests_end ();
  exit (0);
}
