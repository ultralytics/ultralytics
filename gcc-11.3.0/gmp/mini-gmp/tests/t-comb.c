/* Exercise mpz_fac_ui and mpz_bin_uiui.

Copyright 2000-2002, 2012, 2013 Free Software Foundation, Inc.

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

#include "testutils.h"

/* Usage: t-fac_ui [x|num]

   With no arguments testing goes up to the initial value of "limit" below.
   With a number argument tests are carried that far, or with a literal "x"
   tests are continued without limit (this being meant only for development
   purposes).  */

void
try_mpz_bin_uiui (mpz_srcptr want, unsigned long n, unsigned long k)
{
  mpz_t  got;

  mpz_init (got);
  mpz_bin_uiui (got, n, k);
  if (mpz_cmp (got, want) != 0)
    {
      printf ("mpz_bin_uiui wrong\n");
      printf ("  n=%lu\n", n);
      printf ("  k=%lu\n", k);
      printf ("  got="); mpz_out_str (stdout, 10, got); printf ("\n");
      printf ("  want="); mpz_out_str (stdout, 10, want); printf ("\n");
      abort();
    }
  mpz_clear (got);
}

/* Test all bin(n,k) cases, with 0 <= k <= n + 1 <= count.  */
void
bin_smallexaustive (unsigned int count)
{
  mpz_t          want;
  unsigned long  n, k;

  mpz_init (want);

  for (n = 0; n < count; n++)
    {
      mpz_set_ui (want, 1);
      for (k = 0; k <= n; k++)
	{
	  try_mpz_bin_uiui (want, n, k);
	  mpz_mul_ui (want, want, n - k);
	  mpz_fdiv_q_ui (want, want, k + 1);
	}
      try_mpz_bin_uiui (want, n, k);
    }

  mpz_clear (want);
}

/* Test all fac(n) cases, with 0 <= n <= limit.  */
void
fac_smallexaustive (unsigned int limit)
{
  mpz_t          f, r;
  unsigned long  n;
  mpz_init_set_si (f, 1);  /* 0! = 1 */
  mpz_init (r);

  for (n = 0; n < limit; n++)
    {
      mpz_fac_ui (r, n);

      if (mpz_cmp (f, r) != 0)
        {
          printf ("mpz_fac_ui(%lu) wrong\n", n);
          printf ("  got  "); mpz_out_str (stdout, 10, r); printf("\n");
          printf ("  want "); mpz_out_str (stdout, 10, f); printf("\n");
          abort ();
        }

      mpz_mul_ui (f, f, n+1);  /* (n+1)! = n! * (n+1) */
    }

  mpz_clear (f);
  mpz_clear (r);
}

void checkWilson (mpz_t f, unsigned long n)
{
  unsigned long m;

  mpz_fac_ui (f, n - 1);
  m = mpz_fdiv_ui (f, n);
  if ( m != n - 1)
    {
      printf ("mpz_fac_ui(%lu) wrong\n", n - 1);
      printf ("  Wilson's theorem not verified: got %lu, expected %lu.\n",m ,n - 1);
      abort ();
    }
}

void
checkprimes (unsigned long p1, unsigned long p2, unsigned long p3)
{
  mpz_t          b, f;

  if (p1 - 1 != p2 - 1 + p3 - 1)
    {
      printf ("checkprimes(%lu, %lu, %lu) wrong\n", p1, p2, p3);
      printf (" %lu - 1 != %lu - 1 + %lu - 1 \n", p1, p2, p3);
      abort ();
    }

  mpz_init (b);
  mpz_init (f);

  checkWilson (b, p1); /* b = (p1-1)! */
  checkWilson (f, p2); /* f = (p2-1)! */
  mpz_divexact (b, b, f);
  checkWilson (f, p3); /* f = (p3-1)! */
  mpz_divexact (b, b, f); /* b = (p1-1)!/((p2-1)!(p3-1)!) */
  mpz_bin_uiui (f, p1 - 1, p2 - 1);
  if (mpz_cmp (f, b) != 0)
    {
      printf ("checkprimes(%lu, %lu, %lu) wrong\n", p1, p2, p3);
      printf ("  got  "); mpz_out_str (stdout, 10, b); printf("\n");
      printf ("  want "); mpz_out_str (stdout, 10, f); printf("\n");
      abort ();
    }

  mpz_clear (b);
  mpz_clear (f);

}

void
testmain (int argc, char *argv[])
{
  unsigned long  limit = 128;

  if (argc > 1 && argv[1][0] == 'x')
    limit = ~ limit;
  else if (argc > 1)
    limit = atoi (argv[1]);

  checkprimes(1009, 733, 277);
  fac_smallexaustive (limit);
  bin_smallexaustive (limit);
}
