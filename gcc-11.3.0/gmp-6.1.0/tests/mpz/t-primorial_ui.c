/* Exercise mpz_primorial_ui.

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


/* Usage: t-primorial_ui [x|num]

   With no arguments testing goes up to the initial value of "limit" below.
   With a number argument tests are carried that far, or with a literal "x"
   tests are continued without limit (this being meant only for development
   purposes).  */

static int isprime (unsigned long int t);

int
main (int argc, char *argv[])
{
  unsigned long  n;
  unsigned long  limit = 2222;
  mpz_t          f, r;

  tests_start ();

  if (argc > 1 && argv[1][0] == 'x')
    limit = ULONG_MAX;
  else if (argc > 1)
    limit = atoi (argv[1]);

  /* for small limb testing */
  limit = MIN (limit, MP_LIMB_T_MAX);

  mpz_init_set_ui (f, 1);  /* 0# = 1 */
  mpz_init (r);

  for (n = 0; n < limit; n++)
    {
      mpz_primorial_ui (r, n);
      MPZ_CHECK_FORMAT (r);

      if (mpz_cmp (f, r) != 0)
        {
          printf ("mpz_primorial_ui(%lu) wrong\n", n);
          printf ("  got  "); mpz_out_str (stdout, 10, r); printf("\n");
          printf ("  want "); mpz_out_str (stdout, 10, f); printf("\n");
          abort ();
        }

      if (isprime (n+1))
	mpz_mul_ui (f, f, n+1);  /* p# = (p-1)# * (p) */
    }

  mpz_clear (f);
  mpz_clear (r);

  tests_end ();

  exit (0);
}

static int
isprime (unsigned long int t)
{
  unsigned long int q, r, d;

  if (t < 3 || (t & 1) == 0)
    return t == 2;

  for (d = 3, r = 1; r != 0; d += 2)
    {
      q = t / d;
      r = t - q * d;
      if (q < d)
	return 1;
    }
  return 0;
}
