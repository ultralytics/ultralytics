/* Test mpz_lucnum_ui and mpz_lucnum2_ui.

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
#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


/* Usage: t-lucnum_ui [n]

   Test up to L[n], or if n is omitted then the default limit below.  A
   literal "x" for the limit means continue forever, this being meant only
   for development.  */


void
check_sequence (int argc, char *argv[])
{
  unsigned long  n;
  unsigned long  limit = 100 * GMP_LIMB_BITS;
  mpz_t          want_ln, want_ln1, got_ln, got_ln1;

  if (argc > 1 && argv[1][0] == 'x')
    limit = ULONG_MAX;
  else if (argc > 1)
    limit = atoi (argv[1]);

  /* start at n==0 */
  mpz_init_set_si (want_ln1, -1); /* L[-1] */
  mpz_init_set_ui (want_ln,  2);  /* L[0]   */
  mpz_init (got_ln);
  mpz_init (got_ln1);

  for (n = 0; n < limit; n++)
    {
      mpz_lucnum2_ui (got_ln, got_ln1, n);
      MPZ_CHECK_FORMAT (got_ln);
      MPZ_CHECK_FORMAT (got_ln1);
      if (mpz_cmp (got_ln, want_ln) != 0 || mpz_cmp (got_ln1, want_ln1) != 0)
        {
          printf ("mpz_lucnum2_ui(%lu) wrong\n", n);
          mpz_trace ("want ln ", want_ln);
          mpz_trace ("got  ln ",  got_ln);
          mpz_trace ("want ln1", want_ln1);
          mpz_trace ("got  ln1",  got_ln1);
          abort ();
        }

      mpz_lucnum_ui (got_ln, n);
      MPZ_CHECK_FORMAT (got_ln);
      if (mpz_cmp (got_ln, want_ln) != 0)
        {
          printf ("mpz_lucnum_ui(%lu) wrong\n", n);
          mpz_trace ("want ln", want_ln);
          mpz_trace ("got  ln", got_ln);
          abort ();
        }

      mpz_add (want_ln1, want_ln1, want_ln);  /* L[n+1] = L[n] + L[n-1] */
      mpz_swap (want_ln1, want_ln);
    }

  mpz_clear (want_ln);
  mpz_clear (want_ln1);
  mpz_clear (got_ln);
  mpz_clear (got_ln1);
}

int
main (int argc, char *argv[])
{
  tests_start ();
  mp_trace_base = -16;

  check_sequence (argc, argv);

  tests_end ();
  exit (0);
}
