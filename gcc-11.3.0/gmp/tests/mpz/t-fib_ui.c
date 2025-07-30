/* Test mpz_fib_ui and mpz_fib2_ui.

Copyright 2000-2002 Free Software Foundation, Inc.

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


/* Usage: t-fib_ui [x|num]

   Run with no arguments, tests goes up to the initial value of "limit"
   below.  With a number argument tests are carried up that far, or with a
   literal "x" tests are continued without limit (this being only meant for
   development purposes).

   The size tests performed are designed to partially replicate what will be
   going on in mpz_fib_ui.  There's plenty of ASSERTs there, but of course
   they're not normally enabled.

   Misfeatures:

   The tests on MPN_FIB2_SIZE are a bit useless, since that macro includes a
   +2 for the internal purposes of mpn_fib2_ui.  It's probably better to
   give mpn_fib2_ui a run with assertion checking enabled.  */


#define MPZ_FIB_SIZE_FLOAT(n) \
  ((mp_size_t) ((n) * 0.6942419 / GMP_NUMB_BITS + 1))


void
check_fib_table (void)
{
  int        i;
  mp_limb_t  want;

  ASSERT_ALWAYS (FIB_TABLE(-1) == 1);
  ASSERT_ALWAYS (FIB_TABLE(0) == 0);

  for (i = 1; i <= FIB_TABLE_LIMIT; i++)
    {
      want = FIB_TABLE(i-1) + FIB_TABLE(i-2);
      if (FIB_TABLE(i) != want)
        {
          printf ("FIB_TABLE(%d) wrong\n", i);
          gmp_printf ("  got  %#Nx\n", &FIB_TABLE(i), 1);
          gmp_printf ("  want %#Nx\n", &want, 1);
          abort ();
        }
    }
}


int
main (int argc, char *argv[])
{
  unsigned long  n;
  unsigned long  limit = 100 * GMP_LIMB_BITS;
  mpz_t          want_fn, want_fn1, got_fn, got_fn1;

  tests_start ();
  mp_trace_base = -16;
  if (argc > 1 && argv[1][0] == 'x')
    limit = ULONG_MAX;
  else if (argc > 1)
    limit = atoi (argv[1]);

  check_fib_table ();

  /* start at n==0 */
  mpz_init_set_ui (want_fn1, 1);  /* F[-1] */
  mpz_init_set_ui (want_fn,  0);  /* F[0]   */
  mpz_init (got_fn);
  mpz_init (got_fn1);

  for (n = 0; n < limit; n++)
    {
      /* check our float formula seems right */
      if (MPZ_FIB_SIZE_FLOAT (n) < SIZ(want_fn))
        {
          printf ("MPZ_FIB_SIZE_FLOAT wrong at n=%lu\n", n);
          printf ("  MPZ_FIB_SIZE_FLOAT  %ld\n", MPZ_FIB_SIZE_FLOAT (n));
          printf ("  SIZ(want_fn)        %d\n", SIZ(want_fn));
          abort ();
        }

      /* check MPN_FIB2_SIZE seems right, compared to actual size and
         compared to our float formula */
      if (MPN_FIB2_SIZE (n) < MPZ_FIB_SIZE_FLOAT (n))
        {
          printf ("MPN_FIB2_SIZE wrong at n=%lu\n", n);
          printf ("  MPN_FIB2_SIZE       %ld\n", MPN_FIB2_SIZE (n));
          printf ("  MPZ_FIB_SIZE_FLOAT  %ld\n", MPZ_FIB_SIZE_FLOAT (n));
          abort ();
        }
      if (MPN_FIB2_SIZE (n) < SIZ(want_fn))
        {
          printf ("MPN_FIB2_SIZE wrong at n=%lu\n", n);
          printf ("  MPN_FIB2_SIZE  %ld\n", MPN_FIB2_SIZE (n));
          printf ("  SIZ(want_fn)   %d\n", SIZ(want_fn));
          abort ();
        }

      mpz_fib2_ui (got_fn, got_fn1, n);
      MPZ_CHECK_FORMAT (got_fn);
      MPZ_CHECK_FORMAT (got_fn1);
      if (mpz_cmp (got_fn, want_fn) != 0 || mpz_cmp (got_fn1, want_fn1) != 0)
        {
          printf ("mpz_fib2_ui(%lu) wrong\n", n);
          mpz_trace ("want fn ", want_fn);
          mpz_trace ("got  fn ",  got_fn);
          mpz_trace ("want fn1", want_fn1);
          mpz_trace ("got  fn1",  got_fn1);
          abort ();
        }

      mpz_fib_ui (got_fn, n);
      MPZ_CHECK_FORMAT (got_fn);
      if (mpz_cmp (got_fn, want_fn) != 0)
        {
          printf ("mpz_fib_ui(%lu) wrong\n", n);
          mpz_trace ("want fn", want_fn);
          mpz_trace ("got  fn", got_fn);
          abort ();
        }

      mpz_add (want_fn1, want_fn1, want_fn);  /* F[n+1] = F[n] + F[n-1] */
      mpz_swap (want_fn1, want_fn);
    }

  mpz_clear (want_fn);
  mpz_clear (want_fn1);
  mpz_clear (got_fn);
  mpz_clear (got_fn1);

  tests_end ();
  exit (0);
}
