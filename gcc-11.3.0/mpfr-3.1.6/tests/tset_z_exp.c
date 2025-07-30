/* Test file for mpfr_set_z_2exp.

Copyright 1999, 2001-2017 Free Software Foundation, Inc.
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
#include <limits.h>

#include "mpfr-test.h"

static mpfr_exp_t
randexp (void)
{
  return (mpfr_exp_t) (randlimb () % (__gmpfr_emax - __gmpfr_emin))
    + __gmpfr_emin;
}

static void
check0 (void)
{
  mpz_t y;
  mpfr_t x;
  int inexact, r;
  mpfr_exp_t e;

  /* Check for +0 */
  mpfr_init (x);
  mpz_init (y);
  mpz_set_si (y, 0);
  for (r = 0; r < MPFR_RND_MAX; r++)
    {
      e = randexp ();
      inexact = mpfr_set_z_2exp (x, y, e, (mpfr_rnd_t) r);
      if (!MPFR_IS_ZERO(x) || !MPFR_IS_POS(x) || inexact)
        {
          printf ("mpfr_set_z_2exp(x,0,e) failed for e=");
          if (e < LONG_MIN)
            printf ("(<LONG_MIN)");
          else if (e > LONG_MAX)
            printf ("(>LONG_MAX)");
          else
            printf ("%ld", (long) e);
          printf (", rnd=%s\n", mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit (1);
        }
    }
  mpfr_clear(x);
  mpz_clear(y);
}

/* FIXME: It'd be better to examine the actual data in an mpfr_t to see that
   it's as expected.  Comparing mpfr_set_z with mpfr_cmp or against
   mpfr_get_si is a rather indirect test of a low level routine.  */

static void
check (long i, mpfr_rnd_t rnd)
{
  mpfr_t f;
  mpz_t z;
  mpfr_exp_t e;
  int inex;

  /* using CHAR_BIT * sizeof(long) bits of precision ensures that
     mpfr_set_z_2exp is exact below */
  mpfr_init2 (f, CHAR_BIT * sizeof(long));
  mpz_init (z);
  mpz_set_ui (z, i);
  /* the following loop ensures that no overflow occurs */
  do
    e = randexp ();
  while (e > mpfr_get_emax () - CHAR_BIT * sizeof(long));
  inex = mpfr_set_z_2exp (f, z, e, rnd);
  if (inex != 0)
    {
      printf ("Error in mpfr_set_z_2exp for i=%ld, e=%ld,"
              " wrong ternary value\n", i, (long) e);
      printf ("expected 0, got %d\n", inex);
      exit (1);
    }
  mpfr_div_2si (f, f, e, rnd);
  if (mpfr_get_si (f, MPFR_RNDZ) != i)
    {
      printf ("Error in mpfr_set_z_2exp for i=%ld e=", i);
      if (e < LONG_MIN)
        printf ("(<LONG_MIN)");
      else if (e > LONG_MAX)
        printf ("(>LONG_MAX)");
      else
        printf ("%ld", (long) e);
      printf (" rnd_mode=%d\n", rnd);
      printf ("expected %ld\n", i);
      printf ("got      "); mpfr_dump (f);
      exit (1);
    }
  mpfr_clear (f);
  mpz_clear (z);
}

int
main (int argc, char *argv[])
{
  long j;

  tests_start_mpfr ();

  check (0, MPFR_RNDN);
  for (j = 0; j < 200000; j++)
    check (randlimb () & LONG_MAX, RND_RAND ());
  check0 ();

  tests_end_mpfr ();

  return 0;
}
