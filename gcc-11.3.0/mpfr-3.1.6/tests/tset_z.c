/* Test file for mpfr_set_z.

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

static void check0(void)
{
  mpz_t y;
  mpfr_t x;
  int inexact, r;

  /* Check for +0 */
  mpfr_init (x);
  mpz_init (y);
  mpz_set_si (y, 0);
  for(r = 0; r < MPFR_RND_MAX; r++)
    {
      inexact = mpfr_set_z (x, y, (mpfr_rnd_t) r);
      if (!MPFR_IS_ZERO(x) || !MPFR_IS_POS(x) || inexact)
        {
          printf("mpfr_set_z(x,0) failed for %s\n",
                 mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit(1);
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

  mpfr_init2 (f, 8 * sizeof(long));
  mpz_init (z);
  mpz_set_ui (z, i);
  mpfr_set_z (f, z, rnd);
  if (mpfr_get_si (f, MPFR_RNDZ) != i)
    {
      printf ("Error in mpfr_set_z for i=%ld rnd_mode=%d\n", i, rnd);
      exit (1);
    }
  mpfr_clear (f);
  mpz_clear (z);
}

static void
check_large (void)
{
  mpz_t z;
  mpfr_t x, y;
  mpfr_exp_t emax, emin;

  mpz_init (z);
  mpfr_init2 (x, 160);
  mpfr_init2 (y, 160);

  mpz_set_str (z, "77031627725494291259359895954016675357279104942148788042", 10);
  mpfr_set_z (x, z, MPFR_RNDN);
  mpfr_set_str_binary (y, "0.1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000000011011100000111001101000100101001000000100100111000001001E186");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_z on large input\n");
      exit (1);
    }

  /* check overflow */
  emax = mpfr_get_emax ();
  set_emax (2);
  mpz_set_str (z, "7", 10);
  mpfr_set_z (x, z, MPFR_RNDU);
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);
  set_emax (3);
  mpfr_set_prec (x, 2);
  mpz_set_str (z, "7", 10);
  mpfr_set_z (x, z, MPFR_RNDU);
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);
  set_emax (emax);

  /* check underflow */
  emin = mpfr_get_emin ();
  set_emin (3);
  mpz_set_str (z, "1", 10);
  mpfr_set_z (x, z, MPFR_RNDZ);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_POS(x));
  set_emin (2);
  mpfr_set_z (x, z, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_POS(x));
  set_emin (emin);

  mpz_clear (z);

  mpfr_clear (x);
  mpfr_clear (y);
}

int
main (int argc, char *argv[])
{
  long j;

  tests_start_mpfr ();

  check_large ();
  check (0, MPFR_RNDN);
  for (j = 0; j < 200000; j++)
    check (randlimb () & LONG_MAX, RND_RAND ());
  check0 ();

  tests_end_mpfr ();

  return 0;
}
