/* Test file for mpz_set_fr / mpfr_get_z.

Copyright 2004, 2006-2017 Free Software Foundation, Inc.
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

static void
check_diff (void)
{
  int inex;
  mpfr_t x;
  mpz_t  z;
  mpfr_exp_t emin;

  mpz_init   (z);
  mpfr_init2 (x, 2);

  mpfr_set_ui (x, 2047, MPFR_RNDU);
  mpz_set_fr (z, x, MPFR_RNDN);
  if (mpz_cmp_ui (z, 2048) != 0)
    {
      printf ("get_z RU 2048 failed\n");
      exit (1);
    }

  mpfr_set_prec (x, 6);
  mpfr_set_str (x, "17.5", 10, MPFR_RNDN);
  inex = mpfr_get_z (z, x, MPFR_RNDN);
  if (inex <= 0 || mpz_cmp_ui (z, 18) != 0)
    {
      printf ("get_z RN 17.5 failed\n");
      exit (1);
    }

  /* save default emin */
  emin = mpfr_get_emin ();;

  mpfr_set_emin (17);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  inex = mpfr_get_z (z, x, MPFR_RNDN);
  if (inex != 0 || mpz_cmp_ui (z, 0) != 0)
    {
      printf ("get_z 0 failed\n");
      exit (1);
    }

  /* restore default emin */
  mpfr_set_emin (emin);

  mpfr_clear (x);
  mpz_clear  (z);
}

static void
check_one (mpz_ptr z)
{
  int    inex;
  int    sh, neg;
  mpfr_t f;
  mpz_t  got;

  mpfr_init2 (f, MAX( mpz_sizeinbase (z, 2), MPFR_PREC_MIN) );
  mpz_init (got);

  for (sh = -2*GMP_NUMB_BITS ; sh < 2*GMP_NUMB_BITS ; sh++)
    {
      for (neg = 0; neg <= 1; neg++)
        {
          mpz_neg (z, z);
          mpfr_set_z (f, z, MPFR_RNDN);

          if (sh < 0)
            {
              mpz_tdiv_q_2exp (z, z, -sh);
              mpfr_div_2exp (f, f, -sh, MPFR_RNDN);
            }
          else
            {
              mpz_mul_2exp (z, z, sh);
              mpfr_mul_2exp (f, f, sh, MPFR_RNDN);
            }

          inex = mpfr_get_z (got, f, MPFR_RNDZ);

          if (mpz_cmp (got, z) != 0)
            {
              printf ("Wrong result for shift=%d\n", sh);
              printf ("     f "); mpfr_dump (f);
              printf ("   got "); mpz_dump (got);
              printf ("  want "); mpz_dump (z);
              exit (1);
            }
          if (! SAME_SIGN (inex, - mpfr_cmp_z (f, z)))
            {
              printf ("Wrong inexact value for shift=%d\n", sh);
              printf ("    f "); mpfr_dump (f);
              printf ("  got %+d\n", inex);
              printf (" want %+d\n", -mpfr_cmp_z (f, z));
              exit (1);
            }
        }
    }

  mpfr_clear (f);
  mpz_clear (got);
}

static void
check (void)
{
  mpz_t  z;

  mpz_init (z);

  mpz_set_ui (z, 0L);
  check_one (z);

  mpz_set_si (z, 123L);
  check_one (z);

  mpz_rrandomb (z, RANDS, 2*GMP_NUMB_BITS);
  check_one (z);

  mpz_rrandomb (z, RANDS, 5*GMP_NUMB_BITS);
  check_one (z);

  mpz_clear (z);
}

static void
special (void)
{
  int inex;
  mpfr_t x;
  mpz_t z;
  int i;
  mpfr_exp_t e;

  mpfr_init2 (x, 2);
  mpz_init (z);

  for (i = -1; i <= 1; i++)
    {
      if (i != 0)
        mpfr_set_nan (x);
      else
        mpfr_set_inf (x, i);
      mpfr_clear_flags ();
      inex = mpfr_get_z (z, x, MPFR_RNDN);
      if (!mpfr_erangeflag_p () || inex != 0 || mpz_cmp_ui (z, 0) != 0)
        {
          printf ("special() failed on mpfr_get_z for i = %d\n", i);
          exit (1);
        }
      mpfr_clear_flags ();
      e = mpfr_get_z_2exp (z, x);
      if (!mpfr_erangeflag_p () || e != __gmpfr_emin ||
          mpz_cmp_ui (z, 0) != 0)
        {
          printf ("special() failed on mpfr_get_z_2exp for i = %d\n", i);
          exit (1);
        }
    }

  mpfr_clear (x);
  mpz_clear (z);
}

int
main (void)
{
  tests_start_mpfr ();

  check ();
  check_diff ();
  special ();

  tests_end_mpfr ();
  return 0;
}
