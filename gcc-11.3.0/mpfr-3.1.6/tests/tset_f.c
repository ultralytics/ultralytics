/* Test file for mpfr_set_f.

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
#include <limits.h> /* for ULONG_MAX */

#include "mpfr-test.h"

int
main (void)
{
  mpfr_t x, u;
  mpf_t y, z;
  mpfr_exp_t emax;
  unsigned long k, pr;
  int r, inexact;

  tests_start_mpfr ();

  mpf_init (y);
  mpf_init (z);

  mpf_set_d (y, 0.0);

  /* check prototype of mpfr_init_set_f */
  mpfr_init_set_f (x, y, MPFR_RNDN);
  mpfr_set_prec (x, 100);
  mpfr_set_f (x, y, MPFR_RNDN);

  mpf_urandomb (y, RANDS, 10 * GMP_NUMB_BITS);
  mpfr_set_f (x, y, RND_RAND ());

  /* bug found by Jean-Pierre Merlet */
  mpfr_set_prec (x, 256);
  mpf_set_prec (y, 256);
  mpfr_init2 (u, 256);
  mpfr_set_str (u,
                "7.f10872b020c49ba5e353f7ced916872b020c49ba5e353f7ced916872b020c498@2",
                16, MPFR_RNDN);
  mpf_set_str (y, "2033033E-3", 10); /* avoid 2033.033 which is
                                        locale-sensitive */
  mpfr_set_f (x, y, MPFR_RNDN);
  if (mpfr_cmp (x, u))
    {
      printf ("mpfr_set_f failed for y=2033033E-3\n");
      exit (1);
    }
  mpf_set_str (y, "-2033033E-3", 10); /* avoid -2033.033 which is
                                         locale-sensitive */
  mpfr_set_f (x, y, MPFR_RNDN);
  mpfr_neg (u, u, MPFR_RNDN);
  if (mpfr_cmp (x, u))
    {
      printf ("mpfr_set_f failed for y=-2033033E-3\n");
      exit (1);
    }

  mpf_set_prec (y, 300);
  mpf_set_str (y, "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", -2);
  mpf_mul_2exp (y, y, 600);
  mpfr_set_prec (x, 300);
  mpfr_set_f (x, y, MPFR_RNDN);
  if (mpfr_check (x) == 0)
    {
      printf ("Error in mpfr_set_f: corrupted result\n");
      mpfr_dump (x);
      exit (1);
    }
  MPFR_ASSERTN(mpfr_cmp_ui_2exp (x, 1, 901) == 0);

  /* random values */
  for (k = 1; k <= 1000; k++)
    {
      pr = 2 + (randlimb () & 255);
      mpf_set_prec (z, pr);
      mpf_urandomb (z, RANDS, z->_mp_prec);
      mpfr_set_prec (u, ((pr / GMP_NUMB_BITS + 1) * GMP_NUMB_BITS));
      mpfr_set_f (u, z, MPFR_RNDN);
      if (mpfr_cmp_f (u , z) != 0)
        {
          printf ("Error in mpfr_set_f:\n");
          printf ("mpf (precision=%lu)=", pr);
          mpf_out_str (stdout, 16, 0, z);
          printf ("\nmpfr(precision=%lu)=",
                  ((pr / GMP_NUMB_BITS + 1) * GMP_NUMB_BITS));
          mpfr_out_str (stdout, 16, 0, u, MPFR_RNDN);
          putchar ('\n');
          exit (1);
        }
      mpfr_set_prec (x, pr);
      mpfr_set_f (x, z, MPFR_RNDN);
      mpfr_sub (u, u, x, MPFR_RNDN);
      mpfr_abs (u, u, MPFR_RNDN);
      if (mpfr_cmp_ui_2exp (u, 1, -pr - 1) > 0)
        {
          printf ("Error in mpfr_set_f: precision=%lu\n", pr);
          printf ("mpf =");
          mpf_out_str (stdout, 16, 0, z);
          printf ("\nmpfr=");
          mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN);
          putchar ('\n');
          exit (1);
        }
    }

  /* Check for +0 */
  mpfr_set_prec (x, 53);
  mpf_set_prec (y, 53);
  mpf_set_ui (y, 0);
  for (r = 0 ; r < MPFR_RND_MAX ; r++)
    {
      int i;
      for (i = -1; i <= 1; i++)
        {
          if (i)
            mpfr_set_si (x, i, MPFR_RNDN);
          inexact = mpfr_set_f (x, y, (mpfr_rnd_t) r);
          if (!MPFR_IS_ZERO(x) || !MPFR_IS_POS(x) || inexact)
            {
              printf ("mpfr_set_f(x,0) failed for %s, i = %d\n",
                      mpfr_print_rnd_mode ((mpfr_rnd_t) r), i);
              exit (1);
            }
        }
    }

  /* coverage test */
  mpf_set_prec (y, 2);
  mpfr_set_prec (x, 3 * mp_bits_per_limb);
  mpf_set_ui (y, 1);
  for (r = 0; r < mp_bits_per_limb; r++)
    {
      mpfr_urandomb (x, RANDS); /* to fill low limbs with random data */
      inexact = mpfr_set_f (x, y, MPFR_RNDN);
      MPFR_ASSERTN(inexact == 0 && mpfr_cmp_ui_2exp (x, 1, r) == 0);
      mpf_mul_2exp (y, y, 1);
    }

  mpf_set_ui (y, 1);
  mpf_mul_2exp (y, y, ULONG_MAX);
  mpfr_set_f (x, y, MPFR_RNDN);
  mpfr_set_ui (u, 1, MPFR_RNDN);
  mpfr_mul_2ui (u, u, ULONG_MAX, MPFR_RNDN);
  if (!mpfr_equal_p (x, u))
    {
      printf ("Error: mpfr_set_f (x, y, MPFR_RNDN) for y = 2^ULONG_MAX\n");
      exit (1);
    }

  emax = mpfr_get_emax ();

  /* For mpf_mul_2exp, emax must fit in an unsigned long! */
  if (emax >= 0 && emax <= ULONG_MAX)
    {
      mpf_set_ui (y, 1);
      mpf_mul_2exp (y, y, emax);
      mpfr_set_f (x, y, MPFR_RNDN);
      mpfr_set_ui_2exp (u, 1, emax, MPFR_RNDN);
      if (!mpfr_equal_p (x, u))
        {
          printf ("Error: mpfr_set_f (x, y, MPFR_RNDN) for y = 2^emax\n");
          exit (1);
        }
    }

  /* For mpf_mul_2exp, emax - 1 must fit in an unsigned long! */
  if (emax >= 1 && emax - 1 <= ULONG_MAX)
    {
      mpf_set_ui (y, 1);
      mpf_mul_2exp (y, y, emax - 1);
      mpfr_set_f (x, y, MPFR_RNDN);
      mpfr_set_ui_2exp (u, 1, emax - 1, MPFR_RNDN);
      if (!mpfr_equal_p (x, u))
        {
          printf ("Error: mpfr_set_f (x, y, MPFR_RNDN) for y = 2^(emax-1)\n");
          exit (1);
        }
    }

  mpfr_clear (x);
  mpfr_clear (u);
  mpf_clear (y);
  mpf_clear (z);

  tests_end_mpfr ();
  return 0;
}
