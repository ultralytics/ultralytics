/* Test file for mpfr_nextabove, mpfr_nextbelow, mpfr_nexttoward.

Copyright 2003-2017 Free Software Foundation, Inc.
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

/* Generic tests for mpfr_nextabove and mpfr_nextbelow */
static void
generic_abovebelow (void)
{
  int i;

  for (i = 0; i < 20000; i++)
    {
      mpfr_t x, y, z, t;
      mpfr_prec_t prec;
      int neg, below;

      prec = (randlimb () % 300) + MPFR_PREC_MIN;
      mpfr_inits2 (prec, x, y, z, (mpfr_ptr) 0);
      mpfr_init2 (t, 3);

      /* special tests (executed once is enough) */
      if (i == 0)
        {
          mpfr_set_nan (x);
          mpfr_nextabove (x);
          MPFR_ASSERTN(mpfr_nan_p (x));
          mpfr_nextbelow (x);
          MPFR_ASSERTN(mpfr_nan_p (x));
          mpfr_nexttoward (x, y);
          MPFR_ASSERTN(mpfr_nan_p (x));
          mpfr_set_ui (y, 1, MPFR_RNDN);
          mpfr_nexttoward (y, x);
          MPFR_ASSERTN(mpfr_nan_p (y));
          mpfr_set_ui (x, 1, MPFR_RNDN);
          mpfr_set_ui (y, 1, MPFR_RNDN);
          mpfr_nexttoward (x, y);
          MPFR_ASSERTN(mpfr_cmp_ui (x, 1) == 0);
        }

      do
        mpfr_urandomb (x, RANDS);
      while (mpfr_cmp_ui (x, 0) == 0);
      neg = randlimb () & 1;
      if (neg)
        mpfr_neg (x, x, MPFR_RNDN);
      mpfr_set (y, x, MPFR_RNDN);
      below = randlimb () & 1;
      if (below)
        mpfr_nextbelow (y);
      else
        mpfr_nextabove (y);
      mpfr_set_si (t, below ? -5 : 5, MPFR_RNDN);
      mpfr_mul_2si (t, t, (mpfr_get_exp) (x) - prec - 3, MPFR_RNDN);
      /* t = (1/2 + 1/8) ulp(x) */
      mpfr_add (z, x, t, MPFR_RNDN);
      if (!mpfr_number_p (y) || mpfr_cmp (y, z) != 0)
        {
          printf ("Error in mpfr_next%s for\n",
                  below ? "below" : "above");
          mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
          printf (", got\n");
          mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
          printf (" instead of\n");
          mpfr_out_str (stdout, 2, 0, z, MPFR_RNDN);
          printf ("\n");
          exit (1);
        }
      mpfr_clears (x, y, z, t, (mpfr_ptr) 0);
    }
}

static void
inverse_test (void)
{
  static const char *tests[] = { "0", "1", "2", "3.1", "Inf" };
  int i, neg, below;
  mpfr_prec_t prec;

  for (i = 0; i < (int) (sizeof(tests) / sizeof(tests[0])); i++)
    for (neg = 0; neg <= 1; neg++)
      for (below = 0; below <= 1; below++)
        for (prec = MPFR_PREC_MIN; prec < 200; prec += 3)
          {
            mpfr_t x, y;
            int sign;

            mpfr_inits2 (prec, x, y, (mpfr_ptr) 0);
            mpfr_set_str (x, tests[i], 10, MPFR_RNDN);
            if (neg)
              mpfr_neg (x, x, MPFR_RNDN);
            mpfr_set (y, x, MPFR_RNDN);
            if (below)
              mpfr_nextbelow (y);
            else
              mpfr_nextabove (y);
            sign = MPFR_SIGN (y);
            if (!(neg == below && mpfr_inf_p (x)))  /* then x = y */
              {
                if (mpfr_cmp (x, y) == 0)
                  {
                    printf ("Error in inverse_test for %s, neg = %d,"
                            " below = %d, prec = %d: x = y", tests[i],
                            neg, below, (int) prec);
                    printf ("\nx = ");
                    mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
                    printf ("\ny = ");
                    mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
                    printf ("\n");
                    exit (1);
                 }
                mpfr_nexttoward (y, x);
                if (mpfr_cmp_ui (y, 0) == 0 && MPFR_SIGN (y) != sign)
                  {
                    printf ("Sign error in inverse_test for %s, neg = %d,"
                            " below = %d, prec = %d\n", tests[i], neg,
                            below, (int) prec);
                    mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
                    printf ("\n");
                    exit (1);
                  }
              }
            if (mpfr_cmp (x, y) != 0)
              {
                printf ("Error in inverse_test for %s, neg = %d, below = %d,"
                        " prec = %d", tests[i], neg, below, (int) prec);
                printf ("\nx = ");
                mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
                printf ("\ny = ");
                mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
                printf ("\n");
                exit (1);
              }
            mpfr_clears (x, y, (mpfr_ptr) 0);
          }
}

int
main (void)
{
  tests_start_mpfr ();
  generic_abovebelow ();
  inverse_test ();
  tests_end_mpfr ();
  return 0;
}
