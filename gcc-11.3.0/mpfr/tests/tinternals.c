/* tinternals -- Test for internals.

Copyright 2005-2017 Free Software Foundation, Inc.
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

#define MPFR_NEED_LONGLONG_H
#include "mpfr-test.h"

static void
test_int_ceil_log2 (void)
{
  int i;
  int val[16] = { 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4 };

  for (i = 1; i < 17; i++)
    {
      if (MPFR_INT_CEIL_LOG2 (i) != val[i-1])
        {
          printf ("Error 1 in test_int_ceil_log2 for i = %d\n", i);
          exit (1);
        }
      if (MPFR_INT_CEIL_LOG2 (i) != __gmpfr_int_ceil_log2 (i))
        {
          printf ("Error 2 in test_int_ceil_log2 for i = %d\n", i);
          exit (1);
        }
    }
}

static void
test_round_near_x (void)
{
  mpfr_t x, y, z, eps;
  mpfr_exp_t e;
  int failures = 0, mx, neg, err, dir, r, inex, inex2;
  char buffer[7], *p;

  mpfr_inits (x, y, z, eps, (mpfr_ptr) 0);
  mpfr_set_prec (x, 5);
  mpfr_set_prec (y, 3);
  mpfr_set_prec (z, 3);
  mpfr_set_prec (eps, 2);
  mpfr_set_ui_2exp (eps, 1, -32, MPFR_RNDN);

  for (mx = 16; mx < 32; mx++)
    {
      mpfr_set_ui_2exp (x, mx, -2, MPFR_RNDN);
      for (p = buffer, neg = 0;
           neg <= 1;
           mpfr_neg (x, x, MPFR_RNDN), p++, neg++)
        for (err = 2; err <= 6; err++)
          for (dir = 0; dir <= 1; dir++)
            RND_LOOP(r)
              {
                inex = mpfr_round_near_x (y, x, err, dir, (mpfr_rnd_t) r);

                if (inex == 0 && err < 6)
                  {
                    /* The test is more restrictive than necessary.
                       So, no failure in this case. */
                    continue;
                  }

                inex2 = ((dir ^ neg) ? mpfr_add : mpfr_sub)
                  (z, x, eps, (mpfr_rnd_t) r);
                if (inex * inex2 <= 0)
                  printf ("Bad return value (%d instead of %d) for:\n",
                          inex, inex2);
                else if (mpfr_equal_p (y, z))
                  continue;  /* correct inex and y */
                else
                  {
                    printf ("Bad MPFR value (should have got ");
                    mpfr_out_str (stdout, 2, 3, z, MPFR_RNDZ);
                    printf (") for:\n");
                  }

                if (!mpfr_get_str (buffer, &e, 2, 5, x, MPFR_RNDZ) || e != 3)
                  {
                    printf ("mpfr_get_str failed in test_round_near_x\n");
                    exit (1);
                  }
                printf ("x = %c%c%c%c.%c%c, ", neg ? '-' : '+',
                        p[0], p[1], p[2], p[3], p[4]);
                printf ("err = %d, dir = %d, r = %s --> inex = %2d",
                        err, dir, mpfr_print_rnd_mode ((mpfr_rnd_t) r), inex);
                if (inex != 0)
                  {
                    printf (", y = ");
                    mpfr_out_str (stdout, 2, 3, y, MPFR_RNDZ);
                  }
                printf ("\n");
                if (inex == 0)
                  printf ("Rounding was possible!\n");
                if (++failures == 10)  /* show at most 10 failures */
                  exit (1);
              }
    }

  if (failures)
    exit (1);

  mpfr_clears (x, y, z, eps, (mpfr_ptr) 0);
}

static void
test_set_prec_raw (void)
{
  mpfr_t x;
  int i;

  mpfr_init2 (x, 53);
  for (i = 2; i < 11; i++)
    {
      mpfr_set_prec_raw (x, i);
      if (MPFR_PREC (x) != i)
        {
          printf ("[ERROR]: mpfr_set_prec_raw %d\n", i);
          exit (1);
        }
    }
  mpfr_clear (x);
}

int
main (int argc, char **argv)
{
  tests_start_mpfr ();

  /* The tested function and macro exist in MPFR 2.2.0, but with a
     different (incorrect, but with no effect in 2.2.0) behavior. */
#if MPFR_VERSION >= MPFR_VERSION_NUM(2,3,0)
  test_int_ceil_log2 ();
#endif

  test_round_near_x ();
  test_set_prec_raw ();

  tests_end_mpfr ();
  return 0;
}
