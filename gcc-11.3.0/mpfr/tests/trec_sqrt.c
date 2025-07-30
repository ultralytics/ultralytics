/* Test file for mpfr_rec_sqrt.

Copyright 2008-2017 Free Software Foundation, Inc.
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

#if MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0)

#define TEST_FUNCTION mpfr_rec_sqrt
#define TEST_RANDOM_POS 8 /* 8/512 = 1/64 of the tested numbers are negative */
#include "tgeneric.c"

static void
special (void)
{
  mpfr_t x, y;
  int inex;

  mpfr_init (x);
  mpfr_init (y);

  /* rec_sqrt(NaN) = NaN */
  mpfr_set_nan (x);
  inex = mpfr_rec_sqrt (x, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (x) && inex == 0);

  /* rec_sqrt(+Inf) = +0 */
  mpfr_set_inf (x, 1);
  inex = mpfr_rec_sqrt (x, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_zero_p (x) && MPFR_IS_POS(x) && inex == 0);

  /* rec_sqrt(-Inf) = NaN */
  mpfr_set_inf (x, -1);
  inex = mpfr_rec_sqrt (x, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (x) && inex == 0);

  /* rec_sqrt(+0) = +Inf */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  inex = mpfr_rec_sqrt (x, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (x) && MPFR_IS_POS(x) && inex == 0);

  /* rec_sqrt(-0) = +Inf */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  inex = mpfr_rec_sqrt (x, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (x) && MPFR_IS_POS(x) && inex == 0);

  /* rec_sqrt(-1) = NaN */
  mpfr_set_si (x, -1, MPFR_RNDN);
  inex = mpfr_rec_sqrt (x, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (x) && inex == 0);

  /* rec_sqrt(1) = 1 */
  mpfr_set_ui (x, 1, MPFR_RNDN);
  inex = mpfr_rec_sqrt (x, x, MPFR_RNDN);
  MPFR_ASSERTN((mpfr_cmp_ui (x, 1) == 0) && (inex == 0));

  mpfr_set_prec (x, 23);
  mpfr_set_prec (y, 33);
  mpfr_set_str_binary (x, "1.0001110110101001010100e-1");
  inex = mpfr_rec_sqrt (y, x, MPFR_RNDU);
  mpfr_set_prec (x, 33);
  mpfr_set_str_binary (x, "1.01010110101110100100100101011");
  MPFR_ASSERTN (inex > 0 && mpfr_cmp (x, y) == 0);

  mpfr_clear (x);
  mpfr_clear (y);
}

/* Worst case incorrectly rounded in r5573, found with the bad_cases test */
static void
bad_case1 (void)
{
  mpfr_t x, y, z;

  mpfr_init2 (x, 72);
  mpfr_inits2 (6, y, z, (mpfr_ptr) 0);
  mpfr_set_str (x, "1.08310518720928b30e@-120", 16, MPFR_RNDN);
  mpfr_set_str (z, "f.8@59", 16, MPFR_RNDN);
  /* z = rec_sqrt(x) rounded on 6 bits toward 0, the exact value
     being ~= f.bffffffffffffffffa11@59. */
  mpfr_rec_sqrt (y, x, MPFR_RNDZ);
  if (mpfr_cmp0 (y, z) != 0)
    {
      printf ("Error in bad_case1\nexpected ");
      mpfr_out_str (stdout, 16, 0, z, MPFR_RNDN);
      printf ("\ngot      ");
      mpfr_out_str (stdout, 16, 0, y, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  mpfr_clears (x, y, z, (mpfr_ptr) 0);
}

static int
pm2 (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd_mode)
{
  return mpfr_pow_si (y, x, -2, rnd_mode);
}

/* exercises corner cases with inputs around 1 or 2 */
static void
bad_case2 (void)
{
  mpfr_t r, u;
  mpfr_prec_t pr, pu;
  int rnd;

  for (pr = MPFR_PREC_MIN; pr <= 192; pr++)
    for (pu = MPFR_PREC_MIN; pu <= 192; pu++)
      {
        mpfr_init2 (r, pr);
        mpfr_init2 (u, pu);

        mpfr_set_ui (u, 1, MPFR_RNDN);
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          mpfr_rec_sqrt (r, u, (mpfr_rnd_t) rnd);

        mpfr_nextbelow (u);
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          mpfr_rec_sqrt (r, u, (mpfr_rnd_t) rnd);

        mpfr_nextbelow (u);
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          mpfr_rec_sqrt (r, u, (mpfr_rnd_t) rnd);

        mpfr_set_ui (u, 1, MPFR_RNDN);
        mpfr_nextabove (u);
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          mpfr_rec_sqrt (r, u, (mpfr_rnd_t) rnd);

        mpfr_nextabove (u);
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          mpfr_rec_sqrt (r, u, (mpfr_rnd_t) rnd);

        mpfr_set_ui (u, 2, MPFR_RNDN);
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          mpfr_rec_sqrt (r, u, (mpfr_rnd_t) rnd);

        mpfr_nextbelow (u);
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          mpfr_rec_sqrt (r, u, (mpfr_rnd_t) rnd);

        mpfr_nextbelow (u);
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          mpfr_rec_sqrt (r, u, (mpfr_rnd_t) rnd);

        mpfr_set_ui (u, 2, MPFR_RNDN);
        mpfr_nextabove (u);
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          mpfr_rec_sqrt (r, u, (mpfr_rnd_t) rnd);

        mpfr_nextabove (u);
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          mpfr_rec_sqrt (r, u, (mpfr_rnd_t) rnd);

        mpfr_clear (r);
        mpfr_clear (u);
      }
}

int
main (void)
{
  tests_start_mpfr ();

  special ();
  bad_case1 ();
  bad_case2 ();
  test_generic (2, 300, 15);

  data_check ("data/rec_sqrt", mpfr_rec_sqrt, "mpfr_rec_sqrt");
  bad_cases (mpfr_rec_sqrt, pm2, "mpfr_rec_sqrt", 8, -256, 255, 4, 128,
             800, 50);

  tests_end_mpfr ();
  return 0;
}

#else  /* MPFR_VERSION */

int
main (void)
{
  printf ("Warning! Test disabled for this MPFR version.\n");
  return 0;
}

#endif  /* MPFR_VERSION */
