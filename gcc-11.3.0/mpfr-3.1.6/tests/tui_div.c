/* Test file for mpfr_ui_div.

Copyright 2000-2017 Free Software Foundation, Inc.
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

/* checks that y/x gives the right result with 53 bits of precision */
static void
check (unsigned long y, const char *xs, mpfr_rnd_t rnd_mode, const char *zs)
{
  mpfr_t xx, zz;

  mpfr_inits2 (53, xx, zz, (mpfr_ptr) 0);
  mpfr_set_str1 (xx, xs);
  mpfr_ui_div (zz, y, xx, rnd_mode);
  if (mpfr_cmp_str1(zz, zs))
    {
      printf ("expected quotient is %s, got ", zs);
      mpfr_out_str (stdout, 10, 0, zz, MPFR_RNDN);
      printf ("mpfr_ui_div failed for y=%lu x=%s with rnd_mode=%s\n",
              y, xs, mpfr_print_rnd_mode (rnd_mode));
      exit (1);
    }
  mpfr_clears (xx, zz, (mpfr_ptr) 0);
}

static void
check_inexact (void)
{
  mpfr_t x, y, z;
  mpfr_prec_t px, py;
  int inexact, cmp;
  unsigned long int u;
  int rnd;

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);

  for (px = 2; px < 300; px++)
    {
      mpfr_set_prec (x, px);
      do
        {
          mpfr_urandomb (x, RANDS);
        }
      while (mpfr_cmp_ui (x, 0) == 0);
      u = randlimb ();
      for (py = 2; py < 300; py++)
        {
          mpfr_set_prec (y, py);
          mpfr_set_prec (z, py + px);
          for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
            {
              inexact = mpfr_ui_div (y, u, x, (mpfr_rnd_t) rnd);
              if (mpfr_mul (z, y, x, (mpfr_rnd_t) rnd))
                {
                  printf ("z <- y * x should be exact\n");
                  exit (1);
                }
              cmp = mpfr_cmp_ui (z, u);
              if (! SAME_SIGN (inexact, cmp))
                {
                  printf ("Wrong inexact flag for u=%lu, rnd=%s\n",
                          u, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  printf ("expected %d, got %d\n", cmp, inexact);
                  printf ("x = "); mpfr_dump (x);
                  printf ("y = "); mpfr_dump (y);
                  printf ("y*x = "); mpfr_dump (z);
                  exit (1);
                }
            }
        }
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

static void
check_special (void)
{
  mpfr_t  d, q;

  mpfr_init2 (d, 100L);
  mpfr_init2 (q, 100L);

  /* 1/+inf == 0 */
  MPFR_SET_INF (d);
  MPFR_SET_POS (d);
  mpfr_clear_flags ();
  MPFR_ASSERTN (mpfr_ui_div (q, 1L, d, MPFR_RNDZ) == 0); /* exact */
  MPFR_ASSERTN (mpfr_number_p (q));
  MPFR_ASSERTN (mpfr_sgn (q) == 0);
  MPFR_ASSERTN (__gmpfr_flags == 0);

  /* 1/-inf == -0 */
  MPFR_SET_INF (d);
  MPFR_SET_NEG (d);
  mpfr_clear_flags ();
  MPFR_ASSERTN (mpfr_ui_div (q, 1L, d, MPFR_RNDZ) == 0); /* exact */
  MPFR_ASSERTN (mpfr_number_p (q));
  MPFR_ASSERTN (mpfr_sgn (q) == 0);
  MPFR_ASSERTN (__gmpfr_flags == 0);

  /* 1/nan == nan */
  MPFR_SET_NAN (d);
  mpfr_clear_flags ();
  MPFR_ASSERTN (mpfr_ui_div (q, 1L, d, MPFR_RNDZ) == 0); /* exact */
  MPFR_ASSERTN (mpfr_nan_p (q));
  MPFR_ASSERTN (__gmpfr_flags == MPFR_FLAGS_NAN);

  /* 0/0 == nan */
  mpfr_set_ui (d, 0L, MPFR_RNDN);
  mpfr_clear_flags ();
  MPFR_ASSERTN (mpfr_ui_div (q, 0L, d, MPFR_RNDZ) == 0); /* exact */
  MPFR_ASSERTN (mpfr_nan_p (q));
  MPFR_ASSERTN (__gmpfr_flags == MPFR_FLAGS_NAN);

  /* 1/+0 = +inf */
  mpfr_set_ui (d, 0L, MPFR_RNDN);
  mpfr_clear_flags ();
  MPFR_ASSERTN (mpfr_ui_div (q, 1L, d, MPFR_RNDZ) == 0); /* exact */
  MPFR_ASSERTN (mpfr_inf_p (q) && mpfr_sgn (q) > 0);
  MPFR_ASSERTN (__gmpfr_flags == MPFR_FLAGS_DIVBY0);

  /* 1/-0 = -inf */
  mpfr_set_ui (d, 0L, MPFR_RNDN);
  mpfr_neg (d, d, MPFR_RNDN);
  mpfr_clear_flags ();
  MPFR_ASSERTN (mpfr_ui_div (q, 1L, d, MPFR_RNDZ) == 0); /* exact */
  MPFR_ASSERTN (mpfr_inf_p (q) && mpfr_sgn (q) < 0);
  MPFR_ASSERTN (__gmpfr_flags == MPFR_FLAGS_DIVBY0);

  /* 0/1 = +0 */
  mpfr_set_ui (d, 1L, MPFR_RNDN);
  mpfr_clear_flags ();
  MPFR_ASSERTN (mpfr_ui_div (q, 0L, d, MPFR_RNDZ) == 0); /* exact */
  MPFR_ASSERTN (mpfr_cmp_ui (q, 0) == 0 && MPFR_IS_POS (q));
  MPFR_ASSERTN (__gmpfr_flags == 0);

  /* 0/-1 = -0 */
  mpfr_set_si (d, -1, MPFR_RNDN);
  mpfr_clear_flags ();
  MPFR_ASSERTN (mpfr_ui_div (q, 0L, d, MPFR_RNDZ) == 0); /* exact */
  MPFR_ASSERTN (mpfr_cmp_ui (q, 0) == 0 && MPFR_IS_NEG (q));
  MPFR_ASSERTN (__gmpfr_flags == 0);

  mpfr_clear (d);
  mpfr_clear (q);
}

static int
mpfr_inv (mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t r)
{
  return mpfr_ui_div (y, 1, x, r);
}

static void
check_overflow (void)
{
  mpfr_exp_t emin, emax;
  mpfr_t x, y1, y2;
  int inex1, inex2, rnd_mode;
  unsigned int flags1, flags2;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  set_emin (MPFR_EMIN_MIN);
  set_emax (MPFR_EMAX_MAX);

  mpfr_inits2 (32, x, y1, y2, (mpfr_ptr) 0);
  mpfr_setmin (x, MPFR_EMIN_MIN);
  RND_LOOP (rnd_mode)
    {
      inex1 = mpfr_overflow (y1, (mpfr_rnd_t) rnd_mode, 1);
      flags1 = MPFR_FLAGS_OVERFLOW | MPFR_FLAGS_INEXACT;
      mpfr_clear_flags ();
      inex2 = mpfr_ui_div (y2, 1, x, (mpfr_rnd_t) rnd_mode);
      flags2 = __gmpfr_flags;
      if (!(mpfr_equal_p (y1, y2) &&
            SAME_SIGN (inex1, inex2) &&
            flags1 == flags2))
        {
          printf ("Error in check_overflow for %s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) rnd_mode));
          printf ("Expected ");
          mpfr_dump (y1);
          printf ("  with inex = %d, flags =", inex1);
          flags_out (flags1);
          printf ("Got      ");
          mpfr_dump (y2);
          printf ("  with inex = %d, flags =", inex2);
          flags_out (flags2);
          exit (1);
        }
    }
  mpfr_clears (x, y1, y2, (mpfr_ptr) 0);

  set_emin (emin);
  set_emax (emax);
}

#define TEST_FUNCTION mpfr_ui_div
#define ULONG_ARG1
#define RAND_FUNCTION(x) mpfr_random2(x, MPFR_LIMB_SIZE (x), 1, RANDS)
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  check_special ();
  check_inexact ();
  check(948002822, "1.22191250737771397120e+20", MPFR_RNDN,
        "7.758352715731357946e-12");
  check(1976245324, "1.25296395864546893357e+232", MPFR_RNDZ,
        "1.5772563211925444801e-223");
  check(740454110, "2.11496253355831863313e+183", MPFR_RNDZ,
        "3.5010270784996976041e-175");
  check(1690540942, "1.28278599852446657468e-276", MPFR_RNDU,
        "1.3178666932321966062e285");
  check(1476599377, "-2.14191393656148625995e+305", MPFR_RNDD,
        "-6.8938315017943889615e-297");
  check_overflow ();

  test_generic (2, 1000, 100);

  /* inv is for 1/x */
  data_check ("data/inv", mpfr_inv, "mpfr_ui_div(1,x)");

  tests_end_mpfr ();
  return 0;
}
