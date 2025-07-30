/* Test file for mpfr_ui_sub.

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
#include <float.h>

#include "mpfr-test.h"

static void
special (void)
{
  mpfr_t x, y, res;
  int inexact;

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (res);

  mpfr_set_prec (x, 24);
  mpfr_set_prec (y, 24);
  mpfr_set_str_binary (y, "0.111100110001011010111");
  inexact = mpfr_ui_sub (x, 1, y, MPFR_RNDN);
  if (inexact)
    {
      printf ("Wrong inexact flag: got %d, expected 0\n", inexact);
      exit (1);
    }

  mpfr_set_prec (x, 24);
  mpfr_set_prec (y, 24);
  mpfr_set_str_binary (y, "0.111100110001011010111");
  if ((inexact = mpfr_ui_sub (x, 38181761, y, MPFR_RNDN)) >= 0)
    {
      printf ("Wrong inexact flag: got %d, expected -1\n", inexact);
      exit (1);
    }

  mpfr_set_prec (x, 63);
  mpfr_set_prec (y, 63);
  mpfr_set_str_binary (y, "0.111110010010100100110101101010001001100101110001000101110111111E-1");
  if ((inexact = mpfr_ui_sub (x, 1541116494, y, MPFR_RNDN)) <= 0)
    {
      printf ("Wrong inexact flag: got %d, expected +1\n", inexact);
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_set_prec (y, 32);
  mpfr_set_str_binary (y, "0.11011000110111010001011100011100E-1");
  if ((inexact = mpfr_ui_sub (x, 2000375416, y, MPFR_RNDN)) >= 0)
    {
      printf ("Wrong inexact flag: got %d, expected -1\n", inexact);
      exit (1);
    }

  mpfr_set_prec (x, 24);
  mpfr_set_prec (y, 24);
  mpfr_set_str_binary (y, "0.110011011001010011110111E-2");
  if ((inexact = mpfr_ui_sub (x, 927694848, y, MPFR_RNDN)) <= 0)
    {
      printf ("Wrong inexact flag: got %d, expected +1\n", inexact);
      exit (1);
    }

  /* bug found by Mathieu Dutour, 12 Apr 2001 */
  mpfr_set_prec (x, 5);
  mpfr_set_prec (y, 5);
  mpfr_set_prec (res, 5);
  mpfr_set_str_binary (x, "1e-12");

  mpfr_ui_sub (y, 1, x, MPFR_RNDD);
  mpfr_set_str_binary (res, "0.11111");
  if (mpfr_cmp (y, res))
    {
      printf ("Error in mpfr_ui_sub (y, 1, x, MPFR_RNDD) for x=2^(-12)\nexpected 1.1111e-1, got ");
      mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_ui_sub (y, 1, x, MPFR_RNDU);
  mpfr_set_str_binary (res, "1.0");
  if (mpfr_cmp (y, res))
    {
      printf ("Error in mpfr_ui_sub (y, 1, x, MPFR_RNDU) for x=2^(-12)\n"
              "expected 1.0, got ");
      mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_ui_sub (y, 1, x, MPFR_RNDN);
  mpfr_set_str_binary (res, "1.0");
  if (mpfr_cmp (y, res))
    {
      printf ("Error in mpfr_ui_sub (y, 1, x, MPFR_RNDN) for x=2^(-12)\n"
              "expected 1.0, got ");
      mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_set_prec (x, 10);
  mpfr_set_prec (y, 10);
  mpfr_urandomb (x, RANDS);
  mpfr_ui_sub (y, 0, x, MPFR_RNDN);
  if (MPFR_IS_ZERO(x))
    MPFR_ASSERTN(MPFR_IS_ZERO(y));
  else
    MPFR_ASSERTN(mpfr_cmpabs (x, y) == 0 && mpfr_sgn (x) != mpfr_sgn (y));

  mpfr_set_prec (x, 73);
  mpfr_set_str_binary (x, "0.1101111010101011011011100011010000000101110001011111001011011000101111101E-99");
  mpfr_ui_sub (x, 1, x, MPFR_RNDZ);
  mpfr_nextabove (x);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 1) == 0);

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (res);
}

/* checks that (y-x) gives the right results with 53 bits of precision */
static void
check (unsigned long y, const char *xs, mpfr_rnd_t rnd_mode, const char *zs)
{
  mpfr_t xx, zz;

  mpfr_inits2 (53, xx, zz, (mpfr_ptr) 0);
  mpfr_set_str1 (xx, xs);
  mpfr_ui_sub (zz, y, xx, rnd_mode);
  if (mpfr_cmp_str1 (zz, zs) )
    {
      printf ("expected difference is %s, got\n",zs);
      mpfr_out_str(stdout, 10, 0, zz, MPFR_RNDN);
      printf ("mpfr_ui_sub failed for y=%lu x=%s with rnd_mode=%s\n",
              y, xs, mpfr_print_rnd_mode (rnd_mode));
      exit (1);
    }
  mpfr_clears (xx, zz, (mpfr_ptr) 0);
}

/* if u = o(x-y), v = o(u-x), w = o(v+y), then x-y = u-w */
static void
check_two_sum (mpfr_prec_t p)
{
  unsigned int x;
  mpfr_t y, u, v, w;
  mpfr_rnd_t rnd;
  int inexact, cmp;

  mpfr_inits2 (p, y, u, v, w, (mpfr_ptr) 0);
  do
    {
      x = randlimb ();
    }
  while (x < 1);
  mpfr_urandomb (y, RANDS);
  rnd = MPFR_RNDN;
  inexact = mpfr_ui_sub (u, x, y, rnd);
  mpfr_sub_ui (v, u, x, rnd);
  mpfr_add (w, v, y, rnd);
  cmp = mpfr_cmp_ui (w, 0);
  /* as u = (x-y) + w, we should have inexact and w of same sign */
  if (! SAME_SIGN (inexact, cmp))
    {
      printf ("Wrong inexact flag for prec=%u, rnd=%s\n",
              (unsigned int) p, mpfr_print_rnd_mode (rnd));
      printf ("x = %u\n", x);
      printf ("y = "); mpfr_dump (y);
      printf ("u = "); mpfr_dump (u);
      printf ("v = "); mpfr_dump (v);
      printf ("w = "); mpfr_dump (w);
      printf ("inexact = %d\n", inexact);
      exit (1);
    }
  mpfr_clears (y, u, v, w, (mpfr_ptr) 0);
}

static void
check_nans (void)
{
  mpfr_t  x, y;

  mpfr_init2 (x, 123L);
  mpfr_init2 (y, 123L);

  /* 1 - nan == nan */
  mpfr_set_nan (x);
  mpfr_ui_sub (y, 1L, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (y));

  /* 1 - +inf == -inf */
  mpfr_set_inf (x, 1);
  mpfr_ui_sub (y, 1L, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (y));
  MPFR_ASSERTN (mpfr_sgn (y) < 0);

  /* 1 - -inf == +inf */
  mpfr_set_inf (x, -1);
  mpfr_ui_sub (y, 1L, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (y));
  MPFR_ASSERTN (mpfr_sgn (y) > 0);

  mpfr_clear (x);
  mpfr_clear (y);
}

/* Check mpfr_ui_sub with u = 0 (unsigned). */
static void check_neg (void)
{
  mpfr_t x, yneg, ysub;
  int i, s;
  int r;

  mpfr_init2 (x, 64);
  mpfr_init2 (yneg, 32);
  mpfr_init2 (ysub, 32);

  for (i = 0; i <= 25; i++)
    {
      mpfr_sqrt_ui (x, i, MPFR_RNDN);
      for (s = 0; s <= 1; s++)
        {
          RND_LOOP (r)
            {
              int tneg, tsub;

              tneg = mpfr_neg (yneg, x, (mpfr_rnd_t) r);
              tsub = mpfr_ui_sub (ysub, 0, x, (mpfr_rnd_t) r);
              MPFR_ASSERTN (mpfr_equal_p (yneg, ysub));
              MPFR_ASSERTN (!(MPFR_IS_POS (yneg) ^ MPFR_IS_POS (ysub)));
              MPFR_ASSERTN (tneg == tsub);
            }
          mpfr_neg (x, x, MPFR_RNDN);
        }
    }

  mpfr_clear (x);
  mpfr_clear (yneg);
  mpfr_clear (ysub);
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
  mpfr_setmax (x, MPFR_EMAX_MAX);
  mpfr_neg (x, x, MPFR_RNDN);
  RND_LOOP (rnd_mode)
  {
    if (rnd_mode == MPFR_RNDU || rnd_mode == MPFR_RNDA)
      {
        inex1 = mpfr_overflow (y1, (mpfr_rnd_t) rnd_mode, 1);
        flags1 = MPFR_FLAGS_OVERFLOW | MPFR_FLAGS_INEXACT;
      }
    else
      {
        mpfr_neg (y1, x, MPFR_RNDN);
        inex1 = -1;
        flags1 = MPFR_FLAGS_INEXACT;
      }
    mpfr_clear_flags ();
    inex2 = mpfr_ui_sub (y2, 1, x, (mpfr_rnd_t) rnd_mode);
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

#define TEST_FUNCTION mpfr_ui_sub
#define ULONG_ARG1
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  mpfr_prec_t p;
  unsigned k;

  tests_start_mpfr ();

  check_nans ();

  special ();
  for (p=2; p<100; p++)
    for (k=0; k<100; k++)
      check_two_sum (p);

  check(1196426492, "1.4218093058435347e-3", MPFR_RNDN,
        "1.1964264919985781e9");
  check(1092583421, "-1.0880649218158844e9", MPFR_RNDN,
        "2.1806483428158845901e9");
  check(948002822, "1.22191250737771397120e+20", MPFR_RNDN,
        "-1.2219125073682338611e20");
  check(832100416, "4.68311314939691330000e-215", MPFR_RNDD,
        "8.3210041599999988079e8");
  check(1976245324, "1.25296395864546893357e+232", MPFR_RNDZ,
        "-1.2529639586454686577e232");
  check(2128997392, "-1.08496826129284207724e+187", MPFR_RNDU,
        "1.0849682612928422704e187");
  check(293607738, "-1.9967571564050541e-5", MPFR_RNDU,
        "2.9360773800002003e8");
  check(354270183, "2.9469161763489528e3", MPFR_RNDN,
        "3.5426723608382362e8");
  check_overflow ();

  check_neg ();

  test_generic (2, 1000, 100);

  tests_end_mpfr ();
  return 0;
}
