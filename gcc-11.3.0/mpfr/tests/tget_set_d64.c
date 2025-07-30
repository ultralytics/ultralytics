/* Test file for mpfr_get_decimal64 and mpfr_set_decimal64.

Copyright 2006-2017 Free Software Foundation, Inc.
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

#ifdef MPFR_WANT_DECIMAL_FLOATS

#include <stdlib.h> /* for exit */
#include "mpfr-test.h"

#ifndef DEC64_MAX
# define DEC64_MAX 9.999999999999999E384dd
#endif

/* #define DEBUG */

static void
print_decimal64 (_Decimal64 d)
{
  union ieee_double_extract x;
  union ieee_double_decimal64 y;
  unsigned int Gh, i;

  y.d64 = d;
  x.d = y.d;
  Gh = x.s.exp >> 6;
  printf ("|%d%d%d%d%d%d", x.s.sig, Gh >> 4, (Gh >> 3) & 1,
          (Gh >> 2) & 1, (Gh >> 1) & 1, Gh & 1);
  printf ("%d%d%d%d%d%d", (x.s.exp >> 5) & 1, (x.s.exp >> 4) & 1,
          (x.s.exp >> 3) & 1, (x.s.exp >> 2) & 1, (x.s.exp >> 1) & 1,
          x.s.exp & 1);
  for (i = 20; i > 0; i--)
    printf ("%d", (x.s.manh >> (i - 1)) & 1);
  for (i = 32; i > 0; i--)
    printf ("%d", (x.s.manl >> (i - 1)) & 1);
  printf ("|\n");
}

static void
check_inf_nan (void)
{
  mpfr_t  x, y;
  _Decimal64 d;

  mpfr_init2 (x, 123);
  mpfr_init2 (y, 123);

  mpfr_set_nan (x);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 1, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_nan_p (x));

  mpfr_set_inf (x, 1);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 1, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_inf_p (x) && mpfr_sgn (x) > 0);

  mpfr_set_inf (x, -1);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 1, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_inf_p (x) && mpfr_sgn (x) < 0);

  mpfr_set_ui (x, 0, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 1, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp_ui (x, 0) == 0 && MPFR_SIGN (x) > 0);

  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_neg (x, x, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 1, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp_ui (x, 0) == 0 && MPFR_SIGN (x) < 0);

  mpfr_set_ui (x, 1, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp_ui (x, 1) == 0);

  mpfr_set_si (x, -1, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp_si (x, -1) == 0);

  mpfr_set_ui (x, 2, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp_ui (x, 2) == 0);

  mpfr_set_ui (x, 99, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp_ui (x, 99) == 0);

  mpfr_set_str (x, "9999999999999999", 10, MPFR_RNDZ);
  mpfr_set (y, x, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp (x, y) == 0);

  /* smallest normal number */
  mpfr_set_str (x, "1E-383", 10, MPFR_RNDU);
  mpfr_set (y, x, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDU);
  MPFR_ASSERTN (mpfr_cmp (x, y) == 0);

  /* smallest subnormal number */
  mpfr_set_str (x, "1E-398", 10, MPFR_RNDU);
  mpfr_set (y, x, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDU);
  MPFR_ASSERTN (mpfr_cmp (x, y) == 0);

  /* subnormal number with exponent change when we round back
     from 16 digits to 1 digit */
  mpfr_set_str (x, "9.9E-398", 10, MPFR_RNDN);
  d = mpfr_get_decimal64 (x, MPFR_RNDU); /* should be 1E-397 */
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDD);
  mpfr_set_str (y, "1E-397", 10, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp (x, y) == 0);

  /* largest number */
  mpfr_set_str (x, "9.999999999999999E384", 10, MPFR_RNDZ);
  mpfr_set (y, x, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDU);
  MPFR_ASSERTN (d == DEC64_MAX);
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp (x, y) == 0);

  mpfr_set_str (x, "-9.999999999999999E384", 10, MPFR_RNDZ);
  mpfr_set (y, x, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDA);
  MPFR_ASSERTN (d == -DEC64_MAX);
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_set_decimal64 (x, d, MPFR_RNDZ);
  MPFR_ASSERTN (mpfr_cmp (x, y) == 0);

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

  /* largest number */
  mpfr_set_str (x, "9.999999999999999E384", 10, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDZ);
  mpfr_set_decimal64 (y, d, MPFR_RNDU);
  MPFR_ASSERTN (mpfr_cmp (x, y) == 0);

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
check_random (void)
{
  mpfr_t  x, y;
  _Decimal64 d;
  int i;

  mpfr_init2 (x, 49);
  mpfr_init2 (y, 49);

  for (i = 0; i < 100000; i++)
    {
      mpfr_urandomb (x, RANDS); /* 0 <= x < 1 */
      /* the normal decimal64 range contains [2^(-1272), 2^1278] */
      mpfr_mul_2si (x, x, (i % 2550) - 1272, MPFR_RNDN);
      if (mpfr_get_exp (x) <= -1272)
        mpfr_mul_2exp (x, x, -1271 - mpfr_get_exp (x), MPFR_RNDN);
      d = mpfr_get_decimal64 (x, MPFR_RNDN);
      mpfr_set_decimal64 (y, d, MPFR_RNDN);
      if (mpfr_cmp (x, y) != 0)
        {
          printf ("x="); mpfr_dump (x);
          printf ("d="); print_decimal64 (d);
          printf ("y="); mpfr_dump (y);
          exit (1);
        }
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

/* check with native decimal formats */
static void
check_native (void)
{
  mpfr_t x;
  _Decimal64 d;

  mpfr_init2 (x, 53);

  /* check important constants are correctly converted */
  mpfr_set_ui (x, 17, MPFR_RNDN);
  d = mpfr_get_decimal64 (x, MPFR_RNDN);
  MPFR_ASSERTN(d == 17.0dd);

  mpfr_set_ui (x, 42, MPFR_RNDN);
  d = mpfr_get_decimal64 (x, MPFR_RNDN);
  MPFR_ASSERTN(d == 42.0dd);

  mpfr_set_decimal64 (x, 17.0dd, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 17) == 0);

  mpfr_set_decimal64 (x, 42.0dd, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 42) == 0);

  mpfr_clear (x);
}

static void
check_overflow (void)
{
  mpfr_t x;
  int err = 0, neg, rnd;

  mpfr_init2 (x, 96);
  for (neg = 0; neg < 2; neg++)
    RND_LOOP (rnd)
      {
        _Decimal64 d, e;
        mpfr_rnd_t r = (mpfr_rnd_t) rnd;
        int sign = neg ? -1 : 1;

        e = sign * (MPFR_IS_LIKE_RNDZ (r, neg) ? 1 : 2) * DEC64_MAX;
        /* This tests the binary exponent e > 1279 case of get_d64.c */
        mpfr_set_si_2exp (x, sign, 9999, MPFR_RNDN);
        d = mpfr_get_decimal64 (x, r);
        if (d != e)
          {
            printf ("Error 1 in check_overflow for %s, %s\n",
                    neg ? "negative" : "positive",
                    mpfr_print_rnd_mode (r));
            err = 1;
          }
        /* This tests the decimal exponent e > 385 case of get_d64.c */
        mpfr_set_si_2exp (x, sign * 31, 1274, MPFR_RNDN);
        d = mpfr_get_decimal64 (x, r);
        if (d != e)
          {
            printf ("Error 2 in check_overflow for %s, %s\n",
                    neg ? "negative" : "positive",
                    mpfr_print_rnd_mode (r));
            err = 1;
          }
        /* This tests the last else (-382 <= e <= 385) of get_d64.c */
        mpfr_set_decimal64 (x, e, MPFR_RNDA);
        d = mpfr_get_decimal64 (x, r);
        if (d != e)
          {
            printf ("Error 3 in check_overflow for %s, %s\n",
                    neg ? "negative" : "positive",
                    mpfr_print_rnd_mode (r));
            err = 1;
          }
      }
  mpfr_clear (x);
  if (err)
    exit (1);
}

static void
check_tiny (void)
{
  mpfr_t x;
  _Decimal64 d;

  /* If 0.5E-398 < |x| < 1E-398 (smallest subnormal), x should round
     to +/- 1E-398 in MPFR_RNDN. Note: the midpoint 0.5E-398 between
     0 and 1E-398 is not a representable binary number, so that there
     are no tests for it. */
  mpfr_init2 (x, 128);
  mpfr_set_str (x, "1E-398", 10, MPFR_RNDZ);
  d = mpfr_get_decimal64 (x, MPFR_RNDN);
  MPFR_ASSERTN (d == 1.0E-398dd);
  mpfr_neg (x, x, MPFR_RNDN);
  d = mpfr_get_decimal64 (x, MPFR_RNDN);
  MPFR_ASSERTN (d == -1.0E-398dd);
  mpfr_set_str (x, "0.5E-398", 10, MPFR_RNDU);
  d = mpfr_get_decimal64 (x, MPFR_RNDN);
  MPFR_ASSERTN (d == 1.0E-398dd);
  mpfr_neg (x, x, MPFR_RNDN);
  d = mpfr_get_decimal64 (x, MPFR_RNDN);
  MPFR_ASSERTN (d == -1.0E-398dd);
  mpfr_clear (x);
}

int
main (void)
{
  tests_start_mpfr ();
  mpfr_test_init ();

#ifdef DEBUG
#ifdef DPD_FORMAT
  printf ("Using DPD format\n");
#else
  printf ("Using BID format\n");
#endif
#endif
  check_inf_nan ();
  check_random ();
  check_native ();
  check_overflow ();
  check_tiny ();

  tests_end_mpfr ();
  return 0;
}

#else /* MPFR_WANT_DECIMAL_FLOATS */

int
main (void)
{
  return 77;
}

#endif /* MPFR_WANT_DECIMAL_FLOATS */
