/* Test file for mpfr_get_d

Copyright 1999-2017 Free Software Foundation, Inc.
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
#include "ieee_floats.h"

static int
check_denorms (void)
{
  mpfr_rnd_t rnd_mode;
  mpfr_t x;
  double d, d2, dd, f;
  int fail = 0, k, n;

  mpfr_init2 (x, GMP_NUMB_BITS);

  rnd_mode = MPFR_RNDN;
  for (k = -17; k <= 17; k += 2)
    {
      d = (double) k * DBL_MIN; /* k * 2^(-1022) */
      f = 1.0;
      mpfr_set_si (x, k, MPFR_RNDN);
      mpfr_div_2exp (x, x, 1022, MPFR_RNDN); /* k * 2^(-1022) */
      for (n = 0; n <= 58; n++)
        {
          d2 = d * f;
          dd = mpfr_get_d (x, rnd_mode);
          if (d2 != dd) /* should be k * 2^(-1022-n) for n < 53 */
            {
              printf ("Wrong result for %d * 2^(%d), rnd_mode %d\n",
                      k, -1022-n, rnd_mode);
              printf ("got %.20e instead of %.20e\n", dd, d2);
              fail = 1;
            }
          f *= 0.5;
          mpfr_div_2exp (x, x, 1, MPFR_RNDN);
        }
    }

  mpfr_set_str_binary (x, "1e-1074");
  dd = mpfr_get_d (x, MPFR_RNDA);
  d2 = DBL_MIN; /* 2^(-1022) */
  for (k = 0; k < 52; k++)
    d2 *= 0.5;  /* 2^(-1074) */
  /* we first check that d2 is not zero (it could happen on a platform with
     no subnormals) */
  if (d2 != 0.0 && dd != d2)
    {
      printf ("Error for x=1e-1074, RNDA\n");
      exit (1);
    }

  mpfr_set_str_binary (x, "1e-1075");
  dd = mpfr_get_d (x, MPFR_RNDA);
  if (d2 != 0.0 && dd != d2)
    {
      printf ("Error for x=1e-1075, RNDA\n");
      printf ("expected %.16e\n", d2);
      printf ("got      %.16e\n", dd);
      exit (1);
    }

  mpfr_clear (x);
  return fail;
}

static void
check_inf_nan (void)
{
  /* only if nans and infs are available */
#if _GMP_IEEE_FLOATS && !defined(MPFR_ERRDIVZERO)
  mpfr_t  x;
  double  d;

  mpfr_init2 (x, 123);

  mpfr_set_inf (x, 1);
  d = mpfr_get_d (x, MPFR_RNDZ);
  MPFR_ASSERTN (d > 0);
  MPFR_ASSERTN (DOUBLE_ISINF (d));

  mpfr_set_inf (x, -1);
  d = mpfr_get_d (x, MPFR_RNDZ);
  MPFR_ASSERTN (d < 0);
  MPFR_ASSERTN (DOUBLE_ISINF (d));

  mpfr_set_nan (x);
  d = mpfr_get_d (x, MPFR_RNDZ);
  MPFR_ASSERTN (DOUBLE_ISNAN (d));

  mpfr_clear (x);
#endif
}

static void
check_max (void)
{
  double d, e;
  mpfr_t u;

  d = 1.0;
  while (d < (DBL_MAX / 2.0))
    d += d;
  mpfr_init (u);
  if (mpfr_set_d (u, d, MPFR_RNDN) == 0)
    {
      /* If setting is exact */
      e = (mpfr_get_d1) (u);
      if (e != d)
        {
          printf ("get_d(set_d)(1): %1.20e != %1.20e\n", d, e);
          exit (1);
        }
    }

  mpfr_set_str_binary (u, "-1E1024");
  d = mpfr_get_d (u, MPFR_RNDZ);
  MPFR_ASSERTN(d == -DBL_MAX);
  d = mpfr_get_d (u, MPFR_RNDU);
  MPFR_ASSERTN(d == -DBL_MAX);
#if _GMP_IEEE_FLOATS && !defined(MPFR_ERRDIVZERO)
  d = mpfr_get_d (u, MPFR_RNDN);
  MPFR_ASSERTN(DOUBLE_ISINF(d) && d < 0.0);
  d = mpfr_get_d (u, MPFR_RNDD);
  MPFR_ASSERTN(DOUBLE_ISINF(d) && d < 0.0);
#endif

  mpfr_set_str_binary (u, "1E1024");
  d = mpfr_get_d (u, MPFR_RNDZ);
  MPFR_ASSERTN(d == DBL_MAX);
  d = mpfr_get_d (u, MPFR_RNDD);
  MPFR_ASSERTN(d == DBL_MAX);
#if _GMP_IEEE_FLOATS && !defined(MPFR_ERRDIVZERO)
  d = mpfr_get_d (u, MPFR_RNDN);
  MPFR_ASSERTN(DOUBLE_ISINF(d) && d > 0.0);
  d = mpfr_get_d (u, MPFR_RNDU);
  MPFR_ASSERTN(DOUBLE_ISINF(d) && d > 0.0);
#endif

  mpfr_clear (u);
}

static void
check_min(void)
{
  double d, e;
  mpfr_t u;

  d = 1.0; while (d > (DBL_MIN * 2.0)) d /= 2.0;
  mpfr_init(u);
  if (mpfr_set_d(u, d, MPFR_RNDN) == 0)
    {
      /* If setting is exact */
      e = mpfr_get_d1(u);
      if (e != d)
        {
          printf("get_d(set_d)(2): %1.20e != %1.20e\n", d, e);
          exit(1);
        }
    }
  mpfr_clear(u);
}

static void
check_get_d_2exp_inf_nan (void)
{
#if !defined(MPFR_ERRDIVZERO)

  double var_d;
  long exp;
  mpfr_t var;

  mpfr_init2 (var, MPFR_PREC_MIN);

  mpfr_set_nan (var);
  var_d = mpfr_get_d_2exp (&exp, var, MPFR_RNDN);
  if (!DOUBLE_ISNAN (var_d))
    {
      printf ("mpfr_get_d_2exp with a NAN mpfr value returned a wrong value :\n"
              " waiting for %g got %g\n", MPFR_DBL_NAN, var_d);
      exit (1);
    }

  mpfr_set_zero (var, 1);
  var_d = mpfr_get_d_2exp (&exp, var, MPFR_RNDN);
  if ((exp != 0) || (var_d != 0.0))
    {
      printf ("mpfr_get_d_2exp with a +0.0 mpfr value returned a wrong value :\n"
              " double waiting for 0.0 got %g\n exp waiting for 0 got %ld\n",
              var_d, exp);
      exit (1);
    }

  mpfr_set_zero (var, -1);
  var_d = mpfr_get_d_2exp (&exp, var, MPFR_RNDN);
  if ((exp != 0) || (var_d != DBL_NEG_ZERO))
    {
      printf ("mpfr_get_d_2exp with a +0.0 mpfr value returned a wrong value :\n"
              " double waiting for %g got %g\n exp waiting for 0 got %ld\n",
              DBL_NEG_ZERO, var_d, exp);
      exit (1);
    }

  mpfr_set_inf (var, 1);
  var_d = mpfr_get_d_2exp (&exp, var, MPFR_RNDN);
  if (var_d != MPFR_DBL_INFP)
    {
      printf ("mpfr_get_d_2exp with a +Inf mpfr value returned a wrong value :\n"
              " waiting for %g got %g\n", MPFR_DBL_INFP, var_d);
      exit (1);
    }

  mpfr_set_inf (var, -1);
  var_d = mpfr_get_d_2exp (&exp, var, MPFR_RNDN);
  if (var_d != MPFR_DBL_INFM)
    {
      printf ("mpfr_get_d_2exp with a -Inf mpfr value returned a wrong value :\n"
              " waiting for %g got %g\n", MPFR_DBL_INFM, var_d);
      exit (1);
    }

  mpfr_clear (var);

#endif
}

int
main (void)
{
  tests_start_mpfr ();
  mpfr_test_init ();

#ifndef MPFR_DOUBLE_SPEC
  printf ("Warning! The MPFR_DOUBLE_SPEC macro is not defined. This means\n"
          "that you do not have a conforming C implementation and problems\n"
          "may occur with conversions between MPFR numbers and standard\n"
          "floating-point types. Please contact the MPFR team.\n");
#elif MPFR_DOUBLE_SPEC == 0
  /*
  printf ("The type 'double' of your C implementation does not seem to\n"
          "correspond to the IEEE-754 double precision. Though code has\n"
          "been written to support such implementations, tests have been\n"
          "done only on IEEE-754 double-precision implementations and\n"
          "conversions between MPFR numbers and standard floating-point\n"
          "types may be inaccurate. You may wish to contact the MPFR team\n"
          "for further testing.\n");
  */
  printf ("The type 'double' of your C implementation does not seem to\n"
          "correspond to the IEEE-754 double precision. Such particular\n"
          "implementations are not supported yet, and conversions between\n"
          "MPFR numbers and standard floating-point types may be very\n"
          "inaccurate.\n");
  printf ("FLT_RADIX    = %ld\n", (long) FLT_RADIX);
  printf ("DBL_MANT_DIG = %ld\n", (long) DBL_MANT_DIG);
  printf ("DBL_MIN_EXP  = %ld\n", (long) DBL_MIN_EXP);
  printf ("DBL_MAX_EXP  = %ld\n", (long) DBL_MAX_EXP);
#endif

  if (check_denorms ())
    exit (1);

  check_inf_nan ();
  check_min();
  check_max();

  check_get_d_2exp_inf_nan ();

  tests_end_mpfr ();
  return 0;
}

