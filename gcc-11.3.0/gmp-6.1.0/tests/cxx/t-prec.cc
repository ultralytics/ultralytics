/* Test precision of mpf_class expressions.

Copyright 2001-2003, 2008 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include "config.h"

#include <iostream>

#include "gmp.h"
#include "gmpxx.h"
#include "gmp-impl.h"
#include "tests.h"

using namespace std;


const int
small_prec = 64, medium_prec = 128, large_prec = 192, very_large_prec = 256;

#define ASSERT_ALWAYS_PREC(a, s, prec) \
{                                      \
  mpf_srcptr _a = a.get_mpf_t();       \
  mpf_class _b(s, prec);               \
  mpf_srcptr _c = _b.get_mpf_t();      \
  ASSERT_ALWAYS(mpf_eq(_a, _c, prec)); \
}



void
check_mpf (void)
{
  mpf_set_default_prec(medium_prec);

  // simple expressions
  {
    mpf_class f(3.0, small_prec);
    mpf_class g(1 / f, very_large_prec);
    ASSERT_ALWAYS_PREC
      (g, "0.33333 33333 33333 33333 33333 33333 33333 33333 33333 33333"
       "     33333 33333 33333 33333 33333 333", very_large_prec);
  }
  {
    mpf_class f(9.0, medium_prec);
    mpf_class g(0.0, very_large_prec);
    g = 1 / f;
    ASSERT_ALWAYS_PREC
      (g, "0.11111 11111 11111 11111 11111 11111 11111 11111 11111 11111"
       "     11111 11111 11111 11111 11111 111", very_large_prec);
  }
  {
    mpf_class f(15.0, large_prec);
    mpf_class g(0.0, very_large_prec);
    g = 1 / f;
    ASSERT_ALWAYS_PREC
      (g, "0.06666 66666 66666 66666 66666 66666 66666 66666 66666 66666"
       "     66666 66666 66666 66666 66666 667", very_large_prec);
  }

  // compound expressions
  {
    mpf_class f(3.0, small_prec);
    mpf_class g(-(-(-1 / f)), very_large_prec);
    ASSERT_ALWAYS_PREC
      (g, "-0.33333 33333 33333 33333 33333 33333 33333 33333 33333 33333"
       "      33333 33333 33333 33333 33333 333", very_large_prec);
  }
  {
    mpf_class f(3.0, small_prec), g(9.0, medium_prec);
    mpf_class h(0.0, very_large_prec);
    h = 1/f + 1/g;
    ASSERT_ALWAYS_PREC
      (h, "0.44444 44444 44444 44444 44444 44444 44444 44444 44444 44444"
       "     44444 44444 44444 44444 44444 444", very_large_prec);
  }
  {
    mpf_class f(3.0, small_prec), g(9.0, medium_prec), h(15.0, large_prec);
    mpf_class i(0.0, very_large_prec);
    i = f / g + h;
    ASSERT_ALWAYS_PREC
      (i, "15.33333 33333 33333 33333 33333 33333 33333 33333 33333 33333"
       "      33333 33333 33333 33333 33333 3", very_large_prec);
  }
  {
    mpf_class f(3.0, small_prec);
    mpf_class g(-(1 + f) / 3, very_large_prec);
    ASSERT_ALWAYS_PREC
      (g, "-1.33333 33333 33333 33333 33333 33333 33333 33333 33333 33333"
       "      33333 33333 33333 33333 33333 33", very_large_prec);
  }
  {
    mpf_class f(9.0, medium_prec);
    mpf_class g(0.0, very_large_prec);
    g = sqrt(1 / f);
    ASSERT_ALWAYS_PREC
      (g, "0.33333 33333 33333 33333 33333 33333 33333 33333 33333 33333"
       "     33333 33333 33333 33333 33333 333", very_large_prec);
  }
  {
    mpf_class f(15.0, large_prec);
    mpf_class g(0.0, very_large_prec);
    g = hypot(1 + 5 / f, 1.0);
    ASSERT_ALWAYS_PREC
      (g, "1.66666 66666 66666 66666 66666 66666 66666 66666 66666 66666"
       "     66666 66666 66666 66666 66666 67", very_large_prec);
  }

  // compound assignments
  {
    mpf_class f(3.0, small_prec), g(9.0, medium_prec);
    mpf_class h(1.0, very_large_prec);
    h -= f / g;
    ASSERT_ALWAYS_PREC
      (h, "0.66666 66666 66666 66666 66666 66666 66666 66666 66666 66666"
       "     66666 66666 66666 66666 66666 667", very_large_prec);
  }

  // construction from expressions
  {
    mpf_class f(3.0, small_prec);
    mpf_class g(0.0, very_large_prec);
    g = mpf_class(1 / f);
    ASSERT_ALWAYS_PREC(g, "0.33333 33333 33333 33333", small_prec);
  }
  {
    mpf_class f(9.0, medium_prec);
    mpf_class g(0.0, very_large_prec);
    g = mpf_class(1 / f);
    ASSERT_ALWAYS_PREC
      (g, "0.11111 11111 11111 11111 11111 11111 11111 1111", medium_prec);
  }
  {
    mpf_class f(15.0, large_prec);
    mpf_class g(0.0, very_large_prec);
    g = mpf_class(1 / f);
    ASSERT_ALWAYS_PREC
      (g, "0.06666 66666 66666 66666 66666 66666 66666 66666 66666 66666"
       "     66666 6667", large_prec);
  }

  {
    mpf_class f(3.0, small_prec), g(9.0, medium_prec);
    mpf_class h(0.0, very_large_prec);
    h = mpf_class(f / g + 1, large_prec);
    ASSERT_ALWAYS_PREC
      (h, "1.33333 33333 33333 33333 33333 33333 33333 33333 33333 33333"
       "     33333 333",
       large_prec);
  }

  // mixed mpf/mpq expressions
  {
    mpf_class f(3.0, small_prec);
    mpq_class q(1, 3);
    mpf_class g(0.0, very_large_prec);
    g = f - q;
    ASSERT_ALWAYS_PREC
      (g, "2.66666 66666 66666 66666 66666 66666 66666 66666 66666 66666"
       "     66666 66666 66666 66666 66666 67", very_large_prec);
  }

  {
    mpf_class f(3.0, small_prec);
    mpq_class q(1, 3);
    mpf_class g(0.0, very_large_prec);
    g = mpf_class(f - q, large_prec);
    ASSERT_ALWAYS_PREC
      (g, "2.66666 66666 66666 66666 66666 66666 66666 66666 66666 66666"
       "     66666 667",
       large_prec);
  }
  {
    mpf_class f(3.0, small_prec);
    mpq_class q(1, 3);
    mpf_class g(0.0, very_large_prec);
    g = mpf_class(f - q);
    ASSERT_ALWAYS_PREC
      (g, "2.66666 66666 66666 66666 66666 66666 66666 667", medium_prec);
  }
  {
    mpf_class f(15.0, large_prec);
    mpq_class q(1, 3);
    mpf_class g(0.0, very_large_prec);
    g = mpf_class(f + q);
    ASSERT_ALWAYS_PREC
      (g, "15.33333 33333 33333 33333 33333 33333 33333 33333 33333 33333"
       "      33333 33",
       large_prec);
  }
}


int
main (void)
{
  tests_start();

  check_mpf();

  tests_end();
  return 0;
}
