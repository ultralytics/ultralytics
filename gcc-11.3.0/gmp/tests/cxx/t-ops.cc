/* Test mp*_class operators and functions.

Copyright 2001-2003, 2015 Free Software Foundation, Inc.

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


void
check_mpz (void)
{
  // unary operators and functions

  // operator+
  {
    mpz_class a(1);
    mpz_class b;
    b = +a; ASSERT_ALWAYS(b == 1);
  }

  // operator-
  {
    mpz_class a(2);
    mpz_class b;
    b = -a; ASSERT_ALWAYS(b == -2);
  }

  // operator~
  {
    mpz_class a(3);
    mpz_class b;
    b = ~a; ASSERT_ALWAYS(b == -4);
  }

  // abs
  {
    mpz_class a(-123);
    mpz_class b;
    b = abs(a); ASSERT_ALWAYS(b == 123);
    a <<= 300;
    b = abs(a); ASSERT_ALWAYS(a + b == 0);
  }

  // sqrt
  {
    mpz_class a(25);
    mpz_class b;
    b = sqrt(a); ASSERT_ALWAYS(b == 5);
  }
  {
    mpz_class a(125);
    mpz_class b;
    b = sqrt(a); ASSERT_ALWAYS(b == 11); // round toward zero
  }

  // sgn
  {
    mpz_class a(123);
    int b = sgn(a); ASSERT_ALWAYS(b == 1);
  }
  {
    mpz_class a(0);
    int b = sgn(a); ASSERT_ALWAYS(b == 0);
  }
  {
    mpz_class a(-123);
    int b = sgn(a); ASSERT_ALWAYS(b == -1);
  }


  // binary operators and functions

  // operator+
  {
    mpz_class a(1), b(2);
    mpz_class c;
    c = a + b; ASSERT_ALWAYS(c == 3);
  }
  {
    mpz_class a(3);
    signed int b = 4;
    mpz_class c;
    c = a + b; ASSERT_ALWAYS(c == 7);
  }
  {
    mpz_class a(5);
    double b = 6.0;
    mpz_class c;
    c = b + a; ASSERT_ALWAYS(c == 11);
  }

  // operator-
  {
    mpz_class a(3), b(6);
    mpz_class c;
    c = a - b; ASSERT_ALWAYS(c == -3);
  }

  // operator*
  {
    mpz_class a(-2), b(4);
    mpz_class c;
    c = a * b; ASSERT_ALWAYS(c == -8);
  }
  {
    mpz_class a(2);
    long b = -4;
    mpz_class c;
    c = a * b; ASSERT_ALWAYS(c == -8);
    c = b * a; ASSERT_ALWAYS(c == -8);
  }
  {
    mpz_class a(-2);
    unsigned long b = 4;
    mpz_class c;
    c = a * b; ASSERT_ALWAYS(c == -8);
    c = b * a; ASSERT_ALWAYS(c == -8);
  }

  // operator/ and operator%
  {
    mpz_class a(12), b(4);
    mpz_class c;
    c = a / b; ASSERT_ALWAYS(c == 3);
    c = a % b; ASSERT_ALWAYS(c == 0);
  }
  {
    mpz_class a(7), b(5);
    mpz_class c;
    c = a / b; ASSERT_ALWAYS(c == 1);
    c = a % b; ASSERT_ALWAYS(c == 2);
  }
  {
    mpz_class a(-10);
    signed int ai = -10;
    mpz_class b(3);
    signed int bi = 3;
    mpz_class c;
    c = a / b;  ASSERT_ALWAYS(c == -3);
    c = a % b;  ASSERT_ALWAYS(c == -1);
    c = a / bi; ASSERT_ALWAYS(c == -3);
    c = a % bi; ASSERT_ALWAYS(c == -1);
    c = ai / b; ASSERT_ALWAYS(c == -3);
    c = ai % b; ASSERT_ALWAYS(c == -1);
  }
  {
    mpz_class a(-10);
    signed int ai = -10;
    mpz_class b(-3);
    signed int bi = -3;
    mpz_class c;
    c = a / b;  ASSERT_ALWAYS(c == 3);
    c = a % b;  ASSERT_ALWAYS(c == -1);
    c = a / bi; ASSERT_ALWAYS(c == 3);
    c = a % bi; ASSERT_ALWAYS(c == -1);
    c = ai / b; ASSERT_ALWAYS(c == 3);
    c = ai % b; ASSERT_ALWAYS(c == -1);
  }
  {
    mpz_class a (LONG_MIN);
    signed long ai = LONG_MIN;
    mpz_class b = - mpz_class (LONG_MIN);
    mpz_class c;
    c = a / b;  ASSERT_ALWAYS(c == -1);
    c = a % b;  ASSERT_ALWAYS(c == 0);
    c = ai / b; ASSERT_ALWAYS(c == -1);
    c = ai % b; ASSERT_ALWAYS(c == 0);
  }

  // operator&
  // operator|
  // operator^

  // operator<<
  {
    mpz_class a(3);
    unsigned int b = 4;
    mpz_class c;
    c = a << b; ASSERT_ALWAYS(c == 48);
  }

  // operator>>
  {
    mpz_class a(127);
    unsigned int b = 4;
    mpz_class c;
    c = a >> b; ASSERT_ALWAYS(c == 7);
  }

  // operator==
  // operator!=
  // operator<
  // operator<=
  // operator>
  // operator>=

  // cmp
  {
    mpz_class a(123), b(45);
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpz_class a(123);
    unsigned long b = 45;
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpz_class a(123);
    long b = 45;
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpz_class a(123);
    double b = 45;
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }


  // ternary operators

  // mpz_addmul
  {
    mpz_class a(1), b(2), c(3);
    mpz_class d;
    d = a + b * c; ASSERT_ALWAYS(d == 7);
  }
  {
    mpz_class a(1), b(2);
    unsigned int c = 3;
    mpz_class d;
    d = a + b * c; ASSERT_ALWAYS(d == 7);
  }
  {
    mpz_class a(1), b(3);
    unsigned int c = 2;
    mpz_class d;
    d = a + c * b; ASSERT_ALWAYS(d == 7);
  }
  {
    mpz_class a(1), b(2);
    signed int c = 3;
    mpz_class d;
    d = a + b * c; ASSERT_ALWAYS(d == 7);
  }
  {
    mpz_class a(1), b(3);
    signed int c = 2;
    mpz_class d;
    d = a + c * b; ASSERT_ALWAYS(d == 7);
  }
  {
    mpz_class a(1), b(2);
    double c = 3.0;
    mpz_class d;
    d = a + b * c; ASSERT_ALWAYS(d == 7);
  }
  {
    mpz_class a(1), b(3);
    double c = 2.0;
    mpz_class d;
    d = a + c * b; ASSERT_ALWAYS(d == 7);
  }

  {
    mpz_class a(2), b(3), c(4);
    mpz_class d;
    d = a * b + c; ASSERT_ALWAYS(d == 10);
  }
  {
    mpz_class a(2), b(4);
    unsigned int c = 3;
    mpz_class d;
    d = a * c + b; ASSERT_ALWAYS(d == 10);
  }
  {
    mpz_class a(3), b(4);
    unsigned int c = 2;
    mpz_class d;
    d = c * a + b; ASSERT_ALWAYS(d == 10);
  }
  {
    mpz_class a(2), b(4);
    signed int c = 3;
    mpz_class d;
    d = a * c + b; ASSERT_ALWAYS(d == 10);
  }
  {
    mpz_class a(3), b(4);
    signed int c = 2;
    mpz_class d;
    d = c * a + b; ASSERT_ALWAYS(d == 10);
  }
  {
    mpz_class a(2), b(4);
    double c = 3.0;
    mpz_class d;
    d = a * c + b; ASSERT_ALWAYS(d == 10);
  }
  {
    mpz_class a(3), b(4);
    double c = 2.0;
    mpz_class d;
    d = c * a + b; ASSERT_ALWAYS(d == 10);
  }

  // mpz_submul
  {
    mpz_class a(1), b(2), c(3);
    mpz_class d;
    d = a - b * c; ASSERT_ALWAYS(d == -5);
  }
  {
    mpz_class a(1), b(2);
    unsigned int c = 3;
    mpz_class d;
    d = a - b * c; ASSERT_ALWAYS(d == -5);
  }
  {
    mpz_class a(1), b(3);
    unsigned int c = 2;
    mpz_class d;
    d = a - c * b; ASSERT_ALWAYS(d == -5);
  }
  {
    mpz_class a(1), b(2);
    signed int c = 3;
    mpz_class d;
    d = a - b * c; ASSERT_ALWAYS(d == -5);
  }
  {
    mpz_class a(1), b(3);
    signed int c = 2;
    mpz_class d;
    d = a - c * b; ASSERT_ALWAYS(d == -5);
  }
  {
    mpz_class a(1), b(2);
    double c = 3.0;
    mpz_class d;
    d = a - b * c; ASSERT_ALWAYS(d == -5);
  }
  {
    mpz_class a(1), b(3);
    double c = 2.0;
    mpz_class d;
    d = a - c * b; ASSERT_ALWAYS(d == -5);
  }

  {
    mpz_class a(2), b(3), c(4);
    mpz_class d;
    d = a * b - c; ASSERT_ALWAYS(d == 2);
  }
  {
    mpz_class a(2), b(4);
    unsigned int c = 3;
    mpz_class d;
    d = a * c - b; ASSERT_ALWAYS(d == 2);
  }
  {
    mpz_class a(3), b(4);
    unsigned int c = 2;
    mpz_class d;
    d = c * a - b; ASSERT_ALWAYS(d == 2);
  }
  {
    mpz_class a(2), b(4);
    signed int c = 3;
    mpz_class d;
    d = a * c - b; ASSERT_ALWAYS(d == 2);
  }
  {
    mpz_class a(3), b(4);
    signed int c = 2;
    mpz_class d;
    d = c * a - b; ASSERT_ALWAYS(d == 2);
  }
  {
    mpz_class a(2), b(4);
    double c = 3.0;
    mpz_class d;
    d = a * c - b; ASSERT_ALWAYS(d == 2);
  }
  {
    mpz_class a(3), b(4);
    double c = 2.0;
    mpz_class d;
    d = c * a - b; ASSERT_ALWAYS(d == 2);
  }
}

void
check_mpq (void)
{
  // unary operators and functions

  // operator+
  {
    mpq_class a(1, 2);
    mpq_class b;
    b = +a; ASSERT_ALWAYS(b == 0.5);
  }

  // operator-
  {
    mpq_class a(3, 4);
    mpq_class b;
    b = -a; ASSERT_ALWAYS(b == -0.75);
  }

  // abs
  {
    mpq_class a(-123);
    mpq_class b;
    b = abs(a); ASSERT_ALWAYS(b == 123);
  }

  // sgn
  {
    mpq_class a(123);
    int b = sgn(a); ASSERT_ALWAYS(b == 1);
  }
  {
    mpq_class a(0);
    int b = sgn(a); ASSERT_ALWAYS(b == 0);
  }
  {
    mpq_class a(-123);
    int b = sgn(a); ASSERT_ALWAYS(b == -1);
  }


  // binary operators and functions

  // operator+
  {
    mpq_class a(1, 2), b(3, 4);
    mpq_class c;
    c = a + b; ASSERT_ALWAYS(c == 1.25);
  }
  {
    mpq_class a(1, 2);
    signed int b = 2;
    mpq_class c;
    c = a + b; ASSERT_ALWAYS(c == 2.5);
  }
  {
    mpq_class a(1, 2);
    double b = 1.5;
    mpq_class c;
    c = b + a; ASSERT_ALWAYS(c == 2);
  }

  // operator-
  {
    mpq_class a(1, 2), b(3, 4);
    mpq_class c;
    c = a - b; ASSERT_ALWAYS(c == -0.25);
  }

  // operator*
  {
    mpq_class a(1, 3), b(3, 4);
    mpq_class c;
    c = a * b; ASSERT_ALWAYS(c == 0.25);
    c = b * b; ASSERT_ALWAYS(c == 0.5625);
  }

  // operator/
  {
    mpq_class a(1, 2), b(2, 3);
    mpq_class c;
    c = a / b; ASSERT_ALWAYS(c == 0.75);
  }

  // operator<<
  // operator>>
  // operator==
  // operator!=
  // operator<
  // operator<=
  // operator>
  // operator>=

  // cmp
  {
    mpq_class a(123), b(45);
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpq_class a(123);
    unsigned long b = 45;
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpq_class a(123);
    long b = 45;
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpq_class a(123);
    double b = 45;
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpq_class a(123);
    mpz_class b(45);
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
}

void
check_mpf (void)
{
  // unary operators and functions

  // operator+
  {
    mpf_class a(1);
    mpf_class b;
    b = +a; ASSERT_ALWAYS(b == 1);
  }

  // operator-
  {
    mpf_class a(2);
    mpf_class b;
    b = -a; ASSERT_ALWAYS(b == -2);
  }

  // abs
  {
    mpf_class a(-123);
    mpf_class b;
    b = abs(a); ASSERT_ALWAYS(b == 123);
  }

  // trunc
  {
    mpf_class a(1.5);
    mpf_class b;
    b = trunc(a); ASSERT_ALWAYS(b == 1);
  }
  {
    mpf_class a(-1.5);
    mpf_class b;
    b = trunc(a); ASSERT_ALWAYS(b == -1);
  }

  // floor
  {
    mpf_class a(1.9);
    mpf_class b;
    b = floor(a); ASSERT_ALWAYS(b == 1);
  }
  {
    mpf_class a(-1.1);
    mpf_class b;
    b = floor(a); ASSERT_ALWAYS(b == -2);
  }

  // ceil
  {
    mpf_class a(1.1);
    mpf_class b;
    b = ceil(a); ASSERT_ALWAYS(b == 2);
  }
  {
    mpf_class a(-1.9);
    mpf_class b;
    b = ceil(a); ASSERT_ALWAYS(b == -1);
  }

  // sqrt
  {
    mpf_class a(25);
    mpf_class b;
    b = sqrt(a); ASSERT_ALWAYS(b == 5);
  }
  {
    mpf_class a(2.25);
    mpf_class b;
    b = sqrt(a); ASSERT_ALWAYS(b == 1.5);
  }

  // sgn
  {
    mpf_class a(123);
    int b = sgn(a); ASSERT_ALWAYS(b == 1);
  }
  {
    mpf_class a(0);
    int b = sgn(a); ASSERT_ALWAYS(b == 0);
  }
  {
    mpf_class a(-123);
    int b = sgn(a); ASSERT_ALWAYS(b == -1);
  }


  // binary operators and functions

  // operator+
  {
    mpf_class a(1), b(2);
    mpf_class c;
    c = a + b; ASSERT_ALWAYS(c == 3);
  }

  // operator-
  {
    mpf_class a(3), b(4);
    mpf_class c;
    c = a - b; ASSERT_ALWAYS(c == -1);
  }

  // operator*
  {
    mpf_class a(2), b(5);
    mpf_class c;
    c = a * b; ASSERT_ALWAYS(c == 10);
  }

  // operator/
  {
    mpf_class a(7), b(4);
    mpf_class c;
    c = a / b; ASSERT_ALWAYS(c == 1.75);
  }

  // operator<<
  // operator>>
  // operator==
  // operator!=
  // operator<
  // operator<=
  // operator>
  // operator>=

  // hypot
  {
    mpf_class a(3), b(4);
    mpf_class c;
    c = hypot(a, b); ASSERT_ALWAYS(c == 5);
  }

  // cmp
  {
    mpf_class a(123), b(45);
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpf_class a(123);
    unsigned long b = 45;
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpf_class a(123);
    long b = 45;
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpf_class a(123);
    double b = 45;
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpf_class a(123);
    mpz_class b(45);
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
  {
    mpf_class a(123);
    mpq_class b(45);
    int c;
    c = cmp(a, b); ASSERT_ALWAYS(c > 0);
    c = cmp(b, a); ASSERT_ALWAYS(c < 0);
  }
}


int
main (void)
{
  tests_start();

  check_mpz();
  check_mpq();
  check_mpf();

  tests_end();
  return 0;
}
