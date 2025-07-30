/* Test mp*_class binary expressions.

Copyright 2001-2003, 2008, 2012 Free Software Foundation, Inc.

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
  // template <class T, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, __gmp_expr<T, T>, Op> >
  {
    mpz_class a(1), b(2);
    mpz_class c(a + b); ASSERT_ALWAYS(c == 3);
  }
  {
    mpz_class a(3), b(4);
    mpz_class c;
    c = a * b; ASSERT_ALWAYS(c == 12);
  }
  {
    mpz_class a(5), b(3);
    mpz_class c;
    c = a % b; ASSERT_ALWAYS(c == 2);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, U, Op> >
  {
    mpz_class a(1);
    signed int b = 3;
    mpz_class c(a - b); ASSERT_ALWAYS(c == -2);
  }
  {
    mpz_class a(-8);
    unsigned int b = 2;
    mpz_class c;
    c = a / b; ASSERT_ALWAYS(c == -4);
  }
  {
    mpz_class a(2);
    double b = 3.0;
    mpz_class c(a + b); ASSERT_ALWAYS(c == 5);
  }
  {
    mpz_class a(4);
    mpz_class b;
    b = a + 0; ASSERT_ALWAYS(b == 4);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<U, __gmp_expr<T, T>, Op> >
  {
    mpz_class a(3);
    signed int b = 9;
    mpz_class c(b / a); ASSERT_ALWAYS(c == 3);
  }

  // template <class T, class U, class V, class W, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, __gmp_expr<V, W>, Op> >
  // type of result can't be mpz

  // template <class T, class U, class V, class W, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<U, V>, __gmp_expr<T, W>, Op> >
  // type of result can't be mpz

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, __gmp_expr<T, U>, Op> >
  {
    mpz_class a(3), b(4);
    mpz_class c(a * (-b)); ASSERT_ALWAYS(c == -12);
    c = c * (-b); ASSERT_ALWAYS(c == 48);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, __gmp_expr<T, T>, Op> >
  {
    mpz_class a(3), b(2), c(1);
    mpz_class d;
    d = (a % b) + c; ASSERT_ALWAYS(d == 2);
    d = (a % b) + d; ASSERT_ALWAYS(d == 3);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, V, Op> >
  {
    mpz_class a(-5);
    unsigned int b = 2;
    mpz_class c((-a) << b); ASSERT_ALWAYS(c == 20);
  }
  {
    mpz_class a(5), b(-4);
    signed int c = 3;
    mpz_class d;
    d = (a * b) >> c; ASSERT_ALWAYS(d == -3);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<U, __gmp_expr<T, V>, Op> >
  {
    mpz_class a(2), b(4);
    double c = 6;
    mpz_class d(c / (a - b)); ASSERT_ALWAYS(d == -3);
  }
  {
    mpz_class a(3), b(2);
    double c = 1;
    mpz_class d;
    d = c + (a + b); ASSERT_ALWAYS(d == 6);
  }

  // template <class T, class U, class V, class W, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, __gmp_expr<V, W>, Op> >
  // type of result can't be mpz

  // template <class T, class U, class V, class W, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<U, V>, __gmp_expr<T, W>, Op> >
  // type of result can't be mpz

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, __gmp_expr<T, V>, Op> >
  {
    mpz_class a(3), b(5), c(7);
    mpz_class d;
    d = (a - b) * (-c); ASSERT_ALWAYS(d == 14);
    d = (b - d) * (-a); ASSERT_ALWAYS(d == 27);
    d = (a - b) * (-d); ASSERT_ALWAYS(d == 54);
  }

  {
    mpz_class a(0xcafe), b(0xbeef), c, want;
    c = a & b; ASSERT_ALWAYS (c == 0x8aee);
    c = a | b; ASSERT_ALWAYS (c == 0xfeff);
    c = a ^ b; ASSERT_ALWAYS (c == 0x7411);
    c = a & 0xbeef; ASSERT_ALWAYS (c == 0x8aee);
    c = a | 0xbeef; ASSERT_ALWAYS (c == 0xfeff);
    c = a ^ 0xbeef; ASSERT_ALWAYS (c == 0x7411);
    c = a & -0xbeef; ASSERT_ALWAYS (c == 0x4010);
    c = a | -0xbeef; ASSERT_ALWAYS (c == -0x3401);
    c = a ^ -0xbeef; ASSERT_ALWAYS (c == -0x7411);
    c = a & 48879.0; ASSERT_ALWAYS (c == 0x8aee);
    c = a | 48879.0; ASSERT_ALWAYS (c == 0xfeff);
    c = a ^ 48879.0; ASSERT_ALWAYS (c == 0x7411);

    c = a | 1267650600228229401496703205376.0; // 2^100
    want = "0x1000000000000000000000cafe";
    ASSERT_ALWAYS (c == want);
  }

}

void
check_mpq (void)
{
  // template <class T, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, __gmp_expr<T, T>, Op> >
  {
    mpq_class a(1, 2), b(3, 4);
    mpq_class c(a + b); ASSERT_ALWAYS(c == 1.25);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, U, Op> >
  {
    mpq_class a(1, 2);
    signed int b = 3;
    mpq_class c(a - b); ASSERT_ALWAYS(c == -2.5);
  }
  {
    mpq_class a(1, 2);
    mpq_class b;
    b = a + 0; ASSERT_ALWAYS(b == 0.5);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<U, __gmp_expr<T, T>, Op> >
  {
    mpq_class a(2, 3);
    signed int b = 4;
    mpq_class c;
    c = b / a; ASSERT_ALWAYS(c == 6);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, __gmp_expr<U, V>, Op> >
  {
    mpq_class a(1, 2);
    mpz_class b(1);
    mpq_class c(a + b); ASSERT_ALWAYS(c == 1.5);
  }
  {
    mpq_class a(2, 3);
    mpz_class b(1);
    double c = 2.0;
    mpq_class d;
    d = a * (b + c); ASSERT_ALWAYS(d == 2);
    d = d * (b + c); ASSERT_ALWAYS(d == 6);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<U, V>, __gmp_expr<T, T>, Op> >
  {
    mpq_class a(2, 3);
    mpz_class b(4);
    mpq_class c(b / a); ASSERT_ALWAYS(c == 6);
  }
  {
    mpq_class a(2, 3);
    mpz_class b(1), c(4);
    mpq_class d;
    d = (b - c) * a; ASSERT_ALWAYS(d == -2);
    d = (b - c) * d; ASSERT_ALWAYS(d == 6);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, __gmp_expr<T, U>, Op> >
  {
    mpq_class a(1, 3), b(3, 4);
    mpq_class c;
    c = a * (-b); ASSERT_ALWAYS(c == -0.25);
    a = a * (-b); ASSERT_ALWAYS(a == -0.25);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, __gmp_expr<T, T>, Op> >
  {
    mpq_class a(1, 3), b(2, 3), c(1, 4);
    mpq_class d((a / b) + c); ASSERT_ALWAYS(d == 0.75);
    c = (a / b) + c; ASSERT_ALWAYS(c == 0.75);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, V, Op> >
  {
    mpq_class a(3, 8);
    unsigned int b = 4;
    mpq_class c((-a) << b); ASSERT_ALWAYS(c == -6);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<U, __gmp_expr<T, V>, Op> >
  {
    mpq_class a(1, 2), b(1, 4);
    double c = 6.0;
    mpq_class d;
    d = c / (a + b); ASSERT_ALWAYS(d == 8);
  }

  // template <class T, class U, class V, class W, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, __gmp_expr<V, W>, Op> >
  {
    mpq_class a(1, 2), b(1, 4);
    mpz_class c(1);
    mpq_class d((a + b) - c); ASSERT_ALWAYS(d == -0.25);
    d = (a + d) - c; ASSERT_ALWAYS(d == -0.75);
    d = (a + d) - d.get_num(); ASSERT_ALWAYS(d == 2.75);
    d = (2 * d) * d.get_den(); ASSERT_ALWAYS(d == 22);
    d = (b * d) / -d.get_num(); ASSERT_ALWAYS(d == -0.25);
  }
  {
    mpq_class a(1, 3), b(3, 2);
    mpz_class c(2), d(4);
    mpq_class e;
    e = (a * b) / (c - d); ASSERT_ALWAYS(e == -0.25);
    e = (2 * e) / (c - d); ASSERT_ALWAYS(e ==  0.25);
  }

  // template <class T, class U, class V, class W, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<U, V>, __gmp_expr<T, W>, Op> >
  {
    mpq_class a(1, 3), b(3, 4);
    mpz_class c(-3);
    mpq_class d(c * (a * b)); ASSERT_ALWAYS(d == -0.75);
  }
  {
    mpq_class a(1, 3), b(3, 5);
    mpz_class c(6);
    signed int d = 4;
    mpq_class e;
    e = (c % d) / (a * b); ASSERT_ALWAYS(e == 10);
    e = (e.get_num() % d) / (2 / e); ASSERT_ALWAYS(e == 10);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, __gmp_expr<T, V>, Op> >
  {
    mpq_class a(1, 3), b(3, 4), c(2, 5);
    mpq_class d;
    d = (a * b) / (-c); ASSERT_ALWAYS(d == -0.625);
    d = (c * d) / (-b); ASSERT_ALWAYS(3 * d == 1);
    d = (a * c) / (-d); ASSERT_ALWAYS(5 * d == -2);
  }
}

void
check_mpf (void)
{
  // template <class T, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, __gmp_expr<T, T>, Op> >
  {
    mpf_class a(1), b(2);
    mpf_class c(a + b); ASSERT_ALWAYS(c == 3);
  }
  {
    mpf_class a(1.5), b(6);
    mpf_class c;
    c = a / b; ASSERT_ALWAYS(c == 0.25);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, U, Op> >
  {
    mpf_class a(1);
    signed int b = -2;
    mpf_class c(a - b); ASSERT_ALWAYS(c == 3);
  }
  {
    mpf_class a(2);
    mpf_class b;
    b = a + 0; ASSERT_ALWAYS(b == 2);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<U, __gmp_expr<T, T>, Op> >
  {
    mpf_class a(2);
    unsigned int b = 3;
    mpf_class c;
    c = b / a; ASSERT_ALWAYS(c == 1.5);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, __gmp_expr<U, V>, Op> >
  {
    mpf_class a(2);
    mpz_class b(3);
    mpf_class c(a - b); ASSERT_ALWAYS(c == -1);
  }
  {
    mpf_class a(3);
    mpz_class b(2), c(1);
    mpf_class d;
    d = a * (b + c); ASSERT_ALWAYS(d == 9);
    a = a * (b + c); ASSERT_ALWAYS(a == 9);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<U, V>, __gmp_expr<T, T>, Op> >
  {
    mpf_class a(6);
    mpq_class b(3, 4);
    mpf_class c(a * b); ASSERT_ALWAYS(c == 4.5);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, T>, __gmp_expr<T, U>, Op> >
  {
    mpf_class a(2), b(-3);
    mpf_class c;
    c = a * (-b); ASSERT_ALWAYS(c == 6);
    c = c * (-b); ASSERT_ALWAYS(c == 18);
  }

  // template <class T, class U, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, __gmp_expr<T, T>, Op> >
  {
    mpf_class a(3), b(4), c(5);
    mpf_class d;
    d = (a / b) - c; ASSERT_ALWAYS(d == -4.25);
    c = (a / b) - c; ASSERT_ALWAYS(c == -4.25);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, V, Op> >
  {
    mpf_class a(3);
    unsigned int b = 2;
    mpf_class c((-a) >> b); ASSERT_ALWAYS(c == -0.75);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<U, __gmp_expr<T, V>, Op> >
  {
    mpf_class a(2), b(3);
    double c = 5.0;
    mpf_class d;
    d = c / (a + b); ASSERT_ALWAYS(d == 1);
  }

  // template <class T, class U, class V, class W, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, __gmp_expr<V, W>, Op> >
  {
    mpf_class a(2), b(3);
    mpz_class c(4);
    mpf_class d;
    d = (a + b) * c; ASSERT_ALWAYS(d == 20);
  }
  {
    mpf_class a(2), b(3);
    mpq_class c(1, 2), d(1, 4);
    mpf_class e;
    e = (a * b) / (c + d); ASSERT_ALWAYS(e == 8);
  }

  // template <class T, class U, class V, class W, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<U, V>, __gmp_expr<T, W>, Op> >
  {
    mpf_class a(1), b(2);
    mpq_class c(3);
    mpf_class d(c / (a + b)); ASSERT_ALWAYS(d == 1);
  }
  {
    mpf_class a(1);
    mpz_class b(2);
    mpq_class c(3, 4);
    mpf_class d;
    d = (-c) + (a + b); ASSERT_ALWAYS(d == 2.25);
  }

  // template <class T, class U, class V, class Op>
  // __gmp_expr<T, __gmp_binary_expr<__gmp_expr<T, U>, __gmp_expr<T, V>, Op> >
  {
    mpf_class a(1), b(2), c(3);
    mpf_class d;
    d = (a + b) * (-c); ASSERT_ALWAYS(d == -9);
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
