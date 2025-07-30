/* Test mp*_class ternary expressions.

Copyright 2001-2003 Free Software Foundation, Inc.

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


/* The various test cases are broken up into separate functions to keep down
   compiler memory use.  They're static so that any mistakenly omitted from
   main() will provoke warnings (under gcc -Wall at least).  */

static void
check_mpz_1 (void)
{
  // template<class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<mpz_class, __gmp_expr
  // <mpz_t, __gmp_binary_expr<mpz_class, mpz_class, Op1> >, Op2> >
  {
    mpz_class a(1), b(2), c(3);
    mpz_class d;
    d = a + b * c; ASSERT_ALWAYS(d == 7);
  }
  {
    mpz_class a(1), b(2), c(3);
    mpz_class d;
    d = a - b * c; ASSERT_ALWAYS(d == -5);
  }
}

static void
check_mpz_2 (void)
{
  // template <class T, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<mpz_class, __gmp_expr
  // <mpz_t, __gmp_binary_expr<mpz_class, T, Op1> >, Op2> >
  {
    mpz_class a(1), b(2);
    signed int c = 3;
    mpz_class d;
    d = a + b * c; ASSERT_ALWAYS(d == 7);
  }
  {
    mpz_class a(1), b(2);
    signed int c = 3;
    mpz_class d;
    d = a - b * c; ASSERT_ALWAYS(d == -5);
  }
}

static void
check_mpz_3 (void)
{
  // template <class T, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<mpz_class, __gmp_expr
  // <mpz_t, __gmp_binary_expr<T, mpz_class, Op1> >, Op2> >
  {
    mpz_class a(1), b(2);
    unsigned int c = 3;
    mpz_class d;
    d = a + c * b; ASSERT_ALWAYS(d == 7);
  }
  {
    mpz_class a(1), b(2);
    unsigned int c = 3;
    mpz_class d;
    d = a - c * b; ASSERT_ALWAYS(d == -5);
  }
}

static void
check_mpz_4 (void)
{
  // template <class T, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<mpz_class, __gmp_expr
  // <mpz_t, __gmp_binary_expr<mpz_class, __gmp_expr<mpz_t, T>, Op1> >, Op2> >
  {
    mpz_class a(1), b(2), c(3);
    double d = 4.0;
    mpz_class e;
    e = a + b * (c + d); ASSERT_ALWAYS(e == 15);
  }
  {
    mpz_class a(1), b(2), c(3);
    double d = 4.0;
    mpz_class e;
    e = a - b * (c + d); ASSERT_ALWAYS(e == -13);
  }
}

static void
check_mpz_5 (void)
{
  // template <class T, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<mpz_class, __gmp_expr
  // <mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>, mpz_class, Op1> >, Op2> >
  {
    mpz_class a(1), b(2), c(3);
    signed int d = 4;
    mpz_class e;
    e = a + (b - d) * c; ASSERT_ALWAYS(e == -5);
  }
  {
    mpz_class a(1), b(2), c(3);
    signed int d = 4;
    mpz_class e;
    e = a - (b - d) * c; ASSERT_ALWAYS(e == 7);
  }
}

static void
check_mpz_6 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<mpz_class, __gmp_expr
  // <mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>, U, Op1> >, Op2> >
  {
    mpz_class a(1), b(2);
    unsigned int c = 3, d = 4;
    mpz_class e;
    e = a + (b + c) * d; ASSERT_ALWAYS(e == 21);
  }
  {
    mpz_class a(1), b(2);
    unsigned int c = 3, d = 4;
    mpz_class e;
    e = a - (b + c) * d; ASSERT_ALWAYS(e == -19);
  }
}

static void
check_mpz_7 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<mpz_class, __gmp_expr
  // <mpz_t, __gmp_binary_expr<T, __gmp_expr<mpz_t, U>, Op1> >, Op2> >
  {
    mpz_class a(1), b(2);
    double c = 3.0, d = 4.0;
    mpz_class e;
    e = a + c * (b + d); ASSERT_ALWAYS(e == 19);
  }
  {
    mpz_class a(1), b(2);
    double c = 3.0, d = 4.0;
    mpz_class e;
    e = a - c * (b + d); ASSERT_ALWAYS(e == -17);
  }
}

static void
check_mpz_8 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<mpz_class, __gmp_expr
  // <mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>, __gmp_expr<mpz_t, U>,
  // Op1> >, Op2> >
  {
    mpz_class a(1), b(2), c(3);
    signed int d = 4, e = 5;
    mpz_class f;
    f = a + (b - d) * (c + e); ASSERT_ALWAYS(f == -15);
  }
  {
    mpz_class a(1), b(2), c(3);
    signed int d = 4, e = 5;
    mpz_class f;
    f = a - (b - d) * (c + e); ASSERT_ALWAYS(f == 17);
  }
}

static void
check_mpz_9 (void)
{
  // template <class T, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>,
  // __gmp_expr<mpz_t, __gmp_binary_expr<mpz_class, mpz_class, Op1> >, Op2> >
  {
    mpz_class a(1), b(2), c(3);
    unsigned int d = 4;
    mpz_class e;
    e = (a + d) + b * c; ASSERT_ALWAYS(e == 11);
  }
  {
    mpz_class a(1), b(2), c(3);
    unsigned int d = 4;
    mpz_class e;
    e = (a + d) - b * c; ASSERT_ALWAYS(e == -1);
  }
}

static void
check_mpz_10 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>,
  // __gmp_expr<mpz_t, __gmp_binary_expr<mpz_class, U, Op1> >, Op2> >
  {
    mpz_class a(1), b(2);
    double c = 3.0, d = 4.0;
    mpz_class e;
    e = (a - c) + b * d; ASSERT_ALWAYS(e == 6);
  }
  {
    mpz_class a(1), b(2);
    double c = 3.0, d = 4.0;
    mpz_class e;
    e = (a - c) - b * d; ASSERT_ALWAYS(e == -10);
  }
}

static void
check_mpz_11 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>,
  // __gmp_expr<mpz_t, __gmp_binary_expr<U, mpz_class, Op1> >, Op2> >
  {
    mpz_class a(1), b(2);
    signed int c = 3, d = 4;
    mpz_class e;
    e = (a - c) + d * b; ASSERT_ALWAYS(e == 6);
  }
  {
    mpz_class a(1), b(2);
    signed int c = 3, d = 4;
    mpz_class e;
    e = (a - c) - d * b; ASSERT_ALWAYS(e == -10);
  }
}

static void
check_mpz_12 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>, __gmp_expr
  // <mpz_t, __gmp_binary_expr<mpz_class, __gmp_expr<mpz_t, U>, Op1> >, Op2> >
  {
    mpz_class a(1), b(2), c(3);
    unsigned int d = 4, e = 5;
    mpz_class f;
    f = (a + d) + b * (c - e); ASSERT_ALWAYS(f == 1);
  }
  {
    mpz_class a(1), b(2), c(3);
    unsigned int d = 4, e = 5;
    mpz_class f;
    f = (a + d) - b * (c - e); ASSERT_ALWAYS(f == 9);
  }
}

static void
check_mpz_13 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>, __gmp_expr
  // <mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, U>, mpz_class, Op1> >, Op2> >
  {
    mpz_class a(1), b(2), c(3);
    double d = 4.0, e = 5.0;
    mpz_class f;
    f = (a - d) + (b + e) * c; ASSERT_ALWAYS(f == 18);
  }
  {
    mpz_class a(1), b(2), c(3);
    double d = 4.0, e = 5.0;
    mpz_class f;
    f = (a - d) - (b + e) * c; ASSERT_ALWAYS(f == -24);
  }

}

static void
check_mpz_14 (void)
{
  // template <class T, class U, class V, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>, __gmp_expr
  // <mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, U>, V, Op1> >, Op2> >
  {
    mpz_class a(1), b(2);
    signed int c = 3, d = 4, e = 5;
    mpz_class f;
    f = (a + c) + (b + d) * e; ASSERT_ALWAYS(f == 34);
  }
  {
    mpz_class a(1), b(2);
    signed int c = 3, d = 4, e = 5;
    mpz_class f;
    f = (a + c) - (b + d) * e; ASSERT_ALWAYS(f == -26);
  }
}

static void
check_mpz_15 (void)
{
  // template <class T, class U, class V, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>, __gmp_expr
  // <mpz_t, __gmp_binary_expr<U, __gmp_expr<mpz_t, V>, Op1> >, Op2> >
  {
    mpz_class a(1), b(2);
    unsigned int c = 3, d = 4, e = 5;
    mpz_class f;
    f = (a - c) + d * (b - e); ASSERT_ALWAYS(f == -14);
  }
  {
    mpz_class a(1), b(2);
    unsigned int c = 3, d = 4, e = 5;
    mpz_class f;
    f = (a - c) - d * (b - e); ASSERT_ALWAYS(f == 10);
  }

}

static void
check_mpz_16 (void)
{
  // template <class T, class U, class V, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, T>, __gmp_expr
  // <mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, U>, __gmp_expr<mpz_t, V>,
  // Op1> >, Op2> >
  {
    mpz_class a(1), b(2), c(3);
    double d = 4.0, e = 5.0, f = 6.0;
    mpz_class g;
    g = (a + d) + (b - e) * (c + f); ASSERT_ALWAYS(g == -22);
  }
  {
    mpz_class a(1), b(2), c(3);
    double d = 4.0, e = 5.0, f = 6.0;
    mpz_class g;
    g = (a + d) - (b - e) * (c + f); ASSERT_ALWAYS(g == 32);
  }
}

static void
check_mpz_17 (void)
{
  // template <class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr
  // <mpz_t, __gmp_binary_expr<mpz_class, mpz_class, Op1> >, mpz_class, Op2> >
  {
    mpz_class a(2), b(3), c(4);
    mpz_class d;
    d = a * b + c; ASSERT_ALWAYS(d == 10);
  }
  {
    mpz_class a(2), b(3), c(4);
    mpz_class d;
    d = a * b - c; ASSERT_ALWAYS(d == 2);
  }
}

static void
check_mpz_18 (void)
{
  // template <class T, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr
  // <mpz_t, __gmp_binary_expr<mpz_class, T, Op1> >, mpz_class, Op2> >
  {
    mpz_class a(2), b(3);
    signed int c = 4;
    mpz_class d;
    d = a * c + b; ASSERT_ALWAYS(d == 11);
  }
  {
    mpz_class a(2), b(3);
    signed int c = 4;
    mpz_class d;
    d = a * c - b; ASSERT_ALWAYS(d == 5);
  }

}

static void
check_mpz_19 (void)
{
  // template <class T, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr
  // <mpz_t, __gmp_binary_expr<T, mpz_class, Op1> >, mpz_class, Op2> >
  {
    mpz_class a(2), b(3);
    unsigned int c = 4;
    mpz_class d;
    d = c * a + b; ASSERT_ALWAYS(d == 11);
  }
  {
    mpz_class a(2), b(3);
    unsigned int c = 4;
    mpz_class d;
    d = c * a - b; ASSERT_ALWAYS(d == 5);
  }
}

static void
check_mpz_20 (void)
{
  // template <class T, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <mpz_class, __gmp_expr<mpz_t, T>, Op1> >, mpz_class, Op2> >
  {
    mpz_class a(2), b(3), c(4);
    double d = 5.0;
    mpz_class e;
    e = a * (b + d) + c; ASSERT_ALWAYS(e == 20);
  }
  {
    mpz_class a(2), b(3), c(4);
    double d = 5.0;
    mpz_class e;
    e = a * (b + d) - c; ASSERT_ALWAYS(e == 12);
  }
}

static void
check_mpz_21 (void)
{
  // template <class T, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <__gmp_expr<mpz_t, T>, mpz_class, Op1> >, mpz_class, Op2> >
  {
    mpz_class a(2), b(3), c(4);
    signed int d = 5;
    mpz_class e;
    e = (a - d) * b + c; ASSERT_ALWAYS(e == -5);
  }
  {
    mpz_class a(2), b(3), c(4);
    signed int d = 5;
    mpz_class e;
    e = (a - d) * b - c; ASSERT_ALWAYS(e == -13);
  }
}

static void
check_mpz_22 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <__gmp_expr<mpz_t, T>, U, Op1> >, mpz_class, Op2> >
  {
    mpz_class a(2), b(3);
    unsigned int c = 4, d = 5;
    mpz_class e;
    e = (a + c) * d + b; ASSERT_ALWAYS(e == 33);
  }
  {
    mpz_class a(2), b(3);
    unsigned int c = 4, d = 5;
    mpz_class e;
    e = (a + c) * d - b; ASSERT_ALWAYS(e == 27);
  }
}

static void
check_mpz_23 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <T, __gmp_expr<mpz_t, U>, Op1> >, mpz_class, Op2> >
  {
    mpz_class a(2), b(3);
    double c = 4.0, d = 5.0;
    mpz_class e;
    e = c * (a + d) + b; ASSERT_ALWAYS(e == 31);
  }
  {
    mpz_class a(2), b(3);
    double c = 4.0, d = 5.0;
    mpz_class e;
    e = c * (a + d) - b; ASSERT_ALWAYS(e == 25);
  }

}

static void
check_mpz_24 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <__gmp_expr<mpz_t, T>, __gmp_expr<mpz_t, U>, Op1> >, mpz_class, Op2> >
  {
    mpz_class a(2), b(3), c(4);
    signed int d = 5, e = 6;
    mpz_class f;
    f = (a - d) * (b + e) + c; ASSERT_ALWAYS(f == -23);
  }
  {
    mpz_class a(2), b(3), c(4);
    signed int d = 5, e = 6;
    mpz_class f;
    f = (a - d) * (b + e) - c; ASSERT_ALWAYS(f == -31);
  }
}

static void
check_mpz_25 (void)
{
  // template <class T, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <mpz_class, mpz_class, Op1> >, __gmp_expr<mpz_t, T>, Op2> >
  {
    mpz_class a(2), b(3), c(4);
    unsigned int d = 5;
    mpz_class e;
    e = a * b + (c - d); ASSERT_ALWAYS(e == 5);
  }
  {
    mpz_class a(2), b(3), c(4);
    unsigned int d = 5;
    mpz_class e;
    e = a * b - (c - d); ASSERT_ALWAYS(e == 7);
  }
}

static void
check_mpz_26 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <mpz_class, T, Op1> >, __gmp_expr<mpz_t, U>, Op2> >
  {
    mpz_class a(2), b(3);
    double c = 4.0, d = 5.0;
    mpz_class e;
    e = a * c + (b + d); ASSERT_ALWAYS(e == 16);
  }
  {
    mpz_class a(2), b(3);
    double c = 4.0, d = 5.0;
    mpz_class e;
    e = a * c - (b + d); ASSERT_ALWAYS(e == 0);
  }
}

static void
check_mpz_27 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <T, mpz_class, Op1> >, __gmp_expr<mpz_t, U>, Op2> >
  {
    mpz_class a(2), b(3);
    signed int c = 4, d = 5;
    mpz_class e;
    e = c * a + (b - d); ASSERT_ALWAYS(e == 6);
  }
  {
    mpz_class a(2), b(3);
    signed int c = 4, d = 5;
    mpz_class e;
    e = c * a - (b - d); ASSERT_ALWAYS(e == 10);
  }
}

static void
check_mpz_28 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <mpz_class, __gmp_expr<mpz_t, T>, Op1> >, __gmp_expr<mpz_t, U>, Op2> >
  {
    mpz_class a(2), b(3), c(4);
    unsigned int d = 5, e = 6;
    mpz_class f;
    f = a * (b - d) + (c + e); ASSERT_ALWAYS(f == 6);
  }
  {
    mpz_class a(2), b(3), c(4);
    unsigned int d = 5, e = 6;
    mpz_class f;
    f = a * (b - d) - (c + e); ASSERT_ALWAYS(f == -14);
  }
}

static void
check_mpz_29 (void)
{
  // template <class T, class U, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <__gmp_expr<mpz_t, T>, mpz_class, Op1> >, __gmp_expr<mpz_t, U>, Op2> >
  {
    mpz_class a(2), b(3), c(4);
    double d = 5.0, e = 6.0;
    mpz_class f;
    f = (a + d) * b + (c - e); ASSERT_ALWAYS(f == 19);
  }
  {
    mpz_class a(2), b(3), c(4);
    double d = 5.0, e = 6.0;
    mpz_class f;
    f = (a + d) * b - (c - e); ASSERT_ALWAYS(f == 23);
  }
}

static void
check_mpz_30 (void)
{
  // template <class T, class U, class V, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <__gmp_expr<mpz_t, T>, U, Op1> >, __gmp_expr<mpz_t, V>, Op2> >
  {
    mpz_class a(2), b(3);
    signed int c = 4, d = 5, e = 6;
    mpz_class f;
    f = (a + c) * d + (b + e); ASSERT_ALWAYS(f == 39);
  }
  {
    mpz_class a(2), b(3);
    signed int c = 4, d = 5, e = 6;
    mpz_class f;
    f = (a + c) * d - (b + e); ASSERT_ALWAYS(f == 21);
  }
}

static void
check_mpz_31 (void)
{
  // template <class T, class U, class V, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <T, __gmp_expr<mpz_t, U>, Op1> >, __gmp_expr<mpz_t, V>, Op2> >
  {
    mpz_class a(2), b(3);
    unsigned int c = 4, d = 5, e = 6;
    mpz_class f;
    f = c * (a + d) + (b - e); ASSERT_ALWAYS(f == 25);
  }
  {
    mpz_class a(2), b(3);
    unsigned int c = 4, d = 5, e = 6;
    mpz_class f;
    f = c * (a + d) - (b - e); ASSERT_ALWAYS(f == 31);
  }
}

static void
check_mpz_32 (void)
{
  // template <class T, class U, class V, class Op1, class Op2>
  // __gmp_expr<mpz_t, __gmp_binary_expr<__gmp_expr<mpz_t, __gmp_binary_expr
  // <__gmp_expr<mpz_t, T>, __gmp_expr<mpz_t, U>, Op1> >,
  // __gmp_expr<mpz_t, V>, Op2> >
  {
    mpz_class a(2), b(3), c(4);
    double d = 5.0, e = 6.0, f = 7.0;
    mpz_class g;
    g = (a + d) * (b - e) + (c + f); ASSERT_ALWAYS(g == -10);
  }
  {
    mpz_class a(2), b(3), c(4);
    double d = 5.0, e = 6.0, f = 7.0;
    mpz_class g;
    g = (a + d) * (b - e) - (c + f); ASSERT_ALWAYS(g == -32);
  }
}

void
check_mpq (void)
{
  // currently there's no ternary mpq operation
}

void
check_mpf (void)
{
  // currently there's no ternary mpf operation
}


int
main (void)
{
  tests_start();

  check_mpz_1 ();
  check_mpz_2 ();
  check_mpz_3 ();
  check_mpz_4 ();
  check_mpz_5 ();
  check_mpz_6 ();
  check_mpz_7 ();
  check_mpz_8 ();
  check_mpz_9 ();
  check_mpz_10 ();
  check_mpz_11 ();
  check_mpz_12 ();
  check_mpz_13 ();
  check_mpz_14 ();
  check_mpz_15 ();
  check_mpz_16 ();
  check_mpz_17 ();
  check_mpz_18 ();
  check_mpz_19 ();
  check_mpz_20 ();
  check_mpz_21 ();
  check_mpz_22 ();
  check_mpz_23 ();
  check_mpz_24 ();
  check_mpz_25 ();
  check_mpz_26 ();
  check_mpz_27 ();
  check_mpz_28 ();
  check_mpz_29 ();
  check_mpz_30 ();
  check_mpz_31 ();
  check_mpz_32 ();

  check_mpq();
  check_mpf();

  tests_end();
  return 0;
}
