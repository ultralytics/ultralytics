/* Test mp*_class assignment operators.

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
#include <string>

#include "gmp.h"
#include "gmpxx.h"
#include "gmp-impl.h"
#include "tests.h"

using std::string;
using std::invalid_argument;


void
check_mpz (void)
{
  // operator=(const mpz_class &)
  {
    mpz_class a(123), b;
    b = a; ASSERT_ALWAYS(b == 123);
  }

  // template <class T, class U> operator=(const __gmp_expr<T, U> &)
  // not tested here, see t-unary.cc, t-binary.cc

  // operator=(signed char)
  {
    signed char a = -127;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == -127);
  }

  // operator=(unsigned char)
  {
    unsigned char a = 255;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 255);
  }

  // either signed or unsigned char, machine dependent
  {
    mpz_class a;
    a = 'A'; ASSERT_ALWAYS(a == 65);
  }
  {
    mpz_class a;
    a = 'z'; ASSERT_ALWAYS(a == 122);
  }

  // operator=(signed int)
  {
    signed int a = 0;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 0);
  }
  {
    signed int a = -123;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == -123);
  }
  {
    signed int a = 32767;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 32767);
  }

  // operator=(unsigned int)
  {
    unsigned int a = 65535u;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 65535u);
  }

  // operator=(signed short int)
  {
    signed short int a = -12345;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == -12345);
  }

  // operator=(unsigned short int)
  {
    unsigned short int a = 54321u;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 54321u);
  }

  // operator=(signed long int)
  {
    signed long int a = -1234567890L;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == -1234567890L);
  }

  // operator=(unsigned long int)
  {
    unsigned long int a = 3456789012UL;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 3456789012UL);
  }

  // operator=(float)
  {
    float a = 123.0;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 123);
  }

  // operator=(double)
  {
    double a = 0.0;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 0);
  }
  {
    double a = -12.375;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == -12);
  }
  {
    double a = 6.789e+3;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 6789);
  }
  {
    double a = 9.375e-1;
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 0);
  }

  // operator=(long double)
  // currently not implemented

  // operator=(const char *)
  {
    const char *a = "1234567890";
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 1234567890L);
  }

  // operator=(const std::string &)
  {
    string a("1234567890");
    mpz_class b;
    b = a; ASSERT_ALWAYS(b == 1234567890L);
  }

  // operator=(const char *) with invalid
  {
    try {
      const char *a = "abc";
      mpz_class b;
      b = a;
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // operator=(const std::string &) with invalid
  {
    try {
      string a("def");
      mpz_class b;
      b = a;
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // swap(mpz_class &)
  {
    mpz_class a(123);
    mpz_class b(456);
    a.swap(b);
    a.swap(a);
    ASSERT_ALWAYS(a == 456);
    ASSERT_ALWAYS(b == 123);
  }

  // swap(mpz_class &, mpz_class &)
  {
    mpz_class a(123);
    mpz_class b(456);
    ::swap(a, b);
    ::swap(a, a);
    ASSERT_ALWAYS(a == 456);
    ASSERT_ALWAYS(b == 123);
  }
  {
    using std::swap;
    mpz_class a(123);
    mpz_class b(456);
    swap(a, b);
    swap(a, a);
    ASSERT_ALWAYS(a == 456);
    ASSERT_ALWAYS(b == 123);
  }
}

void
check_mpq (void)
{
  // operator=(const mpq_class &)
  {
    mpq_class a(1, 2), b;
    b = a; ASSERT_ALWAYS(b == 0.5);
  }

  // template <class T, class U> operator=(const __gmp_expr<T, U> &)
  // not tested here, see t-unary.cc, t-binary.cc

  // operator=(signed char)
  {
    signed char a = -127;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == -127);
  }

  // operator=(unsigned char)
  {
    unsigned char a = 255;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 255);
  }

  // either signed or unsigned char, machine dependent
  {
    mpq_class a;
    a = 'A'; ASSERT_ALWAYS(a == 65);
  }
  {
    mpq_class a;
    a = 'z'; ASSERT_ALWAYS(a == 122);
  }

  // operator=(signed int)
  {
    signed int a = 0;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 0);
  }
  {
    signed int a = -123;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == -123);
  }
  {
    signed int a = 32767;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 32767);
  }

  // operator=(unsigned int)
  {
    unsigned int a = 65535u;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 65535u);
  }

  // operator=(signed short int)
  {
    signed short int a = -12345;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == -12345);
  }

  // operator=(unsigned short int)
  {
    unsigned short int a = 54321u;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 54321u);
  }

  // operator=(signed long int)
  {
    signed long int a = -1234567890L;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == -1234567890L);
  }

  // operator=(unsigned long int)
  {
    unsigned long int a = 3456789012UL;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 3456789012UL);
  }

  // operator=(float)
  {
    float a = 123.0;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 123);
  }

  // operator=(double)
  {
    double a = 0.0;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 0);
  }
  {
    double a = -12.375;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == -12.375);
  }
  {
    double a = 6.789e+3;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 6789);
  }
  {
    double a = 9.375e-1;
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 0.9375);
  }

  // operator=(long double)
  // currently not implemented

  // operator=(const char *)
  {
    const char *a = "1234567890";
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 1234567890L);
  }

  // operator=(const std::string &)
  {
    string a("1234567890");
    mpq_class b;
    b = a; ASSERT_ALWAYS(b == 1234567890L);
  }

  // operator=(const char *) with invalid
  {
    try {
      const char *a = "abc";
      mpq_class b;
      b = a;
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // operator=(const std::string &) with invalid
  {
    try {
      string a("def");
      mpq_class b;
      b = a;
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // swap(mpq_class &)
  {
    mpq_class a(3, 2);
    mpq_class b(-1, 4);
    a.swap(b);
    a.swap(a);
    ASSERT_ALWAYS(a == -.25);
    ASSERT_ALWAYS(b == 1.5);
  }

  // swap(mpq_class &, mpq_class &)
  {
    mpq_class a(3, 2);
    mpq_class b(-1, 4);
    ::swap(a, b);
    ::swap(a, a);
    ASSERT_ALWAYS(a == -.25);
    ASSERT_ALWAYS(b == 1.5);
  }
  {
    using std::swap;
    mpq_class a(3, 2);
    mpq_class b(-1, 4);
    swap(a, b);
    swap(a, a);
    ASSERT_ALWAYS(a == -.25);
    ASSERT_ALWAYS(b == 1.5);
  }
}

void
check_mpf (void)
{
  // operator=(const mpf_class &)
  {
    mpf_class a(123), b;
    b = a; ASSERT_ALWAYS(b == 123);
  }

  // template <class T, class U> operator=(const __gmp_expr<T, U> &)
  // not tested here, see t-unary.cc, t-binary.cc

  // operator=(signed char)
  {
    signed char a = -127;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == -127);
  }

  // operator=(unsigned char)
  {
    unsigned char a = 255;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 255);
  }

  // either signed or unsigned char, machine dependent
  {
    mpf_class a;
    a = 'A'; ASSERT_ALWAYS(a == 65);
  }
  {
    mpf_class a;
    a = 'z'; ASSERT_ALWAYS(a == 122);
  }

  // operator=(signed int)
  {
    signed int a = 0;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 0);
  }
  {
    signed int a = -123;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == -123);
  }
  {
    signed int a = 32767;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 32767);
  }

  // operator=(unsigned int)
  {
    unsigned int a = 65535u;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 65535u);
  }

  // operator=(signed short int)
  {
    signed short int a = -12345;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == -12345);
  }

  // operator=(unsigned short int)
  {
    unsigned short int a = 54321u;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 54321u);
  }

  // operator=(signed long int)
  {
    signed long int a = -1234567890L;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == -1234567890L);
  }

  // operator=(unsigned long int)
  {
    unsigned long int a = 3456789012UL;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 3456789012UL);
  }

  // operator=(float)
  {
    float a = 123.0;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 123);
  }

  // operator=(double)
  {
    double a = 0.0;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 0);
  }
  {
    double a = -12.375;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == -12.375);
  }
  {
    double a = 6.789e+3;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 6789);
  }
  {
    double a = 9.375e-1;
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 0.9375);
  }

  // operator=(long double)
  // currently not implemented

  // operator=(const char *)
  {
    const char *a = "1234567890";
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 1234567890L);
  }

  // operator=(const std::string &)
  {
    string a("1234567890");
    mpf_class b;
    b = a; ASSERT_ALWAYS(b == 1234567890L);
  }

  // operator=(const char *) with invalid
  {
    try {
      const char *a = "abc";
      mpf_class b;
      b = a;
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // operator=(const std::string &) with invalid
  {
    try {
      string a("def");
      mpf_class b;
      b = a;
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // swap(mpf_class &)
  {
    mpf_class a(123);
    mpf_class b(456);
    a.swap(b);
    a.swap(a);
    ASSERT_ALWAYS(a == 456);
    ASSERT_ALWAYS(b == 123);
  }

  // swap(mpf_class &, mpf_class &)
  {
    mpf_class a(123);
    mpf_class b(456);
    ::swap(a, b);
    ::swap(a, a);
    ASSERT_ALWAYS(a == 456);
    ASSERT_ALWAYS(b == 123);
  }
  {
    using std::swap;
    mpf_class a(123);
    mpf_class b(456);
    swap(a, b);
    swap(a, a);
    ASSERT_ALWAYS(a == 456);
    ASSERT_ALWAYS(b == 123);
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
