/* Test mp*_class constructors.

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

using namespace std;


void
check_mpz (void)
{
  // mpz_class()
  {
    mpz_class a; ASSERT_ALWAYS(a == 0);
  }

  // mpz_class(const mpz_class &)
  // see below

  // template <class T, class U> mpz_class(const __gmp_expr<T, U> &)
  // not tested here, see t-unary.cc, t-binary.cc

  // mpz_class(signed char)
  {
    signed char a = -127;
    mpz_class b(a); ASSERT_ALWAYS(b == -127);
  }

  // mpz_class(unsigned char)
  {
    unsigned char a = 255;
    mpz_class b(a); ASSERT_ALWAYS(b == 255);
  }

  // either signed or unsigned char, machine dependent
  {
    mpz_class a('A'); ASSERT_ALWAYS(a == 65);
  }
  {
    mpz_class a('z'); ASSERT_ALWAYS(a == 122);
  }

  // mpz_class(signed int)
  {
    signed int a = 0;
    mpz_class b(a); ASSERT_ALWAYS(b == 0);
  }
  {
    signed int a = -123;
    mpz_class b(a); ASSERT_ALWAYS(b == -123);
  }
  {
    signed int a = 4567;
    mpz_class b(a); ASSERT_ALWAYS(b == 4567);
  }

  // mpz_class(unsigned int)
  {
    unsigned int a = 890;
    mpz_class b(a); ASSERT_ALWAYS(b == 890);
  }

  // mpz_class(signed short int)
  {
    signed short int a = -12345;
    mpz_class b(a); ASSERT_ALWAYS(b == -12345);
  }

  // mpz_class(unsigned short int)
  {
    unsigned short int a = 54321u;
    mpz_class b(a); ASSERT_ALWAYS(b == 54321u);
  }

  // mpz_class(signed long int)
  {
    signed long int a = -1234567890L;
    mpz_class b(a); ASSERT_ALWAYS(b == -1234567890L);
  }

  // mpz_class(unsigned long int)
  {
    unsigned long int a = 1UL << 30;
    mpz_class b(a); ASSERT_ALWAYS(b == 1073741824L);
  }

  // mpz_class(float)
  {
    float a = 123.45;
    mpz_class b(a); ASSERT_ALWAYS(b == 123);
  }

  // mpz_class(double)
  {
    double a = 3.141592653589793238;
    mpz_class b(a); ASSERT_ALWAYS(b == 3);
  }

  // mpz_class(long double)
  // currently not implemented

  // mpz_class(const char *)
  {
    const char *a = "1234567890";
    mpz_class b(a); ASSERT_ALWAYS(b == 1234567890L);
  }

  // mpz_class(const char *, int)
  {
    const char *a = "FFFF";
    int base = 16;
    mpz_class b(a, base); ASSERT_ALWAYS(b == 65535u);
  }

  // mpz_class(const std::string &)
  {
    string a("1234567890");
    mpz_class b(a); ASSERT_ALWAYS(b == 1234567890L);
  }

  // mpz_class(const std::string &, int)
  {
    string a("7777");
    int base = 8;
    mpz_class b(a, base); ASSERT_ALWAYS(b == 4095);
  }

  // mpz_class(const char *) with invalid
  {
    try {
      const char *a = "ABC";
      mpz_class b(a);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpz_class(const char *, int) with invalid
  {
    try {
      const char *a = "GHI";
      int base = 16;
      mpz_class b(a, base);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpz_class(const std::string &) with invalid
  {
    try {
      string a("abc");
      mpz_class b(a);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpz_class(const std::string &, int) with invalid
  {
    try {
      string a("ZZZ");
      int base = 8;
      mpz_class b(a, base);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpz_class(mpz_srcptr)
  {
    mpz_t a;
    mpz_init_set_ui(a, 100);
    mpz_class b(a); ASSERT_ALWAYS(b == 100);
    mpz_clear(a);
  }

  // mpz_class(const mpz_class &)
  {
    mpz_class a(12345); // tested above, assume it works
    mpz_class b(a); ASSERT_ALWAYS(b == 12345);
  }

  // no constructor for bool, but it gets casted to int
  {
    bool a = true;
    mpz_class b(a); ASSERT_ALWAYS(b == 1);
  }
  {
    bool a = false;
    mpz_class b(a); ASSERT_ALWAYS(b == 0);
  }
}

void
check_mpq (void)
{
  // mpq_class()
  {
    mpq_class a; ASSERT_ALWAYS(a == 0);
  }

  // mpq_class(const mpq_class &)
  // see below

  // template <class T, class U> mpq_class(const __gmp_expr<T, U> &)
  // not tested here, see t-unary.cc, t-binary.cc

  // mpq_class(signed char)
  {
    signed char a = -127;
    mpq_class b(a); ASSERT_ALWAYS(b == -127);
  }

  // mpq_class(unsigned char)
  {
    unsigned char a = 255;
    mpq_class b(a); ASSERT_ALWAYS(b == 255);
  }

  // either signed or unsigned char, machine dependent
  {
    mpq_class a('A'); ASSERT_ALWAYS(a == 65);
  }
  {
    mpq_class a('z'); ASSERT_ALWAYS(a == 122);
  }

  // mpq_class(signed int)
  {
    signed int a = 0;
    mpq_class b(a); ASSERT_ALWAYS(b == 0);
  }
  {
    signed int a = -123;
    mpq_class b(a); ASSERT_ALWAYS(b == -123);
  }
  {
    signed int a = 4567;
    mpq_class b(a); ASSERT_ALWAYS(b == 4567);
  }

  // mpq_class(unsigned int)
  {
    unsigned int a = 890;
    mpq_class b(a); ASSERT_ALWAYS(b == 890);
  }

  // mpq_class(signed short int)
  {
    signed short int a = -12345;
    mpq_class b(a); ASSERT_ALWAYS(b == -12345);
  }

  // mpq_class(unsigned short int)
  {
    unsigned short int a = 54321u;
    mpq_class b(a); ASSERT_ALWAYS(b == 54321u);
  }

  // mpq_class(signed long int)
  {
    signed long int a = -1234567890L;
    mpq_class b(a); ASSERT_ALWAYS(b == -1234567890L);
  }

  // mpq_class(unsigned long int)
  {
    unsigned long int a = 1UL << 30;
    mpq_class b(a); ASSERT_ALWAYS(b == 1073741824L);
  }

  // mpq_class(float)
  {
    float a = 0.625;
    mpq_class b(a); ASSERT_ALWAYS(b == 0.625);
  }

  // mpq_class(double)
  {
    double a = 1.25;
    mpq_class b(a); ASSERT_ALWAYS(b == 1.25);
  }

  // mpq_class(long double)
  // currently not implemented

  // mpq_class(const char *)
  {
    const char *a = "1234567890";
    mpq_class b(a); ASSERT_ALWAYS(b == 1234567890L);
  }

  // mpq_class(const char *, int)
  {
    const char *a = "FFFF";
    int base = 16;
    mpq_class b(a, base); ASSERT_ALWAYS(b == 65535u);
    mpq_class c(0, 1); ASSERT_ALWAYS(c == 0);
  }

  // mpq_class(const std::string &)
  {
    string a("1234567890");
    mpq_class b(a); ASSERT_ALWAYS(b == 1234567890L);
  }

  // mpq_class(const std::string &, int)
  {
    string a("7777");
    int base = 8;
    mpq_class b(a, base); ASSERT_ALWAYS(b == 4095);
  }

  // mpq_class(const char *) with invalid
  {
    try {
      const char *a = "abc";
      mpq_class b(a);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpq_class(const char *, int) with invalid
  {
    try {
      const char *a = "ZZZ";
      int base = 16;
      mpq_class b (a, base);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpq_class(const std::string &) with invalid
  {
    try {
      string a("abc");
      mpq_class b(a);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpq_class(const std::string &, int) with invalid
  {
    try {
      string a("ZZZ");
      int base = 8;
      mpq_class b (a, base);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpq_class(mpq_srcptr)
  {
    mpq_t a;
    mpq_init(a);
    mpq_set_ui(a, 100, 1);
    mpq_class b(a); ASSERT_ALWAYS(b == 100);
    mpq_clear(a);
  }

  // mpq_class(const mpz_class &, const mpz_class &)
  {
    mpz_class a(123), b(4); // tested above, assume it works
    mpq_class c(a, b); ASSERT_ALWAYS(c == 30.75);
  }
  {
    mpz_class a(-1), b(2);  // tested above, assume it works
    mpq_class c(a, b); ASSERT_ALWAYS(c == -0.5);
  }
  {
    mpz_class a(5), b(4); // tested above, assume it works
    mpq_class c(a, b); ASSERT_ALWAYS(c == 1.25);
  }

  // mpq_class(const mpz_class &)
  {
    mpq_class a(12345); // tested above, assume it works
    mpq_class b(a); ASSERT_ALWAYS(b == 12345);
  }

  // no constructor for bool, but it gets casted to int
  {
    bool a = true;
    mpq_class b(a); ASSERT_ALWAYS(b == 1);
  }
  {
    bool a = false;
    mpq_class b(a); ASSERT_ALWAYS(b == 0);
  }
}

void
check_mpf (void)
{
  // mpf_class()
  {
    mpf_class a; ASSERT_ALWAYS(a == 0);
  }

  // mpf_class(const mpf_class &)
  // mpf_class(const mpf_class &, unsigned long int)
  // see below

  // template <class T, class U> mpf_class(const __gmp_expr<T, U> &)
  // template <class T, class U> mpf_class(const __gmp_expr<T, U> &,
  //                                       unsigned long int)
  // not tested here, see t-unary.cc, t-binary.cc

  // mpf_class(signed char)
  {
    signed char a = -127;
    mpf_class b(a); ASSERT_ALWAYS(b == -127);
  }

  // mpf_class(signed char, unsigned long int)
  {
    signed char a = -1;
    int prec = 64;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == -1);
  }

  // mpf_class(unsigned char)
  {
    unsigned char a = 255;
    mpf_class b(a); ASSERT_ALWAYS(b == 255);
  }

  // mpf_class(unsigned char, unsigned long int)
  {
    unsigned char a = 128;
    int prec = 128;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 128);
  }

  // either signed or unsigned char, machine dependent
  {
    mpf_class a('A'); ASSERT_ALWAYS(a == 65);
  }
  {
    int prec = 256;
    mpf_class a('z', prec); ASSERT_ALWAYS(a == 122);
  }

  // mpf_class(signed int)
  {
    signed int a = 0;
    mpf_class b(a); ASSERT_ALWAYS(b == 0);
  }
  {
    signed int a = -123;
    mpf_class b(a); ASSERT_ALWAYS(b == -123);
  }
  {
    signed int a = 4567;
    mpf_class b(a); ASSERT_ALWAYS(b == 4567);
  }

  // mpf_class(signed int, unsigned long int)
  {
    signed int a = -123;
    int prec = 64;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == -123);
  }

  // mpf_class(unsigned int)
  {
    unsigned int a = 890;
    mpf_class b(a); ASSERT_ALWAYS(b == 890);
  }

  // mpf_class(unsigned int, unsigned long int)
  {
    unsigned int a = 890;
    int prec = 128;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 890);
  }

  // mpf_class(signed short int)
  {
    signed short int a = -12345;
    mpf_class b(a); ASSERT_ALWAYS(b == -12345);
  }

  // mpf_class(signed short int, unsigned long int)
  {
    signed short int a = 6789;
    int prec = 256;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 6789);
  }

  // mpf_class(unsigned short int)
  {
    unsigned short int a = 54321u;
    mpf_class b(a); ASSERT_ALWAYS(b == 54321u);
  }

  // mpf_class(unsigned short int, unsigned long int)
  {
    unsigned short int a = 54321u;
    int prec = 64;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 54321u);
  }

  // mpf_class(signed long int)
  {
    signed long int a = -1234567890L;
    mpf_class b(a); ASSERT_ALWAYS(b == -1234567890L);
  }

  // mpf_class(signed long int, unsigned long int)
  {
    signed long int a = -1234567890L;
    int prec = 128;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == -1234567890L);
  }

  // mpf_class(unsigned long int)
  {
    unsigned long int a = 3456789012UL;
    mpf_class b(a); ASSERT_ALWAYS(b == 3456789012UL);
  }

  // mpf_class(unsigned long int, unsigned long int)
  {
    unsigned long int a = 3456789012UL;
    int prec = 256;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 3456789012UL);
  }

  // mpf_class(float)
  {
    float a = 1234.5;
    mpf_class b(a); ASSERT_ALWAYS(b == 1234.5);
  }

  // mpf_class(float, unsigned long int)
  {
    float a = 1234.5;
    int prec = 64;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 1234.5);
  }

  // mpf_class(double)
  {
    double a = 12345.0;
    mpf_class b(a); ASSERT_ALWAYS(b == 12345);
  }
  {
    double a = 1.2345e+4;
    mpf_class b(a); ASSERT_ALWAYS(b == 12345);
  }
  {
    double a = 312.5e-2;
    mpf_class b(a); ASSERT_ALWAYS(b == 3.125);
  }

  // mpf_class(double, unsigned long int)
  {
    double a = 5.4321e+4;
    int prec = 128;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 54321L);
  }

  // mpf_class(long double)
  // mpf_class(long double, unsigned long int)
  // currently not implemented

  // mpf_class(const char *)
  {
    const char *a = "1234567890";
    mpf_class b(a); ASSERT_ALWAYS(b == 1234567890L);
  }

  // mpf_class(const char *, unsigned long int, int = 0)
  {
    const char *a = "1234567890";
    int prec = 256;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 1234567890L);
  }
  {
    const char *a = "777777";
    int prec = 64, base = 8;
    mpf_class b(a, prec, base); ASSERT_ALWAYS(b == 262143L);
  }

  // mpf_class(const std::string &)
  {
    string a("1234567890");
    mpf_class b(a); ASSERT_ALWAYS(b == 1234567890L);
  }

  // mpf_class(const std::string &, unsigned long int, int = 0)
  {
    string a("1234567890");
    int prec = 128;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 1234567890L);
  }
  {
    string a("FFFF");
    int prec = 256, base = 16;
    mpf_class b(a, prec, base); ASSERT_ALWAYS(b == 65535u);
  }

  // mpf_class(const char *) with invalid
  {
    try {
      const char *a = "abc";
      mpf_class b(a);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpf_class(const char *, unsigned long int, int = 0) with invalid
  {
    try {
      const char *a = "def";
      int prec = 256;
      mpf_class b(a, prec); ASSERT_ALWAYS(b == 1234567890L);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }
  {
    try {
      const char *a = "ghi";
      int prec = 64, base = 8;
      mpf_class b(a, prec, base); ASSERT_ALWAYS(b == 262143L);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpf_class(const std::string &) with invalid
  {
    try {
      string a("abc");
      mpf_class b(a); ASSERT_ALWAYS(b == 1234567890L);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpf_class(const std::string &, unsigned long int, int = 0) with invalid
  {
    try {
      string a("def");
      int prec = 128;
      mpf_class b(a, prec); ASSERT_ALWAYS(b == 1234567890L);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }
  {
    try {
      string a("ghi");
      int prec = 256, base = 16;
      mpf_class b(a, prec, base); ASSERT_ALWAYS(b == 65535u);
      ASSERT_ALWAYS (0);  /* should not be reached */
    } catch (invalid_argument) {
    }
  }

  // mpf_class(mpf_srcptr)
  {
    mpf_t a;
    mpf_init_set_ui(a, 100);
    mpf_class b(a); ASSERT_ALWAYS(b == 100);
    mpf_clear(a);
  }

  // mpf_class(mpf_srcptr, unsigned long int)
  {
    mpf_t a;
    int prec = 64;
    mpf_init_set_ui(a, 100);
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 100);
    mpf_clear(a);
  }

  // mpf_class(const mpf_class &)
  {
    mpf_class a(12345); // tested above, assume it works
    mpf_class b(a); ASSERT_ALWAYS(b == 12345);
  }

  // mpf_class(const mpf_class &, unsigned long int)
  {
    mpf_class a(12345); // tested above, assume it works
    int prec = 64;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 12345);
  }

  // no constructors for bool, but it gets casted to int
  {
    bool a = true;
    mpf_class b(a); ASSERT_ALWAYS(b == 1);
  }
  {
    bool a = false;
    mpf_class b(a); ASSERT_ALWAYS(b == 0);
  }
  {
    bool a = true;
    int prec = 128;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 1);
  }
  {
    bool a = false;
    int prec = 256;
    mpf_class b(a, prec); ASSERT_ALWAYS(b == 0);
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
