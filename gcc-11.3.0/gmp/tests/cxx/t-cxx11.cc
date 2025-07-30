/* Test C++11 features

Copyright 2011, 2012 Free Software Foundation, Inc.

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

#include "gmp.h"
#include "gmpxx.h"
#include "gmp-impl.h"
#include "tests.h"

#if __GMPXX_USE_CXX11

#include <utility>
#include <type_traits>

void check_noexcept ()
{
  mpz_class z1, z2;
  mpq_class q1, q2;
  mpf_class f1, f2;
  static_assert(noexcept(z1 = std::move(z2)), "sorry");
  static_assert(noexcept(q1 = std::move(q2)), "sorry");
  static_assert(noexcept(f1 = std::move(f2)), "sorry");
  static_assert(noexcept(q1 = std::move(z1)), "sorry");
}

void check_common_type ()
{
#define CHECK_COMMON_TYPE1(T, Res) \
  static_assert(std::is_same<std::common_type<T>::type, Res>::value, "sorry")
#define CHECK_COMMON_TYPE(T, U, Res) \
  static_assert(std::is_same<std::common_type<T, U>::type, Res>::value, "sorry")
#define CHECK_COMMON_TYPE_BUILTIN1(T, Res) \
  CHECK_COMMON_TYPE(  signed char , T, Res); \
  CHECK_COMMON_TYPE(unsigned char , T, Res); \
  CHECK_COMMON_TYPE(  signed short, T, Res); \
  CHECK_COMMON_TYPE(unsigned short, T, Res); \
  CHECK_COMMON_TYPE(  signed int  , T, Res); \
  CHECK_COMMON_TYPE(unsigned int  , T, Res); \
  CHECK_COMMON_TYPE(  signed long , T, Res); \
  CHECK_COMMON_TYPE(unsigned long , T, Res); \
  CHECK_COMMON_TYPE(float , T, Res); \
  CHECK_COMMON_TYPE(double, T, Res)
#define CHECK_COMMON_TYPE_BUILTIN2(T, Res) \
  CHECK_COMMON_TYPE(T,   signed char , Res); \
  CHECK_COMMON_TYPE(T, unsigned char , Res); \
  CHECK_COMMON_TYPE(T,   signed short, Res); \
  CHECK_COMMON_TYPE(T, unsigned short, Res); \
  CHECK_COMMON_TYPE(T,   signed int  , Res); \
  CHECK_COMMON_TYPE(T, unsigned int  , Res); \
  CHECK_COMMON_TYPE(T,   signed long , Res); \
  CHECK_COMMON_TYPE(T, unsigned long , Res); \
  CHECK_COMMON_TYPE(T, float , Res); \
  CHECK_COMMON_TYPE(T, double, Res)
#define CHECK_COMMON_TYPE_BUILTIN(T, Res) \
  CHECK_COMMON_TYPE_BUILTIN1(T, Res); \
  CHECK_COMMON_TYPE_BUILTIN2(T, Res)
  /* These would just work with implicit conversions */
  CHECK_COMMON_TYPE (mpz_class, mpq_class, mpq_class);
  CHECK_COMMON_TYPE (mpz_class, mpf_class, mpf_class);
  CHECK_COMMON_TYPE (mpf_class, mpq_class, mpf_class);

  CHECK_COMMON_TYPE_BUILTIN (mpz_class, mpz_class);
  CHECK_COMMON_TYPE_BUILTIN (mpq_class, mpq_class);
  CHECK_COMMON_TYPE_BUILTIN (mpf_class, mpf_class);

  mpz_class z; mpq_class q; mpf_class f;

  CHECK_COMMON_TYPE (decltype(-z), mpz_class, mpz_class);
  CHECK_COMMON_TYPE (decltype(-q), mpq_class, mpq_class);
  CHECK_COMMON_TYPE (decltype(-f), mpf_class, mpf_class);

  CHECK_COMMON_TYPE (decltype(-z), mpq_class, mpq_class);
  CHECK_COMMON_TYPE (decltype(-z), mpf_class, mpf_class);
  CHECK_COMMON_TYPE (decltype(-q), mpf_class, mpf_class);

  /* These require a common_type specialization */
  CHECK_COMMON_TYPE (decltype(-z), decltype(z+z), mpz_class);
  CHECK_COMMON_TYPE (decltype(-q), decltype(q+q), mpq_class);
  CHECK_COMMON_TYPE (decltype(-f), decltype(f+f), mpf_class);

  CHECK_COMMON_TYPE (decltype(-q), mpz_class, mpq_class);
  CHECK_COMMON_TYPE (decltype(-f), mpz_class, mpf_class);
  CHECK_COMMON_TYPE (decltype(-f), mpq_class, mpf_class);

  CHECK_COMMON_TYPE (decltype(-z), decltype(-q), mpq_class);
  CHECK_COMMON_TYPE (decltype(-z), decltype(-f), mpf_class);
  CHECK_COMMON_TYPE (decltype(-q), decltype(-f), mpf_class);

  /* common_type now decays */
  CHECK_COMMON_TYPE (decltype(-z), decltype(-z), mpz_class);
  CHECK_COMMON_TYPE (decltype(-q), decltype(-q), mpq_class);
  CHECK_COMMON_TYPE (decltype(-f), decltype(-f), mpf_class);
  CHECK_COMMON_TYPE1 (decltype(-z), mpz_class);
  CHECK_COMMON_TYPE1 (decltype(-q), mpq_class);
  CHECK_COMMON_TYPE1 (decltype(-f), mpf_class);

  /* Painful */
  CHECK_COMMON_TYPE_BUILTIN (decltype(-z), mpz_class);
  CHECK_COMMON_TYPE_BUILTIN (decltype(-q), mpq_class);
  CHECK_COMMON_TYPE_BUILTIN (decltype(-f), mpf_class);
}

template<class T, class U = T>
void check_move_init ()
{
  {
    // Delete moved-from x1
    T x1 = 3;
    U x2 = std::move(x1);
    ASSERT_ALWAYS (x2 == 3);
  }
  {
    // Assign to moved-from x1
    T x1 = 2;
    U x2 = std::move(x1);
    x1 = -7;
    ASSERT_ALWAYS (x1 == -7);
    ASSERT_ALWAYS (x2 == 2);
  }
}

template<class T, class U = T>
void check_move_assign ()
{
  {
    // Delete moved-from x1
    T x1 = 3; U x2;
    x2 = std::move(x1);
    ASSERT_ALWAYS (x2 == 3);
  }
  {
    // Assign to moved-from x1
    T x1 = 2; U x2;
    x2 = std::move(x1);
    x1 = -7;
    ASSERT_ALWAYS (x1 == -7);
    ASSERT_ALWAYS (x2 == 2);
  }
  {
    // Self move-assign (not necessary, but it happens to work...)
    T x = 4;
    x = std::move(x);
    ASSERT_ALWAYS (x == 4);
  }
}

void check_user_defined_literal ()
{
  ASSERT_ALWAYS (123_mpz % 5 == 3);
  ASSERT_ALWAYS (-11_mpq / 22 == -.5);
  ASSERT_ALWAYS (112.5e-1_mpf * 4 == 45);
  {
    mpz_class ref ( "123456789abcdef0123456789abcdef0123", 16);
    ASSERT_ALWAYS (0x123456789abcdef0123456789abcdef0123_mpz == ref);
  }
}

// Check for explicit conversion to bool
void implicit_bool(bool);
int implicit_bool(...);

void check_bool_conversion ()
{
  const mpz_class zn = -2;
  const mpq_class qn = -2;
  const mpf_class fn = -2;
  const mpz_class z0 =  0;
  const mpq_class q0 =  0;
  const mpf_class f0 =  0;
  const mpz_class zp = +2;
  const mpq_class qp = +2;
  const mpf_class fp = +2;
  if (zn && qn && fn && zp && qp && fp && !z0 && !q0 && !f0)
    {
      if (z0 || q0 || f0) ASSERT_ALWAYS(false);
    }
  else ASSERT_ALWAYS(false);
  decltype(implicit_bool(zn)) zi = 1;
  decltype(implicit_bool(qn)) qi = 1;
  decltype(implicit_bool(fn)) fi = 1;
  (void)(zi+qi+fi);
}

int
main (void)
{
  tests_start();

  check_noexcept();
  check_common_type();
  check_move_init<mpz_class>();
  check_move_init<mpq_class>();
  check_move_init<mpf_class>();
  check_move_assign<mpz_class>();
  check_move_assign<mpq_class>();
  check_move_assign<mpf_class>();
  check_move_init<mpz_class,mpq_class>();
  check_move_assign<mpz_class,mpq_class>();
  check_user_defined_literal();
  check_bool_conversion();

  tests_end();
  return 0;
}

#else
int main () { return 0; }
#endif
