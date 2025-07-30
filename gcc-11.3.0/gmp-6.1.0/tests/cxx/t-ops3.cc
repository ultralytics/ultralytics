/* Test mp*_class assignment operators (+=, -=, etc)

Copyright 2011 Free Software Foundation, Inc.

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

using namespace std;

#define FOR_ALL_SIGNED_BUILTIN(F) \
	F(signed char) \
	F(signed short) \
	F(signed int) \
	F(signed long) \
	F(float) \
	F(double)

#define FOR_ALL_BUILTIN(F) \
	FOR_ALL_SIGNED_BUILTIN(F) \
	F(char) \
	F(unsigned char) \
	F(unsigned short) \
	F(unsigned int) \
	F(unsigned long)

#define FOR_ALL_GMPXX(F) \
	F(mpz_class) \
	F(mpq_class) \
	F(mpf_class)

template<class T,class U> void f(T t, U u){
  T a=t;
  ASSERT_ALWAYS((a+=u)==(t+u)); ASSERT_ALWAYS(a==(t+u));
  ASSERT_ALWAYS((a-=u)==t); ASSERT_ALWAYS(a==t);
  ASSERT_ALWAYS((a*=u)==(t*u)); ASSERT_ALWAYS(a==(t*u));
  ASSERT_ALWAYS((a/=u)==t); ASSERT_ALWAYS(a==t);
  ASSERT_ALWAYS((a<<=5)==(t<<5)); ASSERT_ALWAYS(a==(t<<5));
  ASSERT_ALWAYS((a>>=5)==t); ASSERT_ALWAYS(a==t);
}

template<class T,class U> void g(T t, U u){
  T a=t;
  ASSERT_ALWAYS((a%=u)==(t%u)); ASSERT_ALWAYS(a==(t%u));
  a=t;
  ASSERT_ALWAYS((a&=u)==(t&u)); ASSERT_ALWAYS(a==(t&u));
  a=t;
  ASSERT_ALWAYS((a|=u)==(t|u)); ASSERT_ALWAYS(a==(t|u));
  a=t;
  ASSERT_ALWAYS((a^=u)==(t^u)); ASSERT_ALWAYS(a==(t^u));
}

template<class T> void h(T t){
  T a=t;
  ASSERT_ALWAYS((a<<=5)==(t<<5)); ASSERT_ALWAYS(a==(t<<5));
  ASSERT_ALWAYS((a>>=5)==t); ASSERT_ALWAYS(a==t);
}

template<class T, class U> void ffs(T t, U u){
#define F(V) f(t,(V)u);
	FOR_ALL_SIGNED_BUILTIN(F)
	FOR_ALL_GMPXX(F)
#undef F
#define F(V) f(t,-(V)u);
	FOR_ALL_GMPXX(F)
#undef F
}

template<class T, class U> void ff(T t, U u){
#define F(V) f(t,(V)u);
	FOR_ALL_BUILTIN(F)
	FOR_ALL_GMPXX(F)
#undef F
#define F(V) f(t,-(V)u);
	FOR_ALL_GMPXX(F)
#undef F
}

template<class U> void ggs(mpz_class t, U u){
#define F(V) g(t,(V)u);
	FOR_ALL_SIGNED_BUILTIN(F)
#undef F
	g(t,(mpz_class)u);
	g(t,-(mpz_class)u);
}

template<class U> void gg(mpz_class t, U u){
#define F(V) g(t,(V)u);
	FOR_ALL_BUILTIN(F)
#undef F
	g(t,(mpz_class)u);
	g(t,-(mpz_class)u);
}

void check(){
	mpz_class z=18;
	mpq_class q(7,2);
	mpf_class d=3.375;
	h(z); h(q); h(d);
	ff(z,13); ff(q,13); ff(d,13);
	ffs(z,-42); ffs(q,-42); ffs(d,-42);
	gg(z,33); ggs(z,-22);
}


int
main (void)
{
  tests_start();

  check();

  tests_end();
  return 0;
}
