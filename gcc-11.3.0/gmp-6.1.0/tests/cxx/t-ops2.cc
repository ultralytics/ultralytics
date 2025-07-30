/* Test mp*_class operators and functions.

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

#include <math.h>

#include "gmp.h"
#include "gmpxx.h"
#include "gmp-impl.h"
#include "tests.h"


#define CHECK1(Type,a,fun) \
  ASSERT_ALWAYS(fun((Type)(a))==fun(a))
#define CHECK(Type1,Type2,a,b,op) \
  ASSERT_ALWAYS(((Type1)(a) op (Type2)(b))==((a) op (b)))
#define CHECK_G(Type,a,b,op) \
  CHECK(Type,Type,a,b,op)
#define CHECK_UI(Type,a,b,op) \
  CHECK(Type,unsigned long,a,b,op); \
  CHECK(unsigned long,Type,a,b,op)
#define CHECK_SI(Type,a,b,op) \
  CHECK(Type,long,a,b,op); \
  CHECK(long,Type,a,b,op)
#define CHECK_D(Type,a,b,op) \
  CHECK(Type,double,a,b,op); \
  CHECK(double,Type,a,b,op)
#define CHECK_MPZ(Type,a,b,op) \
  CHECK(Type,mpz_class,a,b,op); \
  CHECK(mpz_class,Type,a,b,op)
#define CHECK_MPQ(Type,a,b,op) \
  CHECK(Type,mpq_class,a,b,op); \
  CHECK(mpq_class,Type,a,b,op)
#define CHECK_ALL_SIGNED(Type,a,b,op) \
  CHECK_G(Type,a,b,op); \
  CHECK_SI(Type,a,b,op); \
  CHECK_D(Type,a,b,op)
#define CHECK_ALL_SIGNS(Type,a,b,op) \
  CHECK_ALL_SIGNED(Type,a,b,op); \
  CHECK_ALL_SIGNED(Type,-(a),b,op); \
  CHECK_ALL_SIGNED(Type,a,-(b),op); \
  CHECK_ALL_SIGNED(Type,-(a),-(b),op)
#define CHECK_ALL(Type,a,b,op) \
  CHECK_ALL_SIGNED(Type,a,b,op); \
  CHECK_UI(Type,a,b,op)
#define CHECK_ALL_SIGNED_COMPARISONS(Type,a,b) \
  CHECK_ALL_SIGNED(Type,a,b,<); \
  CHECK_ALL_SIGNED(Type,a,b,>); \
  CHECK_ALL_SIGNED(Type,a,b,<=); \
  CHECK_ALL_SIGNED(Type,a,b,>=); \
  CHECK_ALL_SIGNED(Type,a,b,==); \
  CHECK_ALL_SIGNED(Type,a,b,!=)
#define CHECK_ALL_SIGNS_COMPARISONS(Type,a,b) \
  CHECK_ALL_SIGNS(Type,a,b,<); \
  CHECK_ALL_SIGNS(Type,a,b,>); \
  CHECK_ALL_SIGNS(Type,a,b,<=); \
  CHECK_ALL_SIGNS(Type,a,b,>=); \
  CHECK_ALL_SIGNS(Type,a,b,==); \
  CHECK_ALL_SIGNS(Type,a,b,!=)
#define CHECK_ALL_COMPARISONS(Type,a,b) \
  CHECK_ALL(Type,a,b,<); \
  CHECK_ALL(Type,a,b,>); \
  CHECK_ALL(Type,a,b,<=); \
  CHECK_ALL(Type,a,b,>=); \
  CHECK_ALL(Type,a,b,==); \
  CHECK_ALL(Type,a,b,!=)


void checkz (){
  CHECK_ALL(mpz_class,5,2,+);
  CHECK_ALL(mpz_class,5,2,-);
  CHECK_ALL(mpz_class,5,2,*);
  CHECK_ALL(mpz_class,5,2,/);
  CHECK_ALL(mpz_class,5,2,%);
  CHECK_ALL_COMPARISONS(mpz_class,5,2);
  CHECK_ALL_SIGNS(mpz_class,11,3,+);
  CHECK_ALL_SIGNS(mpz_class,11,3,-);
  CHECK_ALL_SIGNS(mpz_class,11,3,*);
  CHECK_ALL_SIGNS(mpz_class,11,3,/);
  CHECK_ALL_SIGNS(mpz_class,11,3,%);
  CHECK_ALL_SIGNS(mpz_class,17,2,*);
  CHECK_ALL_SIGNS(mpz_class,17,2,/);
  CHECK_ALL_SIGNS(mpz_class,17,2,%);
  CHECK(unsigned long,mpz_class,5,-2,/);
  CHECK(unsigned long,mpz_class,5,-2,%);
  ASSERT_ALWAYS(7ul/mpz_class(1e35)==0);
  ASSERT_ALWAYS(7ul%mpz_class(1e35)==7);
  ASSERT_ALWAYS(7ul/mpz_class(-1e35)==0);
  ASSERT_ALWAYS(7ul%mpz_class(-1e35)==7);
  CHECK_ALL_SIGNS_COMPARISONS(mpz_class,11,3);
  CHECK_ALL(mpz_class,6,3,&);
  CHECK_ALL(mpz_class,6,3,|);
  CHECK_ALL(mpz_class,6,3,^);
  CHECK(mpz_class,unsigned long,6,2,<<);
  CHECK(mpz_class,unsigned long,6,2,>>);
  CHECK(mpz_class,unsigned long,-13,2,<<);
  CHECK(mpz_class,unsigned long,-13,2,>>);
  ASSERT_ALWAYS(++mpz_class(7)==8);
  ASSERT_ALWAYS(++mpz_class(-8)==-7);
  ASSERT_ALWAYS(--mpz_class(8)==7);
  ASSERT_ALWAYS(--mpz_class(-7)==-8);
  ASSERT_ALWAYS(~mpz_class(7)==-8);
  ASSERT_ALWAYS(~mpz_class(-8)==7);
  ASSERT_ALWAYS(+mpz_class(7)==7);
  ASSERT_ALWAYS(+mpz_class(-8)==-8);
  ASSERT_ALWAYS(-mpz_class(7)==-7);
  ASSERT_ALWAYS(-mpz_class(-8)==8);
  ASSERT_ALWAYS(abs(mpz_class(7))==7);
  ASSERT_ALWAYS(abs(mpz_class(-8))==8);
  ASSERT_ALWAYS(sqrt(mpz_class(7))==2);
  ASSERT_ALWAYS(sqrt(mpz_class(0))==0);
  ASSERT_ALWAYS(sgn(mpz_class(0))==0);
  ASSERT_ALWAYS(sgn(mpz_class(9))==1);
  ASSERT_ALWAYS(sgn(mpz_class(-17))==-1);
  ASSERT_ALWAYS(mpz_class(1)+DBL_MAX>2);
  ASSERT_ALWAYS(mpz_class(1)+DBL_MIN<2);
  ASSERT_ALWAYS(mpz_class(1)+std::numeric_limits<double>::denorm_min()<2);
  ASSERT_ALWAYS(gcd(mpz_class(6),mpz_class(8))==2);
  ASSERT_ALWAYS(gcd(-mpz_class(6),mpz_class(8))==2);
  ASSERT_ALWAYS(gcd(-mpz_class(6),-mpz_class(8))==2);
  ASSERT_ALWAYS(gcd(mpz_class(6),8.f)==2);
  ASSERT_ALWAYS(gcd(-mpz_class(6),static_cast<unsigned char>(8))==2);
  ASSERT_ALWAYS(gcd(static_cast<long>(-6),mpz_class(5)+3)==2);
  ASSERT_ALWAYS(lcm(mpz_class(6),mpz_class(8))==24);
  ASSERT_ALWAYS(lcm(-mpz_class(6),mpz_class(8))==24);
  ASSERT_ALWAYS(lcm(-mpz_class(6),-mpz_class(8))==24);
  ASSERT_ALWAYS(lcm(mpz_class(6),static_cast<short>(8))==24);
  ASSERT_ALWAYS(lcm(-mpz_class(6),static_cast<unsigned char>(8))==24);
  ASSERT_ALWAYS(lcm(-6.,mpz_class(5)+3)==24);
}

template<class T>
void checkqf (){
  CHECK_ALL(T,5.,2,+); CHECK_MPZ(T,5.,2,+);
  CHECK_ALL(T,5.,2,-); CHECK_MPZ(T,5.,2,-);
  CHECK_ALL(T,5.,2,*); CHECK_MPZ(T,5.,2,*);
  CHECK_ALL(T,5.,2,/); CHECK_MPZ(T,5.,2,/);
  CHECK_ALL(T,0.,2,/);
  CHECK_ALL_SIGNS(T,11.,3,+);
  CHECK_ALL_SIGNS(T,11.,3,-);
  CHECK_ALL_SIGNS(T,11.,3,*);
  CHECK_ALL_SIGNS(T,11.,4,/);
  CHECK_SI(T,LONG_MIN,1,*);
  CHECK_SI(T,0,3,*);
  CHECK_ALL_COMPARISONS(T,5.,2);
  CHECK_ALL_SIGNS_COMPARISONS(T,11.,3);
  CHECK_MPZ(T,5,-2,<);
  CHECK_MPZ(T,5,-2,>);
  CHECK_MPZ(T,5,-2,<=);
  CHECK_MPZ(T,5,-2,>=);
  CHECK_MPZ(T,5,-2,==);
  CHECK_MPZ(T,5,-2,!=);
  CHECK_MPZ(T,0,0,<);
  CHECK_MPZ(T,0,0,>);
  CHECK_MPZ(T,0,0,<=);
  CHECK_MPZ(T,0,0,>=);
  CHECK_MPZ(T,0,0,==);
  CHECK_MPZ(T,0,0,!=);
  ASSERT_ALWAYS(T(6)<<2==6.*4);
  ASSERT_ALWAYS(T(6)>>2==6./4);
  ASSERT_ALWAYS(T(-13)<<2==-13.*4);
  ASSERT_ALWAYS(T(-13)>>2==-13./4);
  ASSERT_ALWAYS(++T(7)==8);
  ASSERT_ALWAYS(++T(-8)==-7);
  ASSERT_ALWAYS(--T(8)==7);
  ASSERT_ALWAYS(--T(-7)==-8);
  ASSERT_ALWAYS(+T(7)==7);
  ASSERT_ALWAYS(+T(-8)==-8);
  ASSERT_ALWAYS(-T(7)==-7);
  ASSERT_ALWAYS(-T(-8)==8);
  ASSERT_ALWAYS(abs(T(7))==7);
  ASSERT_ALWAYS(abs(T(-8))==8);
  ASSERT_ALWAYS(sgn(T(0))==0);
  ASSERT_ALWAYS(sgn(T(9))==1);
  ASSERT_ALWAYS(sgn(T(-17))==-1);
  ASSERT_ALWAYS(T(1)+DBL_MAX>2);
  ASSERT_ALWAYS(T(1)+DBL_MIN>1);
  ASSERT_ALWAYS(T(1)+DBL_MIN<1.001);
  ASSERT_ALWAYS(T(1)+std::numeric_limits<double>::denorm_min()>1);
  ASSERT_ALWAYS(T(1)+std::numeric_limits<double>::denorm_min()<1.001);
}

void checkf (){
  ASSERT_ALWAYS(sqrt(mpf_class(7))>2.64);
  ASSERT_ALWAYS(sqrt(mpf_class(7))<2.65);
  ASSERT_ALWAYS(sqrt(mpf_class(0))==0);
  // TODO: add some consistency checks, as described in
  // https://gmplib.org/list-archives/gmp-bugs/2013-February/002940.html
  CHECK1(mpf_class,1.9,trunc);
  CHECK1(mpf_class,1.9,floor);
  CHECK1(mpf_class,1.9,ceil);
  CHECK1(mpf_class,4.3,trunc);
  CHECK1(mpf_class,4.3,floor);
  CHECK1(mpf_class,4.3,ceil);
  CHECK1(mpf_class,-7.1,trunc);
  CHECK1(mpf_class,-7.1,floor);
  CHECK1(mpf_class,-7.1,ceil);
  CHECK1(mpf_class,-2.8,trunc);
  CHECK1(mpf_class,-2.8,floor);
  CHECK1(mpf_class,-2.8,ceil);
  CHECK1(mpf_class,-1.5,trunc);
  CHECK1(mpf_class,-1.5,floor);
  CHECK1(mpf_class,-1.5,ceil);
  CHECK1(mpf_class,2.5,trunc);
  CHECK1(mpf_class,2.5,floor);
  CHECK1(mpf_class,2.5,ceil);
  ASSERT_ALWAYS(hypot(mpf_class(-3),mpf_class(4))>4.9);
  ASSERT_ALWAYS(hypot(mpf_class(-3),mpf_class(4))<5.1);
  ASSERT_ALWAYS(hypot(mpf_class(-3),4.)>4.9);
  ASSERT_ALWAYS(hypot(-3.,mpf_class(4))<5.1);
  ASSERT_ALWAYS(hypot(mpf_class(-3),4l)>4.9);
  ASSERT_ALWAYS(hypot(-3l,mpf_class(4))<5.1);
  ASSERT_ALWAYS(hypot(mpf_class(-3),4ul)>4.9);
  ASSERT_ALWAYS(hypot(3ul,mpf_class(4))<5.1);
  CHECK(mpf_class,mpq_class,1.5,2.25,+);
  CHECK(mpf_class,mpq_class,1.5,2.25,-);
  CHECK(mpf_class,mpq_class,1.5,-2.25,*);
  CHECK(mpf_class,mpq_class,1.5,-2,/);
  CHECK_MPQ(mpf_class,-5.5,-2.25,+);
  CHECK_MPQ(mpf_class,-5.5,-2.25,-);
  CHECK_MPQ(mpf_class,-5.5,-2.25,*);
  CHECK_MPQ(mpf_class,-5.25,-0.5,/);
  CHECK_MPQ(mpf_class,5,-2,<);
  CHECK_MPQ(mpf_class,5,-2,>);
  CHECK_MPQ(mpf_class,5,-2,<=);
  CHECK_MPQ(mpf_class,5,-2,>=);
  CHECK_MPQ(mpf_class,5,-2,==);
  CHECK_MPQ(mpf_class,5,-2,!=);
  CHECK_MPQ(mpf_class,0,0,<);
  CHECK_MPQ(mpf_class,0,0,>);
  CHECK_MPQ(mpf_class,0,0,<=);
  CHECK_MPQ(mpf_class,0,0,>=);
  CHECK_MPQ(mpf_class,0,0,==);
  CHECK_MPQ(mpf_class,0,0,!=);
}

int
main (void)
{
  tests_start();

  // Enough precision for 1 + denorm_min
  mpf_set_default_prec(DBL_MANT_DIG-DBL_MIN_EXP+42);
  checkz();
  checkqf<mpq_class>();
  checkqf<mpf_class>();
  checkf();

  tests_end();
  return 0;
}
