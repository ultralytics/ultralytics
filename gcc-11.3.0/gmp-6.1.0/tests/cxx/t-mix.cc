/* Test legality of conversion between the different mp*_class

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

int f_z  (mpz_class){return 0;}
int f_q  (mpq_class){return 1;}
int f_f  (mpf_class){return 2;}
int f_zq (mpz_class){return 0;}
int f_zq (mpq_class){return 1;}
int f_zf (mpz_class){return 0;}
int f_zf (mpf_class){return 2;}
int f_qf (mpq_class){return 1;}
int f_qf (mpf_class){return 2;}
int f_zqf(mpz_class){return 0;}
int f_zqf(mpq_class){return 1;}
int f_zqf(mpf_class){return 2;}

void
check (void)
{
  mpz_class z=42;
  mpq_class q=33;
  mpf_class f=18;

  ASSERT_ALWAYS(f_z  (z)==0); ASSERT_ALWAYS(f_z  (-z)==0);
  ASSERT_ALWAYS(f_q  (z)==1); ASSERT_ALWAYS(f_q  (-z)==1);
  ASSERT_ALWAYS(f_q  (q)==1); ASSERT_ALWAYS(f_q  (-q)==1);
  ASSERT_ALWAYS(f_f  (z)==2); ASSERT_ALWAYS(f_f  (-z)==2);
  ASSERT_ALWAYS(f_f  (q)==2); ASSERT_ALWAYS(f_f  (-q)==2);
  ASSERT_ALWAYS(f_f  (f)==2); ASSERT_ALWAYS(f_f  (-f)==2);
  ASSERT_ALWAYS(f_zq (z)==0);
  ASSERT_ALWAYS(f_zq (q)==1); ASSERT_ALWAYS(f_zq (-q)==1);
  ASSERT_ALWAYS(f_zf (z)==0);
  ASSERT_ALWAYS(f_zf (f)==2); ASSERT_ALWAYS(f_zf (-f)==2);
  ASSERT_ALWAYS(f_qf (q)==1);
  ASSERT_ALWAYS(f_qf (f)==2); ASSERT_ALWAYS(f_qf (-f)==2);
  ASSERT_ALWAYS(f_zqf(z)==0);
  ASSERT_ALWAYS(f_zqf(q)==1);
  ASSERT_ALWAYS(f_zqf(f)==2); ASSERT_ALWAYS(f_zqf(-f)==2);

  ASSERT_ALWAYS(f_zqf(mpz_class(z))==0); ASSERT_ALWAYS(f_zqf(mpz_class(-z))==0);
  ASSERT_ALWAYS(f_zqf(mpz_class(q))==0); ASSERT_ALWAYS(f_zqf(mpz_class(-q))==0);
  ASSERT_ALWAYS(f_zqf(mpz_class(f))==0); ASSERT_ALWAYS(f_zqf(mpz_class(-f))==0);
  ASSERT_ALWAYS(f_zqf(mpq_class(z))==1); ASSERT_ALWAYS(f_zqf(mpq_class(-z))==1);
  ASSERT_ALWAYS(f_zqf(mpq_class(q))==1); ASSERT_ALWAYS(f_zqf(mpq_class(-q))==1);
  ASSERT_ALWAYS(f_zqf(mpq_class(f))==1); ASSERT_ALWAYS(f_zqf(mpq_class(-f))==1);
  ASSERT_ALWAYS(f_zqf(mpf_class(z))==2); ASSERT_ALWAYS(f_zqf(mpf_class(-z))==2);
  ASSERT_ALWAYS(f_zqf(mpf_class(q))==2); ASSERT_ALWAYS(f_zqf(mpf_class(-q))==2);
  ASSERT_ALWAYS(f_zqf(mpf_class(f))==2); ASSERT_ALWAYS(f_zqf(mpf_class(-f))==2);
}

int
main (void)
{
  tests_start();

  check();

  tests_end();
  return 0;
}
