/* Test stream formatted input and output on mp*_class

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

#include <sstream>

#include "gmp.h"
#include "gmpxx.h"
#include "gmp-impl.h"
#include "tests.h"

using namespace std;

// The tests are extremely basic. These functions just forward to the
// ones tested in t-istream.cc and t-ostream.cc; we rely on those for
// advanced tests and only check the syntax here.

void
checki ()
{
  {
    istringstream i("123");
    mpz_class x;
    i >> x;
    ASSERT_ALWAYS (x == 123);
  }
  {
    istringstream i("3/4");
    mpq_class x;
    i >> x;
    ASSERT_ALWAYS (x == .75);
  }
  {
    istringstream i("1.5");
    mpf_class x;
    i >> x;
    ASSERT_ALWAYS (x == 1.5);
  }
}

void
checko ()
{
  {
    ostringstream o;
    mpz_class x=123;
    o << x;
    ASSERT_ALWAYS (o.str() == "123");
  }
  {
    ostringstream o;
    mpz_class x=123;
    o << (x+1);
    ASSERT_ALWAYS (o.str() == "124");
  }
  {
    ostringstream o;
    mpq_class x(3,4);
    o << x;
    ASSERT_ALWAYS (o.str() == "3/4");
  }
  {
    ostringstream o;
    mpq_class x(3,4);
    o << (x+1);
    ASSERT_ALWAYS (o.str() == "7/4");
  }
  {
    ostringstream o;
    mpf_class x=1.5;
    o << x;
    ASSERT_ALWAYS (o.str() == "1.5");
  }
  {
    ostringstream o;
    mpf_class x=1.5;
    o << (x+1);
    ASSERT_ALWAYS (o.str() == "2.5");
  }
}

int
main (int argc, char *argv[])
{
  tests_start ();

  checki ();
  checko ();

  tests_end ();
  return 0;
}
