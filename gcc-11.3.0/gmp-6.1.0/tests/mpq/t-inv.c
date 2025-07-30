/* Test mpq_inv (and set/get_num/den).

Copyright 2012 Free Software Foundation, Inc.

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

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

int
main (int argc, char **argv)
{
  mpq_t a, b;
  mpz_t m, n;
  const char* s = "-420000000000000000000000";

  tests_start ();

  mpq_inits (a, b, (mpq_ptr)0);
  mpz_inits (m, n, (mpz_ptr)0);

  mpz_set_ui (m, 13);
  mpq_set_den (a, m);
  mpz_set_str (m, s, 0);
  mpq_set_num (a, m);
  MPQ_CHECK_FORMAT (a);
  mpq_inv (b, a);
  MPQ_CHECK_FORMAT (b);
  mpq_get_num (n, b);
  ASSERT_ALWAYS (mpz_cmp_si (n, -13) == 0);
  mpq_neg (b, b);
  mpq_inv (a, b);
  MPQ_CHECK_FORMAT (a);
  mpq_inv (b, b);
  MPQ_CHECK_FORMAT (b);
  mpq_get_den (n, b);
  ASSERT_ALWAYS (mpz_cmp_ui (n, 13) == 0);
  mpq_get_num (n, a);
  mpz_add (n, n, m);
  ASSERT_ALWAYS (mpz_sgn (n) == 0);

  mpq_clears (a, b, (mpq_ptr)0);
  mpz_clears (m, n, (mpz_ptr)0);

  tests_end ();
  return 0;
}
