/* Test g++ -Wold-style-cast cleanliness.

Copyright 2003 Free Software Foundation, Inc.

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
#include "gmpxx.h"


/* This code doesn't do anything when run, it just expands various C macros
   to see that they don't trigger compile-time warnings from g++
   -Wold-style-cast.  This option isn't used in a normal build, it has to be
   added manually to make this test worthwhile.  */

void
check_macros (void)
{
  mpz_t          z;
  long           l = 123;
  unsigned long  u = 456;
  int            i;
  mp_limb_t      limb;

  mpz_init_set_ui (z, 0L);
  i = mpz_odd_p (z);
  i = mpz_even_p (z);
  i = mpz_cmp_si (z, l);
  i = mpz_cmp_ui (z, u);
  mpz_clear (z);

  limb = GMP_NUMB_MASK;
  limb = GMP_NUMB_MAX;
  limb = GMP_NAIL_MASK;

  mpn_divmod (&limb, &limb, 1, &limb, 1);
  mpn_divexact_by3 (&limb, &limb, 1);
}

int
main (void)
{
  return 0;
}
