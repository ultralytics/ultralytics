/* Test mpq_equal.

Copyright 2001 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <stdlib.h>
#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


void
check_one (mpq_srcptr x, mpq_srcptr y, int want)
{
  int  got;

  MPQ_CHECK_FORMAT (x);
  MPQ_CHECK_FORMAT (y);

  got = mpq_equal (x, y);
  if ((got != 0) != (want != 0))
    {
      printf ("mpq_equal got %d want %d\n", got, want);
      mpq_trace ("x", x);
      mpq_trace ("y", y);
      abort ();
    }
}


void
check_all (mpq_ptr x, mpq_ptr y, int want)
{
  check_one (x, y, want);
  check_one (y, x, want);

  mpq_neg (x, x);
  mpq_neg (y, y);

  check_one (x, y, want);
  check_one (y, x, want);
}


#define SET4Z(z, size,l3,l2,l1,l0) \
  SIZ(z) = size; PTR(z)[3] = l3; PTR(z)[2] = l2; PTR(z)[1] = l1; PTR(z)[0] = l0

#define SET4(q, nsize,n3,n2,n1,n0, dsize,d3,d2,d1,d0)   \
  SET4Z (mpq_numref(q), nsize,n3,n2,n1,n0);             \
  SET4Z (mpq_denref(q), dsize,d3,d2,d1,d0)


/* Exercise various combinations of same and slightly different values. */

void
check_various (void)
{
  mpq_t  x, y;

  mpq_init (x);
  mpq_init (y);

  mpz_realloc (mpq_numref(x), (mp_size_t) 20);
  mpz_realloc (mpq_denref(x), (mp_size_t) 20);
  mpz_realloc (mpq_numref(y), (mp_size_t) 20);
  mpz_realloc (mpq_denref(y), (mp_size_t) 20);

  /* 0 == 0 */
  SET4 (x, 0,13,12,11,10, 1,23,22,21,1);
  SET4 (y, 0,33,32,31,30, 1,43,42,41,1);
  check_all (x, y, 1);

  /* 83/99 == 83/99 */
  SET4 (x, 1,13,12,11,83, 1,23,22,21,99);
  SET4 (y, 1,33,32,31,83, 1,43,42,41,99);
  check_all (x, y, 1);

  /* 1:2:3:4/5:6:7 == 1:2:3:4/5:6:7 */
  SET4 (x, 4,1,2,3,4, 3,88,5,6,7);
  SET4 (y, 4,1,2,3,4, 3,99,5,6,7);
  check_all (x, y, 1);

  /* various individual changes making != */
  SET4 (x, 4,1,2,3,667, 3,88,5,6,7);
  SET4 (y, 4,1,2,3,4, 3,99,5,6,7);
  check_all (x, y, 0);
  SET4 (x, 4,1,2,666,4, 3,88,5,6,7);
  SET4 (y, 4,1,2,3,4, 3,99,5,6,7);
  check_all (x, y, 0);
  SET4 (x, 4,1,666,3,4, 3,88,5,6,7);
  SET4 (y, 4,1,2,3,4, 3,99,5,6,7);
  check_all (x, y, 0);
#if GMP_NUMB_BITS != 62
  SET4 (x, 4,667,2,3,4, 3,88,5,6,7);
  SET4 (y, 4,1,2,3,4, 3,99,5,6,7);
  check_all (x, y, 0);
#endif
  SET4 (x, 4,1,2,3,4, 3,88,5,6,667);
  SET4 (y, 4,1,2,3,4, 3,99,5,6,7);
  check_all (x, y, 0);
  SET4 (x, 4,1,2,3,4, 3,88,5,667,7);
  SET4 (y, 4,1,2,3,4, 3,99,5,6,7);
  check_all (x, y, 0);
  SET4 (x, 4,1,2,3,4, 3,88,666,6,7);
  SET4 (y, 4,1,2,3,4, 3,99,5,6,7);
  check_all (x, y, 0);
  SET4 (x, -4,1,2,3,4, 3,88,5,6,7);
  SET4 (y,  4,1,2,3,4, 3,99,5,6,7);
  check_all (x, y, 0);
  SET4 (x, 1,1,2,3,4, 3,88,5,6,7);
  SET4 (y, 4,1,2,3,4, 3,99,5,6,7);
  check_all (x, y, 0);
  SET4 (x, 4,1,2,3,4, 3,88,5,6,7);
  SET4 (y, 4,1,2,3,4, 2,99,5,6,7);
  check_all (x, y, 0);

  mpq_clear (x);
  mpq_clear (y);
}


int
main (void)
{
  tests_start ();
  mp_trace_base = -16;

  check_various ();

  tests_end ();
  exit (0);
}
