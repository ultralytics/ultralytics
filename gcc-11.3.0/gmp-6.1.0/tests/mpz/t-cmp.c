/* Test mpz_cmp and mpz_cmpabs.

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


/* Nothing sophisticated here, just exercise some combinations of sizes and
   signs.  */


void
check_one (mpz_ptr x, mpz_ptr y, int want_cmp, int want_cmpabs)
{
  int  got;

  got = mpz_cmp (x, y);
  if ((   got <  0) != (want_cmp <  0)
      || (got == 0) != (want_cmp == 0)
      || (got >  0) != (want_cmp >  0))
    {
      printf ("mpz_cmp got %d want %d\n", got, want_cmp);
      mpz_trace ("x", x);
      mpz_trace ("y", y);
      abort ();
    }

  got = mpz_cmpabs (x, y);
  if ((   got <  0) != (want_cmpabs <  0)
      || (got == 0) != (want_cmpabs == 0)
      || (got >  0) != (want_cmpabs >  0))
    {
      printf ("mpz_cmpabs got %d want %d\n", got, want_cmpabs);
      mpz_trace ("x", x);
      mpz_trace ("y", y);
      abort ();
    }
}


void
check_all (mpz_ptr x, mpz_ptr y, int want_cmp, int want_cmpabs)
{
  check_one (x, y,  want_cmp,  want_cmpabs);
  check_one (y, x, -want_cmp, -want_cmpabs);

  mpz_neg (x, x);
  mpz_neg (y, y);
  want_cmp = -want_cmp;

  check_one (x, y,  want_cmp,  want_cmpabs);
  check_one (y, x, -want_cmp, -want_cmpabs);
}


#define SET1(z,size, n) \
  SIZ(z) = size; PTR(z)[0] = n

#define SET2(z,size, n1,n0) \
  SIZ(z) = size; PTR(z)[1] = n1; PTR(z)[0] = n0

#define SET4(z,size, n3,n2,n1,n0) \
  SIZ(z) = size; PTR(z)[3] = n3; PTR(z)[2] = n2; PTR(z)[1] = n1; PTR(z)[0] = n0

void
check_various (void)
{
  mpz_t  x, y;

  mpz_init (x);
  mpz_init (y);

  mpz_realloc (x, (mp_size_t) 20);
  mpz_realloc (y, (mp_size_t) 20);

  /* 0 cmp 0, junk in low limbs */
  SET1 (x,0, 123);
  SET1 (y,0, 456);
  check_all (x, y, 0, 0);


  /* 123 cmp 0 */
  SET1 (x,1, 123);
  SET1 (y,0, 456);
  check_all (x, y, 1, 1);

  /* 123:456 cmp 0 */
  SET2 (x,2, 456,123);
  SET1 (y,0, 9999);
  check_all (x, y, 1, 1);


  /* 123 cmp 123 */
  SET1(x,1, 123);
  SET1(y,1, 123);
  check_all (x, y, 0, 0);

  /* -123 cmp 123 */
  SET1(x,-1, 123);
  SET1(y,1,  123);
  check_all (x, y, -1, 0);


  /* 123 cmp 456 */
  SET1(x,1, 123);
  SET1(y,1, 456);
  check_all (x, y, -1, -1);

  /* -123 cmp 456 */
  SET1(x,-1, 123);
  SET1(y,1,  456);
  check_all (x, y, -1, -1);

  /* 123 cmp -456 */
  SET1(x,1,  123);
  SET1(y,-1, 456);
  check_all (x, y, 1, -1);


  /* 1:0 cmp 1:0 */
  SET2 (x,2, 1,0);
  SET2 (y,2, 1,0);
  check_all (x, y, 0, 0);

  /* -1:0 cmp 1:0 */
  SET2 (x,-2, 1,0);
  SET2 (y,2,  1,0);
  check_all (x, y, -1, 0);


  /* 2:0 cmp 1:0 */
  SET2 (x,2, 2,0);
  SET2 (y,2, 1,0);
  check_all (x, y, 1, 1);


  /* 4:3:2:1 cmp 2:1 */
  SET4 (x,4, 4,3,2,1);
  SET2 (y,2, 2,1);
  check_all (x, y, 1, 1);

  /* -4:3:2:1 cmp 2:1 */
  SET4 (x,-4, 4,3,2,1);
  SET2 (y,2,  2,1);
  check_all (x, y, -1, 1);


  mpz_clear (x);
  mpz_clear (y);
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
