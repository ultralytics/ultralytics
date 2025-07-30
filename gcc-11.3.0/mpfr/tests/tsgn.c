/* tsgn -- Test for the sign of a floating point number.

Copyright 2003, 2006-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include <stdio.h>
#include <stdlib.h>

#include "mpfr-test.h"

static void
check_special (void)
{
  mpfr_t x;
  int ret = 0;

  mpfr_init (x);
  MPFR_SET_ZERO (x);
  if ((mpfr_sgn) (x) != 0 || mpfr_sgn (x) != 0)
    {
      printf("Sgn error for 0.\n");
      ret = 1;
    }
  MPFR_SET_INF (x);
  MPFR_SET_POS (x);
  if ((mpfr_sgn) (x) != 1 || mpfr_sgn (x) != 1)
    {
      printf("Sgn error for +Inf.\n");
      ret = 1;
    }
  MPFR_SET_INF (x);
  MPFR_SET_NEG (x);
  if ((mpfr_sgn) (x) != -1 || mpfr_sgn (x) != -1)
    {
      printf("Sgn error for -Inf.\n");
      ret = 1;
    }
  MPFR_SET_NAN (x);
  mpfr_clear_flags ();
  if ((mpfr_sgn) (x) != 0 || !mpfr_erangeflag_p ())
    {
      printf("Sgn error for NaN.\n");
      ret = 1;
    }
  mpfr_clear_flags ();
  if (mpfr_sgn (x) != 0 || !mpfr_erangeflag_p ())
    {
      printf("Sgn error for NaN.\n");
      ret = 1;
    }
  mpfr_clear (x);
  if (ret)
    exit (ret);
}

static void
check_sgn(void)
{
  mpfr_t x;
  int i, s1, s2;

  mpfr_init(x);
  for(i = 0 ; i < 100 ; i++)
    {
      mpfr_urandomb (x, RANDS);
      if (i&1)
        {
          MPFR_SET_POS(x);
          s2 = 1;
        }
      else
        {
          MPFR_SET_NEG(x);
          s2 = -1;
        }
      s1 = mpfr_sgn(x);
      if (s1 < -1 || s1 > 1)
        {
          printf("Error for sgn: out of range.\n");
          goto lexit;
        }
      else if (MPFR_IS_NAN(x) || MPFR_IS_ZERO(x))
        {
          if (s1 != 0)
            {
              printf("Error for sgn: Nan or Zero should return 0.\n");
              goto lexit;
            }
        }
      else if (s1 != s2)
        {
          printf("Error for sgn. Return %d instead of %d.\n", s1, s2);
          goto lexit;
        }
    }
  mpfr_clear(x);
  return;

 lexit:
  mpfr_clear(x);
  exit(1);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  check_special ();
  check_sgn ();

  tests_end_mpfr ();
  return 0;
}
