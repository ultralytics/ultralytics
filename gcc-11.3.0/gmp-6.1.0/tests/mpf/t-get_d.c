/* Test mpf_get_d and mpf_set_d.

Copyright 1996, 1999-2001, 2009 Free Software Foundation, Inc.

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
#include "tests.h"

#if defined (__vax) || defined (__vax__)
#define LOW_BOUND 1e-38
#define HIGH_BOUND 8e37
#endif

#if defined (_CRAY) && ! defined (_CRAYIEEE)
/* The range varies mysteriously between Cray version.  On an SV1,
   the range seem to be 1e-600..1e603, but a cfp (non-ieee) T90
   has a much smaller range of 1e-240..1e240.  */
#define LOW_BOUND 1e-240
#define HIGH_BOUND 1e240
#endif

#if ! defined (LOW_BOUND)
#define LOW_BOUND 1e-300
#define HIGH_BOUND 1e300
#endif

void
test_denorms (int prc)
{
#ifdef _GMP_IEEE_FLOATS
  double d1, d2;
  mpf_t f;
  int i;

  mpf_set_default_prec (prc);

  mpf_init (f);

  d1 = 1.9;
  for (i = 0; i < 820; i++)
    {
      mpf_set_d (f, d1);
      d2 = mpf_get_d (f);
      if (d1 != d2)
        abort ();
      d1 *= 0.4;
    }

  mpf_clear (f);
#endif
}

int
main (int argc, char **argv)
{
  double d, e, r;
  mpf_t u, v;

  tests_start ();
  mpf_init (u);
  mpf_init (v);

  mpf_set_d (u, LOW_BOUND);
  for (d = 2.0 * LOW_BOUND; d < HIGH_BOUND; d *= 1.01)
    {
      mpf_set_d (v, d);
      if (mpf_cmp (u, v) >= 0)
	abort ();
      e = mpf_get_d (v);
      r = e/d;
      if (r < 0.99999999999999 || r > 1.00000000000001)
	{
	  fprintf (stderr, "should be one ulp from 1: %.16f\n", r);
	  abort ();
	}
      mpf_set (u, v);
    }

  mpf_clear (u);
  mpf_clear (v);

  test_denorms (10);
  test_denorms (32);
  test_denorms (64);
  test_denorms (100);
  test_denorms (200);

  tests_end ();
  exit (0);
}
