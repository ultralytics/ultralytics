/* Test mpf_pow_ui

Copyright 2015 Free Software Foundation, Inc.

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
check_data (void)
{
  unsigned int b, e;
  mpf_t b1, r, r2, limit;

  mpf_inits (b1, r, r2, NULL);
  mpf_init_set_ui (limit, 1);
  mpf_mul_2exp (limit, limit, MAX (GMP_NUMB_BITS, 53)); 

  /* This test just test integers with results that fit in a single
     limb or 53 bits.  This avoids any rounding.  */

  for (b = 0; b <= 400; b++)
    {
      mpf_set_ui (b1, b);
      mpf_set_ui (r2, 1);
      for (e = 0; e <= GMP_LIMB_BITS; e++)
	{
	  mpf_pow_ui (r, b1, e);

	  if (mpf_cmp (r, r2))
	    abort ();

	  mpf_mul_ui (r2, r2, b);

	  if (mpf_cmp (r2, limit) >= 0)
	    break;
	}
    }

  mpf_clears (b1, r, r2, limit, NULL);
}

int
main (int argc, char **argv)
{
  tests_start ();

  check_data ();

  tests_end ();
  exit (0);
}
