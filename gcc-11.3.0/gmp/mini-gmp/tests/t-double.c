/*

Copyright 2012, 2013 Free Software Foundation, Inc.

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

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "testutils.h"

#define GMP_LIMB_BITS (sizeof(mp_limb_t) * CHAR_BIT)

#define COUNT 10000

static const struct
{
  double d;
  const char *s;
} values[] = {
  { 0.0, "0" },
  { 0.3, "0" },
  { -0.3, "0" },
  { M_PI, "3" },
  { M_PI*1e15, "b29430a256d21" },
  { -M_PI*1e15, "-b29430a256d21" },
  /* 17 * 2^{200} =
     27317946752402834684213355569799764242877450894307478200123392 */
  {0.2731794675240283468421335556979976424288e62,
    "1100000000000000000000000000000000000000000000000000" },
  { 0.0, NULL }
};

void
testmain (int argc, char **argv)
{
  unsigned i;
  mpz_t x;

  for (i = 0; values[i].s; i++)
    {
      char *s;
      mpz_init_set_d (x, values[i].d);
      s = mpz_get_str (NULL, 16, x);
      if (strcmp (s, values[i].s) != 0)
	{
	  fprintf (stderr, "mpz_set_d failed:\n"
		   "d = %.20g\n"
		   "s = %s\n"
		   "r = %s\n",
		   values[i].d, s, values[i].s);
	  abort ();
	}
      testfree (s);
      mpz_clear (x);
    }

  mpz_init (x);

  for (i = 0; i < COUNT; i++)
    {
      /* Use volatile, to avoid extended precision in floating point
	 registers, e.g., on m68k and 80387. */
      volatile double d, f;
      unsigned long m;
      int e;

      mini_rrandomb (x, GMP_LIMB_BITS);
      m = mpz_get_ui (x);
      mini_urandomb (x, 8);
      e = mpz_get_ui (x) - 100;

      d = ldexp ((double) m, e);
      mpz_set_d (x, d);
      f = mpz_get_d (x);
      if (f != floor (d))
	{
	  fprintf (stderr, "mpz_set_d/mpz_get_d failed:\n");
	  goto dumperror;
	}
      if ((f == d) ? (mpz_cmp_d (x, d) != 0) : (mpz_cmp_d (x, d) >= 0))
	{
	  fprintf (stderr, "mpz_cmp_d (x, d) failed:\n");
	  goto dumperror;
	}
      f = d + 1.0;
      if (f > d && ! (mpz_cmp_d (x, f) < 0))
	{
	  fprintf (stderr, "mpz_cmp_d (x, f) failed:\n");
	  goto dumperror;
	}

      d = - d;

      mpz_set_d (x, d);
      f = mpz_get_d (x);
      if (f != ceil (d))
	{
	  fprintf (stderr, "mpz_set_d/mpz_get_d failed:\n");
	dumperror:
	  dump ("x", x);
	  fprintf (stderr, "m = %lx, e = %i\n", m, e);
	  fprintf (stderr, "d = %.15g\n", d);
	  fprintf (stderr, "f = %.15g\n", f);
	  fprintf (stderr, "f - d = %.5g\n", f - d);
	  abort ();
	}
      if ((f == d) ? (mpz_cmp_d (x, d) != 0) : (mpz_cmp_d (x, d) <= 0))
	{
	  fprintf (stderr, "mpz_cmp_d (x, d) failed:\n");
	  goto dumperror;
	}
      f = d - 1.0;
      if (f < d && ! (mpz_cmp_d (x, f) > 0))
	{
	  fprintf (stderr, "mpz_cmp_d (x, f) failed:\n");
	  goto dumperror;
	}
    }

  mpz_clear (x);
}
