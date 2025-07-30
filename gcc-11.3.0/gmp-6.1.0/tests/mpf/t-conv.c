/* Test mpf_get_str and mpf_set_str.

Copyright 1996, 2000, 2001, 2008 Free Software Foundation, Inc.

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
#include <string.h> /* for strlen */

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

#ifndef SIZE
#define SIZE 10
#endif

#ifndef EXPO
#define EXPO 200
#endif

int
main (int argc, char **argv)
{
  mpf_t x, y;
  int reps = 20000;
  int i;
  mp_size_t bprec = 100;
  mpf_t d, rerr, max_rerr, limit_rerr;
  char *str;
  mp_exp_t bexp;
  long size, exp;
  int base;
  char buf[SIZE * GMP_LIMB_BITS + 5];

  tests_start ();

  if (argc > 1)
    {
      reps = strtol (argv[1], 0, 0);
      if (argc > 2)
	bprec = strtol (argv[2], 0, 0);
    }

  mpf_set_default_prec (bprec);

  mpf_init_set_ui (limit_rerr, 1);
  mpf_div_2exp (limit_rerr, limit_rerr, bprec);
#if VERBOSE
  mpf_dump (limit_rerr);
#endif
  mpf_init (rerr);
  mpf_init_set_ui (max_rerr, 0);

  mpf_init (x);
  mpf_init (y);
  mpf_init (d);

  /* First test some specific values.  */

  mpf_set_str (y, "1.23456e1000", 0);

  mpf_set_str (x, "1.23456e1000", 10);
  if (mpf_cmp (x, y) != 0)
    abort ();
  mpf_set_str (x, "1.23456e+1000", 0);
  if (mpf_cmp (x, y) != 0)
    abort ();
  mpf_set_str (x, "1.23456e+1000", 10);
  if (mpf_cmp (x, y) != 0)
    abort ();

  /* Now test random values.  */

  for (i = 0; i < reps; i++)
    {
      if (i == 0)
        {
          /* exercise the special case in get_str for for x==0 */
          mpf_set_ui (x, 0L);
          base = 10;
        }
      else
        {
          size = urandom () % (2 * SIZE) - SIZE;
          exp = urandom () % EXPO;
          mpf_random2 (x, size, exp);
          base = urandom () % 61 + 2;
        }

      str = mpf_get_str (0, &bexp, base, 0, x);

      if (str[0] == '-')
	sprintf (buf, "-0.%s@%ld", str + 1, bexp);
      else
	sprintf (buf, "0.%s@%ld", str, bexp);

      mpf_set_str_or_abort (y, buf, -base);
      (*__gmp_free_func) (str, strlen (str) + 1);

      mpf_reldiff (rerr, x, y);
      if (mpf_cmp (rerr, max_rerr) > 0)
	{
	  mpf_set (max_rerr, rerr);
#if VERBOSE
	  mpf_dump (max_rerr);
#endif
	  if (mpf_cmp (rerr, limit_rerr) > 0)
	    {
	      printf ("ERROR after %d tests\n", i);
	      printf ("base = %d\n", base);
	      printf ("   x = "); mpf_dump (x);
	      printf ("   y = "); mpf_dump (y);
	      abort ();
	    }
	}
    }

  mpf_clear (limit_rerr);
  mpf_clear (rerr);
  mpf_clear (max_rerr);

  mpf_clear (x);
  mpf_clear (y);
  mpf_clear (d);

  tests_end ();
  exit (0);
}
