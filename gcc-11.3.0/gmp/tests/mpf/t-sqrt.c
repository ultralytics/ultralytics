/* Test mpf_sqrt, mpf_mul.

Copyright 1996, 2001, 2004 Free Software Foundation, Inc.

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

#ifndef SIZE
#define SIZE 16
#endif

void
check_rand1 (int argc, char **argv)
{
  mp_size_t size;
  mp_exp_t exp;
  int reps = 20000;
  int i;
  mpf_t x, y, y2;
  mp_size_t bprec = 100;
  mpf_t rerr, max_rerr, limit_rerr;

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
  mpf_init (y2);
  for (i = 0; i < reps; i++)
    {
      size = urandom () % SIZE;
      exp = urandom () % SIZE;
      mpf_random2 (x, size, exp);

      mpf_sqrt (y, x);
      MPF_CHECK_FORMAT (y);
      mpf_mul (y2, y, y);

      mpf_reldiff (rerr, x, y2);
      if (mpf_cmp (rerr, max_rerr) > 0)
	{
	  mpf_set (max_rerr, rerr);
#if VERBOSE
	  mpf_dump (max_rerr);
#endif
	  if (mpf_cmp (rerr, limit_rerr) > 0)
	    {
	      printf ("ERROR after %d tests\n", i);
	      printf ("   x = "); mpf_dump (x);
	      printf ("   y = "); mpf_dump (y);
	      printf ("  y2 = "); mpf_dump (y2);
	      printf ("   rerr       = "); mpf_dump (rerr);
	      printf ("   limit_rerr = "); mpf_dump (limit_rerr);
              printf ("in hex:\n");
              mp_trace_base = 16;
	      mpf_trace ("   x  ", x);
	      mpf_trace ("   y  ", y);
	      mpf_trace ("   y2 ", y2);
	      mpf_trace ("   rerr      ", rerr);
	      mpf_trace ("   limit_rerr", limit_rerr);
	      abort ();
	    }
	}
    }

  mpf_clear (limit_rerr);
  mpf_clear (rerr);
  mpf_clear (max_rerr);

  mpf_clear (x);
  mpf_clear (y);
  mpf_clear (y2);
}

void
check_rand2 (void)
{
  unsigned long      max_prec = 20;
  unsigned long      min_prec = __GMPF_BITS_TO_PREC (1);
  gmp_randstate_ptr  rands = RANDS;
  unsigned long      x_prec, r_prec;
  mpf_t              x, r, s;
  int                i;

  mpf_init (x);
  mpf_init (r);
  mpf_init (s);
  refmpf_set_prec_limbs (s, 2*max_prec+10);

  for (i = 0; i < 500; i++)
    {
      /* input precision */
      x_prec = gmp_urandomm_ui (rands, max_prec-min_prec) + min_prec;
      refmpf_set_prec_limbs (x, x_prec);

      /* result precision */
      r_prec = gmp_urandomm_ui (rands, max_prec-min_prec) + min_prec;
      refmpf_set_prec_limbs (r, r_prec);

      mpf_random2 (x, x_prec, 1000);

      mpf_sqrt (r, x);
      MPF_CHECK_FORMAT (r);

      /* Expect to prec limbs of result.
         In the current implementation there's no stripping of low zero
         limbs in mpf_sqrt, so size should be exactly prec.  */
      if (SIZ(r) != r_prec)
        {
          printf ("mpf_sqrt wrong number of result limbs\n");
          mpf_trace ("  x", x);
          mpf_trace ("  r", r);
          printf    ("  r_prec=%lu\n", r_prec);
          printf    ("  SIZ(r)  %ld\n", (long) SIZ(r));
          printf    ("  PREC(r) %ld\n", (long) PREC(r));
          abort ();
        }

      /* Must have r^2 <= x, since r has been truncated. */
      mpf_mul (s, r, r);
      if (! (mpf_cmp (s, x) <= 0))
        {
          printf    ("mpf_sqrt result too big\n");
          mpf_trace ("  x", x);
          printf    ("  r_prec=%lu\n", r_prec);
          mpf_trace ("  r", r);
          mpf_trace ("  s", s);
          abort ();
        }

      /* Must have (r+ulp)^2 > x, or else r is too small. */
      refmpf_add_ulp (r);
      mpf_mul (s, r, r);
      if (! (mpf_cmp (s, x) > 0))
        {
          printf    ("mpf_sqrt result too small\n");
          mpf_trace ("  x", x);
          printf    ("  r_prec=%lu\n", r_prec);
          mpf_trace ("  r+ulp", r);
          mpf_trace ("  s", s);
          abort ();
        }
    }

  mpf_clear (x);
  mpf_clear (r);
  mpf_clear (s);
}

int
main (int argc, char **argv)
{
  tests_start ();
  mp_trace_base = -16;

  check_rand1 (argc, argv);
  check_rand2 ();

  tests_end ();
  exit (0);
}
