/* Test file for mpfr_pow.

Copyright 2001-2017 Free Software Foundation, Inc.
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
#include <limits.h>
#include <stdlib.h>

#include "mpfr-test.h"


int
main (int argc, char *argv[])
{
  mpfr_t x, y, z;

  mpfr_prec_t prec, yprec;
  mpfr_t t, s;
  mpfr_rnd_t rnd;
  int inexact, compare, compare2;
  unsigned int n, err;

  mpfr_prec_t p0=2, p1=100;
  unsigned int N=25;

  tests_start_mpfr ();

  mpfr_init (x);
  mpfr_init2 (y,sizeof(unsigned long int)*CHAR_BIT);
  mpfr_init (z);

  mpfr_init (s);
  mpfr_init (t);

  /* generic test */
  for (prec = p0; prec <= p1; prec++)
    {
      mpfr_set_prec (x, prec);
      mpfr_set_prec (s, sizeof(unsigned long int)*CHAR_BIT);
      mpfr_set_prec (z, prec);
      mpfr_set_prec (t, prec);
      yprec = prec + 10;

      for (n=0; n<N; n++)
        {
          mpfr_urandomb (x, RANDS);
          mpfr_urandomb (s, RANDS);
          if (randlimb () % 2)
            mpfr_neg (s, s, MPFR_RNDN);
          rnd = RND_RAND ();
          mpfr_set_prec (y, yprec);
          compare = mpfr_pow (y, x, s, rnd);
          err = (rnd == MPFR_RNDN) ? yprec + 1 : yprec;
          if (mpfr_can_round (y, err, rnd, rnd, prec))
            {
              mpfr_set (t, y, rnd);
              inexact = mpfr_pow (z, x, s, rnd);
              if (mpfr_cmp (t, z))
                {
                  printf ("results differ for x^y with x=");
                  mpfr_out_str (stdout, 2, prec, x, MPFR_RNDN);
                  printf (" y=");
                  mpfr_out_str (stdout, 2, 0, s, MPFR_RNDN);
                  printf (" prec=%u rnd_mode=%s\n", (unsigned int) prec,
                          mpfr_print_rnd_mode (rnd));
                  printf ("got      ");
                  mpfr_out_str (stdout, 2, prec, z, MPFR_RNDN);
                  puts ("");
                  printf ("expected ");
                  mpfr_out_str (stdout, 2, prec, t, MPFR_RNDN);
                  puts ("");
                  printf ("approx  ");
                  mpfr_print_binary (y);
                  puts ("");
                  exit (1);
                }
              compare2 = mpfr_cmp (t, y);
              /* if rounding to nearest, cannot know the sign of t - f(x)
                 because of composed rounding: y = o(f(x)) and t = o(y) */
              if ((rnd != MPFR_RNDN) && (compare * compare2 >= 0))
                compare = compare + compare2;
              else
                compare = inexact; /* cannot determine sign(t-f(x)) */
              if (((inexact == 0) && (compare != 0)) ||
                  ((inexact > 0) && (compare <= 0)) ||
                  ((inexact < 0) && (compare >= 0)))
                {
                  printf ("Wrong inexact flag for rnd=%s: expected %d, got %d"
                          "\n", mpfr_print_rnd_mode (rnd), compare, inexact);
                  printf ("x="); mpfr_print_binary (x); puts ("");
                  printf ("y="); mpfr_print_binary (y); puts ("");
                  printf ("t="); mpfr_print_binary (t); puts ("");
                  exit (1);
                }
            }
        }
    }

  mpfr_clear (s);
  mpfr_clear (t);

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);

  tests_end_mpfr ();
  return 0;
}
