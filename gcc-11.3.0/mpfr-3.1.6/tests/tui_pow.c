/* Test file for mpfr_ui_pow and mpfr_ui_pow_ui.

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
#include <stdlib.h>
#include <limits.h>

#include "mpfr-test.h"

static void
test1 (void)
{
  mpfr_t x, y, z, a;
  int res1, res2;

  mpfr_init2 (x, 32);
  mpfr_init2 (y, 65);
  mpfr_init2 (z, 17);
  mpfr_init2 (a, 17);

  mpfr_set_str_binary (x, "-0.101110001001011011011e-9");
  mpfr_ui_pow (y, 7, x, MPFR_RNDN);

  mpfr_set_prec (x, 40);
  mpfr_set_str_binary (x, "-0.1100101100101111011001010010110011110110E-1");
  mpfr_set_prec (y, 74);
  mpfr_ui_pow (y, 8, x, MPFR_RNDN);
  mpfr_set_prec (x, 74);
  mpfr_set_str_binary (x, "0.11100000010100111101000011111011011010011000011000101011010011010101000011E-1");
  if (mpfr_cmp (x, y))
    {
      printf ("Error for input of 40 bits, output of 74 bits\n");
      exit (1);
    }

  /* Check for ui_pow_ui */
  mpfr_ui_pow_ui (x, 0, 1, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_POS (x));
  mpfr_ui_pow_ui (x, 0, 4, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 0) == 0 && MPFR_IS_POS (x));
  res1 = mpfr_ui_pow_ui (z, 17, 42, MPFR_RNDD);
  mpfr_set_ui (x, 17, MPFR_RNDN);
  mpfr_set_ui (y, 42, MPFR_RNDN);
  res2 = mpfr_pow (a, x, y, MPFR_RNDD);
  if (mpfr_cmp (z, a) || res1 != res2)
    {
      printf ("Error for ui_pow_ui for 17^42\n"
              "Inexact1 = %d Inexact2 = %d\n", res1, res2);
      mpfr_dump (z);
      mpfr_dump (a);
      exit (1);
    }
  mpfr_set_prec (x, 2);
  mpfr_ui_pow_ui (x, 65537, 65535, MPFR_RNDN);
  if (mpfr_cmp_str (x, "0.11E1048562", 2, MPFR_RNDN) != 0)
    {
      printf ("Error for ui_pow_ui for 65537 ^65535 with 2 bits of precision\n");
      mpfr_dump (x);
      exit (1);
    }
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (a);
}

static void
check1 (mpfr_ptr x, mpfr_prec_t prec, unsigned long nt, mpfr_rnd_t rnd)
{
  mpfr_t y, z, t;
  int inexact, compare, compare2;
  mpfr_prec_t yprec;
  mpfr_exp_t err;

  yprec = prec + 10;

  mpfr_init (y);
  mpfr_init (z);
  mpfr_init (t);
  mpfr_set_prec (y, yprec);
  mpfr_set_prec (z, prec);
  mpfr_set_prec (t, prec);

  compare = mpfr_ui_pow (y, nt, x, rnd);
  err = (rnd == MPFR_RNDN) ? yprec + 1 : yprec;
  if (mpfr_can_round (y, err, rnd, rnd, prec))
    {
      mpfr_set (t, y, rnd);
      inexact = mpfr_ui_pow (z, nt, x, rnd);
      if (mpfr_cmp (t, z))
        {
          printf ("results differ for x=");
          mpfr_out_str (stdout, 2, prec, x, MPFR_RNDN);
          printf (" n=%lu", nt);
          printf (" prec=%u rnd_mode=%s\n", (unsigned) prec,
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
          printf ("Wrong inexact flag for rnd=%s: expected %d, got %d\n",
                  mpfr_print_rnd_mode (rnd), compare, inexact);
          printf ("x="); mpfr_print_binary (x); puts ("");
          printf ("y="); mpfr_print_binary (y); puts ("");
          printf ("t="); mpfr_print_binary (t); puts ("");
          exit (1);
        }
    }

  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (t);
}

int
main (int argc, char *argv[])
{
  mpfr_t x, y;
  unsigned long int n;

  tests_start_mpfr ();

  mpfr_init (x);
  mpfr_init (y);

  n = randlimb ();

  MPFR_SET_INF(x);
  mpfr_ui_pow (y, n, x, MPFR_RNDN);
  if(!MPFR_IS_INF(y))
    {
      printf ("evaluation of function in INF does not return INF\n");
      exit (1);
    }

  MPFR_CHANGE_SIGN(x);
  mpfr_ui_pow (y, n, x, MPFR_RNDN);
  if(!MPFR_IS_ZERO(y))
    {
      printf ("evaluation of function in -INF does not return 0");
      exit (1);
    }

  MPFR_SET_NAN(x);
  mpfr_ui_pow (y, n, x, MPFR_RNDN);
  if(!MPFR_IS_NAN(y))
    {
      printf ("evaluation of function in NAN does not return NAN");
      exit (1);
    }

  test1 ();

  {
  mpfr_t z, t;
  mpfr_prec_t prec;
  mpfr_rnd_t rnd;
  unsigned int n;

  mpfr_prec_t p0=2, p1=100;
  unsigned int N=20;

  mpfr_init2 (z, 38);
  mpfr_init2 (t, 6);

  /* check exact power */
  mpfr_set_str_binary (t, "0.110000E5");
  mpfr_ui_pow (z, 3, t, MPFR_RNDN);

  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 2);
  mpfr_set_str (x, "-0.5", 10, MPFR_RNDZ);
  mpfr_ui_pow (y, 4, x, MPFR_RNDD);
  if (mpfr_cmp_ui_2exp(y, 1, -1))
    {
      fprintf (stderr, "Error for 4^(-0.5), prec=2, MPFR_RNDD\n");
      fprintf (stderr, "expected 0.5, got ");
      mpfr_out_str (stderr, 2, 0, y, MPFR_RNDN);
      fprintf (stderr, "\n");
      exit (1);
    }

  /* problem found by Kevin on spe175.testdrive.compaq.com
     (03 Sep 2003), ia64 under HP-UX */
  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 2);
  mpfr_set_str (x, "0.5", 10, MPFR_RNDN);
  mpfr_ui_pow (y, 398441521, x, MPFR_RNDN);
  if (mpfr_cmp_ui_2exp(y, 1, 14))
    {
      fprintf (stderr, "Error for 398441521^(0.5), prec=2, MPFR_RNDN\n");
      fprintf (stderr, "expected 1.0e14, got ");
      mpfr_out_str (stderr, 2, 0, y, MPFR_RNDN);
      fprintf (stderr, "\n");
      exit (1);
    }

  mpfr_clear (z);
  mpfr_clear (t);

  mpfr_set_prec (x, 2);
  mpfr_set_str (x, "0.5", 10, MPFR_RNDN);
  check1 (x, 2, 398441521, MPFR_RNDN);  /* 398441521 = 19961^2 */

  /* generic test */
  for (prec = p0; prec <= p1; prec++)
    {
      mpfr_set_prec (x, prec);
      for (n=0; n<N; n++)
        {
          int nt;
          nt = randlimb () & INT_MAX;
          mpfr_urandomb (x, RANDS);
          rnd = RND_RAND ();
          check1 (x, prec, nt, rnd);
        }
    }
  }

  mpfr_clear (x);
  mpfr_clear (y);

  tests_end_mpfr ();
  return 0;
}
