/* Test file for mpfr_modf.

Copyright 2007-2017 Free Software Foundation, Inc.
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

#if MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0)

static void
check (const char *xis, const char *xfs, const char *xs,
       mpfr_prec_t xip, mpfr_prec_t xfp, mpfr_prec_t xp,
       int expected_return, mpfr_rnd_t rnd_mode)
{
  int inexact;
  mpfr_t xi, xf, x;

  mpfr_init2 (xi, xip);
  mpfr_init2 (xf, xfp);
  mpfr_init2 (x, xp);
  mpfr_set_str1 (x, xs);
  inexact = mpfr_modf (xi, xf, x, rnd_mode);
  if (mpfr_cmp_str1 (xi, xis))
    {
      printf ("mpfr_modf failed for x=%s, rnd=%s\n",
              xs, mpfr_print_rnd_mode(rnd_mode));
      printf ("got integer value: ");
      mpfr_out_str (stdout, 10, 0, xi, MPFR_RNDN);
      printf ("\nexpected %s\n", xis);
      exit (1);
    }
  if (mpfr_cmp_str1 (xf, xfs))
    {
      printf ("mpfr_modf failed for x=%s, rnd=%s\n",
              xs, mpfr_print_rnd_mode(rnd_mode));
      printf ("got fractional value: ");
      mpfr_out_str (stdout, 10, 0, xf, MPFR_RNDN);
      printf ("\nexpected %s\n", xfs);
      exit (1);
    }
  if (inexact != expected_return)
    {
      printf ("mpfr_modf failed for x=%s, rnd=%s\n",
              xs, mpfr_print_rnd_mode(rnd_mode));
      printf ("got return value: %d, expected %d\n", inexact, expected_return);
      exit (1);
    }
  mpfr_clears (xi, xf, x, (mpfr_ptr) 0);
}

static void
check_nans (void)
{
  mpfr_t  x, xi, xf;

  mpfr_init2 (x, 123);
  mpfr_init2 (xi, 123);
  mpfr_init2 (xf, 123);

  /* nan */
  mpfr_set_nan (x);
  mpfr_modf (xi, xf, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (xi));
  MPFR_ASSERTN (mpfr_nan_p (xf));

  /* +inf */
  mpfr_set_inf (x, 1);
  mpfr_modf (xi, xf, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (xi));
  MPFR_ASSERTN (mpfr_sgn (xi) > 0);
  MPFR_ASSERTN (mpfr_zero_p (xf));

  /* -inf */
  mpfr_set_inf (x, -1);
  mpfr_modf (xi ,xf, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (xi));
  MPFR_ASSERTN (mpfr_sgn (xi) < 0);
  MPFR_ASSERTN (mpfr_zero_p (xf));

  mpfr_clear (x);
  mpfr_clear (xi);
  mpfr_clear (xf);
}

static void
check_special_exprange (void)
{
  int inexact, ov;
  unsigned int eflags, gflags;
  mpfr_t xi, xf, x;
  mpfr_exp_t emax;

  emax = mpfr_get_emax ();
  mpfr_init2 (xi, 7);
  mpfr_init2 (xf, 7);
  mpfr_init2 (x, 8);

  mpfr_set_str (x, "0.11111111", 2, MPFR_RNDN);
  for (ov = 0; ov <= 1; ov++)
    {
      const char *s = ov ? "@Inf@" : "1";

      if (ov)
        set_emax (0);
      mpfr_clear_flags ();
      inexact = mpfr_modf (xi, xf, x, MPFR_RNDN);
      gflags = __gmpfr_flags;
      set_emax (emax);
      if (MPFR_NOTZERO (xi) || MPFR_IS_NEG (xi) ||
          mpfr_cmp_str1 (xf, s) != 0)
        {
          printf ("Error in check_special_exprange (ov = %d):"
                  " expected 0 and %s, got\n", ov, s);
          mpfr_out_str (stdout, 2, 0, xi, MPFR_RNDN);
          printf (" and ");
          mpfr_out_str (stdout, 2, 0, xf, MPFR_RNDN);
          printf ("\n");
          exit (1);
        }
      if (inexact != 4)
        {
          printf ("Bad inexact value in check_special_exprange (ov = %d):"
                  " expected 4, got %d\n", ov, inexact);
          exit (1);
        }
      eflags = MPFR_FLAGS_INEXACT | (ov ? MPFR_FLAGS_OVERFLOW : 0);
      if (gflags != eflags)
        {
          printf ("Bad flags in check_special_exprange (ov = %d):"
                  " expected %u, got %u\n", ov, eflags, gflags);
          exit (1);
        }
    }

  /* Test if an overflow occurs in mpfr_set for ope >= opq. */
  mpfr_set_emax (MPFR_EMAX_MAX);
  mpfr_set_inf (x, 1);
  mpfr_nextbelow (x);
  mpfr_clear_flags ();
  inexact = mpfr_modf (xi, xf, x, MPFR_RNDN);
  gflags = __gmpfr_flags;
  if (mpfr_cmp_str1 (xi, "@Inf@") != 0 ||
      MPFR_NOTZERO (xf) || MPFR_IS_NEG (xf))
    {
      printf ("Error in check_special_exprange:"
              " expected 0 and @Inf@, got\n");
      mpfr_out_str (stdout, 2, 0, xi, MPFR_RNDN);
      printf (" and ");
      mpfr_out_str (stdout, 2, 0, xf, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  if (inexact != 1)
    {
      printf ("Bad inexact value in check_special_exprange:"
              " expected 1, got %d\n", inexact);
      exit (1);
    }
  eflags = MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW;
  if (gflags != eflags)
    {
      printf ("Bad flags in check_special_exprange:"
              " expected %u, got %u\n", eflags, gflags);
      exit (1);
    }
  set_emax (emax);

  /* Test if an underflow occurs in the general case. TODO */

  mpfr_clears (xi, xf, x, (mpfr_ptr) 0);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  check_nans ();

  /* integer part is exact, frac. part is exact: return value should be 0 */
  check ("61680","3.52935791015625e-1", "61680.352935791015625",
         53, 53, 53, 0, MPFR_RNDZ);
  /* integer part is rounded up, fractional part is rounded up: return value
     should be 1+4*1=5 */
  check ("-53968","-3.529052734375e-1", "-53970.352935791015625",
         13, 13, 53, 5, MPFR_RNDZ);
  /* integer part is rounded down, fractional part is rounded down:
     return value should be 2+4*2=10 */
  check ("61632","3.525390625e-1",      "61648.352935791015625",
         10, 10, 53, 10, MPFR_RNDZ);
  check ("61680", "0", "61680",  53, 53, 53, 0, MPFR_RNDZ);
  /* integer part is rounded up, fractional part is exact: 1 */
  check ("-53968","0", "-53970", 13, 13, 53, 1, MPFR_RNDZ);
  /* integer part is rounded up, fractional part is exact: 1 */
  check ("-43392","0", "-43399", 13, 13, 53, 1, MPFR_RNDU);
  /* integer part is rounded down, fractional part is exact: 2 */
  check ("-52720","0", "-52719", 13, 13, 53, 2, MPFR_RNDD);
  /* integer part is rounded down, fractional part is exact: 2 */
  check ("61632", "0", "61648",  10, 10, 53, 2, MPFR_RNDZ);

  check_special_exprange ();

  tests_end_mpfr ();
  return 0;
}

#else

int
main (void)
{
  printf ("Warning! Test disabled for this MPFR version.\n");
  return 0;
}

#endif
