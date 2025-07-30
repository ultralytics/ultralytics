/* Test file for mpfr_set_q.

Copyright 2000-2017 Free Software Foundation, Inc.
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
check (long int n, long int d, mpfr_rnd_t rnd, const char *ys)
{
  mpq_t q;
  mpfr_t x, t;
  int inexact, compare;
  unsigned int flags, ex_flags;

  mpfr_init2 (x, 53);
  mpfr_init2 (t, mpfr_get_prec (x) + mp_bits_per_limb);
  mpq_init (q);
  mpq_set_si (q, n, d);
  mpfr_clear_flags ();
  inexact = mpfr_set_q (x, q, rnd);
  flags = __gmpfr_flags;

  /* check values */
  if (mpfr_cmp_str1 (x, ys))
    {
      printf ("Error for q = %ld/%ld and rnd = %s\n", n, d,
              mpfr_print_rnd_mode (rnd));
      printf ("correct result is %s, mpfr_set_q gives ", ys);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }

  /* check inexact flag */
  if (mpfr_mul_ui (t, x, (d < 0) ? (-d) : d, rnd))
    {
      printf ("t <- x * d should be exact\n");
      exit (1);
    }
  compare = mpfr_cmp_si (t, n);
  if (! SAME_SIGN (inexact, compare))
    {
      printf ("Wrong ternary value for q = %ld/%ld and rnd = %s:\n"
              "expected %d or equivalent, got %d\n",
              n, d, mpfr_print_rnd_mode (rnd), compare, inexact);
      exit (1);
    }

  ex_flags = compare == 0 ? 0 : MPFR_FLAGS_INEXACT;
  if (flags != ex_flags)
    {
      printf ("Wrong flags for q = %ld/%ld and rnd = %s:\n",
              n, d, mpfr_print_rnd_mode (rnd));
      printf ("Expected flags:");
      flags_out (ex_flags);
      printf ("Got flags:     ");
      flags_out (flags);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (t);
  mpq_clear (q);
}

static void
check0 (void)
{
  mpq_t y;
  mpfr_t x;
  int inexact;
  int r;

  /* Check for +0 */
  mpfr_init (x);
  mpq_init (y);
  mpq_set_si (y, 0, 1);
  for (r = 0; r < MPFR_RND_MAX; r++)
    {
      mpfr_clear_flags ();
      inexact = mpfr_set_q (x, y, (mpfr_rnd_t) r);
      if (!MPFR_IS_ZERO(x) || !MPFR_IS_POS(x) || inexact ||
          __gmpfr_flags != 0)
        {
          printf("mpfr_set_q(x,0) failed for %s\n",
                 mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit(1);
        }
    }
  mpfr_clear (x);
  mpq_clear (y);
}

static void
check_nan_inf_mpq (void)
{
  mpfr_t mpfr_value, mpfr_cmp;
  mpq_t mpq_value;
  int status;

  mpfr_init2 (mpfr_value, MPFR_PREC_MIN);
  mpq_init (mpq_value);
  mpq_set_si (mpq_value, 0, 0);
  mpz_set_si (mpq_denref (mpq_value), 0);

  status = mpfr_set_q (mpfr_value, mpq_value, MPFR_RNDN);

  if ((status != 0) || (!MPFR_IS_NAN (mpfr_value)))
    {
      mpfr_init2 (mpfr_cmp, MPFR_PREC_MIN);
      mpfr_set_nan (mpfr_cmp);
      printf ("mpfr_set_q with a NAN mpq value returned a wrong value :\n"
              " waiting for ");
      mpfr_print_binary (mpfr_cmp);
      printf (" got ");
      mpfr_print_binary (mpfr_value);
      printf ("\n trinary value is %d\n", status);
      exit (1);
    }

  mpq_set_si (mpq_value, -1, 0);
  mpz_set_si (mpq_denref (mpq_value), 0);

  status = mpfr_set_q (mpfr_value, mpq_value, MPFR_RNDN);

  if ((status != 0) || (!MPFR_IS_INF (mpfr_value)) ||
      (MPFR_SIGN(mpfr_value) != mpq_sgn(mpq_value)))
    {
      mpfr_init2 (mpfr_cmp, MPFR_PREC_MIN);
      mpfr_set_inf (mpfr_cmp, -1);
      printf ("mpfr_set_q with a -INF mpq value returned a wrong value :\n"
              " waiting for ");
      mpfr_print_binary (mpfr_cmp);
      printf (" got ");
      mpfr_print_binary (mpfr_value);
      printf ("\n trinary value is %d\n", status);
      exit (1);
    }

  mpq_clear (mpq_value);
  mpfr_clear (mpfr_value);
}

int
main (void)
{
  tests_start_mpfr ();

  check(-1647229822, 40619231, MPFR_RNDZ, "-4.055295438754120596e1");
  check(-148939696, 1673285490, MPFR_RNDZ, "-8.9010331404953485501e-2");
  check(-441322590, 273662545, MPFR_RNDZ, "-1.6126525096812205362");
  check(-1631156895, 1677687197, MPFR_RNDU, "-9.722652100563177191e-1");
  check(2141332571, 3117601, MPFR_RNDZ, "6.8685267004982347316e2");
  check(75504803, 400207282, MPFR_RNDU, "1.8866424074712365155e-1");
  check(643562308, 23100894, MPFR_RNDD, "2.7858762002890447462e1");
  check(632549085, 1831935802, MPFR_RNDN, "3.4528998467600230393e-1");
  check (1, 1, MPFR_RNDN, "1.0");

  check0();

  check_nan_inf_mpq ();

  tests_end_mpfr ();
  return 0;
}
