/* Test file for mpfr_pow_z -- power function x^z with z a MPZ

Copyright 2005-2017 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "mpfr-test.h"

#define ERROR(str) do { printf ("Error for " str "\n"); exit (1); } while (0)

static void
check_special (void)
{
  mpfr_t x, y;
  mpz_t  z;
  int res;

  mpfr_init (x);
  mpfr_init (y);
  mpz_init (z);

  /* x^0 = 1 except for NAN */
  mpfr_set_ui (x, 23, MPFR_RNDN);
  mpz_set_ui (z, 0);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (y, 1) != 0)
    ERROR ("23^0");
  mpfr_set_nan (x);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_nan_p (y) || mpfr_cmp_si (y, 1) != 0)
    ERROR ("NAN^0");
  mpfr_set_inf (x, 1);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (y, 1) != 0)
    ERROR ("INF^0");

  /* sINF^N = INF if s==1 or n even if N > 0*/
  mpz_set_ui (z, 42);
  mpfr_set_inf (x, 1);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_inf_p (y) == 0 || mpfr_sgn (y) <= 0)
    ERROR ("INF^42");
  mpfr_set_inf (x, -1);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_inf_p (y) == 0 || mpfr_sgn (y) <= 0)
    ERROR ("-INF^42");
  mpz_set_ui (z, 17);
  mpfr_set_inf (x, 1);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_inf_p (y) == 0 || mpfr_sgn (y) <= 0)
    ERROR ("INF^17");
  mpfr_set_inf (x, -1);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_inf_p (y) == 0 || mpfr_sgn (y) >= 0)
    ERROR ("-INF^17");

  mpz_set_si (z, -42);
  mpfr_set_inf (x, 1);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_zero_p (y) == 0 || MPFR_SIGN (y) <= 0)
    ERROR ("INF^-42");
  mpfr_set_inf (x, -1);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_zero_p (y) == 0 || MPFR_SIGN (y) <= 0)
    ERROR ("-INF^-42");
  mpz_set_si (z, -17);
  mpfr_set_inf (x, 1);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_zero_p (y) == 0 || MPFR_SIGN (y) <= 0)
    ERROR ("INF^-17");
  mpfr_set_inf (x, -1);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_zero_p (y) == 0 || MPFR_SIGN (y) >= 0)
    ERROR ("-INF^-17");

  /* s0^N = +0 if s==+ or n even if N > 0*/
  mpz_set_ui (z, 42);
  MPFR_SET_ZERO (x); MPFR_SET_POS (x);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_zero_p (y) == 0 || MPFR_SIGN (y) <= 0)
    ERROR ("+0^42");
  MPFR_SET_NEG (x);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_zero_p (y) == 0 || MPFR_SIGN (y) <= 0)
    ERROR ("-0^42");
  mpz_set_ui (z, 17);
  MPFR_SET_POS (x);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_zero_p (y) == 0 || MPFR_SIGN (y) <= 0)
    ERROR ("+0^17");
  MPFR_SET_NEG (x);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_zero_p (y) == 0 || MPFR_SIGN (y) >= 0)
    ERROR ("-0^17");

  mpz_set_si (z, -42);
  MPFR_SET_ZERO (x); MPFR_SET_POS (x);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_inf_p (y) == 0 || MPFR_SIGN (y) <= 0)
    ERROR ("+0^-42");
  MPFR_SET_NEG (x);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_inf_p (y) == 0 || MPFR_SIGN (y) <= 0)
    ERROR ("-0^-42");
  mpz_set_si (z, -17);
  MPFR_SET_POS (x);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_inf_p (y) == 0 || MPFR_SIGN (y) <= 0)
    ERROR ("+0^-17");
  MPFR_SET_NEG (x);
  res = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (res != 0 || mpfr_inf_p (y) == 0 || MPFR_SIGN (y) >= 0)
    ERROR ("-0^-17");

  mpz_clear (z);
  mpfr_clear (y);
  mpfr_clear (x);
}

static void
check_integer (mpfr_prec_t begin, mpfr_prec_t end, unsigned long max)
{
  mpfr_t x, y1, y2;
  mpz_t z;
  unsigned long i, n;
  mpfr_prec_t p;
  int res1, res2;
  mpfr_rnd_t rnd;

  mpfr_inits2 (begin, x, y1, y2, (mpfr_ptr) 0);
  mpz_init (z);
  for (p = begin ; p < end ; p+=4)
    {
      mpfr_set_prec (x, p);
      mpfr_set_prec (y1, p);
      mpfr_set_prec (y2, p);
      for (i = 0 ; i < max ; i++)
        {
          mpz_urandomb (z, RANDS, GMP_NUMB_BITS);
          if ((i & 1) != 0)
            mpz_neg (z, z);
          mpfr_urandomb (x, RANDS);
          mpfr_mul_2ui (x, x, 1, MPFR_RNDN); /* 0 <= x < 2 */
          rnd = RND_RAND ();
          if (mpz_fits_slong_p (z))
            {
              n = mpz_get_si (z);
              /* printf ("New test for x=%ld\nCheck Pow_si\n", n); */
              res1 = mpfr_pow_si (y1, x, n, rnd);
              /* printf ("Check pow_z\n"); */
              res2 = mpfr_pow_z  (y2, x, z, rnd);
              if (mpfr_cmp (y1, y2) != 0)
                {
                  printf ("Error for p = %lu, z = %lu, rnd = %s and x = ",
                          (unsigned long) p, n, mpfr_print_rnd_mode (rnd));
                  mpfr_dump (x);
                  printf ("Ypowsi = "); mpfr_dump (y1);
                  printf ("Ypowz  = "); mpfr_dump (y2);
                  exit (1);
                }
              if (res1 != res2)
                {
                  printf ("Wrong inexact flags for p = %lu, z = %lu, rnd = %s"
                          " and x = ", (unsigned long) p, n,
                          mpfr_print_rnd_mode (rnd));
                  mpfr_dump (x);
                  printf ("Ypowsi(inex = %2d) = ", res1); mpfr_dump (y1);
                  printf ("Ypowz (inex = %2d) = ", res2); mpfr_dump (y2);
                  exit (1);
                }
            }
        } /* for i */
    } /* for p */
  mpfr_clears (x, y1, y2, (mpfr_ptr) 0);
  mpz_clear (z);
}

static void
check_regression (void)
{
  mpfr_t x, y;
  mpz_t  z;
  int res1, res2;

  mpz_init_set_ui (z, 2026876995);
  mpfr_init2 (x, 122);
  mpfr_init2 (y, 122);

  mpfr_set_str_binary (x, "0.10000010010000111101001110100101101010011110011100001111000001001101000110011001001001001011001011010110110110101000111011E1");
  res1 = mpfr_pow_z (y, x, z, MPFR_RNDU);
  res2 = mpfr_pow_ui (x, x, 2026876995UL, MPFR_RNDU);
  if (mpfr_cmp (x, y) || res1 != res2)
    {
      printf ("Regression (1) tested failed (%d=?%d)\n",res1, res2);
      printf ("pow_ui: "); mpfr_dump (x);
      printf ("pow_z:  "); mpfr_dump (y);

      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpz_clear (z);
}

/* Bug found by Kevin P. Rauch */
static void
bug20071104 (void)
{
  mpfr_t x, y;
  mpz_t z;
  int inex;

  mpz_init_set_si (z, -2);
  mpfr_inits2 (20, x, y, (mpfr_ptr) 0);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_nextbelow (x);  /* x = -2^(emin-1) */
  mpfr_clear_flags ();
  inex = mpfr_pow_z (y, x, z, MPFR_RNDN);
  if (! mpfr_inf_p (y) || MPFR_SIGN (y) < 0)
    {
      printf ("Error in bug20071104: expected +Inf, got ");
      mpfr_dump (y);
      exit (1);
    }
  if (inex <= 0)
    {
      printf ("Error in bug20071104: bad ternary value (%d)\n", inex);
      exit (1);
    }
  if (__gmpfr_flags != (MPFR_FLAGS_OVERFLOW | MPFR_FLAGS_INEXACT))
    {
      printf ("Error in bug20071104: bad flags (%u)\n", __gmpfr_flags);
      exit (1);
    }
  mpfr_clears (x, y, (mpfr_ptr) 0);
  mpz_clear (z);
}

static void
check_overflow (void)
{
  mpfr_t a;
  mpz_t z;
  unsigned long n;
  int res;

  mpfr_init2 (a, 53);

  mpfr_set_str_binary (a, "1E10");
  mpz_init_set_ui (z, ULONG_MAX);
  res = mpfr_pow_z (a, a, z, MPFR_RNDN);
  if (! MPFR_IS_INF (a) || MPFR_SIGN (a) < 0 || res <= 0)
    {
      printf ("Error for (1e10)^ULONG_MAX, expected +Inf,\ngot ");
      mpfr_dump (a);
      exit (1);
    }

  /* Bug in pow_z.c up to r5109: if x = y (same mpfr_t argument), the
     input argument is negative and not a power of two, z is positive
     and odd, an overflow or underflow occurs, and the temporary result
     res is positive, then the result gets a wrong sign (positive
     instead of negative). */
  mpfr_set_str_binary (a, "-1.1E10");
  n = (ULONG_MAX ^ (ULONG_MAX >> 1)) + 1;
  mpz_set_ui (z, n);
  res = mpfr_pow_z (a, a, z, MPFR_RNDN);
  if (! MPFR_IS_INF (a) || MPFR_SIGN (a) > 0 || res >= 0)
    {
      printf ("Error for (-1e10)^%lu, expected -Inf,\ngot ", n);
      mpfr_dump (a);
      exit (1);
    }

  mpfr_clear (a);
  mpz_clear (z);
}

/* bug reported by Carl Witty (32-bit architecture) */
static void
bug20080223 (void)
{
  mpfr_t a, exp, answer;

  mpfr_init2 (a, 53);
  mpfr_init2 (exp, 53);
  mpfr_init2 (answer, 53);

  mpfr_set_si (exp, -1073741824, MPFR_RNDN);

  mpfr_set_str (a, "1.999999999", 10, MPFR_RNDN);
  /* a = 562949953139837/2^48 */
  mpfr_pow (answer, a, exp, MPFR_RNDN);
  mpfr_set_str_binary (a, "0.110110101111011001110000111111100011101000111011101E-1073741823");
  MPFR_ASSERTN(mpfr_cmp0 (answer, a) == 0);

  mpfr_clear (a);
  mpfr_clear (exp);
  mpfr_clear (answer);
}

static void
bug20080904 (void)
{
  mpz_t exp;
  mpfr_t a, answer;
  mpfr_exp_t emin_default;

  mpz_init (exp);
  mpfr_init2 (a, 70);
  mpfr_init2 (answer, 70);

  emin_default = mpfr_get_emin ();
  mpfr_set_emin (MPFR_EMIN_MIN);

  mpz_set_str (exp, "-4eb92f8c7b7bf81e", 16);
  mpfr_set_str_binary (a, "1.110000101110100110100011111000011110111101000011111001111001010011100");

  mpfr_pow_z (answer, a, exp, MPFR_RNDN);
  /* The correct result is near 2^(-2^62), so it underflows when
     MPFR_EMIN_MIN > -2^62 (i.e. with 32 and 64 bits machines). */
  mpfr_set_str (a, "AA500C0D7A69275DBp-4632850503556296886", 16, MPFR_RNDN);
  if (! mpfr_equal_p (answer, a))
    {
      printf ("Error in bug20080904:\n");
      printf ("Expected ");
      mpfr_out_str (stdout, 16, 0, a, MPFR_RNDN);
      putchar ('\n');
      printf ("Got      ");
      mpfr_out_str (stdout, 16, 0, answer, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }

  mpfr_set_emin (emin_default);

  mpz_clear (exp);
  mpfr_clear (a);
  mpfr_clear (answer);
}

int
main (void)
{
  tests_start_mpfr ();

  check_special ();

  check_integer (2, 163, 100);
  check_regression ();
  bug20071104 ();
  bug20080223 ();
  bug20080904 ();
  check_overflow ();

  tests_end_mpfr ();
  return 0;
}
