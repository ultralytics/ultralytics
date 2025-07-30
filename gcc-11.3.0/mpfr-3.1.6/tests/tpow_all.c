/* Test file for the various power functions

Copyright 2008-2017 Free Software Foundation, Inc.
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

/* Note: some tests of the other tpow* test files could be moved there.
   The main goal of this test file is to test _all_ the power functions
   on special values, to make sure that they are consistent and give the
   expected result, in particular because such special cases are handled
   in different ways in each function. */

/* Execute with at least an argument to report all the errors found by
   comparisons. */

#include <stdlib.h>

#include "mpfr-test.h"

/* Behavior of cmpres (called by test_others):
 *   0: stop as soon as an error is found.
 *   1: report all errors found by test_others.
 *  -1: the 1 is changed to this value as soon as an error has been found.
 */
static int all_cmpres_errors;

/* Non-zero when extended exponent range */
static int ext = 0;

static const char *val[] =
  { "min", "min+", "max", "@NaN@", "-@Inf@", "-4", "-3", "-2", "-1.5",
    "-1", "-0.5", "-0", "0", "0.5", "1", "1.5", "2", "3", "4", "@Inf@" };

static void
err (const char *s, int i, int j, int rnd, mpfr_srcptr z, int inex)
{
  puts (s);
  if (ext)
    puts ("extended exponent range");
  printf ("x = %s, y = %s, %s\n", val[i], val[j],
          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
  printf ("z = ");
  mpfr_out_str (stdout, 10, 0, z, MPFR_RNDN);
  printf ("\ninex = %d\n", inex);
  exit (1);
}

/* Arguments:
 *   spx: non-zero if px is a stringm zero if px is a MPFR number.
 *   px: value of x (string or MPFR number).
 *   sy: value of y (string).
 *   rnd: rounding mode.
 *   z1: expected result (null pointer if unknown pure FP value).
 *   inex1: expected ternary value (if z1 is not a null pointer).
 *   z2: computed result.
 *   inex2: computed ternary value.
 *   flags1: expected flags (computed flags in __gmpfr_flags).
 *   s1, s2: strings about the context.
 */
static void
cmpres (int spx, const void *px, const char *sy, mpfr_rnd_t rnd,
        mpfr_srcptr z1, int inex1, mpfr_srcptr z2, int inex2,
        unsigned int flags1, const char *s1, const char *s2)
{
  unsigned int flags2 = __gmpfr_flags;

  if (flags1 == flags2)
    {
      /* Note: the test on the sign of z1 and z2 is needed
         in case they are both zeros. */
      if (z1 == NULL)
        {
          if (MPFR_IS_PURE_FP (z2))
            return;
        }
      else if (SAME_SIGN (inex1, inex2) &&
               ((MPFR_IS_NAN (z1) && MPFR_IS_NAN (z2)) ||
                ((MPFR_IS_NEG (z1) ^ MPFR_IS_NEG (z2)) == 0 &&
                 mpfr_equal_p (z1, z2))))
        return;
    }

  printf ("Error in %s\nwith %s%s\nx = ", s1, s2,
          ext ? ", extended exponent range" : "");
  if (spx)
    printf ("%s, ", (char *) px);
  else
    {
      mpfr_out_str (stdout, 16, 0, (mpfr_ptr) px, MPFR_RNDN);
      puts (",");
    }
  printf ("y = %s, %s\n", sy, mpfr_print_rnd_mode (rnd));
  printf ("Expected ");
  if (z1 == NULL)
    {
      printf ("pure FP value, flags =");
      flags_out (flags1);
    }
  else
    {
      mpfr_out_str (stdout, 16, 0, z1, MPFR_RNDN);
      printf (", inex = %d,\n         flags =", SIGN (inex1));
      flags_out (flags1);
    }
  printf ("Got      ");
  mpfr_out_str (stdout, 16, 0, z2, MPFR_RNDN);
  printf (", inex = %d,\n         flags =", SIGN (inex2));
  flags_out (flags2);
  if (all_cmpres_errors != 0)
    all_cmpres_errors = -1;
  else
    exit (1);
}

static int
is_odd (mpfr_srcptr x)
{
  /* works only with the values from val[] */
  return mpfr_integer_p (x) && mpfr_fits_slong_p (x, MPFR_RNDN) &&
    (mpfr_get_si (x, MPFR_RNDN) & 1);
}

/* Compare the result (z1,inex1) of mpfr_pow with all flags cleared
   with those of mpfr_pow with all flags set and of the other power
   functions. Arguments x and y are the input values; sx and sy are
   their string representations (sx may be null); rnd contains the
   rounding mode; s is a string containing the function that called
   test_others. */
static void
test_others (const void *sx, const char *sy, mpfr_rnd_t rnd,
             mpfr_srcptr x, mpfr_srcptr y, mpfr_srcptr z1,
             int inex1, unsigned int flags, const char *s)
{
  mpfr_t z2;
  int inex2;
  int spx = sx != NULL;

  if (!spx)
    sx = x;

  mpfr_init2 (z2, mpfr_get_prec (z1));

  __gmpfr_flags = MPFR_FLAGS_ALL;
  inex2 = mpfr_pow (z2, x, y, rnd);
  cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
          s, "mpfr_pow, flags set");

  /* If y is an integer that fits in an unsigned long and is not -0,
     we can test mpfr_pow_ui. */
  if (MPFR_IS_POS (y) && mpfr_integer_p (y) &&
      mpfr_fits_ulong_p (y, MPFR_RNDN))
    {
      unsigned long yy = mpfr_get_ui (y, MPFR_RNDN);

      mpfr_clear_flags ();
      inex2 = mpfr_pow_ui (z2, x, yy, rnd);
      cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
              s, "mpfr_pow_ui, flags cleared");
      __gmpfr_flags = MPFR_FLAGS_ALL;
      inex2 = mpfr_pow_ui (z2, x, yy, rnd);
      cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
              s, "mpfr_pow_ui, flags set");

      /* If x is an integer that fits in an unsigned long and is not -0,
         we can also test mpfr_ui_pow_ui. */
      if (MPFR_IS_POS (x) && mpfr_integer_p (x) &&
          mpfr_fits_ulong_p (x, MPFR_RNDN))
        {
          unsigned long xx = mpfr_get_ui (x, MPFR_RNDN);

          mpfr_clear_flags ();
          inex2 = mpfr_ui_pow_ui (z2, xx, yy, rnd);
          cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
                  s, "mpfr_ui_pow_ui, flags cleared");
          __gmpfr_flags = MPFR_FLAGS_ALL;
          inex2 = mpfr_ui_pow_ui (z2, xx, yy, rnd);
          cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
                  s, "mpfr_ui_pow_ui, flags set");
        }
    }

  /* If y is an integer but not -0 and not huge, we can test mpfr_pow_z,
     and possibly mpfr_pow_si (and possibly mpfr_ui_div). */
  if (MPFR_IS_ZERO (y) ? MPFR_IS_POS (y) :
      (mpfr_integer_p (y) && MPFR_GET_EXP (y) < 256))
    {
      mpz_t yyy;

      /* If y fits in a long, we can test mpfr_pow_si. */
      if (mpfr_fits_slong_p (y, MPFR_RNDN))
        {
          long yy = mpfr_get_si (y, MPFR_RNDN);

          mpfr_clear_flags ();
          inex2 = mpfr_pow_si (z2, x, yy, rnd);
          cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
                  s, "mpfr_pow_si, flags cleared");
          __gmpfr_flags = MPFR_FLAGS_ALL;
          inex2 = mpfr_pow_si (z2, x, yy, rnd);
          cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
                  s, "mpfr_pow_si, flags set");

          /* If y = -1, we can test mpfr_ui_div. */
          if (yy == -1)
            {
              mpfr_clear_flags ();
              inex2 = mpfr_ui_div (z2, 1, x, rnd);
              cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
                      s, "mpfr_ui_div, flags cleared");
              __gmpfr_flags = MPFR_FLAGS_ALL;
              inex2 = mpfr_ui_div (z2, 1, x, rnd);
              cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
                      s, "mpfr_ui_div, flags set");
            }

          /* If y = 2, we can test mpfr_sqr. */
          if (yy == 2)
            {
              mpfr_clear_flags ();
              inex2 = mpfr_sqr (z2, x, rnd);
              cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
                      s, "mpfr_sqr, flags cleared");
              __gmpfr_flags = MPFR_FLAGS_ALL;
              inex2 = mpfr_sqr (z2, x, rnd);
              cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
                      s, "mpfr_sqr, flags set");
            }
        }

      /* Test mpfr_pow_z. */
      mpz_init (yyy);
      mpfr_get_z (yyy, y, MPFR_RNDN);
      mpfr_clear_flags ();
      inex2 = mpfr_pow_z (z2, x, yyy, rnd);
      cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
              s, "mpfr_pow_z, flags cleared");
      __gmpfr_flags = MPFR_FLAGS_ALL;
      inex2 = mpfr_pow_z (z2, x, yyy, rnd);
      cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
              s, "mpfr_pow_z, flags set");
      mpz_clear (yyy);
    }

  /* If y = 0.5, we can test mpfr_sqrt, except if x is -0 or -Inf (because
     the rule for mpfr_pow on these special values is different). */
  if (MPFR_IS_PURE_FP (y) && mpfr_cmp_str1 (y, "0.5") == 0 &&
      ! ((MPFR_IS_ZERO (x) || MPFR_IS_INF (x)) && MPFR_IS_NEG (x)))
    {
      mpfr_clear_flags ();
      inex2 = mpfr_sqrt (z2, x, rnd);
      cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
              s, "mpfr_sqrt, flags cleared");
      __gmpfr_flags = MPFR_FLAGS_ALL;
      inex2 = mpfr_sqrt (z2, x, rnd);
      cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
              s, "mpfr_sqrt, flags set");
    }

#if MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0)
  /* If y = -0.5, we can test mpfr_rec_sqrt, except if x = -Inf
     (because the rule for mpfr_pow on -Inf is different). */
  if (MPFR_IS_PURE_FP (y) && mpfr_cmp_str1 (y, "-0.5") == 0 &&
      ! (MPFR_IS_INF (x) && MPFR_IS_NEG (x)))
    {
      mpfr_clear_flags ();
      inex2 = mpfr_rec_sqrt (z2, x, rnd);
      cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
              s, "mpfr_rec_sqrt, flags cleared");
      __gmpfr_flags = MPFR_FLAGS_ALL;
      inex2 = mpfr_rec_sqrt (z2, x, rnd);
      cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
              s, "mpfr_rec_sqrt, flags set");
    }
#endif

  /* If x is an integer that fits in an unsigned long and is not -0,
     we can test mpfr_ui_pow. */
  if (MPFR_IS_POS (x) && mpfr_integer_p (x) &&
      mpfr_fits_ulong_p (x, MPFR_RNDN))
    {
      unsigned long xx = mpfr_get_ui (x, MPFR_RNDN);

      mpfr_clear_flags ();
      inex2 = mpfr_ui_pow (z2, xx, y, rnd);
      cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
              s, "mpfr_ui_pow, flags cleared");
      __gmpfr_flags = MPFR_FLAGS_ALL;
      inex2 = mpfr_ui_pow (z2, xx, y, rnd);
      cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
              s, "mpfr_ui_pow, flags set");

      /* If x = 2, we can test mpfr_exp2. */
      if (xx == 2)
        {
          mpfr_clear_flags ();
          inex2 = mpfr_exp2 (z2, y, rnd);
          cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
                  s, "mpfr_exp2, flags cleared");
          __gmpfr_flags = MPFR_FLAGS_ALL;
          inex2 = mpfr_exp2 (z2, y, rnd);
          cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
                  s, "mpfr_exp2, flags set");
        }

      /* If x = 10, we can test mpfr_exp10. */
      if (xx == 10)
        {
          mpfr_clear_flags ();
          inex2 = mpfr_exp10 (z2, y, rnd);
          cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, flags,
                  s, "mpfr_exp10, flags cleared");
          __gmpfr_flags = MPFR_FLAGS_ALL;
          inex2 = mpfr_exp10 (z2, y, rnd);
          cmpres (spx, sx, sy, rnd, z1, inex1, z2, inex2, MPFR_FLAGS_ALL,
                  s, "mpfr_exp10, flags set");
        }
    }

  mpfr_clear (z2);
}

static int
my_setstr (mpfr_ptr t, const char *s)
{
  if (strcmp (s, "min") == 0)
    {
      mpfr_setmin (t, mpfr_get_emin ());
      MPFR_SET_POS (t);
      return 0;
    }
  if (strcmp (s, "min+") == 0)
    {
      mpfr_setmin (t, mpfr_get_emin ());
      MPFR_SET_POS (t);
      mpfr_nextabove (t);
      return 0;
    }
  if (strcmp (s, "max") == 0)
    {
      mpfr_setmax (t, mpfr_get_emax ());
      MPFR_SET_POS (t);
      return 0;
    }
  return mpfr_set_str (t, s, 10, MPFR_RNDN);
}

static void
tst (void)
{
  int sv = sizeof (val) / sizeof (*val);
  int i, j;
  int rnd;
  mpfr_t x, y, z, tmp;

  mpfr_inits2 (53, x, y, z, tmp, (mpfr_ptr) 0);

  for (i = 0; i < sv; i++)
    for (j = 0; j < sv; j++)
      RND_LOOP (rnd)
        {
          int exact, inex;
          unsigned int flags;

          if (my_setstr (x, val[i]) || my_setstr (y, val[j]))
            {
              printf ("internal error for (%d,%d,%d)\n", i, j, rnd);
              exit (1);
            }
          mpfr_clear_flags ();
          inex = mpfr_pow (z, x, y, (mpfr_rnd_t) rnd);
          flags = __gmpfr_flags;
          if (! MPFR_IS_NAN (z) && mpfr_nanflag_p ())
            err ("got NaN flag without NaN value", i, j, rnd, z, inex);
          if (MPFR_IS_NAN (z) && ! mpfr_nanflag_p ())
            err ("got NaN value without NaN flag", i, j, rnd, z, inex);
          if (inex != 0 && ! mpfr_inexflag_p ())
            err ("got non-zero ternary value without inexact flag",
                 i, j, rnd, z, inex);
          if (inex == 0 && mpfr_inexflag_p ())
            err ("got null ternary value with inexact flag",
                 i, j, rnd, z, inex);
          if (i >= 3 && j >= 3)
            {
              if (mpfr_underflow_p ())
                err ("got underflow", i, j, rnd, z, inex);
              if (mpfr_overflow_p ())
                err ("got overflow", i, j, rnd, z, inex);
              exact = MPFR_IS_SINGULAR (z) ||
                (mpfr_mul_2ui (tmp, z, 16, MPFR_RNDN), mpfr_integer_p (tmp));
              if (exact && inex != 0)
                err ("got exact value with ternary flag different from 0",
                     i, j, rnd, z, inex);
              if (! exact && inex == 0)
                err ("got inexact value with ternary flag equal to 0",
                     i, j, rnd, z, inex);
            }
          if (MPFR_IS_ZERO (x) && ! MPFR_IS_NAN (y) && MPFR_NOTZERO (y))
            {
              if (MPFR_IS_NEG (y) && ! MPFR_IS_INF (z))
                err ("expected an infinity", i, j, rnd, z, inex);
              if (MPFR_IS_POS (y) && ! MPFR_IS_ZERO (z))
                err ("expected a zero", i, j, rnd, z, inex);
              if ((MPFR_IS_NEG (x) && is_odd (y)) ^ MPFR_IS_NEG (z))
                err ("wrong sign", i, j, rnd, z, inex);
            }
          if (! MPFR_IS_NAN (x) && mpfr_cmp_si (x, -1) == 0)
            {
              /* x = -1 */
              if (! (MPFR_IS_INF (y) || mpfr_integer_p (y)) &&
                  ! MPFR_IS_NAN (z))
                err ("expected NaN", i, j, rnd, z, inex);
              if ((MPFR_IS_INF (y) || (mpfr_integer_p (y) && ! is_odd (y)))
                  && ! mpfr_equal_p (z, __gmpfr_one))
                err ("expected 1", i, j, rnd, z, inex);
              if (is_odd (y) &&
                  (MPFR_IS_NAN (z) || mpfr_cmp_si (z, -1) != 0))
                err ("expected -1", i, j, rnd, z, inex);
            }
          if ((mpfr_equal_p (x, __gmpfr_one) || MPFR_IS_ZERO (y)) &&
              ! mpfr_equal_p (z, __gmpfr_one))
            err ("expected 1", i, j, rnd, z, inex);
          if (MPFR_IS_PURE_FP (x) && MPFR_IS_NEG (x) &&
              MPFR_IS_FP (y) && ! mpfr_integer_p (y) &&
              ! MPFR_IS_NAN (z))
            err ("expected NaN", i, j, rnd, z, inex);
          if (MPFR_IS_INF (y) && MPFR_NOTZERO (x))
            {
              int cmpabs1 = mpfr_cmpabs (x, __gmpfr_one);

              if ((MPFR_IS_NEG (y) ? (cmpabs1 < 0) : (cmpabs1 > 0)) &&
                  ! (MPFR_IS_POS (z) && MPFR_IS_INF (z)))
                err ("expected +Inf", i, j, rnd, z, inex);
              if ((MPFR_IS_NEG (y) ? (cmpabs1 > 0) : (cmpabs1 < 0)) &&
                  ! (MPFR_IS_POS (z) && MPFR_IS_ZERO (z)))
                err ("expected +0", i, j, rnd, z, inex);
            }
          if (MPFR_IS_INF (x) && ! MPFR_IS_NAN (y) && MPFR_NOTZERO (y))
            {
              if (MPFR_IS_POS (y) && ! MPFR_IS_INF (z))
                err ("expected an infinity", i, j, rnd, z, inex);
              if (MPFR_IS_NEG (y) && ! MPFR_IS_ZERO (z))
                err ("expected a zero", i, j, rnd, z, inex);
              if ((MPFR_IS_NEG (x) && is_odd (y)) ^ MPFR_IS_NEG (z))
                err ("wrong sign", i, j, rnd, z, inex);
            }
          test_others (val[i], val[j], (mpfr_rnd_t) rnd, x, y, z, inex, flags,
                       "tst");
        }
  mpfr_clears (x, y, z, tmp, (mpfr_ptr) 0);
}

static void
underflow_up1 (void)
{
  mpfr_t delta, x, y, z, z0;
  mpfr_exp_t n;
  int inex;
  int rnd;
  int i;

  n = mpfr_get_emin ();
  if (n < LONG_MIN)
    return;

  mpfr_init2 (delta, 2);
  inex = mpfr_set_ui_2exp (delta, 1, -2, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);

  mpfr_init2 (x, 8);
  inex = mpfr_set_ui (x, 2, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);

  mpfr_init2 (y, sizeof (long) * CHAR_BIT + 4);
  inex = mpfr_set_si (y, n, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);

  mpfr_init2 (z0, 2);
  mpfr_set_ui (z0, 0, MPFR_RNDN);

  mpfr_init2 (z, 32);

  for (i = 0; i <= 12; i++)
    {
      unsigned int flags = 0;
      char sy[256];  /* larger than needed, for maintainability */

      /* Test 2^(emin - i/4).
       * --> Underflow iff i > 4.
       * --> Zero in MPFR_RNDN iff i >= 8.
       */

      if (i != 0 && i != 4)
        flags |= MPFR_FLAGS_INEXACT;
      if (i > 4)
        flags |= MPFR_FLAGS_UNDERFLOW;

      sprintf (sy, "emin - %d/4", i);

      RND_LOOP (rnd)
        {
          int zero;

          zero = (i > 4 && (rnd == MPFR_RNDZ || rnd == MPFR_RNDD)) ||
            (i >= 8 && rnd == MPFR_RNDN);

          mpfr_clear_flags ();
          inex = mpfr_pow (z, x, y, (mpfr_rnd_t) rnd);
          cmpres (1, "2", sy, (mpfr_rnd_t) rnd, zero ? z0 : (mpfr_ptr) NULL,
                  -1, z, inex, flags, "underflow_up1", "mpfr_pow");
          test_others ("2", sy, (mpfr_rnd_t) rnd, x, y, z, inex, flags,
                       "underflow_up1");
        }

      inex = mpfr_sub (y, y, delta, MPFR_RNDN);
      MPFR_ASSERTN (inex == 0);
    }

  mpfr_clears (delta, x, y, z, z0, (mpfr_ptr) 0);
}

/* With pow.c r5497, the following test fails on a 64-bit Linux machine
 * due to a double-rounding problem when rescaling the result:
 *   Error with underflow_up2 and extended exponent range
 *   x = 7.fffffffffffffff0@-1,
 *   y = 4611686018427387904, MPFR_RNDN
 *   Expected 1.0000000000000000@-1152921504606846976, inex = 1, flags = 9
 *   Got      0, inex = -1, flags = 9
 * With pow_ui.c r5423, the following test fails on a 64-bit Linux machine
 * as underflows and overflows are not handled correctly (the approximation
 * error is ignored):
 *   Error with mpfr_pow_ui, flags cleared
 *   x = 7.fffffffffffffff0@-1,
 *   y = 4611686018427387904, MPFR_RNDN
 *   Expected 1.0000000000000000@-1152921504606846976, inex = 1, flags = 9
 *   Got      0, inex = -1, flags = 9
 */
static void
underflow_up2 (void)
{
  mpfr_t x, y, z, z0, eps;
  mpfr_exp_t n;
  int inex;
  int rnd;

  n = 1 - mpfr_get_emin ();
  MPFR_ASSERTN (n > 1);
  if (n > ULONG_MAX)
    return;

  mpfr_init2 (eps, 2);
  mpfr_set_ui_2exp (eps, 1, -1, MPFR_RNDN);  /* 1/2 */
  mpfr_div_ui (eps, eps, n, MPFR_RNDZ);      /* 1/(2n) rounded toward zero */

  mpfr_init2 (x, sizeof (unsigned long) * CHAR_BIT + 1);
  inex = mpfr_ui_sub (x, 1, eps, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);  /* since n < 2^(size_of_long_in_bits) */
  inex = mpfr_div_2ui (x, x, 1, MPFR_RNDN);  /* 1/2 - eps/2 exactly */
  MPFR_ASSERTN (inex == 0);

  mpfr_init2 (y, sizeof (unsigned long) * CHAR_BIT);
  inex = mpfr_set_ui (y, n, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);

  /* 0 < eps < 1 / (2n), thus (1 - eps)^n > 1/2,
     and 1/2 (1/2)^n < (1/2 - eps/2)^n < (1/2)^n. */
  mpfr_inits2 (64, z, z0, (mpfr_ptr) 0);
  RND_LOOP (rnd)
    {
      unsigned int ufinex = MPFR_FLAGS_UNDERFLOW | MPFR_FLAGS_INEXACT;
      int expected_inex;
      char sy[256];

      mpfr_set_ui (z0, 0, MPFR_RNDN);
      expected_inex = rnd == MPFR_RNDN || rnd == MPFR_RNDU || rnd == MPFR_RNDA ?
        (mpfr_nextabove (z0), 1) : -1;
      sprintf (sy, "%lu", (unsigned long) n);

      mpfr_clear_flags ();
      inex = mpfr_pow (z, x, y, (mpfr_rnd_t) rnd);
      cmpres (0, x, sy, (mpfr_rnd_t) rnd, z0, expected_inex, z, inex, ufinex,
              "underflow_up2", "mpfr_pow");
      test_others (NULL, sy, (mpfr_rnd_t) rnd, x, y, z, inex, ufinex,
                   "underflow_up2");
    }

  mpfr_clears (x, y, z, z0, eps, (mpfr_ptr) 0);
}

static void
underflow_up3 (void)
{
  mpfr_t x, y, z, z0;
  int inex;
  int rnd;
  int i;

  mpfr_init2 (x, 64);
  mpfr_init2 (y, sizeof (mpfr_exp_t) * CHAR_BIT);
  mpfr_init2 (z, 32);
  mpfr_init2 (z0, 2);

  inex = mpfr_set_exp_t (y, mpfr_get_emin () - 2, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);
  for (i = -1; i <= 1; i++)
    RND_LOOP (rnd)
      {
        unsigned int ufinex = MPFR_FLAGS_UNDERFLOW | MPFR_FLAGS_INEXACT;
        int expected_inex;

        mpfr_set_ui (x, 2, MPFR_RNDN);
        if (i < 0)
          mpfr_nextbelow (x);
        if (i > 0)
          mpfr_nextabove (x);
        /* x = 2 + i * eps, y = emin - 2, x^y ~= 2^(emin - 2) */

        expected_inex = rnd == MPFR_RNDU || rnd == MPFR_RNDA
          || (rnd == MPFR_RNDN && i < 0) ? 1 : -1;

        mpfr_set_ui (z0, 0, MPFR_RNDN);
        if (expected_inex > 0)
          mpfr_nextabove (z0);

        mpfr_clear_flags ();
        inex = mpfr_pow (z, x, y, (mpfr_rnd_t) rnd);
        cmpres (0, x, "emin - 2", (mpfr_rnd_t) rnd, z0, expected_inex, z, inex,
                ufinex, "underflow_up3", "mpfr_pow");
        test_others (NULL, "emin - 2", (mpfr_rnd_t) rnd, x, y, z, inex, ufinex,
                     "underflow_up3");
      }

  mpfr_clears (x, y, z, z0, (mpfr_ptr) 0);
}

static void
underflow_up (void)
{
  underflow_up1 ();
  underflow_up2 ();
  underflow_up3 ();
}

static void
overflow_inv (void)
{
  mpfr_t x, y, z;
  int precx;
  int s, t;
  int inex;
  int rnd;

  mpfr_init2 (y, 2);
  mpfr_init2 (z, 8);

  mpfr_set_si (y, -1, MPFR_RNDN);
  for (precx = 10; precx <= 100; precx += 90)
    {
      const char *sp = precx == 10 ?
        "overflow_inv (precx = 10)" : "overflow_inv (precx = 100)";

      mpfr_init2 (x, precx);
      for (s = -1; s <= 1; s += 2)
        {
          inex = mpfr_set_si_2exp (x, s, - mpfr_get_emax (), MPFR_RNDN);
          MPFR_ASSERTN (inex == 0);
          for (t = 0; t <= 5; t++)
            {
              /* If precx = 10:
               * x = s * 2^(-emax) * (1 + t * 2^(-9)), so that
               * 1/x = s * 2^emax * (1 - t * 2^(-9) + eps) with eps > 0.
               * Values of (1/x) / 2^emax and overflow condition for x > 0:
               * t = 0: 1                           o: always
               * t = 1: 0.11111111 100000000011...  o: MPFR_RNDN and MPFR_RNDU
               * t = 2: 0.11111111 000000001111...  o: MPFR_RNDU
               * t = 3: 0.11111110 100000100011...  o: never
               *
               * If precx = 100:
               * t = 0: always overflow
               * t > 0: overflow for MPFR_RNDN and MPFR_RNDU.
               */
              RND_LOOP (rnd)
                {
                  int inf, overflow;
                  mpfr_rnd_t rnd2;

                  if (rnd == MPFR_RNDA)
                    rnd2 = s < 0 ? MPFR_RNDD : MPFR_RNDU;
                  else
                    rnd2 = (mpfr_rnd_t) rnd;

                  overflow = t == 0 ||
                    ((mpfr_rnd_t) rnd == MPFR_RNDN &&
                     (precx > 10 || t == 1)) ||
                    (rnd2 == (s < 0 ? MPFR_RNDD : MPFR_RNDU) &&
                     (precx > 10 || t <= 2));
                  inf = overflow &&
                    ((mpfr_rnd_t) rnd == MPFR_RNDN ||
                     rnd2 == (s < 0 ? MPFR_RNDD : MPFR_RNDU));
                  mpfr_clear_flags ();
                  inex = mpfr_pow (z, x, y, (mpfr_rnd_t) rnd);
                  if (overflow ^ !! mpfr_overflow_p ())
                    {
                      printf ("Bad overflow flag in %s\nfor mpfr_pow%s\n"
                              "s = %d, t = %d, %s\n", sp,
                              ext ? ", extended exponent range" : "",
                              s, t, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                      exit (1);
                    }
                  if (overflow && (inf ^ !! MPFR_IS_INF (z)))
                    {
                      printf ("Bad value in %s\nfor mpfr_pow%s\n"
                              "s = %d, t = %d, %s\nGot ", sp,
                              ext ? ", extended exponent range" : "",
                              s, t, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                      mpfr_out_str (stdout, 16, 0, z, MPFR_RNDN);
                      printf (" instead of %s value.\n",
                              inf ? "infinite" : "finite");
                      exit (1);
                    }
                  test_others (NULL, "-1", (mpfr_rnd_t) rnd, x, y, z,
                               inex, __gmpfr_flags, sp);
                }  /* RND_LOOP */
              mpfr_nexttoinf (x);
            }  /* for (t = ...) */
        }  /* for (s = ...) */
      mpfr_clear (x);
    }  /* for (precx = ...) */

  mpfr_clears (y, z, (mpfr_ptr) 0);
}

static void
alltst (void)
{
  mpfr_exp_t emin, emax;

  ext = 0;
  tst ();
  underflow_up ();
  overflow_inv ();

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();
  set_emin (MPFR_EMIN_MIN);
  set_emax (MPFR_EMAX_MAX);
  if (mpfr_get_emin () != emin || mpfr_get_emax () != emax)
    {
      ext = 1;
      tst ();
      underflow_up ();
      overflow_inv ();
      set_emin (emin);
      set_emax (emax);
    }
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();
  all_cmpres_errors = argc > 1;
  alltst ();
  tests_end_mpfr ();
  return all_cmpres_errors < 0;
}
