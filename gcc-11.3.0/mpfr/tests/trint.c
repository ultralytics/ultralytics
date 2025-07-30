/* Test file for mpfr_rint, mpfr_trunc, mpfr_floor, mpfr_ceil, mpfr_round,
   mpfr_rint_trunc, mpfr_rint_floor, mpfr_rint_ceil, mpfr_rint_round.

Copyright 2002-2017 Free Software Foundation, Inc.
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

#include "mpfr-test.h"

#if __MPFR_STDC (199901L)
# include <math.h>
#endif

static void
special (void)
{
  mpfr_t x, y;
  mpfr_exp_t emax;

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_nan (x);
  mpfr_rint (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (y));

  mpfr_set_inf (x, 1);
  mpfr_rint (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && mpfr_sgn (y) > 0);

  mpfr_set_inf (x, -1);
  mpfr_rint (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (y) && mpfr_sgn (y) < 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_rint (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 0) == 0 && MPFR_IS_POS(y));

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_rint (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 0) == 0 && MPFR_IS_NEG(y));

  /* coverage test */
  mpfr_set_prec (x, 2);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_mul_2exp (x, x, mp_bits_per_limb, MPFR_RNDN);
  mpfr_rint (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp (y, x) == 0);

  /* another coverage test */
  emax = mpfr_get_emax ();
  set_emax (1);
  mpfr_set_prec (x, 3);
  mpfr_set_str_binary (x, "1.11E0");
  mpfr_set_prec (y, 2);
  mpfr_rint (y, x, MPFR_RNDU); /* x rounds to 1.0E1=0.1E2 which overflows */
  MPFR_ASSERTN(mpfr_inf_p (y) && mpfr_sgn (y) > 0);
  set_emax (emax);

  /* yet another */
  mpfr_set_prec (x, 97);
  mpfr_set_prec (y, 96);
  mpfr_set_str_binary (x, "-0.1011111001101111000111011100011100000110110110110000000111010001000101001111101010101011010111100E97");
  mpfr_rint (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp (y, x) == 0);

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_str_binary (x, "0.10101100000000101001010101111111000000011111010000010E-1");
  mpfr_rint (y, x, MPFR_RNDU);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 1) == 0);
  mpfr_rint (y, x, MPFR_RNDD);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 0) == 0 && MPFR_IS_POS(y));

  mpfr_set_prec (x, 36);
  mpfr_set_prec (y, 2);
  mpfr_set_str_binary (x, "-11000110101010111111110111001.0000100");
  mpfr_rint (y, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "-11E27");
  MPFR_ASSERTN(mpfr_cmp (y, x) == 0);

  mpfr_set_prec (x, 39);
  mpfr_set_prec (y, 29);
  mpfr_set_str_binary (x, "-0.100010110100011010001111001001001100111E39");
  mpfr_rint (y, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "-0.10001011010001101000111100101E39");
  MPFR_ASSERTN(mpfr_cmp (y, x) == 0);

  mpfr_set_prec (x, 46);
  mpfr_set_prec (y, 32);
  mpfr_set_str_binary (x, "-0.1011100110100101000001011111101011001001101001E32");
  mpfr_rint (y, x, MPFR_RNDN);
  mpfr_set_str_binary (x, "-0.10111001101001010000010111111011E32");
  MPFR_ASSERTN(mpfr_cmp (y, x) == 0);

  /* coverage test for mpfr_round */
  mpfr_set_prec (x, 3);
  mpfr_set_str_binary (x, "1.01E1"); /* 2.5 */
  mpfr_set_prec (y, 2);
  mpfr_round (y, x);
  /* since mpfr_round breaks ties away, should give 3 and not 2 as with
     the "round to even" rule */
  MPFR_ASSERTN(mpfr_cmp_ui (y, 3) == 0);
  /* same test for the function */
  (mpfr_round) (y, x);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 3) == 0);

  mpfr_set_prec (x, 6);
  mpfr_set_prec (y, 3);
  mpfr_set_str_binary (x, "110.111");
  mpfr_round (y, x);
  if (mpfr_cmp_ui (y, 7))
    {
      printf ("Error in round(110.111)\n");
      exit (1);
    }

  /* Bug found by  Mark J Watkins */
  mpfr_set_prec (x, 84);
  mpfr_set_str_binary (x,
   "0.110011010010001000000111101101001111111100101110010000000000000" \
                       "000000000000000000000E32");
  mpfr_round (x, x);
  if (mpfr_cmp_str (x, "0.1100110100100010000001111011010100000000000000" \
                    "00000000000000000000000000000000000000E32", 2, MPFR_RNDN))
    {
      printf ("Rounding error when dest=src\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

#define BASIC_TEST(F,J)                                                 \
  do                                                                    \
    {                                                                   \
      int red;                                                          \
      for (red = 0; red <= 1; red++)                                    \
        {                                                               \
          int inex1, inex2;                                             \
          unsigned int ex_flags, flags;                                 \
                                                                        \
          if (red)                                                      \
            {                                                           \
              set_emin (e);                                             \
              set_emax (e);                                             \
            }                                                           \
                                                                        \
          mpfr_clear_flags ();                                          \
          inex1 = mpfr_set_si (y, J, (mpfr_rnd_t) r);                   \
          ex_flags = __gmpfr_flags;                                     \
          mpfr_clear_flags ();                                          \
          inex2 = mpfr_rint_##F (z, x, (mpfr_rnd_t) r);                 \
          flags = __gmpfr_flags;                                        \
          if (! (mpfr_equal_p (y, z) &&                                 \
                 SAME_SIGN (inex1, inex2) &&                            \
                 flags == ex_flags))                                    \
            {                                                           \
              printf ("Basic test failed on mpfr_rint_" #F              \
                      ", prec = %d, i = %d, %s\n", prec, s * i,         \
                      mpfr_print_rnd_mode ((mpfr_rnd_t) r));            \
              printf ("i.e. x = ");                                     \
              mpfr_dump (x);                                            \
              if (red)                                                  \
                printf ("with emin = emax = %d\n", e);                  \
              printf ("Expected ");                                     \
              mpfr_dump (y);                                            \
              printf ("with inex = %d (or equivalent)\n", inex1);       \
              printf ("     flags:");                                   \
              flags_out (ex_flags);                                     \
              printf ("Got      ");                                     \
              mpfr_dump (z);                                            \
              printf ("with inex = %d (or equivalent)\n", inex2);       \
              printf ("     flags:");                                   \
              flags_out (flags);                                        \
              exit (1);                                                 \
            }                                                           \
        }                                                               \
      set_emin (emin);                                                  \
      set_emax (emax);                                                  \
    }                                                                   \
  while (0)

#define BASIC_TEST2(F,J,INEX)                                   \
  do                                                            \
    {                                                           \
      int red;                                                  \
      for (red = 0; red <= 1; red++)                            \
        {                                                       \
          int inex;                                             \
          unsigned int ex_flags, flags;                         \
                                                                \
          if (red)                                              \
            {                                                   \
              set_emin (e);                                     \
              set_emax (e);                                     \
            }                                                   \
                                                                \
          mpfr_clear_flags ();                                  \
          inex = mpfr_set_si (y, J, MPFR_RNDN);                 \
          MPFR_ASSERTN (inex == 0 || mpfr_overflow_p ());       \
          ex_flags = __gmpfr_flags;                             \
          mpfr_clear_flags ();                                  \
          inex = mpfr_##F (z, x);                               \
          if (inex != 0)                                        \
            ex_flags |= MPFR_FLAGS_INEXACT;                     \
          flags = __gmpfr_flags;                                \
          if (! (mpfr_equal_p (y, z) &&                         \
                 inex == (INEX) &&                              \
                 flags == ex_flags))                            \
            {                                                   \
              printf ("Basic test failed on mpfr_" #F           \
                      ", prec = %d, i = %d\n", prec, s * i);    \
              printf ("i.e. x = ");                             \
              mpfr_dump (x);                                    \
              if (red)                                          \
                printf ("with emin = emax = %d\n", e);          \
              printf ("Expected ");                             \
              mpfr_dump (y);                                    \
              printf ("with inex = %d\n", (INEX));              \
              printf ("     flags:");                           \
              flags_out (ex_flags);                             \
              printf ("Got      ");                             \
              mpfr_dump (z);                                    \
              printf ("with inex = %d\n", inex);                \
              printf ("     flags:");                           \
              flags_out (flags);                                \
              exit (1);                                         \
            }                                                   \
        }                                                       \
      set_emin (emin);                                          \
      set_emax (emax);                                          \
    }                                                           \
  while (0)

/* Test mpfr_rint_* on i/4 with |i| between 1 and 72. */
static void
basic_tests (void)
{
  mpfr_t x, y, z;
  int prec, s, i, r;
  mpfr_exp_t emin, emax;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  mpfr_init2 (x, 16);
  for (prec = 2; prec <= 7; prec++)
    {
      mpfr_inits2 (prec, y, z, (mpfr_ptr) 0);
      for (s = 1; s >= -1; s -= 2)
        for (i = 1; i <= 72; i++)
          {
            int k, t, u, v, f, e;

            for (t = i/4, k = 0; t >= 1 << prec; t >>= 1, k++)
              ;
            t <<= k;
            for (u = (i+3)/4, k = 0; u >= 1 << prec; u = (u+1)/2, k++)
              ;
            u <<= k;
            v = i < (t+u) << 1 ? t : u;
            f = t == u ? 0 : i % 4 == 0 ? 1 : 2;

            mpfr_set_si_2exp (x, s * i, -2, MPFR_RNDN);
            e = mpfr_get_exp (x);
            RND_LOOP(r)
              {
                BASIC_TEST (trunc, s * (i/4));
                BASIC_TEST (floor, s > 0 ? i/4 : - ((i+3)/4));
                BASIC_TEST (ceil, s > 0 ? (i+3)/4 : - (i/4));
                BASIC_TEST (round, s * ((i+2)/4));
              }
            BASIC_TEST2 (trunc, s * t, - s * f);
            BASIC_TEST2 (floor, s > 0 ? t : - u, - f);
            BASIC_TEST2 (ceil, s > 0 ? u : - t, f);
            BASIC_TEST2 (round, s * v, v == t ? - s * f : s * f);
          }
      mpfr_clears (y, z, (mpfr_ptr) 0);
    }
  mpfr_clear (x);
}

#if __MPFR_STDC (199901L)

static void
test_fct (double (*f)(double), int (*g)(), char *s, mpfr_rnd_t r)
{
  double d, y;
  mpfr_t dd, yy;

  mpfr_init2 (dd, 53);
  mpfr_init2 (yy, 53);
  for (d = -5.0; d <= 5.0; d += 0.25)
    {
      mpfr_set_d (dd, d, r);
      y = (*f)(d);
      if (g == &mpfr_rint)
        mpfr_rint (yy, dd, r);
      else
        (*g)(yy, dd);
      mpfr_set_d (dd, y, r);
      if (mpfr_cmp (yy, dd))
        {
          printf ("test_against_libc: incorrect result for %s, rnd = %s,"
                  " d = %g\ngot ", s, mpfr_print_rnd_mode (r), d);
          mpfr_out_str (stdout, 10, 0, yy, MPFR_RNDN);
          printf (" instead of %g\n", y);
          exit (1);
        }
    }
  mpfr_clear (dd);
  mpfr_clear (yy);
}

#define TEST_FCT(F) test_fct (&F, &mpfr_##F, #F, r)

static void
test_against_libc (void)
{
  mpfr_rnd_t r = MPFR_RNDN;

  (void) r;  /* avoid a warning by using r */
#if HAVE_ROUND
  TEST_FCT (round);
#endif
#if HAVE_TRUNC
  TEST_FCT (trunc);
#endif
#if HAVE_FLOOR
  TEST_FCT (floor);
#endif
#if HAVE_CEIL
  TEST_FCT (ceil);
#endif
#if HAVE_NEARBYINT
  for (r = 0; r < MPFR_RND_MAX ; r++)
    if (mpfr_set_machine_rnd_mode (r) == 0)
      test_fct (&nearbyint, &mpfr_rint, "rint", r);
#endif
}

#endif

static void
err (const char *str, mp_size_t s, mpfr_t x, mpfr_t y, mpfr_prec_t p,
     mpfr_rnd_t r, int trint, int inexact)
{
  printf ("Error: %s\ns = %u, p = %u, r = %s, trint = %d, inexact = %d\nx = ",
          str, (unsigned int) s, (unsigned int) p, mpfr_print_rnd_mode (r),
          trint, inexact);
  mpfr_print_binary (x);
  printf ("\ny = ");
  mpfr_print_binary (y);
  printf ("\n");
  exit (1);
}

static void
coverage_03032011 (void)
{
  mpfr_t in, out, cmp;
  int status;
  int precIn;
  char strData[(GMP_NUMB_BITS * 4)+256];

  precIn = GMP_NUMB_BITS * 4;

  mpfr_init2 (in, precIn);
  mpfr_init2 (out, GMP_NUMB_BITS);
  mpfr_init2 (cmp, GMP_NUMB_BITS);

  /* cmp = "0.1EprecIn+2" */
  /* The buffer size is sufficient, as precIn is small in practice. */
  sprintf (strData, "0.1E%d", precIn+2);
  mpfr_set_str_binary (cmp, strData);

  /* in = "0.10...01EprecIn+2" use all (precIn) significand bits */
  memset ((void *)strData, '0', precIn+2);
  strData[1] = '.';
  strData[2] = '1';
  sprintf (&strData[precIn+1], "1E%d", precIn+2);
  mpfr_set_str_binary (in, strData);

  status = mpfr_rint (out, in, MPFR_RNDN);
  if ((mpfr_cmp (out, cmp) != 0) || (status >= 0))
    {
      printf("mpfr_rint error :\n status is %d instead of 0\n", status);
      printf(" out value is ");
      mpfr_dump(out);
      printf(" instead of   ");
      mpfr_dump(cmp);
      exit (1);
    }

  mpfr_clear (cmp);
  mpfr_clear (out);

  mpfr_init2 (out, GMP_NUMB_BITS);
  mpfr_init2 (cmp, GMP_NUMB_BITS);

  /* cmp = "0.10...01EprecIn+2" use all (GMP_NUMB_BITS) significand bits */
  strcpy (&strData[GMP_NUMB_BITS+1], &strData[precIn+1]);
  mpfr_set_str_binary (cmp, strData);

  (MPFR_MANT(in))[2] = MPFR_LIMB_HIGHBIT;
  status = mpfr_rint (out, in, MPFR_RNDN);

  if ((mpfr_cmp (out, cmp) != 0) || (status <= 0))
    {
      printf("mpfr_rint error :\n status is %d instead of 0\n", status);
      printf(" out value is\n");
      mpfr_dump(out);
      printf(" instead of\n");
      mpfr_dump(cmp);
      exit (1);
    }

  mpfr_clear (cmp);
  mpfr_clear (out);
  mpfr_clear (in);
}

#define TEST_FUNCTION mpfr_rint_trunc
#define TEST_RANDOM_EMIN -20
#define TEST_RANDOM_ALWAYS_SCALE 1
#define test_generic test_generic_trunc
#include "tgeneric.c"

#define TEST_FUNCTION mpfr_rint_floor
#define TEST_RANDOM_EMIN -20
#define TEST_RANDOM_ALWAYS_SCALE 1
#define test_generic test_generic_floor
#include "tgeneric.c"

#define TEST_FUNCTION mpfr_rint_ceil
#define TEST_RANDOM_EMIN -20
#define TEST_RANDOM_ALWAYS_SCALE 1
#define test_generic test_generic_ceil
#include "tgeneric.c"

#define TEST_FUNCTION mpfr_rint_round
#define TEST_RANDOM_EMIN -20
#define TEST_RANDOM_ALWAYS_SCALE 1
#define test_generic test_generic_round
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  mp_size_t s;
  mpz_t z;
  mpfr_prec_t p;
  mpfr_t x, y, t, u, v;
  int r;
  int inexact, sign_t;

  tests_start_mpfr ();

  mpfr_init (x);
  mpfr_init (y);
  mpz_init (z);
  mpfr_init (t);
  mpfr_init (u);
  mpfr_init (v);
  mpz_set_ui (z, 1);
  for (s = 2; s < 100; s++)
    {
      /* z has exactly s bits */

      mpz_mul_2exp (z, z, 1);
      if (randlimb () % 2)
        mpz_add_ui (z, z, 1);
      mpfr_set_prec (x, s);
      mpfr_set_prec (t, s);
      mpfr_set_prec (u, s);
      if (mpfr_set_z (x, z, MPFR_RNDN))
        {
          printf ("Error: mpfr_set_z should be exact (s = %u)\n",
                  (unsigned int) s);
          exit (1);
        }
      if (randlimb () % 2)
        mpfr_neg (x, x, MPFR_RNDN);
      if (randlimb () % 2)
        mpfr_div_2ui (x, x, randlimb () % s, MPFR_RNDN);
      for (p = 2; p < 100; p++)
        {
          int trint;
          mpfr_set_prec (y, p);
          mpfr_set_prec (v, p);
          for (r = 0; r < MPFR_RND_MAX ; r++)
            for (trint = 0; trint < 3; trint++)
              {
                if (trint == 2)
                  inexact = mpfr_rint (y, x, (mpfr_rnd_t) r);
                else if (r == MPFR_RNDN)
                  inexact = mpfr_round (y, x);
                else if (r == MPFR_RNDZ)
                  inexact = (trint ? mpfr_trunc (y, x) :
                             mpfr_rint_trunc (y, x, MPFR_RNDZ));
                else if (r == MPFR_RNDU)
                  inexact = (trint ? mpfr_ceil (y, x) :
                             mpfr_rint_ceil (y, x, MPFR_RNDU));
                else /* r = MPFR_RNDD */
                  inexact = (trint ? mpfr_floor (y, x) :
                             mpfr_rint_floor (y, x, MPFR_RNDD));
                if (mpfr_sub (t, y, x, MPFR_RNDN))
                  err ("subtraction 1 should be exact",
                       s, x, y, p, (mpfr_rnd_t) r, trint, inexact);
                sign_t = mpfr_cmp_ui (t, 0);
                if (trint != 0 &&
                    (((inexact == 0) && (sign_t != 0)) ||
                     ((inexact < 0) && (sign_t >= 0)) ||
                     ((inexact > 0) && (sign_t <= 0))))
                  err ("wrong inexact flag", s, x, y, p, (mpfr_rnd_t) r, trint, inexact);
                if (inexact == 0)
                  continue; /* end of the test for exact results */

                if (((r == MPFR_RNDD || (r == MPFR_RNDZ && MPFR_SIGN (x) > 0))
                     && inexact > 0) ||
                    ((r == MPFR_RNDU || (r == MPFR_RNDZ && MPFR_SIGN (x) < 0))
                     && inexact < 0))
                  err ("wrong rounding direction",
                       s, x, y, p, (mpfr_rnd_t) r, trint, inexact);
                if (inexact < 0)
                  {
                    mpfr_add_ui (v, y, 1, MPFR_RNDU);
                    if (mpfr_cmp (v, x) <= 0)
                      err ("representable integer between x and its "
                           "rounded value", s, x, y, p, (mpfr_rnd_t) r, trint, inexact);
                  }
                else
                  {
                    mpfr_sub_ui (v, y, 1, MPFR_RNDD);
                    if (mpfr_cmp (v, x) >= 0)
                      err ("representable integer between x and its "
                           "rounded value", s, x, y, p, (mpfr_rnd_t) r, trint, inexact);
                  }
                if (r == MPFR_RNDN)
                  {
                    int cmp;
                    if (mpfr_sub (u, v, x, MPFR_RNDN))
                      err ("subtraction 2 should be exact",
                           s, x, y, p, (mpfr_rnd_t) r, trint, inexact);
                    cmp = mpfr_cmp_abs (t, u);
                    if (cmp > 0)
                      err ("faithful rounding, but not the nearest integer",
                           s, x, y, p, (mpfr_rnd_t) r, trint, inexact);
                    if (cmp < 0)
                      continue;
                    /* |t| = |u|: x is the middle of two consecutive
                       representable integers. */
                    if (trint == 2)
                      {
                        /* halfway case for mpfr_rint in MPFR_RNDN rounding
                           mode: round to an even integer or significand. */
                        mpfr_div_2ui (y, y, 1, MPFR_RNDZ);
                        if (!mpfr_integer_p (y))
                          err ("halfway case for mpfr_rint, result isn't an"
                               " even integer", s, x, y, p, (mpfr_rnd_t) r, trint, inexact);
                        /* If floor(x) and ceil(x) aren't both representable
                           integers, the significand must be even. */
                        mpfr_sub (v, v, y, MPFR_RNDN);
                        mpfr_abs (v, v, MPFR_RNDN);
                        if (mpfr_cmp_ui (v, 1) != 0)
                          {
                            mpfr_div_2si (y, y, MPFR_EXP (y) - MPFR_PREC (y)
                                          + 1, MPFR_RNDN);
                            if (!mpfr_integer_p (y))
                              err ("halfway case for mpfr_rint, significand isn't"
                                   " even", s, x, y, p, (mpfr_rnd_t) r, trint, inexact);
                          }
                      }
                    else
                      { /* halfway case for mpfr_round: x must have been
                           rounded away from zero. */
                        if ((MPFR_SIGN (x) > 0 && inexact < 0) ||
                            (MPFR_SIGN (x) < 0 && inexact > 0))
                          err ("halfway case for mpfr_round, bad rounding"
                               " direction", s, x, y, p, (mpfr_rnd_t) r, trint, inexact);
                      }
                  }
              }
        }
    }
  mpfr_clear (x);
  mpfr_clear (y);
  mpz_clear (z);
  mpfr_clear (t);
  mpfr_clear (u);
  mpfr_clear (v);

  special ();
  basic_tests ();
  coverage_03032011 ();

  test_generic_trunc (2, 300, 20);
  test_generic_floor (2, 300, 20);
  test_generic_ceil (2, 300, 20);
  test_generic_round (2, 300, 20);

#if __MPFR_STDC (199901L)
  if (argc > 1 && strcmp (argv[1], "-s") == 0)
    test_against_libc ();
#endif

  tests_end_mpfr ();
  return 0;
}
