/* Generic test file for functions with one or two arguments (the second being
   either mpfr_t or double or unsigned long).

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

/* Define TWO_ARGS for two-argument functions like mpfr_pow.
   Define DOUBLE_ARG1 or DOUBLE_ARG2 for function with a double operand in
   first or second place like sub_d or d_sub.
   Define ULONG_ARG1 or ULONG_ARG2 for function with an unsigned long
   operand in first or second place like sub_ui or ui_sub. */

#if defined(TWO_ARGS) || defined(DOUBLE_ARG1) || defined(DOUBLE_ARG2) || \
  defined(ULONG_ARG1) || defined(ULONG_ARG2)
#define TWO_ARGS_ALL
#endif

#ifndef TEST_RANDOM_POS
/* For the random function: one number on two is negative. */
#define TEST_RANDOM_POS 256
#endif

#ifndef TEST_RANDOM_POS2
/* For the random function: one number on two is negative. */
#define TEST_RANDOM_POS2 256
#endif

#ifndef TEST_RANDOM_EMIN
#define TEST_RANDOM_EMIN -256
#endif

#ifndef TEST_RANDOM_EMAX
#define TEST_RANDOM_EMAX 255
#endif

#ifndef TEST_RANDOM_ALWAYS_SCALE
#define TEST_RANDOM_ALWAYS_SCALE 0
#endif

/* If the MPFR_SUSPICIOUS_OVERFLOW test fails but this is not a bug,
   then define TGENERIC_SO_TEST with an adequate test (possibly 0) to
   omit this particular case. */
#ifndef TGENERIC_SO_TEST
#define TGENERIC_SO_TEST 1
#endif

#define STR(F) #F
#define MAKE_STR(S) STR(S)

/* The (void *) below is needed to avoid a warning with gcc 4.2+ and functions
 * with 2 arguments. See <http://gcc.gnu.org/bugzilla/show_bug.cgi?id=36299>.
 */
#define TGENERIC_FAIL(S, X, U)                                          \
  do                                                                    \
    {                                                                   \
      printf ("tgeneric: %s\nx = ", (S));                               \
      mpfr_dump (X);;                                                   \
      if ((void *) (U) != 0)                                            \
        {                                                               \
          printf ("u = ");                                              \
          mpfr_dump (U);                                                \
        }                                                               \
      printf ("yprec = %u, rnd_mode = %s, inexact = %d\nflags =",       \
              (unsigned int) yprec, mpfr_print_rnd_mode (rnd),          \
              compare);                                                 \
      flags_out (flags);                                                \
      exit (1);                                                         \
    }                                                                   \
  while (0)

#define TGENERIC_CHECK_AUX(S, EXPR, U)                          \
  do                                                            \
    if (!(EXPR))                                                \
      TGENERIC_FAIL (S " for " MAKE_STR(TEST_FUNCTION), x, U);  \
  while (0)

#undef TGENERIC_CHECK
#if defined(TWO_ARGS_ALL)
#define TGENERIC_CHECK(S, EXPR) TGENERIC_CHECK_AUX(S, EXPR, u)
#else
#define TGENERIC_CHECK(S, EXPR) TGENERIC_CHECK_AUX(S, EXPR, 0)
#endif

#ifdef DEBUG_TGENERIC
#define TGENERIC_IAUX(F,P,X,U)                                          \
  do                                                                    \
    {                                                                   \
      printf ("tgeneric: testing function " STR(F)                      \
              ", %s, target prec = %lu\nx = ",                          \
              mpfr_print_rnd_mode (rnd), (unsigned long) (P));          \
      mpfr_dump (X);                                                    \
      if ((void *) (U) != 0)                                            \
        {                                                               \
          printf ("u = ");                                              \
          mpfr_dump (U);                                                \
        }                                                               \
    }                                                                   \
  while (0)
#undef TGENERIC_INFO
#if defined(TWO_ARGS_ALL)
#define TGENERIC_INFO(F,P) TGENERIC_IAUX(F,P,x,u)
#else
#define TGENERIC_INFO(F,P) TGENERIC_IAUX(F,P,x,0)
#endif
#endif  /* DEBUG_TGENERIC */

/* For some functions (for example cos), the argument reduction is too
   expensive when using mpfr_get_emax(). Then simply define REDUCE_EMAX
   to some reasonable value before including tgeneric.c. */
#ifndef REDUCE_EMAX
#define REDUCE_EMAX mpfr_get_emax ()
#endif

static void
test_generic (mpfr_prec_t p0, mpfr_prec_t p1, unsigned int nmax)
{
  mpfr_prec_t prec, xprec, yprec;
  mpfr_t x, y, z, t, w;
#if defined(TWO_ARGS_ALL)
  mpfr_t u;
#endif
#if defined(DOUBLE_ARG1) || defined(DOUBLE_ARG2)
  double d;
#endif
#if defined(ULONG_ARG1) || defined(ULONG_ARG2)
  unsigned long i;
#endif
  mpfr_rnd_t rnd;
  int inexact, compare, compare2;
  unsigned int n;
  unsigned long ctrt = 0, ctrn = 0;
  int test_of = 1, test_uf = 1;
  mpfr_exp_t old_emin, old_emax;

  old_emin = mpfr_get_emin ();
  old_emax = mpfr_get_emax ();

  mpfr_inits2 (MPFR_PREC_MIN, x, y, z, t, w, (mpfr_ptr) 0);
#if defined(TWO_ARGS_ALL)
  mpfr_init2 (u, MPFR_PREC_MIN);
#endif

  /* generic test */
  for (prec = p0; prec <= p1; prec++)
    {
      mpfr_set_prec (z, prec);
      mpfr_set_prec (t, prec);
      yprec = prec + 10;
      mpfr_set_prec (y, yprec);
      mpfr_set_prec (w, yprec);

      /* Note: in precision p1, we test 4 special cases. */
      for (n = 0; n < (prec == p1 ? nmax + 4 : nmax); n++)
        {
          int infinite_input = 0;
          unsigned int flags;
          mpfr_exp_t oemin, oemax;

          xprec = prec;
          if (randlimb () & 1)
            {
              /* In half cases, modify the precision of the inputs:
                 If the base precision (for the result) is small,
                 take a larger input precision in general, else
                 take a smaller precision. */
              xprec *= (prec < 16 ? 256.0 : 1.0) *
                (double) randlimb () / MP_LIMB_T_MAX;
              if (xprec < MPFR_PREC_MIN)
                xprec = MPFR_PREC_MIN;
            }
          mpfr_set_prec (x, xprec);
#if defined(TWO_ARGS)
          mpfr_set_prec (u, xprec);
#elif defined(DOUBLE_ARG1) || defined(DOUBLE_ARG2)
          mpfr_set_prec (u, IEEE_DBL_MANT_DIG);
#elif defined(ULONG_ARG1) || defined(ULONG_ARG2)
          mpfr_set_prec (u, sizeof (unsigned long) * CHAR_BIT);
#endif

          if (n > 3 || prec < p1)
            {
#if defined(RAND_FUNCTION)
              RAND_FUNCTION (x);
#if defined(TWO_ARGS) || defined(DOUBLE_ARG1) || defined(DOUBLE_ARG2)
              RAND_FUNCTION (u);
#endif
#else  /* ! defined(RAND_FUNCTION) */
              tests_default_random (x, TEST_RANDOM_POS,
                                    TEST_RANDOM_EMIN, TEST_RANDOM_EMAX,
                                    TEST_RANDOM_ALWAYS_SCALE);
#if defined(TWO_ARGS) || defined(DOUBLE_ARG1) || defined(DOUBLE_ARG2)
              tests_default_random (u, TEST_RANDOM_POS2,
                                    TEST_RANDOM_EMIN, TEST_RANDOM_EMAX,
                                    TEST_RANDOM_ALWAYS_SCALE);
#endif
#endif  /* ! defined(RAND_FUNCTION) */
            }
          else
            {
              /* Special cases tested in precision p1 if n <= 3. They are
                 useful really in the extended exponent range. */
#if (defined(DOUBLE_ARG1) || defined(DOUBLE_ARG2)) && defined(MPFR_ERRDIVZERO)
              goto next_n;
#endif
              set_emin (MPFR_EMIN_MIN);
              set_emax (MPFR_EMAX_MAX);
              if (n <= 1)
                {
                  mpfr_set_si (x, n == 0 ? 1 : -1, MPFR_RNDN);
                  mpfr_set_exp (x, mpfr_get_emin ());
#if defined(TWO_ARGS) || defined(DOUBLE_ARG1) || defined(DOUBLE_ARG2)
                  mpfr_set_si (u, randlimb () % 2 == 0 ? 1 : -1, MPFR_RNDN);
                  mpfr_set_exp (u, mpfr_get_emin ());
#endif
                }
              else  /* 2 <= n <= 3 */
                {
                  if (getenv ("MPFR_CHECK_MAX") == NULL)
                    goto next_n;
                  mpfr_set_si (x, n == 0 ? 1 : -1, MPFR_RNDN);
                  mpfr_setmax (x, REDUCE_EMAX);
#if defined(TWO_ARGS) || defined(DOUBLE_ARG1) || defined(DOUBLE_ARG2)
                  mpfr_set_si (u, randlimb () % 2 == 0 ? 1 : -1, MPFR_RNDN);
                  mpfr_setmax (u, mpfr_get_emax ());
#endif
                }
            }

#if defined(ULONG_ARG1) || defined(ULONG_ARG2)
          i = randlimb ();
          inexact = mpfr_set_ui (u, i, MPFR_RNDN);
          MPFR_ASSERTN (inexact == 0);
#endif

          /* Exponent range for the test. */
          oemin = mpfr_get_emin ();
          oemax = mpfr_get_emax ();

          rnd = RND_RAND ();
          mpfr_clear_flags ();
#ifdef DEBUG_TGENERIC
          TGENERIC_INFO (TEST_FUNCTION, MPFR_PREC (y));
#endif
#if defined(TWO_ARGS)
          compare = TEST_FUNCTION (y, x, u, rnd);
#elif defined(DOUBLE_ARG1)
          d = mpfr_get_d (u, rnd);
          compare = TEST_FUNCTION (y, d, x, rnd);
          /* d can be infinite due to overflow in mpfr_get_d */
          infinite_input |= DOUBLE_ISINF (d);
#elif defined(DOUBLE_ARG2)
          d = mpfr_get_d (u, rnd);
          compare = TEST_FUNCTION (y, x, d, rnd);
          /* d can be infinite due to overflow in mpfr_get_d */
          infinite_input |= DOUBLE_ISINF (d);
#elif defined(ULONG_ARG1)
          compare = TEST_FUNCTION (y, i, x, rnd);
#elif defined(ULONG_ARG2)
          compare = TEST_FUNCTION (y, x, i, rnd);
#else
          compare = TEST_FUNCTION (y, x, rnd);
#endif
          flags = __gmpfr_flags;
          if (mpfr_get_emin () != oemin ||
              mpfr_get_emax () != oemax)
            {
              printf ("tgeneric: the exponent range has been modified"
                      " by the tested function!\n");
              exit (1);
            }
          TGENERIC_CHECK ("bad inexact flag",
                          (compare != 0) ^ (mpfr_inexflag_p () == 0));
          ctrt++;

          /* Tests in a reduced exponent range. */
          {
            unsigned int oldflags = flags;
            mpfr_exp_t e, emin, emax;

            /* Determine the smallest exponent range containing the
               exponents of the mpfr_t inputs (x, and u if TWO_ARGS)
               and output (y). */
            emin = MPFR_EMAX_MAX;
            emax = MPFR_EMIN_MIN;
            if (MPFR_IS_PURE_FP (x))
              {
                e = MPFR_GET_EXP (x);
                if (e < emin)
                  emin = e;
                if (e > emax)
                  emax = e;
              }
#if defined(TWO_ARGS)
            if (MPFR_IS_PURE_FP (u))
              {
                e = MPFR_GET_EXP (u);
                if (e < emin)
                  emin = e;
                if (e > emax)
                  emax = e;
              }
#endif
            if (MPFR_IS_PURE_FP (y))
              {
                e = MPFR_GET_EXP (y);
                if (test_of && e - 1 >= emax)
                  {
                    unsigned int ex_flags;

                    mpfr_set_emax (e - 1);
                    mpfr_clear_flags ();
#if defined(TWO_ARGS)
                    inexact = TEST_FUNCTION (w, x, u, rnd);
#elif defined(DOUBLE_ARG1)
                    inexact = TEST_FUNCTION (w, d, x, rnd);
#elif defined(DOUBLE_ARG2)
                    inexact = TEST_FUNCTION (w, x, d, rnd);
#elif defined(ULONG_ARG1)
                    inexact = TEST_FUNCTION (w, i, x, rnd);
#elif defined(ULONG_ARG2)
                    inexact = TEST_FUNCTION (w, x, i, rnd);
#else
                    inexact = TEST_FUNCTION (w, x, rnd);
#endif
                    flags = __gmpfr_flags;
                    mpfr_set_emax (oemax);
                    ex_flags = MPFR_FLAGS_OVERFLOW | MPFR_FLAGS_INEXACT;
                    if (flags != ex_flags)
                      {
                        printf ("tgeneric: error for " MAKE_STR(TEST_FUNCTION)
                                ", reduced exponent range [%"
                                MPFR_EXP_FSPEC "d,%" MPFR_EXP_FSPEC
                                "d] (overflow test) on:\n",
                                (mpfr_eexp_t) oemin, (mpfr_eexp_t) e - 1);
                        printf ("x = ");
                        mpfr_dump (x);
#if defined(TWO_ARGS_ALL)
                        printf ("u = ");
                        mpfr_dump (u);
#endif
                        printf ("yprec = %u, rnd_mode = %s\n",
                                (unsigned int) yprec,
                                mpfr_print_rnd_mode (rnd));
                        printf ("Expected flags =");
                        flags_out (ex_flags);
                        printf ("     got flags =");
                        flags_out (flags);
                        printf ("inex = %d, w = ", inexact);
                        mpfr_dump (w);
                        exit (1);
                      }
                    test_of = 0;  /* Overflow is tested only once. */
                  }
                if (test_uf && e + 1 <= emin)
                  {
                    unsigned int ex_flags;

                    mpfr_set_emin (e + 1);
                    mpfr_clear_flags ();
#if defined(TWO_ARGS)
                    inexact = TEST_FUNCTION (w, x, u, rnd);
#elif defined(DOUBLE_ARG1)
                    inexact = TEST_FUNCTION (w, d, x, rnd);
#elif defined(DOUBLE_ARG2)
                    inexact = TEST_FUNCTION (w, x, d, rnd);
#elif defined(ULONG_ARG1)
                    inexact = TEST_FUNCTION (w, i, x, rnd);
#elif defined(ULONG_ARG2)
                    inexact = TEST_FUNCTION (w, x, i, rnd);
#else
                    inexact = TEST_FUNCTION (w, x, rnd);
#endif
                    flags = __gmpfr_flags;
                    mpfr_set_emin (oemin);
                    ex_flags = MPFR_FLAGS_UNDERFLOW | MPFR_FLAGS_INEXACT;
                    if (flags != ex_flags)
                      {
                        printf ("tgeneric: error for " MAKE_STR(TEST_FUNCTION)
                                ", reduced exponent range [%"
                                MPFR_EXP_FSPEC "d,%" MPFR_EXP_FSPEC
                                "d] (underflow test) on:\n",
                                (mpfr_eexp_t) e + 1, (mpfr_eexp_t) oemax);
                        printf ("x = ");
                        mpfr_dump (x);
#if defined(TWO_ARGS_ALL)
                        printf ("u = ");
                        mpfr_dump (u);
#endif
                        printf ("yprec = %u, rnd_mode = %s\n",
                                (unsigned int) yprec,
                                mpfr_print_rnd_mode (rnd));
                        printf ("Expected flags =");
                        flags_out (ex_flags);
                        printf ("     got flags =");
                        flags_out (flags);
                        printf ("inex = %d, w = ", inexact);
                        mpfr_dump (w);
                        exit (1);
                      }
                    test_uf = 0;  /* Underflow is tested only once. */
                  }
                if (e < emin)
                  emin = e;
                if (e > emax)
                  emax = e;
              }
            if (emin > emax)
              emin = emax;  /* case where all values are singular */
            /* Consistency test in a reduced exponent range. Doing it
               for the first 10 samples and for prec == p1 (which has
               some special cases) should be sufficient. */
            if (ctrt <= 10 || prec == p1)
              {
                mpfr_set_emin (emin);
                mpfr_set_emax (emax);
#ifdef DEBUG_TGENERIC
                /* Useful information in case of assertion failure. */
                printf ("tgeneric: reduced exponent range [%"
                        MPFR_EXP_FSPEC "d,%" MPFR_EXP_FSPEC "d]\n",
                        (mpfr_eexp_t) emin, (mpfr_eexp_t) emax);
#endif
                mpfr_clear_flags ();
#if defined(TWO_ARGS)
                inexact = TEST_FUNCTION (w, x, u, rnd);
#elif defined(DOUBLE_ARG1)
                inexact = TEST_FUNCTION (w, d, x, rnd);
#elif defined(DOUBLE_ARG2)
                inexact = TEST_FUNCTION (w, x, d, rnd);
#elif defined(ULONG_ARG1)
                inexact = TEST_FUNCTION (w, i, x, rnd);
#elif defined(ULONG_ARG2)
                inexact = TEST_FUNCTION (w, x, i, rnd);
#else
                inexact = TEST_FUNCTION (w, x, rnd);
#endif
                flags = __gmpfr_flags;
                mpfr_set_emin (oemin);
                mpfr_set_emax (oemax);
                if (! (SAME_VAL (w, y) &&
                       SAME_SIGN (inexact, compare) &&
                       flags == oldflags))
                  {
                    printf ("tgeneric: error for " MAKE_STR(TEST_FUNCTION)
                            ", reduced exponent range [%"
                            MPFR_EXP_FSPEC "d,%" MPFR_EXP_FSPEC "d] on:\n",
                            (mpfr_eexp_t) emin, (mpfr_eexp_t) emax);
                    printf ("x = ");
                    mpfr_dump (x);
#if defined(TWO_ARGS_ALL)
                    printf ("u = ");
                    mpfr_dump (u);
#endif
                    printf ("yprec = %u, rnd_mode = %s\n",
                            (unsigned int) yprec, mpfr_print_rnd_mode (rnd));
                    printf ("Expected:\n  y = ");
                    mpfr_dump (y);
                    printf ("  inex = %d, flags =", compare);
                    flags_out (oldflags);
                    printf ("Got:\n  w = ");
                    mpfr_dump (w);
                    printf ("  inex = %d, flags =", inexact);
                    flags_out (flags);
                    exit (1);
                  }
              }
            __gmpfr_flags = oldflags;  /* restore the flags */
          }

          if (MPFR_IS_SINGULAR (y))
            {
              if (MPFR_IS_NAN (y) || mpfr_nanflag_p ())
                TGENERIC_CHECK ("bad NaN flag",
                                MPFR_IS_NAN (y) && mpfr_nanflag_p ());
              else if (MPFR_IS_INF (y))
                {
                  TGENERIC_CHECK ("bad overflow flag",
                                  (compare != 0) ^ (mpfr_overflow_p () == 0));
                  TGENERIC_CHECK ("bad divide-by-zero flag",
                                  (compare == 0 && !infinite_input) ^
                                  (mpfr_divby0_p () == 0));
                }
              else if (MPFR_IS_ZERO (y))
                TGENERIC_CHECK ("bad underflow flag",
                                (compare != 0) ^ (mpfr_underflow_p () == 0));
            }
          else if (mpfr_divby0_p ())
            {
              TGENERIC_CHECK ("both overflow and divide-by-zero",
                              ! mpfr_overflow_p ());
              TGENERIC_CHECK ("both underflow and divide-by-zero",
                              ! mpfr_underflow_p ());
              TGENERIC_CHECK ("bad compare value (divide-by-zero)",
                              compare == 0);
            }
          else if (mpfr_overflow_p ())
            {
              TGENERIC_CHECK ("both underflow and overflow",
                              ! mpfr_underflow_p ());
              TGENERIC_CHECK ("bad compare value (overflow)", compare != 0);
              mpfr_nexttoinf (y);
              TGENERIC_CHECK ("should have been max MPFR number (overflow)",
                              MPFR_IS_INF (y));
            }
          else if (mpfr_underflow_p ())
            {
              TGENERIC_CHECK ("bad compare value (underflow)", compare != 0);
              mpfr_nexttozero (y);
              TGENERIC_CHECK ("should have been min MPFR number (underflow)",
                              MPFR_IS_ZERO (y));
            }
          else if (mpfr_can_round (y, yprec, rnd, rnd, prec))
            {
              ctrn++;
              mpfr_set (t, y, rnd);
              /* Risk of failures are known when some flags are already set
                 before the function call. Do not set the erange flag, as
                 it will remain set after the function call and no checks
                 are performed in such a case (see the mpfr_erangeflag_p
                 test below). */
              if (randlimb () & 1)
                __gmpfr_flags = MPFR_FLAGS_ALL ^ MPFR_FLAGS_ERANGE;
#ifdef DEBUG_TGENERIC
              TGENERIC_INFO (TEST_FUNCTION, MPFR_PREC (z));
#endif
              /* Let's increase the precision of the inputs in a random way.
                 In most cases, this doesn't make any difference, but for
                 the mpfr_fmod bug fixed in r6230, this triggers the bug. */
              mpfr_prec_round (x, mpfr_get_prec (x) + (randlimb () & 15),
                               MPFR_RNDN);
#if defined(TWO_ARGS)
              mpfr_prec_round (u, mpfr_get_prec (u) + (randlimb () & 15),
                               MPFR_RNDN);
              inexact = TEST_FUNCTION (z, x, u, rnd);
#elif defined(DOUBLE_ARG1)
              inexact = TEST_FUNCTION (z, d, x, rnd);
#elif defined(DOUBLE_ARG2)
              inexact = TEST_FUNCTION (z, x, d, rnd);
#elif defined(ULONG_ARG1)
              inexact = TEST_FUNCTION (z, i, x, rnd);
#elif defined(ULONG_ARG2)
              inexact = TEST_FUNCTION (z, x, i, rnd);
#else
              inexact = TEST_FUNCTION (z, x, rnd);
#endif
              if (mpfr_erangeflag_p ())
                goto next_n;
              if (! mpfr_equal_p (t, z))
                {
                  printf ("tgeneric: results differ for "
                          MAKE_STR(TEST_FUNCTION) " on\n  x = ");
                  mpfr_dump (x);
#if defined(TWO_ARGS_ALL)
                  printf ("  u = ");
                  mpfr_dump (u);
#endif
                  printf ("  prec = %u, rnd_mode = %s\n",
                          (unsigned int) prec, mpfr_print_rnd_mode (rnd));
                  printf ("Got      ");
                  mpfr_dump (z);
                  printf ("Expected ");
                  mpfr_dump (t);
                  printf ("Approx   ");
                  mpfr_dump (y);
                  exit (1);
                }
              compare2 = mpfr_cmp (t, y);
              /* if rounding to nearest, cannot know the sign of t - f(x)
                 because of composed rounding: y = o(f(x)) and t = o(y) */
              if (compare * compare2 >= 0)
                compare = compare + compare2;
              else
                compare = inexact; /* cannot determine sign(t-f(x)) */
              if (! SAME_SIGN (inexact, compare))
                {
                  printf ("Wrong inexact flag for rnd=%s: expected %d, got %d"
                          "\n", mpfr_print_rnd_mode (rnd), compare, inexact);
                  printf ("x = ");
                  mpfr_dump (x);
#if defined(TWO_ARGS_ALL)
                  printf ("u = ");
                  mpfr_dump (u);
#endif
                  printf ("y = ");
                  mpfr_dump (y);
                  printf ("t = ");
                  mpfr_dump (t);
                  exit (1);
                }
            }
          else if (getenv ("MPFR_SUSPICIOUS_OVERFLOW") != NULL)
            {
              /* For developers only! */
              MPFR_ASSERTN (MPFR_IS_PURE_FP (y));
              mpfr_nexttoinf (y);
              if (MPFR_IS_INF (y) && MPFR_IS_LIKE_RNDZ (rnd, MPFR_IS_NEG (y))
                  && !mpfr_overflow_p () && TGENERIC_SO_TEST)
                {
                  printf ("Possible bug! |y| is the maximum finite number "
                          "and has been obtained when\nrounding toward zero"
                          " (%s). Thus there is a very probable overflow,\n"
                          "but the overflow flag is not set!\n",
                          mpfr_print_rnd_mode (rnd));
                  printf ("x = ");
                  mpfr_dump (x);
#if defined(TWO_ARGS_ALL)
                  printf ("u = ");
                  mpfr_dump (u);
#endif
                  exit (1);
                }
            }

        next_n:
          /* In case the exponent range has been changed by
             tests_default_random() or for special values... */
          mpfr_set_emin (old_emin);
          mpfr_set_emax (old_emax);
        }
    }

#ifndef TGENERIC_NOWARNING
  if (3 * ctrn < 2 * ctrt)
    printf ("Warning! Too few normal cases in generic tests (%lu / %lu)\n",
            ctrn, ctrt);
#endif

  mpfr_clears (x, y, z, t, w, (mpfr_ptr) 0);
#if defined(TWO_ARGS_ALL)
  mpfr_clear (u);
#endif
}

#undef TEST_RANDOM_POS
#undef TEST_RANDOM_POS2
#undef TEST_RANDOM_EMIN
#undef TEST_RANDOM_EMAX
#undef TEST_RANDOM_ALWAYS_SCALE
#undef RAND_FUNCTION
#undef TWO_ARGS
#undef TWO_ARGS_ALL
#undef DOUBLE_ARG1
#undef DOUBLE_ARG2
#undef ULONG_ARG1
#undef ULONG_ARG2
#undef TEST_FUNCTION
#undef test_generic
