/* Test file for:
 mpfr_fits_sint_p, mpfr_fits_slong_p, mpfr_fits_sshort_p,
 mpfr_fits_uint_p, mpfr_fits_ulong_p, mpfr_fits_ushort_p

Copyright 2004-2017 Free Software Foundation, Inc.
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

#ifdef HAVE_CONFIG_H
# include "config.h"       /* for a build within gmp */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "mpfr-intmax.h"
#include "mpfr-test.h"

#define FTEST_AUX(N,NOT,FCT)                                    \
  do                                                            \
    {                                                           \
      __gmpfr_flags = ex_flags;                                 \
      if (NOT FCT (x, (mpfr_rnd_t) r))                          \
        {                                                       \
          printf ("Error %d for %s, rnd = %s and x = ",         \
                  N, #FCT,                                      \
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r));        \
          mpfr_dump (x);                                        \
          exit (1);                                             \
        }                                                       \
      if (__gmpfr_flags != ex_flags)                            \
        {                                                       \
          unsigned int flags = __gmpfr_flags;                   \
          printf ("Flags error %d for %s, rnd = %s and x = ",   \
                  N, #FCT,                                      \
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r));        \
          mpfr_dump(x);                                         \
          printf ("Expected flags:");                           \
          flags_out (ex_flags);                                 \
          printf ("Got flags:     ");                           \
          flags_out (flags);                                    \
          exit (1);                                             \
        }                                                       \
    }                                                           \
  while (0)

#define FTEST(N,NOT,FCT)                                        \
  do                                                            \
    {                                                           \
      mpfr_exp_t e;                                             \
      FTEST_AUX (N,NOT,FCT);                                    \
      if (MPFR_IS_SINGULAR (x))                                 \
        break;                                                  \
      e = mpfr_get_exp (x);                                     \
      set_emin (e);                                             \
      set_emax (e);                                             \
      FTEST_AUX (N,NOT,FCT);                                    \
      set_emin (emin);                                          \
      set_emax (emax);                                          \
    }                                                           \
  while (0)

#define CHECK_ALL(N,NOT)                                        \
  do                                                            \
    {                                                           \
      FTEST (N, NOT, mpfr_fits_ulong_p);                        \
      FTEST (N, NOT, mpfr_fits_slong_p);                        \
      FTEST (N, NOT, mpfr_fits_uint_p);                         \
      FTEST (N, NOT, mpfr_fits_sint_p);                         \
      FTEST (N, NOT, mpfr_fits_ushort_p);                       \
      FTEST (N, NOT, mpfr_fits_sshort_p);                       \
    }                                                           \
  while (0)

#define CHECK_MAX(N,NOT)                                        \
  do                                                            \
    {                                                           \
      FTEST (N, NOT, mpfr_fits_uintmax_p);                      \
      FTEST (N, NOT, mpfr_fits_intmax_p);                       \
    }                                                           \
  while (0)

/* V is a non-zero limit for the type (*_MIN for a signed type or *_MAX).
 * If V is positive, then test V, V + 1/4, V + 3/4 and V + 1.
 * If V is negative, then test V, V - 1/4, V - 3/4 and V - 1.
 */
#define CHECK_LIM(N,V,SET,FCT)                                  \
  do                                                            \
    {                                                           \
      SET (x, V, MPFR_RNDN);                                    \
      FTEST (N, !, FCT);                                        \
      mpfr_set_si_2exp (y, (V) < 0 ? -1 : 1, -2, MPFR_RNDN);    \
      mpfr_add (x, x, y, MPFR_RNDN);                            \
      FTEST (N+1, (r == MPFR_RNDN ||                            \
                   MPFR_IS_LIKE_RNDZ (r, (V) < 0)) ^ !!, FCT);  \
      mpfr_add (x, x, y, MPFR_RNDN);                            \
      mpfr_add (x, x, y, MPFR_RNDN);                            \
      FTEST (N+3, MPFR_IS_LIKE_RNDZ (r, (V) < 0) ^ !!, FCT);    \
      mpfr_add (x, x, y, MPFR_RNDN);                            \
      FTEST (N+4, !!, FCT);                                     \
    }                                                           \
  while (0)

int
main (void)
{
  mpfr_exp_t emin, emax;
  mpfr_t x, y;
  unsigned int flags[2] = { 0, MPFR_FLAGS_ALL }, ex_flags;
  int i, r, fi;

  tests_start_mpfr ();

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  mpfr_init2 (x, sizeof (unsigned long) * CHAR_BIT + 2);
  mpfr_init2 (y, 8);

  RND_LOOP (r)
    for (fi = 0; fi < numberof (flags); fi++)
      {
        ex_flags = flags[fi];

        /* Check NaN */
        mpfr_set_nan (x);
        CHECK_ALL (1, !!);

        /* Check +Inf */
        mpfr_set_inf (x, 1);
        CHECK_ALL (2, !!);

        /* Check -Inf */
        mpfr_set_inf (x, -1);
        CHECK_ALL (3, !!);

        /* Check +0 */
        mpfr_set_zero (x, 1);
        CHECK_ALL (4, !);

        /* Check -0 */
        mpfr_set_zero (x, -1);
        CHECK_ALL (5, !);

        /* Check small positive op */
        mpfr_set_str1 (x, "1@-1");
        CHECK_ALL (6, !);

        /* Check 17 */
        mpfr_set_ui (x, 17, MPFR_RNDN);
        CHECK_ALL (7, !);

        /* Check large values (no fit) */
        mpfr_set_ui (x, ULONG_MAX, MPFR_RNDN);
        mpfr_mul_2exp (x, x, 1, MPFR_RNDN);
        CHECK_ALL (8, !!);
        mpfr_mul_2exp (x, x, 40, MPFR_RNDN);
        CHECK_ALL (9, !!);

        /* Check a non-integer number just below a power of two. */
        mpfr_set_ui_2exp (x, 255, -2, MPFR_RNDN);
        CHECK_ALL (10, !);

        /* Check the limits of the types (except 0 for unsigned types) */
        CHECK_LIM (20, ULONG_MAX, mpfr_set_ui, mpfr_fits_ulong_p);
        CHECK_LIM (30, LONG_MAX, mpfr_set_si, mpfr_fits_slong_p);
        CHECK_LIM (35, LONG_MIN, mpfr_set_si, mpfr_fits_slong_p);
        CHECK_LIM (40, UINT_MAX, mpfr_set_ui, mpfr_fits_uint_p);
        CHECK_LIM (50, INT_MAX, mpfr_set_si, mpfr_fits_sint_p);
        CHECK_LIM (55, INT_MIN, mpfr_set_si, mpfr_fits_sint_p);
        CHECK_LIM (60, USHRT_MAX, mpfr_set_ui, mpfr_fits_ushort_p);
        CHECK_LIM (70, SHRT_MAX, mpfr_set_si, mpfr_fits_sshort_p);
        CHECK_LIM (75, SHRT_MIN, mpfr_set_si, mpfr_fits_sshort_p);

        /* Check negative op */
        for (i = 1; i <= 4; i++)
          {
            int inv;

            mpfr_set_si_2exp (x, -i, -2, MPFR_RNDN);
            mpfr_rint (y, x, (mpfr_rnd_t) r);
            inv = MPFR_NOTZERO (y);
            FTEST (80, inv ^ !, mpfr_fits_ulong_p);
            FTEST (81,       !, mpfr_fits_slong_p);
            FTEST (82, inv ^ !, mpfr_fits_uint_p);
            FTEST (83,       !, mpfr_fits_sint_p);
            FTEST (84, inv ^ !, mpfr_fits_ushort_p);
            FTEST (85,       !, mpfr_fits_sshort_p);
          }
      }

#ifdef _MPFR_H_HAVE_INTMAX_T

  mpfr_set_prec (x, sizeof (uintmax_t) * CHAR_BIT + 2);

  RND_LOOP (r)
    {
      /* Check NaN */
      mpfr_set_nan (x);
      CHECK_MAX (1, !!);

      /* Check +Inf */
      mpfr_set_inf (x, 1);
      CHECK_MAX (2, !!);

      /* Check -Inf */
      mpfr_set_inf (x, -1);
      CHECK_MAX (3, !!);

      /* Check +0 */
      mpfr_set_zero (x, 1);
      CHECK_MAX (4, !);

      /* Check -0 */
      mpfr_set_zero (x, -1);
      CHECK_MAX (5, !);

      /* Check small positive op */
      mpfr_set_str1 (x, "1@-1");
      CHECK_MAX (6, !);

      /* Check 17 */
      mpfr_set_ui (x, 17, MPFR_RNDN);
      CHECK_MAX (7, !);

      /* Check hugest */
      mpfr_set_ui_2exp (x, 42, sizeof (uintmax_t) * 32, MPFR_RNDN);
      CHECK_MAX (8, !!);

      /* Check a non-integer number just below a power of two. */
      mpfr_set_ui_2exp (x, 255, -2, MPFR_RNDN);
      CHECK_MAX (10, !);

      /* Check the limits of the types (except 0 for uintmax_t) */
      CHECK_LIM (20, MPFR_UINTMAX_MAX, mpfr_set_uj, mpfr_fits_uintmax_p);
      CHECK_LIM (30, MPFR_INTMAX_MAX, mpfr_set_sj, mpfr_fits_intmax_p);
      CHECK_LIM (35, MPFR_INTMAX_MIN, mpfr_set_sj, mpfr_fits_intmax_p);

      /* Check negative op */
      for (i = 1; i <= 4; i++)
        {
          int inv;

          mpfr_set_si_2exp (x, -i, -2, MPFR_RNDN);
          mpfr_rint (y, x, (mpfr_rnd_t) r);
          inv = MPFR_NOTZERO (y);
          FTEST (80, inv ^ !, mpfr_fits_uintmax_p);
          FTEST (81,       !, mpfr_fits_intmax_p);
        }
    }

#endif  /* _MPFR_H_HAVE_INTMAX_T */

  mpfr_clear (x);
  mpfr_clear (y);

  tests_end_mpfr ();
  return 0;
}
