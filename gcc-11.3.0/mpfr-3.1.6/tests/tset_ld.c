/* Test file for mpfr_set_ld and mpfr_get_ld.

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

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <limits.h>
#ifdef WITH_FPU_CONTROL
#include <fpu_control.h>
#endif

#include "mpfr-test.h"

static void
check_gcc33_bug (void)
{
  volatile long double x;
  x = (long double) 9007199254740992.0 + 1.0;
  if (x != 0.0)
    return;  /* OK */
  printf
    ("Detected optimization bug of gcc 3.3 on Alpha concerning long double\n"
     "comparisons; set_ld tests might fail (set_ld won't work correctly).\n"
     "See http://gcc.gnu.org/ml/gcc-bugs/2003-10/msg00853.html for more\n"
     "information.\n");
}

static int
Isnan_ld (long double d)
{
  /* Do not convert d to double as this can give an overflow, which
     may confuse compilers without IEEE 754 support (such as clang
     -fsanitize=undefined), or trigger a trap if enabled.
     The DOUBLE_ISNAN macro should work fine on long double. */
  if (DOUBLE_ISNAN (d))
    return 1;
  LONGDOUBLE_NAN_ACTION (d, goto yes);
  return 0;
 yes:
  return 1;
}

/* checks that a long double converted to a mpfr (with precision >=113),
   then converted back to a long double gives the initial value,
   or in other words mpfr_get_ld(mpfr_set_ld(d)) = d.
*/
static void
check_set_get (long double d, mpfr_t x)
{
  int r;
  long double e;
  int inex;

  for (r = 0; r < MPFR_RND_MAX; r++)
    {
      inex = mpfr_set_ld (x, d, (mpfr_rnd_t) r);
      if (inex != 0)
        {
          mpfr_exp_t emin, emax;
          emin = mpfr_get_emin ();
          emax = mpfr_get_emax ();
          printf ("Error: mpfr_set_ld should be exact\n");
          printf ("d=%1.30Le inex=%d\n", d, inex);
          if (emin >= LONG_MIN)
            printf ("emin=%ld\n", (long) emin);
          if (emax <= LONG_MAX)
            printf ("emax=%ld\n", (long) emax);
          mpfr_dump (x);
          exit (1);
        }
      e = mpfr_get_ld (x, (mpfr_rnd_t) r);
      if ((Isnan_ld(d) && ! Isnan_ld(e)) ||
          (Isnan_ld(e) && ! Isnan_ld(d)) ||
          (e != d && !(Isnan_ld(e) && Isnan_ld(d))))
        {
          printf ("Error: mpfr_get_ld o mpfr_set_ld <> Id\n");
          printf ("  r=%d\n", r);
          printf ("  d=%1.30Le get_ld(set_ld(d))=%1.30Le\n", d, e);
          ld_trace ("  d", d);
          printf ("  x="); mpfr_out_str (NULL, 16, 0, x, MPFR_RNDN);
          printf ("\n");
          ld_trace ("  e", e);
#ifdef MPFR_NANISNAN
          if (Isnan_ld(d) || Isnan_ld(e))
            printf ("The reason is that NAN == NAN. Please look at the "
                    "configure output\nand Section \"In case of problem\" "
                    "of the INSTALL file.\n");
#endif
          exit (1);
        }
    }
}

static void
test_small (void)
{
  mpfr_t x, y, z;
  long double d;

  mpfr_init2 (x, 64);
  mpfr_init2 (y, 64);
  mpfr_init2 (z, 64);

  /* x = 11906603631607553907/2^(16381+64) */
  mpfr_set_str (x, "0.1010010100111100110000001110101101000111010110000001111101110011E-16381", 2, MPFR_RNDN);
  d = mpfr_get_ld (x, MPFR_RNDN);  /* infinite loop? */
  mpfr_set_ld (y, d, MPFR_RNDN);
  mpfr_sub (z, x, y, MPFR_RNDN);
  mpfr_abs (z, z, MPFR_RNDN);
  mpfr_clear_erangeflag ();
  /* If long double = double, d should be equal to 0;
     in this case, everything is OK. */
  if (d != 0 && (mpfr_cmp_str (z, "1E-16434", 2, MPFR_RNDN) > 0 ||
                 mpfr_erangeflag_p ()))
    {
      printf ("Error with x = ");
      mpfr_out_str (NULL, 10, 21, x, MPFR_RNDN);
      printf (" = ");
      mpfr_out_str (NULL, 16, 0, x, MPFR_RNDN);
      printf ("\n        -> d = %.21Lg", d);
      printf ("\n        -> y = ");
      mpfr_out_str (NULL, 10, 21, y, MPFR_RNDN);
      printf (" = ");
      mpfr_out_str (NULL, 16, 0, y, MPFR_RNDN);
      printf ("\n        -> |x-y| = ");
      mpfr_out_str (NULL, 16, 0, z, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

static void
test_fixed_bugs (void)
{
  mpfr_t x;
  long double l, m;

  /* bug found by Steve Kargl (2009-03-14) */
  mpfr_init2 (x, 64);
  mpfr_set_ui_2exp (x, 1, -16447, MPFR_RNDN);
  mpfr_get_ld (x, MPFR_RNDN);  /* an assertion failed in init2.c:50 */

  /* bug reported by Jakub Jelinek (2010-10-17)
     https://gforge.inria.fr/tracker/?func=detail&aid=11300 */
  mpfr_set_prec (x, MPFR_LDBL_MANT_DIG);
  /* l = 0x1.23456789abcdef0123456789abcdp-914L; */
  l = 8.215640181713713164092636634579e-276;
  mpfr_set_ld (x, l, MPFR_RNDN);
  m = mpfr_get_ld (x, MPFR_RNDN);
  if (m != l)
    {
      printf ("Error in get_ld o set_ld for l=%Le\n", l);
      printf ("Got m=%Le instead of l\n", m);
      exit (1);
    }

  /* another similar test which failed with extended double precision and the
     generic code for mpfr_set_ld */
  /* l = 0x1.23456789abcdef0123456789abcdp-968L; */
  l = 4.560596445887084662336528403703e-292;
  mpfr_set_ld (x, l, MPFR_RNDN);
  m = mpfr_get_ld (x, MPFR_RNDN);
  if (m != l)
    {
      printf ("Error in get_ld o set_ld for l=%Le\n", l);
      printf ("Got m=%Le instead of l\n", m);
      exit (1);
    }

  mpfr_clear (x);
}

/* bug reported by Walter Mascarenhas
   https://sympa.inria.fr/sympa/arc/mpfr/2016-09/msg00005.html */
static void
bug_20160907 (void)
{
#if HAVE_LDOUBLE_IEEE_EXT_LITTLE
  long double dn, ld;
  mpfr_t mp;
  long e;
  mpfr_long_double_t x;

  /* the following is the encoding of the smallest subnormal number
     for HAVE_LDOUBLE_IEEE_EXT_LITTLE */
  x.s.manl = 1;
  x.s.manh = 0;
  x.s.expl = 0;
  x.s.exph = 0;
  x.s.sign= 0;
  dn = x.ld;
  e = -16445;
  /* dn=2^e is now the smallest subnormal. */

  mpfr_init2 (mp, 64);
  mpfr_set_ui_2exp (mp, 1, e - 1, MPFR_RNDN);
  ld = mpfr_get_ld (mp, MPFR_RNDU);
  /* since mp = 2^(e-1) and ld is rounded upwards, we should have
     ld = 2^e */
  if (ld != dn)
    {
      printf ("Error, ld = %Le <> dn = %Le\n", ld, dn);
      printf ("mp=");
      mpfr_out_str (stdout, 10, 0, mp, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  /* check a few more numbers */
  for (e = -16446; e <= -16381; e++)
    {
      mpfr_set_ui_2exp (mp, 1, e, MPFR_RNDN);
      ld = mpfr_get_ld (mp, MPFR_RNDU);
      mpfr_set_ld (mp, ld, MPFR_RNDU);
      /* mp is 2^e rounded up, thus should be >= 2^e */
      MPFR_ASSERTN(mpfr_cmp_ui_2exp (mp, 1, e) >= 0);

      mpfr_set_ui_2exp (mp, 1, e, MPFR_RNDN);
      ld = mpfr_get_ld (mp, MPFR_RNDD);
      mpfr_set_ld (mp, ld, MPFR_RNDD);
      /* mp is 2^e rounded down, thus should be <= 2^e */
      if (mpfr_cmp_ui_2exp (mp, 3, e) > 0)
        {
          printf ("Error, expected value <= 2^%ld\n", e);
          printf ("got "); mpfr_dump (mp);
          exit (1);
        }
    }

  mpfr_clear (mp);
#endif
}

int
main (int argc, char *argv[])
{
  long double d, e;
  mpfr_t x;
  int i;
  mpfr_exp_t emax;
#ifdef WITH_FPU_CONTROL
  fpu_control_t cw;

  if (argc > 1)
    {
      cw = strtol(argv[1], NULL, 0);
      printf ("FPU control word: 0x%x\n", (unsigned int) cw);
      _FPU_SETCW (cw);
    }
#endif

  tests_start_mpfr ();
  mpfr_test_init ();

  check_gcc33_bug ();
  test_fixed_bugs ();

  mpfr_init2 (x, MPFR_LDBL_MANT_DIG);

#if !defined(MPFR_ERRDIVZERO)
  /* check NaN */
  mpfr_set_nan (x);
  d = mpfr_get_ld (x, MPFR_RNDN);
  check_set_get (d, x);
#endif

  /* check +0.0 and -0.0 */
  d = 0.0;
  check_set_get (d, x);
  d = DBL_NEG_ZERO;
  check_set_get (d, x);

  /* check that the sign of -0.0 is set */
  mpfr_set_ld (x, DBL_NEG_ZERO, MPFR_RNDN);
  if (MPFR_SIGN(x) > 0)
    {
      printf ("Error: sign of -0.0 is not set correctly\n");
#if _GMP_IEEE_FLOATS
      exit (1);
      /* Non IEEE doesn't support negative zero yet */
#endif
    }

#if !defined(MPFR_ERRDIVZERO)
  /* check +Inf */
  mpfr_set_inf (x, 1);
  d = mpfr_get_ld (x, MPFR_RNDN);
  check_set_get (d, x);

  /* check -Inf */
  mpfr_set_inf (x, -1);
  d = mpfr_get_ld (x, MPFR_RNDN);
  check_set_get (d, x);
#endif

  /* check the largest power of two */
  d = 1.0; while (d < LDBL_MAX / 2.0) d += d;
  check_set_get (d, x);
  check_set_get (-d, x);

  /* check largest long double */
  d = LDBL_MAX;
  check_set_get (d, x);
  check_set_get (-d, x);

  /* check the smallest power of two */
  d = 1.0;
  while ((e = d / 2.0) != (long double) 0.0 && e != d)
    d = e;
  check_set_get (d, x);
  check_set_get (-d, x);

  /* check largest 2^(2^k) that is representable as a long double */
  d = (LDBL_MAX / 2) + (LDBL_MAX / 4 * LDBL_EPSILON);
  check_set_get (d, x);

  /* check that 2^i, 2^i+1 and 2^i-1 are correctly converted */
  d = 1.0;
  for (i = 1; i < MPFR_LDBL_MANT_DIG; i++)
    {
      d = 2.0 * d; /* d = 2^i */
      check_set_get (d, x);
      check_set_get (d + 1.0, x);
      check_set_get (d - 1.0, x);
    }

  for (i = 0; i < 10000; i++)
    {
      mpfr_urandomb (x, RANDS);
      d = mpfr_get_ld (x, MPFR_RNDN);
      check_set_get (d, x);
    }

  /* check with reduced emax to exercise overflow */
  emax = mpfr_get_emax ();
  mpfr_set_prec (x, 2);
  set_emax (1);
  mpfr_set_ld (x, (long double) 2.0, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);
  for (d = (long double) 2.0, i = 0; i < 13; i++, d *= d);
  /* now d = 2^8192 */
  mpfr_set_ld (x, d, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);
  set_emax (emax);

  mpfr_clear (x);

  test_small ();

  bug_20160907 ();

  tests_end_mpfr ();

  return 0;
}
