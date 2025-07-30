/* Test file for mpfr_sqr.

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

#include <stdio.h>
#include <stdlib.h>

#include "mpfr-test.h"

#define TEST_FUNCTION mpfr_sqr
#include "tgeneric.c"

static int
inexact_sign (int x)
{
  return (x < 0) ? -1 : (x > 0);
}

static void
error1 (mpfr_rnd_t rnd, mpfr_prec_t prec,
        mpfr_t in, mpfr_t outmul, mpfr_t outsqr)
{
  printf("ERROR: for %s and prec=%lu\nINPUT=", mpfr_print_rnd_mode(rnd),
         (unsigned long) prec);
  mpfr_dump(in);
  printf("OutputMul="); mpfr_dump(outmul);
  printf("OutputSqr="); mpfr_dump(outsqr);
  exit(1);
}

static void
error2 (mpfr_rnd_t rnd, mpfr_prec_t prec, mpfr_t in, mpfr_t out,
        int inexactmul, int inexactsqr)
{
  printf("ERROR: for %s and prec=%lu\nINPUT=", mpfr_print_rnd_mode(rnd),
         (unsigned long) prec);
  mpfr_dump(in);
  printf("Output="); mpfr_dump(out);
  printf("InexactMul= %d InexactSqr= %d\n", inexactmul, inexactsqr);
  exit(1);
}

static void
check_random (mpfr_prec_t p)
{
  mpfr_t x,y,z;
  int r;
  int i, inexact1, inexact2;

  mpfr_inits2 (p, x, y, z, (mpfr_ptr) 0);
  for(i = 0 ; i < 500 ; i++)
    {
      mpfr_urandomb (x, RANDS);
      if (MPFR_IS_PURE_FP(x))
        for (r = 0 ; r < MPFR_RND_MAX ; r++)
          {
            inexact1 = mpfr_mul (y, x, x, (mpfr_rnd_t) r);
            inexact2 = mpfr_sqr (z, x, (mpfr_rnd_t) r);
            if (mpfr_cmp (y, z))
              error1 ((mpfr_rnd_t) r,p,x,y,z);
            if (inexact_sign (inexact1) != inexact_sign (inexact2))
              error2 ((mpfr_rnd_t) r,p,x,y,inexact1,inexact2);
          }
    }
  mpfr_clears (x, y, z, (mpfr_ptr) 0);
}

static void
check_special (void)
{
  mpfr_t x, y;
  mpfr_exp_t emin;

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_nan (x);
  mpfr_sqr (y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (y));

  mpfr_set_inf (x, 1);
  mpfr_sqr (y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (y) && mpfr_sgn (y) > 0);

  mpfr_set_inf (x, -1);
  mpfr_sqr (y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (y) && mpfr_sgn (y) > 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_sqr (y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_zero_p (y));

  emin = mpfr_get_emin ();
  mpfr_set_emin (0);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_div_2ui (x, x, 1, MPFR_RNDN);
  MPFR_ASSERTN (!mpfr_zero_p (x));
  mpfr_sqr (y, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_zero_p (y));
  mpfr_set_emin (emin);

  mpfr_clear (y);
  mpfr_clear (x);
}

/* Test of a bug seen with GCC 4.5.2 and GMP 5.0.1 on m68k (m68000 target).
     https://sympa.inria.fr/sympa/arc/mpfr/2011-05/msg00003.html
     https://sympa.inria.fr/sympa/arc/mpfr/2011-05/msg00041.html
*/
static void
check_mpn_sqr (void)
{
#if GMP_NUMB_BITS == 32 && __GNU_MP_VERSION >= 5
  /* Note: since we test a low-level bug, src is initialized
     without using a GMP function, just in case. */
  mp_limb_t src[5] =
    { 0x90000000, 0xbaa55f4f, 0x2cbec4d9, 0xfcef3242, 0xda827999 };
  mp_limb_t exd[10] =
    { 0x00000000, 0x31000000, 0xbd4bc59a, 0x41fbe2b5, 0x33471e7e,
      0x90e826a7, 0xbaa55f4f, 0x2cbec4d9, 0xfcef3242, 0xba827999 };
  mp_limb_t dst[10];
  int i;

  mpn_sqr (dst, src, 5);  /* new in GMP 5 */
  for (i = 0; i < 10; i++)
    {
      if (dst[i] != exd[i])
        {
          printf ("Error in check_mpn_sqr\n");
          printf ("exd[%d] = 0x%08lx\n", i, (unsigned long) exd[i]);
          printf ("dst[%d] = 0x%08lx\n", i, (unsigned long) dst[i]);
          printf ("Note: This is not a bug in MPFR, but a bug in"
                  " either GMP or, more\nprobably, in the compiler."
                  " It may cause other tests to fail.\n");
          exit (1);
        }
    }
#endif
}

int
main (void)
{
  mpfr_prec_t p;

  tests_start_mpfr ();

  check_mpn_sqr ();

  check_special ();
  for (p = 2; p < 200; p++)
    check_random (p);

  test_generic (2, 200, 15);
  data_check ("data/sqr", mpfr_sqr, "mpfr_sqr");
  bad_cases (mpfr_sqr, mpfr_sqrt, "mpfr_sqr", 8, -256, 255, 4, 128, 800, 50);

  tests_end_mpfr ();
  return 0;
}
