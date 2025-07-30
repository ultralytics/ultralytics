/* Test file for mpfr_cmp_ui and mpfr_cmp_si.

Copyright 1999, 2001-2017 Free Software Foundation, Inc.
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

#ifdef TCMP_UI_CHECK_NAN

mpfr_clear_erangeflag ();
c = mpfr_cmp_si (x, TCMP_UI_CHECK_NAN);
if (c != 0 || !mpfr_erangeflag_p ())
  {
    printf ("NaN error on %d (1)\n", TCMP_UI_CHECK_NAN);
    exit (1);
  }
mpfr_clear_erangeflag ();
c = (mpfr_cmp_si) (x, TCMP_UI_CHECK_NAN);
if (c != 0 || !mpfr_erangeflag_p ())
  {
    printf ("NaN error on %d (2)\n", TCMP_UI_CHECK_NAN);
    exit (1);
  }
if (TCMP_UI_CHECK_NAN >= 0)
  {
    mpfr_clear_erangeflag ();
    c = mpfr_cmp_ui (x, TCMP_UI_CHECK_NAN);
    if (c != 0 || !mpfr_erangeflag_p ())
      {
        printf ("NaN error on %d (3)\n", TCMP_UI_CHECK_NAN);
        exit (1);
      }
    mpfr_clear_erangeflag ();
    c = (mpfr_cmp_ui) (x, TCMP_UI_CHECK_NAN);
    if (c != 0 || !mpfr_erangeflag_p ())
      {
        printf ("NaN error on %d (4)\n", TCMP_UI_CHECK_NAN);
        exit (1);
      }
  }

#else

#include <stdio.h>
#include <stdlib.h>

#include "mpfr-test.h"

static void
check_nan (void)
{
  mpfr_t x;
  int c, i;

  mpfr_init (x);
  mpfr_set_nan (x);
  /* We need constants to completely test the macros. */
#undef TCMP_UI_CHECK_NAN
#define TCMP_UI_CHECK_NAN -17
#include "tcmp_ui.c"
#undef TCMP_UI_CHECK_NAN
#define TCMP_UI_CHECK_NAN 0
#include "tcmp_ui.c"
#undef TCMP_UI_CHECK_NAN
#define TCMP_UI_CHECK_NAN 17
#include "tcmp_ui.c"
  for (i = -17; i <= 17; i += 17)
    {
#undef TCMP_UI_CHECK_NAN
#define TCMP_UI_CHECK_NAN i
#include "tcmp_ui.c"
    }
  mpfr_clear (x);
}

/* Since mpfr_cmp_ui and mpfr_cmp_si are also implemented by a macro
   with __builtin_constant_p for GCC, check that side effects are
   handled correctly. */
static void
check_macros (void)
{
  mpfr_t x;
  int c;

  mpfr_init2 (x, 32);

  c = 0;
  mpfr_set_ui (x, 17, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 17) != 0)
    {
      printf ("Error 1 on mpfr_cmp_ui(x,17) in check_macros\n");
      exit (1);
    }
  if (mpfr_cmp_ui (x, (c++, 17)) != 0)
    {
      printf ("Error 2 on mpfr_cmp_ui(x,17) in check_macros\n");
      exit (1);
    }
  if (c != 1)
    {
      printf ("Error 3 on mpfr_cmp_ui(x,17) in check_macros\n"
              "(c = %d instead of 1)\n", c);
      exit (1);
    }
  if (mpfr_cmp_si (x, 17) != 0)
    {
      printf ("Error 1 on mpfr_cmp_si(x,17) in check_macros\n");
      exit (1);
    }
  if (mpfr_cmp_si (x, (c++, 17)) != 0)
    {
      printf ("Error 2 on mpfr_cmp_si(x,17) in check_macros\n");
      exit (1);
    }
  if (c != 2)
    {
      printf ("Error 3 on mpfr_cmp_si(x,17) in check_macros\n"
              "(c = %d instead of 2)\n", c);
      exit (1);
    }

  c = 0;
  mpfr_set_ui (x, 0, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 0) != 0)
    {
      printf ("Error 1 on mpfr_cmp_ui(x,0) in check_macros\n");
      exit (1);
    }
  if (mpfr_cmp_ui (x, (c++, 0)) != 0)
    {
      printf ("Error 2 on mpfr_cmp_ui(x,0) in check_macros\n");
      exit (1);
    }
  if (c != 1)
    {
      printf ("Error 3 on mpfr_cmp_ui(x,0) in check_macros\n"
              "(c = %d instead of 1)\n", c);
      exit (1);
    }
  if (mpfr_cmp_si (x, 0) != 0)
    {
      printf ("Error 1 on mpfr_cmp_si(x,0) in check_macros\n");
      exit (1);
    }
  if (mpfr_cmp_si (x, (c++, 0)) != 0)
    {
      printf ("Error 2 on mpfr_cmp_si(x,0) in check_macros\n");
      exit (1);
    }
  if (c != 2)
    {
      printf ("Error 3 on mpfr_cmp_si(x,0) in check_macros\n"
              "(c = %d instead of 2)\n", c);
      exit (1);
    }

  mpfr_clear (x);
}

/* Bug in r7114 */
static void
test_macros (void)
{
  mpfr_t x[3];
  mpfr_ptr p;

  mpfr_inits (x[0], x[1], x[2], (mpfr_ptr) 0);
  mpfr_set_ui (x[0], 0, MPFR_RNDN);
  p = x[0];
  if (mpfr_cmp_ui (p++, 0) != 0)
    {
      printf ("Error in mpfr_cmp_ui macro: result should be 0.\n");
      exit (1);
    }
  if (p != x[1])
    {
      printf ("Error in mpfr_cmp_ui macro: p - x[0] = %d (expecting 1)\n",
              (int) (p - x[0]));
      exit (1);
    }
  p = x[0];
  if (mpfr_cmp_si (p++, 0) != 0)
    {
      printf ("Error in mpfr_cmp_si macro: result should be 0.\n");
      exit (1);
    }
  if (p != x[1])
    {
      printf ("Error in mpfr_cmp_si macro: p - x[0] = %d (expecting 1)\n",
              (int) (p - x[0]));
      exit (1);
    }
  mpfr_clears (x[0], x[1], x[2], (mpfr_ptr) 0);
}

int
main (void)
{
  mpfr_t x;
  unsigned long i;
  long s;

  tests_start_mpfr ();

  mpfr_init(x);

  /* tests for cmp_ui */
  mpfr_set_ui (x, 3, MPFR_RNDZ);
  if ((mpfr_cmp_ui) (x, i = 3) != 0)
    {
      printf ("Error in mpfr_cmp_ui(3.0, 3)\n");
      exit (1);
    }
  if (mpfr_cmp_ui (x, i = 2) <= 0)
    {
      printf ("Error in mpfr_cmp_ui(3.0,2)\n");
      exit (1);
    }
  if (mpfr_cmp_ui (x, i = 4) >= 0)
    {
      printf ("Error in mpfr_cmp_ui(3.0,4)\n");
      exit (1);
    }
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  mpfr_neg (x, x, MPFR_RNDZ);
  if (mpfr_cmp_ui (x, i = 0))
    {
      printf ("Error in mpfr_cmp_ui(0.0,0)\n");
      exit (1);
    }
  mpfr_set_ui (x, 1, MPFR_RNDZ);
  if (mpfr_cmp_ui (x, i = 0) == 0)
    {
      printf ("Error in mpfr_cmp_ui(1.0,0)\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  if (mpfr_cmp_ui (x, 1) <= 0)
    {
      printf ("Error in mpfr_cmp_ui (Inf, 0)\n");
      exit (1);
    }
  mpfr_set_inf (x, -1);
  if (mpfr_cmp_ui (x, 1) >= 0)
    {
      printf ("Error in mpfr_cmp_ui (-Inf, 0)\n");
      exit (1);
    }

  mpfr_set_si (x, -1, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 1) < 0);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 0) < 0);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 0) > 0);

  /* tests for cmp_si */
  (mpfr_set_si) (x, -3, MPFR_RNDZ);
  if ((mpfr_cmp_si) (x, s = -3) != 0)
    {
      printf ("Error in mpfr_cmp_si(-3.0,-3)\n");
      exit (1);
    }
  if (mpfr_cmp_si (x, s = -4) <= 0)
    {
      printf ("Error in mpfr_cmp_si(-3.0,-4)\n");
      exit (1);
    }
  if (mpfr_cmp_si (x, s = 1) >= 0)
    {
      printf ("Error in mpfr_cmp_si(-3.0,1)\n");
      exit (1);
    }

  mpfr_set_inf (x, 1);
  if (mpfr_cmp_si (x, -1) <= 0)
    {
      printf ("Error in mpfr_cmp_si (Inf, 0)\n");
      exit (1);
    }
  mpfr_set_inf (x, -1);
  if (mpfr_cmp_si (x, -1) >= 0)
    {
      printf ("Error in mpfr_cmp_si (-Inf, 0)\n");
      exit (1);
    }

  /* case b=0 */
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  MPFR_ASSERTN(mpfr_cmp_si (x, 0) == 0);
  MPFR_ASSERTN(mpfr_cmp_si (x, 1) < 0);
  MPFR_ASSERTN(mpfr_cmp_si (x, -1) > 0);

  /* case i=0 */
  mpfr_set_ui (x, 1, MPFR_RNDZ);
  MPFR_ASSERTN(mpfr_cmp_si (x, 0) > 0);
  mpfr_set_ui (x, 0, MPFR_RNDZ);
  MPFR_ASSERTN(mpfr_cmp_si (x, 0) == 0);
  mpfr_neg (x, x, MPFR_RNDZ);
  MPFR_ASSERTN(mpfr_cmp_si (x, 0) == 0);
  mpfr_set_si (x, -1, MPFR_RNDZ);
  MPFR_ASSERTN(mpfr_cmp_si (x, 0) < 0);

  /* case large x */
  mpfr_set_str_binary (x, "1E100");
  MPFR_ASSERTN(mpfr_cmp_si (x, 0) > 0);
  MPFR_ASSERTN(mpfr_cmp_si (x, 1) > 0);
  MPFR_ASSERTN(mpfr_cmp_si (x, -1) > 0);
  mpfr_set_str_binary (x, "-1E100");
  MPFR_ASSERTN(mpfr_cmp_si (x, 0) < 0);
  MPFR_ASSERTN(mpfr_cmp_si (x, 1) < 0);
  MPFR_ASSERTN(mpfr_cmp_si (x, -1) < 0);

  /* corner case */
  mpfr_set_ui (x, 1, MPFR_RNDZ);
  mpfr_mul_2exp (x, x, GMP_NUMB_BITS - 1, MPFR_RNDZ);
  /* now EXP(x)=GMP_NUMB_BITS */
  MPFR_ASSERTN(mpfr_cmp_si (x, 1) > 0);

  mpfr_clear (x);

  check_nan ();
  check_macros ();
  test_macros ();

  tests_end_mpfr ();
  return 0;
}

#endif
