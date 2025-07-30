/* Test file for
   mpfr_set_sj, mpfr_set_uj, mpfr_set_sj_2exp and mpfr_set_uj_2exp.

Copyright 2004, 2006-2017 Free Software Foundation, Inc.
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

#ifndef _MPFR_H_HAVE_INTMAX_T

int
main (void)
{
  return 77;
}

#else

#define ERROR(str) do { printf ("Error for " str "\n"); exit (1); } while (0)

static int
inexact_sign (int x)
{
  return (x < 0) ? -1 : (x > 0);
}

static void
check_set_uj (mpfr_prec_t pmin, mpfr_prec_t pmax, int N)
{
  mpfr_t x, y;
  mpfr_prec_t p;
  int inex1, inex2, n;
  mp_limb_t limb;

  mpfr_inits2 (pmax, x, y, (mpfr_ptr) 0);

  for ( p = pmin ; p < pmax ; p++)
    {
      mpfr_set_prec (x, p);
      mpfr_set_prec (y, p);
      for (n = 0 ; n < N ; n++)
        {
          /* mp_limb_t may be unsigned long long */
          limb = (unsigned long) randlimb ();
          inex1 = mpfr_set_uj (x, limb, MPFR_RNDN);
          inex2 = mpfr_set_ui (y, limb, MPFR_RNDN);
          if (mpfr_cmp (x, y))
            {
              printf ("ERROR for mpfr_set_uj and j=%lu and p=%lu\n",
                      (unsigned long) limb, (unsigned long) p);
              printf ("X="); mpfr_dump (x);
              printf ("Y="); mpfr_dump (y);
              exit (1);
            }
          if (inexact_sign (inex1) != inexact_sign (inex2))
            {
              printf ("ERROR for inexact(set_uj): j=%lu p=%lu\n"
                      "Inexact1= %d Inexact2= %d\n",
                      (unsigned long) limb, (unsigned long) p, inex1, inex2);
              exit (1);
            }
        }
    }
  /* Special case */
  mpfr_set_prec (x, sizeof(uintmax_t)*CHAR_BIT);
  inex1 = mpfr_set_uj (x, MPFR_UINTMAX_MAX, MPFR_RNDN);
  if (inex1 != 0 || mpfr_sgn(x) <= 0)
    ERROR ("inexact / UINTMAX_MAX");
  inex1 = mpfr_add_ui (x, x, 1, MPFR_RNDN);
  if (inex1 != 0 || !mpfr_powerof2_raw (x)
      || MPFR_EXP (x) != (sizeof(uintmax_t)*CHAR_BIT+1) )
    ERROR ("power of 2");
  mpfr_set_uj (x, 0, MPFR_RNDN);
  if (!MPFR_IS_ZERO (x))
    ERROR ("Setting 0");

  mpfr_clears (x, y, (mpfr_ptr) 0);
}

static void
check_set_uj_2exp (void)
{
  mpfr_t x;
  int inex;

  mpfr_init2 (x, sizeof(uintmax_t)*CHAR_BIT);

  inex = mpfr_set_uj_2exp (x, 1, 0, MPFR_RNDN);
  if (inex || mpfr_cmp_ui(x, 1))
    ERROR("(1U,0)");

  inex = mpfr_set_uj_2exp (x, 1024, -10, MPFR_RNDN);
  if (inex || mpfr_cmp_ui(x, 1))
    ERROR("(1024U,-10)");

  inex = mpfr_set_uj_2exp (x, 1024, 10, MPFR_RNDN);
  if (inex || mpfr_cmp_ui(x, 1024L * 1024L))
    ERROR("(1024U,+10)");

  inex = mpfr_set_uj_2exp (x, MPFR_UINTMAX_MAX, 1000, MPFR_RNDN);
  inex |= mpfr_div_2ui (x, x, 1000, MPFR_RNDN);
  inex |= mpfr_add_ui (x, x, 1, MPFR_RNDN);
  if (inex || !mpfr_powerof2_raw (x)
      || MPFR_EXP (x) != (sizeof(uintmax_t)*CHAR_BIT+1) )
    ERROR("(UINTMAX_MAX)");

  inex = mpfr_set_uj_2exp (x, MPFR_UINTMAX_MAX, MPFR_EMAX_MAX-10, MPFR_RNDN);
  if (inex == 0 || !mpfr_inf_p (x))
    ERROR ("Overflow");

  inex = mpfr_set_uj_2exp (x, MPFR_UINTMAX_MAX, MPFR_EMIN_MIN-1000, MPFR_RNDN);
  if (inex == 0 || !MPFR_IS_ZERO (x))
    ERROR ("Underflow");

  mpfr_clear (x);
}

static void
check_set_sj (void)
{
  mpfr_t x;
  int inex;

  mpfr_init2 (x, sizeof(intmax_t)*CHAR_BIT-1);

  inex = mpfr_set_sj (x, -MPFR_INTMAX_MAX, MPFR_RNDN);
  inex |= mpfr_add_si (x, x, -1, MPFR_RNDN);
  if (inex || mpfr_sgn (x) >=0 || !mpfr_powerof2_raw (x)
      || MPFR_EXP (x) != (sizeof(intmax_t)*CHAR_BIT) )
    ERROR("set_sj (-INTMAX_MAX)");

  inex = mpfr_set_sj (x, 1742, MPFR_RNDN);
  if (inex || mpfr_cmp_ui (x, 1742))
    ERROR ("set_sj (1742)");

  mpfr_clear (x);
}

static void
check_set_sj_2exp (void)
{
  mpfr_t x;
  int inex;

  mpfr_init2 (x, sizeof(intmax_t)*CHAR_BIT-1);

  inex = mpfr_set_sj_2exp (x, MPFR_INTMAX_MIN, 1000, MPFR_RNDN);
  if (inex || mpfr_sgn (x) >=0 || !mpfr_powerof2_raw (x)
      || MPFR_EXP (x) != (sizeof(intmax_t)*CHAR_BIT+1000) )
    ERROR("set_sj_2exp (INTMAX_MIN)");

  mpfr_clear (x);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  check_set_uj (2, 128, 50);
  check_set_uj_2exp ();
  check_set_sj ();
  check_set_sj_2exp ();

  tests_end_mpfr ();
  return 0;
}

#endif
