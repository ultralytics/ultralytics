/* Test file for mpfr_get_sj and mpfr_get_uj.

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

#include "mpfr-intmax.h"
#include "mpfr-test.h"

#ifndef _MPFR_H_HAVE_INTMAX_T

int
main (void)
{
  return 77;
}

#else

static void
check_sj (intmax_t s, mpfr_ptr x)
{
  mpfr_t y;
  int i;

  mpfr_init2 (y, MPFR_PREC (x));

  for (i = -1; i <= 1; i++)
    {
      int rnd;

      mpfr_set_si_2exp (y, i, -2, MPFR_RNDN);
      mpfr_add (y, y, x, MPFR_RNDN);
      for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
        {
          intmax_t r;

          if (rnd == MPFR_RNDZ && i < 0 && s >= 0)
            continue;
          if (rnd == MPFR_RNDZ && i > 0 && s <= 0)
            continue;
          if (rnd == MPFR_RNDD && i < 0)
            continue;
          if (rnd == MPFR_RNDU && i > 0)
            continue;
          if (rnd == MPFR_RNDA && ((MPFR_IS_POS(y) && i > 0) ||
                                  (MPFR_IS_NEG(y) && i < 0)))
            continue;
          /* rint (y) == x == s */
          r = mpfr_get_sj (y, (mpfr_rnd_t) rnd);
          if (r != s)
            {
              printf ("Error in check_sj for y = ");
              mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
              printf (" in %s\n", mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
              printf ("Got %jd instead of %jd.\n", r, s);
              exit (1);
            }
        }
    }

  mpfr_clear (y);
}

static void
check_uj (uintmax_t u, mpfr_ptr x)
{
  mpfr_t y;
  int i;

  mpfr_init2 (y, MPFR_PREC (x));

  for (i = -1; i <= 1; i++)
    {
      int rnd;

      mpfr_set_si_2exp (y, i, -2, MPFR_RNDN);
      mpfr_add (y, y, x, MPFR_RNDN);
      for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
        {
          uintmax_t r;

          if (rnd == MPFR_RNDZ && i < 0)
            continue;
          if (rnd == MPFR_RNDD && i < 0)
            continue;
          if (rnd == MPFR_RNDU && i > 0)
            continue;
          if (rnd == MPFR_RNDA && ((MPFR_IS_POS(y) && i > 0) ||
                                  (MPFR_IS_NEG(y) && i < 0)))
            continue;
          /* rint (y) == x == u */
          r = mpfr_get_uj (y, (mpfr_rnd_t) rnd);
          if (r != u)
            {
              printf ("Error in check_uj for y = ");
              mpfr_out_str (stdout, 2, 0, y, MPFR_RNDN);
              printf (" in %s\n", mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
              printf ("Got %ju instead of %ju.\n", r, u);
              exit (1);
            }
        }
    }

  mpfr_clear (y);
}

static void
check_erange (void)
{
  mpfr_t x;
  uintmax_t dl;
  intmax_t d;

  /* Test for ERANGE flag + correct behaviour if overflow */

  mpfr_init2 (x, 256);
  mpfr_set_uj (x, MPFR_UINTMAX_MAX, MPFR_RNDN);
  mpfr_clear_erangeflag ();
  dl = mpfr_get_uj (x, MPFR_RNDN);
  if (dl != MPFR_UINTMAX_MAX || mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_uj + ERANGE + UINTMAX_MAX (1)\n");
      exit (1);
    }
  mpfr_add_ui (x, x, 1, MPFR_RNDN);
  dl = mpfr_get_uj (x, MPFR_RNDN);
  if (dl != MPFR_UINTMAX_MAX || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_uj + ERANGE + UINTMAX_MAX (2)\n");
      exit (1);
    }
  mpfr_set_sj (x, -1, MPFR_RNDN);
  mpfr_clear_erangeflag ();
  dl = mpfr_get_uj (x, MPFR_RNDN);
  if (dl != 0 || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_uj + ERANGE + -1 \n");
      exit (1);
    }
  mpfr_set_sj (x, MPFR_INTMAX_MAX, MPFR_RNDN);
  mpfr_clear_erangeflag ();
  d = mpfr_get_sj (x, MPFR_RNDN);
  if (d != MPFR_INTMAX_MAX || mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_sj + ERANGE + INTMAX_MAX (1)\n");
      exit (1);
    }
  mpfr_add_ui (x, x, 1, MPFR_RNDN);
  d = mpfr_get_sj (x, MPFR_RNDN);
  if (d != MPFR_INTMAX_MAX || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_sj + ERANGE + INTMAX_MAX (2)\n");
      exit (1);
    }
  mpfr_set_sj (x, MPFR_INTMAX_MIN, MPFR_RNDN);
  mpfr_clear_erangeflag ();
  d = mpfr_get_sj (x, MPFR_RNDN);
  if (d != MPFR_INTMAX_MIN || mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_sj + ERANGE + INTMAX_MIN (1)\n");
      exit (1);
    }
  mpfr_sub_ui (x, x, 1, MPFR_RNDN);
  d = mpfr_get_sj (x, MPFR_RNDN);
  if (d != MPFR_INTMAX_MIN || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_sj + ERANGE + INTMAX_MIN (2)\n");
      exit (1);
    }

  mpfr_set_nan (x);
  mpfr_clear_erangeflag ();
  d = mpfr_get_uj (x, MPFR_RNDN);
  if (d != 0 || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_uj + NaN\n");
      exit (1);
    }
  mpfr_clear_erangeflag ();
  d = mpfr_get_sj (x, MPFR_RNDN);
  if (d != 0 || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_sj + NaN\n");
      exit (1);
    }

  mpfr_clear (x);
}

int
main (void)
{
  mpfr_prec_t prec;
  mpfr_t x, y;
  intmax_t s;
  uintmax_t u;

  tests_start_mpfr ();

  for (u = MPFR_UINTMAX_MAX, prec = 0; u != 0; u /= 2, prec++)
    { }

  mpfr_init2 (x, prec + 4);
  mpfr_init2 (y, prec + 4);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  check_sj (0, x);
  check_uj (0, x);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  check_sj (1, x);
  check_uj (1, x);

  mpfr_neg (x, x, MPFR_RNDN);
  check_sj (-1, x);

  mpfr_set_si_2exp (x, 1, prec, MPFR_RNDN);
  mpfr_sub_ui (x, x, 1, MPFR_RNDN); /* UINTMAX_MAX */

  mpfr_div_ui (y, x, 2, MPFR_RNDZ);
  mpfr_trunc (y, y); /* INTMAX_MAX */
  for (s = MPFR_INTMAX_MAX; s != 0; s /= 17)
    {
      check_sj (s, y);
      mpfr_div_ui (y, y, 17, MPFR_RNDZ);
      mpfr_trunc (y, y);
    }

  mpfr_div_ui (y, x, 2, MPFR_RNDZ);
  mpfr_trunc (y, y); /* INTMAX_MAX */
  mpfr_neg (y, y, MPFR_RNDN);
  if (MPFR_INTMAX_MIN + MPFR_INTMAX_MAX != 0)
    mpfr_sub_ui (y, y, 1, MPFR_RNDN); /* INTMAX_MIN */
  for (s = MPFR_INTMAX_MIN; s != 0; s /= 17)
    {
      check_sj (s, y);
      mpfr_div_ui (y, y, 17, MPFR_RNDZ);
      mpfr_trunc (y, y);
    }

  for (u = MPFR_UINTMAX_MAX; u != 0; u /= 17)
    {
      check_uj (u, x);
      mpfr_div_ui (x, x, 17, MPFR_RNDZ);
      mpfr_trunc (x, x);
    }

  mpfr_clear (x);
  mpfr_clear (y);

  check_erange ();

  tests_end_mpfr ();
  return 0;
}

#endif
