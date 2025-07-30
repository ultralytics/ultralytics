/* Test file for mpfr_sin_cos.

Copyright 2000-2004, 2006-2017 Free Software Foundation, Inc.
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

static void
large_test (char *X, int prec, int N)
{
  int i;
  mpfr_t x, s, c;

  mpfr_init2 (x, prec);
  mpfr_init2 (s, prec);
  mpfr_init2 (c, prec);
  mpfr_set_str (x, X, 10, MPFR_RNDN);

  for (i = 0; i < N; i++)
    mpfr_sin_cos (s, c, x, MPFR_RNDN);

  mpfr_clear (x);
  mpfr_clear (s);
  mpfr_clear (c);
}

static void
check53 (const char *xs, const char *sin_xs, const char *cos_xs,
         mpfr_rnd_t rnd_mode)
{
  mpfr_t xx, s, c;

  mpfr_inits2 (53, xx, s, c, (mpfr_ptr) 0);
  mpfr_set_str1 (xx, xs); /* should be exact */
  mpfr_sin_cos (s, c, xx, rnd_mode);
  if (mpfr_cmp_str1 (s, sin_xs))
    {
      printf ("mpfr_sin_cos failed for x=%s, rnd=%s\n",
              xs, mpfr_print_rnd_mode (rnd_mode));
      printf ("mpfr_sin_cos gives sin(x)=");
      mpfr_out_str(stdout, 10, 0, s, MPFR_RNDN);
      printf(", expected %s\n", sin_xs);
      exit (1);
    }
  if (mpfr_cmp_str1 (c, cos_xs))
    {
      printf ("mpfr_sin_cos failed for x=%s, rnd=%s\n",
              xs, mpfr_print_rnd_mode (rnd_mode));
      printf ("mpfr_sin_cos gives cos(x)=");
      mpfr_out_str(stdout, 10, 0, c, MPFR_RNDN);
      printf(", expected %s\n", cos_xs);
      exit (1);
    }
  mpfr_clears (xx, s, c, (mpfr_ptr) 0);
}

static void
check53sin (const char *xs, const char *sin_xs, mpfr_rnd_t rnd_mode)
{
  mpfr_t xx, s, c;

  mpfr_inits2 (53, xx, s, c, (mpfr_ptr) 0);
  mpfr_set_str1 (xx, xs); /* should be exact */
  mpfr_sin_cos (s, c, xx, rnd_mode);
  if (mpfr_cmp_str1 (s, sin_xs))
    {
      printf ("mpfr_sin_cos failed for x=%s, rnd=%s\n",
              xs, mpfr_print_rnd_mode (rnd_mode));
      printf ("mpfr_sin_cos gives sin(x)=");
      mpfr_out_str(stdout, 10, 0, s, MPFR_RNDN);
      printf(", expected %s\n", sin_xs);
      exit (1);
    }
  mpfr_clears (xx, s, c, (mpfr_ptr) 0);
}

static void
check53cos (const char *xs, const char *cos_xs, mpfr_rnd_t rnd_mode)
{
  mpfr_t xx, c, s;

  mpfr_inits2 (53, xx, s, c, (mpfr_ptr) 0);
  mpfr_set_str1 (xx, xs); /* should be exact */
  mpfr_sin_cos (s, c, xx, rnd_mode);
  if (mpfr_cmp_str1 (c, cos_xs))
    {
      printf ("mpfr_sin_cos failed for x=%s, rnd=%s\n",
              xs, mpfr_print_rnd_mode (rnd_mode));
      printf ("mpfr_sin_cos gives cos(x)=");
      mpfr_out_str(stdout, 10, 0, c, MPFR_RNDN);
      printf(", expected %s\n", cos_xs);
      exit (1);
    }
  mpfr_clears (xx, s, c, (mpfr_ptr) 0);
}

static void
check_nans (void)
{
  mpfr_t  x, s, c;

  mpfr_init2 (x, 123L);
  mpfr_init2 (s, 123L);
  mpfr_init2 (c, 123L);

  /* sin(NaN)==NaN, cos(NaN)==NaN */
  mpfr_set_nan (x);
  mpfr_sin_cos (s, c, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (s));
  MPFR_ASSERTN (mpfr_nan_p (c));

  /* sin(+Inf)==NaN, cos(+Inf)==NaN */
  mpfr_set_inf (x, 1);
  mpfr_sin_cos (s, c, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (s));
  MPFR_ASSERTN (mpfr_nan_p (c));

  /* sin(-Inf)==NaN, cos(-Inf)==NaN */
  mpfr_set_inf (x, -1);
  mpfr_sin_cos (s, c, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (s));
  MPFR_ASSERTN (mpfr_nan_p (c));

  /* check zero */
  mpfr_set_ui  (x, 0, MPFR_RNDN);
  mpfr_sin_cos (s, c, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (s, 0) == 0 && MPFR_IS_POS (s));
  MPFR_ASSERTN (mpfr_cmp_ui (c, 1) == 0);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_sin_cos (s, c, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (s, 0) == 0 && MPFR_IS_NEG (s));
  MPFR_ASSERTN (mpfr_cmp_ui (c, 1) == 0);

  /* coverage test */
  mpfr_set_prec (x, 2);
  mpfr_set_ui (x, 4, MPFR_RNDN);
  mpfr_set_prec (s, 2);
  mpfr_set_prec (c, 2);
  mpfr_sin_cos (s, c, x, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_si_2exp (s, -3, -2) == 0);
  MPFR_ASSERTN (mpfr_cmp_si_2exp (c, -3, -2) == 0);

  mpfr_clear (x);
  mpfr_clear (s);
  mpfr_clear (c);
}

static void
overflowed_sin_cos0 (void)
{
  mpfr_t x, y, z;
  int emax, inex, rnd, err = 0;
  mpfr_exp_t old_emax;

  old_emax = mpfr_get_emax ();

  mpfr_init2 (x, 8);
  mpfr_init2 (y, 8);
  mpfr_init2 (z, 8);

  for (emax = -1; emax <= 0; emax++)
    {
      mpfr_set_ui_2exp (z, 1, emax, MPFR_RNDN);
      mpfr_nextbelow (z);
      set_emax (emax);  /* 1 is not representable. */
      /* and if emax < 0, 1 - eps is not representable either. */
      RND_LOOP (rnd)
        {
          mpfr_set_si (x, 0, MPFR_RNDN);
          mpfr_neg (x, x, MPFR_RNDN);
          mpfr_clear_flags ();
          inex = mpfr_sin_cos (x, y, x, (mpfr_rnd_t) rnd);
          if (! mpfr_overflow_p ())
            {
              printf ("Error in overflowed_sin_cos0 (rnd = %s):\n"
                      "  The overflow flag is not set.\n",
                      mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
              err = 1;
            }
          if (! (mpfr_zero_p (x) && MPFR_SIGN (x) < 0))
            {
              printf ("Error in overflowed_sin_cos0 (rnd = %s):\n"
                      "  Got sin = ", mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
              mpfr_print_binary (x);
              printf (" instead of -0.\n");
              err = 1;
            }
          if (rnd == MPFR_RNDZ || rnd == MPFR_RNDD)
            {
              if (inex == 0)
                {
                  printf ("Error in overflowed_sin_cos0 (rnd = %s):\n"
                          "  The inexact value must be non-zero.\n",
                          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  err = 1;
                }
              if (! mpfr_equal_p (y, z))
                {
                  printf ("Error in overflowed_sin_cos0 (rnd = %s):\n"
                          "  Got cos = ",
                          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  mpfr_print_binary (y);
                  printf (" instead of 0.11111111E%d.\n", emax);
                  err = 1;
                }
            }
          else
            {
              if (inex == 0)
                {
                  printf ("Error in overflowed_sin_cos0 (rnd = %s):\n"
                          "  The inexact value must be non-zero.\n",
                          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  err = 1;
                }
              if (! (mpfr_inf_p (y) && MPFR_SIGN (y) > 0))
                {
                  printf ("Error in overflowed_sin_cos0 (rnd = %s):\n"
                          "  Got cos = ",
                          mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                  mpfr_print_binary (y);
                  printf (" instead of +Inf.\n");
                  err = 1;
                }
            }
        }
      set_emax (old_emax);
    }

  if (err)
    exit (1);
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

static void
tiny (void)
{
  mpfr_t x, s, c;
  int i, inex;

  mpfr_inits2 (64, x, s, c, (mpfr_ptr) 0);

  for (i = -1; i <= 1; i += 2)
    {
      mpfr_set_si (x, i, MPFR_RNDN);
      mpfr_set_exp (x, mpfr_get_emin ());
      inex = mpfr_sin_cos (s, c, x, MPFR_RNDN);
      MPFR_ASSERTN (inex != 0);
      MPFR_ASSERTN (mpfr_equal_p (s, x));
      MPFR_ASSERTN (!mpfr_nan_p (c) && mpfr_cmp_ui (c, 1) == 0);
    }

  mpfr_clears (x, s, c, (mpfr_ptr) 0);
}

/* bug found in nightly tests */
static void
test20071214 (void)
{
  mpfr_t a, b;
  int inex;

  mpfr_init2 (a, 4);
  mpfr_init2 (b, 4);

  mpfr_set_ui_2exp (a, 3, -4, MPFR_RNDN);
  inex = mpfr_sin_cos (a, b, a, MPFR_RNDD);
  MPFR_ASSERTN(mpfr_cmp_ui_2exp (a, 11, -6) == 0);
  MPFR_ASSERTN(mpfr_cmp_ui_2exp (b, 15, -4) == 0);
  MPFR_ASSERTN(inex == 10);

  mpfr_set_ui_2exp (a, 3, -4, MPFR_RNDN);
  inex = mpfr_sin_cos (a, b, a, MPFR_RNDU);
  MPFR_ASSERTN(mpfr_cmp_ui_2exp (a, 3, -4) == 0);
  MPFR_ASSERTN(mpfr_cmp_ui (b, 1) == 0);
  MPFR_ASSERTN(inex == 5);

  mpfr_set_ui_2exp (a, 3, -4, MPFR_RNDN);
  inex = mpfr_sin_cos (a, b, a, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui_2exp (a, 3, -4) == 0);
  MPFR_ASSERTN(mpfr_cmp_ui (b, 1) == 0);
  MPFR_ASSERTN(inex == 5);

  mpfr_clear (a);
  mpfr_clear (b);
}

/* check that mpfr_sin_cos and test_mpfr_sincos_fast agree */
static void
test_mpfr_sincos_fast (void)
{
  mpfr_t x, y, z, yref, zref, h;
  mpfr_prec_t p = 1000;
  int i, inex, inexref;
  mpfr_rnd_t r;

  mpfr_init2 (x, p);
  mpfr_init2 (y, p);
  mpfr_init2 (z, p);
  mpfr_init2 (yref, p);
  mpfr_init2 (zref, p);
  mpfr_init2 (h, p);
  mpfr_set_ui (x, 0, MPFR_RNDN);
  /* we generate a random value x, compute sin(x) and cos(x) with both
     mpfr_sin_cos and mpfr_sincos_fast, and check the values and the flags
     agree */
  for (i = 0; i < 100; i++)
    {
      mpfr_urandomb (h, RANDS);
      mpfr_add (x, x, h, MPFR_RNDN);
      r = RND_RAND ();
      inexref = mpfr_sin_cos (yref, zref, x, r);
      inex = mpfr_sincos_fast (y, z, x, r);
      if (mpfr_cmp (y, yref))
        {
          printf ("mpfr_sin_cos and mpfr_sincos_fast disagree\n");
          printf ("x="); mpfr_dump (x);
          printf ("rnd=%s\n", mpfr_print_rnd_mode (r));
          printf ("yref="); mpfr_dump (yref);
          printf ("y="); mpfr_dump (y);
          exit (1);
        }
      if (mpfr_cmp (z, zref))
        {
          printf ("mpfr_sin_cos and mpfr_sincos_fast disagree\n");
          printf ("x="); mpfr_dump (x);
          printf ("rnd=%s\n", mpfr_print_rnd_mode (r));
          printf ("zref="); mpfr_dump (zref);
          printf ("z="); mpfr_dump (z);
          exit (1);
        }
      if (inex != inexref)
        {
          printf ("mpfr_sin_cos and mpfr_sincos_fast disagree\n");
          printf ("x="); mpfr_dump (x);
          printf ("rnd=%s\n", mpfr_print_rnd_mode (r));
          printf ("inexref=%d inex=%d\n", inexref, inex);
          exit (1);
        }
    }
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (yref);
  mpfr_clear (zref);
  mpfr_clear (h);
}

static void
bug20091007 (void)
{
  mpfr_t x, y, z, yref, zref;
  mpfr_prec_t p = 1000;
  int inex, inexref;
  mpfr_rnd_t r = MPFR_RNDZ;

  mpfr_init2 (x, p);
  mpfr_init2 (y, p);
  mpfr_init2 (z, p);
  mpfr_init2 (yref, p);
  mpfr_init2 (zref, p);

  mpfr_set_str (x, "1.9ecdc22ba77a5ab2560f7e84289e2a328906f47377ea3fd4c82d1bb2f13ee05c032cffc1933eadab7b0a5498e03e3bd0508968e59c25829d97a0b54f20cd4662c8dfffa54e714de41fc8ee3e0e0b244d110a194db05b70022b7d77f88955d415b09f17dd404576098dc51a583a3e49c35839551646e880c7eb790a01a4@1", 16, MPFR_RNDN);
  inexref = mpfr_sin_cos (yref, zref, x, r);
  inex = mpfr_sincos_fast (y, z, x, r);

  if (mpfr_cmp (y, yref))
    {
      printf ("mpfr_sin_cos and mpfr_sincos_fast disagree (bug20091007)\n");
      printf ("yref="); mpfr_dump (yref);
      printf ("y="); mpfr_dump (y);
      exit (1);
    }
  if (mpfr_cmp (z, zref))
    {
      printf ("mpfr_sin_cos and mpfr_sincos_fast disagree (bug20091007)\n");
      printf ("zref="); mpfr_dump (zref);
      printf ("z="); mpfr_dump (z);
      exit (1);
    }
  if (inex != inexref)
    {
      printf ("mpfr_sin_cos and mpfr_sincos_fast disagree (bug20091007)\n");
      printf ("inexref=%d inex=%d\n", inexref, inex);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (yref);
  mpfr_clear (zref);
}

/* Note: with the sin_cos.c code before r6507, the disagreement occurs
   only on the return ("inexact") value, which is new in r6444. */
static void
bug20091008 (void)
{
  mpfr_t x, y, z, yref, zref;
  mpfr_prec_t p = 1000;
  int inex, inexref;
  mpfr_rnd_t r = MPFR_RNDN;

  mpfr_init2 (x, p);
  mpfr_init2 (y, p);
  mpfr_init2 (z, p);
  mpfr_init2 (yref, p);
  mpfr_init2 (zref, p);

  mpfr_set_str (x, "c.91813724e28ef6a711d33e6505984699daef7fe93636c1ed5d0168bc96989cc6802f7f9e405c902ec62fb90cd39c9d21084c8ad8b5af4c4aa87bf402e2e4a78e6fe1ffeb6dbbbdbbc2983c196c518966ccc1e094ed39ee77984ef2428069d65de37928e75247edbe7007245e682616b5ebbf05f2fdefc74ad192024f10", 16, MPFR_RNDN);
  inexref = mpfr_sin_cos (yref, zref, x, r);
  inex = mpfr_sincos_fast (y, z, x, r);

  if (mpfr_cmp (y, yref))
    {
      printf ("mpfr_sin_cos and mpfr_sincos_fast disagree (bug20091008)\n");
      printf ("yref="); mpfr_dump (yref);
      printf ("y="); mpfr_dump (y);
      exit (1);
    }
  if (mpfr_cmp (z, zref))
    {
      printf ("mpfr_sin_cos and mpfr_sincos_fast disagree (bug20091008)\n");
      printf ("zref="); mpfr_dump (zref);
      printf ("z="); mpfr_dump (z);
      exit (1);
    }
  /* sin(x) is rounded up, cos(x) is rounded up too, thus we should get 5
     for the return value */
  if (inex != inexref)
    {
      printf ("mpfr_sin_cos and mpfr_sincos_fast disagree (bug20091008)\n");
      printf ("inexref=%d inex=%d (5 expected)\n", inexref, inex);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (yref);
  mpfr_clear (zref);
}

static void
bug20091013 (void)
{
  mpfr_t x, y, z, yref, zref;
  mpfr_prec_t p = 1000;
  int inex, inexref;
  mpfr_rnd_t r = MPFR_RNDN;

  mpfr_init2 (x, p);
  mpfr_init2 (y, p);
  mpfr_init2 (z, p);
  mpfr_init2 (yref, p);
  mpfr_init2 (zref, p);

  mpfr_set_str (x, "3.240ff3fdcb1ee7cd667b96287593ae24e20fb63ed7c2d5bf4bd0f2cc5509283b04e7628e66382605f14ed5967cef15296041539a1bdaa626c777c7fbb6f2068414759b78cee14f37848689b3a170f583656be4e0837f464d8210556a3a822d4ecfdd59f4e0d5fdb76bf7e15b8a57234e2160b98e14c17bbdf27c4643b8@1", 16, MPFR_RNDN);
  inexref = mpfr_sin_cos (yref, zref, x, r);
  inex = mpfr_sincos_fast (y, z, x, r);

  if (mpfr_cmp (y, yref))
    {
      printf ("mpfr_sin_cos and mpfr_sincos_fast disagree (bug20091013)\n");
      printf ("yref="); mpfr_dump (yref);
      printf ("y="); mpfr_dump (y);
      exit (1);
    }
  if (mpfr_cmp (z, zref))
    {
      printf ("mpfr_sin_cos and mpfr_sincos_fast disagree (bug20091013)\n");
      printf ("zref="); mpfr_dump (zref);
      printf ("z="); mpfr_dump (z);
      exit (1);
    }
  /* sin(x) is rounded down and cos(x) is rounded down, thus we should get
     2+4*2 = 10 as return value */
  if (inex != inexref)
    {
      printf ("mpfr_sin_cos and mpfr_sincos_fast disagree (bug20091013)\n");
      printf ("inexref=%d inex=%d (10 expected)\n", inexref, inex);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (yref);
  mpfr_clear (zref);
}

/* Bug reported by Laurent Fousse for the 2.4 branch.
   No problem in the trunk.
   https://sympa.inria.fr/sympa/arc/mpfr/2009-11/msg00044.html */
static void
bug20091122 (void)
{
  mpfr_t x, y, z, yref, zref;
  mpfr_prec_t p = 3;
  mpfr_rnd_t r = MPFR_RNDN;

  mpfr_init2 (x, 5);
  mpfr_init2 (y, p);
  mpfr_init2 (z, p);
  mpfr_init2 (yref, p);
  mpfr_init2 (zref, p);

  mpfr_set_str (x, "0.11111E49", 2, MPFR_RNDN);
  mpfr_sin_cos (yref, zref, x, r);

  mpfr_sin (y, x, r);
  mpfr_cos (z, x, r);

  if (! mpfr_equal_p (y, yref))
    {
      printf ("mpfr_sin_cos and mpfr_sin disagree (bug20091122)\n");
      printf ("yref = "); mpfr_dump (yref);
      printf ("y    = "); mpfr_dump (y);
      exit (1);
    }
  if (! mpfr_equal_p (z, zref))
    {
      printf ("mpfr_sin_cos and mpfr_cos disagree (bug20091122)\n");
      printf ("zref = "); mpfr_dump (zref);
      printf ("z    = "); mpfr_dump (z);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (yref);
  mpfr_clear (zref);
}

static void
consistency (void)
{
  mpfr_t x, s1, s2, c1, c2;
  mpfr_exp_t emin, emax;
  mpfr_rnd_t rnd;
  unsigned int flags_sin, flags_cos, flags, flags_before, flags_ref;
  int inex_sin, is, inex_cos, ic, inex, inex_ref;
  int i;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  for (i = 0; i <= 10000; i++)
    {
      mpfr_init2 (x, MPFR_PREC_MIN + (randlimb () % 8));
      mpfr_inits2 (MPFR_PREC_MIN + (randlimb () % 8), s1, s2, c1, c2,
                   (mpfr_ptr) 0);
      if (i < 8 * MPFR_RND_MAX)
        {
          int j = i / MPFR_RND_MAX;
          if (j & 1)
            mpfr_set_emin (MPFR_EMIN_MIN);
          mpfr_set_si (x, (j & 2) ? 1 : -1, MPFR_RNDN);
          mpfr_set_exp (x, mpfr_get_emin ());
          rnd = (mpfr_rnd_t) (i % MPFR_RND_MAX);
          flags_before = 0;
          if (j & 4)
            mpfr_set_emax (-17);
        }
      else
        {
          tests_default_random (x, 256, -5, 50, 0);
          rnd = RND_RAND ();
          flags_before = (randlimb () & 1) ?
            (unsigned int) (MPFR_FLAGS_ALL ^ MPFR_FLAGS_ERANGE) :
            (unsigned int) 0;
        }
      __gmpfr_flags = flags_before;
      inex_sin = mpfr_sin (s1, x, rnd);
      is = inex_sin < 0 ? 2 : inex_sin > 0 ? 1 : 0;
      flags_sin = __gmpfr_flags;
      __gmpfr_flags = flags_before;
      inex_cos = mpfr_cos (c1, x, rnd);
      ic = inex_cos < 0 ? 2 : inex_cos > 0 ? 1 : 0;
      flags_cos = __gmpfr_flags;
      __gmpfr_flags = flags_before;
      inex = mpfr_sin_cos (s2, c2, x, rnd);
      flags = __gmpfr_flags;
      inex_ref = is + 4 * ic;
      flags_ref = flags_sin | flags_cos;
      if (!(mpfr_equal_p (s1, s2) && mpfr_equal_p (c1, c2)) ||
          inex != inex_ref || flags != flags_ref)
        {
          printf ("mpfr_sin_cos and mpfr_sin/mpfr_cos disagree on %s,"
                  " i = %d\nx = ", mpfr_print_rnd_mode (rnd), i);
          mpfr_dump (x);
          printf ("s1 = ");
          mpfr_dump (s1);
          printf ("s2 = ");
          mpfr_dump (s2);
          printf ("c1 = ");
          mpfr_dump (c1);
          printf ("c2 = ");
          mpfr_dump (c2);
          printf ("inex_sin = %d (s = %d), inex_cos = %d (c = %d), "
                  "inex = %d (expected %d)\n",
                  inex_sin, is, inex_cos, ic, inex, inex_ref);
          printf ("flags_sin = 0x%x, flags_cos = 0x%x, "
                  "flags = 0x%x (expected 0x%x)\n",
                  flags_sin, flags_cos, flags, flags_ref);
          exit (1);
        }
      mpfr_clears (x, s1, s2, c1, c2, (mpfr_ptr) 0);
      mpfr_set_emin (emin);
      mpfr_set_emax (emax);
    }
}

static void
coverage_01032011 (void)
{
  mpfr_t val, cval, sval, svalf;
  int status_f, status;

  mpfr_init2 (val, MPFR_PREC_MIN);
  mpfr_init2 (cval, MPFR_PREC_MIN);
  mpfr_init2 (sval, MPFR_PREC_MIN);
  mpfr_init2 (svalf, MPFR_PREC_MIN);

  mpfr_set_str1 (val, "-0.7");

  status_f = mpfr_sincos_fast (svalf, NULL, val, MPFR_RNDN);
  status = mpfr_sin_cos (sval, cval, val, MPFR_RNDN);
  if (! mpfr_equal_p (svalf, sval) || SIGN (status_f) != SIGN (status))
    {
      printf ("mpfr_sincos_fast differ from mpfr_sin_cos result:\n"
              " sin fast is ");
      mpfr_dump (svalf);
      printf (" sin is ");
      mpfr_dump (sval);
      printf ("status_f = %d, status = %d\n", status_f, status);
      exit (1);
    }

  mpfr_clears(val, cval, sval, svalf, (mpfr_ptr) 0);
}

/* tsin_cos prec [N] performs N tests with prec bits */
int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  if (argc > 1)
    {
      if (argc != 3 && argc != 4)
        {
          fprintf (stderr, "Usage: tsin_cos x prec [n]\n");
          exit (1);
        }
      large_test (argv[1], atoi (argv[2]), (argc > 3) ? atoi (argv[3]) : 1);
      goto end;
    }

  bug20091013 ();
  bug20091008 ();
  bug20091007 ();
  bug20091122 ();
  consistency ();

  test_mpfr_sincos_fast ();

  check_nans ();

  /* worst case from PhD thesis of Vincent Lefe`vre: x=8980155785351021/2^54 */
  check53 ("4.984987858808754279e-1", "4.781075595393330379e-1",
           "8.783012931285841817e-1", MPFR_RNDN);
  check53 ("4.984987858808754279e-1", "4.781075595393329824e-1",
           "8.783012931285840707e-1", MPFR_RNDD);
  check53 ("4.984987858808754279e-1", "4.781075595393329824e-1",
           "8.783012931285840707e-1", MPFR_RNDZ);
  check53 ("4.984987858808754279e-1", "4.781075595393330379e-1",
           "8.783012931285841817e-1", MPFR_RNDU);
  check53 ("1.00031274099908640274",  "8.416399183372403892e-1",
           "0.540039116973283217504", MPFR_RNDN);
  check53 ("1.00229256850978698523",  "8.427074524447979442e-1",
           "0.538371757797526551137", MPFR_RNDZ);
  check53 ("1.00288304857059840103",  "8.430252033025980029e-1",
           "0.537874062022526966409", MPFR_RNDZ);
  check53 ("1.00591265847407274059",  "8.446508805292128885e-1",
           "0.53531755997839769456",  MPFR_RNDN);

  /* check one argument only */
  check53sin ("1.00591265847407274059", "8.446508805292128885e-1", MPFR_RNDN);
  check53cos ("1.00591265847407274059", "0.53531755997839769456",  MPFR_RNDN);

  overflowed_sin_cos0 ();
  tiny ();
  test20071214 ();

  coverage_01032011 ();

 end:
  tests_end_mpfr ();
  return 0;
}
