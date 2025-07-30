/* Test file for mpfr_set_si and mpfr_set_ui.

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

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "mpfr-test.h"

#define ERROR(str) do { printf ("Error for " str "\n"); exit (1); } while (0)

static void
test_2exp (void)
{
  mpfr_t x;
  int res;

  mpfr_init2 (x, 32);

  mpfr_set_ui_2exp (x, 1, 0, MPFR_RNDN);
  if (mpfr_cmp_ui(x, 1))
    ERROR("(1U,0)");

  mpfr_set_ui_2exp (x, 1024, -10, MPFR_RNDN);
  if (mpfr_cmp_ui(x, 1))
    ERROR("(1024U,-10)");

  mpfr_set_ui_2exp (x, 1024, 10, MPFR_RNDN);
  if (mpfr_cmp_ui(x, 1024*1024))
    ERROR("(1024U,+10)");

  mpfr_set_si_2exp (x, -1024L * 1024L, -10, MPFR_RNDN);
  if (mpfr_cmp_si(x, -1024))
    ERROR("(1M,-10)");

  mpfr_set_ui_2exp (x, 0x92345678, 16, MPFR_RNDN);
  if (mpfr_cmp_str (x, "92345678@4", 16, MPFR_RNDN))
    ERROR("(x92345678U,+16)");

  mpfr_set_si_2exp (x, -0x1ABCDEF0, -256, MPFR_RNDN);
  if (mpfr_cmp_str (x, "-1ABCDEF0@-64", 16, MPFR_RNDN))
    ERROR("(-x1ABCDEF0,-256)");

  mpfr_set_prec (x, 2);
  res = mpfr_set_si_2exp (x, 7, 10, MPFR_RNDU);
  if (mpfr_cmp_ui (x, 1<<13) || res <= 0)
    ERROR ("Prec 2 + si_2exp");

  res = mpfr_set_ui_2exp (x, 7, 10, MPFR_RNDU);
  if (mpfr_cmp_ui (x, 1<<13) || res <= 0)
    ERROR ("Prec 2 + ui_2exp");

  mpfr_clear_flags ();
  mpfr_set_ui_2exp (x, 17, MPFR_EMAX_MAX, MPFR_RNDN);
  if (!mpfr_inf_p (x) || MPFR_IS_NEG (x))
    ERROR ("mpfr_set_ui_2exp and overflow (bad result)");
  if (!mpfr_overflow_p ())
    ERROR ("mpfr_set_ui_2exp and overflow (overflow flag not set)");

  mpfr_clear_flags ();
  mpfr_set_si_2exp (x, 17, MPFR_EMAX_MAX, MPFR_RNDN);
  if (!mpfr_inf_p (x) || MPFR_IS_NEG (x))
    ERROR ("mpfr_set_si_2exp (pos) and overflow (bad result)");
  if (!mpfr_overflow_p ())
    ERROR ("mpfr_set_si_2exp (pos) and overflow (overflow flag not set)");

  mpfr_clear_flags ();
  mpfr_set_si_2exp (x, -17, MPFR_EMAX_MAX, MPFR_RNDN);
  if (!mpfr_inf_p (x) || MPFR_IS_POS (x))
    ERROR ("mpfr_set_si_2exp (neg) and overflow (bad result)");
  if (!mpfr_overflow_p ())
    ERROR ("mpfr_set_si_2exp (neg) and overflow (overflow flag not set)");

  mpfr_clear (x);
}

static void
test_macros (void)
{
  mpfr_t x[3];
  mpfr_ptr p;
  int r;

  mpfr_inits (x[0], x[1], x[2], (mpfr_ptr) 0);
  p = x[0];
  r = 0;
  mpfr_set_ui (p++, 0, (mpfr_rnd_t) r++);
  if (p != x[1] || r != 1)
    {
      printf ("Error in mpfr_set_ui macro: p - x[0] = %d (expecting 1), "
              "r = %d (expecting 1)\n", (int) (p - x[0]), r);
      exit (1);
    }
  p = x[0];
  r = 0;
  mpfr_set_si (p++, 0, (mpfr_rnd_t) r++);
  if (p != x[1] || r != 1)
    {
      printf ("Error in mpfr_set_si macro: p - x[0] = %d (expecting 1), "
              "r = %d (expecting 1)\n", (int) (p - x[0]), r);
      exit (1);
    }
  mpfr_clears (x[0], x[1], x[2], (mpfr_ptr) 0);
}

static void
test_macros_keyword (void)
{
  mpfr_t x;
  unsigned long i;

  mpfr_init2 (x, 64);
#define MKN 0x1000000
#define long short
  mpfr_set_ui (x, MKN, MPFR_RNDN);
#undef long
  i = mpfr_get_ui (x, MPFR_RNDN);
  if (i != MKN)
    {
      printf ("Error in test_macros_keyword: expected 0x%lx, got 0x%lx.\n",
              (unsigned long) MKN, i);
      exit (1);
    }
  mpfr_clear (x);
}

/* FIXME: Comparing against mpfr_get_si/ui is not ideal, it'd be better to
   have all tests examine the bits in mpfr_t for what should come out.  */

int
main (int argc, char *argv[])
{
  mpfr_t x;
  long k, z, d, N;
  unsigned long zl, dl;
  int inex;
  int r;
  mpfr_exp_t emin, emax;
  int flag;

  tests_start_mpfr ();

  mpfr_init2 (x, 100);

  N = (argc==1) ? 100000 : atol (argv[1]);

  for (k = 1; k <= N; k++)
    {
      z = (long) (randlimb () & LONG_MAX) + LONG_MIN / 2;
      inex = mpfr_set_si (x, z, MPFR_RNDZ);
      d = mpfr_get_si (x, MPFR_RNDZ);
      if (d != z)
        {
          printf ("Error in mpfr_set_si: expected %ld got %ld\n", z, d);
          exit (1);
        }
      if (inex)
        {
          printf ("Error in mpfr_set_si: inex value incorrect for %ld: %d\n",
                  z, inex);
          exit (1);
        }
    }

  for (k = 1; k <= N; k++)
    {
      zl = randlimb ();
      inex = mpfr_set_ui (x, zl, MPFR_RNDZ);
      dl = mpfr_get_ui (x, MPFR_RNDZ);
      if (dl != zl)
        {
          printf ("Error in mpfr_set_ui: expected %lu got %lu\n", zl, dl);
          exit (1);
        }
      if (inex)
        {
          printf ("Error in mpfr_set_ui: inex value incorrect for %lu: %d\n",
                  zl, inex);
          exit (1);
        }
    }

  mpfr_set_prec (x, 2);
  if (mpfr_set_si (x, 5, MPFR_RNDZ) >= 0)
    {
      printf ("Wrong inexact flag for x=5, rnd=MPFR_RNDZ\n");
      exit (1);
    }

  mpfr_set_prec (x, 2);
  if (mpfr_set_si (x, -5, MPFR_RNDZ) <= 0)
    {
      printf ("Wrong inexact flag for x=-5, rnd=MPFR_RNDZ\n");
      exit (1);
    }

  mpfr_set_prec (x, 3);
  inex = mpfr_set_si (x, 77617, MPFR_RNDD); /* should be 65536 */
  if (MPFR_MANT(x)[0] != ((mp_limb_t)1 << (mp_bits_per_limb-1))
      || inex >= 0)
    {
      printf ("Error in mpfr_set_si(x:3, 77617, MPFR_RNDD)\n");
      mpfr_print_binary (x);
      puts ("");
      exit (1);
    }
  inex = mpfr_set_ui (x, 77617, MPFR_RNDD); /* should be 65536 */
  if (MPFR_MANT(x)[0] != ((mp_limb_t)1 << (mp_bits_per_limb-1))
      || inex >= 0)
    {
      printf ("Error in mpfr_set_ui(x:3, 77617, MPFR_RNDD)\n");
      mpfr_print_binary (x);
      puts ("");
      exit (1);
    }

  mpfr_set_prec (x, 2);
  inex = mpfr_set_si (x, 33096, MPFR_RNDU);
  if (mpfr_get_si (x, MPFR_RNDZ) != 49152 || inex <= 0)
    {
      printf ("Error in mpfr_set_si, exp. 49152, got %ld, inex %d\n",
              mpfr_get_si (x, MPFR_RNDZ), inex);
      exit (1);
    }
  inex = mpfr_set_ui (x, 33096, MPFR_RNDU);
  if (mpfr_get_si (x, MPFR_RNDZ) != 49152)
    {
      printf ("Error in mpfr_set_ui, exp. 49152, got %ld, inex %d\n",
              mpfr_get_si (x, MPFR_RNDZ), inex);
      exit (1);
    }
  /* Also test the mpfr_set_ui function (instead of macro). */
  inex = (mpfr_set_ui) (x, 33096, MPFR_RNDU);
  if (mpfr_get_si (x, MPFR_RNDZ) != 49152)
    {
      printf ("Error in mpfr_set_ui function, exp. 49152, got %ld, inex %d\n",
              mpfr_get_si (x, MPFR_RNDZ), inex);
      exit (1);
    }

  for (r = 0 ; r < MPFR_RND_MAX ; r++)
    {
      mpfr_set_si (x, -1, (mpfr_rnd_t) r);
      mpfr_set_ui (x, 0, (mpfr_rnd_t) r);
      if (MPFR_IS_NEG (x) || mpfr_get_ui (x, (mpfr_rnd_t) r) != 0)
        {
          printf ("mpfr_set_ui (x, 0) gives -0 for %s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit (1);
        }

      mpfr_set_si (x, -1, (mpfr_rnd_t) r);
      mpfr_set_si (x, 0, (mpfr_rnd_t) r);
      if (MPFR_IS_NEG (x) || mpfr_get_si (x, (mpfr_rnd_t) r) != 0)
        {
          printf ("mpfr_set_si (x, 0) gives -0 for %s\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) r));
          exit (1);
        }
    }

  /* check potential bug in case mp_limb_t is unsigned */
  emax = mpfr_get_emax ();
  set_emax (0);
  mpfr_set_si (x, -1, MPFR_RNDN);
  if (mpfr_sgn (x) >= 0)
    {
      printf ("mpfr_set_si (x, -1) fails\n");
      exit (1);
    }
  set_emax (emax);

  emax = mpfr_get_emax ();
  set_emax (5);
  mpfr_set_prec (x, 2);
  mpfr_set_si (x, -31, MPFR_RNDN);
  if (mpfr_sgn (x) >= 0)
    {
      printf ("mpfr_set_si (x, -31) fails\n");
      exit (1);
    }
  set_emax (emax);

  /* test for get_ui */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_get_ui (x, MPFR_RNDN) == 0);
  mpfr_set_ui (x, ULONG_MAX, MPFR_RNDU);
  mpfr_nextabove (x);
  mpfr_get_ui (x, MPFR_RNDU);

  /* another test for get_ui */
  mpfr_set_prec (x, 10);
  mpfr_set_str_binary (x, "10.101");
  dl = mpfr_get_ui (x, MPFR_RNDN);
  MPFR_ASSERTN (dl == 3);

  mpfr_set_str_binary (x, "-1.0");
  mpfr_get_ui (x, MPFR_RNDN);

  mpfr_set_str_binary (x, "0.1");
  dl = mpfr_get_ui (x, MPFR_RNDN);
  MPFR_ASSERTN (dl == 0);
  dl = mpfr_get_ui (x, MPFR_RNDZ);
  MPFR_ASSERTN (dl == 0);
  dl = mpfr_get_ui (x, MPFR_RNDD);
  MPFR_ASSERTN (dl == 0);
  dl = mpfr_get_ui (x, MPFR_RNDU);
  MPFR_ASSERTN (dl == 1);

  /* coverage tests */
  mpfr_set_prec (x, 2);
  mpfr_set_si (x, -7, MPFR_RNDD);
  MPFR_ASSERTN(mpfr_cmp_si (x, -8) == 0);
  mpfr_set_prec (x, 2);
  mpfr_set_ui (x, 7, MPFR_RNDU);
  MPFR_ASSERTN(mpfr_cmp_ui (x, 8) == 0);
  emax = mpfr_get_emax ();
  set_emax (3);
  mpfr_set_ui (x, 7, MPFR_RNDU);
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);
  set_emax (1);
  MPFR_ASSERTN( mpfr_set_ui (x, 7, MPFR_RNDU) );
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);
  set_emax (emax);
  mpfr_set_ui_2exp (x, 17, -50, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_get_ui (x, MPFR_RNDD) == 0);
  MPFR_ASSERTN (mpfr_get_si (x, MPFR_RNDD) == 0);

  /* Test for ERANGE flag + correct behaviour if overflow */
  mpfr_set_prec (x, 256);
  mpfr_set_ui (x, ULONG_MAX, MPFR_RNDN);
  mpfr_clear_erangeflag ();
  dl = mpfr_get_ui (x, MPFR_RNDN);
  if (dl != ULONG_MAX || mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_ui + ERANGE + ULONG_MAX (1)\n");
      exit (1);
    }
  mpfr_add_ui (x, x, 1, MPFR_RNDN);
  dl = mpfr_get_ui (x, MPFR_RNDN);
  if (dl != ULONG_MAX || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_ui + ERANGE + ULONG_MAX (2)\n");
      exit (1);
    }
  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_clear_erangeflag ();
  dl = mpfr_get_ui (x, MPFR_RNDN);
  if (dl != 0 || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_ui + ERANGE + -1 \n");
      exit (1);
    }
  mpfr_set_si (x, LONG_MAX, MPFR_RNDN);
  mpfr_clear_erangeflag ();
  d = mpfr_get_si (x, MPFR_RNDN);
  if (d != LONG_MAX || mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_si + ERANGE + LONG_MAX (1): %ld\n", d);
      exit (1);
    }
  mpfr_add_ui (x, x, 1, MPFR_RNDN);
  d = mpfr_get_si (x, MPFR_RNDN);
  if (d != LONG_MAX || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_si + ERANGE + LONG_MAX (2)\n");
      exit (1);
    }
  mpfr_set_si (x, LONG_MIN, MPFR_RNDN);
  mpfr_clear_erangeflag ();
  d = mpfr_get_si (x, MPFR_RNDN);
  if (d != LONG_MIN || mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_si + ERANGE + LONG_MIN (1)\n");
      exit (1);
    }
  mpfr_sub_ui (x, x, 1, MPFR_RNDN);
  d = mpfr_get_si (x, MPFR_RNDN);
  if (d != LONG_MIN || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_si + ERANGE + LONG_MIN (2)\n");
      exit (1);
    }

  mpfr_set_nan (x);
  mpfr_clear_erangeflag ();
  d = mpfr_get_ui (x, MPFR_RNDN);
  if (d != 0 || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_ui + NaN\n");
      exit (1);
    }
  mpfr_clear_erangeflag ();
  d = mpfr_get_si (x, MPFR_RNDN);
  if (d != 0 || !mpfr_erangeflag_p ())
    {
      printf ("ERROR for get_si + NaN\n");
      exit (1);
    }

  emin = mpfr_get_emin ();
  mpfr_set_prec (x, 2);

  mpfr_set_emin (4);
  mpfr_clear_flags ();
  mpfr_set_ui (x, 7, MPFR_RNDU);
  flag = mpfr_underflow_p ();
  mpfr_set_emin (emin);
  if (mpfr_cmp_ui (x, 8) != 0)
    {
      printf ("Error for mpfr_set_ui (x, 7, MPFR_RNDU), prec = 2, emin = 4\n");
      exit (1);
    }
  if (flag)
    {
      printf ("mpfr_set_ui (x, 7, MPFR_RNDU) should not underflow "
              "with prec = 2, emin = 4\n");
      exit (1);
    }

  mpfr_set_emin (4);
  mpfr_clear_flags ();
  mpfr_set_si (x, -7, MPFR_RNDD);
  flag = mpfr_underflow_p ();
  mpfr_set_emin (emin);
  if (mpfr_cmp_si (x, -8) != 0)
    {
      printf ("Error for mpfr_set_si (x, -7, MPFR_RNDD), prec = 2, emin = 4\n");
      exit (1);
    }
  if (flag)
    {
      printf ("mpfr_set_si (x, -7, MPFR_RNDD) should not underflow "
              "with prec = 2, emin = 4\n");
      exit (1);
    }

  mpfr_clear (x);

  test_2exp ();
  test_macros ();
  test_macros_keyword ();
  tests_end_mpfr ();
  return 0;
}
