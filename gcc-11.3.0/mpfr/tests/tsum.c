/* tsum -- test file for the list summation function

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

#include <stdlib.h>
#include <stdio.h>
#include "mpfr-test.h"

static int
check_is_sorted (unsigned long n, mpfr_srcptr *perm)
{
  unsigned long i;

  for (i = 0; i < n - 1; i++)
    if (MPFR_GET_EXP(perm[i]) < MPFR_GET_EXP(perm[i+1]))
      return 0;
  return 1;
}

static int
sum_tab (mpfr_ptr ret, mpfr_t *tab, unsigned long n, mpfr_rnd_t rnd)
{
  mpfr_ptr *tabtmp;
  unsigned long i;
  int inexact;
  MPFR_TMP_DECL(marker);

  MPFR_TMP_MARK(marker);
  tabtmp = (mpfr_ptr *) MPFR_TMP_ALLOC(n * sizeof(mpfr_srcptr));
  for (i = 0; i < n; i++)
    tabtmp[i] = tab[i];

  inexact = mpfr_sum (ret, tabtmp, n, rnd);
  MPFR_TMP_FREE(marker);
  return inexact;
}


static mpfr_prec_t
get_prec_max (mpfr_t *tab, unsigned long n, mpfr_prec_t f)
{
  mpfr_prec_t res;
  mpfr_exp_t min, max;
  unsigned long i;

  i = 0;
  while (MPFR_IS_ZERO (tab[i]))
    {
      i++;
      if (i == n)
        return MPFR_PREC_MIN;  /* all values are 0 */
    }

  if (! mpfr_check (tab[i]))
    {
      printf ("tab[%lu] is not valid.\n", i);
      exit (1);
    }
  MPFR_ASSERTN (MPFR_IS_FP (tab[i]));
  min = max = MPFR_GET_EXP(tab[i]);
  for (i++; i < n; i++)
    {
      if (! mpfr_check (tab[i]))
        {
          printf ("tab[%lu] is not valid.\n", i);
          exit (1);
        }
      MPFR_ASSERTN (MPFR_IS_FP (tab[i]));
      if (! MPFR_IS_ZERO (tab[i]))
        {
          if (MPFR_GET_EXP(tab[i]) > max)
            max = MPFR_GET_EXP(tab[i]);
          if (MPFR_GET_EXP(tab[i]) < min)
            min = MPFR_GET_EXP(tab[i]);
        }
    }
  res = max - min;
  res += f;
  res += __gmpfr_ceil_log2 (n) + 1;
  return res;
}


static void
algo_exact (mpfr_t somme, mpfr_t *tab, unsigned long n, mpfr_prec_t f)
{
  unsigned long i;
  mpfr_prec_t prec_max;

  prec_max = get_prec_max(tab, n, f);
  mpfr_set_prec (somme, prec_max);
  mpfr_set_ui (somme, 0, MPFR_RNDN);
  for (i = 0; i < n; i++)
    {
      if (mpfr_add(somme, somme, tab[i], MPFR_RNDN))
        {
          printf ("FIXME: algo_exact is buggy.\n");
          exit (1);
        }
    }
}

/* Test the sorting function */
static void
test_sort (mpfr_prec_t f, unsigned long n)
{
  mpfr_t *tab;
  mpfr_ptr *tabtmp;
  mpfr_srcptr *perm;
  unsigned long i;
  mpfr_prec_t prec = MPFR_PREC_MIN;

  /* Init stuff */
  tab = (mpfr_t *) tests_allocate (n * sizeof (mpfr_t));
  for (i = 0; i < n; i++)
    mpfr_init2 (tab[i], f);
  tabtmp = (mpfr_ptr *) tests_allocate (n * sizeof(mpfr_ptr));
  perm = (mpfr_srcptr *) tests_allocate (n * sizeof(mpfr_srcptr));

  for (i = 0; i < n; i++)
    {
      mpfr_urandomb (tab[i], RANDS);
      tabtmp[i] = tab[i];
    }

  mpfr_sum_sort ((mpfr_srcptr *)tabtmp, n, perm, &prec);

  if (check_is_sorted (n, perm) == 0)
    {
      printf ("mpfr_sum_sort incorrect.\n");
      for (i = 0; i < n; i++)
        mpfr_dump (perm[i]);
      exit (1);
    }

  /* Clear stuff */
  for (i = 0; i < n; i++)
    mpfr_clear (tab[i]);
  tests_free (tab, n * sizeof (mpfr_t));
  tests_free (tabtmp, n * sizeof(mpfr_ptr));
  tests_free (perm, n * sizeof(mpfr_srcptr));
}

static void
test_sum (mpfr_prec_t f, unsigned long n)
{
  mpfr_t sum, real_sum, real_non_rounded;
  mpfr_t *tab;
  unsigned long i;
  int rnd_mode;

  /* Init */
  tab = (mpfr_t *) tests_allocate (n * sizeof(mpfr_t));
  for (i = 0; i < n; i++)
    mpfr_init2 (tab[i], f);
  mpfr_inits2 (f, sum, real_sum, real_non_rounded, (mpfr_ptr) 0);

  /* First Uniform */
  for (i = 0; i < n; i++)
    mpfr_urandomb (tab[i], RANDS);
  algo_exact (real_non_rounded, tab, n, f);
  for (rnd_mode = 0; rnd_mode < MPFR_RND_MAX; rnd_mode++)
    {
      sum_tab (sum, tab, n, (mpfr_rnd_t) rnd_mode);
      mpfr_set (real_sum, real_non_rounded, (mpfr_rnd_t) rnd_mode);
      if (mpfr_cmp (real_sum, sum) != 0)
        {
          printf ("mpfr_sum incorrect.\n");
          mpfr_dump (real_sum);
          mpfr_dump (sum);
          exit (1);
        }
    }

  /* Then non uniform */
  for (i = 0; i < n; i++)
    {
      mpfr_urandomb (tab[i], RANDS);
      if (! mpfr_zero_p (tab[i]))
        mpfr_set_exp (tab[i], randlimb () % 1000);
    }
  algo_exact (real_non_rounded, tab, n, f);
  for (rnd_mode = 0; rnd_mode < MPFR_RND_MAX; rnd_mode++)
    {
      sum_tab (sum, tab, n, (mpfr_rnd_t) rnd_mode);
      mpfr_set (real_sum, real_non_rounded, (mpfr_rnd_t) rnd_mode);
      if (mpfr_cmp (real_sum, sum) != 0)
        {
          printf ("mpfr_sum incorrect.\n");
          mpfr_dump (real_sum);
          mpfr_dump (sum);
          exit (1);
        }
    }

  /* Clear stuff */
  for (i = 0; i < n; i++)
    mpfr_clear (tab[i]);
  mpfr_clears (sum, real_sum, real_non_rounded, (mpfr_ptr) 0);
  tests_free (tab, n * sizeof(mpfr_t));
}

static
void check_special (void)
{
  mpfr_t tab[3], r;
  mpfr_ptr tabp[3];
  int i;

  mpfr_inits (tab[0], tab[1], tab[2], r, (mpfr_ptr) 0);
  tabp[0] = tab[0];
  tabp[1] = tab[1];
  tabp[2] = tab[2];

  i = mpfr_sum (r, tabp, 0, MPFR_RNDN);
  if (!MPFR_IS_ZERO (r) || !MPFR_IS_POS (r) || i != 0)
    {
      printf ("Special case n==0 failed!\n");
      exit (1);
    }

  mpfr_set_ui (tab[0], 42, MPFR_RNDN);
  i = mpfr_sum (r, tabp, 1, MPFR_RNDN);
  if (mpfr_cmp_ui (r, 42) || i != 0)
    {
      printf ("Special case n==1 failed!\n");
      exit (1);
    }

  mpfr_set_ui (tab[1], 17, MPFR_RNDN);
  MPFR_SET_NAN (tab[2]);
  i = mpfr_sum (r, tabp, 3, MPFR_RNDN);
  if (!MPFR_IS_NAN (r) || i != 0)
    {
      printf ("Special case NAN failed!\n");
      exit (1);
    }

  MPFR_SET_INF (tab[2]);
  MPFR_SET_POS (tab[2]);
  i = mpfr_sum (r, tabp, 3, MPFR_RNDN);
  if (!MPFR_IS_INF (r) || !MPFR_IS_POS (r) || i != 0)
    {
      printf ("Special case +INF failed!\n");
      exit (1);
    }

  MPFR_SET_INF (tab[2]);
  MPFR_SET_NEG (tab[2]);
  i = mpfr_sum (r, tabp, 3, MPFR_RNDN);
  if (!MPFR_IS_INF (r) || !MPFR_IS_NEG (r) || i != 0)
    {
      printf ("Special case -INF failed!\n");
      exit (1);
    }

  MPFR_SET_ZERO (tab[1]);
  i = mpfr_sum (r, tabp, 2, MPFR_RNDN);
  if (mpfr_cmp_ui (r, 42) || i != 0)
    {
      printf ("Special case 42+0 failed!\n");
      exit (1);
    }

  MPFR_SET_NAN (tab[0]);
  i = mpfr_sum (r, tabp, 3, MPFR_RNDN);
  if (!MPFR_IS_NAN (r) || i != 0)
    {
      printf ("Special case NAN+0+-INF failed!\n");
      exit (1);
    }

  mpfr_set_inf (tab[0], 1);
  mpfr_set_ui  (tab[1], 59, MPFR_RNDN);
  mpfr_set_inf (tab[2], -1);
  i = mpfr_sum (r, tabp, 3, MPFR_RNDN);
  if (!MPFR_IS_NAN (r) || i != 0)
    {
      printf ("Special case +INF + 59 +-INF failed!\n");
      exit (1);
    }

  mpfr_clears (tab[0], tab[1], tab[2], r, (mpfr_ptr) 0);
}

/* bug reported by Joseph S. Myers on 2013-10-27
   https://sympa.inria.fr/sympa/arc/mpfr/2013-10/msg00015.html */
static void
bug20131027 (void)
{
  mpfr_t r, t[4];
  mpfr_ptr p[4];
  char *s[4] = {
    "0x1p1000",
    "-0x0.fffffffffffff80000000000000001p1000",
    "-0x1p947",
    "0x1p880"
  };
  int i;

  mpfr_init2 (r, 53);
  for (i = 0; i < 4; i++)
    {
      mpfr_init2 (t[i], i == 0 ? 53 : 1000);
      mpfr_set_str (t[i], s[i], 0, MPFR_RNDN);
      p[i] = t[i];
    }
  mpfr_sum (r, p, 4, MPFR_RNDN);

  if (MPFR_NOTZERO (r))
    {
      printf ("mpfr_sum incorrect in bug20131027: expected 0, got\n");
      mpfr_dump (r);
      exit (1);
    }

  for (i = 0; i < 4; i++)
    mpfr_clear (t[i]);
  mpfr_clear (r);
}

int
main (void)
{
  mpfr_prec_t p;
  unsigned long n;

  tests_start_mpfr ();

  check_special ();
  bug20131027 ();
  test_sort (1764, 1026);
  for (p = 2 ; p < 444 ; p += 17)
    for (n = 2 ; n < 1026 ; n += 42 + p)
      test_sum (p, n);

  tests_end_mpfr ();
  return 0;
}
