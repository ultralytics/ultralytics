/* Test file for mpfr_urandom

Copyright 1999-2004, 2006-2017 Free Software Foundation, Inc.
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
test_urandom (long nbtests, mpfr_prec_t prec, mpfr_rnd_t rnd, long bit_index,
              int verbose)
{
  mpfr_t x;
  int *tab, size_tab, k, sh, xn;
  double d, av = 0, var = 0, chi2 = 0, th;
  mpfr_exp_t emin;
  mp_size_t limb_index = 0;
  mp_limb_t limb_mask = 0;
  long count = 0;
  int i;
  int inex = 1;
  unsigned int ex_flags, flags;

  size_tab = (nbtests >= 1000 ? nbtests / 50 : 20);
  tab = (int *) calloc (size_tab, sizeof(int));
  if (tab == NULL)
    {
      fprintf (stderr, "trandom: can't allocate memory in test_urandom\n");
      exit (1);
    }

  mpfr_init2 (x, prec);
  xn = 1 + (prec - 1) / mp_bits_per_limb;
  sh = xn * mp_bits_per_limb - prec;
  if (bit_index >= 0 && bit_index < prec)
    {
      /* compute the limb index and limb mask to fetch the bit #bit_index */
      limb_index = (prec - bit_index) / mp_bits_per_limb;
      i = 1 + bit_index - (bit_index / mp_bits_per_limb) * mp_bits_per_limb;
      limb_mask = MPFR_LIMB_ONE << (mp_bits_per_limb - i);
    }

  for (k = 0; k < nbtests; k++)
    {
      mpfr_clear_flags ();
      ex_flags = MPFR_FLAGS_INEXACT;
      i = mpfr_urandom (x, RANDS, rnd);
      flags = __gmpfr_flags;
      inex = (i != 0) && inex;
      /* check that lower bits are zero */
      if (MPFR_MANT(x)[0] & MPFR_LIMB_MASK(sh) && !MPFR_IS_ZERO (x))
        {
          printf ("Error: mpfr_urandom() returns invalid numbers:\n");
          mpfr_print_binary (x); puts ("");
          exit (1);
        }
      /* check that the value is in [0,1] */
      if (mpfr_cmp_ui (x, 0) < 0 || mpfr_cmp_ui (x, 1) > 0)
        {
          printf ("Error: mpfr_urandom() returns number outside [0, 1]:\n");
          mpfr_print_binary (x); puts ("");
          exit (1);
        }
      /* check the flags (an underflow is theoretically possible, but
         impossible in practice due to the huge exponent range) */
      if (flags != ex_flags)
        {
          printf ("Error: mpfr_urandom() returns incorrect flags.\n");
          printf ("Expected ");
          flags_out (ex_flags);
          printf ("Got      ");
          flags_out (flags);
          exit (1);
        }

      d = mpfr_get_d1 (x); av += d; var += d*d;
      i = (int)(size_tab * d);
      if (d == 1.0) i --;
      tab[i]++;

      if (limb_mask && (MPFR_MANT (x)[limb_index] & limb_mask))
        count ++;
    }

  if (inex == 0)
    {
      /* one call in the loop pretended to return an exact number! */
      printf ("Error: mpfr_urandom() returns a zero ternary value.\n");
      exit (1);
    }

  /* coverage test */
  emin = mpfr_get_emin ();
  for (k = 0; k < 5; k++)
    {
      set_emin (k+1);
      mpfr_clear_flags ();
      ex_flags = MPFR_FLAGS_UNDERFLOW | MPFR_FLAGS_INEXACT;
      inex = mpfr_urandom (x, RANDS, rnd);
      flags = __gmpfr_flags;
      if (flags != ex_flags)
        {
          printf ("Error: mpfr_urandom() returns incorrect flags"
                  " for emin = %d.\n", k+1);
          printf ("Expected ");
          flags_out (ex_flags);
          printf ("Got      ");
          flags_out (flags);
          exit (1);
        }
      if ((   (rnd == MPFR_RNDZ || rnd == MPFR_RNDD)
              && (!MPFR_IS_ZERO (x) || inex != -1))
          || ((rnd == MPFR_RNDU || rnd == MPFR_RNDA)
              && (mpfr_cmp_ui (x, 1 << k) != 0 || inex != +1))
          || (rnd == MPFR_RNDN
              && (k > 0 || mpfr_cmp_ui (x, 1 << k) != 0 || inex != +1)
              && (!MPFR_IS_ZERO (x) || inex != -1)))
        {
          printf ("Error: mpfr_urandom() do not handle correctly a restricted"
                  " exponent range.\nrounding mode: %s\nternary value: %d\n"
                  "random value: ", mpfr_print_rnd_mode (rnd), inex);
          mpfr_print_binary (x); puts ("");
          exit (1);
        }
    }
  set_emin (emin);

  mpfr_clear (x);
  if (!verbose)
    {
      free(tab);
      return;
    }

  av /= nbtests;
  var = (var / nbtests) - av * av;

  th = (double)nbtests / size_tab;
  printf ("Average = %.5f\nVariance = %.5f\n", av, var);
  printf ("Repartition for urandom with rounding mode %s. "
          "Each integer should be close to %d.\n",
          mpfr_print_rnd_mode (rnd), (int) th);

  for (k = 0; k < size_tab; k++)
    {
      chi2 += (tab[k] - th) * (tab[k] - th) / th;
      printf("%d ", tab[k]);
      if (((k+1) & 7) == 0)
        printf("\n");
    }

  printf("\nChi2 statistics value (with %d degrees of freedom) : %.5f\n",
         size_tab - 1, chi2);

  if (limb_mask)
    printf ("Bit #%ld is set %ld/%ld = %.1f %% of time\n",
            bit_index, count, nbtests, count * 100.0 / nbtests);

  puts ("");

  free(tab);
  return;
}

/* Problem reported by Carl Witty. This test assumes the random generator
   used by GMP is deterministic (for a given seed). We need to distinguish
   two cases since the random generator changed in GMP 4.2.0. */
static void
bug20100914 (void)
{
  mpfr_t x;
  gmp_randstate_t s;

#if __MPFR_GMP(4,2,0)
# define C1 "0.8488312"
# define C2 "0.8156509"
#else
# define C1 "0.6485367"
# define C2 "0.9362717"
#endif

  gmp_randinit_default (s);
  gmp_randseed_ui (s, 42);
  mpfr_init2 (x, 17);
  mpfr_urandom (x, s, MPFR_RNDN);
  if (mpfr_cmp_str1 (x, C1) != 0)
    {
      printf ("Error in bug20100914, expected " C1 ", got ");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  mpfr_urandom (x, s, MPFR_RNDN);
  if (mpfr_cmp_str1 (x, C2) != 0)
    {
      printf ("Error in bug20100914, expected " C2 ", got ");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }
  mpfr_clear (x);
  gmp_randclear (s);
}

/* non-regression test for bug reported by Trevor Spiteri
   https://sympa.inria.fr/sympa/arc/mpfr/2017-01/msg00020.html */
static void
bug20170123 (void)
{
#if __MPFR_GMP(4,2,0)
  mpfr_t x;
  mpfr_exp_t emin;
  gmp_randstate_t s;

  emin = mpfr_get_emin ();
  mpfr_set_emin (-7);
  mpfr_init2 (x, 53);
  gmp_randinit_default (s);
  gmp_randseed_ui (s, 398);
  mpfr_urandom (x, s, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui_2exp (x, 1, -8) == 0);
  mpfr_clear (x);
  gmp_randclear (s);
  mpfr_set_emin (emin);
#endif
}

static void
underflow_tests (void)
{
  mpfr_t x;
  mpfr_exp_t emin;
  int i, k;
  int inex;
  int rnd;
  unsigned int ex_flags, flags;

  emin = mpfr_get_emin ();
  mpfr_init2 (x, 4);
  ex_flags = MPFR_FLAGS_UNDERFLOW | MPFR_FLAGS_INEXACT; /* if underflow */
  for (i = 2; i >= -4; i--)
    {
      mpfr_set_emin (i);
      RND_LOOP (rnd)
        for (k = 0; k < 100; k++)
          {
            mpfr_clear_flags ();
            inex = mpfr_urandom (x, RANDS, (mpfr_rnd_t) rnd);
            flags = __gmpfr_flags;
            MPFR_ASSERTN (mpfr_inexflag_p ());
            if (MPFR_IS_NEG (x))
              {
                printf ("Error in underflow_tests: got a negative sign"
                        " for i=%d rnd=%s k=%d.\n",
                        i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), k);
                exit (1);
              }
            if (MPFR_IS_ZERO (x))
              {
                if (rnd == MPFR_RNDU || rnd == MPFR_RNDA)
                  {
                    printf ("Error in underflow_tests: the value cannot"
                            " be 0 for i=%d rnd=%s k=%d.\n",
                            i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), k);
                    exit (1);
                  }
                if (flags != ex_flags)
                  {
                    printf ("Error in underflow_tests: incorrect flags"
                            " for i=%d rnd=%s k=%d.\n",
                            i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), k);
                    printf ("Expected ");
                    flags_out (ex_flags);
                    printf ("Got      ");
                    flags_out (flags);
                    exit (1);
                  }
              }
            if (inex == 0 || (MPFR_IS_ZERO (x) && inex > 0))
              {
                printf ("Error in underflow_tests: incorrect inex (%d)"
                        " for i=%d rnd=%s k=%d.\n", inex,
                        i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), k);
                exit (1);
              }
          }
    }
  mpfr_clear (x);
  mpfr_set_emin (emin);
}

static void
test_underflow (int verbose)
{
  mpfr_t x;
  mpfr_exp_t emin = mpfr_get_emin ();
  long i, exp[6] = {0, 0, 0, 0, 0, 0};

  mpfr_init2 (x, 2);
  mpfr_set_emin (-3);
#define N 1000000
  for (i = 0; i < N; i++)
    {
      mpfr_urandom (x, RANDS, MPFR_RNDN);
      if (mpfr_zero_p (x))
        exp[5] ++;
      else
        /* exp=1 is possible if the generated number is 0.111111... */
        exp[1-mpfr_get_exp(x)] ++;
    }
  if (verbose)
    printf ("exp=1:%.3f(%.3f) 0:%.3f(%.3f) -1:%.3f(%.3f) -2:%.3f(%.3f) "
            "-3:%.3f(%.3f) zero:%.3f(%.3f)\n",
            100.0 * (double) exp[0] / (double) N, 12.5,
            100.0 * (double) exp[1] / (double) N, 43.75,
            100.0 * (double) exp[2] / (double) N, 21.875,
            100.0 * (double) exp[3] / (double) N, 10.9375,
            100.0 * (double) exp[4] / (double) N, 7.8125,
            100.0 * (double) exp[5] / (double) N, 3.125);
  mpfr_clear (x);
  mpfr_set_emin (emin);
#undef N
}

static void
overflow_tests (void)
{
  mpfr_t x;
  mpfr_exp_t emax;
  int i, k;
  int inex;
  int rnd;
  unsigned int ex_flags, flags;

  emax = mpfr_get_emax ();
  mpfr_init2 (x, 4);
  ex_flags = MPFR_FLAGS_OVERFLOW | MPFR_FLAGS_INEXACT; /* if overflow */
  for (i = -4; i <= 0; i++)
    {
      mpfr_set_emax (i);
      RND_LOOP (rnd)
        for (k = 0; k < 100; k++)
          {
            mpfr_clear_flags ();
            inex = mpfr_urandom (x, RANDS, (mpfr_rnd_t) rnd);
            flags = __gmpfr_flags;
            MPFR_ASSERTN (mpfr_inexflag_p ());
            if (MPFR_IS_NEG (x))
              {
                printf ("Error in overflow_tests: got a negative sign"
                        " for i=%d rnd=%s k=%d.\n",
                        i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), k);
                exit (1);
              }
            if (MPFR_IS_INF (x))
              {
                if (rnd == MPFR_RNDD || rnd == MPFR_RNDZ)
                  {
                    printf ("Error in overflow_tests: the value cannot"
                            " be +inf for i=%d rnd=%s k=%d.\n",
                            i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), k);
                    exit (1);
                  }
                if (flags != ex_flags)
                  {
                    printf ("Error in overflow_tests: incorrect flags"
                            " for i=%d rnd=%s k=%d.\n",
                            i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), k);
                    printf ("Expected ");
                    flags_out (ex_flags);
                    printf ("Got      ");
                    flags_out (flags);
                    exit (1);
                  }
              }
            if (inex == 0 || (MPFR_IS_INF (x) && inex < 0))
              {
                printf ("Error in overflow_tests: incorrect inex (%d)"
                        " for i=%d rnd=%s k=%d.\n", inex,
                        i, mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), k);
                exit (1);
              }
          }
    }
  mpfr_clear (x);
  mpfr_set_emax (emax);
}

/* Reproducibility test: check that the behavior does not depend on
   the platform ABI or MPFR version (new, incompatible MPFR versions
   may introduce changes, in which case the hardcoded values should
   depend on MPFR_VERSION).
   It is not necessary to test with different rounding modes and
   exponent ranges as this has already been done in reprod_rnd_exp.
   We do not need to check the status of the PRNG after mpfr_urandom
   since this is done implicitly by comparing the next value, except
   for the last itaration.
*/
static void
reprod_abi (void)
{
#if __MPFR_GMP(4,2,0)
#define N 6
  /* Run this program with the MPFR_REPROD_ABI_OUTPUT environment variable
     set to get the array of strings. */
  char *t[5 * N] = {
    "1.8@-1",
    "5.0@-1",
    "5.0@-1",
    "a.0@-1",
    "f.c0@-1",
    "a.51738280@-1",
    "e.53dc74e0@-1",
    "6.4edc72d0@-1",
    "3.de4fdd30@-1",
    "5.e1fe0b5a0@-1",
    "5.9ce70076d54d1980@-1",
    "1.f84df9a49d68e0ec@-1",
    "b.18aa08297dbe2cf0@-1",
    "6.def0fb5a8190c754@-1",
    "d.17a57ed62a602ad40@-1",
    "1.812f1fa6c4e024363ae95d58@-1",
    "2.09b0e06eb0a7e16ce51c8b90@-1",
    "5.86a1ac9115417c51af272238@-1",
    "c.9e3b21f7dfb431dd6533c008@-1",
    "6.6b26e12c345f96fd2929d4200@-1",
    "7.d039f4414022c863224e6c641dd71900@-2",
    "8.18e0674a0a2b318d3c99911a45e4cf40@-1",
    "d.426ed83227d849f06424b5e86022c620@-1",
    "7.1638e5483b69f800691942a63307fc98@-1",
    "2.6ed009ee96c8788f6b88212f5f2a4d4b0@-1",
    "3.8429cc14328d744c0b3cda780f0a962fcb397400@-1",
    "4.257841b744575cc65865e54f4cf43f4f1527ebd0@-1",
    "1.b593a359e6e146dfd7db2e5b768fe4a07efc114e@-1",
    "2.543b005741d935e8e3081bbae9bffa33ce75ddc6@-1",
    "a.2f9030421a312f0bb16db20c4783c6438725ed600@-1"
  };
  gmp_randstate_t s;
  int generate, i;

  /* We must hardcode the seed to be able to compare with hardcoded values. */
  gmp_randinit_default (s);
  gmp_randseed_ui (s, 17);

  generate = getenv ("MPFR_REPROD_ABI_OUTPUT") != NULL;

  for (i = 0; i < 5 * N; i++)
    {
      mpfr_prec_t prec;
      mpfr_t x;

      prec = i < 5 ? MPFR_PREC_MIN + i : (i / 5) * 32 + (i % 5) - 2;
      mpfr_init2 (x, prec);
      mpfr_urandom (x, s, MPFR_RNDN);
      if (generate)
        {
          printf ("    \"");
          mpfr_out_str (stdout, 16, 0, x, MPFR_RNDZ);
          printf (i < 5 * N - 1 ? "\",\n" : "\"\n");
        }
      else if (mpfr_cmp_str (x, t[i], 16, MPFR_RNDN) != 0)
        {
          printf ("Error in reprod_abi for i=%d\n", i);
          printf ("Expected %s\n", t[i]);
          printf ("Got      ");
          mpfr_out_str (stdout, 16, 0, x, MPFR_RNDZ);
          printf ("\n");
          exit (1);
        }
      mpfr_clear (x);
    }

  gmp_randclear (s);
#endif
}

int
main (int argc, char *argv[])
{
  long nbtests;
  mpfr_prec_t prec;
  int verbose = 0;
  int rnd;
  long bit_index;

  tests_start_mpfr ();

  if (argc > 1)
    verbose = 1;

  nbtests = 10000;
  if (argc > 1)
    {
      long a = atol(argv[1]);
      if (a != 0)
        nbtests = a;
    }

  if (argc <= 2)
    prec = 1000;
  else
    prec = atol(argv[2]);

  if (argc <= 3)
    bit_index = -1;
  else
    {
      bit_index = atol(argv[3]);
      if (bit_index >= prec)
        {
          printf ("Warning. Cannot compute the bit frequency: the given bit "
                  "index (= %ld) is not less than the precision (= %ld).\n",
                  bit_index, (long) prec);
          bit_index = -1;
        }
    }

  RND_LOOP(rnd)
    {
      test_urandom (nbtests, prec, (mpfr_rnd_t) rnd, bit_index, verbose);

      if (argc == 1)  /* check also small precision */
        {
          test_urandom (nbtests, 2, (mpfr_rnd_t) rnd, -1, 0);
        }
    }

  underflow_tests ();
  overflow_tests ();

  bug20100914 ();
  bug20170123 ();
  reprod_abi ();
  test_underflow (verbose);

  tests_end_mpfr ();
  return 0;
}
