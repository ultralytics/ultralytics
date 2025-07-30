/* Test file for mpfr_factorial.

Copyright 2001-2017 Free Software Foundation, Inc.
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

#define TEST_FUNCTION mpfr_fac_ui

static void
special (void)
{
  mpfr_t x, y;
  int inex;

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_prec (x, 21);
  mpfr_set_prec (y, 21);
  mpfr_fac_ui (x, 119, MPFR_RNDZ);
  mpfr_set_str_binary (y, "0.101111101110100110110E654");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_fac_ui (119)\n");
      exit (1);
    }

  mpfr_set_prec (y, 206);
  inex = mpfr_fac_ui (y, 767, MPFR_RNDN);
  mpfr_set_prec (x, 206);
  mpfr_set_str_binary (x, "0.110111100001000001101010010001000111000100000100111000010011100011011111001100011110101000111101101100110001001100110100001001111110000101010000100100011100010011101110000001000010001100010000101001111E6250");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_fac_ui (767)\n");
      exit (1);
    }
  if (inex <= 0)
    {
      printf ("Wrong flag for mpfr_fac_ui (767)\n");
      exit (1);
    }

  mpfr_set_prec (y, 202);
  mpfr_fac_ui (y, 69, MPFR_RNDU);

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
test_int (void)
{
  unsigned long n0 = 1, n1 = 80, n;
  mpz_t f;
  mpfr_t x, y;
  mpfr_prec_t prec_f, p;
  int r;
  int inex1, inex2;

  mpz_init (f);
  mpfr_init (x);
  mpfr_init (y);

  mpz_fac_ui (f, n0 - 1);
  for (n = n0; n <= n1; n++)
    {
      mpz_mul_ui (f, f, n); /* f = n! */
      prec_f = mpz_sizeinbase (f, 2) - mpz_scan1 (f, 0);
      for (p = MPFR_PREC_MIN; p <= prec_f; p++)
        {
          mpfr_set_prec (x, p);
          mpfr_set_prec (y, p);
          for (r = 0; r < MPFR_RND_MAX; r++)
            {
              inex1 = mpfr_fac_ui (x, n, (mpfr_rnd_t) r);
              inex2 = mpfr_set_z (y, f, (mpfr_rnd_t) r);
              if (mpfr_cmp (x, y))
                {
                  printf ("Error for n=%lu prec=%lu rnd=%s\n",
                          n, (unsigned long) p, mpfr_print_rnd_mode ((mpfr_rnd_t) r));
                  exit (1);
                }
              if ((inex1 < 0 && inex2 >= 0) || (inex1 == 0 && inex2 != 0)
                  || (inex1 > 0 && inex2 <= 0))
                {
                  printf ("Wrong inexact flag for n=%lu prec=%lu rnd=%s\n",
                          n, (unsigned long) p, mpfr_print_rnd_mode ((mpfr_rnd_t) r));
                  exit (1);
                }
            }
        }
    }

  mpz_clear (f);
  mpfr_clear (x);
  mpfr_clear (y);
}

static void
overflowed_fac0 (void)
{
  mpfr_t x, y;
  int inex, rnd, err = 0;
  mpfr_exp_t old_emax;

  old_emax = mpfr_get_emax ();

  mpfr_init2 (x, 8);
  mpfr_init2 (y, 8);

  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_nextbelow (y);
  set_emax (0);  /* 1 is not representable. */
  RND_LOOP (rnd)
    {
      mpfr_clear_flags ();
      inex = mpfr_fac_ui (x, 0, (mpfr_rnd_t) rnd);
      if (! mpfr_overflow_p ())
        {
          printf ("Error in overflowed_fac0 (rnd = %s):\n"
                  "  The overflow flag is not set.\n",
                  mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
          err = 1;
        }
      if (rnd == MPFR_RNDZ || rnd == MPFR_RNDD)
        {
          if (inex >= 0)
            {
              printf ("Error in overflowed_fac0 (rnd = %s):\n"
                      "  The inexact value must be negative.\n",
                      mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
              err = 1;
            }
          if (! mpfr_equal_p (x, y))
            {
              printf ("Error in overflowed_fac0 (rnd = %s):\n"
                      "  Got ", mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
              mpfr_print_binary (x);
              printf (" instead of 0.11111111E0.\n");
              err = 1;
            }
        }
      else
        {
          if (inex <= 0)
            {
              printf ("Error in overflowed_fac0 (rnd = %s):\n"
                      "  The inexact value must be positive.\n",
                      mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
              err = 1;
            }
          if (! (mpfr_inf_p (x) && MPFR_SIGN (x) > 0))
            {
              printf ("Error in overflowed_fac0 (rnd = %s):\n"
                      "  Got ", mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
              mpfr_print_binary (x);
              printf (" instead of +Inf.\n");
              err = 1;
            }
        }
    }
  set_emax (old_emax);

  if (err)
    exit (1);
  mpfr_clear (x);
  mpfr_clear (y);
}

int
main (int argc, char *argv[])
{
  unsigned int prec, err, yprec, n, k, zeros;
  int rnd;
  mpfr_t x, y, z, t;
  int inexact;

  tests_start_mpfr ();

  special ();

  test_int ();

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);
  mpfr_init (t);

  mpfr_fac_ui (y, 0, MPFR_RNDN);

  if (mpfr_cmp_ui (y, 1))
    {
      printf ("mpfr_fac_ui(0) does not give 1\n");
      exit (1);
    }

  for (prec = 2; prec <= 100; prec++)
    {
      mpfr_set_prec (x, prec);
      mpfr_set_prec (z, prec);
      mpfr_set_prec (t, prec);
      yprec = prec + 10;
      mpfr_set_prec (y, yprec);

      for (n = 0; n < 50; n++)
        for (rnd = 0; rnd < MPFR_RND_MAX; rnd++)
          {
            inexact = mpfr_fac_ui (y, n, (mpfr_rnd_t) rnd);
            err = (rnd == MPFR_RNDN) ? yprec + 1 : yprec;
            if (mpfr_can_round (y, err, (mpfr_rnd_t) rnd, (mpfr_rnd_t) rnd, prec))
              {
                mpfr_set (t, y, (mpfr_rnd_t) rnd);
                inexact = mpfr_fac_ui (z, n, (mpfr_rnd_t) rnd);
                /* fact(n) ends with floor(n/2)+floor(n/4)+... zeros */
                for (k=n/2, zeros=0; k; k >>= 1)
                  zeros += k;
                if (MPFR_EXP(y) <= (mpfr_exp_t) (prec + zeros))
                  /* result should be exact */
                  {
                    if (inexact)
                      {
                        printf ("Wrong inexact flag: expected exact\n");
                        exit (1);
                      }
                  }
                else /* result is inexact */
                  {
                    if (!inexact)
                      {
                        printf ("Wrong inexact flag: expected inexact\n");
                        printf ("n=%u prec=%u\n", n, prec);
                        mpfr_print_binary(z); puts ("");
                        exit (1);
                      }
                  }
                if (mpfr_cmp (t, z))
                  {
                    printf ("results differ for x=");
                    mpfr_out_str (stdout, 2, prec, x, MPFR_RNDN);
                    printf (" prec=%u rnd_mode=%s\n", prec,
                            mpfr_print_rnd_mode ((mpfr_rnd_t) rnd));
                    printf ("   got ");
                    mpfr_out_str (stdout, 2, prec, z, MPFR_RNDN);
                    puts ("");
                    printf ("   expected ");
                    mpfr_out_str (stdout, 2, prec, t, MPFR_RNDN);
                    puts ("");
                    printf ("   approximation was ");
                    mpfr_print_binary (y);
                    puts ("");
                    exit (1);
                  }
              }
          }
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (t);

  overflowed_fac0 ();

  tests_end_mpfr ();
  return 0;
}
