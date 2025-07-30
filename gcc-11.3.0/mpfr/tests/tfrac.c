/* Test file for mpfr_frac.

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

#include "mpfr-test.h"

#define PIP 70
#define PFP 70
#define PMAX (PIP+2*PFP)

static void
check0 (mpfr_ptr ip, mpfr_ptr fp, mpfr_prec_t prec, mpfr_rnd_t rnd)
{
  mpfr_t sum, tmp, dst, fp2;
  int inex1, inex2;

  mpfr_init2 (sum, PMAX);
  mpfr_init2 (tmp, PMAX);
  mpfr_init2 (dst, prec);
  mpfr_init2 (fp2, prec);

  if (MPFR_SIGN (ip) != MPFR_SIGN (fp))
    {
      printf ("Internal error (1)\n");
      exit (1);
    }
  if (mpfr_add (sum, ip, fp, MPFR_RNDZ))
    {
      printf ("Wrong inexact flag in mpfr_add\n");
      exit (1);
    }
  if (MPFR_SIGN (sum) != MPFR_SIGN (fp))
    {
      printf ("Internal error (2)\n");
      exit (1);
    }

  inex1 = mpfr_frac (dst, sum, rnd);
  inex2 = mpfr_set (fp2, fp, rnd);
  if (inex1 != inex2)
    {
      printf ("Wrong inexact flag in mpfr_frac for\n");
      mpfr_out_str (stdout, 2, 0, sum, MPFR_RNDN);
      printf ("\nGot %d instead of %d\n", inex1, inex2);
      exit (1);
    }
  if (!mpfr_number_p (dst) ||
      MPFR_SIGN (dst) != MPFR_SIGN (fp2) ||
      mpfr_cmp (dst, fp2))
    {
      printf ("Error in mpfr_frac (y, x, %s) with\nx = ",
              mpfr_print_rnd_mode (rnd));
      mpfr_out_str (stdout, 2, 0, sum, MPFR_RNDN);
      printf ("\nGot        ");
      mpfr_out_str (stdout, 2, 0, dst, MPFR_RNDN);
      printf ("\ninstead of ");
      mpfr_out_str (stdout, 2, 0, fp2, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  if (prec == PMAX)
    {
      inex1 = mpfr_frac (sum, sum, rnd);
      if (inex1)
        {
          printf ("Wrong inexact flag in mpfr_frac\n");
          exit (1);
        }
      if (!mpfr_number_p (sum) ||
          MPFR_SIGN (sum) != MPFR_SIGN (fp) ||
          mpfr_cmp (sum, fp))
        {
          printf ("Error in mpfr_frac (x, x, %s) with\nx = ",
                  mpfr_print_rnd_mode (rnd));
          mpfr_add (tmp, ip, fp, MPFR_RNDZ);
          mpfr_out_str (stdout, 2, 0, tmp, MPFR_RNDN);
          printf ("\nGot        ");
          mpfr_out_str (stdout, 2, 0, sum, MPFR_RNDN);
          printf ("\ninstead of ");
          mpfr_out_str (stdout, 2, 0, fp, MPFR_RNDN);
          printf ("\n");
          exit (1);
        }
    }

  mpfr_clear (fp2);
  mpfr_clear (dst);
  mpfr_clear (tmp);
  mpfr_clear (sum);
}

static void
check1 (mpfr_ptr ip, mpfr_ptr fp)
{
  int rnd;

  for (rnd = 0; rnd < MPFR_RND_MAX ; rnd++)
    {
      check0 (ip, fp, PMAX, (mpfr_rnd_t) rnd);
      check0 (ip, fp, 70, (mpfr_rnd_t) rnd);
      mpfr_neg (fp, fp, MPFR_RNDN);
      mpfr_neg (ip, ip, MPFR_RNDN);
      check0 (ip, fp, PMAX, (mpfr_rnd_t) rnd);
      check0 (ip, fp, 70, (mpfr_rnd_t) rnd);
      mpfr_neg (fp, fp, MPFR_RNDN);
      mpfr_neg (ip, ip, MPFR_RNDN);
    }
}

static void
special (void)
{
  mpfr_t z, t;

  mpfr_init (z);
  mpfr_init (t);

  mpfr_set_nan (z);
  mpfr_frac (t, z, MPFR_RNDN);
  if (!mpfr_nan_p (t))
    {
      printf ("Error for frac(NaN)\n");
      exit (1);
    }

  mpfr_set_prec (z, 6);
  mpfr_set_prec (t, 3);

  mpfr_set_str_binary (z, "0.101101E3");
  mpfr_frac (t, z, MPFR_RNDN);
  mpfr_set_str_binary (z, "0.101");
  if (mpfr_cmp (t, z))
    {
      printf ("Error in frac(0.101101E3)\n");
      exit (1);
    }

  mpfr_set_prec (z, 34);
  mpfr_set_prec (t, 26);
  mpfr_set_str_binary (z, "0.101101010000010011110011001101E9");
  mpfr_frac (t, z, MPFR_RNDN);
  mpfr_set_str_binary (z, "0.000010011110011001101");
  if (mpfr_cmp (t, z))
    {
      printf ("Error in frac(0.101101010000010011110011001101E9)\n");
      exit (1);
    }

  mpfr_clear (z);
  mpfr_clear (t);
}

static void
bug20090918 (void)
{
  mpfr_t x, y, z;
  mp_limb_t y0;
  int inexy, inexz;
  int r, i;
  const char *s[] = { "61680.352935791015625", "61680.999999" };
  mpfr_exp_t emin;

  emin = mpfr_get_emin ();
  mpfr_init2 (x, 32);
  mpfr_init2 (y, 13);

  for (i = 0; i <= 9; i++)
    {
      mpfr_set_str (x, s[i & 1], 10, MPFR_RNDZ);

      RND_LOOP(r)
        {
          set_emin ((i >> 1) - 3);
          inexy = mpfr_frac (y, x, (mpfr_rnd_t) r);
          set_emin (emin);
          y0 = MPFR_MANT(y)[0];
          while (y0 != 0 && (y0 >> 1) << 1 == y0)
            y0 >>= 1;
          if (y0 > 0x2000)
            {
              printf ("Error in bug20090918 (significand has more than"
                      " 13 bits), i = %d, %s.\n", i,
                      mpfr_print_rnd_mode ((mpfr_rnd_t) r));
              exit (1);
            }
          mpfr_init2 (z, 32);
          inexz = mpfr_frac (z, x, MPFR_RNDN);
          MPFR_ASSERTN (inexz == 0);  /* exact */
          inexz = mpfr_prec_round (z, 13, (mpfr_rnd_t) r);
          set_emin ((i >> 1) - 3);
          inexz = mpfr_check_range (z, inexz, (mpfr_rnd_t) r);
          set_emin (emin);
          if (mpfr_cmp0 (y, z) != 0)
            {
              printf ("Error in bug20090918, i = %d, %s.\n", i,
                      mpfr_print_rnd_mode ((mpfr_rnd_t) r));
              printf ("Expected ");
              mpfr_dump (z);
              printf ("Got      ");
              mpfr_dump (y);
              exit (1);
            }
          if (! SAME_SIGN (inexy, inexz))
            {
              printf ("Incorrect ternary value in bug20090918, i = %d, %s.\n",
                      i, mpfr_print_rnd_mode ((mpfr_rnd_t) r));
              printf ("Expected %d, got %d.\n", inexz, inexy);
              exit (1);
            }
          mpfr_clear (z);
        }
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

#define TEST_FUNCTION mpfr_frac
#include "tgeneric.c"

int
main (void)
{
  mpfr_t ip, fp;
  int ni, nf1, nf2;

  tests_start_mpfr ();

  special ();

  mpfr_init2 (ip, PIP);
  mpfr_init2 (fp, PFP);

  for (ni = -1; ni < PIP; ni++)
    {
      if (ni <= 0)
        { /* ni + 1 */
          mpfr_set_si (ip, ni, MPFR_RNDN);
          mpfr_add_ui (ip, ip, 1, MPFR_RNDN);
        }
      else
        { /* 2^ni + 1 */
          mpfr_set_ui (ip, 1, MPFR_RNDN);
          mpfr_mul_2ui (ip, ip, ni, MPFR_RNDN);
          mpfr_add_ui (ip, ip, 1, MPFR_RNDN);
        }

      mpfr_set_ui (fp, 0, MPFR_RNDN);
      check1 (ip, fp);

      for (nf1 = 1; nf1 < PFP; nf1++)
        {
          mpfr_set_ui (fp, 1, MPFR_RNDN);
          mpfr_div_2ui (fp, fp, nf1, MPFR_RNDN);
          check1 (ip, fp);
          nf2 = 1 + (randlimb () % (PFP - 1));
          mpfr_set_ui (fp, 1, MPFR_RNDN);
          mpfr_div_2ui (fp, fp, nf2, MPFR_RNDN);
          mpfr_add_ui (fp, fp, 1, MPFR_RNDN);
          mpfr_div_2ui (fp, fp, nf1, MPFR_RNDN);
          check1 (ip, fp);
        }
    }

  mpfr_set_ui (ip, 1, MPFR_RNDN);
  mpfr_div_ui (ip, ip, 0, MPFR_RNDN);
  mpfr_set_ui (fp, 0, MPFR_RNDN);
  check1 (ip, fp);  /* test infinities */

  mpfr_clear (ip);
  mpfr_clear (fp);

  bug20090918 ();

  test_generic (2, 1000, 10);

  tests_end_mpfr ();
  return 0;
}
