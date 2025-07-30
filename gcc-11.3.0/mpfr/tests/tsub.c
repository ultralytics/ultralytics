/* Test file for mpfr_sub.

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

#ifdef CHECK_EXTERNAL
static int
test_sub (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mpfr_rnd_t rnd_mode)
{
  int res;
  int ok = rnd_mode == MPFR_RNDN && mpfr_number_p (b) && mpfr_number_p (c);
  if (ok)
    {
      mpfr_print_raw (b);
      printf (" ");
      mpfr_print_raw (c);
    }
  res = mpfr_sub (a, b, c, rnd_mode);
  if (ok)
    {
      printf (" ");
      mpfr_print_raw (a);
      printf ("\n");
    }
  return res;
}
#else
#define test_sub mpfr_sub
#endif

static void
check_diverse (void)
{
  mpfr_t x, y, z;
  int inexact;

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);

  /* check corner case cancel=0, but add_exp=1 */
  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 4);
  mpfr_set_prec (z, 2);
  mpfr_setmax (y, __gmpfr_emax);
  mpfr_set_str_binary (z, "0.1E-10"); /* tiny */
  test_sub (x, y, z, MPFR_RNDN); /* should round to 2^emax, i.e. overflow */
  if (!mpfr_inf_p (x) || mpfr_sgn (x) < 0)
    {
      printf ("Error in mpfr_sub(a,b,c,RNDN) for b=maxfloat, prec(a)<prec(b), c tiny\n");
      exit (1);
    }

  /* other coverage test */
  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 2);
  mpfr_set_prec (z, 2);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_set_si (z, -2, MPFR_RNDN);
  test_sub (x, y, z, MPFR_RNDD);
  if (mpfr_cmp_ui (x, 3))
    {
      printf ("Error in mpfr_sub(1,-2,RNDD)\n");
      exit (1);
    }

  mpfr_set_prec (x, 288);
  mpfr_set_prec (y, 288);
  mpfr_set_prec (z, 288);
  mpfr_set_str_binary (y, "0.111000110011000001000111101010111011110011101001101111111110000011100101000001001010110010101010011001010100000001110011110001010101101010001011101110100100001011110100110000101101100011010001001011011010101010000010001101001000110010010111111011110001111101001000101101001100101100101000E80");
  mpfr_set_str_binary (z, "0.100001111111101001011010001100110010100111001110000110011101001011010100001000000100111011010110110010000000000010101101011000010000110001110010100001100101011100100100001011000100011110000001010101000100011101001000010111100000111000111011001000100100011000100000010010111000000100100111E-258");
  inexact = test_sub (x, y, z, MPFR_RNDN);
  if (inexact <= 0)
    {
      printf ("Wrong inexact flag for prec=288\n");
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_set_prec (y, 63);
  mpfr_set_prec (z, 63);
  mpfr_set_str_binary (x, "0.101101111011011100100100100111E31");
  mpfr_set_str_binary (y, "0.111110010010100100110101101010001001100101110001000101110111111E-1");
  test_sub (z, x, y, MPFR_RNDN);
  mpfr_set_str_binary (y, "0.1011011110110111001001001001101100000110110101101100101001011E31");
  if (mpfr_cmp (z, y))
    {
      printf ("Error in mpfr_sub (5)\n");
      printf ("expected "); mpfr_print_binary (y); puts ("");
      printf ("got      "); mpfr_print_binary (z); puts ("");
      exit (1);
    }

  mpfr_set_prec (y, 63);
  mpfr_set_prec (z, 63);
  mpfr_set_str_binary (y, "0.1011011110110111001001001001101100000110110101101100101001011E31");
  mpfr_sub_ui (z, y, 1541116494, MPFR_RNDN);
  mpfr_set_str_binary (y, "-0.11111001001010010011010110101E-1");
  if (mpfr_cmp (z, y))
    {
      printf ("Error in mpfr_sub (7)\n");
      printf ("expected "); mpfr_print_binary (y); puts ("");
      printf ("got      "); mpfr_print_binary (z); puts ("");
      exit (1);
    }

  mpfr_set_prec (y, 63);
  mpfr_set_prec (z, 63);
  mpfr_set_str_binary (y, "0.1011011110110111001001001001101100000110110101101100101001011E31");
  mpfr_sub_ui (z, y, 1541116494, MPFR_RNDN);
  mpfr_set_str_binary (y, "-0.11111001001010010011010110101E-1");
  if (mpfr_cmp (z, y))
    {
      printf ("Error in mpfr_sub (6)\n");
      printf ("expected "); mpfr_print_binary (y); puts ("");
      printf ("got      "); mpfr_print_binary (z); puts ("");
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_set_prec (y, 32);
  mpfr_set_str_binary (x, "0.10110111101001110100100101111000E0");
  mpfr_set_str_binary (y, "0.10001100100101000100110111000100E0");
  if ((inexact = test_sub (x, x, y, MPFR_RNDN)))
    {
      printf ("Wrong inexact flag (2): got %d instead of 0\n", inexact);
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_set_prec (y, 32);
  mpfr_set_str_binary (x, "0.11111000110111011000100111011010E0");
  mpfr_set_str_binary (y, "0.10011111101111000100001000000000E-3");
  if ((inexact = test_sub (x, x, y, MPFR_RNDN)))
    {
      printf ("Wrong inexact flag (1): got %d instead of 0\n", inexact);
      exit (1);
    }

  mpfr_set_prec (x, 33);
  mpfr_set_prec (y, 33);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_str_binary (y, "-0.1E-32");
  mpfr_add (x, x, y, MPFR_RNDN);
  mpfr_set_str_binary (y, "0.111111111111111111111111111111111E0");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_sub (1 - 1E-33) with prec=33\n");
      printf ("Expected "); mpfr_print_binary (y); puts ("");
      printf ("got      "); mpfr_print_binary (x); puts ("");
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_set_prec (y, 33);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_str_binary (y, "-0.1E-32");
  mpfr_add (x, x, y, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 1))
    {
      printf ("Error in mpfr_sub (1 - 1E-33) with prec=32\n");
      printf ("Expected 1.0, got "); mpfr_print_binary (x); puts ("");
      exit (1);
    }

  mpfr_set_prec (x, 65);
  mpfr_set_prec (y, 65);
  mpfr_set_prec (z, 64);
  mpfr_set_str_binary (x, "1.1110111011110001110111011111111111101000011001011100101100101101");
  mpfr_set_str_binary (y, "0.1110111011110001110111011111111111101000011001011100101100101100");
  test_sub (z, x, y, MPFR_RNDZ);
  if (mpfr_cmp_ui (z, 1))
    {
      printf ("Error in mpfr_sub (1)\n");
      exit (1);
    }
  test_sub (z, x, y, MPFR_RNDU);
  mpfr_set_str_binary (x, "1.000000000000000000000000000000000000000000000000000000000000001");
  if (mpfr_cmp (z, x))
    {
      printf ("Error in mpfr_sub (2)\n");
      printf ("Expected "); mpfr_print_binary (x); puts ("");
      printf ("Got      "); mpfr_print_binary (z); puts ("");
      exit (1);
    }
  mpfr_set_str_binary (x, "1.1110111011110001110111011111111111101000011001011100101100101101");
  test_sub (z, x, y, MPFR_RNDN);
  if (mpfr_cmp_ui (z, 1))
    {
      printf ("Error in mpfr_sub (3)\n");
      exit (1);
    }
  inexact = test_sub (z, x, y, MPFR_RNDA);
  mpfr_set_str_binary (x, "1.000000000000000000000000000000000000000000000000000000000000001");
  if (mpfr_cmp (z, x) || inexact <= 0)
    {
      printf ("Error in mpfr_sub (4)\n");
      exit (1);
    }
  mpfr_set_prec (x, 66);
  mpfr_set_str_binary (x, "1.11101110111100011101110111111111111010000110010111001011001010111");
  test_sub (z, x, y, MPFR_RNDN);
  if (mpfr_cmp_ui (z, 1))
    {
      printf ("Error in mpfr_sub (5)\n");
      exit (1);
    }

  /* check in-place operations */
  mpfr_set_ui (x, 1, MPFR_RNDN);
  test_sub (x, x, x, MPFR_RNDN);
  if (mpfr_cmp_ui(x, 0))
    {
      printf ("Error for mpfr_sub (x, x, x, MPFR_RNDN) with x=1.0\n");
      exit (1);
    }

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_prec (z, 53);
  mpfr_set_str1 (x, "1.229318102e+09");
  mpfr_set_str1 (y, "2.32221184180698677665e+05");
  test_sub (z, x, y, MPFR_RNDN);
  if (mpfr_cmp_str1 (z, "1229085880.815819263458251953125"))
    {
      printf ("Error in mpfr_sub (1.22e9 - 2.32e5)\n");
      printf ("expected 1229085880.815819263458251953125, got ");
      mpfr_out_str(stdout, 10, 0, z, MPFR_RNDN);
      putchar('\n');
      exit (1);
    }

  mpfr_set_prec (x, 112);
  mpfr_set_prec (y, 98);
  mpfr_set_prec (z, 54);
  mpfr_set_str_binary (x, "0.11111100100000000011000011100000101101010001000111E-401");
  mpfr_set_str_binary (y, "0.10110000100100000101101100011111111011101000111000101E-464");
  test_sub (z, x, y, MPFR_RNDN);
  if (mpfr_cmp (z, x)) {
    printf ("mpfr_sub(z, x, y) failed for prec(x)=112, prec(y)=98\n");
    printf ("expected "); mpfr_print_binary (x); puts ("");
    printf ("got      "); mpfr_print_binary (z); puts ("");
    exit (1);
  }

  mpfr_set_prec (x, 33);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_div_2exp (x, x, 32, MPFR_RNDN);
  mpfr_sub_ui (x, x, 1, MPFR_RNDN);

  mpfr_set_prec (x, 5);
  mpfr_set_prec (y, 5);
  mpfr_set_str_binary (x, "1e-12");
  mpfr_set_ui (y, 1, MPFR_RNDN);
  test_sub (x, y, x, MPFR_RNDD);
  mpfr_set_str_binary (y, "0.11111");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_sub (x, y, x, MPFR_RNDD) for x=2^(-12), y=1\n");
      exit (1);
    }

  mpfr_set_prec (x, 24);
  mpfr_set_prec (y, 24);
  mpfr_set_str_binary (x, "-0.100010000000000000000000E19");
  mpfr_set_str_binary (y, "0.100000000000000000000100E15");
  mpfr_add (x, x, y, MPFR_RNDD);
  mpfr_set_str_binary (y, "-0.1E19");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_add (2)\n");
      exit (1);
    }

  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 10);
  mpfr_set_prec (z, 10);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_set_str_binary (z, "0.10001");
  if (test_sub (x, y, z, MPFR_RNDN) <= 0)
    {
      printf ("Wrong inexact flag in x=mpfr_sub(0,z) for prec(z)>prec(x)\n");
      exit (1);
    }
  if (test_sub (x, z, y, MPFR_RNDN) >= 0)
    {
      printf ("Wrong inexact flag in x=mpfr_sub(z,0) for prec(z)>prec(x)\n");
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

static void
bug_ddefour(void)
{
    mpfr_t ex, ex1, ex2, ex3, tot, tot1;

    mpfr_init2(ex, 53);
    mpfr_init2(ex1, 53);
    mpfr_init2(ex2, 53);
    mpfr_init2(ex3, 53);
    mpfr_init2(tot, 150);
    mpfr_init2(tot1, 150);

    mpfr_set_ui( ex, 1, MPFR_RNDN);
    mpfr_mul_2exp( ex, ex, 906, MPFR_RNDN);
    mpfr_log( tot, ex, MPFR_RNDN);
    mpfr_set( ex1, tot, MPFR_RNDN); /* ex1 = high(tot) */
    test_sub( ex2, tot, ex1, MPFR_RNDN); /* ex2 = high(tot - ex1) */
    test_sub( tot1, tot, ex1, MPFR_RNDN); /* tot1 = tot - ex1 */
    mpfr_set( ex3, tot1, MPFR_RNDN); /* ex3 = high(tot - ex1) */

    if (mpfr_cmp(ex2, ex3))
      {
        printf ("Error in ddefour test.\n");
        printf ("ex2="); mpfr_print_binary (ex2); puts ("");
        printf ("ex3="); mpfr_print_binary (ex3); puts ("");
        exit (1);
      }

    mpfr_clear (ex);
    mpfr_clear (ex1);
    mpfr_clear (ex2);
    mpfr_clear (ex3);
    mpfr_clear (tot);
    mpfr_clear (tot1);
}

/* if u = o(x-y), v = o(u-x), w = o(v+y), then x-y = u-w */
static void
check_two_sum (mpfr_prec_t p)
{
  mpfr_t x, y, u, v, w;
  mpfr_rnd_t rnd;
  int inexact;

  mpfr_init2 (x, p);
  mpfr_init2 (y, p);
  mpfr_init2 (u, p);
  mpfr_init2 (v, p);
  mpfr_init2 (w, p);
  mpfr_urandomb (x, RANDS);
  mpfr_urandomb (y, RANDS);
  if (mpfr_cmpabs (x, y) < 0)
    mpfr_swap (x, y);
  rnd = MPFR_RNDN;
  inexact = test_sub (u, x, y, rnd);
  test_sub (v, u, x, rnd);
  mpfr_add (w, v, y, rnd);
  /* as u = (x-y) - w, we should have inexact and w of opposite signs */
  if (((inexact == 0) && mpfr_cmp_ui (w, 0)) ||
      ((inexact > 0) && (mpfr_cmp_ui (w, 0) <= 0)) ||
      ((inexact < 0) && (mpfr_cmp_ui (w, 0) >= 0)))
    {
      printf ("Wrong inexact flag for prec=%u, rnd=%s\n", (unsigned)p,
               mpfr_print_rnd_mode (rnd));
      printf ("x="); mpfr_print_binary(x); puts ("");
      printf ("y="); mpfr_print_binary(y); puts ("");
      printf ("u="); mpfr_print_binary(u); puts ("");
      printf ("v="); mpfr_print_binary(v); puts ("");
      printf ("w="); mpfr_print_binary(w); puts ("");
      printf ("inexact = %d\n", inexact);
      exit (1);
    }
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (u);
  mpfr_clear (v);
  mpfr_clear (w);
}

#define MAX_PREC 200

static void
check_inexact (void)
{
  mpfr_t x, y, z, u;
  mpfr_prec_t px, py, pu, pz;
  int inexact, cmp;
  mpfr_rnd_t rnd;

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);
  mpfr_init (u);

  mpfr_set_prec (x, 2);
  mpfr_set_ui (x, 6, MPFR_RNDN);
  mpfr_div_2exp (x, x, 4, MPFR_RNDN); /* x = 6/16 */
  mpfr_set_prec (y, 2);
  mpfr_set_si (y, -1, MPFR_RNDN);
  mpfr_div_2exp (y, y, 4, MPFR_RNDN); /* y = -1/16 */
  inexact = test_sub (y, y, x, MPFR_RNDN); /* y = round(-7/16) = -1/2 */
  if (inexact >= 0)
    {
      printf ("Error: wrong inexact flag for -1/16 - (6/16)\n");
      exit (1);
    }

  for (px=2; px<MAX_PREC; px++)
    {
      mpfr_set_prec (x, px);
      do
        {
          mpfr_urandomb (x, RANDS);
        }
      while (mpfr_cmp_ui (x, 0) == 0);
      for (pu=2; pu<MAX_PREC; pu++)
        {
          mpfr_set_prec (u, pu);
          do
            {
              mpfr_urandomb (u, RANDS);
            }
          while (mpfr_cmp_ui (u, 0) == 0);
          {
              py = 2 + (randlimb () % (MAX_PREC - 2));
              mpfr_set_prec (y, py);
              /* warning: MPFR_EXP is undefined for 0 */
              pz =  (mpfr_cmpabs (x, u) >= 0) ? MPFR_EXP(x) - MPFR_EXP(u)
                : MPFR_EXP(u) - MPFR_EXP(x);
              pz = pz + MAX(MPFR_PREC(x), MPFR_PREC(u));
              mpfr_set_prec (z, pz);
              rnd = RND_RAND ();
              if (test_sub (z, x, u, rnd))
                {
                  printf ("z <- x - u should be exact\n");
                  exit (1);
                }
                {
                  rnd = RND_RAND ();
                  inexact = test_sub (y, x, u, rnd);
                  cmp = mpfr_cmp (y, z);
                  if (((inexact == 0) && (cmp != 0)) ||
                      ((inexact > 0) && (cmp <= 0)) ||
                      ((inexact < 0) && (cmp >= 0)))
                    {
                      printf ("Wrong inexact flag for rnd=%s\n",
                              mpfr_print_rnd_mode(rnd));
                      printf ("expected %d, got %d\n", cmp, inexact);
                      printf ("x="); mpfr_print_binary (x); puts ("");
                      printf ("u="); mpfr_print_binary (u); puts ("");
                      printf ("y=  "); mpfr_print_binary (y); puts ("");
                      printf ("x-u="); mpfr_print_binary (z); puts ("");
                      exit (1);
                    }
                }
            }
        }
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (u);
}

/* Bug found by Jakub Jelinek
 * http://bugzilla.redhat.com/643657
 * https://gforge.inria.fr/tracker/index.php?func=detail&aid=11301
 * The consequence can be either an assertion failure (i = 2 in the
 * testcase below, in debug mode) or an incorrectly rounded value.
 */
static void
bug20101017 (void)
{
  mpfr_t a, b, c;
  int inex;
  int i;

  mpfr_init2 (a, GMP_NUMB_BITS * 2);
  mpfr_init2 (b, GMP_NUMB_BITS);
  mpfr_init2 (c, GMP_NUMB_BITS);

  /* a = 2^(2N) + k.2^(2N-1) + 2^N and b = 1
     with N = GMP_NUMB_BITS and k = 0 or 1.
     c = a - b should round to the same value as a. */

  for (i = 2; i <= 3; i++)
    {
      mpfr_set_ui_2exp (a, i, GMP_NUMB_BITS - 1, MPFR_RNDN);
      mpfr_add_ui (a, a, 1, MPFR_RNDN);
      mpfr_mul_2ui (a, a, GMP_NUMB_BITS, MPFR_RNDN);
      mpfr_set_ui (b, 1, MPFR_RNDN);
      inex = mpfr_sub (c, a, b, MPFR_RNDN);
      mpfr_set (b, a, MPFR_RNDN);
      if (! mpfr_equal_p (c, b))
        {
          printf ("Error in bug20101017 for i = %d.\n", i);
          printf ("Expected ");
          mpfr_out_str (stdout, 16, 0, b, MPFR_RNDN);
          putchar ('\n');
          printf ("Got      ");
          mpfr_out_str (stdout, 16, 0, c, MPFR_RNDN);
          putchar ('\n');
          exit (1);
        }
      if (inex >= 0)
        {
          printf ("Error in bug20101017 for i = %d: bad inex value.\n", i);
          printf ("Expected negative, got %d.\n", inex);
          exit (1);
        }
    }

  mpfr_set_prec (a, 64);
  mpfr_set_prec (b, 129);
  mpfr_set_prec (c, 2);
  mpfr_set_str_binary (b, "0.100000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000001E65");
  mpfr_set_str_binary (c, "0.10E1");
  inex = mpfr_sub (a, b, c, MPFR_RNDN);
  if (mpfr_cmp_ui_2exp (a, 1, 64) != 0 || inex >= 0)
    {
      printf ("Error in mpfr_sub for b-c for b=2^64+1+2^(-64), c=1\n");
      printf ("Expected result 2^64 with inex < 0\n");
      printf ("Got "); mpfr_print_binary (a);
      printf (" with inex=%d\n", inex);
      exit (1);
    }

  mpfr_clears (a, b, c, (mpfr_ptr) 0);
}

/* hard test of rounding */
static void
check_rounding (void)
{
  mpfr_t a, b, c, res;
  mpfr_prec_t p;
  long k, l;
  int i;

#define MAXKL (2 * GMP_NUMB_BITS)
  for (p = MPFR_PREC_MIN; p <= GMP_NUMB_BITS; p++)
    {
      mpfr_init2 (a, p);
      mpfr_init2 (res, p);
      mpfr_init2 (b, p + 1 + MAXKL);
      mpfr_init2 (c, MPFR_PREC_MIN);

      /* b = 2^p + 1 + 2^(-k), c = 2^(-l) */
      for (k = 0; k <= MAXKL; k++)
        for (l = 0; l <= MAXKL; l++)
          {
            mpfr_set_ui_2exp (b, 1, p, MPFR_RNDN);
            mpfr_add_ui (b, b, 1, MPFR_RNDN);
            mpfr_mul_2ui (b, b, k, MPFR_RNDN);
            mpfr_add_ui (b, b, 1, MPFR_RNDN);
            mpfr_div_2ui (b, b, k, MPFR_RNDN);
            mpfr_set_ui_2exp (c, 1, -l, MPFR_RNDN);
            i = mpfr_sub (a, b, c, MPFR_RNDN);
            /* b - c = 2^p + 1 + 2^(-k) - 2^(-l), should be rounded to
               2^p for l <= k, and 2^p+2 for l < k */
            if (l <= k)
              {
                if (mpfr_cmp_ui_2exp (a, 1, p) != 0)
                  {
                    printf ("Wrong result in check_rounding\n");
                    printf ("p=%lu k=%ld l=%ld\n", (unsigned long) p, k, l);
                    printf ("b="); mpfr_print_binary (b); puts ("");
                    printf ("c="); mpfr_print_binary (c); puts ("");
                    printf ("Expected 2^%lu\n", (unsigned long) p);
                    printf ("Got      "); mpfr_print_binary (a); puts ("");
                    exit (1);
                  }
                if (i >= 0)
                  {
                    printf ("Wrong ternary value in check_rounding\n");
                    printf ("p=%lu k=%ld l=%ld\n", (unsigned long) p, k, l);
                    printf ("b="); mpfr_print_binary (b); puts ("");
                    printf ("c="); mpfr_print_binary (c); puts ("");
                    printf ("a="); mpfr_print_binary (a); puts ("");
                    printf ("Expected < 0, got %d\n", i);
                    exit (1);
                  }
              }
            else /* l < k */
              {
                mpfr_set_ui_2exp (res, 1, p, MPFR_RNDN);
                mpfr_add_ui (res, res, 2, MPFR_RNDN);
                if (mpfr_cmp (a, res) != 0)
                  {
                    printf ("Wrong result in check_rounding\n");
                    printf ("b="); mpfr_print_binary (b); puts ("");
                    printf ("c="); mpfr_print_binary (c); puts ("");
                    printf ("Expected "); mpfr_print_binary (res); puts ("");
                    printf ("Got      "); mpfr_print_binary (a); puts ("");
                    exit (1);
                  }
                if (i <= 0)
                  {
                    printf ("Wrong ternary value in check_rounding\n");
                    printf ("b="); mpfr_print_binary (b); puts ("");
                    printf ("c="); mpfr_print_binary (c); puts ("");
                    printf ("Expected > 0, got %d\n", i);
                    exit (1);
                  }
              }
          }

      mpfr_clear (a);
      mpfr_clear (res);
      mpfr_clear (b);
      mpfr_clear (c);
    }
}

/* Check a = b - c, where the significand of b has all 1's, c is small
   compared to b, and PREC(a) = PREC(b) - 1. Thus b is a midpoint for
   the precision of the result a. The test is done with the extended
   exponent range and with some reduced exponent range. Two choices
   are made for the exponent of b: the maximum exponent - 1 (similar
   to some normal case) and the maximum exponent (overflow case or
   near overflow case, depending on the rounding mode).
   This test is useful to trigger a bug in r10382: Since c is small,
   the computation in sub1.c was done by first rounding b in the
   precision of a, then correcting the result if b was a breakpoint
   for this precision (exactly representable number for the directed
   rounding modes, or midpoint for the round-to-nearest mode). The
   problem was that for a midpoint in the round-to-nearest mode, the
   rounding of b gave a spurious overflow; not only the overflow flag
   was incorrect, but the result could not be corrected, since due to
   this overflow, the "even rounding" information was lost.
   In the case of reduced exponent range, an additional test is done
   for consistency checks: the subtraction is done in the extended
   exponent range (no overflow), then the result is converted to the
   initial exponent range with mpfr_check_range. */
static void
check_max_almosteven (void)
{
  mpfr_exp_t old_emin, old_emax;
  mpfr_exp_t emin[2] = { MPFR_EMIN_MIN, -1000 };
  mpfr_exp_t emax[2] = { MPFR_EMAX_MAX, 1000 };
  int i;

  old_emin = mpfr_get_emin ();
  old_emax = mpfr_get_emax ();

  for (i = 0; i < 2; i++)
    {
      mpfr_t a1, a2, b, c;
      mpfr_prec_t p;
      int neg, j, rnd;

      set_emin (emin[i]);
      set_emax (emax[i]);

      p = MPFR_PREC_MIN + randlimb () % 70;
      mpfr_init2 (a1, p);
      mpfr_init2 (a2, p);
      mpfr_init2 (b, p+1);
      mpfr_init2 (c, MPFR_PREC_MIN);

      mpfr_setmax (b, 0);
      mpfr_set_ui (c, 1, MPFR_RNDN);

      for (neg = 0; neg < 2; neg++)
        {
          for (j = 1; j >= 0; j--)
            {
              mpfr_set_exp (b, __gmpfr_emax - j);
              RND_LOOP (rnd)
                {
                  unsigned int flags1, flags2;
                  int inex1, inex2;

                  /* Expected result. */
                  flags1 = MPFR_FLAGS_INEXACT;
                  if (rnd == MPFR_RNDN || MPFR_IS_LIKE_RNDZ (rnd, neg))
                    {
                      inex1 = neg ? 1 : -1;
                      mpfr_setmax (a1, __gmpfr_emax - j);
                    }
                  else
                    {
                      inex1 = neg ? -1 : 1;
                      if (j == 0)
                        {
                          flags1 |= MPFR_FLAGS_OVERFLOW;
                          mpfr_set_inf (a1, 1);
                        }
                      else
                        {
                          mpfr_setmin (a1, __gmpfr_emax);
                        }
                    }
                  MPFR_SET_SIGN (a1, neg ? -1 : 1);

                  /* Computed result. */
                  mpfr_clear_flags ();
                  inex2 = mpfr_sub (a2, b, c, (mpfr_rnd_t) rnd);
                  flags2 = __gmpfr_flags;

                  if (! (flags1 == flags2 && SAME_SIGN (inex1, inex2) &&
                         mpfr_equal_p (a1, a2)))
                    {
                      printf ("Error 1 in check_max_almosteven for %s,"
                              " i = %d, j = %d, neg = %d\n",
                              mpfr_print_rnd_mode ((mpfr_rnd_t) rnd),
                              i, j, neg);
                      printf ("     b = ");
                      mpfr_dump (b);
                      printf ("Expected ");
                      mpfr_dump (a1);
                      printf ("  with inex = %d, flags =", inex1);
                      flags_out (flags1);
                      printf ("Got      ");
                      mpfr_dump (a2);
                      printf ("  with inex = %d, flags =", inex2);
                      flags_out (flags2);
                      exit (1);
                    }

                  if (i == 0)
                    break;

                  /* Additional test for the reduced exponent range. */
                  mpfr_clear_flags ();
                  set_emin (MPFR_EMIN_MIN);
                  set_emax (MPFR_EMAX_MAX);
                  inex2 = mpfr_sub (a2, b, c, (mpfr_rnd_t) rnd);
                  set_emin (emin[i]);
                  set_emax (emax[i]);
                  inex2 = mpfr_check_range (a2, inex2, (mpfr_rnd_t) rnd);
                  flags2 = __gmpfr_flags;

                  if (! (flags1 == flags2 && SAME_SIGN (inex1, inex2) &&
                         mpfr_equal_p (a1, a2)))
                    {
                      printf ("Error 2 in check_max_almosteven for %s,"
                              " i = %d, j = %d, neg = %d\n",
                              mpfr_print_rnd_mode ((mpfr_rnd_t) rnd),
                              i, j, neg);
                      printf ("     b = ");
                      mpfr_dump (b);
                      printf ("Expected ");
                      mpfr_dump (a1);
                      printf ("  with inex = %d, flags =", inex1);
                      flags_out (flags1);
                      printf ("Got      ");
                      mpfr_dump (a2);
                      printf ("  with inex = %d, flags =", inex2);
                      flags_out (flags2);
                      exit (1);
                    }
                }
            }  /* j */

          mpfr_neg (b, b, MPFR_RNDN);
          mpfr_neg (c, c, MPFR_RNDN);
        }  /* neg */

      mpfr_clears (a1, a2, b, c, (mpfr_ptr) 0);
    }  /* i */

  set_emin (old_emin);
  set_emax (old_emax);
}

#define TEST_FUNCTION test_sub
#define TWO_ARGS
#define RAND_FUNCTION(x) mpfr_random2(x, MPFR_LIMB_SIZE (x), randlimb () % 100, RANDS)
#include "tgeneric.c"

int
main (void)
{
  mpfr_prec_t p;
  unsigned int i;

  tests_start_mpfr ();

  bug20101017 ();
  check_rounding ();
  check_diverse ();
  check_inexact ();
  check_max_almosteven ();
  bug_ddefour ();
  for (p=2; p<200; p++)
    for (i=0; i<50; i++)
      check_two_sum (p);
  test_generic (2, 800, 100);

  tests_end_mpfr ();
  return 0;
}
