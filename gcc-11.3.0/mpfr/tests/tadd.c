/* Test file for mpfr_add and mpfr_sub.

Copyright 1999-2017 Free Software Foundation, Inc.
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

#define N 30000

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "mpfr-test.h"

/* If the precisions are the same, we want to test both mpfr_add1sp
   and mpfr_add1. */

static int usesp;

static int
test_add (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mpfr_rnd_t rnd_mode)
{
  int res;
#ifdef CHECK_EXTERNAL
  int ok = rnd_mode == MPFR_RNDN && mpfr_number_p (b) && mpfr_number_p (c);
  if (ok)
    {
      mpfr_print_raw (b);
      printf (" ");
      mpfr_print_raw (c);
    }
#endif
  if (usesp || MPFR_ARE_SINGULAR(b,c) || MPFR_SIGN(b) != MPFR_SIGN(c))
    res = mpfr_add (a, b, c, rnd_mode);
  else
    {
      if (MPFR_GET_EXP(b) < MPFR_GET_EXP(c))
        res = mpfr_add1(a, c, b, rnd_mode);
      else
        res = mpfr_add1(a, b, c, rnd_mode);
    }
#ifdef CHECK_EXTERNAL
  if (ok)
    {
      printf (" ");
      mpfr_print_raw (a);
      printf ("\n");
    }
#endif
  return res;
}

/* checks that xs+ys gives the expected result zs */
static void
check (const char *xs, const char *ys, mpfr_rnd_t rnd_mode,
        unsigned int px, unsigned int py, unsigned int pz, const char *zs)
{
  mpfr_t xx,yy,zz;

  mpfr_init2 (xx, px);
  mpfr_init2 (yy, py);
  mpfr_init2 (zz, pz);

  mpfr_set_str1 (xx, xs);
  mpfr_set_str1 (yy, ys);
  test_add (zz, xx, yy, rnd_mode);
  if (mpfr_cmp_str1 (zz, zs) )
    {
      printf ("expected sum is %s, got ", zs);
      mpfr_out_str(stdout, 10, 0, zz, MPFR_RNDN);
      printf ("mpfr_add failed for x=%s y=%s with rnd_mode=%s\n",
              xs, ys, mpfr_print_rnd_mode (rnd_mode));
      exit (1);
    }
  mpfr_clears (xx, yy, zz, (mpfr_ptr) 0);
}

static void
check2b (const char *xs, int px,
         const char *ys, int py,
         const char *rs, int pz,
         mpfr_rnd_t rnd_mode)
{
  mpfr_t xx, yy, zz;

  mpfr_init2 (xx,px);
  mpfr_init2 (yy,py);
  mpfr_init2 (zz,pz);
  mpfr_set_str_binary (xx, xs);
  mpfr_set_str_binary (yy, ys);
  test_add (zz, xx, yy, rnd_mode);
  if (mpfr_cmp_str (zz, rs, 2, MPFR_RNDN))
    {
      printf ("(2) x=%s,%d y=%s,%d pz=%d,rnd=%s\n",
              xs, px, ys, py, pz, mpfr_print_rnd_mode (rnd_mode));
      printf ("got        "); mpfr_print_binary(zz); puts ("");
      mpfr_set_str(zz, rs, 2, MPFR_RNDN);
      printf ("instead of "); mpfr_print_binary(zz); puts ("");
      exit (1);
    }
  mpfr_clear(xx); mpfr_clear(yy); mpfr_clear(zz);
}

static void
check64 (void)
{
  mpfr_t x, t, u;

  mpfr_init (x);
  mpfr_init (t);
  mpfr_init (u);

  mpfr_set_prec (x, 29);
  mpfr_set_str_binary (x, "1.1101001000101111011010010110e-3");
  mpfr_set_prec (t, 58);
  mpfr_set_str_binary (t, "0.11100010011111001001100110010111110110011000000100101E-1");
  mpfr_set_prec (u, 29);
  test_add (u, x, t, MPFR_RNDD);
  mpfr_set_str_binary (t, "1.0101011100001000011100111110e-1");
  if (mpfr_cmp (u, t))
    {
      printf ("mpfr_add(u, x, t) failed for prec(x)=29, prec(t)=58\n");
      printf ("expected "); mpfr_out_str (stdout, 2, 29, t, MPFR_RNDN);
      puts ("");
      printf ("got      "); mpfr_out_str (stdout, 2, 29, u, MPFR_RNDN);
      puts ("");
      exit(1);
    }

  mpfr_set_prec (x, 4);
  mpfr_set_str_binary (x, "-1.0E-2");
  mpfr_set_prec (t, 2);
  mpfr_set_str_binary (t, "-1.1e-2");
  mpfr_set_prec (u, 2);
  test_add (u, x, t, MPFR_RNDN);
  if (MPFR_MANT(u)[0] << 2)
    {
      printf ("result not normalized for prec=2\n");
      mpfr_print_binary (u); puts ("");
      exit (1);
    }
  mpfr_set_str_binary (t, "-1.0e-1");
  if (mpfr_cmp (u, t))
    {
      printf ("mpfr_add(u, x, t) failed for prec(x)=4, prec(t)=2\n");
      printf ("expected -1.0e-1\n");
      printf ("got      "); mpfr_out_str (stdout, 2, 4, u, MPFR_RNDN);
      puts ("");
      exit (1);
    }

  mpfr_set_prec (x, 8);
  mpfr_set_str_binary (x, "-0.10011010"); /* -77/128 */
  mpfr_set_prec (t, 4);
  mpfr_set_str_binary (t, "-1.110e-5"); /* -7/128 */
  mpfr_set_prec (u, 4);
  test_add (u, x, t, MPFR_RNDN); /* should give -5/8 */
  mpfr_set_str_binary (t, "-1.010e-1");
  if (mpfr_cmp (u, t)) {
    printf ("mpfr_add(u, x, t) failed for prec(x)=8, prec(t)=4\n");
    printf ("expected -1.010e-1\n");
    printf ("got      "); mpfr_out_str (stdout, 2, 4, u, MPFR_RNDN);
    puts ("");
    exit (1);
  }

  mpfr_set_prec (x, 112); mpfr_set_prec (t, 98); mpfr_set_prec (u, 54);
  mpfr_set_str_binary (x, "-0.11111100100000000011000011100000101101010001000111E-401");
  mpfr_set_str_binary (t, "0.10110000100100000101101100011111111011101000111000101E-464");
  test_add (u, x, t, MPFR_RNDN);
  if (mpfr_cmp (u, x))
    {
      printf ("mpfr_add(u, x, t) failed for prec(x)=112, prec(t)=98\n");
      exit (1);
    }

  mpfr_set_prec (x, 92); mpfr_set_prec (t, 86); mpfr_set_prec (u, 53);
  mpfr_set_str (x, "-5.03525136761487735093e-74", 10, MPFR_RNDN);
  mpfr_set_str (t, "8.51539046314262304109e-91", 10, MPFR_RNDN);
  test_add (u, x, t, MPFR_RNDN);
  if (mpfr_cmp_str1 (u, "-5.0352513676148773509283672e-74") )
    {
      printf ("mpfr_add(u, x, t) failed for prec(x)=92, prec(t)=86\n");
      exit (1);
    }

  mpfr_set_prec(x, 53); mpfr_set_prec(t, 76); mpfr_set_prec(u, 76);
  mpfr_set_str_binary(x, "-0.10010010001001011011110000000000001010011011011110001E-32");
  mpfr_set_str_binary(t, "-0.1011000101110010000101111111011111010001110011110111100110101011110010011111");
  mpfr_sub(u, x, t, MPFR_RNDU);
  mpfr_set_str_binary(t, "0.1011000101110010000101111111011100111111101010011011110110101011101000000100");
  if (mpfr_cmp(u,t))
    {
      printf ("expect "); mpfr_print_binary(t); puts ("");
      printf ("mpfr_add failed for precisions 53-76\n");
      exit (1);
    }
  mpfr_set_prec(x, 53); mpfr_set_prec(t, 108); mpfr_set_prec(u, 108);
  mpfr_set_str_binary(x, "-0.10010010001001011011110000000000001010011011011110001E-32");
  mpfr_set_str_binary(t, "-0.101100010111001000010111111101111101000111001111011110011010101111001001111000111011001110011000000000111111");
  mpfr_sub(u, x, t, MPFR_RNDU);
  mpfr_set_str_binary(t, "0.101100010111001000010111111101110011111110101001101111011010101110100000001011000010101110011000000000111111");
  if (mpfr_cmp(u,t))
    {
      printf ("expect "); mpfr_print_binary(t); puts ("");
      printf ("mpfr_add failed for precisions 53-108\n");
      exit (1);
    }
  mpfr_set_prec(x, 97); mpfr_set_prec(t, 97); mpfr_set_prec(u, 97);
  mpfr_set_str_binary(x, "0.1111101100001000000001011000110111101000001011111000100001000101010100011111110010000000000000000E-39");
  mpfr_set_ui(t, 1, MPFR_RNDN);
  test_add (u, x, t, MPFR_RNDN);
  mpfr_set_str_binary(x, "0.1000000000000000000000000000000000000000111110110000100000000101100011011110100000101111100010001E1");
  if (mpfr_cmp(u,x))
    {
      printf ("mpfr_add failed for precision 97\n");
      exit (1);
    }
  mpfr_set_prec(x, 128); mpfr_set_prec(t, 128); mpfr_set_prec(u, 128);
  mpfr_set_str_binary(x, "0.10101011111001001010111011001000101100111101000000111111111011010100001100011101010001010111111101111010100110111111100101100010E-4");
  mpfr_set(t, x, MPFR_RNDN);
  mpfr_sub(u, x, t, MPFR_RNDN);
  mpfr_set_prec(x, 96); mpfr_set_prec(t, 96); mpfr_set_prec(u, 96);
  mpfr_set_str_binary(x, "0.111000000001110100111100110101101001001010010011010011100111100011010100011001010011011011000010E-4");
  mpfr_set(t, x, MPFR_RNDN);
  mpfr_sub(u, x, t, MPFR_RNDN);
  mpfr_set_prec(x, 85); mpfr_set_prec(t, 85); mpfr_set_prec(u, 85);
  mpfr_set_str_binary(x, "0.1111101110100110110110100010101011101001100010100011110110110010010011101100101111100E-4");
  mpfr_set_str_binary(t, "0.1111101110100110110110100010101001001000011000111000011101100101110100001110101010110E-4");
  mpfr_sub(u, x, t, MPFR_RNDU);
  mpfr_sub(x, x, t, MPFR_RNDU);
  if (mpfr_cmp(x, u) != 0)
    {
      printf ("Error in mpfr_sub: u=x-t and x=x-t give different results\n");
      exit (1);
    }
  if ((MPFR_MANT(u)[(MPFR_PREC(u)-1)/mp_bits_per_limb] &
       ((mp_limb_t)1<<(mp_bits_per_limb-1)))==0)
    {
      printf ("Error in mpfr_sub: result is not msb-normalized (1)\n");
      exit (1);
    }
  mpfr_set_prec(x, 65); mpfr_set_prec(t, 65); mpfr_set_prec(u, 65);
  mpfr_set_str_binary(x, "0.10011010101000110101010000000011001001001110001011101011111011101E623");
  mpfr_set_str_binary(t, "0.10011010101000110101010000000011001001001110001011101011111011100E623");
  mpfr_sub(u, x, t, MPFR_RNDU);
  if (mpfr_cmp_ui_2exp(u, 1, 558))
    { /* 2^558 */
      printf ("Error (1) in mpfr_sub\n");
      exit (1);
    }

  mpfr_set_prec(x, 64); mpfr_set_prec(t, 64); mpfr_set_prec(u, 64);
  mpfr_set_str_binary(x, "0.1000011110101111011110111111000011101011101111101101101100000100E-220");
  mpfr_set_str_binary(t, "0.1000011110101111011110111111000011101011101111101101010011111101E-220");
  test_add (u, x, t, MPFR_RNDU);
  if ((MPFR_MANT(u)[0] & 1) != 1)
    {
      printf ("error in mpfr_add with rnd_mode=MPFR_RNDU\n");
      printf ("b=  "); mpfr_print_binary(x); puts ("");
      printf ("c=  "); mpfr_print_binary(t); puts ("");
      printf ("b+c="); mpfr_print_binary(u); puts ("");
      exit (1);
    }

  /* bug found by Norbert Mueller, 14 Sep 2000 */
  mpfr_set_prec(x, 56); mpfr_set_prec(t, 83); mpfr_set_prec(u, 10);
  mpfr_set_str_binary(x, "0.10001001011011001111101100110100000101111010010111010111E-7");
  mpfr_set_str_binary(t, "0.10001001011011001111101100110100000101111010010111010111000000000111110110110000100E-7");
  mpfr_sub(u, x, t, MPFR_RNDU);

  /* array bound write found by Norbert Mueller, 26 Sep 2000 */
  mpfr_set_prec(x, 109); mpfr_set_prec(t, 153); mpfr_set_prec(u, 95);
  mpfr_set_str_binary(x,"0.1001010000101011101100111000110001111111111111111111111111111111111111111111111111111111111111100000000000000E33");
  mpfr_set_str_binary(t,"-0.100101000010101110110011100011000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011100101101000000100100001100110111E33");
  test_add (u, x, t, MPFR_RNDN);

  /* array bound writes found by Norbert Mueller, 27 Sep 2000 */
  mpfr_set_prec(x, 106); mpfr_set_prec(t, 53); mpfr_set_prec(u, 23);
  mpfr_set_str_binary(x, "-0.1000011110101111111001010001000100001011000000000000000000000000000000000000000000000000000000000000000000E-59");
  mpfr_set_str_binary(t, "-0.10000111101011111110010100010001101100011100110100000E-59");
  mpfr_sub(u, x, t, MPFR_RNDN);
  mpfr_set_prec(x, 177); mpfr_set_prec(t, 217); mpfr_set_prec(u, 160);
  mpfr_set_str_binary(x, "-0.111010001011010000111001001010010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000E35");
  mpfr_set_str_binary(t, "0.1110100010110100001110010010100100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111011010011100001111001E35");
  test_add (u, x, t, MPFR_RNDN);
  mpfr_set_prec(x, 214); mpfr_set_prec(t, 278); mpfr_set_prec(u, 207);
  mpfr_set_str_binary(x, "0.1000100110100110101101101101000000010000100111000001001110001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000E66");
  mpfr_set_str_binary(t, "-0.10001001101001101011011011010000000100001001110000010011100010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001111011111001001100011E66");
  test_add (u, x, t, MPFR_RNDN);
  mpfr_set_prec(x, 32); mpfr_set_prec(t, 247); mpfr_set_prec(u, 223);
  mpfr_set_str_binary(x, "0.10000000000000000000000000000000E1");
  mpfr_set_str_binary(t, "0.1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100000110001110100000100011110000101110110011101110100110110111111011010111100100000000000000000000000000E0");
  mpfr_sub(u, x, t, MPFR_RNDN);
  if ((MPFR_MANT(u)[(MPFR_PREC(u)-1)/mp_bits_per_limb] &
       ((mp_limb_t)1<<(mp_bits_per_limb-1)))==0)
    {
      printf ("Error in mpfr_sub: result is not msb-normalized (2)\n");
      exit (1);
    }

  /* bug found by Nathalie Revol, 21 March 2001 */
  mpfr_set_prec (x, 65);
  mpfr_set_prec (t, 65);
  mpfr_set_prec (u, 65);
  mpfr_set_str_binary (x, "0.11100100101101001100111011111111110001101001000011101001001010010E-35");
  mpfr_set_str_binary (t, "0.10000000000000000000000000000000000001110010010110100110011110000E1");
  mpfr_sub (u, t, x, MPFR_RNDU);
  if ((MPFR_MANT(u)[(MPFR_PREC(u)-1)/mp_bits_per_limb] &
       ((mp_limb_t)1<<(mp_bits_per_limb-1)))==0)
    {
      printf ("Error in mpfr_sub: result is not msb-normalized (3)\n");
      exit (1);
    }

  /* bug found by Fabrice Rouillier, 27 Mar 2001 */
  mpfr_set_prec (x, 107);
  mpfr_set_prec (t, 107);
  mpfr_set_prec (u, 107);
  mpfr_set_str_binary (x, "0.10111001001111010010001000000010111111011011011101000001001000101000000000000000000000000000000000000000000E315");
  mpfr_set_str_binary (t, "0.10000000000000000000000000000000000101110100100101110110000001100101011111001000011101111100100100111011000E350");
  mpfr_sub (u, x, t, MPFR_RNDU);
  if ((MPFR_MANT(u)[(MPFR_PREC(u)-1)/mp_bits_per_limb] &
       ((mp_limb_t)1<<(mp_bits_per_limb-1)))==0)
    {
      printf ("Error in mpfr_sub: result is not msb-normalized (4)\n");
      exit (1);
    }

  /* checks that NaN flag is correctly reset */
  mpfr_set_ui (t, 1, MPFR_RNDN);
  mpfr_set_ui (u, 1, MPFR_RNDN);
  mpfr_set_nan (x);
  test_add (x, t, u, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 2))
    {
      printf ("Error in mpfr_add: 1+1 gives ");
      mpfr_out_str(stdout, 10, 0, x, MPFR_RNDN);
      exit (1);
    }

  mpfr_clear(x); mpfr_clear(t); mpfr_clear(u);
}

/* check case when c does not overlap with a, but both b and c count
   for rounding */
static void
check_case_1b (void)
{
  mpfr_t a, b, c;
  unsigned int prec_a, prec_b, prec_c, dif;

  mpfr_init (a);
  mpfr_init (b);
  mpfr_init (c);

    {
      prec_a = MPFR_PREC_MIN + (randlimb () % 63);
      mpfr_set_prec (a, prec_a);
      for (prec_b = prec_a + 2; prec_b <= 64; prec_b++)
        {
          dif = prec_b - prec_a;
          mpfr_set_prec (b, prec_b);
          /* b = 1 - 2^(-prec_a) + 2^(-prec_b) */
          mpfr_set_ui (b, 1, MPFR_RNDN);
          mpfr_div_2exp (b, b, dif, MPFR_RNDN);
          mpfr_sub_ui (b, b, 1, MPFR_RNDN);
          mpfr_div_2exp (b, b, prec_a, MPFR_RNDN);
          mpfr_add_ui (b, b, 1, MPFR_RNDN);
          for (prec_c = dif; prec_c <= 64; prec_c++)
            {
              /* c = 2^(-prec_a) - 2^(-prec_b) */
              mpfr_set_prec (c, prec_c);
              mpfr_set_si (c, -1, MPFR_RNDN);
              mpfr_div_2exp (c, c, dif, MPFR_RNDN);
              mpfr_add_ui (c, c, 1, MPFR_RNDN);
              mpfr_div_2exp (c, c, prec_a, MPFR_RNDN);
              test_add (a, b, c, MPFR_RNDN);
              if (mpfr_cmp_ui (a, 1) != 0)
                {
                  printf ("case (1b) failed for prec_a=%u, prec_b=%u,"
                          " prec_c=%u\n", prec_a, prec_b, prec_c);
                  printf ("b="); mpfr_print_binary(b); puts ("");
                  printf ("c="); mpfr_print_binary(c); puts ("");
                  printf ("a="); mpfr_print_binary(a); puts ("");
                  exit (1);
                }
            }
        }
    }

  mpfr_clear (a);
  mpfr_clear (b);
  mpfr_clear (c);
}

/* check case when c overlaps with a */
static void
check_case_2 (void)
{
  mpfr_t a, b, c, d;

  mpfr_init2 (a, 300);
  mpfr_init2 (b, 800);
  mpfr_init2 (c, 500);
  mpfr_init2 (d, 800);

  mpfr_set_str_binary(a, "1E110");  /* a = 2^110 */
  mpfr_set_str_binary(b, "1E900");  /* b = 2^900 */
  mpfr_set_str_binary(c, "1E500");  /* c = 2^500 */
  test_add (c, c, a, MPFR_RNDZ);   /* c = 2^500 + 2^110 */
  mpfr_sub (d, b, c, MPFR_RNDZ);   /* d = 2^900 - 2^500 - 2^110 */
  test_add (b, b, c, MPFR_RNDZ);   /* b = 2^900 + 2^500 + 2^110 */
  test_add (a, b, d, MPFR_RNDZ);   /* a = 2^901 */
  if (mpfr_cmp_ui_2exp (a, 1, 901))
    {
      printf ("b + d fails for b=2^900+2^500+2^110, d=2^900-2^500-2^110\n");
      printf ("expected 1.0e901, got ");
      mpfr_out_str (stdout, 2, 0, a, MPFR_RNDN);
      printf ("\n");
      exit (1);
    }

  mpfr_clear (a);
  mpfr_clear (b);
  mpfr_clear (c);
  mpfr_clear (d);
}

/* checks when source and destination are equal */
static void
check_same (void)
{
  mpfr_t x;

  mpfr_init(x); mpfr_set_ui(x, 1, MPFR_RNDZ);
  test_add (x, x, x, MPFR_RNDZ);
  if (mpfr_cmp_ui (x, 2))
    {
      printf ("Error when all 3 operands are equal\n");
      exit (1);
    }
  mpfr_clear(x);
}

#define check53(x, y, r, z) check(x, y, r, 53, 53, 53, z)

#define MAX_PREC 256

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
  mpfr_set_str_binary (x, "0.1E-4");
  mpfr_set_prec (u, 33);
  mpfr_set_str_binary (u, "0.101110100101101100000000111100000E-1");
  mpfr_set_prec (y, 31);
  if ((inexact = test_add (y, x, u, MPFR_RNDN)))
    {
      printf ("Wrong inexact flag (2): expected 0, got %d\n", inexact);
      exit (1);
    }

  mpfr_set_prec (x, 2);
  mpfr_set_str_binary (x, "0.1E-4");
  mpfr_set_prec (u, 33);
  mpfr_set_str_binary (u, "0.101110100101101100000000111100000E-1");
  mpfr_set_prec (y, 28);
  if ((inexact = test_add (y, x, u, MPFR_RNDN)))
    {
      printf ("Wrong inexact flag (1): expected 0, got %d\n", inexact);
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
              py = MPFR_PREC_MIN + (randlimb () % (MAX_PREC - 1));
              mpfr_set_prec (y, py);
              pz =  (mpfr_cmpabs (x, u) >= 0) ? MPFR_EXP(x) - MPFR_EXP(u)
                : MPFR_EXP(u) - MPFR_EXP(x);
              /* x + u is exactly representable with precision
                 abs(EXP(x)-EXP(u)) + max(prec(x), prec(u)) + 1 */
              pz = pz + MAX(MPFR_PREC(x), MPFR_PREC(u)) + 1;
              mpfr_set_prec (z, pz);
              rnd = RND_RAND ();
              if (test_add (z, x, u, rnd))
                {
                  printf ("z <- x + u should be exact\n");
                  printf ("x="); mpfr_print_binary (x); puts ("");
                  printf ("u="); mpfr_print_binary (u); puts ("");
                  printf ("z="); mpfr_print_binary (z); puts ("");
                  exit (1);
                }
                {
                  rnd = RND_RAND ();
                  inexact = test_add (y, x, u, rnd);
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
                      printf ("x+u="); mpfr_print_binary (z); puts ("");
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

static void
check_nans (void)
{
  mpfr_t  s, x, y;

  mpfr_init2 (x, 8L);
  mpfr_init2 (y, 8L);
  mpfr_init2 (s, 8L);

  /* +inf + -inf == nan */
  mpfr_set_inf (x, 1);
  mpfr_set_inf (y, -1);
  test_add (s, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (s));

  /* +inf + 1 == +inf */
  mpfr_set_inf (x, 1);
  mpfr_set_ui (y, 1L, MPFR_RNDN);
  test_add (s, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (s));
  MPFR_ASSERTN (mpfr_sgn (s) > 0);

  /* -inf + 1 == -inf */
  mpfr_set_inf (x, -1);
  mpfr_set_ui (y, 1L, MPFR_RNDN);
  test_add (s, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (s));
  MPFR_ASSERTN (mpfr_sgn (s) < 0);

  /* 1 + +inf == +inf */
  mpfr_set_ui (x, 1L, MPFR_RNDN);
  mpfr_set_inf (y, 1);
  test_add (s, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (s));
  MPFR_ASSERTN (mpfr_sgn (s) > 0);

  /* 1 + -inf == -inf */
  mpfr_set_ui (x, 1L, MPFR_RNDN);
  mpfr_set_inf (y, -1);
  test_add (s, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (s));
  MPFR_ASSERTN (mpfr_sgn (s) < 0);

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (s);
}

static void
check_alloc (void)
{
  mpfr_t a;

  mpfr_init2 (a, 10000);
  mpfr_set_prec (a, 53);
  mpfr_set_ui (a, 15236, MPFR_RNDN);
  test_add (a, a, a, MPFR_RNDN);
  mpfr_mul (a, a, a, MPFR_RNDN);
  mpfr_div (a, a, a, MPFR_RNDN);
  mpfr_sub (a, a, a, MPFR_RNDN);
  mpfr_clear (a);
}

static void
check_overflow (void)
{
  mpfr_t a, b, c;
  mpfr_prec_t prec_a;
  int r;

  mpfr_init2 (a, 256);
  mpfr_init2 (b, 256);
  mpfr_init2 (c, 256);

  mpfr_set_ui (b, 1, MPFR_RNDN);
  mpfr_setmax (b, mpfr_get_emax ());
  mpfr_set_ui (c, 1, MPFR_RNDN);
  mpfr_set_exp (c, mpfr_get_emax () - 192);
  RND_LOOP(r)
    for (prec_a = 128; prec_a < 512; prec_a += 64)
      {
        mpfr_set_prec (a, prec_a);
        mpfr_clear_overflow ();
        test_add (a, b, c, (mpfr_rnd_t) r);
        if (!mpfr_overflow_p ())
          {
            printf ("No overflow in check_overflow\n");
            exit (1);
          }
      }

  mpfr_set_exp (c, mpfr_get_emax () - 512);
  mpfr_set_prec (a, 256);
  mpfr_clear_overflow ();
  test_add (a, b, c, MPFR_RNDU);
  if (!mpfr_overflow_p ())
    {
      printf ("No overflow in check_overflow\n");
      exit (1);
    }

  mpfr_clear (a);
  mpfr_clear (b);
  mpfr_clear (c);
}

static void
check_1111 (void)
{
  mpfr_t one;
  long n;

  mpfr_init2 (one, MPFR_PREC_MIN);
  mpfr_set_ui (one, 1, MPFR_RNDN);
  for (n = 0; n < N; n++)
    {
      mpfr_prec_t prec_a, prec_b, prec_c;
      mpfr_exp_t tb=0, tc, diff;
      mpfr_t a, b, c, s;
      int m = 512;
      int sb, sc;
      int inex_a, inex_s;
      mpfr_rnd_t rnd_mode;

      prec_a = MPFR_PREC_MIN + (randlimb () % m);
      prec_b = MPFR_PREC_MIN + (randlimb () % m);
      prec_c = MPFR_PREC_MIN + (randlimb () % m);
      mpfr_init2 (a, prec_a);
      mpfr_init2 (b, prec_b);
      mpfr_init2 (c, prec_c);
      sb = randlimb () % 3;
      if (sb != 0)
        {
          tb = 1 + (randlimb () % (prec_b - (sb != 2)));
          mpfr_div_2ui (b, one, tb, MPFR_RNDN);
          if (sb == 2)
            mpfr_neg (b, b, MPFR_RNDN);
          test_add (b, b, one, MPFR_RNDN);
        }
      else
        mpfr_set (b, one, MPFR_RNDN);
      tc = 1 + (randlimb () % (prec_c - 1));
      mpfr_div_2ui (c, one, tc, MPFR_RNDN);
      sc = randlimb () % 2;
      if (sc)
        mpfr_neg (c, c, MPFR_RNDN);
      test_add (c, c, one, MPFR_RNDN);
      diff = (randlimb () % (2*m)) - m;
      mpfr_mul_2si (c, c, diff, MPFR_RNDN);
      rnd_mode = RND_RAND ();
      inex_a = test_add (a, b, c, rnd_mode);
      mpfr_init2 (s, MPFR_PREC_MIN + 2*m);
      inex_s = test_add (s, b, c, MPFR_RNDN); /* exact */
      if (inex_s)
        {
          printf ("check_1111: result should have been exact.\n");
          exit (1);
        }
      inex_s = mpfr_prec_round (s, prec_a, rnd_mode);
      if ((inex_a < 0 && inex_s >= 0) ||
          (inex_a == 0 && inex_s != 0) ||
          (inex_a > 0 && inex_s <= 0) ||
          !mpfr_equal_p (a, s))
        {
          printf ("check_1111: results are different.\n");
          printf ("prec_a = %d, prec_b = %d, prec_c = %d\n",
                  (int) prec_a, (int) prec_b, (int) prec_c);
          printf ("tb = %d, tc = %d, diff = %d, rnd = %s\n",
                  (int) tb, (int) tc, (int) diff,
                  mpfr_print_rnd_mode (rnd_mode));
          printf ("sb = %d, sc = %d\n", sb, sc);
          printf ("a = "); mpfr_print_binary (a); puts ("");
          printf ("s = "); mpfr_print_binary (s); puts ("");
          printf ("inex_a = %d, inex_s = %d\n", inex_a, inex_s);
          exit (1);
        }
      mpfr_clear (a);
      mpfr_clear (b);
      mpfr_clear (c);
      mpfr_clear (s);
    }
  mpfr_clear (one);
}

static void
check_1minuseps (void)
{
  static mpfr_prec_t prec_a[] = {
    MPFR_PREC_MIN, 30, 31, 32, 33, 62, 63, 64, 65, 126, 127, 128, 129
  };
  static int supp_b[] = {
    0, 1, 2, 3, 4, 29, 30, 31, 32, 33, 34, 35, 61, 62, 63, 64, 65, 66, 67
  };
  mpfr_t a, b, c;
  unsigned int ia, ib, ic;

  mpfr_init2 (c, MPFR_PREC_MIN);

  for (ia = 0; ia < numberof (prec_a); ia++)
    for (ib = 0; ib < numberof(supp_b); ib++)
      {
        mpfr_prec_t prec_b;
        int rnd_mode;

        prec_b = prec_a[ia] + supp_b[ib];

        mpfr_init2 (a, prec_a[ia]);
        mpfr_init2 (b, prec_b);

        mpfr_set_ui (c, 1, MPFR_RNDN);
        mpfr_div_ui (b, c, prec_a[ia], MPFR_RNDN);
        mpfr_sub (b, c, b, MPFR_RNDN);  /* b = 1 - 2^(-prec_a) */

        for (ic = 0; ic < numberof(supp_b); ic++)
          for (rnd_mode = 0; rnd_mode < MPFR_RND_MAX; rnd_mode++)
            {
              mpfr_t s;
              int inex_a, inex_s;

              mpfr_set_ui (c, 1, MPFR_RNDN);
              mpfr_div_ui (c, c, prec_a[ia] + supp_b[ic], MPFR_RNDN);
              inex_a = test_add (a, b, c, (mpfr_rnd_t) rnd_mode);
              mpfr_init2 (s, 256);
              inex_s = test_add (s, b, c, MPFR_RNDN); /* exact */
              if (inex_s)
                {
                  printf ("check_1minuseps: result should have been exact "
                          "(ia = %u, ib = %u, ic = %u)\n", ia, ib, ic);
                  exit (1);
                }
              inex_s = mpfr_prec_round (s, prec_a[ia], (mpfr_rnd_t) rnd_mode);
              if ((inex_a < 0 && inex_s >= 0) ||
                  (inex_a == 0 && inex_s != 0) ||
                  (inex_a > 0 && inex_s <= 0) ||
                  !mpfr_equal_p (a, s))
                {
                  printf ("check_1minuseps: results are different.\n");
                  printf ("ia = %u, ib = %u, ic = %u\n", ia, ib, ic);
                  exit (1);
                }
              mpfr_clear (s);
            }

        mpfr_clear (a);
        mpfr_clear (b);
      }

  mpfr_clear (c);
}

/* Test case bk == 0 in add1.c (b has entirely been read and
   c hasn't been taken into account). */
static void
coverage_bk_eq_0 (void)
{
  mpfr_t a, b, c;
  int inex;

  mpfr_init2 (a, GMP_NUMB_BITS);
  mpfr_init2 (b, 2 * GMP_NUMB_BITS);
  mpfr_init2 (c, GMP_NUMB_BITS);

  mpfr_set_ui_2exp (b, 1, 2 * GMP_NUMB_BITS, MPFR_RNDN);
  mpfr_sub_ui (b, b, 1, MPFR_RNDN);
  /* b = 111...111 (in base 2) where the 1's fit 2 whole limbs */

  mpfr_set_ui_2exp (c, 1, -1, MPFR_RNDN);  /* c = 1/2 */

  inex = mpfr_add (a, b, c, MPFR_RNDU);
  mpfr_set_ui_2exp (c, 1, 2 * GMP_NUMB_BITS, MPFR_RNDN);
  if (! mpfr_equal_p (a, c))
    {
      printf ("Error in coverage_bk_eq_0\n");
      printf ("Expected ");
      mpfr_dump (c);
      printf ("Got      ");
      mpfr_dump (a);
      exit (1);
    }
  MPFR_ASSERTN (inex > 0);

  mpfr_clear (a);
  mpfr_clear (b);
  mpfr_clear (c);
}

static void
tests (void)
{
  check_alloc ();
  check_nans ();
  check_inexact ();
  check_case_1b ();
  check_case_2 ();
  check64();
  coverage_bk_eq_0 ();

  check("293607738.0", "1.9967571564050541e-5", MPFR_RNDU, 64, 53, 53,
        "2.9360773800002003e8");
  check("880524.0", "-2.0769715792901673e-5", MPFR_RNDN, 64, 53, 53,
        "8.8052399997923023e5");
  check("1196426492.0", "-1.4218093058435347e-3", MPFR_RNDN, 64, 53, 53,
        "1.1964264919985781e9");
  check("982013018.0", "-8.941829477291838e-7", MPFR_RNDN, 64, 53, 53,
        "9.8201301799999905e8");
  check("1092583421.0", "1.0880649218158844e9", MPFR_RNDN, 64, 53, 53,
        "2.1806483428158846e9");
  check("1.8476886419022969e-6", "961494401.0", MPFR_RNDN, 53, 64, 53,
        "9.6149440100000179e8");
  check("-2.3222118418069868e5", "1229318102.0", MPFR_RNDN, 53, 64, 53,
        "1.2290858808158193e9");
  check("-3.0399171300395734e-6", "874924868.0", MPFR_RNDN, 53, 64, 53,
        "8.749248679999969e8");
  check("9.064246624706179e1", "663787413.0", MPFR_RNDN, 53, 64, 53,
        "6.6378750364246619e8");
  check("-1.0954322421551264e2", "281806592.0", MPFR_RNDD, 53, 64, 53,
        "2.8180648245677572e8");
  check("5.9836930386056659e-8", "1016217213.0", MPFR_RNDN, 53, 64, 53,
        "1.0162172130000001e9");
  check("-1.2772161928500301e-7", "1237734238.0", MPFR_RNDN, 53, 64, 53,
        "1.2377342379999998e9");
  check("-4.567291988483277e8", "1262857194.0", MPFR_RNDN, 53, 64, 53,
        "8.0612799515167236e8");
  check("4.7719471752925262e7", "196089880.0", MPFR_RNDN, 53, 53, 53,
        "2.4380935175292528e8");
  check("4.7719471752925262e7", "196089880.0", MPFR_RNDN, 53, 64, 53,
        "2.4380935175292528e8");
  check("-1.716113812768534e-140", "1271212614.0", MPFR_RNDZ, 53, 64, 53,
        "1.2712126139999998e9");
  check("-1.2927455200185474e-50", "1675676122.0", MPFR_RNDD, 53, 64, 53,
        "1.6756761219999998e9");

  check53("1.22191250737771397120e+20", "948002822.0", MPFR_RNDN,
          "122191250738719408128.0");
  check53("9966027674114492.0", "1780341389094537.0", MPFR_RNDN,
          "11746369063209028.0");
  check53("2.99280481918991653800e+272", "5.34637717585790933424e+271",
          MPFR_RNDN, "3.5274425367757071711e272");
  check_same();
  check53("6.14384195492641560499e-02", "-6.14384195401037683237e-02",
          MPFR_RNDU, "9.1603877261370314499e-12");
  check53("1.16809465359248765399e+196", "7.92883212101990665259e+196",
          MPFR_RNDU, "9.0969267746123943065e196");
  check53("3.14553393112021279444e-67", "3.14553401015952024126e-67", MPFR_RNDU,
          "6.2910679412797336946e-67");

  check53("5.43885304644369509058e+185","-1.87427265794105342763e-57",MPFR_RNDN,
          "5.4388530464436950905e185");
  check53("5.43885304644369509058e+185","-1.87427265794105342763e-57",MPFR_RNDZ,
          "5.4388530464436944867e185");
  check53("5.43885304644369509058e+185","-1.87427265794105342763e-57",MPFR_RNDU,
          "5.4388530464436950905e185");
  check53("5.43885304644369509058e+185","-1.87427265794105342763e-57",MPFR_RNDD,
          "5.4388530464436944867e185");

  check2b("1.001010101110011000000010100101110010111001010000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e358",187,
          "-1.11100111001101100010001111111110101101110001000000000000000000000000000000000000000000e160",87,
          "1.001010101110011000000010100101110010111001010000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111e358",178,
          MPFR_RNDD);
  check2b("-1.111100100011100111010101010101001010100100000111001000000000000000000e481",70,
          "1.1111000110100011110101111110110010010000000110101000000000000000e481",65,
          "-1.001010111111101011010000001100011101100101000000000000000000e472",61,
          MPFR_RNDD);
  check2b("1.0100010111010000100101000000111110011100011001011010000000000000000000000000000000e516",83,
          "-1.1001111000100001011100000001001100110011110010111111000000e541",59,
          "-1.1001111000100001011011110111000001001011100000011110100000110001110011010011000000000000000000000000000000000000000000000000e541",125,
          MPFR_RNDZ);
  check2b("-1.0010111100000100110001011011010000000011000111101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e261",155,
          "-1.00111110100011e239",15,
          "-1.00101111000001001100101010101110001100110001111010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e261",159,
          MPFR_RNDD);
  check2b("-1.110111000011111011000000001001111101101001010100111000000000000000000000000e880",76,
          "-1.1010010e-634",8,
          "-1.11011100001111101100000000100111110110100101010011100000000000000000000000e880",75,
          MPFR_RNDZ);
  check2b("1.00100100110110101001010010101111000001011100100101010000000000000000000000000000e-530",81,
          "-1.101101111100000111000011001010110011001011101001110100000e-908",58,
          "1.00100100110110101001010010101111000001011100100101010e-530",54,
          MPFR_RNDN);
  check2b("1.0101100010010111101000000001000010010010011000111011000000000000000000000000000000000000000000000000000000000000000000e374",119,
          "1.11100101100101e358",15,
          "1.01011000100110011000010110100100100100100110001110110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e374",150,
          MPFR_RNDZ);
  check2b("-1.10011001000010100000010100100110110010011111101111000000000000000000000000000000000000000000000000000000000000000000e-172",117,
          "1.111011100000101010110000100100110100100001001000011100000000e-173",61,
          "-1.0100010000001001010110011011101001001011101011110001000000000000000e-173",68,
          MPFR_RNDZ);
  check2b("-1.011110000111101011100001100110100011100101000011011000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-189",175,
          "1.1e631",2,
          "1.011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111e631",115,
          MPFR_RNDZ);
  check2b("-1.101011101001011101100011001000001100010100001101011000000000000000000000000000000000000000000e-449",94,
          "-1.01111101010111000011000110011101000111001100110111100000000000000e-429",66,
          "-1.01111101010111000100110010000110100100101111111111101100010100001101011000000000000000000000000000000000000000e-429",111,
          MPFR_RNDU);
  check2b("-1.101011101001011101100011001000001100010100001101011000000000000000000000000000000000000000000e-449",94,
          "-1.01111101010111000011000110011101000111001100110111100000000000000e-429",66,
          "-1.01111101010111000100110010000110100100101111111111101100010100001101011000000000000000000000000000000000000000e-429",111,
          MPFR_RNDD);
  check2b("-1.1001000011101000110000111110010100100101110101111100000000000000000000000000000000000000000000000000000000e-72",107,
          "-1.001100011101100100010101101010101011010010111111010000000000000000000000000000e521",79,
          "-1.00110001110110010001010110101010101101001011111101000000000000000000000000000000000000000000000001e521",99,
          MPFR_RNDD);
  check2b("-1.01010001111000000101010100100100110101011011100001110000000000e498",63,
          "1.010000011010101111000100111100011100010101011110010100000000000e243",64,
          "-1.010100011110000001010101001001001101010110111000011100000000000e498",64,
          MPFR_RNDN);
  check2b("1.00101100010101000011010000011000111111011110010111000000000000000000000000000000000000000000000000000000000e178",108,
          "-1.10101101010101000110011011111001001101111111110000100000000e160",60,
          "1.00101100010100111100100011000011111001000010011101110010000000001111100000000000000000000000000000000000e178",105,
          MPFR_RNDN);
  check2b("1.00110011010100111110011010110100111101110101100100110000000000000000000000000000000000000000000000e559",99,
          "-1.011010110100111011100110100110011100000000111010011000000000000000e559",67,
          "-1.101111111101011111111111001001100100011100001001100000000000000000000000000000000000000000000e556",94,
          MPFR_RNDU);
  check2b("-1.100000111100101001100111011100011011000001101001111100000000000000000000000000e843",79,
          "-1.1101101010110000001001000100001100110011000110110111000000000000000000000000000000000000000000e414",95,
          "-1.1000001111001010011001110111000110110000011010100000e843",53,
          MPFR_RNDD);
  check2b("-1.110110010110100010100011000110111001010000010111110000000000e-415",61,
          "-1.0000100101100001111100110011111111110100011101101011000000000000000000e751",71,
          "-1.00001001011000011111001100111111111101000111011010110e751",54,
          MPFR_RNDN);
  check2b("-1.1011011011110001001101010101001000010100010110111101000000000000000000000e258",74,
          "-1.00011100010110110101001011000100100000100010101000010000000000000000000000000000000000000000000000e268",99,
          "-1.0001110011001001000011110001000111010110101011110010011011110100000000000000000000000000000000000000e268",101,
          MPFR_RNDD);
  check2b("-1.1011101010011101011000000100100110101101101110000001000000000e629",62,
          "1.111111100000011100100011100000011101100110111110111000000000000000000000000000000000000000000e525",94,
          "-1.101110101001110101100000010010011010110110111000000011111111111111111111111111111111111111111111111111101e629",106,
          MPFR_RNDD);
  check2b("1.111001000010001100010000001100000110001011110111011000000000000000000000000000000000000e152",88,
          "1.111110111001100100000100111111010111000100111111001000000000000000e152",67,
          "1.1110111111011110000010101001011011101010000110110100e153",53,
          MPFR_RNDN);
  check2b("1.000001100011110010110000110100001010101101111011110100e696",55,
          "-1.1011001111011100100001011110100101010101110111010101000000000000000000000000000000000000000000000000000000000000e730",113,
          "-1.1011001111011100100001011110100100010100010011100010e730",53,
          MPFR_RNDN);
  check2b("-1.11010111100001001111000001110101010010001111111001100000000000000000000000000000000000000000000000000000000000e530",111,
          "1.01110100010010000000010110111101011101000001111101100000000000000000000000000000000000000000000000e530",99,
          "-1.1000110011110011101010101101111101010011011111000000000000000e528",62,
          MPFR_RNDD);
  check2b("-1.0001100010010100111101101011101000100100010011100011000000000000000000000000000000000000000000000000000000000e733",110,
          "-1.001000000111110010100101010100110111001111011011001000000000000000000000000000000000000000000000000000000000e710",109,
          "-1.000110001001010011111000111110110001110110011000110110e733",55,
          MPFR_RNDN);
  check2b("-1.1101011110000100111100000111010101001000111111100110000000000000000000000e530",74,
          "1.01110100010010000000010110111101011101000001111101100000000000000000000000000000000000000000000000000000000000e530",111,
          "-1.10001100111100111010101011011111010100110111110000000000000000000000000000e528",75,
          MPFR_RNDU);
  check2b("1.00110011010100111110011010110100111101110101100100110000000000000000000000000000000000000000000000e559",99,
          "-1.011010110100111011100110100110011100000000111010011000000000000000e559",67,
          "-1.101111111101011111111111001001100100011100001001100000000000000000000000000000000000000000000e556",94,
          MPFR_RNDU);
  check2b("-1.100101111110110000000110111111011010011101101111100100000000000000e-624",67,
          "1.10111010101110100000010110101000000000010011100000100000000e-587",60,
          "1.1011101010111010000001011010011111110100011110001011111111001000000100101100010010000011100000000000000000000e-587",110,
          MPFR_RNDU);
  check2b("-1.10011001000010100000010100100110110010011111101111000000000000000000000000000000000000000000000000000000000000000000e-172",117,
          "1.111011100000101010110000100100110100100001001000011100000000e-173",61,
          "-1.0100010000001001010110011011101001001011101011110001000000000000000e-173",68,
          MPFR_RNDZ);
  check2b("1.1000111000110010101001010011010011101100010110001001000000000000000000000000000000000000000000000000e167",101,
          "1.0011110010000110000000101100100111000001110110110000000000000000000000000e167",74,
          "1.01100101010111000101001111111111010101110001100111001000000000000000000000000000000000000000000000000000e168",105,
          MPFR_RNDZ);
  check2b("1.100101111111110010100101110111100001110000100001010000000000000000000000000000000000000000000000e808",97,
          "-1.1110011001100000100000111111110000110010100111001011000000000000000000000000000000e807",83,
          "1.01001001100110001100011111000000000001011010010111010000000000000000000000000000000000000000000e807",96,
          MPFR_RNDN);
  check2b("1e128",128,
          "1e0",128,
          "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001e0",256,
          MPFR_RNDN);

  /* Checking double precision (53 bits) */
  check53("-8.22183238641455905806e-19", "7.42227178769761587878e-19",MPFR_RNDD,
          "-7.9956059871694317927e-20");
  check53("5.82106394662028628236e+234","-5.21514064202368477230e+89",MPFR_RNDD,
          "5.8210639466202855763e234");
  check53("5.72931679569871602371e+122","-5.72886070363264321230e+122",
          MPFR_RNDN, "4.5609206607281141508e118");
  check53("-5.09937369394650450820e+238", "2.70203299854862982387e+250",
          MPFR_RNDD, "2.7020329985435301323e250");
  check53("-2.96695924472363684394e+27", "1.22842938251111500000e+16",MPFR_RNDD,
          "-2.96695924471135255027e27");
  check53("1.74693641655743793422e-227", "-7.71776956366861843469e-229",
          MPFR_RNDN, "1.669758720920751867e-227");
  /*  x = -7883040437021647.0; for (i=0; i<468; i++) x = x / 2.0;*/
  check53("-1.03432206392780011159e-125", "1.30127034799251347548e-133",
          MPFR_RNDN,
          "-1.0343220509150965661100887242027378881805094180354e-125");
  check53("1.05824655795525779205e+71", "-1.06022698059744327881e+71",MPFR_RNDZ,
          "-1.9804226421854867632e68");
  check53("-5.84204911040921732219e+240", "7.26658169050749590763e+240",
          MPFR_RNDD, "1.4245325800982785854e240");
  check53("1.00944884131046636376e+221","2.33809162651471520268e+215",MPFR_RNDN,
          "1.0094511794020929787e221");
  /*x = 7045852550057985.0; for (i=0; i<986; i++) x = x / 2.0;*/
  check53("4.29232078932667367325e-278",
          "1.0773525047389793833221116707010783793203080117586e-281"
          , MPFR_RNDU, "4.2933981418314132787e-278");
  check53("5.27584773801377058681e-80", "8.91207657803547196421e-91", MPFR_RNDN,
          "5.2758477381028917269e-80");
  check53("2.99280481918991653800e+272", "5.34637717585790933424e+271",
          MPFR_RNDN, "3.5274425367757071711e272");
  check53("4.67302514390488041733e-184", "2.18321376145645689945e-190",
          MPFR_RNDN, "4.6730273271186420541e-184");
  check53("5.57294120336300389254e+71", "2.60596167942024924040e+65", MPFR_RNDZ,
          "5.5729438093246831053e71");
  check53("6.6052588496951015469e24", "4938448004894539.0", MPFR_RNDU,
          "6.6052588546335505068e24");
  check53("1.23056185051606761523e-190", "1.64589756643433857138e-181",
          MPFR_RNDU, "1.6458975676649006598e-181");
  check53("2.93231171510175981584e-280", "3.26266919161341483877e-273",
          MPFR_RNDU, "3.2626694848445867288e-273");
  check53("5.76707395945001907217e-58", "4.74752971449827687074e-51", MPFR_RNDD,
          "4.747530291205672325e-51");
  check53("277363943109.0", "11.0", MPFR_RNDN, "277363943120.0");
  check53("1.44791789689198883921e-140", "-1.90982880222349071284e-121",
          MPFR_RNDN, "-1.90982880222349071e-121");


  /* tests for particular cases (Vincent Lefevre, 22 Aug 2001) */
  check53("9007199254740992.0", "1.0", MPFR_RNDN, "9007199254740992.0");
  check53("9007199254740994.0", "1.0", MPFR_RNDN, "9007199254740996.0");
  check53("9007199254740992.0", "-1.0", MPFR_RNDN, "9007199254740991.0");
  check53("9007199254740994.0", "-1.0", MPFR_RNDN, "9007199254740992.0");
  check53("9007199254740996.0", "-1.0", MPFR_RNDN, "9007199254740996.0");

  check_overflow ();
  check_1111 ();
  check_1minuseps ();
}

#define TEST_FUNCTION test_add
#define TWO_ARGS
#define RAND_FUNCTION(x) mpfr_random2(x, MPFR_LIMB_SIZE (x), randlimb () % 100, RANDS)
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  usesp = 0;
  tests ();

#ifndef CHECK_EXTERNAL /* no need to check twice */
  usesp = 1;
  tests ();
#endif
  test_generic (2, 1000, 100);

  tests_end_mpfr ();
  return 0;
}
