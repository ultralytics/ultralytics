/* Test file for mpfr_sub1sp.

Copyright 2003-2017 Free Software Foundation, Inc.
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

static void check_special (void);
static void check_random (mpfr_prec_t p);

int
main (void)
{
  mpfr_prec_t p;

  tests_start_mpfr ();

  check_special ();
  for (p = 2 ; p < 200 ; p++)
    check_random (p);

  tests_end_mpfr ();
  return 0;
}

#define STD_ERROR                                                       \
  do                                                                    \
    {                                                                   \
      printf("ERROR: for %s and p=%lu and i=%d:\nY=",                   \
             mpfr_print_rnd_mode ((mpfr_rnd_t) r), (unsigned long) p, i); \
      mpfr_print_binary(y);                                             \
      printf("\nZ="); mpfr_print_binary(z);                             \
      printf("\nReal: "); mpfr_print_binary(x2);                        \
      printf("\nGot : "); mpfr_print_binary(x);                         \
      putchar('\n');                                                    \
      exit(1);                                                          \
    }                                                                   \
 while (0)

#define STD_ERROR2                                                      \
  do                                                                    \
    {                                                                   \
      printf("ERROR: for %s and p=%lu and i=%d:\nY=",                   \
             mpfr_print_rnd_mode ((mpfr_rnd_t) r), (unsigned long) p, i); \
      mpfr_print_binary(y);                                             \
      printf("\nZ="); mpfr_print_binary(z);                             \
      printf("\nR="); mpfr_print_binary(x);                             \
      printf("\nWrong inexact flag. Real: %d. Got: %d\n",               \
             inexact1, inexact2);                                       \
      exit(1);                                                          \
    }                                                                   \
 while (0)

static void
check_random (mpfr_prec_t p)
{
  mpfr_t x,y,z,x2;
  int r;
  int i, inexact1, inexact2;

  mpfr_inits2 (p, x, y, z, x2, (mpfr_ptr) 0);

  for (i = 0 ; i < 500 ; i++)
    {
      mpfr_urandomb (y, RANDS);
      mpfr_urandomb (z, RANDS);
      if (MPFR_IS_PURE_FP(y) && MPFR_IS_PURE_FP(z))
        for(r = 0 ; r < MPFR_RND_MAX ; r++)
          {
            inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
            inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
            if (mpfr_cmp(x, x2))
              STD_ERROR;
            if (inexact1 != inexact2)
              STD_ERROR2;
          }
    }

  mpfr_clears (x, y, z, x2, (mpfr_ptr) 0);
}

static void
check_special (void)
{
  mpfr_t x,y,z,x2;
  int r;
  mpfr_prec_t p;
  int i = -1, inexact1, inexact2;
  mpfr_exp_t es;

  mpfr_inits (x, y, z, x2, (mpfr_ptr) 0);

  for (r = 0 ; r < MPFR_RND_MAX ; r++)
    {
      p = 53;
      mpfr_set_prec(x, 53);
      mpfr_set_prec(x2, 53);
      mpfr_set_prec(y, 53);
      mpfr_set_prec(z, 53);

      mpfr_set_str_binary (y,
       "0.10110111101101110010010010011011000001101101011011001E31");

      mpfr_sub1sp (x, y, y, (mpfr_rnd_t) r);
      if (mpfr_cmp_ui(x, 0))
        {
          printf("Error for x-x with p=%lu. Expected 0. Got:",
                 (unsigned long) p);
          mpfr_print_binary(x);
          exit(1);
        }

      mpfr_set(z, y, (mpfr_rnd_t) r);
      mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp_ui(x, 0))
        {
          printf("Error for x-y with y=x and p=%lu. Expected 0. Got:",
                 (unsigned long) p);
          mpfr_print_binary(x);
          exit(1);
        }
      /* diff = 0 */
      mpfr_set_str_binary (y,
       "0.10110111101101110010010010011011001001101101011011001E31");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      /* Diff = 1 */
      mpfr_set_str_binary (y,
       "0.10110111101101110010010010011011000001101101011011001E30");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      /* Diff = 2 */
      mpfr_set_str_binary (y,
       "0.10110111101101110010010010011011000101101101011011001E32");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      /* Diff = 32 */
      mpfr_set_str_binary (y,
       "0.10110111101101110010010010011011000001101101011011001E63");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      /* Diff = 52 */
      mpfr_set_str_binary (y,
       "0.10110111101101110010010010011011010001101101011011001E83");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      /* Diff = 53 */
      mpfr_set_str_binary (y,
       "0.10110111101101110010010010011111000001101101011011001E31");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      /* Diff > 200 */
      mpfr_set_str_binary (y,
       "0.10110111101101110010010010011011000001101101011011001E331");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
       "0.10000000000000000000000000000000000000000000000000000E31");
      mpfr_set_str_binary (z,
       "0.11111111111111111111111111111111111111111111111111111E30");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
       "0.10000000000000000000000000000000000000000000000000000E31");
      mpfr_set_str_binary (z,
       "0.11111111111111111111111111111111111111111111111111111E29");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
       "0.10000000000000000000000000000000000000000000000000000E52");
      mpfr_set_str_binary (z,
       "0.10000000000010000000000000000000000000000000000000000E00");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
        "0.11100000000000000000000000000000000000000000000000000E53");
      mpfr_set_str_binary (z,
        "0.10000000000000000000000000000000000000000000000000000E00");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(z, y, z, (mpfr_rnd_t) r);
      mpfr_set(x, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
       "0.10000000000000000000000000000000000000000000000000000E53");
      mpfr_set_str_binary (z,
       "0.10100000000000000000000000000000000000000000000000000E00");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
        "0.10000000000000000000000000000000000000000000000000000E54");
      mpfr_set_str_binary (z,
        "0.10100000000000000000000000000000000000000000000000000E00");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      p = 63;
      mpfr_set_prec(x, p);
      mpfr_set_prec(x2, p);
      mpfr_set_prec(y, p);
      mpfr_set_prec(z, p);
      mpfr_set_str_binary (y,
      "0.100000000000000000000000000000000000000000000000000000000000000E62");
      mpfr_set_str_binary (z,
      "0.110000000000000000000000000000000000000000000000000000000000000E00");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      p = 64;
      mpfr_set_prec(x, 64);
      mpfr_set_prec(x2, 64);
      mpfr_set_prec(y, 64);
      mpfr_set_prec(z, 64);

      mpfr_set_str_binary (y,
      "0.1100000000000000000000000000000000000000000000000000000000000000E31");
      mpfr_set_str_binary (z,
      "0.1111111111111111111111111110000000000000000000000000011111111111E29");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
      "0.1000000000000000000000000000000000000000000000000000000000000000E63");
      mpfr_set_str_binary (z,
      "0.1011000000000000000000000000000000000000000000000000000000000000E00");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
      "0.1000000000000000000000000000000000000000000000000000000000000000E63");
      mpfr_set_str_binary (z,
      "0.1110000000000000000000000000000000000000000000000000000000000000E00");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
        "0.10000000000000000000000000000000000000000000000000000000000000E63");
      mpfr_set_str_binary (z,
        "0.10000000000000000000000000000000000000000000000000000000000000E00");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
      "0.1000000000000000000000000000000000000000000000000000000000000000E64");
      mpfr_set_str_binary (z,
      "0.1010000000000000000000000000000000000000000000000000000000000000E00");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      MPFR_SET_NAN(x);
      MPFR_SET_NAN(x2);
      mpfr_set_str_binary (y,
      "0.1000000000000000000000000000000000000000000000000000000000000000"
                          "E-1073741823");
      mpfr_set_str_binary (z,
      "0.1100000000000000000000000000000000000000000000000000000000000000"
                          "E-1073741823");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      p = 9;
      mpfr_set_prec(x, p);
      mpfr_set_prec(x2, p);
      mpfr_set_prec(y, p);
      mpfr_set_prec(z, p);

      mpfr_set_str_binary (y, "0.100000000E1");
      mpfr_set_str_binary (z, "0.100000000E-8");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      p = 34;
      mpfr_set_prec(x, p);
      mpfr_set_prec(x2, p);
      mpfr_set_prec(y, p);
      mpfr_set_prec(z, p);

      mpfr_set_str_binary (y, "-0.1011110000111100010111011100110100E-18");
      mpfr_set_str_binary (z, "0.1000101010110011010101011110000000E-14");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      p = 124;
      mpfr_set_prec(x, p);
      mpfr_set_prec(x2, p);
      mpfr_set_prec(y, p);
      mpfr_set_prec(z, p);

      mpfr_set_str_binary (y,
"0.1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000E1");
      mpfr_set_str_binary (z,
"0.1011111000100111000011001000011101010101101100101010101001000001110100001101110110001110111010000011101001100010111110001100E-31");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      p = 288;
      mpfr_set_prec(x, p);
      mpfr_set_prec(x2, p);
      mpfr_set_prec(y, p);
      mpfr_set_prec(z, p);

      mpfr_set_str_binary (y,
     "0.111000110011000001000111101010111011110011101001101111111110000011100101000001001010110010101010011001010100000001110011110001010101101010001011101110100100001011110100110000101101100011010001001011011010101010000010001101001000110010010111111011110001111101001000101101001100101100101000E80");
      mpfr_set_str_binary (z,
     "-0.100001111111101001011010001100110010100111001110000110011101001011010100001000000100111011010110110010000000000010101101011000010000110001110010100001100101011100100100001011000100011110000001010101000100011101001000010111100000111000111011001000100100011000100000010010111000000100100111E-258");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      p = 85;
      mpfr_set_prec(x, p);
      mpfr_set_prec(x2, p);
      mpfr_set_prec(y, p);
      mpfr_set_prec(z, p);

      mpfr_set_str_binary (y,
"0.1111101110100110110110100010101011101001100010100011110110110010010011101100101111100E-4");
      mpfr_set_str_binary (z,
"0.1111101110100110110110100010101001001000011000111000011101100101110100001110101010110E-4");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      p = 64;
      mpfr_set_prec(x, p); mpfr_set_prec(x2, p);
      mpfr_set_prec(y, p); mpfr_set_prec(z, p);

      mpfr_set_str_binary (y,
                          "0.11000000000000000000000000000000"
                          "00000000000000000000000000000000E1");
      mpfr_set_str_binary (z,
                          "0.10000000000000000000000000000000"
                          "00000000000000000000000000000001E0");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
                          "0.11000000000000000000000000000000"
                          "000000000000000000000000000001E1");
      mpfr_set_str_binary (z,
                          "0.10000000000000000000000000000000"
                          "00000000000000000000000000000001E0");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      es = mpfr_get_emin ();
      set_emin (-1024);

      mpfr_set_str_binary (y,
                          "0.10000000000000000000000000000000"
                          "000000000000000000000000000000E-1023");
      mpfr_set_str_binary (z,
                          "0.10000000000000000000000000000000"
                          "00000000000000000000000000000001E-1023");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      mpfr_set_str_binary (y,
                           "0.10000000000000000000000000000000"
                           "000000000000000000000000000000E-1023");
      mpfr_set_str_binary (z,
                           "0.1000000000000000000000000000000"
                           "000000000000000000000000000000E-1023");
      inexact1 = mpfr_sub1(x2, y, z, (mpfr_rnd_t) r);
      inexact2 = mpfr_sub1sp(x, y, z, (mpfr_rnd_t) r);
      if (mpfr_cmp(x, x2))
        STD_ERROR;
      if (inexact1 != inexact2)
        STD_ERROR2;

      set_emin (es);
    }

  mpfr_clears (x, y, z, x2, (mpfr_ptr) 0);
}
