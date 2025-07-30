/* Test file for mpfr_get_str.

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

#include <stdlib.h>

#include "mpfr-test.h"

static void
check3 (const char *d, mpfr_rnd_t rnd, const char *res)
{
  mpfr_t x;
  char *str;
  mpfr_exp_t e;

  mpfr_init2 (x, 53);
  mpfr_set_str (x, d, 10, rnd);
  str = mpfr_get_str (NULL, &e, 10, 5, x, rnd);
  if (strcmp (str, res))
    {
      printf ("Error in mpfr_get_str for x=%s\n", d);
      printf ("got %s instead of %s\n", str, res);
      exit (1);
    }
  mpfr_clear (x);
  mpfr_free_str (str);
}

static void
check_small (void)
{
  mpfr_t x;
  char *s;
  mpfr_exp_t e;
  mpfr_prec_t p;

  mpfr_init (x);

  mpfr_set_prec (x, 20);
  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_nexttozero (x);
  s = mpfr_get_str (NULL, &e, 4, 2, x, MPFR_RNDU);
  if (strcmp (s, "20") || (e != 1))
    {
      printf ("Error in mpfr_get_str: 2- rounded up with 2 digits"
              " in base 4\n");
      exit (1);
    }
  mpfr_free_str (s);

  /* check n_digits=0 */
  mpfr_set_prec (x, 5);
  mpfr_set_ui (x, 17, MPFR_RNDN);
  s = mpfr_get_str (NULL, &e, 3, 0, x, MPFR_RNDN);
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 36, 0, x, MPFR_RNDN);
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 62, 0, x, MPFR_RNDN);
  mpfr_free_str (s);

  mpfr_set_prec (x, 64);
  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_div_2exp (x, x, 63, MPFR_RNDN); /* x = -2^(-63) */
  mpfr_add_ui (x, x, 1, MPFR_RNDN); /* x = 1 - 2^(-63) */
  mpfr_mul_2exp (x, x, 32, MPFR_RNDN); /* x = 2^32 - 2^(-31) */
  s = mpfr_get_str (NULL, &e, 3, 21, x, MPFR_RNDU);
  if (strcmp (s, "102002022201221111211") || (e != 21))
    {
      printf ("Error in mpfr_get_str: 2^32-2^(-31) rounded up with"
              " 21 digits in base 3\n");
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 3, 20, x, MPFR_RNDU);
  if (strcmp (s, "10200202220122111122") || (e != 21))
    {
      printf ("Error in mpfr_get_str: 2^32-2^(-31) rounded up with"
              " 20 digits in base 3\n");
      exit (1);
    }
  mpfr_free_str (s);

  /* check corner case ret!=0, j0!=0 in mpfr_get_str_aux */
  mpfr_set_prec (x, 100);
  mpfr_set_str_binary (x, "0.1001011111010001101110010101010101111001010111111101101101100110100011110110000101110110001011110000E-9");
  s = mpfr_get_str (NULL, &e, 3, 2, x, MPFR_RNDU);
  if (strcmp (s, "22") || (e != -6))
    {
      printf ("Error in mpfr_get_str: 100-bit number rounded up with"
              " 2 digits in base 3\n");
      exit (1);
    }
  mpfr_free_str (s);

  /* check corner case exact=0 in mpfr_get_str_aux */
  mpfr_set_prec (x, 100);
  mpfr_set_str_binary (x, "0.1001001111101101111000101000110111111010101100000110010001111111011001101011101100001100110000000000E8");
  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDZ);
  if (strcmp (s, "14") || (e != 3))
    {
      printf ("Error in mpfr_get_str: 100-bit number rounded to zero with"
              " 2 digits in base 10\n");
      exit (1);
    }
  mpfr_free_str (s);

  for (p=4; p<=200; p++)
    {
      mpfr_set_prec (x, p);
      mpfr_set_str (x, "6.5", 10, MPFR_RNDN);

      s = mpfr_get_str (NULL, &e, 6, 2, x, MPFR_RNDN);
      if (strcmp (s, "10") || (e != 2))
        {
          printf ("Error in mpfr_get_str: 6.5 rounded to nearest with"
                  " 2 digits in base 6\n");
          exit (1);
        }
      mpfr_free_str (s);

      mpfr_nexttoinf (x);
      s = mpfr_get_str (NULL, &e, 6, 2, x, MPFR_RNDN);
      if (strcmp (s, "11") || (e != 2))
        {
          printf ("Error in mpfr_get_str: 6.5+ rounded to nearest with"
                  " 2 digits in base 6\ngot %se%d instead of 11e2\n",
                  s, (int) e);
          exit (1);
        }
      mpfr_free_str (s);

      mpfr_set_str (x, "6.5", 10, MPFR_RNDN);
      mpfr_nexttozero (x);
      s = mpfr_get_str (NULL, &e, 6, 2, x, MPFR_RNDN);
      if (strcmp (s, "10") || (e != 2))
        {
          printf ("Error in mpfr_get_str: 6.5- rounded to nearest with"
                  " 2 digits in base 6\n");
          exit (1);
        }
      mpfr_free_str (s);
    }

  mpfr_set_prec (x, 3);
  mpfr_set_ui (x, 7, MPFR_RNDN);
  s = mpfr_get_str (NULL, &e, 2, 2, x, MPFR_RNDU);
  if (strcmp (s, "10") || (e != 4))
    {
      printf ("Error in mpfr_get_str: 7 rounded up with 2 bits should"
              " give 0.10e3 instead of 0.%s*2^%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* problem found by Fabrice Rouillier */
  mpfr_set_prec (x, 63);
  mpfr_set_str (x, "5e14", 10, MPFR_RNDN);
  s = mpfr_get_str (NULL, &e, 10, 18, x, MPFR_RNDU);
  mpfr_free_str (s);

  /* bug found by Johan Vervloet */
  mpfr_set_prec (x, 6);
  mpfr_set_str (x, "688.0", 10, MPFR_RNDN);
  s = mpfr_get_str (NULL, &e, 2, 4, x, MPFR_RNDU);
  if (strcmp (s, "1011") || (e != 10))
    {
      printf ("Error in mpfr_get_str: 688 printed up to 4 bits should"
              " give 1.011e9\ninstead of ");
      mpfr_out_str (stdout, 2, 4, x, MPFR_RNDU);
      puts ("");
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_prec (x, 38);
  mpfr_set_str_binary (x, "1.0001110111110100011010100010010100110e-6");
  s = mpfr_get_str (NULL, &e, 8, 10, x, MPFR_RNDU);
  if (strcmp (s, "1073721522") || (e != -1))
    {
      printf ("Error in mpfr_get_str (3): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_prec (x, 53);
  mpfr_set_str_binary (x, "0.11010111011101100010000100010101110001000000010111001E454");
  s = mpfr_get_str (NULL, &e, 19, 12, x, MPFR_RNDU);
  if (strcmp (s, "b1cgfa4gha0h") || (e != 107))
    {
      printf ("Error in mpfr_get_str (4): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_prec (x, 145);
  mpfr_set_str_binary (x, "-0.1000110011000001011000010101101010110110101100101110100011111100011110011001001001010000100001000011000011000000010111011001000111101001110100110e6");
  s = mpfr_get_str (NULL, &e, 4, 53, x, MPFR_RNDU);
  if (strcmp (s, "-20303001120111222312230232203330132121021100201003003") || (e != 3))
    {
      printf ("Error in mpfr_get_str (5): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_prec (x, 45);
  mpfr_set_str_binary (x, "-0.00100111010110010001011001110111010001010010010");
  s = mpfr_get_str (NULL, &e, 32, 9, x, MPFR_RNDN);
  if (strcmp (s, "-4tchctq54") || (e != 0))
    {
      printf ("Error in mpfr_get_str (6): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* worst case found by Vincent Lefe`vre */
  mpfr_set_prec (x, 53);
  mpfr_set_str_binary (x, "10011110111100000000001011011110101100010000011011111E164");
  s = mpfr_get_str (NULL, &e, 10, 17, x, MPFR_RNDN);
  if (strcmp (s, "13076622631878654") || (e != 66))
    {
      printf ("Error in mpfr_get_str (7): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "10000001001001001100011101010011011011111000011000100E93");
  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDU);
  if (strcmp (s, "46") || e != 44)
    {
       printf ("Error in mpfr_get_str (8): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "10010001111100000111001111010101001010000010111010101E55");
  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDN);
  if (strcmp (s, "19") || e != 33)
    {
       printf ("Error in mpfr_get_str (9): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "11011001010010111110010101101100111110111000010110110E44");
  s = mpfr_get_str (NULL, &e, 10, 3, x, MPFR_RNDN);
  if (strcmp (s, "135") || e != 30)
    {
       printf ("Error in mpfr_get_str (10): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "11101111101000001011100001111000011111101111011001100E72");
  s = mpfr_get_str (NULL, &e, 10, 4, x, MPFR_RNDN);
  if (strcmp (s, "3981") || e != 38)
    {
       printf ("Error in mpfr_get_str (11): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "10011001001100100010111100001101110101001001111110000E46");
  s = mpfr_get_str (NULL, &e, 10, 5, x, MPFR_RNDN);
  if (strcmp (s, "37930") || e != 30)
    {
       printf ("Error in mpfr_get_str (12): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "10001100110111001011011110011011011101100011010001011E-72");
  s = mpfr_get_str (NULL, &e, 10, 6, x, MPFR_RNDN);
  if (strcmp (s, "104950") || e != -5)
    {
       printf ("Error in mpfr_get_str (13): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "10100100001011001000011001101101000110100110000010111E89");
  s = mpfr_get_str (NULL, &e, 10, 7, x, MPFR_RNDN);
  if (strcmp (s, "3575392") || e != 43)
    {
       printf ("Error in mpfr_get_str (14): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "11000011011110110010100110001010000001010011001011001E-73");
  s = mpfr_get_str (NULL, &e, 10, 8, x, MPFR_RNDN);
  if (strcmp (s, "72822386") || e != -6)
    {
       printf ("Error in mpfr_get_str (15): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "10101010001101000111001100001000100011100010010001010E78");
  s = mpfr_get_str (NULL, &e, 10, 9, x, MPFR_RNDN);
  if (strcmp (s, "180992873") || e != 40)
    {
      printf ("Error in mpfr_get_str (16): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "10110111001000100000001101111001100101101110011011101E91");
  s = mpfr_get_str (NULL, &e, 10, 10, x, MPFR_RNDN);
  if (strcmp (s, "1595312255") || e != 44)
    {
      printf ("Error in mpfr_get_str (17): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "10011101010111101111000100111011101011110100110110101E93");
  s = mpfr_get_str (NULL, &e, 10, 11, x, MPFR_RNDN);
  if (strcmp (s, "54835744350") || e != 44)
    {
      printf ("Error in mpfr_get_str (18): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "10011101010111101111000100111011101011110100110110101E92");
  s = mpfr_get_str (NULL, &e, 10, 12, x, MPFR_RNDN);
  if (strcmp (s, "274178721752") || e != 44)
    {
      printf ("Error in mpfr_get_str (19): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "10011101010111101111000100111011101011110100110110101E91");
  s = mpfr_get_str (NULL, &e, 10, 13, x, MPFR_RNDN);
  if (strcmp (s, "1370893608762") || e != 44)
    {
      printf ("Error in mpfr_get_str (20): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "10010011010110011100010010100101100011101000011111111E92");
  s = mpfr_get_str (NULL, &e, 10, 14, x, MPFR_RNDN);
  if (strcmp (s, "25672105101864") || e != 44)
    {
      printf ("Error in mpfr_get_str (21): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "100110111110110001000101110100100101101000011111001E87");
  s = mpfr_get_str (NULL, &e, 10, 15, x, MPFR_RNDN);
  if (strcmp (s, "212231308858721") || e != 42)
    {
      printf ("Error in mpfr_get_str (22): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "10111010110000111000101100101111001011011100101001111E-128");
  s = mpfr_get_str (NULL, &e, 10, 15, x, MPFR_RNDN);
  if (strcmp (s, "193109287087290") || e != -22)
    {
      printf ("Error in mpfr_get_str (22b): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "10001101101011010001111110000111010111010000110101010E80");
  s = mpfr_get_str (NULL, &e, 10, 16, x, MPFR_RNDN);
  if (strcmp (s, "6026241735727920") || e != 40)
    {
      printf ("Error in mpfr_get_str (23): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "100010001011101001110101000110011001001000110001001E-81");
  s = mpfr_get_str (NULL, &e, 10, 17, x, MPFR_RNDN);
  if (strcmp (s, "49741483709103481") || e != -9)
    {
      printf ("Error in mpfr_get_str (24): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "11000100001001001110111010011001111001001010110101111E-101");
  s = mpfr_get_str (NULL, &e, 10, 7, x, MPFR_RNDN);
  if (strcmp (s, "2722049") || e != -14)
    {
      printf ("Error in mpfr_get_str (25): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "11111001010011100101000001111111110001001001110110001E-135");
  s = mpfr_get_str (NULL, &e, 10, 8, x, MPFR_RNDN);
  if (strcmp (s, "20138772") || e != -24)
    {
      printf ("Error in mpfr_get_str (26): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "11111001010011100101000001111111110001001001110110001E-136");
  s = mpfr_get_str (NULL, &e, 10, 9, x, MPFR_RNDN);
  if (strcmp (s, "100693858") || e != -24)
    {
      printf ("Error in mpfr_get_str (27): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
    mpfr_free_str (s);
  mpfr_set_str_binary (x, "10001000001110010110001011111011111011011010000110001E-110");
  s = mpfr_get_str (NULL, &e, 10, 14, x, MPFR_RNDN);
  if (strcmp (s, "36923634350619") || e != -17)
    {
      printf ("Error in mpfr_get_str (28): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "11001100010111000111100010000110011101110001000101111E-87");
  s = mpfr_get_str (NULL, &e, 10, 16, x, MPFR_RNDN);
  if (strcmp (s, "4646636036100804") || e != -10)
    {
      printf ("Error in mpfr_get_str (29): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "10011111001111110100001001010111111011010101111111000E-99");
  s = mpfr_get_str (NULL, &e, 10, 17, x, MPFR_RNDN);
  if (strcmp (s, "88399901882446712") || e != -14)
    {
      printf ("Error in mpfr_get_str (30): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 8116315218207718*2^(-293) ~ 0.5100000000000000000015*10^(-72) */
  mpfr_set_str_binary (x, "11100110101011011111011100101011101110110001111100110E-293");
  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDU);
  if (strcmp (s, "52") || e != -72)
    {
      printf ("Error in mpfr_get_str (31u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDD);
  if (strcmp (s, "51") || e != -72)
    {
      printf ("Error in mpfr_get_str (31d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 6712731423444934*2^536 ~ .151000000000000000000067*10^178 */
  mpfr_set_str_binary (x, "10111110110010011000110010011111101111000111111000110E536");
  s = mpfr_get_str (NULL, &e, 10, 3, x, MPFR_RNDU);
  if (strcmp (s, "152") || e != 178)
    {
      printf ("Error in mpfr_get_str (32u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 3, x, MPFR_RNDD);
  if (strcmp (s, "151") || e != 178)
    {
      printf ("Error in mpfr_get_str (32d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 3356365711722467*2^540 ~ .120800000000000000000054*10^179 */
  mpfr_set_str_binary (x, "1011111011001001100011001001111110111100011111100011E540");
  s = mpfr_get_str (NULL, &e, 10, 4, x, MPFR_RNDU);
  if (strcmp (s, "1209") || e != 179)
    {
      printf ("Error in mpfr_get_str (33u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 4, x, MPFR_RNDD);
  if (strcmp (s, "1208") || e != 179)
    {
      printf ("Error in mpfr_get_str (33d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 6475049196144587*2^100 ~ .8208099999999999999999988*10^46 */
  mpfr_set_str_binary (x, "10111000000010000010111011111001111010100011111001011E100");
  s = mpfr_get_str (NULL, &e, 10, 5, x, MPFR_RNDU);
  if (strcmp (s, "82081") || e != 46)
    {
      printf ("Error in mpfr_get_str (34u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 5, x, MPFR_RNDD);
  if (strcmp (s, "82080") || e != 46)
    {
      printf ("Error in mpfr_get_str (34d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 6722280709661868*2^364 ~ .25260100000000000000000012*10^126 */
  mpfr_set_str_binary (x, "10111111000011110000011110001110001111010010010101100E364");
  s = mpfr_get_str (NULL, &e, 10, 6, x, MPFR_RNDU);
  if (strcmp (s, "252602") || e != 126)
    {
      printf ("Error in mpfr_get_str (35u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 6, x, MPFR_RNDD);
  if (strcmp (s, "252601") || e != 126)
    {
      printf ("Error in mpfr_get_str (35d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 5381065484265332*2^(-455) ~ .578389299999999999999999982*10^(-121) */
  mpfr_set_str_binary (x, "10011000111100000110011110000101100111110011101110100E-455");
  s = mpfr_get_str (NULL, &e, 10, 7, x, MPFR_RNDU);
  if (strcmp (s, "5783893") || e != -121)
    {
      printf ("Error in mpfr_get_str (36u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 7, x, MPFR_RNDD);
  if (strcmp (s, "5783892") || e != -121)
    {
      printf ("Error in mpfr_get_str (36d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 8369123604277281*2^(-852) ~ .27869147000000000000000000056*10^(-240) */
  mpfr_set_str_binary (x, "11101101110111010110001101111100000111010100000100001E-852");
  s = mpfr_get_str (NULL, &e, 10, 8, x, MPFR_RNDU);
  if (strcmp (s, "27869148") || e != -240)
    {
      printf ("Error in mpfr_get_str (37u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 8, x, MPFR_RNDD);
  if (strcmp (s, "27869147") || e != -240)
    {
      printf ("Error in mpfr_get_str (37d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 7976538478610756*2^377 ~ .245540326999999999999999999982*10^130 */
  mpfr_set_str_binary (x, "11100010101101001111010010110100011100000100101000100E377");
  s = mpfr_get_str (NULL, &e, 10, 9, x, MPFR_RNDU);
  if (strcmp (s, "245540327") || e != 130)
    {
      printf ("Error in mpfr_get_str (38u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 9, x, MPFR_RNDD);
  if (strcmp (s, "245540326") || e != 130)
    {
      printf ("Error in mpfr_get_str (38d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 8942832835564782*2^(-382) ~ .9078555839000000000000000000038*10^(-99) */
  mpfr_set_str_binary (x, "11111110001010111010110000110011100110001010011101110E-382");
  s = mpfr_get_str (NULL, &e, 10, 10, x, MPFR_RNDU);
  if (strcmp (s, "9078555840") || e != -99)
    {
      printf ("Error in mpfr_get_str (39u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 10, x, MPFR_RNDD);
  if (strcmp (s, "9078555839") || e != -99)
    {
      printf ("Error in mpfr_get_str (39d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 4471416417782391*2^(-380) ~ .18157111678000000000000000000077*10^(-98) */
  mpfr_set_str_binary (x, "1111111000101011101011000011001110011000101001110111E-380");
  s = mpfr_get_str (NULL, &e, 10, 11, x, MPFR_RNDU);
  if (strcmp (s, "18157111679") || e != -98)
    {
      printf ("Error in mpfr_get_str (40u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 11, x, MPFR_RNDD);
  if (strcmp (s, "18157111678") || e != -98)
    {
      printf ("Error in mpfr_get_str (40d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 7225450889282194*2^711 ~ .778380362292999999999999999999971*10^230 */
  mpfr_set_str_binary (x, "11001101010111000001001100001100110010000001010010010E711");
  s = mpfr_get_str (NULL, &e, 10, 12, x, MPFR_RNDU);
  if (strcmp (s, "778380362293") || e != 230)
    {
      printf ("Error in mpfr_get_str (41u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 12, x, MPFR_RNDD);
  if (strcmp (s, "778380362292") || e != 230)
    {
      printf ("Error in mpfr_get_str (41d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 3612725444641097*2^713 ~ .1556760724585999999999999999999942*10^231 */
  mpfr_set_str_binary (x, "1100110101011100000100110000110011001000000101001001E713");
  s = mpfr_get_str (NULL, &e, 10, 13, x, MPFR_RNDU);
  if (strcmp (s, "1556760724586") || e != 231)
    {
      printf ("Error in mpfr_get_str (42u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 13, x, MPFR_RNDD);
  if (strcmp (s, "1556760724585") || e != 231)
    {
      printf ("Error in mpfr_get_str (42d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 6965949469487146*2^(-248) ~ .15400733123779000000000000000000016*10^(-58) */
  mpfr_set_str_binary (x, "11000101111110111111001111111101001101111000000101010E-248");
  s = mpfr_get_str (NULL, &e, 10, 14, x, MPFR_RNDU);
  if (strcmp (s, "15400733123780") || e != -58)
    {
      printf ("Error in mpfr_get_str (43u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 14, x, MPFR_RNDD);
  if (strcmp (s, "15400733123779") || e != -58)
    {
      printf ("Error in mpfr_get_str (43d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 3482974734743573*2^(-244) ~ .12320586499023200000000000000000013*10^(-57) */
  mpfr_set_str_binary (x, "1100010111111011111100111111110100110111100000010101E-244");
  s = mpfr_get_str (NULL, &e, 10, 15, x, MPFR_RNDU);
  if (strcmp (s, "123205864990233") || e != -57)
    {
      printf ("Error in mpfr_get_str (44u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 15, x, MPFR_RNDD);
  if (strcmp (s, "123205864990232") || e != -57)
    {
      printf ("Error in mpfr_get_str (44d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 7542952370752766*2^(-919) ~ .170206189963739699999999999999999974*10^(-260) */
  mpfr_set_str_binary (x, "11010110011000100011001110100100111011100110011111110E-919");
  s = mpfr_get_str (NULL, &e, 10, 16, x, MPFR_RNDU);
  if (strcmp (s, "1702061899637397") || e != -260)
    {
      printf ("Error in mpfr_get_str (45u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 16, x, MPFR_RNDD);
  if (strcmp (s, "1702061899637396") || e != -260)
    {
      printf ("Error in mpfr_get_str (45d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  /* 5592117679628511*2^165 ~ .26153245263757307000000000000000000074*10^66 */
  mpfr_set_str_binary (x, "10011110111100000000001011011110101100010000011011111E165");
  s = mpfr_get_str (NULL, &e, 10, 17, x, MPFR_RNDU);
  if (strcmp (s, "26153245263757308") || e != 66)
    {
      printf ("Error in mpfr_get_str (46u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 17, x, MPFR_RNDD);
  if (strcmp (s, "26153245263757307") || e != 66)
    {
      printf ("Error in mpfr_get_str (46d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "11010010110111100001011010000110010000100001011011101E1223");
  s = mpfr_get_str (NULL, &e, 10, 17, x, MPFR_RNDN);
  if (strcmp (s, "10716284017294180") || e != 385)
    {
      printf ("Error in mpfr_get_str (47n): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 18, x, MPFR_RNDU);
  if (strcmp (s, "107162840172941805") || e != 385)
    {
      printf ("Error in mpfr_get_str (47u): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 18, x, MPFR_RNDD);
  if (strcmp (s, "107162840172941804") || e != 385)
    {
      printf ("Error in mpfr_get_str (47d): s=%s e=%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_str_binary (x, "11111101111011000001010100001101101000010010001111E122620");
  s = mpfr_get_str (NULL, &e, 10, 17, x, MPFR_RNDN);
  if (strcmp (s, "22183435284042374") || e != 36928)
    {
      printf ("Error in mpfr_get_str (48n): s=%s e=%ld\n", s, (long) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 18, x, MPFR_RNDU);
  if (strcmp (s, "221834352840423736") || e != 36928)
    {
      printf ("Error in mpfr_get_str (48u): s=%s e=%ld\n", s, (long) e);
      exit (1);
    }
  mpfr_free_str (s);
  s = mpfr_get_str (NULL, &e, 10, 18, x, MPFR_RNDD);
  if (strcmp (s, "221834352840423735") || e != 36928)
    {
      printf ("Error in mpfr_get_str (48d): s=%s e=%ld\n", s, (long) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_prec (x, 45);
  mpfr_set_str_binary (x, "1E45");
  s = mpfr_get_str (NULL, &e, 32, 9, x, MPFR_RNDN);
  mpfr_free_str (s);

  mpfr_set_prec (x, 7);
  mpfr_set_str_binary (x, "0.1010101E10");
  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDU);
  mpfr_free_str (s);

  /* checks rounding of negative numbers */
  mpfr_set_prec (x, 7);
  mpfr_set_str (x, "-11.5", 10, MPFR_RNDN);
  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDD);
  if (strcmp (s, "-12"))
    {
      printf ("Error in mpfr_get_str for x=-11.5 and rnd=MPFR_RNDD\n"
              "got %s instead of -12\n", s);
      exit (1);
  }
  mpfr_free_str (s);

  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDU);
  if (strcmp (s, "-11"))
    {
      printf ("Error in mpfr_get_str for x=-11.5 and rnd=MPFR_RNDU\n");
      exit (1);
    }
  mpfr_free_str (s);

  /* bug found by Jean-Pierre Merlet, produced error in mpfr_get_str */
  mpfr_set_prec (x, 128);
  mpfr_set_str_binary (x, "0.10111001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011010E3");
  s = mpfr_get_str (NULL, &e, 10, 0, x, MPFR_RNDU);
  mpfr_free_str (s);

  mpfr_set_prec (x, 381);
  mpfr_set_str_binary (x, "0.111111111111111111111111111111111111111111111111111111111111111111101110110000100110011101101101001010111000101111000100100011110101010110101110100000010100001000110100000100011111001000010010000010001010111001011110000001110010111101100001111000101101100000010110000101100100000101010110010110001010100111001111100011100101100000100100111001100010010011110011011010110000001000010");
  s = mpfr_get_str (NULL, &e, 10, 0, x, MPFR_RNDD);
  if (e != 0)
    {
      printf ("Error in mpfr_get_str for x=0.999999..., exponent is %d"
              " instead of 0\n", (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_prec (x, 5);
  mpfr_set_str_binary (x, "1101.1"); /* 13.5, or (16)_7 + 1/2 */
  s = mpfr_get_str (NULL, &e, 7, 2, x, MPFR_RNDN);
  /* we are in the tie case: both surrounding numbers are (16)_7 and
     (20)_7: since (16)_7 = 13 is odd and (20)_7 = 14 is even,
     we should have s = "20" and e = 2 */
  if (e != 2 || strcmp (s, "20"))
    {
      printf ("Error in mpfr_get_str for x=13.5, base 7\n");
      printf ("Expected s=20, e=2, got s=%s, e=%ld\n", s, (long) e);
      exit (1);
    }
  mpfr_free_str (s);
  /* try the same example, with input just below or above 13.5 */
  mpfr_set_prec (x, 1000);
  mpfr_set_str_binary (x, "1101.1");
  mpfr_nextabove (x);
  s = mpfr_get_str (NULL, &e, 7, 2, x, MPFR_RNDN);
  if (e != 2 || strcmp (s, "20"))
    {
      printf ("Error in mpfr_get_str for x=13.5+tiny, base 7\n");
      printf ("Expected s=20, e=2, got s=%s, e=%ld\n", s, (long) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "1101.1");
  mpfr_nextbelow (x);
  s = mpfr_get_str (NULL, &e, 7, 2, x, MPFR_RNDN);
  if (e != 2 || strcmp (s, "16"))
    {
      printf ("Error in mpfr_get_str for x=13.5-tiny, base 7\n");
      printf ("Expected s=16, e=2, got s=%s, e=%ld\n", s, (long) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_prec (x, 7);
  mpfr_set_str_binary (x, "110000.1"); /* 48.5, or (66)_7 + 1/2 */
  s = mpfr_get_str (NULL, &e, 7, 2, x, MPFR_RNDN);
  /* we are in the tie case: both surrounding numbers are (66)_7 and
     (100)_7: since (66)_7 = 48 is even and (100)_7 is odd,
     we should hase s = "66" and e = 2 */
  if (e != 2 || strcmp (s, "66"))
    {
      printf ("Error in mpfr_get_str for x=48.5, base 7\n");
      printf ("Expected s=66, e=2, got s=%s, e=%ld\n", s, (long) e);
      exit (1);
    }
  mpfr_free_str (s);
  /* try the same example, with input just below or above 48.5 */
  mpfr_set_prec (x, 1000);
  mpfr_set_str_binary (x, "110000.1");
  mpfr_nextabove (x);
  s = mpfr_get_str (NULL, &e, 7, 2, x, MPFR_RNDN);
  if (e != 3 || strcmp (s, "10"))
    {
      printf ("Error in mpfr_get_str for x=48.5+tiny, base 7\n");
      printf ("Expected s=10, e=3, got s=%s, e=%ld\n", s, (long) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_set_str_binary (x, "110000.1");
  mpfr_nextbelow (x);
  s = mpfr_get_str (NULL, &e, 7, 2, x, MPFR_RNDN);
  if (e != 2 || strcmp (s, "66"))
    {
      printf ("Error in mpfr_get_str for x=48.5-tiny, base 7\n");
      printf ("Expected s=66, e=2, got s=%s, e=%ld\n", s, (long) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_clear (x);
}

/* bugs found by Alain Delplanque */
static void
check_large (void)
{
  mpfr_t x;
  char *s, s1[7];
  const char xm[] = { '1', '1', '9', '1', '3', '2', '9', '3', '7', '3',
                      '5', '8', '4', '4', '5', '4', '9', '0', '2', '9',
                      '6', '3', '4', '4', '6', '9', '9', '1', '9', '5',
                      '5', '7', '2', '0', '1', '7', '5', '2', '8', '6',
                      '1', '2', '5', '2', '5', '2', '7', '4', '0', '2',
                      '7', '9', '1', '1', '7', '4', '5', '6', '7', '5',
                      '9', '3', '1', '4', '2', '5', '5', '6', '6', '6',
                      '1', '6', '4', '3', '8', '1', '2', '8', '7', '6',
                      '2', '9', '2', '0', '8', '8', '9', '4', '3', '9',
                      '6', '2', '8', '4', '1', '1', '8', '1', '0', '6',
                      '2', '3', '7', '6', '3', '8', '1', '5', '1', '7',
                      '3', '4', '6', '1', '2', '4', '0', '1', '3', '0',
                      '8', '4', '1', '3', '9', '3', '2', '0', '1', '6',
                      '3', '6', '7', '1', '5', '1', '7', '5', '0', '1',
                      '9', '8', '4', '0', '8', '2', '7', '9', '1', '3',
                      '2', '2', '8', '3', '4', '1', '6', '2', '3', '9',
                      '6', '2', '0', '7', '3', '5', '5', '5', '3', '4',
                      '2', '1', '7', '0', '9', '7', '6', '2', '1', '0',
                      '3', '3', '5', '4', '7', '6', '0', '9', '7', '6',
                      '9', '3', '5', '1', '7', '8', '6', '8', '8', '2',
                      '8', '1', '4', '3', '7', '4', '3', '3', '2', '4',
                      '1', '5', '4', '7', '8', '1', '1', '4', '2', '1',
                      '2', '4', '2', '7', '6', '5', '9', '5', '4', '5',
                      '2', '6', '7', '3', '0', '3', '4', '0', '6', '9',
                      '1', '8', '9', '9', '9', '8', '0', '5', '7', '0',
                      '9', '3', '8', '7', '6', '2', '4', '6', '1', '6',
                      '7', '2', '0', '3', '5', '9', '3', '5', '8', '8',
                      '9', '7', '7', '9', '2', '7', '0', '8', '1', '6',
                      '8', '7', '4', '8', '5', '3', '0', '8', '4', '3',
                      '5', '6', '5', '1', '6', '6', '0', '9', '7', '9',
                      '8', '9', '2', '7', '2', '6', '8', '5', '9', '4',
                      '5', '8', '1', '3', '7', '2', '9', '3', '8', '3',
                      '7', '9', '1', '7', '9', '9', '7', '7', '2', '8',
                      '4', '6', '5', '5', '7', '3', '3', '8', '3', '6',
                      '6', '9', '7', '1', '4', '3', '3', '7', '1', '4',
                      '9', '4', '1', '2', '4', '9', '5', '1', '4', '7',
                      '2', '6', '4', '4', '8', '0', '6', '2', '6', '0',
                      '6', '9', '8', '1', '1', '7', '9', '9', '3', '9',
                      '3', '8', '4', '7', '3', '1', '9', '0', '2', '3',
                      '5', '3', '5', '4', '2', '1', '1', '7', '6', '7',
                      '4', '3', '2', '2', '0', '6', '5', '9', '9', '3',
                      '2', '6', '7', '1', '2', '0', '0', '3', '7', '3',
                      '8', '7', '4', '3', '3', '3', '3', '3', '2', '3',
                      '8', '2', '8', '6', '3', '1', '5', '5', '2', '2',
                      '5', '9', '3', '3', '7', '0', '6', '2', '8', '1',
                      '0', '3', '6', '7', '6', '9', '6', '5', '9', '0',
                      '6', '6', '6', '3', '6', '9', '9', '3', '8', '7',
                      '6', '5', '4', '5', '3', '5', '9', '4', '0', '0',
                      '7', '5', '8', '5', '4', '1', '4', '3', '1', '5',
                      '7', '6', '6', '3', '4', '4', '5', '0', '8', '7',
                      '5', '7', '5', '0', '1', '0', '1', '8', '4', '7',
                      '3', '1', '9', '9', '2', '7', '1', '1', '1', '2',
                      '3', '9', '9', '6', '5', '9', '2', '3', '2', '8',
                      '1', '5', '5', '1', '2', '6', '4', '9', '6', '6',
                      '4', '5', '1', '1', '6', '0', '0', '3', '2', '8',
                      '4', '8', '7', '1', '4', '9', '6', '8', '1', '6',
                      '5', '9', '8', '3', '4', '2', '9', '7', '0', '1',
                      '9', '2', '6', '6', '9', '1', '3', '5', '9', '3',
                      '2', '9', '6', '2', '3', '0', '6', '0', '1', '1',
                      '6', '5', '1', '7', '9', '0', '7', '5', '8', '6',
                      '8', '4', '2', '1', '0', '3', '8', '6', '6', '4',
                      '4', '9', '9', '7', '5', '8', '1', '7', '5', '7',
                      '9', '6', '6', '8', '8', '5', '8', '6', '7', '4',
                      '0', '7', '2', '0', '2', '9', '9', '4', '4', '1',
                      '9', '5', '8', '6', '5', '0', '6', '7', '4', '2',
                      '7', '3', '2', '3', '2', '7', '0', '2', '1', '3',
                      '0', '5', '9', '0', '3', '9', '1', '4', '5', '3',
                      '7', '2', '7', '0', '8', '5', '5', '4', '6', '1',
                      '1', '0', '0', '9', '2', '0', '4', '1', '6', '6',
                      '4', '6', '9', '1', '3', '2', '8', '5', '0', '3',
                      '3', '8', '9', '8', '7', '8', '5', '9', '5', '5',
                      '9', '1', '9', '3', '6', '5', '4', '1', '7', '4',
                      '0', '2', '4', '7', '2', '9', '7', '1', '2', '4',
                      '5', '8', '1', '4', '4', '6', '1', '8', '5', '8',
                      '7', '6', '9', '7', '2', '1', '2', '0', '8', '9',
                      '5', '9', '5', '5', '3', '8', '1', '2', '5', '4',
                      '3', '0', '7', '6', '5', '1', '7', '8', '2', '0',
                      '0', '7', '6', '7', '4', '8', '1', '0', '6', '3',
                      '2', '3', '0', '5', '2', '5', '0', '1', '1', '4',
                      '3', '8', '4', '5', '2', '3', '9', '5', '0', '9',
                      '8', '2', '6', '4', '7', '4', '8', '0', '1', '1',
                      '7', '1', '5', '4', '9', '0', '9', '2', '2', '3',
                      '8', '1', '6', '9', '0', '4', '6', '4', '5', '4',
                      '6', '3', '8', '7', '3', '6', '1', '7', '2', '3',
                      '4', '5', '5', '2', '0', '2', '5', '8', '1', '4',
                      '9', '3', '0', '7', '4', '1', '6', '8', '7', '8',
                      '2', '6', '2', '5', '1', '0', '7', '4', '7', '3',
                      '6', '6', '4', '5', '6', '6', '6', '6', '8', '5',
                      '1', '3', '5', '7', '1', '6', '2', '0', '9', '2',
                      '3', '2', '6', '0', '7', '9', '8', '1', '6', '2',
                      '0', '3', '8', '8', '0', '2', '8', '7', '7', '5',
                      '9', '3', '1', '0', '6', '7', '5', '7', '3', '1',
                      '2', '7', '7', '2', '0', '0', '4', '1', '2', '8',
                      '2', '0', '8', '4', '0', '5', '0', '5', '0', '1',
                      '9', '3', '3', '6', '3', '6', '9', '6', '2', '8',
                      '2', '9', '7', '5', '3', '8', '8', '9', '1', '1',
                      '4', '5', '7', '7', '5', '6', '0', '2', '7', '9',
                      '7', '2', '1', '7', '4', '3', '0', '3', '6', '7',
                      '3', '7', '2', '2', '7', '5', '6', '2', '3', '1',
                      '2', '1', '3', '1', '4', '2', '6', '9', '2', '3',
                      '\0' };
  mpfr_exp_t e;

  mpfr_init2 (x, 3322);
  mpfr_set_str (x, xm, 10, MPFR_RNDN);
  mpfr_div_2exp (x, x, 4343, MPFR_RNDN);
  s = mpfr_get_str (NULL, &e, 10, 1000, x, MPFR_RNDN);
  if (s[999] != '1') /* s must be 5.04383...689071e-309 */
    {
      printf ("Error in check_large: expected '689071', got '%s'\n",
              s + 994);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_mul_2exp (x, x, 4343, MPFR_RNDN);
  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDN);
  if (strcmp (s, "12") || (e != 1000))
    {
      printf ("Error in check_large: expected 0.12e1000\n");
      printf ("got %se%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_set_nan (x);
  s = mpfr_get_str (NULL, &e, 10, 1000, x, MPFR_RNDN);
  if (strcmp (s, "@NaN@"))
    {
      printf ("Error for NaN\n");
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_get_str (s1, &e, 10, 1000, x, MPFR_RNDN);

  mpfr_set_inf (x, 1);
  s = mpfr_get_str (NULL, &e, 10, 1000, x, MPFR_RNDN);
  if (strcmp (s, "@Inf@"))
    {
      printf ("Error for Inf\n");
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_get_str (s1, &e, 10, 1000, x, MPFR_RNDN);

  mpfr_set_inf (x, -1);
  s = mpfr_get_str (NULL, &e, 10, 1000, x, MPFR_RNDN);
  if (strcmp (s, "-@Inf@"))
    {
      printf ("Error for -Inf\n");
      exit (1);
    }
  mpfr_free_str (s);

  mpfr_get_str (s1, &e, 10, 1000, x, MPFR_RNDN);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDN);
  if (e != 0 || strcmp (s, "00"))
    {
      printf ("Error for 0.0\n");
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_get_str (s1, &e, 10, 2, x, MPFR_RNDN);

  mpfr_neg (x, x, MPFR_RNDN); /* -0.0 */
  s = mpfr_get_str (NULL, &e, 10, 2, x, MPFR_RNDN);
  if (e != 0 || strcmp (s, "-00"))
    {
      printf ("Error for -0.0\ngot %se%d\n", s, (int) e);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_get_str (s1, &e, 10, 2, x, MPFR_RNDN);

  mpfr_clear (x);
}

#define MAX_DIGITS 100

static void
check_special (int b, mpfr_prec_t p)
{
  mpfr_t x;
  int i, j;
  char s[MAX_DIGITS + 2], s2[MAX_DIGITS + 2], c;
  mpfr_exp_t e;
  int r;
  size_t m;

  /* check for invalid base */
  if (mpfr_get_str (s, &e, 1, 10, x, MPFR_RNDN) != NULL)
    {
      printf ("Error: mpfr_get_str should not accept base = 1\n");
      exit (1);
    }
  if (mpfr_get_str (s, &e, 63, 10, x, MPFR_RNDN) != NULL)
    {
      printf ("Error: mpfr_get_str should not accept base = 63\n");
      exit (1);
    }

  s2[0] = '1';
  for (i=1; i<MAX_DIGITS+2; i++)
    s2[i] = '0';

  mpfr_init2 (x, p);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  for (i=1; i<MAX_DIGITS && mpfr_mul_ui (x, x, b, MPFR_RNDN) == 0; i++)
    {
      /* x = b^i (exact) */
      for (r = 0; r < MPFR_RND_MAX; r++)
        for (m= (i<3)? 2 : i-1 ; (int) m <= i+1 ; m++)
          {
            mpfr_get_str (s, &e, b, m, x, (mpfr_rnd_t) r);
            /* s should be 1 followed by (m-1) zeros, and e should be i+1 */
            if ((e != i+1) || strncmp (s, s2, m) != 0)
              {
                printf ("Error in mpfr_get_str for %d^%d\n", b, i);
                exit (1);
              }
          }
      if (mpfr_sub_ui (x, x, 1, MPFR_RNDN) != 0)
        break;
      /* now x = b^i-1 (exact) */
      for (r = 0; r < MPFR_RND_MAX; r++)
        if (i >= 2)
          {
            mpfr_get_str (s, &e, b, i, x, (mpfr_rnd_t) r);
            /* should be i times (b-1) */
            c = (b <= 10) ? '0' + b - 1 : 'a' + (b - 11);
            for (j=0; (j < i) && (s[j] == c); j++);
            if ((j < i) || (e != i))
              {
                printf ("Error in mpfr_get_str for %d^%d-1\n", b, i);
                printf ("got 0.%s*2^%d\n", s, (int) e);
                exit (1);
              }
          }
      if (i >= 3)
        {
          mpfr_get_str (s, &e, b, i - 1, x, MPFR_RNDU);
          /* should be b^i */
          if ((e != i+1) || strncmp (s, s2, i - 1) != 0)
            {
              printf ("Error in mpfr_get_str for %d^%d-1\n", b, i);
              printf ("got 0.%s*2^%d\n", s, (int) e);
              exit (1);
            }
        }

      mpfr_add_ui (x, x, 1, MPFR_RNDN);
    }
  mpfr_clear (x);
}

static void
check_bug_base2k (void)
{
  /*
   * -2.63b22b55697e800000000000@130
   * +-0.1001100011101100100010101101010101011010010111111010000000000000000000000000+00000000000000000000001E522
  */
  mpfr_t xx, yy, zz;
  char *s;
  mpfr_exp_t e;

  mpfr_init2 (xx, 107);
  mpfr_init2 (yy, 79);
  mpfr_init2 (zz, 99);

  mpfr_set_str (xx, "-1.90e8c3e525d7c0000000000000@-18", 16, MPFR_RNDN);
  mpfr_set_str (yy, "-2.63b22b55697e8000000@130", 16, MPFR_RNDN);
  mpfr_add (zz, xx, yy, MPFR_RNDD);
  s = mpfr_get_str (NULL, &e, 16, 0, zz, MPFR_RNDN);
  if (strcmp (s, "-263b22b55697e8000000000008"))
    {
      printf ("Error for get_str base 16\n"
              "Got %s expected -263b22b55697e8000000000008\n", s);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_clears (xx, yy, zz, (mpfr_ptr) 0);
}

static void
check_reduced_exprange (void)
{
  mpfr_t x;
  char *s;
  mpfr_exp_t emax, e;

  emax = mpfr_get_emax ();
  mpfr_init2 (x, 8);
  mpfr_set_str (x, "0.11111111E0", 2, MPFR_RNDN);
  set_emax (0);
  s = mpfr_get_str (NULL, &e, 16, 0, x, MPFR_RNDN);
  set_emax (emax);
  if (strcmp (s, "ff0"))
    {
      printf ("Error for mpfr_get_str on 0.11111111E0 in base 16:\n"
              "Got \"%s\" instead of \"ff0\".\n", s);
      exit (1);
    }
  mpfr_free_str (s);
  mpfr_clear (x);
}

#define ITER 1000

int
main (int argc, char *argv[])
{
  int b;
  mpfr_t x;
  mpfr_rnd_t r;
  char s[MAX_DIGITS + 2];
  mpfr_exp_t e, f;
  size_t m;
  mpfr_prec_t p;
  int i;

  tests_start_mpfr ();

  check_small ();

  check_special (2, 2);
  for (i = 0; i < ITER; i++)
    {
      p = 2 + (randlimb () % (MAX_DIGITS - 1));
      b = 2 + (randlimb () % 35);
      check_special (b, p);
    }

  mpfr_init2 (x, MAX_DIGITS);
  for (i = 0; i < ITER; i++)
    {
      m = 2 + (randlimb () % (MAX_DIGITS - 1));
      mpfr_urandomb (x, RANDS);
      e = (mpfr_exp_t) (randlimb () % 21) - 10;
      mpfr_set_exp (x, (e == -10) ? mpfr_get_emin () :
                    ((e == 10) ? mpfr_get_emax () : e));
      b = 2 + (randlimb () % 35);
      r = RND_RAND ();
      mpfr_get_str (s, &f, b, m, x, r);
    }
  mpfr_clear (x);

  check_large ();
  check3 ("4.059650008e-83", MPFR_RNDN, "40597");
  check3 ("-6.606499965302424244461355e233", MPFR_RNDN, "-66065");
  check3 ("-7.4", MPFR_RNDN, "-74000");
  check3 ("0.997", MPFR_RNDN, "99700");
  check3 ("-4.53063926135729747564e-308", MPFR_RNDN, "-45306");
  check3 ("2.14478198760196000000e+16", MPFR_RNDN, "21448");
  check3 ("7.02293374921793516813e-84", MPFR_RNDN, "70229");

  check3 ("-6.7274500420134077e-87", MPFR_RNDN, "-67275");
  check3 ("-6.7274500420134077e-87", MPFR_RNDZ, "-67274");
  check3 ("-6.7274500420134077e-87", MPFR_RNDU, "-67274");
  check3 ("-6.7274500420134077e-87", MPFR_RNDD, "-67275");
  check3 ("-6.7274500420134077e-87", MPFR_RNDA, "-67275");

  check3 ("6.7274500420134077e-87", MPFR_RNDN, "67275");
  check3 ("6.7274500420134077e-87", MPFR_RNDZ, "67274");
  check3 ("6.7274500420134077e-87", MPFR_RNDU, "67275");
  check3 ("6.7274500420134077e-87", MPFR_RNDD, "67274");
  check3 ("6.7274500420134077e-87", MPFR_RNDA, "67275");

  check_bug_base2k ();
  check_reduced_exprange ();

  tests_end_mpfr ();
  return 0;
}
