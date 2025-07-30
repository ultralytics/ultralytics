/* Test file for mpfr_set_str.

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

#define N 30000

#define CHECK53(y, s, r, x, t, n) \
  mpfr_set_str (y, s, 10, r); \
  mpfr_set_str_binary (x, t); \
  if (mpfr_cmp (x, y)) \
    { \
      printf ("Error in mpfr_set_str (%d):\n", n); \
      mpfr_print_binary (x); \
      puts (""); \
      mpfr_print_binary (y); \
      puts (""); \
      mpfr_clear (x); \
      mpfr_clear (y); \
      exit (1); \
    }

static void
check_underflow (void)
{
  mpfr_t a;
  mpfr_exp_t emin, emax;
  int res;

  mpfr_init (a);

  /* Check underflow */
  emin = mpfr_get_emin ();
  set_emin (-20);
  res = mpfr_set_str (a, "0.00000000001", 10, MPFR_RNDZ);
  if (!MPFR_IS_ZERO (a))
    {
      printf("ERROR for mpfr_set_str (a, \"0.00000000001\", 10, MPFR_RNDN)\n"
             " with emin=-20\n"
             "res=%d\n", res);
      mpfr_dump (a);
      exit (1);
    }
  set_emin (emin);

  /* check overflow */
  emax = mpfr_get_emax ();
  set_emax (1073741823); /* 2^30-1 */
  mpfr_set_str (a, "2E1000000000", 10, MPFR_RNDN);
  if (!mpfr_inf_p (a) || mpfr_sgn (a) < 0)
    {
      printf("ERROR for mpfr_set_str (a, \"2E1000000000\", 10, MPFR_RNDN);\n");
      exit (1);
    }
  set_emax (emax);

  mpfr_clear (a);
}

/* Bug found by Christoph Lauter. */
static void
bug20081028 (void)
{
  mpfr_t x;
  const char *s = "0.10000000000000000000000000000001E1";

  mpfr_init2 (x, 32);
  mpfr_set_str (x, "1.00000000000000000006", 10, MPFR_RNDU);
  if (! mpfr_greater_p (x, __gmpfr_one))
    {
      printf ("Error in bug20081028:\nExpected %s\nGot      ", s);
      mpfr_dump (x);
      exit (1);
    }
  mpfr_clear (x);
}

int
main (int argc, char *argv[])
{
  mpfr_t x, y;
  unsigned long k, bd, nc, i;
  char *str, *str2;
  mpfr_exp_t e;
  int base, logbase, prec, baseprec, ret, obase;

  tests_start_mpfr ();

  if (argc >= 2) /* tset_str <string> [<prec>] [<ibase>] [<obase>] */
    {
      prec = (argc >= 3) ? atoi (argv[2]) : 53;
      base = (argc >= 4) ? atoi (argv[3]) : 2;
      obase = (argc >= 5) ? atoi (argv[4]) : 10;
      mpfr_init2 (x, prec);
      mpfr_set_str (x, argv[1], base, MPFR_RNDN);
      mpfr_out_str (stdout, obase, 0, x, MPFR_RNDN);
      puts ("");
      mpfr_clear (x);
      return 0;
    }

  mpfr_init2 (x, 2);

  nc = (argc > 1) ? atoi(argv[1]) : 53;
  if (nc < 100)
    nc = 100;

  bd = randlimb () & 8;

  str2 = str = (char*) tests_allocate (nc);

  if (bd)
    {
      for(k = 1; k <= bd; k++)
        *(str2++) = (randlimb () & 1) + '0';
    }
  else
    *(str2++) = '0';

  *(str2++) = '.';

  for (k = 1; k < nc - 17 - bd; k++)
    *(str2++) = '0' + (char) (randlimb () & 1);

  *(str2++) = 'e';
  sprintf (str2, "%d", (int) (randlimb () & INT_MAX) + INT_MIN/2);

  mpfr_set_prec (x, nc + 10);
  mpfr_set_str_binary (x, str);

  mpfr_set_prec (x, 54);
  mpfr_set_str_binary (x, "0.100100100110110101001010010101111000001011100100101010E-529");
  mpfr_init2 (y, 54);
  mpfr_set_str (y, "4.936a52bc17254@-133", 16, MPFR_RNDN);
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (1a):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str_binary (x, "0.111111101101110010111010100110000111011001010100001101E-529");
  mpfr_set_str (y, "0.fedcba98765434P-529", 16, MPFR_RNDN);
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (1b):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  tests_free (str, nc);

  mpfr_set_prec (x, 53);
  mpfr_set_str_binary (x, "+110101100.01010000101101000000100111001000101011101110E00");

  mpfr_set_str_binary (x, "1.0");
  if (mpfr_cmp_ui (x, 1))
    {
      printf ("Error in mpfr_set_str_binary for s=1.0\n");
      mpfr_clear(x);
      mpfr_clear(y);
      exit(1);
    }

  mpfr_set_str_binary (x, "+0000");
  mpfr_set_str_binary (x, "+0000E0");
  mpfr_set_str_binary (x, "0000E0");
  if (mpfr_cmp_ui (x, 0))
    {
      printf ("Error in mpfr_set_str_binary for s=0.0\n");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (x, "+243495834958.53452345E1", 10, MPFR_RNDN);
  mpfr_set_str (x, "9007199254740993", 10, MPFR_RNDN);
  mpfr_set_str (x, "9007199254740992", 10, MPFR_RNDU);
  mpfr_set_str (x, "9007199254740992", 10, MPFR_RNDD);
  mpfr_set_str (x, "9007199254740992", 10, MPFR_RNDZ);

  /* check a random number printed and read is not modified */
  prec = 53;
  mpfr_set_prec (x, prec);
  mpfr_set_prec (y, prec);
  for (i=0;i<N;i++)
    {
      mpfr_rnd_t rnd;

      mpfr_urandomb (x, RANDS);
      rnd = RND_RAND ();
      logbase = (randlimb () % 5) + 1;
      base = 1 << logbase;
      /* Warning: the number of bits needed to print exactly a number of
         'prec' bits in base 2^logbase may be greater than ceil(prec/logbase),
         for example 0.11E-1 in base 2 cannot be written exactly with only
         one digit in base 4 */
      if (base == 2)
        baseprec = prec;
      else
        baseprec = 1 + (prec - 2 + logbase) / logbase;
      str = mpfr_get_str (NULL, &e, base, baseprec, x, rnd);
      mpfr_set_str (y, str, base, rnd);
      MPFR_EXP(y) += logbase * (e - strlen (str));
      if (mpfr_cmp (x, y))
        {
          printf ("mpfr_set_str o mpfr_get_str <> id for rnd_mode=%s\n",
                  mpfr_print_rnd_mode (rnd));
          printf ("x=");
          mpfr_print_binary (x);
          puts ("");
          printf ("s=%s, exp=%d, base=%d\n", str, (int) e, base);
          printf ("y=");
          mpfr_print_binary (y);
          puts ("");
          mpfr_clear (x);
          mpfr_clear (y);
          exit (1);
        }
      mpfr_free_str (str);
    }

  for (i = 2; i <= 62; i++)
    {
      if (mpfr_set_str (x, "@NaN@(garbage)", i, MPFR_RNDN) != 0 ||
          !mpfr_nan_p(x))
        {
          printf ("mpfr_set_str failed on @NaN@(garbage)\n");
          exit (1);
        }

      /*
      if (mpfr_set_str (x, "@Inf@garbage", i, MPFR_RNDN) != 0 ||
          !mpfr_inf_p(x) || MPFR_SIGN(x) < 0)
        {
          printf ("mpfr_set_str failed on @Inf@garbage\n");
          exit (1);
        }

      if (mpfr_set_str (x, "-@Inf@garbage", i, MPFR_RNDN) != 0 ||
          !mpfr_inf_p(x) || MPFR_SIGN(x) > 0)
        {
          printf ("mpfr_set_str failed on -@Inf@garbage\n");
          exit (1);
        }

      if (mpfr_set_str (x, "+@Inf@garbage", i, MPFR_RNDN) != 0 ||
          !mpfr_inf_p(x) || MPFR_SIGN(x) < 0)
        {
          printf ("mpfr_set_str failed on +@Inf@garbage\n");
          exit (1);
        }
      */

      if (i > 16)
        continue;

      if (mpfr_set_str (x, "NaN", i, MPFR_RNDN) != 0 ||
          !mpfr_nan_p(x))
        {
          printf ("mpfr_set_str failed on NaN\n");
          exit (1);
        }

      if (mpfr_set_str (x, "Inf", i, MPFR_RNDN) != 0 ||
          !mpfr_inf_p(x) || MPFR_SIGN(x) < 0)
        {
          printf ("mpfr_set_str failed on Inf\n");
          exit (1);
        }

      if (mpfr_set_str (x, "-Inf", i, MPFR_RNDN) != 0 ||
          !mpfr_inf_p(x) || MPFR_SIGN(x) > 0)
        {
          printf ("mpfr_set_str failed on -Inf\n");
          exit (1);
        }

      if (mpfr_set_str (x, "+Inf", i, MPFR_RNDN) != 0 ||
          !mpfr_inf_p(x) || MPFR_SIGN(x) < 0)
        {
          printf ("mpfr_set_str failed on +Inf\n");
          exit (1);
        }
    }

  /* check that mpfr_set_str works for uppercase letters too */
  mpfr_set_prec (x, 10);
  mpfr_set_str (x, "B", 16, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 11) != 0)
    {
      printf ("mpfr_set_str does not work for uppercase letters\n");
      exit (1);
    }

  /* start of tests added by Alain Delplanque */

  /* in this example an overflow can occur */
  mpfr_set_prec (x, 64);
  mpfr_set_prec (y, 64);
  mpfr_set_str_binary (x, "1.0E-532");
  mpfr_set_str (y, "0.71128279983522479470@-160", 10, MPFR_RNDU);
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (2):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  /* in this example, I think there was a pb in the old function :
     result of mpfr_set_str_old for the same number , but with more
     precision is: 1.111111111110000000000000000111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100111000100001100000010101100111010e184
     this result is the same as mpfr_set_str */
  mpfr_set_prec (x, 64);
  mpfr_set_prec (y, 64);
  mpfr_set_str_binary (x, "1.111111111110000000000000000111111111111111111111111110000000001E184");
  mpfr_set_str (y, "0.jo08hg31hc5mmpj5mjjmgn55p2h35g@39", 27, MPFR_RNDU);
  /* y = 49027884868983130654865109690613178467841148597221480052 */
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (3):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  /* not exact rounding in mpfr_set_str
     same number with more precision is : 1.111111111111111111111111111000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011011111101000001101110110010101101000010100110011101110010001110e195
     this result is the same as mpfr_set_str */
  /* problem was : can_round was call with MPFR_RNDN round mode,
     so can_round use an error : 1/2 * 2^err * ulp(y)
     instead of 2^err * ulp(y)
     I have increase err by 1 */
  mpfr_set_prec (x, 64);  /* it was round down instead of up */
  mpfr_set_prec (y, 64);
  mpfr_set_str_binary (x, "1.111111111111111111111111111000000000000000000000000000000000001e195");
  mpfr_set_str (y, "0.6e23ekb6acgh96abk10b6c9f2ka16i@45", 21, MPFR_RNDU);
  /* y = 100433627392042473064661483711179345482301462325708736552078 */
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (4):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  /* may be an error in mpfr_set_str_old
     with more precision : 1.111111100000001111110000000000011111011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110111101010001110111011000010111001011100110110e180 */
  mpfr_set_prec (x, 64);  /* it was round down instead of up */
  mpfr_set_prec (y, 64);
  mpfr_set_str_binary (x, "1.111111100000001111110000000000011111011111111111111111111111111e180");
  mpfr_set_str (y, "0.10j8j2k82ehahha56390df0a1de030@41", 23, MPFR_RNDZ);
  /* y = 3053110535624388280648330929253842828159081875986159414 */
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (5):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_prec (x, 64);
  mpfr_set_prec (y, 64);
  mpfr_set_str (y, "0.jrchfhpp9en7hidqm9bmcofid9q3jg@39", 28, MPFR_RNDU);
  /* y = 196159429139499688661464718784226062699788036696626429952 */
  mpfr_set_str_binary (x, "0.1111111111111111111111111111111000000000000011100000001111100001E187");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (6):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_prec (x, 64);
  mpfr_set_prec (y, 64);
  mpfr_set_str (y, "0.h148m5ld5cf8gk1kd70b6ege92g6ba@47", 24, MPFR_RNDZ);
  /* y = 52652933527468502324759448399183654588831274530295083078827114496 */
  mpfr_set_str_binary (x, "0.1111111111111100000000001000000000000000000011111111111111101111E215");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (7):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  /* worst cases for rounding to nearest in double precision */
  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);

  mpfr_set_str (y, "5e125", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.10111101000101110110011000100000101001010000000111111E418");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (8):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (y, "69e267", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.10000101101111100101101100000110010011001010011011010E894");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (9):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (y, "623e100", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.10110010000001010011000101111001110101000001111011111E342");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (10):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (y, "3571e263", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.10110001001100100010011000110000111010100000110101010E886");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (11):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (y, "75569e-254", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.10101101001000110001011011001000111000110101010110011E-827");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (12):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (y, "920657e-23", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.10101001110101001100110000101110110111101111001101100E-56");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (13):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (y, "9210917e80", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.11101101000100011001000110100011111100110000000110010E289");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (14):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (y, "87575437e-309", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.11110000001110011001000000110000000100000010101101100E-1000");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (15):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (y, "245540327e122", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.10001101101100010001100011110000110001100010111001011E434");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (16):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (y, "491080654e122", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.10001101101100010001100011110000110001100010111001011E435");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (17):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  mpfr_set_str (y, "83356057653e193", 10, MPFR_RNDN);
  mpfr_set_str_binary (x, "0.10101010001001110011011011010111011100010101000011000E678");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_set_str (18):\n");
      mpfr_print_binary (x);
      puts ("");
      mpfr_print_binary (y);
      puts ("");
      mpfr_clear (x);
      mpfr_clear (y);
      exit (1);
    }

  CHECK53(y, "83356057653e193", MPFR_RNDN, x,
          "0.10101010001001110011011011010111011100010101000011000E678",
          18);

  CHECK53(y, "619534293513e124", MPFR_RNDN, x,
          "0.10001000011000010000000110000001111111110000011110001e452",
          19);

  CHECK53(y, "3142213164987e-294", MPFR_RNDN, x,
          "0.11101001101000000100111011111101111001010001001101111e-935",
          20);

  CHECK53(y, "36167929443327e-159", MPFR_RNDN, x,
          "0.11100111001110111110000101011001100110010100011111100e-483",
          21);

  CHECK53(y, "904198236083175e-161", MPFR_RNDN, x,
          "0.11100111001110111110000101011001100110010100011111100e-485",
          22);

  CHECK53(y, "3743626360493413e-165", MPFR_RNDN, x,
          "0.11000100000100011101001010111101011011011111011111001e-496",
          23);

  CHECK53(y, "94080055902682397e-242", MPFR_RNDN, x,
          "0.10110010010011000000111100011100111100110011011001010e-747",
          24);

  CHECK53(y, "7e-303", MPFR_RNDD, x,
          "0.10011001100111001000100110001110001000110111110001011e-1003",
          25);
  CHECK53(y, "7e-303", MPFR_RNDU, x,
          "0.10011001100111001000100110001110001000110111110001100e-1003",
          26);

  CHECK53(y, "93e-234", MPFR_RNDD, x,
          "0.10010011110110010111001001111001000010000000001110101E-770",
          27);
  CHECK53(y, "93e-234", MPFR_RNDU, x,
          "0.10010011110110010111001001111001000010000000001110110E-770",
          28);

  CHECK53(y, "755e174", MPFR_RNDD, x,
          "0.10111110110010011000110010011111101111000111111000101E588",
          29);
  CHECK53(y, "755e174", MPFR_RNDU, x,
          "0.10111110110010011000110010011111101111000111111000110E588",
          30);

  CHECK53(y, "8699e-276", MPFR_RNDD, x,
          "0.10010110100101101111100100100011011101100110100101100E-903",
          31);
  CHECK53(y, "8699e-276", MPFR_RNDU, x,
          "0.10010110100101101111100100100011011101100110100101101E-903",
          32);

  CHECK53(y, "82081e41", MPFR_RNDD, x,
          "0.10111000000010000010111011111001111010100011111001011E153",
          33);
  CHECK53(y, "82081e41", MPFR_RNDU, x,
          "0.10111000000010000010111011111001111010100011111001100E153",
          34);

  CHECK53(y, "584169e229", MPFR_RNDD, x,
          "0.11101011001010111000001011001110111000111100110101010E780",
          35);
  CHECK53(y, "584169e229", MPFR_RNDU, x,
          "0.11101011001010111000001011001110111000111100110101011E780",
          36);

  CHECK53(y, "5783893e-128", MPFR_RNDD, x,
          "0.10011000111100000110011110000101100111110011101110100E-402",
          37);
  CHECK53(y, "5783893e-128", MPFR_RNDU, x,
          "0.10011000111100000110011110000101100111110011101110101E-402",
          38);

  CHECK53(y, "87575437e-310", MPFR_RNDD, x,
          "0.11000000001011100000110011110011010000000010001010110E-1003",
          39);
  CHECK53(y, "87575437e-310", MPFR_RNDU, x,
          "0.11000000001011100000110011110011010000000010001010111E-1003",
          40);

  CHECK53(y, "245540327e121", MPFR_RNDD, x,
          "0.11100010101101001111010010110100011100000100101000100E430",
          41);
  CHECK53(y, "245540327e121", MPFR_RNDU, x,
          "0.11100010101101001111010010110100011100000100101000101E430",
          42);

  CHECK53(y, "9078555839e-109", MPFR_RNDD, x,
          "0.11111110001010111010110000110011100110001010011101101E-329",
          43);
  CHECK53(y, "9078555839e-109", MPFR_RNDU, x,
          "0.11111110001010111010110000110011100110001010011101110E-329",
          44);

  CHECK53(y, "42333842451e201", MPFR_RNDD, x,
          "0.10000000110001001101000100110110111110101011101011111E704",
          45);
  CHECK53(y, "42333842451e201", MPFR_RNDU, x,
          "0.10000000110001001101000100110110111110101011101100000E704",
          46);

  CHECK53(y, "778380362293e218", MPFR_RNDD, x,
          "0.11001101010111000001001100001100110010000001010010010E764",
          47);
  CHECK53(y, "778380362293e218", MPFR_RNDU, x,
          "0.11001101010111000001001100001100110010000001010010011E764",
          48);

  CHECK53(y, "7812878489261e-179", MPFR_RNDD, x,
          "0.10010011011011010111001111011101111101101101001110100E-551",
          49);
  CHECK53(y, "7812878489261e-179", MPFR_RNDU, x,
          "0.10010011011011010111001111011101111101101101001110101E-551",
          50);

  CHECK53(y, "77003665618895e-73", MPFR_RNDD, x,
          "0.11000101111110111111001111111101001101111000000101001E-196",
          51);
  CHECK53(y, "77003665618895e-73", MPFR_RNDU, x,
          "0.11000101111110111111001111111101001101111000000101010E-196",
          52);

  CHECK53(y, "834735494917063e-300", MPFR_RNDD, x,
          "0.11111110001101100001001101111100010011001110111010001E-947",
          53);
  CHECK53(y, "834735494917063e-300", MPFR_RNDU, x,
          "0.11111110001101100001001101111100010011001110111010010E-947",
          54);

  CHECK53(y, "6182410494241627e-119", MPFR_RNDD, x,
          "0.10001101110010110010001011000010001000101110100000111E-342",
          55);
  CHECK53(y, "6182410494241627e-119", MPFR_RNDU, x,
          "0.10001101110010110010001011000010001000101110100001000E-342",
          56);

  CHECK53(y, "26153245263757307e49", MPFR_RNDD, x,
          "0.10011110111100000000001011011110101100010000011011110E218",
          57);
  CHECK53(y, "26153245263757307e49", MPFR_RNDU, x,
          "0.10011110111100000000001011011110101100010000011011111E218",
          58);

  /* to check this problem : I convert limb (10--0 or 101--1) into base b
     with more than mp_bits_per_limb digits,
     so when convert into base 2 I should have
     the limb that I have choose */
  /* this use mpfr_get_str */
  {
    size_t nb_digit = mp_bits_per_limb;
    mp_limb_t check_limb[2] = {MPFR_LIMB_HIGHBIT, ~(MPFR_LIMB_HIGHBIT >> 1)};
    int base[3] = {10, 16, 19};
    mpfr_rnd_t rnd[3] = {MPFR_RNDU, MPFR_RNDN, MPFR_RNDD};
    int cbase, climb, crnd;
    char *str;

    mpfr_set_prec (x, mp_bits_per_limb); /* x and y have only one limb */
    mpfr_set_prec (y, mp_bits_per_limb);

    str = (char *) tests_allocate (N + 20);

    mpfr_set_ui (x, 1, MPFR_RNDN); /* ensures that x is not NaN or Inf */
    for (; nb_digit < N; nb_digit *= 10)
      for (cbase = 0; cbase < 3; cbase++)
        for (climb = 0; climb < 2; climb++)
          for (crnd = 0; crnd < 3; crnd++)
            {
              char *str1;
              mpfr_exp_t exp;

              *(MPFR_MANT(x)) = check_limb[climb];
              MPFR_EXP(x) = 0;

              mpfr_get_str (str + 2, &exp, base[cbase],
                            nb_digit, x, rnd[crnd]);
              str[0] = '-';
              str[(str[2] == '-')] =  '0';
              str[(str[2] == '-') + 1] =  '.';

              for (str1 = str; *str1 != 0; str1++)
                ;
              sprintf (str1, "@%i", (int) exp);

              mpfr_set_str (y, str, base[cbase], rnd[2 - crnd]);

              if (mpfr_cmp (x, y) != 0)
                {
                  printf ("Error in mpfr_set_str for nb_digit=%u, base=%d, "
                          "rnd=%s:\n", (unsigned int) nb_digit, base[cbase],
                          mpfr_print_rnd_mode (rnd[crnd]));
                  printf ("instead of: ");
                  mpfr_print_binary (x);
                  puts ("");
                  printf ("return    : ");
                  mpfr_print_binary (y);
                  puts ("");
                  exit (1);
                }
            }

    tests_free (str, N + 20);
  }

  /* end of tests added by Alain Delplanque */

  /* check that flags are correctly cleared */
  mpfr_set_nan (x);
  mpfr_set_str (x, "+0.0", 10, MPFR_RNDN);
  if (!mpfr_number_p(x) || mpfr_cmp_ui (x, 0) != 0 || mpfr_sgn (x) < 0)
    {
      printf ("x <- +0.0 failed after x=NaN\n");
      exit (1);
    }
  mpfr_set_str (x, "-0.0", 10, MPFR_RNDN);
  if (!mpfr_number_p(x) || mpfr_cmp_ui (x, 0) != 0 || mpfr_sgn (x) > 0)
    {
      printf ("x <- -0.0 failed after x=NaN\n");
      exit (1);
    }

  /* check invalid input */
  ret = mpfr_set_str (x, "1E10toto", 10, MPFR_RNDN);
  MPFR_ASSERTN (ret == -1);
  ret = mpfr_set_str (x, "1p10toto", 16, MPFR_RNDN);
  MPFR_ASSERTN (ret == -1);
  ret = mpfr_set_str (x, "", 16, MPFR_RNDN);
  MPFR_ASSERTN (ret == -1);
  ret = mpfr_set_str (x, "+", 16, MPFR_RNDN);
  MPFR_ASSERTN (ret == -1);
  ret = mpfr_set_str (x, "-", 16, MPFR_RNDN);
  MPFR_ASSERTN (ret == -1);
  ret = mpfr_set_str (x, "this_is_an_invalid_number_in_base_36", 36, MPFR_RNDN);
  MPFR_ASSERTN (ret == -1);
  ret = mpfr_set_str (x, "1.2.3", 10, MPFR_RNDN);
  MPFR_ASSERTN (ret == -1);
  mpfr_set_prec (x, 135);
  ret = mpfr_set_str (x, "thisisavalidnumberinbase36", 36, MPFR_RNDN);
  mpfr_set_prec (y, 135);
  mpfr_set_str (y, "23833565676460972739462619524519814462546", 10, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp (x, y) == 0 && ret == 0);

  /* coverage test for set_str_binary */
  mpfr_set_str_binary (x, "NaN");
  MPFR_ASSERTN(mpfr_nan_p (x));
  mpfr_set_str_binary (x, "Inf");
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);
  mpfr_set_str_binary (x, "+Inf");
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) > 0);
  mpfr_set_str_binary (x, "-Inf");
  MPFR_ASSERTN(mpfr_inf_p (x) && mpfr_sgn (x) < 0);
  mpfr_set_prec (x, 3);
  mpfr_set_str_binary (x, "0.01E2");
  MPFR_ASSERTN(mpfr_cmp_ui (x, 1) == 0);
  mpfr_set_str_binary (x, "-0.01E2");
  MPFR_ASSERTN(mpfr_cmp_si (x, -1) == 0);

  mpfr_clear (x);
  mpfr_clear (y);

  check_underflow ();
  bug20081028 ();

  tests_end_mpfr ();
  return 0;
}
