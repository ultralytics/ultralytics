/* tzeta -- test file for the Riemann Zeta function

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

static void
test1 (void)
{
  mpfr_t x, y;

  mpfr_init2 (x, 32);
  mpfr_init2 (y, 42);

  mpfr_set_str_binary (x, "1.1111111101000111011010010010100e-1");
  mpfr_zeta (y, x, MPFR_RNDN); /* shouldn't crash */

  mpfr_set_prec (x, 40);
  mpfr_set_prec (y, 50);
  mpfr_set_str_binary (x, "1.001101001101000010011010110100110000101e-1");
  mpfr_zeta (y, x, MPFR_RNDU);
  mpfr_set_prec (x, 50);
  mpfr_set_str_binary (x, "-0.11111100011100111111101111100011110111001111111111E1");
  if (mpfr_cmp (x, y))
    {
      printf ("Error for input on 40 bits, output on 50 bits\n");
      printf ("Expected "); mpfr_print_binary (x); puts ("");
      printf ("Got      "); mpfr_print_binary (y); puts ("");
      mpfr_set_str_binary (x, "1.001101001101000010011010110100110000101e-1");
      mpfr_zeta (y, x, MPFR_RNDU);
      mpfr_print_binary (x); puts ("");
      mpfr_print_binary (y); puts ("");
      exit (1);
    }

  mpfr_set_prec (x, 2);
  mpfr_set_prec (y, 55);
  mpfr_set_str_binary (x, "0.11e3");
  mpfr_zeta (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 55);
  mpfr_set_str_binary (x, "0.1000001000111000010011000010011000000100100100100010010E1");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_zeta (1)\n");
      printf ("Expected "); mpfr_print_binary (x); puts ("");
      printf ("Got      "); mpfr_print_binary (y); puts ("");
      exit (1);
    }

  mpfr_set_prec (x, 3);
  mpfr_set_prec (y, 47);
  mpfr_set_str_binary (x, "0.111e4");
  mpfr_zeta (y, x, MPFR_RNDN);
  mpfr_set_prec (x, 47);
  mpfr_set_str_binary (x, "1.0000000000000100000000111001001010111100101011");
  if (mpfr_cmp (x, y))
    {
      printf ("Error in mpfr_zeta (2)\n");
      exit (1);
    }

  /* coverage test */
  mpfr_set_prec (x, 7);
  mpfr_set_str_binary (x, "1.000001");
  mpfr_set_prec (y, 2);
  mpfr_zeta (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 64) == 0);

  /* another coverage test */
  mpfr_set_prec (x, 24);
  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_set_prec (y, 2);
  mpfr_zeta (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui_2exp (y, 3, -1) == 0);

  mpfr_set_nan (x);
  mpfr_zeta (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (y));

  mpfr_set_inf (x, 1);
  mpfr_zeta (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_cmp_ui (y, 1) == 0);

  mpfr_set_inf (x, -1);
  mpfr_zeta (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (y));

  mpfr_clear (x);
  mpfr_clear (y);
}

static const char *const val[] = {
  "-2000", "0.0",
  "-2.0", "0.0",
  "-1.0", "-0.000101010101010101010101010101010101010101010101010101010101010",
  "-0.9", "-0.000110011110011111010001010001100010111101001010100110001110110",
  /*  "-0.8", "-0.000111110011101010001011100011010010000001010011110100010001110",
  "-0.7", "-0.00100101011011111100110011110011111010111111000110110100010110",
  "-0.6", "-0.00101100101100100100110111111000110010111010110010111000001100",
  "-0.5", "-0.00110101001110000000100000011001100100010000111100010001111100",
  "-0.4", "-0.00111111010001100011110001010010111110010001010101111101110001",
  "-0.3", "-0.0100101100110111010101010100111011000001001010111010110101010",
  "-0.2", "-0.0101100110000011101110101011011110101111000010000010110101111",
  "-0.1", "-0.0110101011001111011101001111011000010001111010110011011111011",
  "-0.0", "-0.100000000000000000000000000000000000000000000000000000000000",
  "0.1", "-0.100110100110000010101010101110100000101100100011011001000101",
  "0.2", "-0.10111011111000100011110111100010010001111010010010010100010110",
  "0.3", "-0.11100111100100010011001000001011001100110010110101101110110110",
  "0.4", "-1.0010001010000010000110111000100101001000001011101010110101011",
  "0.5", "-1.0111010111011001110010110000011111100111001111111110111000110",
  "0.6", "-1.1111001111100001100111101110010001001000001101100110110000100",
  "0.7", "-10.110001110100010001110111000101010011110011000110010100101000",
  "0.8", "-100.01110000000000101000010010000011000000111101100101100011010",
  "0.9", "-1001.0110111000011011111100111100111011100010001111111010000100",
  "0.99","-0.11000110110110001101011010110001011010011000110001011100101110E7",
  "0.997", "-0.10100110011000001100111110011111100011110000111011101110001010E9",
  "0.9995", "-0.11111001111011011000011110111111010111101001000110001111110010E11",
  "0.99998", "-0.11000011010011110110110000111011101100001000101101011001110100E16",
  "1.00001", "0.11000011010100000100100111100010001110100000110101110011111011E17",
  "1.0002", "0.10011100010001001001111000101010111000011011011111110010110100E13",
  "1.003","0.10100110111101001001010000000110101101110100001010100000110000E9",
  "1.04", "11001.100101001000001011000111010110011010000001000010111101101",
  "1.1", "1010.1001010110011110011010100010001100101001001111111101100001",
  "1.2", "101.10010111011100011111001001100101101111110000110001101100010",
  "1.3", "11.111011101001010000111001001110100100000101000101101011010100",
  "1.4", "11.000110110000010100100101011110110001100001110100100100111111",
  "1.5", "10.100111001100010010100001011111110111101100010011101011011100",
  "1.6", "10.010010010010011111110000010011000110101001110011101010100110",
  "1.7", "10.000011011110010111011110001100110010100010011100011111110010",
  "1.8", "1.1110000111011001110011001101110101010000011011101100010111001",
  "1.9", "1.1011111111101111011000011110001100100111100110111101101000101",
  "2.0", "1.1010010100011010011001100010010100110000011111010011001000110",
  "42.17", "1.0000000000000000000000000000000000000000001110001110001011001",
  "-17.42", "-11.101110101010101000000001001000001111111101000100001100101100",
  "-24.17", "-0.10001111010010011111000010001011111010010111101011000010010011E13"*/
};

static void
test2 (void)
{
  mpfr_t x, y;
  int i, n = numberof(val);

  mpfr_inits2 (55, x, y, (mpfr_ptr) 0);

  for(i = 0 ; i < n ; i+=2)
    {
      mpfr_set_str1 (x, val[i]);
      mpfr_zeta(y, x, MPFR_RNDZ);
      if (mpfr_cmp_str (y, val[i+1] , 2, MPFR_RNDZ))
        {
          printf("Wrong result for zeta(%s=", val[i]);
          mpfr_print_binary (x);
          printf (").\nGot     : ");
          mpfr_print_binary(y); putchar('\n');
          printf("Expected: ");
          mpfr_set_str (y, val[i+1], 2, MPFR_RNDZ);
          mpfr_print_binary(y); putchar('\n');
          mpfr_set_prec(y, 65);
          mpfr_zeta(y, x, MPFR_RNDZ);
          printf("+ Prec  : ");
          mpfr_print_binary(y); putchar('\n');
          exit(1);
        }
    }
  mpfr_clears (x, y, (mpfr_ptr) 0);
}

#define TEST_FUNCTION mpfr_zeta
#define TEST_RANDOM_EMIN -48
#define TEST_RANDOM_EMAX 31
#include "tgeneric.c"

/* Usage: tzeta - generic tests
          tzeta s prec rnd_mode - compute zeta(s) with precision 'prec'
                                  and rounding mode 'mode' */
int
main (int argc, char *argv[])
{
  mpfr_t s, y, z;
  mpfr_prec_t prec;
  mpfr_rnd_t rnd_mode;
  int inex;

  tests_start_mpfr ();

  if (argc != 1 && argc != 4)
    {
      printf ("Usage: tzeta\n"
              "    or tzeta s prec rnd_mode\n");
      exit (1);
    }

  if (argc == 4)
    {
      prec = atoi(argv[2]);
      mpfr_init2 (s, prec);
      mpfr_init2 (z, prec);
      mpfr_set_str (s, argv[1], 10, MPFR_RNDN);
      rnd_mode = (mpfr_rnd_t) atoi(argv[3]);

      mpfr_zeta (z, s, rnd_mode);
      mpfr_out_str (stdout, 10, 0, z, MPFR_RNDN);
      printf ("\n");

      mpfr_clear (s);
      mpfr_clear (z);

      return 0;
    }

  test1();

  mpfr_init2 (s, MPFR_PREC_MIN);
  mpfr_init2 (y, MPFR_PREC_MIN);
  mpfr_init2 (z, MPFR_PREC_MIN);


  /* the following seems to loop */
  mpfr_set_prec (s, 6);
  mpfr_set_prec (z, 6);
  mpfr_set_str_binary (s, "1.10010e4");
  mpfr_zeta (z, s, MPFR_RNDZ);

  mpfr_set_prec (s, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_prec (z, 53);

  mpfr_set_ui (s, 1, MPFR_RNDN);
  mpfr_clear_divby0();
  mpfr_zeta (z, s, MPFR_RNDN);
  if (!mpfr_inf_p (z) || MPFR_SIGN (z) < 0 || !mpfr_divby0_p())
    {
      printf ("Error in mpfr_zeta for s = 1 (should be +inf) with divby0 flag\n");
      exit (1);
    }

  mpfr_set_str_binary (s, "0.1100011101110111111111111010000110010111001011001011");
  mpfr_set_str_binary (y, "-0.11111101111011001001001111111000101010000100000100100E2");
  mpfr_zeta (z, s, MPFR_RNDN);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (1,RNDN)\n");
      exit (1);
    }
  mpfr_zeta (z, s, MPFR_RNDZ);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (1,RNDZ)\n");
      exit (1);
    }
  mpfr_zeta (z, s, MPFR_RNDU);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (1,RNDU)\n");
      exit (1);
    }
  mpfr_zeta (z, s, MPFR_RNDD);
  mpfr_nexttoinf (y);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (1,RNDD)\n");
      exit (1);
    }

  mpfr_set_str_binary (s, "0.10001011010011100110010001100100001011000010011001011");
  mpfr_set_str_binary (y, "-0.11010011010010101101110111011010011101111101111010110E1");
  mpfr_zeta (z, s, MPFR_RNDN);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (2,RNDN)\n");
      exit (1);
    }
  mpfr_zeta (z, s, MPFR_RNDZ);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (2,RNDZ)\n");
      exit (1);
    }
  mpfr_zeta (z, s, MPFR_RNDU);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (2,RNDU)\n");
      exit (1);
    }
  mpfr_zeta (z, s, MPFR_RNDD);
  mpfr_nexttoinf (y);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (2,RNDD)\n");
      exit (1);
    }

  mpfr_set_str_binary (s, "0.1100111110100001111110111000110101111001011101000101");
  mpfr_set_str_binary (y, "-0.10010111010110000111011111001101100001111011000001010E3");
  mpfr_zeta (z, s, MPFR_RNDN);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (3,RNDN)\n");
      exit (1);
    }
  mpfr_zeta (z, s, MPFR_RNDD);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (3,RNDD)\n");
      exit (1);
    }
  mpfr_nexttozero (y);
  mpfr_zeta (z, s, MPFR_RNDZ);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (3,RNDZ)\n");
      exit (1);
    }
  mpfr_zeta (z, s, MPFR_RNDU);
  if (mpfr_cmp (z, y) != 0)
    {
      printf ("Error in mpfr_zeta (3,RNDU)\n");
      exit (1);
    }

  mpfr_set_str (s, "-400000001", 10, MPFR_RNDZ);
  mpfr_zeta (z, s, MPFR_RNDN);
  if (!(mpfr_inf_p (z) && MPFR_SIGN(z) < 0))
    {
      printf ("Error in mpfr_zeta (-400000001)\n");
      exit (1);
    }
  mpfr_set_str (s, "-400000003", 10, MPFR_RNDZ);
  mpfr_zeta (z, s, MPFR_RNDN);
  if (!(mpfr_inf_p (z) && MPFR_SIGN(z) > 0))
    {
      printf ("Error in mpfr_zeta (-400000003)\n");
      exit (1);
    }

  mpfr_set_prec (s, 34);
  mpfr_set_prec (z, 34);
  mpfr_set_str_binary (s, "-1.111111100001011110000010001010000e-35");
  mpfr_zeta (z, s, MPFR_RNDD);
  mpfr_set_str_binary (s, "-1.111111111111111111111111111111111e-2");
  if (mpfr_cmp (s, z))
    {
      printf ("Error in mpfr_zeta, prec=34, MPFR_RNDD\n");
      mpfr_dump (z);
      exit (1);
    }

  /* bug found by nightly tests on June 7, 2007 */
  mpfr_set_prec (s, 23);
  mpfr_set_prec (z, 25);
  mpfr_set_str_binary (s, "-1.0110110110001000000000e-27");
  mpfr_zeta (z, s, MPFR_RNDN);
  mpfr_set_prec (s, 25);
  mpfr_set_str_binary (s, "-1.111111111111111111111111e-2");
  if (mpfr_cmp (s, z))
    {
      printf ("Error in mpfr_zeta, prec=25, MPFR_RNDN\n");
      printf ("expected "); mpfr_dump (s);
      printf ("got      "); mpfr_dump (z);
      exit (1);
    }

  /* bug reported by Kevin Rauch on 26 Oct 2007 */
  mpfr_set_prec (s, 128);
  mpfr_set_prec (z, 128);
  mpfr_set_str_binary (s, "-0.1000000000000000000000000000000000000000000000000000000000000001E64");
  inex = mpfr_zeta (z, s, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (z) && MPFR_SIGN (z) < 0 && inex < 0);
  inex = mpfr_zeta (z, s, MPFR_RNDU);
  mpfr_set_inf (s, -1);
  mpfr_nextabove (s);
  MPFR_ASSERTN (mpfr_equal_p (z, s) && inex > 0);

  /* bug reported by Fredrik Johansson on 19 Jan 2016 */
  mpfr_set_prec (s, 536);
  mpfr_set_ui_2exp (s, 1, -424, MPFR_RNDN);
  mpfr_sub_ui (s, s, 128, MPFR_RNDN);  /* -128 + 2^(-424) */
  for (prec = 6; prec <= 536; prec += 8) /* should go through 318 */
    {
      mpfr_set_prec (z, prec);
      mpfr_zeta (z, s, MPFR_RNDD);
      mpfr_set_prec (y, prec + 10);
      mpfr_zeta (y, s, MPFR_RNDD);
      mpfr_prec_round (y, prec, MPFR_RNDD);
      if (! mpfr_equal_p (z, y))
        {
          printf ("mpfr_zeta fails near -128 for inprec=%lu outprec=%lu\n",
                  (unsigned long) mpfr_get_prec (s), (unsigned long) prec);
          printf ("expected "); mpfr_dump (y);
          printf ("got      "); mpfr_dump (z);
          exit (1);
        }
    }

  mpfr_clear (s);
  mpfr_clear (y);
  mpfr_clear (z);

  /* FIXME: change the last argument back to 5 once the working precision
     in the mpfr_zeta implementation no longer depends on the precision of
     the input. */
  test_generic (MPFR_PREC_MIN, 70, 1);
  test2 ();

  tests_end_mpfr ();
  return 0;
}
