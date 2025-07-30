/* Test file for mpfr_const_log2.

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

#include <stdio.h>
#include <stdlib.h>

#include "mpfr-test.h"

/* tlog2 [prec] [rnd] [0 = no print] */

static void
check (mpfr_prec_t p0, mpfr_prec_t p1)
{
  mpfr_t x, y, z;
  mpfr_rnd_t rnd;
  int dif;

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init2 (z, p1 + 10);
  mpfr_const_log2 (z, MPFR_RNDN);
  mpfr_clear_cache (__gmpfr_cache_const_log2);

  for (; p0<=p1; p0++)
    {
      mpfr_set_prec (x, p0);
      mpfr_set_prec (y, p0);
        {
          rnd = RND_RAND ();
          mpfr_const_log2 (x, rnd);
          mpfr_set (y, z, rnd);
          if ((dif = mpfr_cmp (x, y))
              && mpfr_can_round (z, mpfr_get_prec(z), MPFR_RNDN,
                                                 rnd, p0))
            {
              printf ("mpfr_const_log2 fails for prec=%u, rnd=%s Diff=%d\n",
                      (unsigned int) p0, mpfr_print_rnd_mode (rnd), dif);
              printf ("expected "), mpfr_dump (y);
              printf ("got      "), mpfr_dump (x);
              exit (1);
            }
        }
    }

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

static void
check_large (void)
{
  mpfr_t x, y;
  mpfr_init2 (x, 25000);
  mpfr_init2 (y, 26000);
  (mpfr_const_log2) (x, MPFR_RNDN); /* First one ! */
  (mpfr_const_log2) (y, MPFR_RNDN); /* Then the other - cache - */
  mpfr_prec_round (y, 25000, MPFR_RNDN);
  if (mpfr_cmp (x, y))
    {
      printf ("const_log2: error for large prec\n");
      exit (1);
    }

  /* worst-case with 15 successive ones after last bit,
     to exercise can_round loop */
  mpfr_set_prec (x, 26249);
  mpfr_const_log2 (x, MPFR_RNDZ);

  mpfr_clears (x, y, (mpfr_ptr) 0);
}

static void
check_cache (void)
{
  mpfr_t x;
  int i;

  mpfr_init2 (x, 195);
  mpfr_free_cache ();
  i = mpfr_const_log2 (x, MPFR_RNDN);
  if (i == 0)
    {
      printf("Error for log2. Invalid ternary value (1).\n");
      exit (1);
    }
  mpfr_set_prec (x, 194);
  i = mpfr_const_log2 (x, MPFR_RNDN);
  if (i == 0)
    {
      printf("Error for log2. Invalid ternary value (2).\n");
      exit (1);
    }

  mpfr_free_cache ();
  mpfr_set_prec (x, 9);
  mpfr_const_log2 (x, MPFR_RNDN);
  mpfr_set_prec (x, 8);
  mpfr_const_log2 (x, MPFR_RNDN);
  if (mpfr_cmp_str (x, "0.10110001E0", 2, MPFR_RNDN))
    {
      printf("Error for log2. Wrong rounding.\n");
      exit (1);
    }

  mpfr_clear (x);
}

/* Wrapper for tgeneric */
static int
my_const_log2 (mpfr_ptr x, mpfr_srcptr y, mpfr_rnd_t r)
{
  return mpfr_const_log2 (x, r);
}

#define RAND_FUNCTION(x) mpfr_set_ui ((x), 0, MPFR_RNDN)
#define TEST_FUNCTION my_const_log2
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  mpfr_t x;
  int p;
  mpfr_rnd_t rnd;

  tests_start_mpfr ();

  p = (argc>1) ? atoi(argv[1]) : 53;
  rnd = (argc>2) ? (mpfr_rnd_t) atoi(argv[2]) : MPFR_RNDZ;

  mpfr_init (x);

  check (2, 1000);

  /* check precision of 2 bits */
  mpfr_set_prec (x, 2);
  mpfr_const_log2 (x, MPFR_RNDN);
  if (mpfr_cmp_ui_2exp(x, 3, -2)) /* 3*2^-2 */
    {
      printf ("mpfr_const_log2 failed for prec=2, rnd=MPFR_RNDN\n"
              "expected 0.75, got ");
      mpfr_out_str(stdout, 10, 0, x, MPFR_RNDN);
      putchar('\n');
      exit (1);
    }

  if (argc>=2)
    {
      mpfr_set_prec (x, p);
      mpfr_const_log2 (x, rnd);
      printf ("log(2)=");
      mpfr_out_str (stdout, 10, 0, x, rnd);
      puts ("");
    }

  mpfr_set_prec (x, 53);
  mpfr_const_log2 (x, MPFR_RNDZ);
  if (mpfr_cmp_str1 (x, "6.9314718055994530941e-1") )
    {
      printf ("mpfr_const_log2 failed for prec=53\n");
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_const_log2 (x, MPFR_RNDN);
  if (mpfr_cmp_str1 (x, "0.69314718060195446"))
    {
      printf ("mpfr_const_log2 failed for prec=32\n");
      exit (1);
    }

  mpfr_clear(x);

  check_large();
  check_cache ();

  test_generic (2, 200, 1);

  tests_end_mpfr ();
  return 0;
}
