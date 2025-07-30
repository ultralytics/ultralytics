/* Test file for mpfr_const_pi.

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

/* tconst_pi [prec] [rnd] [0 = no print] */

static void
check_large (void)
{
  mpfr_t x, y, z;

  mpfr_init2 (x, 20000);
  mpfr_init2 (y, 21000);
  mpfr_init2 (z, 11791);

  /* The algo failed to round for p=11791. */
  (mpfr_const_pi) (z, MPFR_RNDU);
  mpfr_const_pi (x, MPFR_RNDN); /* First one ! */
  mpfr_const_pi (y, MPFR_RNDN); /* Then the other - cache - */
  mpfr_prec_round (y, 20000, MPFR_RNDN);
  if (mpfr_cmp (x, y)) {
    printf ("const_pi: error for large prec (%d)\n", 1);
    exit (1);
  }
  mpfr_prec_round (y, 11791, MPFR_RNDU);
  if (mpfr_cmp (z, y)) {
    printf ("const_pi: error for large prec (%d)\n", 2);
    exit (1);
  }

  /* a worst-case to exercise recomputation */
  if (MPFR_PREC_MAX > 33440) {
    mpfr_set_prec (x, 33440);
    mpfr_const_pi (x, MPFR_RNDZ);
  }

  mpfr_clears (x, y, z, (mpfr_ptr) 0);
}

/* Wrapper for tgeneric */
static int
my_const_pi (mpfr_ptr x, mpfr_srcptr y, mpfr_rnd_t r)
{
  return mpfr_const_pi (x, r);
}

#define RAND_FUNCTION(x) mpfr_set_ui ((x), 0, MPFR_RNDN)
#define TEST_FUNCTION my_const_pi
#include "tgeneric.c"

static void
bug20091030 (void)
{
  mpfr_t x, x_ref;
  int inex, inex_ref;
  mpfr_prec_t p;
  int r;

  mpfr_free_cache ();
  mpfr_init2 (x, MPFR_PREC_MIN);
  for (p = MPFR_PREC_MIN; p <= 100; p++)
    {
      mpfr_set_prec (x, p);
      inex = mpfr_const_pi (x, MPFR_RNDU);
      if (inex < 0)
        {
          printf ("Error, inex < 0 for RNDU (prec=%lu)\n",
                  (unsigned long) p);
          exit (1);
        }
      inex = mpfr_const_pi (x, MPFR_RNDD);
      if (inex > 0)
        {
          printf ("Error, inex > 0 for RNDD (prec=%lu)\n",
                  (unsigned long) p);
          exit (1);
        }
    }
  mpfr_free_cache ();
  mpfr_init2 (x_ref, MPFR_PREC_MIN);
  for (p = MPFR_PREC_MIN; p <= 100; p++)
    {
      mpfr_set_prec (x, p + 10);
      mpfr_const_pi (x, MPFR_RNDN);
      mpfr_set_prec (x, p);
      mpfr_set_prec (x_ref, p);
      for (r = 0; r < MPFR_RND_MAX; r++)
        {
          inex = mpfr_const_pi (x, (mpfr_rnd_t) r);
          inex_ref = mpfr_const_pi_internal (x_ref, (mpfr_rnd_t) r);
          if (inex != inex_ref || mpfr_cmp (x, x_ref) != 0)
            {
              printf ("mpfr_const_pi and mpfr_const_pi_internal disagree\n");
              printf ("mpfr_const_pi gives ");
              mpfr_dump (x);
              printf ("mpfr_const_pi_internal gives ");
              mpfr_dump (x_ref);
              printf ("inex=%d inex_ref=%d\n", inex, inex_ref);
              exit (1);
            }
        }
    }
  mpfr_clear (x);
  mpfr_clear (x_ref);
}

int
main (int argc, char *argv[])
{
  mpfr_t x;
  mpfr_prec_t p;
  mpfr_rnd_t rnd;

  tests_start_mpfr ();

  p = 53;
  if (argc > 1)
    {
      long a = atol (argv[1]);
      if (a >= MPFR_PREC_MIN && a <= MPFR_PREC_MAX)
        p = a;
    }

  rnd = (argc > 2) ? (mpfr_rnd_t) atoi(argv[2]) : MPFR_RNDZ;

  mpfr_init2 (x, p);
  mpfr_const_pi (x, rnd);
  if (argc >= 2)
    {
      if (argc < 4 || atoi (argv[3]) != 0)
        {
          printf ("Pi=");
          mpfr_out_str (stdout, 10, 0, x, rnd);
          puts ("");
        }
    }
  else if (mpfr_cmp_str1 (x, "3.141592653589793116") )
    {
      printf ("mpfr_const_pi failed for prec=53\n");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar('\n');
      exit (1);
    }

  mpfr_set_prec (x, 32);
  mpfr_const_pi (x, MPFR_RNDN);
  if (mpfr_cmp_str1 (x, "3.141592653468251") )
    {
      printf ("mpfr_const_pi failed for prec=32\n");
      exit (1);
    }

  mpfr_clear (x);

  bug20091030 ();

  check_large ();

  test_generic (2, 200, 1);

  tests_end_mpfr ();

  return 0;
}
