/* Test file for mpfr_zeta_ui.

Copyright 2005-2017 Free Software Foundation, Inc.
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

#define TEST_FUNCTION mpfr_zeta_ui

int
main (int argc, char *argv[])
{
#if MPFR_VERSION >= MPFR_VERSION_NUM(2,3,0)
  unsigned int prec, yprec;
  int rnd;
  mpfr_t x, y, z, t;
  unsigned long n;
  int inex;
  mpfr_exp_t emin, emax;
  unsigned int flags, ex_flags;
  int i;

  tests_start_mpfr ();

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  mpfr_init (x);
  mpfr_init (y);
  mpfr_init (z);
  mpfr_init (t);

  if (argc >= 3) /* tzeta_ui n prec [rnd] */
    {
      mpfr_set_prec (x, atoi (argv[2]));
      mpfr_zeta_ui (x, atoi (argv[1]),
                    argc > 3 ? (mpfr_rnd_t) atoi (argv[3]) : MPFR_RNDN);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      printf ("\n");
      goto clear_and_exit;
    }

  mpfr_set_prec (x, 33);
  mpfr_set_prec (y, 33);
  mpfr_zeta_ui (x, 3, MPFR_RNDZ);
  mpfr_set_str_binary (y, "0.100110011101110100000000001001111E1");
  if (mpfr_cmp (x, y))
    {
      printf ("Error for zeta(3), prec=33, MPFR_RNDZ\n");
      printf ("expected "); mpfr_dump (y);
      printf ("got      "); mpfr_dump (x);
      exit (1);
    }

  mpfr_clear_flags ();
  inex = mpfr_zeta_ui (x, 0, MPFR_RNDN);
  flags = __gmpfr_flags;
  MPFR_ASSERTN (inex == 0 && mpfr_cmp_si_2exp (x, -1, -1) == 0 && flags == 0);

  for (i = -2; i <= 2; i += 2)
    RND_LOOP (rnd)
      {
        int ex_inex;

        set_emin (i);
        set_emax (i);
        mpfr_clear_flags ();
        inex = mpfr_zeta_ui (x, 0, (mpfr_rnd_t) rnd);
        flags = __gmpfr_flags;
        if (i < 0)
          {
            mpfr_set_inf (y, -1);
            if (rnd == MPFR_RNDU || rnd == MPFR_RNDZ)
              {
                mpfr_nextabove (y);
                ex_inex = 1;
              }
            else
              {
                ex_inex = -1;
              }
            ex_flags = MPFR_FLAGS_OVERFLOW | MPFR_FLAGS_INEXACT;
          }
        else if (i > 0)
          {
            mpfr_set_zero (y, -1);
            if (rnd == MPFR_RNDD || rnd == MPFR_RNDA)
              {
                mpfr_nextbelow (y);
                ex_inex = -1;
              }
            else
              {
                ex_inex = 1;
              }
            ex_flags = MPFR_FLAGS_UNDERFLOW | MPFR_FLAGS_INEXACT;
          }
        else
          {
            mpfr_set_str_binary (y, "-1e-1");
            ex_inex = 0;
            ex_flags = 0;
          }
        set_emin (emin);
        set_emax (emax);
        if (! (mpfr_equal_p (x, y) && MPFR_IS_NEG (x) &&
               SAME_SIGN (inex, ex_inex) && flags == ex_flags))
          {
            printf ("Failure for zeta(0) in %s, exponent range [%d,%d]\n",
                    mpfr_print_rnd_mode ((mpfr_rnd_t) rnd), i, i);
            printf ("Expected ");
            mpfr_dump (y);
            printf ("  with inex ~ %d, flags =", ex_inex);
            flags_out (ex_flags);
            printf ("Got      ");
            mpfr_dump (x);
            printf ("  with inex = %d, flags =", inex);
            flags_out (flags);
            exit (1);
          }
      }

  mpfr_clear_divby0 ();
  inex = mpfr_zeta_ui (x, 1, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0 && MPFR_IS_INF (x) && MPFR_IS_POS (x)
                && mpfr_divby0_p ());

  for (prec = MPFR_PREC_MIN; prec <= 100; prec++)
    {
      mpfr_set_prec (x, prec);
      mpfr_set_prec (z, prec);
      mpfr_set_prec (t, prec);
      yprec = prec + 10;
      mpfr_set_prec (y, yprec);

      for (n = 0; n < 50; n++)
        RND_LOOP (rnd)
          {
            mpfr_zeta_ui (y, n, MPFR_RNDN);
            if (mpfr_can_round (y, yprec, MPFR_RNDN, MPFR_RNDZ, prec
                                + (rnd == MPFR_RNDN)))
              {
                mpfr_set (t, y, (mpfr_rnd_t) rnd);
                for (i = 0; i <= 1; i++)
                  {
                    if (i)
                      {
                        mpfr_exp_t e;

                        if (MPFR_IS_SINGULAR (t))
                          break;
                        e = mpfr_get_exp (t);
                        set_emin (e);
                        set_emax (e);
                      }
                    mpfr_zeta_ui (z, n, (mpfr_rnd_t) rnd);
                    if (i)
                      {
                        set_emin (emin);
                        set_emax (emax);
                      }
                    if (mpfr_cmp (t, z))
                      {
                        printf ("results differ for n = %lu, prec = %u,"
                                " %s%s\n", n, prec,
                                mpfr_print_rnd_mode ((mpfr_rnd_t) rnd),
                                i ? ", reduced exponent range" : "");
                        printf ("  got      ");
                        mpfr_dump (z);
                        printf ("  expected ");
                        mpfr_dump (t);
                        printf ("  approx   ");
                        mpfr_dump (y);
                        exit (1);
                      }
                  }
              }
          }
    }

 clear_and_exit:
  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
  mpfr_clear (t);

  tests_end_mpfr ();
#endif
  return 0;
}
