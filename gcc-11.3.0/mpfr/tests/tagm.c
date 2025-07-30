/* Test file for mpfr_agm.

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

#define check(a,b,r) check4(a,b,r,0.0)

static void
check4 (const char *as, const char *bs, mpfr_rnd_t rnd_mode,
        const char *res, int inex)
{
  mpfr_t ta, tb, tc, tres;
  mpfr_exp_t emin, emax;
  int i;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  mpfr_inits2 (53, ta, tb, tc, tres, (mpfr_ptr) 0);

  for (i = 0; i <= 2; i++)
    {
      unsigned int expflags, newflags;
      int inex2;

      mpfr_set_str1 (ta, as);
      mpfr_set_str1 (tb, bs);
      mpfr_set_str1 (tc, res);

      if (i > 0)
        {
          mpfr_exp_t ea, eb, ec, e0;

          set_emin (MPFR_EMIN_MIN);
          set_emax (MPFR_EMAX_MAX);

          ea = mpfr_get_exp (ta);
          eb = mpfr_get_exp (tb);
          ec = mpfr_get_exp (tc);

          e0 = i == 1 ? __gmpfr_emin : __gmpfr_emax;
          if ((i == 1 && ea < eb) || (i == 2 && ea > eb))
            {
              mpfr_set_exp (ta, e0);
              mpfr_set_exp (tb, e0 + (eb - ea));
              mpfr_set_exp (tc, e0 + (ec - ea));
            }
          else
            {
              mpfr_set_exp (ta, e0 + (ea - eb));
              mpfr_set_exp (tb, e0);
              mpfr_set_exp (tc, e0 + (ec - eb));
            }
        }

      __gmpfr_flags = expflags =
        (randlimb () & 1) ? MPFR_FLAGS_ALL ^ MPFR_FLAGS_ERANGE : 0;
      inex2 = mpfr_agm (tres, ta, tb, rnd_mode);
      newflags = __gmpfr_flags;
      expflags |= MPFR_FLAGS_INEXACT;

      if (SIGN (inex2) != inex || newflags != expflags ||
          ! mpfr_equal_p (tres, tc))
        {
          printf ("mpfr_agm failed in rnd_mode=%s for\n",
                  mpfr_print_rnd_mode (rnd_mode));
          printf ("  a = ");
          mpfr_out_str (stdout, 10, 0, ta, MPFR_RNDN);
          printf ("\n");
          printf ("  b = ");
          mpfr_out_str (stdout, 10, 0, tb, MPFR_RNDN);
          printf ("\n");
          printf ("expected inex = %d, flags = %u,\n"
                  "         ", inex, expflags);
          mpfr_dump (tc);
          printf ("got      inex = %d, flags = %u,\n"
                  "         ", inex2, newflags);
          mpfr_dump (tres);
          exit (1);
        }

      set_emin (emin);
      set_emax (emax);
    }

  mpfr_clears (ta, tb, tc, tres, (mpfr_ptr) 0);
}

static void
check_large (void)
{
  mpfr_t a, b, agm;
  int inex;

  mpfr_init2 (a, 82);
  mpfr_init2 (b, 82);
  mpfr_init2 (agm, 82);

  mpfr_set_ui (a, 1, MPFR_RNDN);
  mpfr_set_str_binary (b, "0.1111101100001000000001011000110111101000001011111000100001000101010100011111110010E-39");
  mpfr_agm (agm, a, b, MPFR_RNDN);
  mpfr_set_str_binary (a, "0.1110001000111101101010101010101101001010001001001011100101111011110101111001111100E-4");
  if (mpfr_cmp (agm, a))
    {
      printf ("mpfr_agm failed for precision 82\n");
      exit (1);
    }

  /* problem found by Damien Fischer <damien@maths.usyd.edu.au> 4 Aug 2003:
     produced a divide-by-zero exception */
  mpfr_set_prec (a, 268);
  mpfr_set_prec (b, 268);
  mpfr_set_prec (agm, 268);
  mpfr_set_str (a, "703.93543315330225238487276503953366664991725089988315253092140138947103394917006", 10, MPFR_RNDN);
  mpfr_set_str (b, "703.93543315330225238487279020523738740563816490895994499256063816906728642622316", 10, MPFR_RNDN);
  mpfr_agm (agm, a, b, MPFR_RNDN);

  mpfr_set_prec (a, 18);
  mpfr_set_prec (b, 70);
  mpfr_set_prec (agm, 67);
  mpfr_set_str_binary (a, "0.111001111100101000e8");
  mpfr_set_str_binary (b, "0.1101110111100100010100110000010111011011011100110100111001010100100001e10");
  inex = mpfr_agm (agm, a, b, MPFR_RNDN);
  mpfr_set_str_binary (b, "0.1111110010011101101100010101011011010010010000001010100011000110011e9");
  if (mpfr_cmp (agm, b))
    {
      printf ("Error in mpfr_agm (1)\n");
      exit (1);
    }
  if (inex >= 0)
    {
      printf ("Wrong flag for mpfr_agm (1)\n");
      exit (1);
    }

  /* test worst case: 9 consecutive ones after the last bit */
  mpfr_set_prec (a, 2);
  mpfr_set_prec (b, 2);
  mpfr_set_ui (a, 1, MPFR_RNDN);
  mpfr_set_ui (b, 2, MPFR_RNDN);
  mpfr_set_prec (agm, 904);
  mpfr_agm (agm, a, b, MPFR_RNDZ);

  mpfr_clear (a);
  mpfr_clear (b);
  mpfr_clear (agm);
}

static void
check_eq (void)
{
  mpfr_t a, b, agm;
  int p;

  mpfr_init2 (a, 17);
  mpfr_init2 (b, 9);

  mpfr_set_str_binary (b, "0.101000000E-3");
  mpfr_set (a, b, MPFR_RNDN);

  for (p = MPFR_PREC_MIN; p <= 2; p++)
    {
      int inex;

      mpfr_init2 (agm, p);
      inex = mpfr_agm (agm, a, b, MPFR_RNDU);
      if (mpfr_cmp_ui_2exp (agm, 5 - p, -5) != 0)
        {
          printf ("Error in check_eq for p = %d: expected %d*2^(-5), got ",
                  p, 5 - p);
          mpfr_dump (agm);
          exit (1);
        }
      if (inex <= 0)
        {
          printf ("Wrong ternary value in check_eq for p = %d\n", p);
          printf ("expected 1\n");
          printf ("got      %d\n", inex);
          exit (1);
        }
      mpfr_clear (agm);
    }

  mpfr_clear (a);
  mpfr_clear (b);
}

static void
check_nans (void)
{
  mpfr_t  x, y, m;

  mpfr_init2 (x, 123L);
  mpfr_init2 (y, 123L);
  mpfr_init2 (m, 123L);

  /* agm(1,nan) == nan */
  mpfr_set_ui (x, 1L, MPFR_RNDN);
  mpfr_set_nan (y);
  mpfr_agm (m, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (m));

  /* agm(1,+inf) == +inf */
  mpfr_set_ui (x, 1L, MPFR_RNDN);
  mpfr_set_inf (y, 1);
  mpfr_agm (m, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (m));
  MPFR_ASSERTN (mpfr_sgn (m) > 0);

  /* agm(+inf,+inf) == +inf */
  mpfr_set_inf (x, 1);
  mpfr_set_inf (y, 1);
  mpfr_agm (m, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_inf_p (m));
  MPFR_ASSERTN (mpfr_sgn (m) > 0);

  /* agm(-inf,+inf) == nan */
  mpfr_set_inf (x, -1);
  mpfr_set_inf (y, 1);
  mpfr_agm (m, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (m));

  /* agm(+0,+inf) == nan */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_set_inf (y, 1);
  mpfr_agm (m, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (m));

  /* agm(+0,1) == +0 */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_agm (m, x, y, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO (m) && MPFR_IS_POS(m));

  /* agm(-0,1) == +0 */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_agm (m, x, y, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO (m) && MPFR_IS_POS(m));

  /* agm(-0,+0) == +0 */
  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  mpfr_agm (m, x, y, MPFR_RNDN);
  MPFR_ASSERTN (MPFR_IS_ZERO (m) && MPFR_IS_POS(m));

  /* agm(1,1) == 1 */
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  mpfr_agm (m, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_cmp_ui (m ,1) == 0);

  /* agm(-1,-2) == NaN */
  mpfr_set_si (x, -1, MPFR_RNDN);
  mpfr_set_si (y, -2, MPFR_RNDN);
  mpfr_agm (m, x, y, MPFR_RNDN);
  MPFR_ASSERTN (mpfr_nan_p (m));

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (m);
}

#define TEST_FUNCTION mpfr_agm
#define TWO_ARGS
#define TEST_RANDOM_POS 4
#define TEST_RANDOM_POS2 4
#include "tgeneric.c"

int
main (int argc, char* argv[])
{
  tests_start_mpfr ();

  check_nans ();

  check_large ();
  check_eq ();
  check4 ("2.0", "1.0", MPFR_RNDN, "1.456791031046906869", -1);
  check4 ("6.0", "4.0", MPFR_RNDN, "4.949360872472608925", 1);
  check4 ("62.0", "61.0", MPFR_RNDN, "61.498983718845075902", -1);
  check4 ("0.5", "1.0", MPFR_RNDN, "0.72839551552345343459", -1);
  check4 ("1.0", "2.0", MPFR_RNDN, "1.456791031046906869", -1);
  check4 ("234375765.0", "234375000.0", MPFR_RNDN, "234375382.49984394025", 1);
  check4 ("8.0", "1.0", MPFR_RNDU, "3.61575617759736274873", 1);
  check4 ("1.0", "44.0", MPFR_RNDU, "13.3658354512981243907", 1);
  check4 ("1.0", "3.7252902984619140625e-9", MPFR_RNDU,
          "0.07553933569711989657765", 1);
  test_generic (2, 300, 17);

  tests_end_mpfr ();
  return 0;
}
