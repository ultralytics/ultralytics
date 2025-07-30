/* Test file for mpfr_eq.

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

#include <stdio.h>
#include <stdlib.h>

#include "mpfr-test.h"

static void
teq (mpfr_t x)
{
  mpfr_t y;
  unsigned long k, px, mx;

  mpfr_init2 (y, MPFR_PREC(x));

  mx = (MPFR_PREC(x) - 1) / mp_bits_per_limb;
  px = mp_bits_per_limb - 2;

  for (k = 2; k < MPFR_PREC(x); k++)
    {
      mpfr_set (y, x, MPFR_RNDN);

      MPFR_MANT(y) [mx] ^= (mp_limb_t) 1 << px;

      if (mpfr_eq(y, x, k) || !mpfr_eq(y, x, k - 1))
        {
          printf ("Error in eq.\n");
          printf ("x = "); mpfr_print_binary (x); printf ("\n");
          printf ("y = "); mpfr_print_binary (y); printf ("\n");
          printf ("k = %lu\n", k);
          printf ("mpfr_eq(y, x, k) = %d\nmpfr_eq(y, x, k - 1) = %d\n",
                  mpfr_eq (y, x, k), mpfr_eq (y, x, k - 1));
          exit (1);
        }

      if (px)
        {
          --px;
        }
      else
        {
          --mx;
          px = mp_bits_per_limb - 1;
        }
    }
  mpfr_clear (y);
}

static void
special (void)
{
  mpfr_t x, y, z;
  int i, error = 0;

  mpfr_init2 (x, 53);
  mpfr_init2 (y, 53);
  mpfr_init2 (z, 53);

  mpfr_set_str (x, "1", 10, (mpfr_rnd_t) 0);
  mpfr_set_str (y, "1e-10000", 10, (mpfr_rnd_t) 0);
  mpfr_add (z, x, y, MPFR_RNDU);

  for (i = 1; i <= 52; i++)
    if (mpfr_eq (x, z, i) == 0)
      error = 1;
  for (i = 53; i <= 100; i++)
    if (mpfr_eq (x, z, i) != 0)
      error = 1;
  if (mpfr_eq (x, z, 1000) != 0)
    error = 1;

  if (error)
    {
      printf ("Error in mpfr_eq (1, 1+1e-1000)\n");
      exit (1);
    }

  mpfr_set_nan (x);
  mpfr_set_nan (y);
  MPFR_ASSERTN(mpfr_eq (x, y, 1) == 0);

  mpfr_set_inf (y, 1);
  MPFR_ASSERTN(mpfr_eq (x, y, 1) == 0);

  mpfr_set_ui (y, 0, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_eq (x, y, 1) == 0);

  mpfr_set_inf (x, 1);
  mpfr_set_inf (y, 1);
  MPFR_ASSERTN(mpfr_eq (x, y, 1));

  mpfr_set_inf (x, 1);
  mpfr_set_inf (y, -1);
  MPFR_ASSERTN(mpfr_eq (x, y, 1) == 0);

  mpfr_set_inf (x, -1);
  mpfr_set_inf (y, -1);
  MPFR_ASSERTN(mpfr_eq (x, y, 1));

  mpfr_set_inf (x, 1);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_eq (x, y, 1) == 0);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_eq (x, y, 1) == 0);
  MPFR_ASSERTN(mpfr_eq (y, x, 1) == 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_set_ui (y, 0, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_eq (x, y, 1));

  mpfr_neg (y, y, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_eq (x, y, 1));

  mpfr_neg (x, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_eq (x, y, 1));

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_neg (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_eq (x, y, 1) == 0);

  mpfr_set_prec (x, 2 * mp_bits_per_limb);
  mpfr_set_prec (y, mp_bits_per_limb);
  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_ui (y, 1, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_eq (x, y, mp_bits_per_limb - 1));
  MPFR_ASSERTN(mpfr_eq (x, y, mp_bits_per_limb));
  MPFR_ASSERTN(mpfr_eq (x, y, mp_bits_per_limb + 1));
  MPFR_ASSERTN(mpfr_eq (x, y, 2 * mp_bits_per_limb - 1));
  MPFR_ASSERTN(mpfr_eq (x, y, 2 * mp_bits_per_limb));
  MPFR_ASSERTN(mpfr_eq (x, y, 2 * mp_bits_per_limb + 1));

  mpfr_nextabove (x);
  MPFR_ASSERTN(mpfr_eq (x, y, mp_bits_per_limb - 1));
  MPFR_ASSERTN(mpfr_eq (x, y, mp_bits_per_limb));
  MPFR_ASSERTN(mpfr_eq (x, y, mp_bits_per_limb + 1));
  MPFR_ASSERTN(mpfr_eq (x, y, 2 * mp_bits_per_limb - 1));
  MPFR_ASSERTN(mpfr_eq (x, y, 2 * mp_bits_per_limb) == 0);
  MPFR_ASSERTN(mpfr_eq (x, y, 2 * mp_bits_per_limb + 1) == 0);
  MPFR_ASSERTN(mpfr_eq (y, x, mp_bits_per_limb - 1));
  MPFR_ASSERTN(mpfr_eq (y, x, mp_bits_per_limb));
  MPFR_ASSERTN(mpfr_eq (y, x, mp_bits_per_limb + 1));
  MPFR_ASSERTN(mpfr_eq (y, x, 2 * mp_bits_per_limb - 1));
  MPFR_ASSERTN(mpfr_eq (y, x, 2 * mp_bits_per_limb) == 0);
  MPFR_ASSERTN(mpfr_eq (y, x, 2 * mp_bits_per_limb + 1) == 0);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_set_ui (y, 2, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_eq (x, y, 1) == 0);

  mpfr_set_prec (x, 2 * mp_bits_per_limb);
  mpfr_set_prec (y, 2 * mp_bits_per_limb);
  mpfr_set_ui (x, 2, MPFR_RNDN);
  mpfr_set_ui (y, 3, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_eq (x, y, 1));
  MPFR_ASSERTN(mpfr_eq (x, y, 2) == 0);
  MPFR_ASSERTN(mpfr_eq (x, y, mp_bits_per_limb) == 0);
  MPFR_ASSERTN(mpfr_eq (x, y, mp_bits_per_limb + 1) == 0);

  mpfr_clear (x);
  mpfr_clear (y);
  mpfr_clear (z);
}

int
main (void)
{
  int j;
  mpfr_t x;

  tests_start_mpfr ();

  special ();

  mpfr_init2 (x, 500);

  for (j = 0; j < 500; j++)
    {
      mpfr_urandomb (x, RANDS);
      teq (x);
    }

  mpfr_clear (x);

  tests_end_mpfr ();
  return 0;
}
