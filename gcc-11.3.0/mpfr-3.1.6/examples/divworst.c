/* Test of the double rounding effect.
 *
 * This example was presented at the CNC'2 summer school on MPFR and MPC
 * at LORIA, Nancy, France.
 *
 * Arguments: max difference of exponents dmax, significand size n.
 * Optional argument: extended precision p (with double rounding).
 *
 * Return all the couples of positive machine numbers (x,y) such that
 * 1/2 <= y < 1, 0 <= Ex - Ey <= dmax, x - y is exactly representable
 * in precision n and the results of floor(x/y) in the rounding modes
 * toward 0 and to nearest are different.
 */

/*
Copyright 2009-2017 Free Software Foundation, Inc.
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
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpfr.h>

#define PRECN x, y, z
#define VARS PRECN, t

static unsigned long
eval (mpfr_t x, mpfr_t y, mpfr_t z, mpfr_t t, mpfr_rnd_t rnd)
{
  mpfr_div (t, x, y, rnd);  /* the division x/y in precision p */
  mpfr_set (z, t, rnd);     /* the rounding to the precision n */
  mpfr_rint_floor (z, z, rnd);
  return mpfr_get_ui (z, rnd);
}

int main (int argc, char *argv[])
{
  int dmax, n, p;
  mpfr_t VARS;

  if (argc != 3 && argc != 4)
    {
      fprintf (stderr, "Usage: divworst <dmax> <n> [ <p> ]\n");
      exit (EXIT_FAILURE);
    }

  dmax = atoi (argv[1]);
  n = atoi (argv[2]);
  p = argc == 3 ? n : atoi (argv[3]);
  if (p < n)
    {
      fprintf (stderr, "divworst: p must be greater or equal to n\n");
      exit (EXIT_FAILURE);
    }

  mpfr_inits2 (n, PRECN, (mpfr_ptr) 0);
  mpfr_init2 (t, p);

  for (mpfr_set_ui_2exp (x, 1, -1, MPFR_RNDN);
       mpfr_get_exp (x) <= dmax;
       mpfr_nextabove (x))
    for (mpfr_set_ui_2exp (y, 1, -1, MPFR_RNDN);
         mpfr_get_exp (y) == 0;
         mpfr_nextabove (y))
      {
        unsigned long rz, rn;

        if (mpfr_sub (z, x, y, MPFR_RNDZ) != 0)
          continue;  /* x - y is not representable in precision n */
        rz = eval (x, y, z, t, MPFR_RNDZ);
        rn = eval (x, y, z, t, MPFR_RNDN);
        if (rz == rn)
          continue;
        mpfr_printf ("x = %.*Rb ; y = %.*Rb ; Z: %lu ; N: %lu\n",
                     n - 1, x, n - 1, y, rz, rn);
      }

  mpfr_clears (VARS, (mpfr_ptr) 0);
  return 0;
}
