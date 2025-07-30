/* This example was presented at the CNC'2 summer school on MPFR and MPC at
 * LORIA, Nancy, France. It shows how one can use different rounding modes.
 * This example implements the OddRoundedAdd algorithm, which returns the
 * sum z = x + y rounded-to-odd:
 *   * RO(z) = z if z is exactly representable;
 *   * otherwise RO(z) is the value among RD(z) and RU(z) whose
 *     least significant bit is a one.
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
#include <gmp.h>
#include <mpfr.h>

#define LIST x, y, d, u, e, z

int main (int argc, char **argv)
{
  mpfr_t LIST;
  mpfr_prec_t prec;
  int pprec;       /* will be prec - 1 for mpfr_printf */

  if (argc != 4)
    {
      fprintf (stderr, "Usage: rndo-add <prec> <x> <y>\n");
      exit (1);
    }

  prec = atoi (argv[1]);
  if (prec < 2)
    {
      fprintf (stderr, "rndo-add: bad precision\n");
      exit (1);
    }
  pprec = prec - 1;

  mpfr_inits2 (prec, LIST, (mpfr_ptr) 0);

  if (mpfr_set_str (x, argv[2], 0, MPFR_RNDN))
    {
      fprintf (stderr, "rndo-add: bad x value\n");
      exit (1);
    }
  mpfr_printf ("x = %.*Rb\n", pprec, x);

  if (mpfr_set_str (y, argv[3], 0, MPFR_RNDN))
    {
      fprintf (stderr, "rndo-add: bad y value\n");
      exit (1);
    }
  mpfr_printf ("y = %.*Rb\n", pprec, y);

  mpfr_add (d, x, y, MPFR_RNDD);
  mpfr_printf ("d = %.*Rb\n", pprec, d);

  mpfr_add (u, x, y, MPFR_RNDU);
  mpfr_printf ("u = %.*Rb\n", pprec, u);

  mpfr_add (e, d, u, MPFR_RNDN);
  mpfr_div_2ui (e, e, 1, MPFR_RNDN);
  mpfr_printf ("e = %.*Rb\n", pprec, e);

  mpfr_sub (z, u, e, MPFR_RNDN);
  mpfr_add (z, z, d, MPFR_RNDN);
  mpfr_printf ("z = %.*Rb\n", pprec, z);

  mpfr_clears (LIST, (mpfr_ptr) 0);
  return 0;
}
