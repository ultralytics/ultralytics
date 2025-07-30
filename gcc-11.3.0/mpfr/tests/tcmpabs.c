/* Test file for mpfr_cmpabs.

Copyright 2004-2017 Free Software Foundation, Inc.
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

#define ERROR(s) do { printf(s); exit(1); } while (0)

int
main (void)
{
  mpfr_t xx, yy;
  int c;

  tests_start_mpfr ();

  mpfr_init2 (xx, 2);
  mpfr_init2 (yy, 2);

  mpfr_clear_erangeflag ();
  MPFR_SET_NAN (xx);
  MPFR_SET_NAN (yy);
  if (mpfr_cmpabs (xx, yy) != 0)
    ERROR ("mpfr_cmpabs (NAN,NAN) returns non-zero\n");
  if (!mpfr_erangeflag_p ())
    ERROR ("mpfr_cmpabs (NAN,NAN) doesn't set erange flag\n");

  mpfr_set_str_binary (xx, "0.10E0");
  mpfr_set_str_binary (yy, "-0.10E0");
  if (mpfr_cmpabs (xx, yy) != 0)
    ERROR ("mpfr_cmpabs (xx, yy) returns non-zero for prec=2\n");

  mpfr_set_prec (xx, 65);
  mpfr_set_prec (yy, 65);
  mpfr_set_str_binary (xx, "-0.10011010101000110101010000000011001001001110001011101011111011101E623");
  mpfr_set_str_binary (yy, "0.10011010101000110101010000000011001001001110001011101011111011100E623");
  if (mpfr_cmpabs (xx, yy) <= 0)
    ERROR ("Error (1) in mpfr_cmpabs\n");

  mpfr_set_str_binary (xx, "-0.10100010001110110111000010001000010011111101000100011101000011100");
  mpfr_set_str_binary (yy, "-0.10100010001110110111000010001000010011111101000100011101000011011");
  if (mpfr_cmpabs (xx, yy) <= 0)
    ERROR ("Error (2) in mpfr_cmpabs\n");

  mpfr_set_prec (xx, 160);
  mpfr_set_prec (yy, 160);
  mpfr_set_str_binary (xx, "0.1E1");
  mpfr_set_str_binary (yy, "-0.1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100000110001110100");
  if (mpfr_cmpabs (xx, yy) <= 0)
    ERROR ("Error (3) in mpfr_cmpabs\n");

  mpfr_set_prec(xx, 53);
  mpfr_set_prec(yy, 200);
  mpfr_set_ui (xx, 1, (mpfr_rnd_t) 0);
  mpfr_set_ui (yy, 1, (mpfr_rnd_t) 0);
  if (mpfr_cmpabs(xx, yy) != 0)
    ERROR ("Error in mpfr_cmpabs: 1.0 != 1.0\n");

  mpfr_set_prec (yy, 31);
  mpfr_set_str (xx, "-1.0000000002", 10, (mpfr_rnd_t) 0);
  mpfr_set_ui (yy, 1, (mpfr_rnd_t) 0);
  if (!(mpfr_cmpabs(xx,yy)>0))
    ERROR ("Error in mpfr_cmpabs: not 1.0000000002 > 1.0\n");
  mpfr_set_prec(yy, 53);

  mpfr_set_ui(xx, 0, MPFR_RNDN);
  mpfr_set_str (yy, "-0.1", 10, MPFR_RNDN);
  if (mpfr_cmpabs(xx, yy) >= 0)
    ERROR ("Error in mpfr_cmpabs(0.0, 0.1)\n");

  mpfr_set_inf (xx, -1);
  mpfr_set_str (yy, "23489745.0329", 10, MPFR_RNDN);
  if (mpfr_cmpabs(xx, yy) <= 0)
    ERROR ("Error in mpfr_cmp(-Inf, 23489745.0329)\n");

  mpfr_set_inf (xx, 1);
  mpfr_set_inf (yy, -1);
  if (mpfr_cmpabs(xx, yy) != 0)
    ERROR ("Error in mpfr_cmpabs(Inf, -Inf)\n");

  mpfr_set_inf (yy, -1);
  mpfr_set_str (xx, "2346.09234", 10, MPFR_RNDN);
  if (mpfr_cmpabs (xx, yy) >= 0)
    ERROR ("Error in mpfr_cmpabs(-Inf, 2346.09234)\n");

  mpfr_set_prec (xx, 2);
  mpfr_set_prec (yy, 128);
  mpfr_set_str_binary (xx, "0.1E10");
  mpfr_set_str_binary (yy,
                       "0.100000000000000000000000000000000000000000000000"
                       "00000000000000000000000000000000000000000000001E10");
  if (mpfr_cmpabs (xx, yy) >= 0)
    ERROR ("Error in mpfr_cmpabs(10.235, 2346.09234)\n");
  mpfr_swap (xx, yy);
  if (mpfr_cmpabs(xx, yy) <= 0)
    ERROR ("Error in mpfr_cmpabs(2346.09234, 10.235)\n");
  mpfr_swap (xx, yy);

  /* Check for NAN */
  mpfr_set_nan (xx);
  mpfr_clear_erangeflag ();
  c = (mpfr_cmp) (xx, yy);
  if (c != 0 || !mpfr_erangeflag_p () )
    {
      printf ("NAN error (1)\n");
      exit (1);
    }
  mpfr_clear_erangeflag ();
  c = (mpfr_cmp) (yy, xx);
  if (c != 0 || !mpfr_erangeflag_p () )
    {
      printf ("NAN error (2)\n");
      exit (1);
    }
  mpfr_clear_erangeflag ();
  c = (mpfr_cmp) (xx, xx);
  if (c != 0 || !mpfr_erangeflag_p () )
    {
      printf ("NAN error (3)\n");
      exit (1);
    }

  mpfr_clear (xx);
  mpfr_clear (yy);

  tests_end_mpfr ();
  return 0;
}
