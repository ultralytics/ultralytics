/* Test file for mpfr_cmp.

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

int
main (void)
{
  double x, y;
  mpfr_t xx, yy;
  int c;
  long i;
  mpfr_prec_t p;

  tests_start_mpfr ();

  mpfr_init (xx);
  mpfr_init (yy);

  mpfr_set_prec (xx, 2);
  mpfr_set_prec (yy, 2);
  mpfr_set_str_binary(xx, "-0.10E0");
  mpfr_set_str_binary(yy, "-0.10E0");
  if ((mpfr_cmp) (xx, yy))
    {
      printf ("mpfr_cmp (xx, yy) returns non-zero for prec=2\n");
      exit (1);
    }

  mpfr_set_prec (xx, 65);
  mpfr_set_prec (yy, 65);
  mpfr_set_str_binary(xx, "0.10011010101000110101010000000011001001001110001011101011111011101E623");
  mpfr_set_str_binary(yy, "0.10011010101000110101010000000011001001001110001011101011111011100E623");
  p = 0;
  if (mpfr_cmp2 (xx, yy, &p) <= 0 || p != 64)
    {
      printf ("Error (1) in mpfr_cmp2\n");
      exit (1);
    }
  mpfr_set_str_binary(xx, "0.10100010001110110111000010001000010011111101000100011101000011100");
  mpfr_set_str_binary(yy, "0.10100010001110110111000010001000010011111101000100011101000011011");
  p = 0;
  if (mpfr_cmp2 (xx, yy, &p) <= 0 || p != 64)
    {
      printf ("Error (2) in mpfr_cmp2\n");
      exit (1);
    }

  mpfr_set_prec (xx, 160); mpfr_set_prec (yy, 160);
  mpfr_set_str_binary (xx, "0.1E1");
  mpfr_set_str_binary (yy, "0.1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100000110001110100");
  p = 0;
  if (mpfr_cmp2 (xx, yy, &p) <= 0 || p != 144)
    {
      printf ("Error (3) in mpfr_cmp2\n");
      exit (1);
    }

  mpfr_set_prec (xx, 53);
  mpfr_set_prec (yy, 200);
  mpfr_set_ui (xx, 1, (mpfr_rnd_t) 0);
  mpfr_set_ui (yy, 1, (mpfr_rnd_t) 0);
  if (mpfr_cmp (xx, yy) != 0)
    {
      printf ("Error in mpfr_cmp: 1.0 != 1.0\n");
      exit (1);
    }
  mpfr_set_prec (yy, 31);
  mpfr_set_str (xx, "1.0000000002", 10, (mpfr_rnd_t) 0);
  mpfr_set_ui (yy, 1, (mpfr_rnd_t) 0);
  if (!(mpfr_cmp (xx,yy)>0))
    {
      printf ("Error in mpfr_cmp: not 1.0000000002 > 1.0\n");
      exit (1);
    }
  mpfr_set_prec (yy, 53);

  /* bug found by Gerardo Ballabio */
  mpfr_set_ui(xx, 0, MPFR_RNDN);
  mpfr_set_str (yy, "0.1", 10, MPFR_RNDN);
  if ((c = mpfr_cmp (xx, yy)) >= 0)
    {
      printf ("Error in mpfr_cmp(0.0, 0.1), gives %d\n", c);
      exit (1);
    }

  mpfr_set_inf (xx, 1);
  mpfr_set_str (yy, "-23489745.0329", 10, MPFR_RNDN);
  if ((c = mpfr_cmp (xx, yy)) <= 0)
    {
      printf ("Error in mpfr_cmp(Infp, 23489745.0329), gives %d\n", c);
      exit (1);
    }

  mpfr_set_inf (xx, 1);
  mpfr_set_inf (yy, -1);
  if ((c = mpfr_cmp (xx, yy)) <= 0)
    {
      printf ("Error in mpfr_cmp(Infp, Infm), gives %d\n", c);
      exit (1);
    }

  mpfr_set_inf (xx, -1);
  mpfr_set_inf (yy, 1);
  if ((c = mpfr_cmp (xx, yy)) >= 0)
    {
      printf ("Error in mpfr_cmp(Infm, Infp), gives %d\n", c);
      exit (1);
    }

  mpfr_set_inf (xx, 1);
  mpfr_set_inf (yy, 1);
  if ((c = mpfr_cmp (xx, yy)) != 0)
    {
      printf ("Error in mpfr_cmp(Infp, Infp), gives %d\n", c);
      exit (1);
    }

  mpfr_set_inf (xx, -1);
  mpfr_set_inf (yy, -1);
  if ((c = mpfr_cmp (xx, yy)) != 0)
    {
      printf ("Error in mpfr_cmp(Infm, Infm), gives %d\n", c);
      exit (1);
    }

  mpfr_set_inf (xx, -1);
  mpfr_set_str (yy, "2346.09234", 10, MPFR_RNDN);
  if ((c = mpfr_cmp (xx, yy)) >= 0)
    {
      printf ("Error in mpfr_cmp(Infm, 2346.09234), gives %d\n", c);
      exit (1);
    }

  mpfr_set_ui (xx, 0, MPFR_RNDN);
  mpfr_set_ui (yy, 1, MPFR_RNDN);
  if ((c = mpfr_cmp3 (xx, yy, 1)) >= 0)
    {
      printf ("Error: mpfr_cmp3 (0, 1, 1) gives %d instead of"
              " a negative value\n", c);
      exit (1);
    }
  if ((c = mpfr_cmp3 (xx, yy, -1)) <= 0)
    {
      printf ("Error: mpfr_cmp3 (0, 1, -1) gives %d instead of"
              " a positive value\n", c);
      exit (1);
    }

  for (i=0; i<500000; )
    {
      x = DBL_RAND ();
      y = DBL_RAND ();
      if (!Isnan(x) && !Isnan(y))
        {
          i++;
          mpfr_set_d (xx, x, MPFR_RNDN);
          mpfr_set_d (yy, y, MPFR_RNDN);
          c = mpfr_cmp (xx,yy);
          if ((c>0 && x<=y) || (c==0 && x!=y) || (c<0 && x>=y))
            {
              printf ("Error in mpfr_cmp with x=%1.20e, y=%1.20e"
                      " mpfr_cmp(x,y)=%d\n", x, y, c);
              exit (1);
            }
        }
    }

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
