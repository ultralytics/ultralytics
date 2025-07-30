/* Test file for mpfr_inp_str.

Copyright 2004, 2006-2017 Free Software Foundation, Inc.
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
main (int argc, char *argv[])
{
  mpfr_t x;
  mpfr_t y;
  FILE *f;
  int i;
  tests_start_mpfr ();

  mpfr_init (x);
  mpfr_init (y);

  mpfr_set_prec (x, 15);
  f = src_fopen ("inp_str.data", "r");
  if (f == NULL)
    {
      printf ("Error, can't open inp_str.data\n");
      exit (1);
    }
  i = mpfr_inp_str (x, f, 10, MPFR_RNDN);
  if (i == 0 || mpfr_cmp_ui (x, 31415))
    {
      printf ("Error in reading 1st line from file inp_str.data (%d)\n", i);
      mpfr_dump (x);
      exit (1);
    }
  getc (f);
  i = mpfr_inp_str (x, f, 10, MPFR_RNDN);
  if ((i == 0) || mpfr_cmp_ui (x, 31416))
    {
      printf ("Error in reading 2nd line from file inp_str.data (%d)\n", i);
      mpfr_dump (x);
      exit (1);
    }
  getc (f);
  i = mpfr_inp_str (x, f, 10, MPFR_RNDN);
  if (i != 0)
    {
      printf ("Error in reading 3rd line from file inp_str.data (%d)\n", i);
      mpfr_dump (x);
      exit (1);
    }

  mpfr_set_prec (x, 53);
  mpfr_set_prec (y, 53);
  mpfr_set_str (y, "1.0010010100001110100101001110011010111011100001110010e226",
                2, MPFR_RNDN);
  for (i = 2; i < 63; i++)
    {
      getc (f);
      if (mpfr_inp_str (x, f, i, MPFR_RNDN) == 0 || !mpfr_equal_p (x, y))
        {
          printf ("Error in reading %dth line from file inp_str.data\n", i+2);
          mpfr_dump (x);
          exit (1);
        }
      mpfr_set_ui (x, 0, MPFR_RNDN);
    }

  fclose (f);

  mpfr_clear (x);
  mpfr_clear (y);

  tests_end_mpfr ();
  return 0;
}
