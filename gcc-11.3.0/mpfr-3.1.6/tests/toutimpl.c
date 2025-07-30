/* Test file for internal debugging-out functions:
   mpfr_dump, mpfr_print_binary, mpfr_print_rnd_mode.

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

#include <stdlib.h>
#include "mpfr-test.h"

#define FILE_NAME "toutimpl_out.txt"

static const char Buffer[] =
"@NaN@\n"
"-@Inf@\n"
"-0\n"
"0.10101010101011111001000110001100010000100000000000000E32\n";

int
main (void)
{
  mpfr_t x;
  FILE *f;
  int i;

  tests_start_mpfr ();

  /* Check RND_MODE */
  if (strcmp (mpfr_print_rnd_mode(MPFR_RNDN), "MPFR_RNDN"))
    {
      printf ("Error for printing MPFR_RNDN\n");
      exit (1);
    }
  if (strcmp (mpfr_print_rnd_mode(MPFR_RNDU), "MPFR_RNDU"))
    {
      printf ("Error for printing MPFR_RNDU\n");
      exit (1);
    }
  if (strcmp (mpfr_print_rnd_mode(MPFR_RNDD), "MPFR_RNDD"))
    {
      printf ("Error for printing MPFR_RNDD\n");
      exit (1);
    }
  if (strcmp (mpfr_print_rnd_mode(MPFR_RNDZ), "MPFR_RNDZ"))
    {
      printf ("Error for printing MPFR_RNDZ\n");
      exit (1);
    }
  if (mpfr_print_rnd_mode ((mpfr_rnd_t) -1) != NULL ||
      mpfr_print_rnd_mode (MPFR_RND_MAX) != NULL)
    {
      printf ("Error for illegal rounding mode values.\n");
      exit (1);
    }

  /* Reopen stdout to a file. All errors will be put to stderr. */
  if (freopen (FILE_NAME, "w", stdout) == NULL)
    {
      printf ("Error can't redirect stdout\n");
      exit (1);
    }
  mpfr_init (x);
  mpfr_set_nan (x);
  mpfr_dump (x);
  mpfr_set_inf (x, -1);
  mpfr_dump (x);
  MPFR_SET_ZERO (x); MPFR_SET_NEG (x);
  mpfr_dump (x);
  mpfr_set_str_binary (x, "0.101010101010111110010001100011000100001E32");
  mpfr_dump (x);
  mpfr_print_mant_binary ("x=",MPFR_MANT(x), MPFR_PREC(x));


  mpfr_clear (x);
  fclose (stdout);
  /* Open it and check for it */
  f = fopen (FILE_NAME, "r");
  if (f == NULL)
    {
      fprintf (stderr, "Can't reopen file!\n");
      exit (1);
    }
  for(i = 0 ; i < sizeof(Buffer)-1 ; i++)
    {
      if (feof (f))
        {
          fprintf (stderr, "Error EOF\n");
          exit (1);
        }
      if (Buffer[i] != fgetc (f))
        {
          fprintf (stderr, "Character mismatch for i=%d / %lu\n",
                  i, (unsigned long) sizeof(Buffer));
          exit (1);
        }
    }
  fclose (f);

  remove (FILE_NAME);
  tests_end_mpfr ();
  return 0;
}
