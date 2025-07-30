/* Test file for l2b constants.

Copyright 2007-2017 Free Software Foundation, Inc.
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

/* Execute this program with an argument to generate code that initializes
   the l2b constants. */

#include <stdio.h>
#include <stdlib.h>
#include "mpfr-test.h"

/* Must be a multiple of 4 */
static const int bits2use[] = {16, 32, 64, 96, 128, 256};
#define size_of_bits2use ((sizeof bits2use) / sizeof bits2use[0])

static __mpfr_struct l2b[BASE_MAX-1][2];

static void
print_mpfr (mpfr_srcptr x, const char *name)
{
  unsigned char temp[16];   /* buffer for the base-256 string */
  unsigned char *ptr;       /* pointer to its first non-zero byte */
  int size;                 /* size of the string */
  int i;                    /* bits2use index */
  int j;                    /* output limb index */
  int k;                    /* byte index (in output limb) */
  int r;                    /* digit index, relative to ptr */

  if (printf ("#if 0\n") < 0)
    { fprintf (stderr, "Error in printf\n"); exit (1); }
  for (i = 0; i < size_of_bits2use; i++)
    {
      if (printf ("#elif GMP_NUMB_BITS == %d\n"
                  "const mp_limb_t %s__tab[] = { 0x", bits2use[i], name) < 0)
        { fprintf (stderr, "Error in printf\n"); exit (1); }
      size = mpn_get_str (temp, 256, MPFR_MANT (x), MPFR_LIMB_SIZE (x));
      MPFR_ASSERTN (size <= 16);
      ptr = temp;
      /* Skip leading zeros. */
      while (*ptr == 0)
        {
          ptr++;
          size--;
          MPFR_ASSERTN (size > 0);
        }
      MPFR_ASSERTN (*ptr >= 128);
      for (j = (MPFR_PREC (x) - 1) / bits2use[i]; j >= 0; j--)
        {
          r = j * (bits2use[i] / 8);
          for (k = 0; k < bits2use[i] / 8; k++)
            if (printf ("%02x", r < size ? ptr[r++] : 0) < 0)
              { fprintf (stderr, "Error in printf\n"); exit (1); }
          if (printf (j == 0 ? " };\n" : ", 0x") < 0)
            { fprintf (stderr, "Error in printf\n"); exit (1); }
        }
    }
  if (printf ("#endif\n\n") < 0)
    { fprintf (stderr, "Error in printf\n"); exit (1); }
}

static void
compute_l2b (int output)
{
  mpfr_ptr p;
  mpfr_srcptr t;
  int beta, i;
  int error = 0;
  char buffer[256];  /* larger than needed, for maintainability */

  for (beta = 2; beta <= BASE_MAX; beta++)
    {
      for (i = 0; i < 2; i++)
        {
          p = &l2b[beta-2][i];

          /* Compute the value */
          if (i == 0)
            {
              /* 23-bit upper approximation to log(b)/log(2) */
              mpfr_init2 (p, 23);
              mpfr_set_ui (p, beta, MPFR_RNDU);
              mpfr_log2 (p, p, MPFR_RNDU);
            }
          else
            {
              /* 76-bit upper approximation to log(2)/log(b) */
              mpfr_init2 (p, 77);
              mpfr_set_ui (p, beta, MPFR_RNDD);
              mpfr_log2 (p, p, MPFR_RNDD);
              mpfr_ui_div (p, 1, p, MPFR_RNDU);
            }

          sprintf (buffer, "mpfr_l2b_%d_%d", beta, i);
          if (output)
            print_mpfr (p, buffer);

          /* Check the value */
          t = &__gmpfr_l2b[beta-2][i];
          if (t == NULL || MPFR_PREC (t) == 0 || !mpfr_equal_p (p, t))
            {
              if (!output)
                {
                  error = 1;
                  printf ("Error for constant %s\n", buffer);
                }
            }

          if (!output)
            mpfr_clear (p);
        }
    }

  if (output)
    {
      if (printf ("const __mpfr_struct __gmpfr_l2b[BASE_MAX-1][2] = {\n")
          < 0)
        { fprintf (stderr, "Error in printf\n"); exit (1); }
      for (beta = 2; beta <= BASE_MAX; beta++)
        {
          for (i = 0; i < 2; i++)
            {
              p = &l2b[beta-2][i];
              if (printf ("  %c {%3d,%2d,%3ld, (mp_limb_t *) "
                          "mpfr_l2b_%d_%d__tab }%s\n", i == 0 ? '{' : ' ',
                          (int) MPFR_PREC (p), MPFR_SIGN (p),
                          (long) MPFR_GET_EXP (p), beta, i,
                          i == 0 ? "," : beta < BASE_MAX ? " }," : " } };")
                  < 0)
                { fprintf (stderr, "Error in printf\n"); exit (1); }
              mpfr_clear (p);
            }
        }
    }

  /* If there was an error, the test fails. */
  if (error)
    exit (1);
}

int
main (int argc, char *argv[])
{
  if (argc != 1)
    {
      /* Generate code that initializes the l2b constants. */
      compute_l2b (1);
    }
  else
    {
      /* Check the l2b constants. */
      tests_start_mpfr ();
      compute_l2b (0);
      tests_end_mpfr ();
    }
  return 0;
}
