/* gmp_randinit_lc_2exp_size -- initialize a random state with a linear
   congruential generator of a requested size.

Copyright 1999-2001 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include <stdio.h> /* for NULL */
#include "gmp.h"
#include "gmp-impl.h"


/* Array of LC-schemes, ordered in increasing order of the first
   member (the 'm2exp' value).  The end of the array is indicated with
   an entry containing all zeros.  */

/* All multipliers are in the range 0.01*m and 0.99*m, and are
congruent to 5 (mod 8).
They all pass the spectral test with Vt >= 2^(30/t) and merit >= 1.
(Up to and including 196 bits, merit is >= 3.)  */

struct __gmp_rand_lc_scheme_struct
{
  unsigned long int m2exp;	/* Modulus is 2 ^ m2exp. */
  const char *astr;		/* Multiplier in string form. */
  unsigned long int c;		/* Addend. */
};

static const struct __gmp_rand_lc_scheme_struct __gmp_rand_lc_scheme[] =
{
  {32, "29CF535",	     1},
  {33, "51F666D",	     1},
  {34, "A3D73AD",	     1},
  {35, "147E5B85",	     1},
  {36, "28F725C5",	     1},
  {37, "51EE3105",	     1},
  {38, "A3DD5CDD",	     1},
  {39, "147AF833D",	     1},
  {40, "28F5DA175",	     1},
  {56, "AA7D735234C0DD",  1},
  {64, "BAECD515DAF0B49D", 1},
  {100, "292787EBD3329AD7E7575E2FD", 1},
  {128, "48A74F367FA7B5C8ACBB36901308FA85", 1},
  {156, "78A7FDDDC43611B527C3F1D760F36E5D7FC7C45", 1},
  {196, "41BA2E104EE34C66B3520CE706A56498DE6D44721E5E24F5", 1},
  {200, "4E5A24C38B981EAFE84CD9D0BEC48E83911362C114F30072C5", 1},
  {256, "AF66BA932AAF58A071FD8F0742A99A0C76982D648509973DB802303128A14CB5", 1},
  {0, NULL, 0}			/* End of array. */
};

int
gmp_randinit_lc_2exp_size (gmp_randstate_t rstate, mp_bitcnt_t size)
{
  const struct __gmp_rand_lc_scheme_struct *sp;
  mpz_t a;

  /* Pick a scheme.  */
  for (sp = __gmp_rand_lc_scheme; sp->m2exp != 0; sp++)
    if (sp->m2exp / 2 >= size)
      goto found;
  return 0;

 found:
  /* Install scheme.  */
  mpz_init_set_str (a, sp->astr, 16);
  gmp_randinit_lc_2exp (rstate, a, sp->c, sp->m2exp);
  mpz_clear (a);
  return 1;
}
