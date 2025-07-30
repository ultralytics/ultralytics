/* gen-jacobi.c

   Contributed to the GNU project by Niels Möller.

Copyright 2010 Free Software Foundation, Inc.

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

/* Generate the lookup table needed for fast left-to-right computation
   of the Jacobi symbol. */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static const struct
{
  unsigned char a;
  unsigned char b;
} decode_table[13] = {
  /*  0 */ { 0, 1 },
  /*  1 */ { 0, 3 },
  /*  2 */ { 1, 1 },
  /*  3 */ { 1, 3 },
  /*  4 */ { 2, 1 },
  /*  5 */ { 2, 3 },
  /*  6 */ { 3, 1 },
  /*  7 */ { 3, 3 }, /* d = 1 */
  /*  8 */ { 1, 0 },
  /*  9 */ { 1, 2 },
  /* 10 */ { 3, 0 },
  /* 11 */ { 3, 2 },
  /* 12 */ { 3, 3 }, /* d = 0 */

};
#define JACOBI_A(bits) (decode_table[(bits)>>1].a)
#define JACOBI_B(bits) (decode_table[(bits)>>1].b)

#define JACOBI_E(bits) ((bits) & 1)
#define JACOBI_D(bits) (((bits)>>1) == 7) /* Gives 0 for don't care states. */

static unsigned
encode (unsigned a, unsigned b, unsigned d)
{
  unsigned i;

  assert (d < 2);
  assert (a < 4);
  assert (b < 4);
  assert ( (a | b ) & 1);

  if (a == 3 && b == 3)
    return d ? 7 : 12;

  for (i = 0; i < 12; i++)
    if (decode_table[i].a == a
	&& decode_table[i].b == b)
      return i;

  abort ();
}

int
main (int argc, char **argv)
{
  unsigned bits;

  for (bits = 0; bits < 208; bits++)
    {
      unsigned e, a, b, d_old, d, q;

      if (bits && !(bits & 0xf))
	printf("\n");

      q = bits & 3;
      d = (bits >> 2) & 1;

      e = JACOBI_E (bits >> 3);
      a = JACOBI_A (bits >> 3);
      b = JACOBI_B (bits >> 3);
      d_old = JACOBI_D (bits >> 3);

      if (d != d_old && a == 3 && b == 3)
	e ^= 1;

      if (d == 1)
	{
	  if (b == 2)
	    e ^= (q & (a >> 1)) ^ (q >> 1);
	  a = (a - q * b) & 3;
	}
      else
	{
	  if (a == 2)
	    e ^= (q & (b >> 1)) ^ (q >> 1);
	  b = (b - q * a) & 3;
	}

      printf("%2d,", (encode (a, b, d) << 1) | e);
    }
  printf("\n");

  return 0;
}
