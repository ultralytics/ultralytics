/*

Copyright 2013 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "testutils.h"

#define MAX_WORDS 20
#define MAX_WORD_SIZE 10

static void
dump_bytes (const char *label, const unsigned char *s, size_t n)
{
  size_t i;
  fprintf (stderr, "%s:", label);
  for (i = 0; i < n; i++)
    {
      if (i && (i % 16) == 0)
	fprintf (stderr, "\n");
      fprintf (stderr, " %02x", s[i]);
    }
  fprintf (stderr, "\n");
}

/* Tests both mpz_import and mpz_export. */
void
testmain (int argc, char **argv)
{
  unsigned char input[MAX_WORDS * MAX_WORD_SIZE];
  unsigned char output[MAX_WORDS * MAX_WORD_SIZE + 2];
  size_t count, in_count, out_count, size;
  int endian, order;

  mpz_t a, res;

  mpz_init (a);
  mpz_init (res);

  for (size = 0; size <= MAX_WORD_SIZE; size++)
    for (count = 0; count <= MAX_WORDS; count++)
      for (endian = -1; endian <= 1; endian++)
	for (order = -1; order <= 1; order += 2)
	  {
	    mini_rrandomb_export (a, input, &in_count,
				  order, size, endian, size*count * 8);
	    mpz_import (res, in_count, order, size, endian, 0, input);
	    if (mpz_cmp (a, res))
	      {
		fprintf (stderr, "mpz_import failed:\n"
			 "in_count %lu, out_count %lu, endian = %d, order = %d\n",
			 (unsigned long) in_count, (unsigned long) out_count, endian, order);
		dump ("a", a);
		dump ("res", res);
		abort ();
	      }
	    output[0] = 17;
	    output[1+in_count*size] = 17;

	    mpz_export (output+1, &out_count, order, size, endian, 0, a);
	    if (out_count != in_count
		|| memcmp (output+1, input, in_count * size)
		|| output[0] != 17
		|| output[1+in_count*size] != 17)
	      {
		fprintf (stderr, "mpz_export failed:\n"
			 "in_count %lu, out_count %lu, endian = %d, order = %d\n",
			 (unsigned long) in_count, (unsigned long) out_count, endian, order);
		dump_bytes ("input", input, in_count * size);
		dump_bytes ("output", output+1, out_count * size);
		if (output[0] != 17)
		  fprintf (stderr, "Overwrite at -1, value %02x\n", output[0]);
		if (output[1+in_count*size] != 17)
		  fprintf (stderr, "Overwrite at %lu, value %02x\n",
			   (unsigned long) (in_count*size), output[1+in_count*size]);

		abort ();
	      }
	  }
  mpz_clear (a);
  mpz_clear (res);
}
