/* mpf_inp_str(dest_float, stream, base) -- Input a number in base
   BASE from stdio stream STREAM and store the result in DEST_FLOAT.

Copyright 1996, 2000-2002, 2005 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <ctype.h>
#include "gmp.h"
#include "gmp-impl.h"

size_t
mpf_inp_str (mpf_ptr rop, FILE *stream, int base)
{
  char *str;
  size_t alloc_size, str_size;
  int c;
  int res;
  size_t nread;

  if (stream == 0)
    stream = stdin;

  alloc_size = 100;
  str = (char *) (*__gmp_allocate_func) (alloc_size);
  str_size = 0;
  nread = 0;

  /* Skip whitespace.  */
  do
    {
      c = getc (stream);
      nread++;
    }
  while (isspace (c));

  for (;;)
    {
      if (str_size >= alloc_size)
	{
	  size_t old_alloc_size = alloc_size;
	  alloc_size = alloc_size * 3 / 2;
	  str = (char *) (*__gmp_reallocate_func) (str, old_alloc_size, alloc_size);
	}
      if (c == EOF || isspace (c))
	break;
      str[str_size++] = c;
      c = getc (stream);
    }
  ungetc (c, stream);
  nread--;

  if (str_size >= alloc_size)
    {
      size_t old_alloc_size = alloc_size;
      alloc_size = alloc_size * 3 / 2;
      str = (char *) (*__gmp_reallocate_func) (str, old_alloc_size, alloc_size);
    }
  str[str_size] = 0;

  res = mpf_set_str (rop, str, base);
  (*__gmp_free_func) (str, alloc_size);

  if (res == -1)
    return 0;			/* error */

  return str_size + nread;
}
