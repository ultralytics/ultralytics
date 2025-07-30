/* mpq_inp_str -- read an mpq from a FILE.

Copyright 2001 Free Software Foundation, Inc.

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
mpq_inp_str (mpq_ptr q, FILE *fp, int base)
{
  size_t  nread;
  int     c;

  if (fp == NULL)
    fp = stdin;

  SIZ(DEN(q)) = 1;
  PTR(DEN(q))[0] = 1;

  nread = mpz_inp_str (mpq_numref(q), fp, base);
  if (nread == 0)
    return 0;

  c = getc (fp);
  nread++;

  if (c == '/')
    {
      c = getc (fp);
      nread++;

      nread = mpz_inp_str_nowhite (mpq_denref(q), fp, base, c, nread);
      if (nread == 0)
	{
	  SIZ(NUM(q)) = 0;
	  SIZ(DEN(q)) = 1;
	  PTR(DEN(q))[0] = 1;
	}
    }
  else
    {
      ungetc (c, fp);
      nread--;
    }

  return nread;
}
