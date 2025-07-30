/* Auxiliary functions for C++-style input of GMP types.

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

#include <cctype>
#include <iostream>
#include <string>
#include "gmp.h"
#include "gmp-impl.h"

using namespace std;


int
__gmp_istream_set_base (istream &i, char &c, bool &zero, bool &showbase)
{
  int base;

  zero = showbase = false;
  switch (i.flags() & ios::basefield)
    {
    case ios::dec:
      base = 10;
      break;
    case ios::hex:
      base = 16;
      break;
    case ios::oct:
      base = 8;
      break;
    default:
      showbase = true; // look for initial "0" or "0x" or "0X"
      if (c == '0')
	{
	  if (! i.get(c))
	    c = 0; // reset or we might loop indefinitely

	  if (c == 'x' || c == 'X')
	    {
	      base = 16;
	      i.get(c);
	    }
	  else
	    {
	      base = 8;
	      zero = true; // if no other digit is read, the "0" counts
	    }
	}
      else
	base = 10;
      break;
    }

  return base;
}

void
__gmp_istream_set_digits (string &s, istream &i, char &c, bool &ok, int base)
{
  switch (base)
    {
    case 10:
      while (isdigit(c))
	{
	  ok = true; // at least a valid digit was read
	  s += c;
	  if (! i.get(c))
	    break;
	}
      break;
    case 8:
      while (isdigit(c) && c != '8' && c != '9')
	{
	  ok = true; // at least a valid digit was read
	  s += c;
	  if (! i.get(c))
	    break;
	}
      break;
    case 16:
      while (isxdigit(c))
	{
	  ok = true; // at least a valid digit was read
	  s += c;
	  if (! i.get(c))
	    break;
	}
      break;
    }
}
