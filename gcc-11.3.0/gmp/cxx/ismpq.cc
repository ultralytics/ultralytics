/* operator>> -- C++-style input of mpq_t.

Copyright 2003 Free Software Foundation, Inc.

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


istream &
operator>> (istream &i, mpq_ptr q)
{
  if (! (i >> mpq_numref(q)))
    return i;

  char  c = 0;
  i.get(c); // start reading

  if (c == '/')
    {
      // skip slash, read denominator
      i.get(c);
      return __gmpz_operator_in_nowhite (i, mpq_denref(q), c);
    }
  else
    {
      // no denominator, set 1
      q->_mp_den._mp_size = 1;
      q->_mp_den._mp_d[0] = 1;
      if (i.good())
        i.putback(c);
      else if (i.eof())
        i.clear(ios::eofbit);
    }

  return i;
}
