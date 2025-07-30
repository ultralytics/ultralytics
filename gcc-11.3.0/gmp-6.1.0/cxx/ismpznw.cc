/* __gmpz_operator_in_nowhite -- C++-style input of mpz_t, no whitespace skip.

Copyright 2001, 2003 Free Software Foundation, Inc.

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


// For g++ libstdc++ parsing see num_get<chartype,initer>::_M_extract_int in
// include/bits/locale_facets.tcc.

istream &
__gmpz_operator_in_nowhite (istream &i, mpz_ptr z, char c)
{
  int base;
  string s;
  bool ok = false, zero, showbase;

  if (c == '-' || c == '+') // sign
    {
      if (c == '-') // mpz_set_str doesn't accept '+'
	s = "-";
      i.get(c);
    }

  base = __gmp_istream_set_base(i, c, zero, showbase); // select the base
  __gmp_istream_set_digits(s, i, c, ok, base);         // read the number

  if (i.good()) // last character read was non-numeric
    i.putback(c);
  else if (i.eof() && (ok || zero)) // stopped just before eof
    i.clear(ios::eofbit);

  if (ok)
    ASSERT_NOCARRY (mpz_set_str (z, s.c_str(), base)); // extract the number
  else if (zero)
    mpz_set_ui(z, 0);
  else
    i.setstate(ios::failbit); // read failed

  return i;
}
