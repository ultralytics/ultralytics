/* operator>> -- C++-style input of mpf_t.

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
#include <clocale>    // for localeconv

#include "gmp.h"
#include "gmp-impl.h"

using namespace std;


// For g++ libstdc++ parsing see num_get<chartype,initer>::_M_extract_float
// in include/bits/locale_facets.tcc.
//
// There are no plans to accept hex or octal floats, not unless the standard
// C++ library does so.  Although such formats might be of use, it's
// considered more important to be compatible with what the normal
// operator>> does on "double"s etc.

istream &
operator>> (istream &i, mpf_ptr f)
{
  int base;
  char c = 0;
  string s;
  bool ok = false;

  // C decimal point, as expected by mpf_set_str
  const char *lconv_point = GMP_DECIMAL_POINT;

  // C++ decimal point
#if HAVE_STD__LOCALE
  const locale& loc = i.getloc();
  char point_char = use_facet< numpunct<char> >(loc).decimal_point();
#else
  const char *point = lconv_point;
  char point_char = *point;
#endif

  i.get(c); // start reading

  if (i.flags() & ios::skipws) // skip initial whitespace
    {
      // C++ isspace
#if HAVE_STD__LOCALE
      const ctype<char>& ct = use_facet< ctype<char> >(loc);
#define cxx_isspace(c)  (ct.is(ctype_base::space,(c)))
#else
#define cxx_isspace(c)  isspace(c)
#endif

      while (cxx_isspace(c) && i.get(c))
        ;
    }

  if (c == '-' || c == '+') // sign
    {
      if (c == '-')
	s = "-";
      i.get(c);
    }

  base = 10;
  __gmp_istream_set_digits(s, i, c, ok, base); // read the number

  // look for the C++ radix point, but put the C one in for mpf_set_str
  if (c == point_char)
    {
#if HAVE_STD__LOCALE
      i.get(c);
#else // lconv point can be multi-char
      for (;;)
        {
          i.get(c);
          point++;
          if (*point == '\0')
            break;
          if (c != *point)
            goto fail;
        }
#endif
      s += lconv_point;
      __gmp_istream_set_digits(s, i, c, ok, base); // read the mantissa
    }

  if (ok && (c == 'e' || c == 'E')) // exponent
    {
      s += c;
      i.get(c);
      ok = false; // exponent is mandatory

      if (c == '-' || c == '+') // sign
	{
	  s += c;
	  i.get(c);
	}

      __gmp_istream_set_digits(s, i, c, ok, base); // read the exponent
    }

  if (i.good()) // last character read was non-numeric
    i.putback(c);
  else if (i.eof() && ok) // stopped just before eof
    i.clear(ios::eofbit);

  if (ok)
    ASSERT_NOCARRY (mpf_set_str(f, s.c_str(), base)); // extract the number
  else
    {
    fail:
      i.setstate(ios::failbit); // read failed
    }

  return i;
}
