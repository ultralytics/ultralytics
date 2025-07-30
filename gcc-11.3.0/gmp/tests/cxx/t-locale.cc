/* Test locale support in C++ functions.

Copyright 2001-2003, 2007 Free Software Foundation, Inc.

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

#include <clocale>
#include <iostream>
#include <cstdlib>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

using namespace std;


extern "C" {
  char point_string[2];
}

#if HAVE_STD__LOCALE
// Like std::numpunct, but with decimal_point coming from point_string[].
class my_numpunct : public numpunct<char> {
 public:
  explicit my_numpunct (size_t r = 0) : numpunct<char>(r) { }
 protected:
  char do_decimal_point() const { return point_string[0]; }
};
#endif

void
set_point (char c)
{
  point_string[0] = c;

#if HAVE_STD__LOCALE
  locale loc (locale::classic(), new my_numpunct ());
  locale::global (loc);
#endif
}


void
check_input (void)
{
  static const struct {
    const char  *str1;
    const char  *str2;
    double      want;
  } data[] = {

    { "1","",   1.0 },
    { "1","0",  1.0 },
    { "1","00", 1.0 },

    { "","5",    0.5 },
    { "0","5",   0.5 },
    { "00","5",  0.5 },
    { "00","50", 0.5 },

    { "1","5",    1.5 },
    { "1","5e1", 15.0 },
  };

  static char point[] = {
    '.', ',', 'x', '\xFF'
  };

  mpf_t  got;
  mpf_init (got);

  for (size_t i = 0; i < numberof (point); i++)
    {
      set_point (point[i]);

      for (int neg = 0; neg <= 1; neg++)
        {
          for (size_t j = 0; j < numberof (data); j++)
            {
              string str = string(data[j].str1)+point[i]+string(data[j].str2);
              if (neg)
                str = "-" + str;

              istringstream is (str.c_str());

              mpf_set_ui (got, 123);   // dummy initial value

              if (! (is >> got))
                {
                  cout << "istream mpf_t operator>> error\n";
                  cout << "  point " << point[i] << "\n";
                  cout << "  str   \"" << str << "\"\n";
                  cout << "  localeconv point \""
                       << GMP_DECIMAL_POINT << "\"\n";
                  abort ();
                }

              double want = data[j].want;
              if (neg)
                want = -want;
              if (mpf_cmp_d (got, want) != 0)
                {
                  cout << "istream mpf_t operator>> wrong\n";
                  cout << "  point " << point[i] << "\n";
                  cout << "  str   \"" << str << "\"\n";
                  cout << "  got   " << got << "\n";
                  cout << "  want  " << want << "\n";
                  cout << "  localeconv point \""
                       << GMP_DECIMAL_POINT << "\"\n";
                  abort ();
                }
            }
        }
    }

  mpf_clear (got);
}

void
check_output (void)
{
  static char point[] = {
    '.', ',', 'x', '\xFF'
  };

  for (size_t i = 0; i < numberof (point); i++)
    {
      set_point (point[i]);
      ostringstream  got;

      mpf_t  f;
      mpf_init (f);
      mpf_set_d (f, 1.5);
      got << f;
      mpf_clear (f);

      string  want = string("1") + point[i] + string("5");

      if (want.compare (got.str()) != 0)
        {
          cout << "ostream mpf_t operator<< doesn't respect locale\n";
          cout << "  point " << point[i] << "\n";
          cout << "  got   \"" << got.str() << "\"\n";
          cout << "  want  \"" << want      << "\"\n";
          abort ();
        }
    }
}

int
replacement_works (void)
{
  set_point ('x');
  mpf_t  f;
  mpf_init (f);
  mpf_set_d (f, 1.5);
  ostringstream s;
  s << f;
  mpf_clear (f);

  return (s.str().compare("1x5") == 0);
}

int
main (void)
{
  tests_start ();

  if (replacement_works())
    {
      check_input ();
      check_output ();
    }
  else
    {
      cout << "Replacing decimal point didn't work, tests skipped\n";
    }

  tests_end ();
  return 0;
}
