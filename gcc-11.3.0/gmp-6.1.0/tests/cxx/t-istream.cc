/* Test istream formatted input.

Copyright 2001-2004 Free Software Foundation, Inc.

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

#include <iostream>
#include <cstdlib>
#include <cstring>

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

using namespace std;


// Under option_check_standard, the various test cases for mpz operator>>
// are put through the standard operator>> for long, and likewise mpf
// operator>> is put through double.
//
// In g++ 3.3 this results in some printouts about the final position
// indicated for something like ".e123".  Our mpf code stops at the "e"
// since there's no mantissa digits, but g++ reads the whole thing and only
// then decides it's bad.

bool option_check_standard = false;


// On some versions of g++ 2.96 it's been observed that putback() may leave
// tellg() unchanged.  We believe this is incorrect and presumably the
// result of a bug, since for instance it's ok in g++ 2.95 and g++ 3.3.  We
// detect the problem at runtime and disable affected checks.

bool putback_tellg_works = true;

void
check_putback_tellg (void)
{
  istringstream input ("hello");
  streampos  old_pos, new_pos;
  char  c;

  input.get(c);
  old_pos = input.tellg();
  input.putback(c);
  new_pos = input.tellg();

  if (old_pos == new_pos)
    {
      cout << "Warning, istringstream has a bug: putback() doesn't update tellg().\n";;
      cout << "Tests on tellg() will be skipped.\n";
      putback_tellg_works = false;
    }
}


#define WRONG(str)                                              \
  do {                                                          \
    cout << str ", data[" << i << "]\n";                        \
    cout << "  input: \"" << data[i].input << "\"\n";           \
    cout << "  flags: " << hex << input.flags() << dec << "\n"; \
  } while (0)

void
check_mpz (void)
{
  static const struct {
    const char     *input;
    int            want_pos;
    const char     *want;
    ios::fmtflags  flags;

  } data[] = {

    { "0",      -1, "0",    (ios::fmtflags) 0 },
    { "123",    -1, "123",  (ios::fmtflags) 0 },
    { "0123",   -1, "83",   (ios::fmtflags) 0 },
    { "0x123",  -1, "291",  (ios::fmtflags) 0 },
    { "-123",   -1, "-123", (ios::fmtflags) 0 },
    { "-0123",  -1, "-83",  (ios::fmtflags) 0 },
    { "-0x123", -1, "-291", (ios::fmtflags) 0 },
    { "+123",   -1, "123", (ios::fmtflags) 0 },
    { "+0123",  -1, "83",  (ios::fmtflags) 0 },
    { "+0x123", -1, "291", (ios::fmtflags) 0 },

    { "0",     -1, "0",    ios::dec },
    { "1f",     1, "1",    ios::dec },
    { "011f",   3, "11",   ios::dec },
    { "123",   -1, "123",  ios::dec },
    { "-1f",    2, "-1",   ios::dec },
    { "-011f",  4, "-11",  ios::dec },
    { "-123",  -1, "-123", ios::dec },
    { "+1f",    2, "1",    ios::dec },
    { "+011f",  4, "11",   ios::dec },
    { "+123",  -1, "123",  ios::dec },

    { "0",    -1, "0",   ios::oct },
    { "123",  -1, "83",  ios::oct },
    { "-123", -1, "-83", ios::oct },
    { "+123", -1, "83",  ios::oct },

    { "0",    -1, "0",    ios::hex },
    { "123",  -1, "291",  ios::hex },
    { "ff",   -1, "255",  ios::hex },
    { "FF",   -1, "255",  ios::hex },
    { "-123", -1, "-291", ios::hex },
    { "-ff",  -1, "-255", ios::hex },
    { "-FF",  -1, "-255", ios::hex },
    { "+123", -1, "291",  ios::hex },
    { "+ff",  -1, "255",  ios::hex },
    { "+FF",  -1, "255",  ios::hex },
    { "ab",   -1, "171",  ios::hex },
    { "cd",   -1, "205",  ios::hex },
    { "ef",   -1, "239",  ios::hex },

    { " 123",  0, NULL,  (ios::fmtflags) 0 },   // not without skipws
    { " 123", -1, "123", ios::skipws },
  };

  mpz_t      got, want;
  bool       got_ok, want_ok;
  bool       got_eof, want_eof;
  long       got_si, want_si;
  streampos  init_tellg, got_pos, want_pos;

  mpz_init (got);
  mpz_init (want);

  for (size_t i = 0; i < numberof (data); i++)
    {
      size_t input_length = strlen (data[i].input);
      want_pos = (data[i].want_pos == -1
                  ? input_length : data[i].want_pos);
      want_eof = (want_pos == streampos(input_length));

      want_ok = (data[i].want != NULL);

      if (data[i].want != NULL)
        mpz_set_str_or_abort (want, data[i].want, 0);
      else
        mpz_set_ui (want, 0L);

      if (option_check_standard && mpz_fits_slong_p (want))
        {
          istringstream  input (data[i].input);
          input.flags (data[i].flags);
          init_tellg = input.tellg();
          want_si = mpz_get_si (want);

          input >> got_si;
          got_ok = !input.fail();
          got_eof = input.eof();
          input.clear();
          got_pos = input.tellg() - init_tellg;

          if (got_ok != want_ok)
            {
              WRONG ("stdc++ operator>> wrong status, check_mpz");
              cout << "  want_ok: " << want_ok << "\n";
              cout << "  got_ok:  " << got_ok << "\n";
            }
          if (want_ok && got_si != want_si)
            {
              WRONG ("stdc++ operator>> wrong result, check_mpz");
              cout << "  got_si:  " << got_si << "\n";
              cout << "  want_si: " << want_si << "\n";
            }
          if (want_ok && got_eof != want_eof)
            {
              WRONG ("stdc++ operator>> wrong EOF state, check_mpz");
              cout << "  got_eof:  " << got_eof << "\n";
              cout << "  want_eof: " << want_eof << "\n";
            }
          if (putback_tellg_works && got_pos != want_pos)
            {
              WRONG ("stdc++ operator>> wrong position, check_mpz");
              cout << "  want_pos: " << want_pos << "\n";
              cout << "  got_pos:  " << got_pos << "\n";
            }
        }

      {
        istringstream  input (data[i].input);
        input.flags (data[i].flags);
        init_tellg = input.tellg();

        mpz_set_ui (got, 0xDEAD);
        input >> got;
        got_ok = !input.fail();
	got_eof = input.eof();
        input.clear();
        got_pos = input.tellg() - init_tellg;

        if (got_ok != want_ok)
          {
            WRONG ("mpz operator>> wrong status");
            cout << "  want_ok: " << want_ok << "\n";
            cout << "  got_ok:  " << got_ok << "\n";
            abort ();
          }
        if (want_ok && mpz_cmp (got, want) != 0)
          {
            WRONG ("mpz operator>> wrong result");
            mpz_trace ("  got ", got);
            mpz_trace ("  want", want);
            abort ();
          }
        if (want_ok && got_eof != want_eof)
          {
            WRONG ("mpz operator>> wrong EOF state");
            cout << "  want_eof: " << want_eof << "\n";
            cout << "  got_eof:  " << got_eof << "\n";
            abort ();
          }
        if (putback_tellg_works && got_pos != want_pos)
          {
            WRONG ("mpz operator>> wrong position");
            cout << "  want_pos: " << want_pos << "\n";
            cout << "  got_pos:  " << got_pos << "\n";
            abort ();
          }
      }
    }

  mpz_clear (got);
  mpz_clear (want);
}

void
check_mpq (void)
{
  static const struct {
    const char     *input;
    int            want_pos;
    const char     *want;
    ios::fmtflags  flags;

  } data[] = {

    { "0",   -1, "0", (ios::fmtflags) 0 },
    { "00",  -1, "0", (ios::fmtflags) 0 },
    { "0x0", -1, "0", (ios::fmtflags) 0 },

    { "123/456",   -1, "123/456", ios::dec },
    { "0123/456",  -1, "123/456", ios::dec },
    { "123/0456",  -1, "123/456", ios::dec },
    { "0123/0456", -1, "123/456", ios::dec },

    { "123/456",   -1, "83/302", ios::oct },
    { "0123/456",  -1, "83/302", ios::oct },
    { "123/0456",  -1, "83/302", ios::oct },
    { "0123/0456", -1, "83/302", ios::oct },

    { "ab",   -1, "171",  ios::hex },
    { "cd",   -1, "205",  ios::hex },
    { "ef",   -1, "239",  ios::hex },

    { "0/0",     -1, "0/0", (ios::fmtflags) 0 },
    { "5/8",     -1, "5/8", (ios::fmtflags) 0 },
    { "0x5/0x8", -1, "5/8", (ios::fmtflags) 0 },

    { "123/456",   -1, "123/456",  (ios::fmtflags) 0 },
    { "123/0456",  -1, "123/302",  (ios::fmtflags) 0 },
    { "123/0x456", -1, "123/1110", (ios::fmtflags) 0 },
    { "123/0X456", -1, "123/1110", (ios::fmtflags) 0 },

    { "0123/123",   -1, "83/123", (ios::fmtflags) 0 },
    { "0123/0123",  -1, "83/83",  (ios::fmtflags) 0 },
    { "0123/0x123", -1, "83/291", (ios::fmtflags) 0 },
    { "0123/0X123", -1, "83/291", (ios::fmtflags) 0 },

    { "0x123/123",   -1, "291/123", (ios::fmtflags) 0 },
    { "0X123/0123",  -1, "291/83",  (ios::fmtflags) 0 },
    { "0x123/0x123", -1, "291/291", (ios::fmtflags) 0 },

    { " 123",  0, NULL,  (ios::fmtflags) 0 },   // not without skipws
    { " 123", -1, "123", ios::skipws },

    { "123 /456",    3, "123",  (ios::fmtflags) 0 },
    { "123/ 456",    4,  NULL,  (ios::fmtflags) 0 },
    { "123/"    ,   -1,  NULL,  (ios::fmtflags) 0 },
    { "123 /456",    3, "123",  ios::skipws },
    { "123/ 456",    4,  NULL,  ios::skipws },
  };

  mpq_t      got, want;
  bool       got_ok, want_ok;
  bool       got_eof, want_eof;
  long       got_si, want_si;
  streampos  init_tellg, got_pos, want_pos;

  mpq_init (got);
  mpq_init (want);

  for (size_t i = 0; i < numberof (data); i++)
    {
      size_t input_length = strlen (data[i].input);
      want_pos = (data[i].want_pos == -1
                  ? input_length : data[i].want_pos);
      want_eof = (want_pos == streampos(input_length));

      want_ok = (data[i].want != NULL);

      if (data[i].want != NULL)
        mpq_set_str_or_abort (want, data[i].want, 0);
      else
        mpq_set_ui (want, 0L, 1L);

      if (option_check_standard
          && mpz_fits_slong_p (mpq_numref(want))
          && mpz_cmp_ui (mpq_denref(want), 1L) == 0
          && strchr (data[i].input, '/') == NULL)
        {
          istringstream  input (data[i].input);
          input.flags (data[i].flags);
          init_tellg = input.tellg();
          want_si = mpz_get_si (mpq_numref(want));

          input >> got_si;
          got_ok = !input.fail();
          got_eof = input.eof();
          input.clear();
          got_pos = input.tellg() - init_tellg;

          if (got_ok != want_ok)
            {
              WRONG ("stdc++ operator>> wrong status, check_mpq");
              cout << "  want_ok: " << want_ok << "\n";
              cout << "  got_ok:  " << got_ok << "\n";
            }
          if (want_ok && want_si != got_si)
            {
              WRONG ("stdc++ operator>> wrong result, check_mpq");
              cout << "  got_si:  " << got_si << "\n";
              cout << "  want_si: " << want_si << "\n";
            }
          if (want_ok && got_eof != want_eof)
            {
              WRONG ("stdc++ operator>> wrong EOF state, check_mpq");
              cout << "  got_eof:  " << got_eof << "\n";
              cout << "  want_eof: " << want_eof << "\n";
            }
          if (putback_tellg_works && got_pos != want_pos)
            {
              WRONG ("stdc++ operator>> wrong position, check_mpq");
              cout << "  want_pos: " << want_pos << "\n";
              cout << "  got_pos:  " << got_pos << "\n";
            }
        }

      {
        istringstream  input (data[i].input);
        input.flags (data[i].flags);
        init_tellg = input.tellg();
        mpq_set_si (got, 0xDEAD, 0xBEEF);

        input >> got;
        got_ok = !input.fail();
	got_eof = input.eof();
        input.clear();
        got_pos = input.tellg() - init_tellg;

        if (got_ok != want_ok)
          {
            WRONG ("mpq operator>> wrong status");
            cout << "  want_ok: " << want_ok << "\n";
            cout << "  got_ok:  " << got_ok << "\n";
            abort ();
          }
        // don't use mpq_equal, since we allow non-normalized values to be
        // read, which can trigger ASSERTs in mpq_equal
        if (want_ok && (mpz_cmp (mpq_numref (got), mpq_numref(want)) != 0
                        || mpz_cmp (mpq_denref (got), mpq_denref(want)) != 0))
          {
            WRONG ("mpq operator>> wrong result");
            mpq_trace ("  got ", got);
            mpq_trace ("  want", want);
            abort ();
          }
        if (want_ok && got_eof != want_eof)
          {
            WRONG ("mpq operator>> wrong EOF state");
            cout << "  want_eof: " << want_eof << "\n";
            cout << "  got_eof:  " << got_eof << "\n";
            abort ();
          }
        if (putback_tellg_works && got_pos != want_pos)
          {
            WRONG ("mpq operator>> wrong position");
            cout << "  want_pos: " << want_pos << "\n";
            cout << "  got_pos:  " << got_pos << "\n";
            abort ();
          }
      }
    }

  mpq_clear (got);
  mpq_clear (want);
}


void
check_mpf (void)
{
  static const struct {
    const char     *input;
    int            want_pos;
    const char     *want;
    ios::fmtflags  flags;

  } data[] = {

    { "0",      -1, "0", (ios::fmtflags) 0 },
    { "+0",     -1, "0", (ios::fmtflags) 0 },
    { "-0",     -1, "0", (ios::fmtflags) 0 },
    { "0.0",    -1, "0", (ios::fmtflags) 0 },
    { "0.",     -1, "0", (ios::fmtflags) 0 },
    { ".0",     -1, "0", (ios::fmtflags) 0 },
    { "+.0",    -1, "0", (ios::fmtflags) 0 },
    { "-.0",    -1, "0", (ios::fmtflags) 0 },
    { "+0.00",  -1, "0", (ios::fmtflags) 0 },
    { "-0.000", -1, "0", (ios::fmtflags) 0 },
    { "+0.00",  -1, "0", (ios::fmtflags) 0 },
    { "-0.000", -1, "0", (ios::fmtflags) 0 },
    { "0.0e0",  -1, "0", (ios::fmtflags) 0 },
    { "0.e0",   -1, "0", (ios::fmtflags) 0 },
    { ".0e0",   -1, "0", (ios::fmtflags) 0 },
    { "0.0e-0", -1, "0", (ios::fmtflags) 0 },
    { "0.e-0",  -1, "0", (ios::fmtflags) 0 },
    { ".0e-0",  -1, "0", (ios::fmtflags) 0 },
    { "0.0e+0", -1, "0", (ios::fmtflags) 0 },
    { "0.e+0",  -1, "0", (ios::fmtflags) 0 },
    { ".0e+0",  -1, "0", (ios::fmtflags) 0 },

    { "1",  -1,  "1", (ios::fmtflags) 0 },
    { "+1", -1,  "1", (ios::fmtflags) 0 },
    { "-1", -1, "-1", (ios::fmtflags) 0 },

    { " 0",  0,  NULL, (ios::fmtflags) 0 },  // not without skipws
    { " 0",  -1, "0", ios::skipws },
    { " +0", -1, "0", ios::skipws },
    { " -0", -1, "0", ios::skipws },

    { "+-123", 1, NULL, (ios::fmtflags) 0 },
    { "-+123", 1, NULL, (ios::fmtflags) 0 },
    { "1e+-123", 3, NULL, (ios::fmtflags) 0 },
    { "1e-+123", 3, NULL, (ios::fmtflags) 0 },

    { "e123",   0, NULL, (ios::fmtflags) 0 }, // at least one mantissa digit
    { ".e123",  1, NULL, (ios::fmtflags) 0 },
    { "+.e123", 2, NULL, (ios::fmtflags) 0 },
    { "-.e123", 2, NULL, (ios::fmtflags) 0 },

    { "123e",   4, NULL, (ios::fmtflags) 0 }, // at least one exponent digit
    { "123e-",  5, NULL, (ios::fmtflags) 0 },
    { "123e+",  5, NULL, (ios::fmtflags) 0 },
  };

  mpf_t      got, want;
  bool       got_ok, want_ok;
  bool       got_eof, want_eof;
  double     got_d, want_d;
  streampos  init_tellg, got_pos, want_pos;

  mpf_init (got);
  mpf_init (want);

  for (size_t i = 0; i < numberof (data); i++)
    {
      size_t input_length = strlen (data[i].input);
      want_pos = (data[i].want_pos == -1
                  ? input_length : data[i].want_pos);
      want_eof = (want_pos == streampos(input_length));

      want_ok = (data[i].want != NULL);

      if (data[i].want != NULL)
        mpf_set_str_or_abort (want, data[i].want, 0);
      else
        mpf_set_ui (want, 0L);

      want_d = mpf_get_d (want);
      if (option_check_standard && mpf_cmp_d (want, want_d) == 0)
        {
          istringstream  input (data[i].input);
          input.flags (data[i].flags);
          init_tellg = input.tellg();

          input >> got_d;
          got_ok = !input.fail();
          got_eof = input.eof();
          input.clear();
          got_pos = input.tellg() - init_tellg;

          if (got_ok != want_ok)
            {
              WRONG ("stdc++ operator>> wrong status, check_mpf");
              cout << "  want_ok: " << want_ok << "\n";
              cout << "  got_ok:  " << got_ok << "\n";
            }
          if (want_ok && want_d != got_d)
            {
              WRONG ("stdc++ operator>> wrong result, check_mpf");
              cout << "  got:   " << got_d << "\n";
              cout << "  want:  " << want_d << "\n";
            }
          if (want_ok && got_eof != want_eof)
            {
              WRONG ("stdc++ operator>> wrong EOF state, check_mpf");
              cout << "  got_eof:  " << got_eof << "\n";
              cout << "  want_eof: " << want_eof << "\n";
            }
          if (putback_tellg_works && got_pos != want_pos)
            {
              WRONG ("stdc++ operator>> wrong position, check_mpf");
              cout << "  want_pos: " << want_pos << "\n";
              cout << "  got_pos:  " << got_pos << "\n";
            }
        }

      {
        istringstream  input (data[i].input);
        input.flags (data[i].flags);
        init_tellg = input.tellg();

        mpf_set_ui (got, 0xDEAD);
        input >> got;
        got_ok = !input.fail();
	got_eof = input.eof();
        input.clear();
        got_pos = input.tellg() - init_tellg;

        if (got_ok != want_ok)
          {
            WRONG ("mpf operator>> wrong status");
            cout << "  want_ok: " << want_ok << "\n";
            cout << "  got_ok:  " << got_ok << "\n";
            abort ();
          }
        if (want_ok && mpf_cmp (got, want) != 0)
          {
            WRONG ("mpf operator>> wrong result");
            mpf_trace ("  got ", got);
            mpf_trace ("  want", want);
            abort ();
          }
        if (want_ok && got_eof != want_eof)
          {
            WRONG ("mpf operator>> wrong EOF state");
            cout << "  want_eof: " << want_eof << "\n";
            cout << "  got_eof:  " << got_eof << "\n";
            abort ();
          }
        if (putback_tellg_works && got_pos != want_pos)
          {
            WRONG ("mpf operator>> wrong position");
            cout << "  want_pos: " << want_pos << "\n";
            cout << "  got_pos:  " << got_pos << "\n";
            abort ();
          }
      }
    }

  mpf_clear (got);
  mpf_clear (want);
}



int
main (int argc, char *argv[])
{
  if (argc > 1 && strcmp (argv[1], "-s") == 0)
    option_check_standard = true;

  tests_start ();

  check_putback_tellg ();
  check_mpz ();
  check_mpq ();
  check_mpf ();

  tests_end ();
  return 0;
}
