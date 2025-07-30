/* Test ostream formatted output.

Copyright 2001, 2002 Free Software Foundation, Inc.

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

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"

using namespace std;


bool option_check_standard = false;


#define CALL(expr)							\
  do {									\
    got.flags (data[i].flags);						\
    got.width (data[i].width);						\
    got.precision (data[i].precision);					\
    if (data[i].fill == '\0')						\
      got.fill (' ');							\
    else								\
      got.fill (data[i].fill);						\
									\
    if (! (expr))							\
      {									\
	cout << "\"got\" output error\n";				\
	abort ();							\
      }									\
    if (got.width() != 0)						\
      {									\
	cout << "\"got\" width not reset to 0\n";			\
	abort ();							\
      }									\
									\
  } while (0)


#define DUMP()								\
  do {									\
    cout << "  want:  |" << data[i].want << "|\n";			\
    cout << "  got:   |" << got.str() << "|\n";				\
    cout << "  width: " << data[i].width << "\n";			\
    cout << "  prec:  " << got.precision() << "\n";			\
    cout << "  flags: " << hex << (unsigned long) got.flags() << "\n";	\
  } while (0)

#define ABORT() \
  do {          \
    DUMP ();    \
    abort ();   \
  } while (0)

void
check_mpz (void)
{
  static const struct {
    const char     *z;
    const char     *want;
    ios::fmtflags  flags;
    int            width;
    int            precision;
    char           fill;

  } data[] = {

    { "0", "0", ios::dec },

    { "0", "0", ios::oct },
    { "0", "0", ios::oct | ios::showbase },

    { "0", "0", ios::hex },
    { "0", "0x0", ios::hex | ios::showbase },
    { "0", "0X0", ios::hex | ios::showbase | ios::uppercase },

    { "1", "****1", ios::dec, 5, 0, '*' },

    { "-1", "   -1",  ios::dec | ios::right,    5 },
    { "-1", "-   1",  ios::dec | ios::internal, 5 },
    { "-1", "-1   ",  ios::dec | ios::left,     5 },

    { "1", "   0x1", ios::hex | ios::showbase | ios::right,    6 },
    { "1", "0x   1", ios::hex | ios::showbase | ios::internal, 6 },
    { "1", "0x1   ", ios::hex | ios::showbase | ios::left,     6 },

    { "1", "   +0x1", ios::hex | ios::showbase | ios::showpos | ios::right,
      7 },
    { "1", "+0x   1", ios::hex | ios::showbase | ios::showpos | ios::internal,
      7 },
    { "1", "+0x1   ", ios::hex | ios::showbase | ios::showpos | ios::left,
      7 },

    {  "123",    "7b", ios::hex },
    {  "123",    "7B", ios::hex | ios::uppercase },
    {  "123",  "0x7b", ios::hex | ios::showbase },
    {  "123",  "0X7B", ios::hex | ios::showbase | ios::uppercase },
    { "-123", "-0x7b", ios::hex | ios::showbase },
    { "-123", "-0X7B", ios::hex | ios::showbase | ios::uppercase },

    {  "123",   "173", ios::oct },
    {  "123",   "173", ios::oct | ios::uppercase },
    {  "123",  "0173", ios::oct | ios::showbase },
    {  "123",  "0173", ios::oct | ios::showbase | ios::uppercase },
    { "-123", "-0173", ios::oct | ios::showbase },
    { "-123", "-0173", ios::oct | ios::showbase | ios::uppercase },

  };

  size_t  i;
  mpz_t   z;

  mpz_init (z);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (z, data[i].z, 0);

      if (option_check_standard
	  && mpz_fits_slong_p (z)

	  // no negatives or showpos in hex or oct
	  && (((data[i].flags & ios::basefield) == ios::hex
	       || (data[i].flags & ios::basefield) == ios::oct)
	      ? (mpz_sgn (z) >= 0
		 && ! (data[i].flags & ios::showpos))
	      : 1)
	  )
	{
	  ostringstream  got;
	  long  n = mpz_get_si (z);
	  CALL (got << n);
	  if (got.str().compare (data[i].want) != 0)
	    {
	      cout << "check_mpz data[" << i
		   << "] doesn't match standard ostream output\n";
	      cout << "  z:     " << data[i].z << "\n";
	      cout << "  n:     " << n << "\n";
	      DUMP ();
	    }
	}

      {
	ostringstream  got;
	CALL (got << z);
	if (got.str().compare (data[i].want) != 0)
	  {
	    cout << "mpz operator<< wrong, data[" << i << "]\n";
	    cout << "  z:     " << data[i].z << "\n";
	    ABORT ();
	  }
      }
    }

  mpz_clear (z);
}

void
check_mpq (void)
{
  static const struct {
    const char     *q;
    const char     *want;
    ios::fmtflags  flags;
    int            width;
    int            precision;
    char           fill;

  } data[] = {

    { "0", "0", ios::dec },
    { "0", "0", ios::hex },
    { "0", "0x0", ios::hex | ios::showbase },
    { "0", "0X0", ios::hex | ios::showbase | ios::uppercase },

    { "5/8", "5/8", ios::dec },
    { "5/8", "0X5/0X8", ios::hex | ios::showbase | ios::uppercase },

    // zero denominator with showbase
    { "0/0",   "       0/0", ios::oct | ios::showbase, 10 },
    { "0/0",   "       0/0", ios::dec | ios::showbase, 10 },
    { "0/0",   "   0x0/0x0", ios::hex | ios::showbase, 10 },
    { "123/0", "    0173/0", ios::oct | ios::showbase, 10 },
    { "123/0", "     123/0", ios::dec | ios::showbase, 10 },
    { "123/0", "  0x7b/0x0", ios::hex | ios::showbase, 10 },
    { "123/0", "  0X7B/0X0", ios::hex | ios::showbase | ios::uppercase, 10 },
    { "0/123", "    0/0173", ios::oct | ios::showbase, 10 },
    { "0/123", "     0/123", ios::dec | ios::showbase, 10 },
    { "0/123", "  0x0/0x7b", ios::hex | ios::showbase, 10 },
    { "0/123", "  0X0/0X7B", ios::hex | ios::showbase | ios::uppercase, 10 },
  };

  size_t  i;
  mpq_t   q;

  mpq_init (q);

#define mpq_integer_p(q)  (mpz_cmp_ui (mpq_denref(q), 1L) == 0)

  for (i = 0; i < numberof (data); i++)
    {
      mpq_set_str_or_abort (q, data[i].q, 0);
      MPZ_CHECK_FORMAT (mpq_numref (q));
      MPZ_CHECK_FORMAT (mpq_denref (q));

      if (option_check_standard
	  && mpz_fits_slong_p (mpq_numref(q))
	  && mpq_integer_p (q))
	{
	  ostringstream  got;
	  long  n = mpz_get_si (mpq_numref(q));
	  CALL (got << n);
	  if (got.str().compare (data[i].want) != 0)
	    {
	      cout << "check_mpq data[" << i
		   << "] doesn't match standard ostream output\n";
	      cout << "  q:     " << data[i].q << "\n";
	      cout << "  n:     " << n << "\n";
	      DUMP ();
	    }
	}

      {
	ostringstream  got;
	CALL (got << q);
	if (got.str().compare (data[i].want) != 0)
	  {
	    cout << "mpq operator<< wrong, data[" << i << "]\n";
	    cout << "  q:     " << data[i].q << "\n";
	    ABORT ();
	  }
      }
    }

  mpq_clear (q);
}


void
check_mpf (void)
{
  static const struct {
    const char     *f;
    const char     *want;
    ios::fmtflags  flags;
    int            width;
    int            precision;
    char           fill;

  } data[] = {

    { "0", "0",            ios::dec },
    { "0", "+0",           ios::dec | ios::showpos },
    { "0", "0.00000",      ios::dec | ios::showpoint },
    { "0", "0",            ios::dec | ios::fixed },
    { "0", "0.",           ios::dec | ios::fixed | ios::showpoint },
    { "0", "0.000000e+00", ios::dec | ios::scientific },
    { "0", "0.000000e+00", ios::dec | ios::scientific | ios::showpoint },

    { "0", "0",          ios::dec, 0, 4 },
    { "0", "0.000",      ios::dec | ios::showpoint, 0, 4 },
    { "0", "0.0000",     ios::dec | ios::fixed, 0, 4 },
    { "0", "0.0000",     ios::dec | ios::fixed | ios::showpoint, 0, 4 },
    { "0", "0.0000e+00", ios::dec | ios::scientific, 0, 4 },
    { "0", "0.0000e+00", ios::dec | ios::scientific | ios::showpoint, 0, 4 },

    { "1", "1",       ios::dec },
    { "1", "+1",      ios::dec | ios::showpos },
    { "1", "1.00000", ios::dec | ios::showpoint },
    { "1", "1",       ios::dec | ios::fixed },
    { "1", "1.",      ios::dec | ios::fixed | ios::showpoint },
    { "1", "1.000000e+00",   ios::dec | ios::scientific },
    { "1", "1.000000e+00",  ios::dec | ios::scientific | ios::showpoint },

    { "1", "1",          ios::dec,                   0, 4 },
    { "1", "1.000",      ios::dec | ios::showpoint,  0, 4 },
    { "1", "1.0000",     ios::dec | ios::fixed,      0, 4 },
    { "1", "1.0000",     ios::dec | ios::fixed | ios::showpoint, 0, 4 },
    { "1", "1.0000e+00", ios::dec | ios::scientific, 0, 4 },
    { "1", "1.0000e+00", ios::dec | ios::scientific | ios::showpoint, 0, 4 },

    { "-1", "-1",        ios::dec | ios::showpos },

    { "-1", "  -1",      ios::dec, 4 },
    { "-1", "-  1",      ios::dec | ios::internal, 4 },
    { "-1", "-1  ",      ios::dec | ios::left, 4 },

    { "-1", "  -0x1",    ios::hex | ios::showbase, 6 },
    { "-1", "-0x  1",    ios::hex | ios::showbase | ios::internal, 6 },
    { "-1", "-0x1  ",    ios::hex | ios::showbase | ios::left, 6 },

    {    "1", "*********1", ios::dec, 10, 4, '*' },
    { "1234", "******1234", ios::dec, 10, 4, '*' },
    { "1234", "*****1234.", ios::dec | ios::showpoint, 10, 4, '*' },

    { "12345", "1.23e+04", ios::dec, 0, 3 },

    { "12345", "12345.", ios::dec | ios::fixed | ios::showpoint },

    { "1.9999999",    "2",     ios::dec, 0, 1 },
    { "1.0009999999", "1.001", ios::dec, 0, 4 },
    { "1.0001",       "1",     ios::dec, 0, 4 },
    { "1.0004",       "1",     ios::dec, 0, 4 },
    { "1.000555",     "1.001", ios::dec, 0, 4 },

    { "1.0002",       "1.000", ios::dec | ios::fixed, 0, 3 },
    { "1.0008",       "1.001", ios::dec | ios::fixed, 0, 3 },

    { "0", "0", ios::hex },
    { "0", "0x0", ios::hex | ios::showbase },
    { "0", "0X0", ios::hex | ios::showbase | ios::uppercase },
    { "123",   "7b", ios::hex },
    { "123", "0x7b", ios::hex | ios::showbase },
    { "123", "0X7B", ios::hex | ios::showbase | ios::uppercase },

    { "0", "0.000@+00", ios::hex | ios::scientific, 0, 3 },
    { "256", "1.000@+02", ios::hex | ios::scientific, 0, 3 },

    { "123",   "7.b@+01", ios::hex | ios::scientific, 0, 1 },
    { "123",   "7.B@+01", ios::hex | ios::scientific | ios::uppercase, 0, 1 },
    { "123", "0x7.b@+01", ios::hex | ios::scientific | ios::showbase, 0, 1 },
    { "123", "0X7.B@+01",
      ios::hex | ios::scientific | ios::showbase | ios::uppercase, 0, 1 },

    { "1099511627776", "1.0@+10", ios::hex | ios::scientific, 0, 1 },
    { "1099511627776", "1.0@+10",
      ios::hex | ios::scientific | ios::uppercase, 0, 1 },

    { "0.0625", "1.00@-01", ios::hex | ios::scientific, 0, 2 },

    { "0", "0", ios::oct },
    { "123",  "173", ios::oct },
    { "123", "0173", ios::oct | ios::showbase },

    // octal showbase suppressed for 0
    { "0", "0", ios::oct | ios::showbase },
    { ".125",    "00.1",  ios::oct | ios::showbase, 0, 1 },
    { ".015625", "00.01", ios::oct | ios::showbase, 0, 2 },
    { ".125",    "00.1",  ios::fixed | ios::oct | ios::showbase, 0, 1 },
    { ".015625", "0.0",   ios::fixed | ios::oct | ios::showbase, 0, 1 },
    { ".015625", "00.01", ios::fixed | ios::oct | ios::showbase, 0, 2 },

    {  "0.125",  "1.000000e-01", ios::oct | ios::scientific },
    {  "0.125", "+1.000000e-01", ios::oct | ios::scientific | ios::showpos },
    { "-0.125", "-1.000000e-01", ios::oct | ios::scientific },
    { "-0.125", "-1.000000e-01", ios::oct | ios::scientific | ios::showpos },

    { "0", "0.000e+00", ios::oct | ios::scientific, 0, 3 },
    { "256",  "4.000e+02", ios::oct | ios::scientific, 0, 3 },
    { "256", "04.000e+02", ios::oct | ios::scientific | ios::showbase, 0, 3 },
    { "256",  "4.000E+02", ios::oct | ios::scientific | ios::uppercase, 0, 3 },
    { "256", "04.000E+02",
      ios::oct | ios::scientific | ios::showbase | ios::uppercase, 0, 3 },

    { "16777216",    "1.000000e+08", ios::oct | ios::scientific },
    { "16777216",    "1.000000E+08",
      ios::oct | ios::scientific | ios::uppercase },
    { "16777216",   "01.000000e+08",
      ios::oct | ios::scientific | ios::showbase },
    { "16777216",   "01.000000E+08",
      ios::oct | ios::scientific | ios::showbase | ios::uppercase },
    { "16777216",  "+01.000000e+08",
      ios::oct | ios::scientific | ios::showbase | ios::showpos },
    { "16777216",  "+01.000000E+08", ios::oct | ios::scientific
      | ios::showbase | ios::showpos | ios::uppercase },
    { "-16777216", "-01.000000e+08",
      ios::oct | ios::scientific | ios::showbase | ios::showpos },
    { "-16777216", "-01.000000E+08", ios::oct | ios::scientific
      | ios::showbase | ios::showpos | ios::uppercase },

  };

  size_t  i;
  mpf_t   f, f2;
  double  d;

  mpf_init (f);
  mpf_init (f2);

  for (i = 0; i < numberof (data); i++)
    {
      mpf_set_str_or_abort (f, data[i].f, 0);

      d = mpf_get_d (f);
      mpf_set_d (f2, d);
      if (option_check_standard && mpf_cmp (f, f2) == 0
	  && ! (data[i].flags & (ios::hex | ios::oct | ios::showbase)))
	{
	  ostringstream  got;
	  CALL (got << d);
	  if (got.str().compare (data[i].want) != 0)
	    {
	      cout << "check_mpf data[" << i
		   << "] doesn't match standard ostream output\n";
	      cout << "  f:     " << data[i].f << "\n";
	      cout << "  d:     " << d << "\n";
	      DUMP ();
	    }
	}

      {
	ostringstream  got;
	CALL (got << f);
	if (got.str().compare (data[i].want) != 0)
	  {
	    cout << "mpf operator<< wrong, data[" << i << "]\n";
	    cout << "  f:     " << data[i].f << "\n";
	    ABORT ();
	  }
      }
    }

  mpf_clear (f);
  mpf_clear (f2);
}



int
main (int argc, char *argv[])
{
  if (argc > 1 && strcmp (argv[1], "-s") == 0)
    option_check_standard = true;

  tests_start ();

  check_mpz ();
  check_mpq ();
  check_mpf ();

  tests_end ();
  return 0;
}
