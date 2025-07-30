/* Test gmp_printf and related functions.

Copyright 2001-2003 Free Software Foundation, Inc.

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


/* Usage: t-printf [-s]

   -s  Check the data against the system printf, where possible.  This is
       only an option since we don't want to fail if the system printf is
       faulty or strange.  */


#include "config.h"	/* needed for the HAVE_, could also move gmp incls */

#include <stdarg.h>
#include <stddef.h>    /* for ptrdiff_t */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if HAVE_OBSTACK_VPRINTF
#define obstack_chunk_alloc tests_allocate
#define obstack_chunk_free  tests_free_nosize
#include <obstack.h>
#endif

#if HAVE_INTTYPES_H
# include <inttypes.h> /* for intmax_t */
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

#if HAVE_UNISTD_H
#include <unistd.h>  /* for unlink */
#endif

#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


int   option_check_printf = 0;


#define CHECK_VFPRINTF_FILENAME  "t-printf.tmp"
FILE  *check_vfprintf_fp;


/* From any of the tests run here. */
#define MAX_OUTPUT  1024


void
check_plain (const char *want, const char *fmt_orig, ...)
{
  char        got[MAX_OUTPUT];
  int         got_len, want_len;
  size_t      fmtsize;
  char        *fmt, *q;
  const char  *p;
  va_list     ap;
  va_start (ap, fmt_orig);

  if (! option_check_printf)
    return;

  fmtsize = strlen (fmt_orig) + 1;
  fmt = (char *) (*__gmp_allocate_func) (fmtsize);

  for (p = fmt_orig, q = fmt; *p != '\0'; p++)
    {
      switch (*p) {
      case 'a':
      case 'A':
	/* The exact value of the exponent isn't guaranteed in glibc, and it
	   and gmp_printf do slightly different things, so don't compare
	   directly. */
	goto done;
      case 'F':
	if (p > fmt_orig && *(p-1) == '.')
	  goto done;  /* don't test the "all digits" cases */
	/* discard 'F' type */
	break;
      case 'Z':
	/* transmute */
	*q++ = 'l';
	break;
      default:
	*q++ = *p;
	break;
      }
    }
  *q = '\0';

  want_len = strlen (want);
  ASSERT_ALWAYS (want_len < sizeof(got));

  got_len = vsprintf (got, fmt, ap);

  if (got_len != want_len || strcmp (got, want) != 0)
    {
      printf ("wanted data doesn't match plain vsprintf\n");
      printf ("  fmt      |%s|\n", fmt);
      printf ("  got      |%s|\n", got);
      printf ("  want     |%s|\n", want);
      printf ("  got_len  %d\n", got_len);
      printf ("  want_len %d\n", want_len);
      abort ();
    }

 done:
  (*__gmp_free_func) (fmt, fmtsize);
}

void
check_vsprintf (const char *want, const char *fmt, va_list ap)
{
  char  got[MAX_OUTPUT];
  int   got_len, want_len;

  want_len = strlen (want);
  got_len = gmp_vsprintf (got, fmt, ap);

  if (got_len != want_len || strcmp (got, want) != 0)
    {
      printf ("gmp_vsprintf wrong\n");
      printf ("  fmt      |%s|\n", fmt);
      printf ("  got      |%s|\n", got);
      printf ("  want     |%s|\n", want);
      printf ("  got_len  %d\n", got_len);
      printf ("  want_len %d\n", want_len);
      abort ();
    }
}

void
check_vfprintf (const char *want, const char *fmt, va_list ap)
{
  char  got[MAX_OUTPUT];
  int   got_len, want_len, fread_len;
  long  ftell_len;

  want_len = strlen (want);

  rewind (check_vfprintf_fp);
  got_len = gmp_vfprintf (check_vfprintf_fp, fmt, ap);
  ASSERT_ALWAYS (got_len != -1);
  ASSERT_ALWAYS (fflush (check_vfprintf_fp) == 0);

  ftell_len = ftell (check_vfprintf_fp);
  ASSERT_ALWAYS (ftell_len != -1);

  rewind (check_vfprintf_fp);
  ASSERT_ALWAYS (ftell_len <= sizeof(got));
  fread_len = fread (got, 1, ftell_len, check_vfprintf_fp);

  if (got_len != want_len
      || ftell_len != want_len
      || fread_len != want_len
      || memcmp (got, want, want_len) != 0)
    {
      printf ("gmp_vfprintf wrong\n");
      printf ("  fmt       |%s|\n", fmt);
      printf ("  got       |%.*s|\n", fread_len, got);
      printf ("  want      |%s|\n", want);
      printf ("  got_len   %d\n", got_len);
      printf ("  ftell_len %ld\n", ftell_len);
      printf ("  fread_len %d\n", fread_len);
      printf ("  want_len  %d\n", want_len);
      abort ();
    }
}

void
check_vsnprintf (const char *want, const char *fmt, va_list ap)
{
  char    got[MAX_OUTPUT+1];
  int     ret, got_len, want_len;
  size_t  bufsize;

  want_len = strlen (want);

  bufsize = -1;
  for (;;)
    {
      /* do 0 to 5, then want-5 to want+5 */
      bufsize++;
      if (bufsize > 5 && bufsize < want_len-5)
	bufsize = want_len-5;
      if (bufsize > want_len + 5)
	break;
      ASSERT_ALWAYS (bufsize+1 <= sizeof (got));

      got[bufsize] = '!';
      ret = gmp_vsnprintf (got, bufsize, fmt, ap);

      got_len = MIN (MAX(1,bufsize)-1, want_len);

      if (got[bufsize] != '!')
	{
	  printf ("gmp_vsnprintf overwrote bufsize sentinel\n");
	  goto error;
	}

      if (ret != want_len)
	{
	  printf ("gmp_vsnprintf return value wrong\n");
	  goto error;
	}

      if (bufsize > 0)
	{
	  if (memcmp (got, want, got_len) != 0 || got[got_len] != '\0')
	    {
	      printf ("gmp_vsnprintf wrong result string\n");
	    error:
	      printf ("  fmt       |%s|\n", fmt);
	      printf ("  bufsize   %lu\n", (unsigned long) bufsize);
	      printf ("  got       |%s|\n", got);
	      printf ("  want      |%.*s|\n", got_len, want);
	      printf ("  want full |%s|\n", want);
	      printf ("  ret       %d\n", ret);
	      printf ("  want_len  %d\n", want_len);
	      abort ();
	    }
	}
    }
}

void
check_vasprintf (const char *want, const char *fmt, va_list ap)
{
  char  *got;
  int   got_len, want_len;

  want_len = strlen (want);
  got_len = gmp_vasprintf (&got, fmt, ap);

  if (got_len != want_len || strcmp (got, want) != 0)
    {
      printf ("gmp_vasprintf wrong\n");
      printf ("  fmt      |%s|\n", fmt);
      printf ("  got      |%s|\n", got);
      printf ("  want     |%s|\n", want);
      printf ("  got_len  %d\n", got_len);
      printf ("  want_len %d\n", want_len);
      abort ();
    }
  (*__gmp_free_func) (got, strlen(got)+1);
}

void
check_obstack_vprintf (const char *want, const char *fmt, va_list ap)
{
#if HAVE_OBSTACK_VPRINTF
  struct obstack  ob;
  int   got_len, want_len, ob_len;
  char  *got;

  want_len = strlen (want);

  obstack_init (&ob);
  got_len = gmp_obstack_vprintf (&ob, fmt, ap);
  got = (char *) obstack_base (&ob);
  ob_len = obstack_object_size (&ob);

  if (got_len != want_len
      || ob_len != want_len
      || memcmp (got, want, want_len) != 0)
    {
      printf ("gmp_obstack_vprintf wrong\n");
      printf ("  fmt      |%s|\n", fmt);
      printf ("  got      |%s|\n", got);
      printf ("  want     |%s|\n", want);
      printf ("  got_len  %d\n", got_len);
      printf ("  ob_len   %d\n", ob_len);
      printf ("  want_len %d\n", want_len);
      abort ();
    }
  obstack_free (&ob, NULL);
#endif
}


void
check_one (const char *want, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);

  /* simplest first */
  check_vsprintf (want, fmt, ap);
  check_vfprintf (want, fmt, ap);
  check_vsnprintf (want, fmt, ap);
  check_vasprintf (want, fmt, ap);
  check_obstack_vprintf (want, fmt, ap);
}


#define hex_or_octal_p(fmt)             \
  (strchr (fmt, 'x') != NULL            \
   || strchr (fmt, 'X') != NULL         \
   || strchr (fmt, 'o') != NULL)

void
check_z (void)
{
  static const struct {
    const char  *fmt;
    const char  *z;
    const char  *want;
  } data[] = {
    { "%Zd", "0",    "0" },
    { "%Zd", "1",    "1" },
    { "%Zd", "123",  "123" },
    { "%Zd", "-1",   "-1" },
    { "%Zd", "-123", "-123" },

    { "%+Zd", "0",      "+0" },
    { "%+Zd", "123",  "+123" },
    { "%+Zd", "-123", "-123" },

    { "%Zx",  "123",   "7b" },
    { "%ZX",  "123",   "7B" },
    { "%Zx", "-123",  "-7b" },
    { "%ZX", "-123",  "-7B" },
    { "%Zo",  "123",  "173" },
    { "%Zo", "-123", "-173" },

    { "%#Zx",    "0",     "0" },
    { "%#ZX",    "0",     "0" },
    { "%#Zx",  "123",  "0x7b" },
    { "%#ZX",  "123",  "0X7B" },
    { "%#Zx", "-123", "-0x7b" },
    { "%#ZX", "-123", "-0X7B" },

    { "%#Zo",    "0",     "0" },
    { "%#Zo",  "123",  "0173" },
    { "%#Zo", "-123", "-0173" },

    { "%10Zd",      "0", "         0" },
    { "%10Zd",    "123", "       123" },
    { "%10Zd",   "-123", "      -123" },

    { "%-10Zd",     "0", "0         " },
    { "%-10Zd",   "123", "123       " },
    { "%-10Zd",  "-123", "-123      " },

    { "%+10Zd",   "123", "      +123" },
    { "%+-10Zd",  "123", "+123      " },
    { "%+10Zd",  "-123", "      -123" },
    { "%+-10Zd", "-123", "-123      " },

    { "%08Zd",    "0", "00000000" },
    { "%08Zd",  "123", "00000123" },
    { "%08Zd", "-123", "-0000123" },

    { "%+08Zd",    "0", "+0000000" },
    { "%+08Zd",  "123", "+0000123" },
    { "%+08Zd", "-123", "-0000123" },

    { "%#08Zx",    "0", "00000000" },
    { "%#08Zx",  "123", "0x00007b" },
    { "%#08Zx", "-123", "-0x0007b" },

    { "%+#08Zx",    "0", "+0000000" },
    { "%+#08Zx",  "123", "+0x0007b" },
    { "%+#08Zx", "-123", "-0x0007b" },

    { "%.0Zd", "0", "" },
    { "%.1Zd", "0", "0" },
    { "%.2Zd", "0", "00" },
    { "%.3Zd", "0", "000" },
  };

  int        i, j;
  mpz_t      z;
  char       *nfmt;
  mp_size_t  nsize, zeros;

  mpz_init (z);

  for (i = 0; i < numberof (data); i++)
    {
      mpz_set_str_or_abort (z, data[i].z, 0);

      /* don't try negatives or forced sign in hex or octal */
      if (mpz_fits_slong_p (z)
	  && ! (hex_or_octal_p (data[i].fmt)
		&& (strchr (data[i].fmt, '+') != NULL || mpz_sgn(z) < 0)))
	{
	  check_plain (data[i].want, data[i].fmt, mpz_get_si (z));
	}

      check_one (data[i].want, data[i].fmt, z);

      /* Same again, with %N and possibly some high zero limbs */
      nfmt = __gmp_allocate_strdup (data[i].fmt);
      for (j = 0; nfmt[j] != '\0'; j++)
	if (nfmt[j] == 'Z')
	  nfmt[j] = 'N';
      for (zeros = 0; zeros <= 3; zeros++)
	{
	  nsize = ABSIZ(z)+zeros;
	  MPZ_REALLOC (z, nsize);
	  nsize = (SIZ(z) >= 0 ? nsize : -nsize);
	  refmpn_zero (PTR(z)+ABSIZ(z), zeros);
	  check_one (data[i].want, nfmt, PTR(z), nsize);
	}
      __gmp_free_func (nfmt, strlen(nfmt)+1);
    }

  mpz_clear (z);
}

void
check_q (void)
{
  static const struct {
    const char  *fmt;
    const char  *q;
    const char  *want;
  } data[] = {
    { "%Qd",    "0",    "0" },
    { "%Qd",    "1",    "1" },
    { "%Qd",  "123",  "123" },
    { "%Qd",   "-1",   "-1" },
    { "%Qd", "-123", "-123" },
    { "%Qd",  "3/2",  "3/2" },
    { "%Qd", "-3/2", "-3/2" },

    { "%+Qd", "0",      "+0" },
    { "%+Qd", "123",  "+123" },
    { "%+Qd", "-123", "-123" },
    { "%+Qd", "5/8",  "+5/8" },
    { "%+Qd", "-5/8", "-5/8" },

    { "%Qx",  "123",   "7b" },
    { "%QX",  "123",   "7B" },
    { "%Qx",  "15/16", "f/10" },
    { "%QX",  "15/16", "F/10" },
    { "%Qx", "-123",  "-7b" },
    { "%QX", "-123",  "-7B" },
    { "%Qx", "-15/16", "-f/10" },
    { "%QX", "-15/16", "-F/10" },
    { "%Qo",  "123",  "173" },
    { "%Qo", "-123", "-173" },
    { "%Qo",  "16/17",  "20/21" },
    { "%Qo", "-16/17", "-20/21" },

    { "%#Qx",    "0",     "0" },
    { "%#QX",    "0",     "0" },
    { "%#Qx",  "123",  "0x7b" },
    { "%#QX",  "123",  "0X7B" },
    { "%#Qx",  "5/8",  "0x5/0x8" },
    { "%#QX",  "5/8",  "0X5/0X8" },
    { "%#Qx", "-123", "-0x7b" },
    { "%#QX", "-123", "-0X7B" },
    { "%#Qx", "-5/8", "-0x5/0x8" },
    { "%#QX", "-5/8", "-0X5/0X8" },
    { "%#Qo",    "0",     "0" },
    { "%#Qo",  "123",  "0173" },
    { "%#Qo", "-123", "-0173" },
    { "%#Qo",  "5/7",  "05/07" },
    { "%#Qo", "-5/7", "-05/07" },

    /* zero denominator and showbase */
    { "%#10Qo", "0/0",     "       0/0" },
    { "%#10Qd", "0/0",     "       0/0" },
    { "%#10Qx", "0/0",     "       0/0" },
    { "%#10Qo", "123/0",   "    0173/0" },
    { "%#10Qd", "123/0",   "     123/0" },
    { "%#10Qx", "123/0",   "    0x7b/0" },
    { "%#10QX", "123/0",   "    0X7B/0" },
    { "%#10Qo", "-123/0",  "   -0173/0" },
    { "%#10Qd", "-123/0",  "    -123/0" },
    { "%#10Qx", "-123/0",  "   -0x7b/0" },
    { "%#10QX", "-123/0",  "   -0X7B/0" },

    { "%10Qd",      "0", "         0" },
    { "%-10Qd",     "0", "0         " },
    { "%10Qd",    "123", "       123" },
    { "%-10Qd",   "123", "123       " },
    { "%10Qd",   "-123", "      -123" },
    { "%-10Qd",  "-123", "-123      " },

    { "%+10Qd",   "123", "      +123" },
    { "%+-10Qd",  "123", "+123      " },
    { "%+10Qd",  "-123", "      -123" },
    { "%+-10Qd", "-123", "-123      " },

    { "%08Qd",    "0", "00000000" },
    { "%08Qd",  "123", "00000123" },
    { "%08Qd", "-123", "-0000123" },

    { "%+08Qd",    "0", "+0000000" },
    { "%+08Qd",  "123", "+0000123" },
    { "%+08Qd", "-123", "-0000123" },

    { "%#08Qx",    "0", "00000000" },
    { "%#08Qx",  "123", "0x00007b" },
    { "%#08Qx", "-123", "-0x0007b" },

    { "%+#08Qx",    "0", "+0000000" },
    { "%+#08Qx",  "123", "+0x0007b" },
    { "%+#08Qx", "-123", "-0x0007b" },
  };

  int    i;
  mpq_t  q;

  mpq_init (q);

  for (i = 0; i < numberof (data); i++)
    {
      mpq_set_str_or_abort (q, data[i].q, 0);
      check_one (data[i].want, data[i].fmt, q);
    }

  mpq_clear (q);
}

void
check_f (void)
{
  static const struct {
    const char  *fmt;
    const char  *f;
    const char  *want;

  } data[] = {

    { "%Ff",    "0",    "0.000000" },
    { "%Ff",  "123",  "123.000000" },
    { "%Ff", "-123", "-123.000000" },

    { "%+Ff",    "0",   "+0.000000" },
    { "%+Ff",  "123", "+123.000000" },
    { "%+Ff", "-123", "-123.000000" },

    { "%.0Ff",    "0",    "0" },
    { "%.0Ff",  "123",  "123" },
    { "%.0Ff", "-123", "-123" },

    { "%8.0Ff",    "0", "       0" },
    { "%8.0Ff",  "123", "     123" },
    { "%8.0Ff", "-123", "    -123" },

    { "%08.0Ff",    "0", "00000000" },
    { "%08.0Ff",  "123", "00000123" },
    { "%08.0Ff", "-123", "-0000123" },

    { "%10.2Ff",       "0", "      0.00" },
    { "%10.2Ff",    "0.25", "      0.25" },
    { "%10.2Ff",  "123.25", "    123.25" },
    { "%10.2Ff", "-123.25", "   -123.25" },

    { "%-10.2Ff",       "0", "0.00      " },
    { "%-10.2Ff",    "0.25", "0.25      " },
    { "%-10.2Ff",  "123.25", "123.25    " },
    { "%-10.2Ff", "-123.25", "-123.25   " },

    { "%.2Ff", "0.00000000000001", "0.00" },
    { "%.2Ff", "0.002",            "0.00" },
    { "%.2Ff", "0.008",            "0.01" },

    { "%.0Ff", "123.00000000000001", "123" },
    { "%.0Ff", "123.2",              "123" },
    { "%.0Ff", "123.8",              "124" },

    { "%.0Ff",  "999999.9", "1000000" },
    { "%.0Ff", "3999999.9", "4000000" },

    { "%Fe",    "0",  "0.000000e+00" },
    { "%Fe",    "1",  "1.000000e+00" },
    { "%Fe",  "123",  "1.230000e+02" },

    { "%FE",    "0",  "0.000000E+00" },
    { "%FE",    "1",  "1.000000E+00" },
    { "%FE",  "123",  "1.230000E+02" },

    { "%Fe",    "0",  "0.000000e+00" },
    { "%Fe",    "1",  "1.000000e+00" },

    { "%.0Fe",     "10000000000",    "1e+10" },
    { "%.0Fe",    "-10000000000",   "-1e+10" },

    { "%.2Fe",     "10000000000",  "1.00e+10" },
    { "%.2Fe",    "-10000000000", "-1.00e+10" },

    { "%8.0Fe",    "10000000000", "   1e+10" },
    { "%8.0Fe",   "-10000000000", "  -1e+10" },

    { "%-8.0Fe",   "10000000000", "1e+10   " },
    { "%-8.0Fe",  "-10000000000", "-1e+10  " },

    { "%12.2Fe",   "10000000000", "    1.00e+10" },
    { "%12.2Fe",  "-10000000000", "   -1.00e+10" },

    { "%012.2Fe",  "10000000000", "00001.00e+10" },
    { "%012.2Fe", "-10000000000", "-0001.00e+10" },

    { "%Fg",   "0", "0" },
    { "%Fg",   "1", "1" },
    { "%Fg",   "-1", "-1" },

    { "%.0Fg", "0", "0" },
    { "%.0Fg", "1", "1" },
    { "%.0Fg", "-1", "-1" },

    { "%.1Fg", "100", "1e+02" },
    { "%.2Fg", "100", "1e+02" },
    { "%.3Fg", "100", "100" },
    { "%.4Fg", "100", "100" },

    { "%Fg", "0.001",    "0.001" },
    { "%Fg", "0.0001",   "0.0001" },
    { "%Fg", "0.00001",  "1e-05" },
    { "%Fg", "0.000001", "1e-06" },

    { "%.4Fg", "1.00000000000001", "1" },
    { "%.4Fg", "100000000000001",  "1e+14" },

    { "%.4Fg", "12345678", "1.235e+07" },

    { "%Fa", "0","0x0p+0" },
    { "%FA", "0","0X0P+0" },

    { "%Fa", "1","0x1p+0" },
    { "%Fa", "65535","0xf.fffp+12" },
    { "%Fa", "65536","0x1p+16" },
    { "%F.10a", "65536","0x1.0000000000p+16" },
    { "%F.1a", "65535","0x1.0p+16" },
    { "%F.0a", "65535","0x1p+16" },

    { "%.2Ff", "0.99609375", "1.00" },
    { "%.Ff",  "0.99609375", "0.99609375" },
    { "%.Fe",  "0.99609375", "9.9609375e-01" },
    { "%.Fg",  "0.99609375", "0.99609375" },
    { "%.20Fg",  "1000000", "1000000" },
    { "%.Fg",  "1000000", "1000000" },

    { "%#.0Ff", "1", "1." },
    { "%#.0Fe", "1", "1.e+00" },
    { "%#.0Fg", "1", "1." },

    { "%#.1Ff", "1", "1.0" },
    { "%#.1Fe", "1", "1.0e+00" },
    { "%#.1Fg", "1", "1." },

    { "%#.4Ff", "1234", "1234.0000" },
    { "%#.4Fe", "1234", "1.2340e+03" },
    { "%#.4Fg", "1234", "1234." },

    { "%#.8Ff", "1234", "1234.00000000" },
    { "%#.8Fe", "1234", "1.23400000e+03" },
    { "%#.8Fg", "1234", "1234.0000" },

  };

  int     i;
  mpf_t   f;
  double  d;

  mpf_init2 (f, 256L);

  for (i = 0; i < numberof (data); i++)
    {
      if (data[i].f[0] == '0' && data[i].f[1] == 'x')
	mpf_set_str_or_abort (f, data[i].f, 16);
      else
	mpf_set_str_or_abort (f, data[i].f, 10);

      /* if mpf->double doesn't truncate, then expect same result */
      d = mpf_get_d (f);
      if (mpf_cmp_d (f, d) == 0)
	check_plain (data[i].want, data[i].fmt, d);

      check_one (data[i].want, data[i].fmt, f);
    }

  mpf_clear (f);
}


void
check_limb (void)
{
  int        i;
  mp_limb_t  limb;
  mpz_t      z;
  char       *s;

  check_one ("0", "%Md", CNST_LIMB(0));
  check_one ("1", "%Md", CNST_LIMB(1));

  /* "i" many 1 bits, tested against mpz_get_str in decimal and hex */
  limb = 1;
  mpz_init_set_ui (z, 1L);
  for (i = 1; i <= GMP_LIMB_BITS; i++)
    {
      s = mpz_get_str (NULL, 10, z);
      check_one (s, "%Mu", limb);
      (*__gmp_free_func) (s, strlen (s) + 1);

      s = mpz_get_str (NULL, 16, z);
      check_one (s, "%Mx", limb);
      (*__gmp_free_func) (s, strlen (s) + 1);

      s = mpz_get_str (NULL, -16, z);
      check_one (s, "%MX", limb);
      (*__gmp_free_func) (s, strlen (s) + 1);

      limb = 2*limb + 1;
      mpz_mul_2exp (z, z, 1L);
      mpz_add_ui (z, z, 1L);
    }

  mpz_clear (z);
}


void
check_n (void)
{
  {
    int  n = -1;
    check_one ("blah", "%nblah", &n);
    ASSERT_ALWAYS (n == 0);
  }

  {
    int  n = -1;
    check_one ("hello ", "hello %n", &n);
    ASSERT_ALWAYS (n == 6);
  }

  {
    int  n = -1;
    check_one ("hello  world", "hello %n world", &n);
    ASSERT_ALWAYS (n == 6);
  }

#define CHECK_N(type, string)                           \
  do {                                                  \
    type  x[2];                                         \
    char  fmt[128];                                     \
							\
    x[0] = ~ (type) 0;                                  \
    x[1] = ~ (type) 0;                                  \
    sprintf (fmt, "%%d%%%sn%%d", string);               \
    check_one ("123456", fmt, 123, &x[0], 456);         \
							\
    /* should write whole of x[0] and none of x[1] */   \
    ASSERT_ALWAYS (x[0] == 3);                          \
    ASSERT_ALWAYS (x[1] == (type) ~ (type) 0);		\
							\
  } while (0)

  CHECK_N (mp_limb_t, "M");
  CHECK_N (char,      "hh");
  CHECK_N (long,      "l");
#if HAVE_LONG_LONG
  CHECK_N (long long, "L");
#endif
#if HAVE_INTMAX_T
  CHECK_N (intmax_t,  "j");
#endif
#if HAVE_PTRDIFF_T
  CHECK_N (ptrdiff_t, "t");
#endif
  CHECK_N (short,     "h");
  CHECK_N (size_t,    "z");

  {
    mpz_t  x[2];
    mpz_init_set_si (x[0], -987L);
    mpz_init_set_si (x[1],  654L);
    check_one ("123456", "%d%Zn%d", 123, x[0], 456);
    MPZ_CHECK_FORMAT (x[0]);
    MPZ_CHECK_FORMAT (x[1]);
    ASSERT_ALWAYS (mpz_cmp_ui (x[0], 3L) == 0);
    ASSERT_ALWAYS (mpz_cmp_ui (x[1], 654L) == 0);
    mpz_clear (x[0]);
    mpz_clear (x[1]);
  }

  {
    mpq_t  x[2];
    mpq_init (x[0]);
    mpq_init (x[1]);
    mpq_set_ui (x[0], 987L, 654L);
    mpq_set_ui (x[1], 4115L, 226L);
    check_one ("123456", "%d%Qn%d", 123, x[0], 456);
    MPQ_CHECK_FORMAT (x[0]);
    MPQ_CHECK_FORMAT (x[1]);
    ASSERT_ALWAYS (mpq_cmp_ui (x[0], 3L, 1L) == 0);
    ASSERT_ALWAYS (mpq_cmp_ui (x[1], 4115L, 226L) == 0);
    mpq_clear (x[0]);
    mpq_clear (x[1]);
  }

  {
    mpf_t  x[2];
    mpf_init (x[0]);
    mpf_init (x[1]);
    mpf_set_ui (x[0], 987L);
    mpf_set_ui (x[1], 654L);
    check_one ("123456", "%d%Fn%d", 123, x[0], 456);
    MPF_CHECK_FORMAT (x[0]);
    MPF_CHECK_FORMAT (x[1]);
    ASSERT_ALWAYS (mpf_cmp_ui (x[0], 3L) == 0);
    ASSERT_ALWAYS (mpf_cmp_ui (x[1], 654L) == 0);
    mpf_clear (x[0]);
    mpf_clear (x[1]);
  }

  {
    mp_limb_t  a[5];
    mp_limb_t  a_want[numberof(a)];
    mp_size_t  i;

    a[0] = 123;
    check_one ("blah", "bl%Nnah", a, (mp_size_t) 0);
    ASSERT_ALWAYS (a[0] == 123);

    MPN_ZERO (a_want, numberof (a_want));
    for (i = 1; i < numberof (a); i++)
      {
	check_one ("blah", "bl%Nnah", a, i);
	a_want[0] = 2;
	ASSERT_ALWAYS (mpn_cmp (a, a_want, i) == 0);
      }
  }
}


void
check_misc (void)
{
  mpz_t  z;
  mpf_t  f;

  mpz_init (z);
  mpf_init2 (f, 128L);

  check_one ("!", "%c", '!');

  check_one ("hello world", "hello %s", "world");
  check_one ("hello:", "%s:", "hello");
  mpz_set_ui (z, 0L);
  check_one ("hello0", "%s%Zd", "hello", z, z);

  {
    static char  xs[801];
    memset (xs, 'x', sizeof(xs)-1);
    check_one (xs, "%s", xs);
  }

  mpz_set_ui (z, 12345L);
  check_one ("     12345", "%*Zd", 10, z);
  check_one ("0000012345", "%0*Zd", 10, z);
  check_one ("12345     ", "%*Zd", -10, z);
  check_one ("12345 and 678", "%Zd and %d", z, 678);
  check_one ("12345,1,12345,2,12345", "%Zd,%d,%Zd,%d,%Zd", z, 1, z, 2, z);

  /* from the glibc info docs */
  mpz_set_si (z, 0L);
  check_one ("|    0|0    |   +0|+0   |    0|00000|     |   00|0|",
	     "|%5Zd|%-5Zd|%+5Zd|%+-5Zd|% 5Zd|%05Zd|%5.0Zd|%5.2Zd|%Zd|",
	     /**/ z,    z,    z,     z,    z,    z,     z,     z,  z);
  mpz_set_si (z, 1L);
  check_one ("|    1|1    |   +1|+1   |    1|00001|    1|   01|1|",
	     "|%5Zd|%-5Zd|%+5Zd|%+-5Zd|% 5Zd|%05Zd|%5.0Zd|%5.2Zd|%Zd|",
	     /**/ z,    z,    z,     z,    z,    z,     z,     z,  z);
  mpz_set_si (z, -1L);
  check_one ("|   -1|-1   |   -1|-1   |   -1|-0001|   -1|  -01|-1|",
	     "|%5Zd|%-5Zd|%+5Zd|%+-5Zd|% 5Zd|%05Zd|%5.0Zd|%5.2Zd|%Zd|",
	     /**/ z,    z,    z,     z,    z,    z,     z,     z,  z);
  mpz_set_si (z, 100000L);
  check_one ("|100000|100000|+100000|+100000| 100000|100000|100000|100000|100000|",
	     "|%5Zd|%-5Zd|%+5Zd|%+-5Zd|% 5Zd|%05Zd|%5.0Zd|%5.2Zd|%Zd|",
	     /**/ z,    z,    z,     z,    z,    z,     z,     z,  z);
  mpz_set_si (z, 0L);
  check_one ("|    0|    0|    0|    0|    0|    0|  00000000|",
	     "|%5Zo|%5Zx|%5ZX|%#5Zo|%#5Zx|%#5ZX|%#10.8Zx|",
	     /**/ z,   z,   z,    z,    z,    z,       z);
  mpz_set_si (z, 1L);
  check_one ("|    1|    1|    1|   01|  0x1|  0X1|0x00000001|",
	     "|%5Zo|%5Zx|%5ZX|%#5Zo|%#5Zx|%#5ZX|%#10.8Zx|",
	     /**/ z,   z,   z,    z,    z,    z,       z);
  mpz_set_si (z, 100000L);
  check_one ("|303240|186a0|186A0|0303240|0x186a0|0X186A0|0x000186a0|",
	     "|%5Zo|%5Zx|%5ZX|%#5Zo|%#5Zx|%#5ZX|%#10.8Zx|",
	     /**/ z,   z,   z,    z,    z,    z,       z);

  /* %zd for size_t won't be available on old systems, and running something
     to see if it works might be bad, so only try it on glibc, and only on a
     new enough version (glibc 2.0 doesn't have %zd) */
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 0)
  mpz_set_ui (z, 789L);
  check_one ("456 789 blah", "%zd %Zd blah", (size_t) 456, z);
#endif

  mpz_clear (z);
  mpf_clear (f);
}


int
main (int argc, char *argv[])
{
  if (argc > 1 && strcmp (argv[1], "-s") == 0)
    option_check_printf = 1;

  tests_start ();
  check_vfprintf_fp = fopen (CHECK_VFPRINTF_FILENAME, "w+");
  ASSERT_ALWAYS (check_vfprintf_fp != NULL);

  check_z ();
  check_q ();
  check_f ();
  check_limb ();
  check_n ();
  check_misc ();

  ASSERT_ALWAYS (fclose (check_vfprintf_fp) == 0);
  unlink (CHECK_VFPRINTF_FILENAME);
  tests_end ();
  exit (0);
}
