/* mpf_out_str (stream, base, n_digits, op) -- Print N_DIGITS digits from
   the float OP to STREAM in base BASE.  Return the number of characters
   written, or 0 if an error occurred.

Copyright 1996, 1997, 2001, 2002, 2005, 2011 Free Software Foundation, Inc.

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

#define _GNU_SOURCE    /* for DECIMAL_POINT in langinfo.h */

#include "config.h"

#include <stdio.h>
#include <string.h>

#if HAVE_LANGINFO_H
#include <langinfo.h>  /* for nl_langinfo */
#endif

#if HAVE_LOCALE_H
#include <locale.h>    /* for localeconv */
#endif

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"


size_t
mpf_out_str (FILE *stream, int base, size_t n_digits, mpf_srcptr op)
{
  char *str;
  mp_exp_t exp;
  size_t written;
  TMP_DECL;

  TMP_MARK;

  if (base == 0)
    base = 10;
  if (n_digits == 0)
    MPF_SIGNIFICANT_DIGITS (n_digits, base, op->_mp_prec);

  if (stream == 0)
    stream = stdout;

  /* Consider these changes:
     * Don't allocate memory here for huge n_digits; pass NULL to mpf_get_str.
     * Make mpf_get_str allocate extra space when passed NULL, to avoid
       allocating two huge string buffers.
     * Implement more/other allocation reductions tricks.  */

  str = (char *) TMP_ALLOC (n_digits + 2); /* extra for minus sign and \0 */

  mpf_get_str (str, &exp, base, n_digits, op);
  n_digits = strlen (str);

  written = 0;

  /* Write sign */
  if (str[0] == '-')
    {
      str++;
      fputc ('-', stream);
      written = 1;
      n_digits--;
    }

  {
    const char  *point = GMP_DECIMAL_POINT;
    size_t      pointlen = strlen (point);
    putc ('0', stream);
    fwrite (point, 1, pointlen, stream);
    written += pointlen + 1;
  }

  /* Write mantissa */
  {
    size_t fwret;
    fwret = fwrite (str, 1, n_digits, stream);
    written += fwret;
  }

  /* Write exponent */
  {
    int fpret;
    fpret = fprintf (stream, (base <= 10 ? "e%ld" : "@%ld"), exp);
    written += fpret;
  }

  TMP_FREE;
  return ferror (stream) ? 0 : written;
}
