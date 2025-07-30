/* mpz_set_str(mp_dest, string, base) -- Convert the \0-terminated
   string STRING in base BASE to multiple precision integer in
   MP_DEST.  Allow white space in the string.  If BASE == 0 determine
   the base in the C standard way, i.e.  0xhh...h means base 16,
   0oo...o means base 8, otherwise assume base 10.

Copyright 1991, 1993, 1994, 1996-1998, 2000-2003, 2005, 2011-2013 Free Software
Foundation, Inc.

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

#include <string.h>
#include <ctype.h>
#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"

#define digit_value_tab __gmp_digit_value_tab

int
mpz_set_str (mpz_ptr x, const char *str, int base)
{
  size_t str_size;
  char *s, *begs;
  size_t i;
  mp_size_t xsize;
  int c;
  int negative;
  const unsigned char *digit_value;
  TMP_DECL;

  digit_value = digit_value_tab;
  if (base > 36)
    {
      /* For bases > 36, use the collating sequence
	 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.  */
      digit_value += 208;
      if (base > 62)
	return -1;		/* too large base */
    }

  /* Skip whitespace.  */
  do
    c = (unsigned char) *str++;
  while (isspace (c));

  negative = 0;
  if (c == '-')
    {
      negative = 1;
      c = (unsigned char) *str++;
    }

  if (digit_value[c] >= (base == 0 ? 10 : base))
    return -1;			/* error if no valid digits */

  /* If BASE is 0, try to find out the base by looking at the initial
     characters.  */
  if (base == 0)
    {
      base = 10;
      if (c == '0')
	{
	  base = 8;
	  c = (unsigned char) *str++;
	  if (c == 'x' || c == 'X')
	    {
	      base = 16;
	      c = (unsigned char) *str++;
	    }
	  else if (c == 'b' || c == 'B')
	    {
	      base = 2;
	      c = (unsigned char) *str++;
	    }
	}
    }

  /* Skip leading zeros and white space.  */
  while (c == '0' || isspace (c))
    c = (unsigned char) *str++;
  /* Make sure the string does not become empty, mpn_set_str would fail.  */
  if (c == 0)
    {
      SIZ (x) = 0;
      return 0;
    }

  TMP_MARK;
  str_size = strlen (str - 1);
  s = begs = (char *) TMP_ALLOC (str_size + 1);

  /* Remove spaces from the string and convert the result from ASCII to a
     byte array.  */
  for (i = 0; i < str_size; i++)
    {
      if (!isspace (c))
	{
	  int dig = digit_value[c];
	  if (dig >= base)
	    {
	      TMP_FREE;
	      return -1;
	    }
	  *s++ = dig;
	}
      c = (unsigned char) *str++;
    }

  str_size = s - begs;

  LIMBS_PER_DIGIT_IN_BASE (xsize, str_size, base);
  MPZ_REALLOC (x, xsize);

  /* Convert the byte array in base BASE to our bignum format.  */
  xsize = mpn_set_str (PTR (x), (unsigned char *) begs, str_size, base);
  SIZ (x) = negative ? -xsize : xsize;

  TMP_FREE;
  return 0;
}
