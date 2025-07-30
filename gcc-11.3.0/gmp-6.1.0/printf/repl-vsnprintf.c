/* __gmp_replacement_vsnprintf -- for systems which don't have vsnprintf, or
   only have a broken one.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2001, 2002 Free Software Foundation, Inc.

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

#include "config.h"

#if ! HAVE_VSNPRINTF   /* only need this file if we don't have vsnprintf */


#define _GNU_SOURCE    /* for strnlen prototype */

#include <stdarg.h>
#include <ctype.h>     /* for isdigit */
#include <stddef.h>    /* for ptrdiff_t */
#include <string.h>
#include <stdio.h>     /* for NULL */
#include <stdlib.h>

#if HAVE_FLOAT_H
#include <float.h>     /* for DBL_MAX_10_EXP etc */
#endif

#if HAVE_INTTYPES_H
# include <inttypes.h> /* for intmax_t */
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

#if HAVE_SYS_TYPES_H
#include <sys/types.h> /* for quad_t */
#endif

#include "gmp.h"
#include "gmp-impl.h"


/* Autoconf notes that AIX 4.3 has a broken strnlen, but fortunately it
   doesn't affect us since __gmp_replacement_vsnprintf is not required on
   that system.  */
#if ! HAVE_STRNLEN
static size_t
strnlen (const char *s, size_t n)
{
  size_t  i;
  for (i = 0; i < n; i++)
    if (s[i] == '\0')
      break;
  return i;
}
#endif


/* The approach here is to parse the fmt string, and decide how much space
   it requires, then use vsprintf into a big enough buffer.  The space
   calculated isn't an exact amount, but it's certainly no less than
   required.

   This code was inspired by GNU libiberty/vasprintf.c but we support more
   datatypes, when available.

   mingw32 - doesn't have vsnprintf, it seems.  Because gcc is used a full
       set of types are available, but "long double" is just a plain IEEE
       64-bit "double" and LDBL_MAX_EXP_10 is correspondingly defined, so we
       avoid the big 15-bit exponent estimate.  */

int
__gmp_replacement_vsnprintf (char *buf, size_t buf_size,
			     const char *orig_fmt, va_list orig_ap)
{
  va_list     ap;
  const char  *fmt;
  size_t      total_width, integer_sizeof, floating_sizeof, len;
  char        fchar, type;
  int         width, prec, seen_prec, double_digits, long_double_digits;
  int         *value;

  /* preserve orig_ap for use after size estimation */
  va_copy (ap, orig_ap);

  fmt = orig_fmt;
  total_width = strlen (fmt) + 1;   /* 1 extra for the '\0' */

  integer_sizeof = sizeof (long);
#if HAVE_LONG_LONG
  integer_sizeof = MAX (integer_sizeof, sizeof (long long));
#endif
#if HAVE_QUAD_T
  integer_sizeof = MAX (integer_sizeof, sizeof (quad_t));
#endif

  floating_sizeof = sizeof (double);
#if HAVE_LONG_DOUBLE
  floating_sizeof = MAX (floating_sizeof, sizeof (long double));
#endif

  /* IEEE double or VAX G floats have an 11 bit exponent, so the default is
     a maximum 308 decimal digits.  VAX D floats have only an 8 bit
     exponent, but we don't bother trying to detect that directly.  */
  double_digits = 308;
#ifdef DBL_MAX_10_EXP
  /* but in any case prefer a value the compiler says */
  double_digits = DBL_MAX_10_EXP;
#endif

  /* IEEE 128-bit quad, Intel 80-bit temporary, or VAX H floats all have 15
     bit exponents, so the default is a maximum 4932 decimal digits.  */
  long_double_digits = 4932;
  /* but if double == long double, then go with that size */
#if HAVE_LONG_DOUBLE
  if (sizeof (double) == sizeof (long double))
    long_double_digits = double_digits;
#endif
#ifdef LDBL_MAX_10_EXP
  /* but in any case prefer a value the compiler says */
  long_double_digits = LDBL_MAX_10_EXP;
#endif

  for (;;)
    {
      fmt = strchr (fmt, '%');
      if (fmt == NULL)
	break;
      fmt++;

      type = '\0';
      width = 0;
      prec = 6;
      seen_prec = 0;
      value = &width;

      for (;;)
	{
	  fchar = *fmt++;
	  switch (fchar) {

	  case 'c':
	    /* char, already accounted for by strlen(fmt) */
	    goto next;

	  case 'd':
	  case 'i':
	  case 'o':
	  case 'x':
	  case 'X':
	  case 'u':
	    /* at most 3 digits per byte in hex, dec or octal, plus a sign */
	    total_width += 3 * integer_sizeof + 1;

	    switch (type) {
	    case 'j':
	      /* Let's assume uintmax_t is the same size as intmax_t. */
#if HAVE_INTMAX_T
	      (void) va_arg (ap, intmax_t);
#else
	      ASSERT_FAIL (intmax_t not available);
#endif
	      break;
	    case 'l':
	      (void) va_arg (ap, long);
	      break;
	    case 'L':
#if HAVE_LONG_LONG
	      (void) va_arg (ap, long long);
#else
	      ASSERT_FAIL (long long not available);
#endif
	      break;
	    case 'q':
	      /* quad_t is probably the same as long long, but let's treat
		 it separately just to be sure.  Also let's assume u_quad_t
		 will be the same size as quad_t.  */
#if HAVE_QUAD_T
	      (void) va_arg (ap, quad_t);
#else
	      ASSERT_FAIL (quad_t not available);
#endif
	      break;
	    case 't':
#if HAVE_PTRDIFF_T
	      (void) va_arg (ap, ptrdiff_t);
#else
	      ASSERT_FAIL (ptrdiff_t not available);
#endif
	      break;
	    case 'z':
	      (void) va_arg (ap, size_t);
	      break;
	    default:
	      /* default is an "int", and this includes h=short and hh=char
		 since they're promoted to int in a function call */
	      (void) va_arg (ap, int);
	      break;
	    }
	    goto next;

	  case 'E':
	  case 'e':
	  case 'G':
	  case 'g':
	    /* Requested decimals, sign, point and e, plus an overestimate
	       of exponent digits (the assumption is all the float is
	       exponent!).  */
	    total_width += prec + 3 + floating_sizeof * 3;
	    if (type == 'L')
	      {
#if HAVE_LONG_DOUBLE
		(void) va_arg (ap, long double);
#else
		ASSERT_FAIL (long double not available);
#endif
	      }
	    else
	      (void) va_arg (ap, double);
	    break;

	  case 'f':
	    /* Requested decimals, sign and point, and a margin for error,
	       then add the maximum digits that can be in the integer part,
	       based on the maximum exponent value. */
	    total_width += prec + 2 + 10;
	    if (type == 'L')
	      {
#if HAVE_LONG_DOUBLE
		(void) va_arg (ap, long double);
		total_width += long_double_digits;
#else
		ASSERT_FAIL (long double not available);
#endif
	      }
	    else
	      {
		(void) va_arg (ap, double);
		total_width += double_digits;
	      }
	    break;

	  case 'h':  /* short or char */
	  case 'j':  /* intmax_t */
	  case 'L':  /* long long or long double */
	  case 'q':  /* quad_t */
	  case 't':  /* ptrdiff_t */
	  case 'z':  /* size_t */
	  set_type:
	    type = fchar;
	    break;

	  case 'l':
	    /* long or long long */
	    if (type != 'l')
	      goto set_type;
	    type = 'L';   /* "ll" means "L" */
	    break;

	  case 'n':
	    /* bytes written, no output as such */
	    (void) va_arg (ap, void *);
	    goto next;

	  case 's':
	    /* If no precision was given, then determine the string length
	       and put it there, to be added to the total under "next".  If
	       a precision was given then that's already the maximum from
	       this field, but see whether the string is shorter than that,
	       in case the limit was very big.  */
	    {
	      const char  *s = va_arg (ap, const char *);
	      prec = (seen_prec ? strnlen (s, prec) : strlen (s));
	    }
	    goto next;

	  case 'p':
	    /* pointer, let's assume at worst it's octal with some padding */
	    (void) va_arg (ap, const void *);
	    total_width += 3 * sizeof (void *) + 16;
	    goto next;

	  case '%':
	    /* literal %, already accounted for by strlen(fmt) */
	    goto next;

	  case '#':
	    /* showbase, at most 2 for "0x" */
	    total_width += 2;
	    break;

	  case '+':
	  case ' ':
	    /* sign, already accounted for under numerics */
	    break;

	  case '-':
	    /* left justify, no effect on total width */
	    break;

	  case '.':
	    seen_prec = 1;
	    value = &prec;
	    break;

	  case '*':
	    {
	      /* negative width means left justify which can be ignored,
		 negative prec would be invalid, just use absolute value */
	      int n = va_arg (ap, int);
	      *value = ABS (n);
	    }
	    break;

	  case '0': case '1': case '2': case '3': case '4':
	  case '5': case '6': case '7': case '8': case '9':
	    /* process all digits to form a value */
	    {
	      int  n = 0;
	      do {
		n = n * 10 + (fchar-'0');
		fchar = *fmt++;
	      } while (isascii (fchar) && isdigit (fchar));
	      fmt--; /* unget the non-digit */
	      *value = n;
	    }
	    break;

	  default:
	    /* incomplete or invalid % sequence */
	    ASSERT (0);
	    goto next;
	  }
	}

    next:
      total_width += width;
      total_width += prec;
    }

  if (total_width <= buf_size)
    {
      vsprintf (buf, orig_fmt, orig_ap);
      len = strlen (buf);
    }
  else
    {
      char  *s;

      s = __GMP_ALLOCATE_FUNC_TYPE (total_width, char);
      vsprintf (s, orig_fmt, orig_ap);
      len = strlen (s);
      if (buf_size != 0)
	{
	  size_t  copylen = MIN (len, buf_size-1);
	  memcpy (buf, s, copylen);
	  buf[copylen] = '\0';
	}
      (*__gmp_free_func) (s, total_width);
    }

  /* If total_width was somehow wrong then chances are we've already
     clobbered memory, but maybe this check will still work.  */
  ASSERT_ALWAYS (len < total_width);

  return len;
}

#endif /* ! HAVE_VSNPRINTF */
