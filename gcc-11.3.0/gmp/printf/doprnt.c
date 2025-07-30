/* __gmp_doprnt -- printf style formatted output.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2001-2003 Free Software Foundation, Inc.

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

#define _GNU_SOURCE    /* for DECIMAL_POINT in glibc langinfo.h */

#include "config.h"	/* needed for the HAVE_, could also move gmp incls */

#include <stdarg.h>
#include <ctype.h>     /* for isdigit */
#include <stddef.h>    /* for ptrdiff_t */
#include <string.h>
#include <stdio.h>     /* for NULL */
#include <stdlib.h>

#if HAVE_INTTYPES_H
# include <inttypes.h> /* for intmax_t */
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

#if HAVE_LANGINFO_H
#include <langinfo.h>  /* for nl_langinfo */
#endif

#if HAVE_LOCALE_H
#include <locale.h>    /* for localeconv */
#endif

#if HAVE_SYS_TYPES_H
#include <sys/types.h> /* for quad_t */
#endif

#include "gmp.h"
#include "gmp-impl.h"


/* change this to "#define TRACE(x) x" for diagnostics */
#define TRACE(x)


/* Should be portable, but in any case this is only used under some ASSERTs. */
#define va_equal(x, y)                           \
  (memcmp (&(x), &(y), sizeof(va_list)) == 0)


/* printf is convenient because it allows various types to be printed in one
   fairly compact call, so having gmp_printf support the standard types as
   well as the gmp ones is important.  This ends up meaning all the standard
   parsing must be duplicated, to get a new routine recognising the gmp
   extras.

   With the currently favoured handling of mpz etc as Z, Q and F type
   markers, it's not possible to use glibc register_printf_function since
   that only accepts new conversion characters, not new types.  If Z was a
   conversion there'd be no way to specify hex, decimal or octal, or
   similarly with F no way to specify fixed point or scientific format.

   It seems wisest to pass conversions %f, %e and %g of float, double and
   long double over to the standard printf.  It'd be hard to be sure of
   getting the right handling for NaNs, rounding, etc.  Integer conversions
   %d etc and string conversions %s on the other hand could be easily enough
   handled within gmp_doprnt, but if floats are going to libc then it's just
   as easy to send all non-gmp types there.

   "Z" was a type marker for size_t in old glibc, but there seems no need to
   provide access to that now "z" is standard.

   In GMP 4.1.1 we documented "ll" and "L" as being equivalent, but in C99
   in fact "ll" is just for long long and "L" just for long double.
   Apparently GLIBC allows "L" for long long though.  This doesn't affect
   us as such, since both are passed through to the C library.  To be
   consistent with what we said before, the two are treated equivalently
   here, and it's left to the C library to do what it thinks with them.

   Possibilities:

   "b" might be nice for binary output, and could even be supported for the
   standard C types too if desired.

   POSIX style "%n$" parameter numbering would be possible, but would need
   to be handled completely within gmp_doprnt, since the numbering will be
   all different once the format string it cut into pieces.

   Some options for mpq formatting would be good.  Perhaps a non-zero
   precision field could give a width for the denominator and mean always
   put a "/".  A form "n+p/q" might interesting too, though perhaps that's
   better left to applications.

   Right now there's no way for an application to know whether types like
   intmax_t are supported here.  If configure is doing its job and the same
   compiler is used for gmp as for the application then there shouldn't be
   any problem, but perhaps gmp.h should have some preprocessor symbols to
   say what libgmp can do.  */



/* If a gmp format is the very first thing or there are two gmp formats with
   nothing in between then we'll reach here with this_fmt == last_fmt and we
   can do nothing in that case.

   last_ap is always replaced after a FLUSH, so it doesn't matter if va_list
   is a call-by-reference and the funs->format routine modifies it.  */

#define FLUSH()                                         \
  do {                                                  \
    if (this_fmt == last_fmt)                           \
      {                                                 \
	TRACE (printf ("nothing to flush\n"));          \
	ASSERT (va_equal (this_ap, last_ap));           \
      }                                                 \
    else                                                \
      {                                                 \
	ASSERT (*this_fmt == '%');                      \
	*this_fmt = '\0';                               \
	TRACE (printf ("flush \"%s\"\n", last_fmt));    \
	DOPRNT_FORMAT (last_fmt, last_ap);              \
      }                                                 \
  } while (0)


/* Parse up the given format string and do the appropriate output using the
   given "funs" routines.  The data parameter is passed through to those
   routines.  */

int
__gmp_doprnt (const struct doprnt_funs_t *funs, void *data,
	      const char *orig_fmt, va_list orig_ap)
{
  va_list  ap, this_ap, last_ap;
  size_t   alloc_fmt_size;
  char     *fmt, *alloc_fmt, *last_fmt, *this_fmt, *gmp_str;
  int      retval = 0;
  int      type, fchar, *value, seen_precision;
  struct doprnt_params_t param;

  TRACE (printf ("gmp_doprnt \"%s\"\n", orig_fmt));

  /* Don't modify orig_ap, if va_list is actually an array and hence call by
     reference.  It could be argued that it'd be more efficient to leave the
     caller to make a copy if it cared, but doing so here is going to be a
     very small part of the total work, and we may as well keep applications
     out of trouble.  */
  va_copy (ap, orig_ap);

  /* The format string is chopped up into pieces to be passed to
     funs->format.  Unfortunately that means it has to be copied so each
     piece can be null-terminated.  We're not going to be very fast here, so
     use __gmp_allocate_func rather than TMP_ALLOC, to avoid overflowing the
     stack if a long output string is given.  */
  alloc_fmt_size = strlen (orig_fmt) + 1;
#if _LONG_LONG_LIMB
  /* for a long long limb we change %Mx to %llx, so could need an extra 1
     char for every 3 existing */
  alloc_fmt_size += alloc_fmt_size / 3;
#endif
  alloc_fmt = __GMP_ALLOCATE_FUNC_TYPE (alloc_fmt_size, char);
  fmt = alloc_fmt;
  memcpy (fmt, orig_fmt, alloc_fmt_size);

  /* last_fmt and last_ap are just after the last output, and hence where
     the next output will begin, when that's done */
  last_fmt = fmt;
  va_copy (last_ap, ap);

  for (;;)
    {
      TRACE (printf ("next: \"%s\"\n", fmt));

      fmt = strchr (fmt, '%');
      if (fmt == NULL)
	break;

      /* this_fmt and this_ap are the current '%' sequence being considered */
      this_fmt = fmt;
      va_copy (this_ap, ap);
      fmt++; /* skip the '%' */

      TRACE (printf ("considering\n");
	     printf ("  last: \"%s\"\n", last_fmt);
	     printf ("  this: \"%s\"\n", this_fmt));

      type = '\0';
      value = &param.width;

      param.base = 10;
      param.conv = 0;
      param.expfmt = "e%c%02ld";
      param.exptimes4 = 0;
      param.fill = ' ';
      param.justify = DOPRNT_JUSTIFY_RIGHT;
      param.prec = 6;
      param.showbase = DOPRNT_SHOWBASE_NO;
      param.showpoint = 0;
      param.showtrailing = 1;
      param.sign = '\0';
      param.width = 0;
      seen_precision = 0;

      /* This loop parses a single % sequence.  "break" from the switch
	 means continue with this %, "goto next" means the conversion
	 character has been seen and a new % should be sought.  */
      for (;;)
	{
	  fchar = *fmt++;
	  if (fchar == '\0')
	    break;

	  switch (fchar) {

	  case 'a':
	    /* %a behaves like %e, but defaults to all significant digits,
	       and there's no leading zeros on the exponent (which is in
	       fact bit-based) */
	    param.base = 16;
	    param.expfmt = "p%c%ld";
	    goto conv_a;
	  case 'A':
	    param.base = -16;
	    param.expfmt = "P%c%ld";
	  conv_a:
	    param.conv = DOPRNT_CONV_SCIENTIFIC;
	    param.exptimes4 = 1;
	    if (! seen_precision)
	      param.prec = -1;  /* default to all digits */
	    param.showbase = DOPRNT_SHOWBASE_YES;
	    param.showtrailing = 1;
	    goto floating_a;

	  case 'c':
	    /* Let's assume wchar_t will be promoted to "int" in the call,
	       the same as char will be. */
	    (void) va_arg (ap, int);
	    goto next;

	  case 'd':
	  case 'i':
	  case 'u':
	  integer:
	    TRACE (printf ("integer, base=%d\n", param.base));
	    if (! seen_precision)
	      param.prec = -1;
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
	    case 'N':
	      {
		mp_ptr     xp;
		mp_size_t  xsize, abs_xsize;
		mpz_t      z;
		FLUSH ();
		xp = va_arg (ap, mp_ptr);
		PTR(z) = xp;
		xsize = (int) va_arg (ap, mp_size_t);
		abs_xsize = ABS (xsize);
		MPN_NORMALIZE (xp, abs_xsize);
		SIZ(z) = (xsize >= 0 ? abs_xsize : -abs_xsize);
		ASSERT_CODE (ALLOC(z) = abs_xsize);
		gmp_str = mpz_get_str (NULL, param.base, z);
		goto gmp_integer;
	      }
	      /* break; */
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
	    case 'Q':
	      FLUSH ();
	      gmp_str = mpq_get_str (NULL, param.base, va_arg(ap, mpq_srcptr));
	      goto gmp_integer;
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
	    case 'Z':
	      {
		int   ret;
		FLUSH ();
		gmp_str = mpz_get_str (NULL, param.base,
				       va_arg (ap, mpz_srcptr));
	      gmp_integer:
		ret = __gmp_doprnt_integer (funs, data, &param, gmp_str);
		(*__gmp_free_func) (gmp_str, strlen(gmp_str)+1);
		DOPRNT_ACCUMULATE (ret);
		va_copy (last_ap, ap);
		last_fmt = fmt;
	      }
	      break;
	    default:
	      /* default is an "int", and this includes h=short and hh=char
		 since they're promoted to int in a function call */
	      (void) va_arg (ap, int);
	      break;
	    }
	    goto next;

	  case 'E':
	    param.base = -10;
	    param.expfmt = "E%c%02ld";
	    /*FALLTHRU*/
	  case 'e':
	    param.conv = DOPRNT_CONV_SCIENTIFIC;
	  floating:
	    if (param.showbase == DOPRNT_SHOWBASE_NONZERO)
	      {
		/* # in %e, %f and %g */
		param.showpoint = 1;
		param.showtrailing = 1;
	      }
	  floating_a:
	    switch (type) {
	    case 'F':
	      FLUSH ();
	      DOPRNT_ACCUMULATE (__gmp_doprnt_mpf (funs, data, &param,
						   GMP_DECIMAL_POINT,
						   va_arg (ap, mpf_srcptr)));
	      va_copy (last_ap, ap);
	      last_fmt = fmt;
	      break;
	    case 'L':
#if HAVE_LONG_DOUBLE
	      (void) va_arg (ap, long double);
#else
	      ASSERT_FAIL (long double not available);
#endif
	      break;
	    default:
	      (void) va_arg (ap, double);
	      break;
	    }
	    goto next;

	  case 'f':
	    param.conv = DOPRNT_CONV_FIXED;
	    goto floating;

	  case 'F': /* mpf_t     */
	  case 'j': /* intmax_t  */
	  case 'L': /* long long */
	  case 'N': /* mpn       */
	  case 'q': /* quad_t    */
	  case 'Q': /* mpq_t     */
	  case 't': /* ptrdiff_t */
	  case 'z': /* size_t    */
	  case 'Z': /* mpz_t     */
	  set_type:
	    type = fchar;
	    break;

	  case 'G':
	    param.base = -10;
	    param.expfmt = "E%c%02ld";
	    /*FALLTHRU*/
	  case 'g':
	    param.conv = DOPRNT_CONV_GENERAL;
	    param.showtrailing = 0;
	    goto floating;

	  case 'h':
	    if (type != 'h')
	      goto set_type;
	    type = 'H';   /* internal code for "hh" */
	    break;

	  case 'l':
	    if (type != 'l')
	      goto set_type;
	    type = 'L';   /* "ll" means "L" */
	    break;

	  case 'm':
	    /* glibc strerror(errno), no argument */
	    goto next;

	  case 'M': /* mp_limb_t */
	    /* mung format string to l or ll and let plain printf handle it */
#if _LONG_LONG_LIMB
	    memmove (fmt+1, fmt, strlen (fmt)+1);
	    fmt[-1] = 'l';
	    fmt[0] = 'l';
	    fmt++;
	    type = 'L';
#else
	    fmt[-1] = 'l';
	    type = 'l';
#endif
	    break;

	  case 'n':
	    {
	      void  *p;
	      FLUSH ();
	      p = va_arg (ap, void *);
	      switch (type) {
	      case '\0': * (int       *) p = retval; break;
	      case 'F':  mpf_set_si ((mpf_ptr) p, (long) retval); break;
	      case 'H':  * (char      *) p = retval; break;
	      case 'h':  * (short     *) p = retval; break;
#if HAVE_INTMAX_T
	      case 'j':  * (intmax_t  *) p = retval; break;
#else
	      case 'j':  ASSERT_FAIL (intmax_t not available); break;
#endif
	      case 'l':  * (long      *) p = retval; break;
#if HAVE_QUAD_T && HAVE_LONG_LONG
	      case 'q':
		ASSERT_ALWAYS (sizeof (quad_t) == sizeof (long long));
		/*FALLTHRU*/
#else
	      case 'q':  ASSERT_FAIL (quad_t not available); break;
#endif
#if HAVE_LONG_LONG
	      case 'L':  * (long long *) p = retval; break;
#else
	      case 'L':  ASSERT_FAIL (long long not available); break;
#endif
	      case 'N':
		{
		  mp_size_t  n;
		  n = va_arg (ap, mp_size_t);
		  n = ABS (n);
		  if (n != 0)
		    {
		      * (mp_ptr) p = retval;
		      MPN_ZERO ((mp_ptr) p + 1, n - 1);
		    }
		}
		break;
	      case 'Q':  mpq_set_si ((mpq_ptr) p, (long) retval, 1L); break;
#if HAVE_PTRDIFF_T
	      case 't':  * (ptrdiff_t *) p = retval; break;
#else
	      case 't':  ASSERT_FAIL (ptrdiff_t not available); break;
#endif
	      case 'z':  * (size_t    *) p = retval; break;
	      case 'Z':  mpz_set_si ((mpz_ptr) p, (long) retval); break;
	      }
	    }
	    va_copy (last_ap, ap);
	    last_fmt = fmt;
	    goto next;

	  case 'o':
	    param.base = 8;
	    goto integer;

	  case 'p':
	  case 's':
	    /* "void *" will be good enough for "char *" or "wchar_t *", no
	       need for separate code.  */
	    (void) va_arg (ap, const void *);
	    goto next;

	  case 'x':
	    param.base = 16;
	    goto integer;
	  case 'X':
	    param.base = -16;
	    goto integer;

	  case '%':
	    goto next;

	  case '#':
	    param.showbase = DOPRNT_SHOWBASE_NONZERO;
	    break;

	  case '\'':
	    /* glibc digit grouping, just pass it through, no support for it
	       on gmp types */
	    break;

	  case '+':
	  case ' ':
	    param.sign = fchar;
	    break;

	  case '-':
	    param.justify = DOPRNT_JUSTIFY_LEFT;
	    break;
	  case '.':
	    seen_precision = 1;
	    param.prec = -1; /* "." alone means all necessary digits */
	    value = &param.prec;
	    break;

	  case '*':
	    {
	      int n = va_arg (ap, int);

	      if (value == &param.width)
		{
		  /* negative width means left justify */
		  if (n < 0)
		    {
		      param.justify = DOPRNT_JUSTIFY_LEFT;
		      n = -n;
		    }
		  param.width = n;
		}
	      else
		{
		  /* don't allow negative precision */
		  param.prec = MAX (0, n);
		}
	    }
	    break;

	  case '0':
	    if (value == &param.width)
	      {
		/* in width field, set fill */
		param.fill = '0';

		/* for right justify, put the fill after any minus sign */
		if (param.justify == DOPRNT_JUSTIFY_RIGHT)
		  param.justify = DOPRNT_JUSTIFY_INTERNAL;
	      }
	    else
	      {
		/* in precision field, set value */
		*value = 0;
	      }
	    break;

	  case '1': case '2': case '3': case '4': case '5':
	  case '6': case '7': case '8': case '9':
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
	    /* something invalid */
	    ASSERT (0);
	    goto next;
	  }
	}

    next:
      /* Stop parsing the current "%" format, look for a new one. */
      ;
    }

  TRACE (printf ("remainder: \"%s\"\n", last_fmt));
  if (*last_fmt != '\0')
    DOPRNT_FORMAT (last_fmt, last_ap);

  if (funs->final != NULL)
    if ((*funs->final) (data) == -1)
      goto error;

 done:
  (*__gmp_free_func) (alloc_fmt, alloc_fmt_size);
  return retval;

 error:
  retval = -1;
  goto done;
}
