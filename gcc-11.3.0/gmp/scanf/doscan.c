/* __gmp_doscan -- formatted input internals.

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

#define _GNU_SOURCE    /* for DECIMAL_POINT in langinfo.h */

#include "config.h"	/* needed for the HAVE_, could also move gmp incls */

#include <stdarg.h>
#include <ctype.h>
#include <stddef.h>    /* for ptrdiff_t */
#include <stdio.h>
#include <stdlib.h>    /* for strtol */
#include <string.h>

#if HAVE_LANGINFO_H
#include <langinfo.h>  /* for nl_langinfo */
#endif

#if HAVE_LOCALE_H
#include <locale.h>    /* for localeconv */
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


/* Change this to "#define TRACE(x) x" for some traces. */
#define TRACE(x)


/* General:

       It's necessary to parse up the format string to recognise the GMP
       extra types F, Q and Z.  Other types and conversions are passed
       across to the standard sscanf or fscanf via funs->scan, for ease of
       implementation.  This is essential in the case of something like glibc
       %p where the pointer format isn't actually documented.

       Because funs->scan doesn't get the whole input it can't put the right
       values in for %n, so that's handled in __gmp_doscan.  Neither sscanf
       nor fscanf directly indicate how many characters were read, so an
       extra %n is appended to each run for that.  For fscanf this merely
       supports our %n output, but for sscanf it lets funs->step move us
       along the input string.

       Whitespace and literal matches in the format string, including %%,
       are handled directly within __gmp_doscan.  This is reasonably
       efficient, and avoids some suspicious behaviour observed in various
       system libc's.  GLIBC 2.2.4 for instance returns 0 on

	   sscanf(" ", " x")
       or
	   sscanf(" ", " x%d",&n)

       whereas we think they should return EOF, since end-of-string is
       reached when a match of "x" is required.

       For standard % conversions, funs->scan is called once for each
       conversion.  If we had vfscanf and vsscanf and could rely on their
       fixed text matching behaviour then we could call them with multiple
       consecutive standard conversions.  But plain fscanf and sscanf work
       fine, and parsing one field at a time shouldn't be too much of a
       slowdown.

   gmpscan:

       gmpscan reads a gmp type.  It's only used from one place, but is a
       separate subroutine to avoid a big chunk of complicated code in the
       middle of __gmp_doscan.  Within gmpscan a couple of loopbacks make it
       possible to share code for parsing integers, rationals and floats.

       In gmpscan normally one char of lookahead is maintained, but when width
       is reached that stops, on the principle that an fgetc/ungetc of a char
       past where we're told to stop would be undesirable.  "chars" is how many
       characters have been read so far, including the current c.  When
       chars==width and another character is desired then a jump is done to the
       "convert" stage.  c is invalid and mustn't be unget'ed in this case;
       chars is set to width+1 to indicate that.

       gmpscan normally returns the number of characters read.  -1 means an
       invalid field, -2 means EOF reached before any matching characters
       were read.

       For hex floats, the mantissa part is passed to mpf_set_str, then the
       exponent is applied with mpf_mul_exp or mpf_div_2exp.  This is easier
       than teaching mpf_set_str about an exponent factor (ie. 2) differing
       from the mantissa radix point factor (ie. 16).  mpf_mul_exp and
       mpf_div_2exp will preserve the application requested precision, so
       nothing in that respect is lost by making this a two-step process.

   Matching and errors:

       C99 7.19.6.2 paras 9 and 10 say an input item is read as the longest
       string which is a match for the appropriate type, or a prefix of a
       match.  With that done, if it's only a prefix then the result is a
       matching failure, ie. invalid input.

       This rule seems fairly clear, but doesn't seem to be universally
       applied in system C libraries.  Even GLIBC doesn't seem to get it
       right, insofar as it seems to accept some apparently invalid forms.
       Eg. glibc 2.3.1 accepts "0x" for a "%i", where a reading of the
       standard would suggest a non-empty sequence of digits should be
       required after an "0x".

       A footnote to 7.19.6.2 para 17 notes how this input item reading can
       mean inputs acceptable to strtol are not acceptable to fscanf.  We
       think this confirms our reading of "0x" as invalid.

       Clearly gmp_sscanf could backtrack to a longest input which was a
       valid match for a given item, but this is not done, since C99 says
       sscanf is identical to fscanf, so we make gmp_sscanf identical to
       gmp_fscanf.

   Types:

       C99 says "ll" is for long long, and "L" is for long double floats.
       Unfortunately in GMP 4.1.1 we documented the two as equivalent.  This
       doesn't affect us directly, since both are passed through to plain
       scanf.  It seems wisest not to try to enforce the C99 rule.  This is
       consistent with what we said before, though whether it actually
       worked was always up to the C library.

   Alternatives:

       Consideration was given to using separate code for gmp_fscanf and
       gmp_sscanf.  The sscanf case could zip across a string doing literal
       matches or recognising digits in gmpscan, rather than making a
       function call fun->get per character.  The fscanf could use getc
       rather than fgetc too, which might help those systems where getc is a
       macro or otherwise inlined.  But none of this scanning and converting
       will be particularly fast, so the two are done together to keep it a
       little simpler for now.

       Various multibyte string issues are not addressed, for a start C99
       scanf says the format string is multibyte.  Since we pass %c, %s and
       %[ to the system scanf, they might do multibyte reads already, but
       it's another matter whether or not that can be used, since our digit
       and whitespace parsing is only unibyte.  The plan is to quietly
       ignore multibyte locales for now.  This is not as bad as it sounds,
       since GMP is presumably used mostly on numbers, which can be
       perfectly adequately treated in plain ASCII.

*/


struct gmp_doscan_params_t {
  int	base;
  int	ignore;
  char	type;
  int	width;
};


#define GET(c)			\
  do {				\
    ASSERT (chars <= width);	\
    chars++;			\
    if (chars > width)		\
      goto convert;		\
    (c) = (*funs->get) (data);	\
  } while (0)

/* store into "s", extending if necessary */
#define STORE(c)							\
  do {									\
    ASSERT (s_upto <= s_alloc);						\
    if (s_upto >= s_alloc)						\
      {									\
	size_t	s_alloc_new = s_alloc + S_ALLOC_STEP;			\
	s = __GMP_REALLOCATE_FUNC_TYPE (s, s_alloc, s_alloc_new, char); \
	s_alloc = s_alloc_new;						\
      }									\
    s[s_upto++] = c;							\
  } while (0)

#define S_ALLOC_STEP  512

static int
gmpscan (const struct gmp_doscan_funs_t *funs, void *data,
	 const struct gmp_doscan_params_t *p, void *dst)
{
  int	  chars, c, base, first, width, seen_point, seen_digit, hexfloat;
  size_t  s_upto, s_alloc, hexexp;
  char	  *s;
  int	  invalid = 0;

  TRACE (printf ("gmpscan\n"));

  ASSERT (p->type == 'F' || p->type == 'Q' || p->type == 'Z');

  c = (*funs->get) (data);
  if (c == EOF)
    return -2;

  chars = 1;
  first = 1;
  seen_point = 0;
  width = (p->width == 0 ? INT_MAX-1 : p->width);
  base = p->base;
  s_alloc = S_ALLOC_STEP;
  s = __GMP_ALLOCATE_FUNC_TYPE (s_alloc, char);
  s_upto = 0;
  hexfloat = 0;
  hexexp = 0;

 another:
  seen_digit = 0;
  if (c == '-')
    {
      STORE (c);
      goto get_for_sign;
    }
  else if (c == '+')
    {
      /* don't store '+', it's not accepted by mpz_set_str etc */
    get_for_sign:
      GET (c);
    }

  if (base == 0)
    {
      base = 10;		  /* decimal if no base indicator */
      if (c == '0')
	{
	  seen_digit = 1;	  /* 0 alone is a valid number */
	  if (p->type != 'F')
	    base = 8;		  /* leading 0 is octal, for non-floats */
	  STORE (c);
	  GET (c);
	  if (c == 'x' || c == 'X')
	    {
	      base = 16;
	      seen_digit = 0;	  /* must have digits after an 0x */
	      if (p->type == 'F') /* don't pass 'x' to mpf_set_str_point */
		hexfloat = 1;
	      else
		STORE (c);
	      GET (c);
	    }
	}
    }

 digits:
  for (;;)
    {
      if (base == 16)
	{
	  if (! isxdigit (c))
	    break;
	}
      else
	{
	  if (! isdigit (c))
	    break;
	  if (base == 8 && (c == '8' || c == '9'))
	    break;
	}

      seen_digit = 1;
      STORE (c);
      GET (c);
    }

  if (first)
    {
      /* decimal point */
      if (p->type == 'F' && ! seen_point)
	{
	  /* For a multi-character decimal point, if the first character is
	     present then all of it must be, otherwise the input is
	     considered invalid.  */
	  const char  *point = GMP_DECIMAL_POINT;
	  int	      pc = (unsigned char) *point++;
	  if (c == pc)
	    {
	      for (;;)
		{
		  STORE (c);
		  GET (c);
		  pc = (unsigned char) *point++;
		  if (pc == '\0')
		    break;
		  if (c != pc)
		    goto set_invalid;
		}
	      seen_point = 1;
	      goto digits;
	    }
	}

      /* exponent */
      if (p->type == 'F')
	{
	  if (hexfloat && (c == 'p' || c == 'P'))
	    {
	      hexexp = s_upto; /* exponent location */
	      base = 10;       /* exponent in decimal */
	      goto exponent;
	    }
	  else if (! hexfloat && (c == 'e' || c == 'E'))
	    {
	    exponent:
	      /* must have at least one digit in the mantissa, just an exponent
		 is not good enough */
	      if (! seen_digit)
		goto set_invalid;

	    do_second:
	      first = 0;
	      STORE (c);
	      GET (c);
	      goto another;
	    }
	}

      /* denominator */
      if (p->type == 'Q' && c == '/')
	{
	  /* must have at least one digit in the numerator */
	  if (! seen_digit)
	    goto set_invalid;

	  /* now look for at least one digit in the denominator */
	  seen_digit = 0;

	  /* allow the base to be redetermined for "%i" */
	  base = p->base;
	  goto do_second;
	}
    }

 convert:
  if (! seen_digit)
    {
    set_invalid:
      invalid = 1;
      goto done;
    }

  if (! p->ignore)
    {
      STORE ('\0');
      TRACE (printf ("	convert \"%s\"\n", s));

      /* We ought to have parsed out a valid string above, so just test
	 mpz_set_str etc with an ASSERT.  */
      switch (p->type) {
      case 'F':
	{
	  mpf_ptr  f = (mpf_ptr) dst;
	  if (hexexp != 0)
	    s[hexexp] = '\0';
	  ASSERT_NOCARRY (mpf_set_str (f, s, hexfloat ? 16 : 10));
	  if (hexexp != 0)
	    {
	      char *dummy;
	      long  exp;
	      exp = strtol (s + hexexp + 1, &dummy, 10);
	      if (exp >= 0)
		mpf_mul_2exp (f, f, (unsigned long) exp);
	      else
		mpf_div_2exp (f, f, - (unsigned long) exp);
	    }
	}
	break;
      case 'Q':
	ASSERT_NOCARRY (mpq_set_str ((mpq_ptr) dst, s, p->base));
	break;
      case 'Z':
	ASSERT_NOCARRY (mpz_set_str ((mpz_ptr) dst, s, p->base));
	break;
      default:
	ASSERT (0);
	/*FALLTHRU*/
	break;
      }
    }

 done:
  ASSERT (chars <= width+1);
  if (chars != width+1)
    {
      (*funs->unget) (c, data);
      TRACE (printf ("	ungetc %d, to give %d chars\n", c, chars-1));
    }
  chars--;

  (*__gmp_free_func) (s, s_alloc);

  if (invalid)
    {
      TRACE (printf ("	invalid\n"));
      return -1;
    }

  TRACE (printf ("  return %d chars (cf width %d)\n", chars, width));
  return chars;
}


/* Read and discard whitespace, if any.  Return number of chars skipped.
   Whitespace skipping never provokes the EOF return from __gmp_doscan, so
   it's not necessary to watch for EOF from funs->get, */
static int
skip_white (const struct gmp_doscan_funs_t *funs, void *data)
{
  int  c;
  int  ret = 0;

  do
    {
      c = (funs->get) (data);
      ret++;
    }
  while (isspace (c));

  (funs->unget) (c, data);
  ret--;

  TRACE (printf ("  skip white %d\n", ret));
  return ret;
}


int
__gmp_doscan (const struct gmp_doscan_funs_t *funs, void *data,
	      const char *orig_fmt, va_list orig_ap)
{
  struct gmp_doscan_params_t  param;
  va_list     ap;
  char	      *alloc_fmt;
  const char  *fmt, *this_fmt, *end_fmt;
  size_t      orig_fmt_len, alloc_fmt_size, len;
  int	      new_fields, new_chars;
  char	      fchar;
  int	      fields = 0;
  int	      chars = 0;

  TRACE (printf ("__gmp_doscan \"%s\"\n", orig_fmt);
	 if (funs->scan == (gmp_doscan_scan_t) sscanf)
	   printf ("  s=\"%s\"\n", * (const char **) data));

  /* Don't modify orig_ap, if va_list is actually an array and hence call by
     reference.  It could be argued that it'd be more efficient to leave
     callers to make a copy if they care, but doing so here is going to be a
     very small part of the total work, and we may as well keep applications
     out of trouble.  */
  va_copy (ap, orig_ap);

  /* Parts of the format string are going to be copied so that a " %n" can
     be appended.  alloc_fmt is some space for that.  orig_fmt_len+4 will be
     needed if fmt consists of a single "%" specifier, but otherwise is an
     overestimate.  We're not going to be very fast here, so use
     __gmp_allocate_func rather than TMP_ALLOC.  */
  orig_fmt_len = strlen (orig_fmt);
  alloc_fmt_size = orig_fmt_len + 4;
  alloc_fmt = __GMP_ALLOCATE_FUNC_TYPE (alloc_fmt_size, char);

  fmt = orig_fmt;
  end_fmt = orig_fmt + orig_fmt_len;

  for (;;)
    {
    next:
      fchar = *fmt++;

      if (fchar == '\0')
	break;

      if (isspace (fchar))
	{
	  chars += skip_white (funs, data);
	  continue;
	}

      if (fchar != '%')
	{
	  int  c;
	literal:
	  c = (funs->get) (data);
	  if (c != fchar)
	    {
	      (funs->unget) (c, data);
	      if (c == EOF)
		{
		eof_no_match:
		  if (fields == 0)
		    fields = EOF;
		}
	      goto done;
	    }
	  chars++;
	  continue;
	}

      param.type = '\0';
      param.base = 0;	 /* for e,f,g,i */
      param.ignore = 0;
      param.width = 0;

      this_fmt = fmt-1;
      TRACE (printf ("	this_fmt \"%s\"\n", this_fmt));

      for (;;)
	{
	  ASSERT (fmt <= end_fmt);

	  fchar = *fmt++;
	  switch (fchar) {

	  case '\0':  /* unterminated % sequence */
	    ASSERT (0);
	    goto done;

	  case '%':   /* literal % */
	    goto literal;

	  case '[':   /* character range */
	    fchar = *fmt++;
	    if (fchar == '^')
	      fchar = *fmt++;
	    /* ']' allowed as the first char (possibly after '^') */
	    if (fchar == ']')
	      fchar = *fmt++;
	    for (;;)
	      {
		ASSERT (fmt <= end_fmt);
		if (fchar == '\0')
		  {
		    /* unterminated % sequence */
		    ASSERT (0);
		    goto done;
		  }
		if (fchar == ']')
		  break;
		fchar = *fmt++;
	      }
	    /*FALLTHRU*/
	  case 'c':   /* characters */
	  case 's':   /* string of non-whitespace */
	  case 'p':   /* pointer */
	  libc_type:
	    len = fmt - this_fmt;
	    memcpy (alloc_fmt, this_fmt, len);
	    alloc_fmt[len++] = '%';
	    alloc_fmt[len++] = 'n';
	    alloc_fmt[len] = '\0';

	    TRACE (printf ("  scan \"%s\"\n", alloc_fmt);
		   if (funs->scan == (gmp_doscan_scan_t) sscanf)
		     printf ("	s=\"%s\"\n", * (const char **) data));

	    new_chars = -1;
	    if (param.ignore)
	      {
		new_fields = (*funs->scan) (data, alloc_fmt, &new_chars, NULL);
		ASSERT (new_fields == 0 || new_fields == EOF);
	      }
	    else
	      {
		void *arg = va_arg (ap, void *);
		new_fields = (*funs->scan) (data, alloc_fmt, arg, &new_chars);
		ASSERT (new_fields==0 || new_fields==1 || new_fields==EOF);

		if (new_fields == 0)
		  goto done;  /* invalid input */

		if (new_fields == 1)
		  ASSERT (new_chars != -1);
	      }
	    TRACE (printf ("  new_fields %d   new_chars %d\n",
			   new_fields, new_chars));

	    if (new_fields == -1)
	      goto eof_no_match;  /* EOF before anything matched */

	    /* Under param.ignore, when new_fields==0 we don't know if
	       it's a successful match or an invalid field.  new_chars
	       won't have been assigned if it was an invalid field.  */
	    if (new_chars == -1)
	      goto done;  /* invalid input */

	    chars += new_chars;
	    (*funs->step) (data, new_chars);

	  increment_fields:
	    if (! param.ignore)
	      fields++;
	    goto next;

	  case 'd':   /* decimal */
	  case 'u':   /* decimal */
	    param.base = 10;
	    goto numeric;

	  case 'e':   /* float */
	  case 'E':   /* float */
	  case 'f':   /* float */
	  case 'g':   /* float */
	  case 'G':   /* float */
	  case 'i':   /* integer with base marker */
	  numeric:
	    if (param.type != 'F' && param.type != 'Q' && param.type != 'Z')
	      goto libc_type;

	    chars += skip_white (funs, data);

	    new_chars = gmpscan (funs, data, &param,
				 param.ignore ? NULL : va_arg (ap, void*));
	    if (new_chars == -2)
	      goto eof_no_match;
	    if (new_chars == -1)
	      goto done;

	    ASSERT (new_chars >= 0);
	    chars += new_chars;
	    goto increment_fields;

	  case 'a':   /* glibc allocate string */
	  case '\'':  /* glibc digit groupings */
	    break;

	  case 'F':   /* mpf_t */
	  case 'j':   /* intmax_t */
	  case 'L':   /* long long */
	  case 'q':   /* quad_t */
	  case 'Q':   /* mpq_t */
	  case 't':   /* ptrdiff_t */
	  case 'z':   /* size_t */
	  case 'Z':   /* mpz_t */
	  set_type:
	    param.type = fchar;
	    break;

	  case 'h':   /* short or char */
	    if (param.type != 'h')
	      goto set_type;
	    param.type = 'H';	/* internal code for "hh" */
	    break;

	    goto numeric;

	  case 'l':   /* long, long long, double or long double */
	    if (param.type != 'l')
	      goto set_type;
	    param.type = 'L';	/* "ll" means "L" */
	    break;

	  case 'n':
	    if (! param.ignore)
	      {
		void  *p;
		p = va_arg (ap, void *);
		TRACE (printf ("  store %%n to %p\n", p));
		switch (param.type) {
		case '\0': * (int	*) p = chars; break;
		case 'F':  mpf_set_si ((mpf_ptr) p, (long) chars); break;
		case 'H':  * (char	*) p = chars; break;
		case 'h':  * (short	*) p = chars; break;
#if HAVE_INTMAX_T
		case 'j':  * (intmax_t	*) p = chars; break;
#else
		case 'j':  ASSERT_FAIL (intmax_t not available); break;
#endif
		case 'l':  * (long	*) p = chars; break;
#if HAVE_QUAD_T && HAVE_LONG_LONG
		case 'q':
		  ASSERT_ALWAYS (sizeof (quad_t) == sizeof (long long));
		  /*FALLTHRU*/
#else
		case 'q':  ASSERT_FAIL (quad_t not available); break;
#endif
#if HAVE_LONG_LONG
		case 'L':  * (long long *) p = chars; break;
#else
		case 'L':  ASSERT_FAIL (long long not available); break;
#endif
		case 'Q':  mpq_set_si ((mpq_ptr) p, (long) chars, 1L); break;
#if HAVE_PTRDIFF_T
		case 't':  * (ptrdiff_t *) p = chars; break;
#else
		case 't':  ASSERT_FAIL (ptrdiff_t not available); break;
#endif
		case 'z':  * (size_t	*) p = chars; break;
		case 'Z':  mpz_set_si ((mpz_ptr) p, (long) chars); break;
		default: ASSERT (0); break;
		}
	      }
	    goto next;

	  case 'o':
	    param.base = 8;
	    goto numeric;

	  case 'x':
	  case 'X':
	    param.base = 16;
	    goto numeric;

	  case '0': case '1': case '2': case '3': case '4':
	  case '5': case '6': case '7': case '8': case '9':
	    param.width = 0;
	    do {
	      param.width = param.width * 10 + (fchar-'0');
	      fchar = *fmt++;
	    } while (isdigit (fchar));
	    fmt--; /* unget the non-digit */
	    break;

	  case '*':
	    param.ignore = 1;
	    break;

	  default:
	    /* something invalid in a % sequence */
	    ASSERT (0);
	    goto next;
	  }
	}
    }

 done:
  (*__gmp_free_func) (alloc_fmt, alloc_fmt_size);
  return fields;
}
