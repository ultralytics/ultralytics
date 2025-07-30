/* __gmp_doprnt_mpf -- mpf formatted output.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2001, 2002, 2011 Free Software Foundation, Inc.

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

#include <stdarg.h>    /* for va_list and hence doprnt_funs_t */
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"


/* change this to "#define TRACE(x) x" for diagnostics */
#define TRACE(x)


/* The separate of __gmp_doprnt_float_digits and __gmp_doprnt_float is so
   some C++ can do the mpf_get_str and release it in case of an exception */

#define DIGIT_VALUE(c)                  \
  (isdigit (c)   ? (c) - '0'            \
   : islower (c) ? (c) - 'a' + 10       \
   :               (c) - 'A' + 10)

int
__gmp_doprnt_mpf (const struct doprnt_funs_t *funs,
		  void *data,
		  const struct doprnt_params_t *p,
		  const char *point,
		  mpf_srcptr f)
{
  int         prec, ndigits, free_size, len, newlen, justify, justlen, explen;
  int         showbaselen, sign, signlen, intlen, intzeros, pointlen;
  int         fraczeros, fraclen, preczeros;
  char        *s, *free_ptr;
  mp_exp_t    exp;
  char        exponent[GMP_LIMB_BITS + 10];
  const char  *showbase;
  int         retval = 0;

  TRACE (printf ("__gmp_doprnt_float\n");
	 printf ("  conv=%d prec=%d\n", p->conv, p->prec));

  prec = p->prec;
  if (prec <= -1)
    {
      /* all digits */
      ndigits = 0;

      /* arrange the fixed/scientific decision on a "prec" implied by how
	 many significant digits there are */
      if (p->conv == DOPRNT_CONV_GENERAL)
	MPF_SIGNIFICANT_DIGITS (prec, PREC(f), ABS(p->base));
    }
  else
    {
      switch (p->conv) {
      case DOPRNT_CONV_FIXED:
	/* Precision is digits after the radix point.  Try not to generate
	   too many more than will actually be required.  If f>=1 then
	   overestimate the integer part, and add prec.  If f<1 then
	   underestimate the zeros between the radix point and the first
	   digit and subtract that from prec.  In either case add 2 so the
	   round to nearest can be applied accurately.  Finally, we add 1 to
	   handle the case of 1-eps where EXP(f) = 0 but mpf_get_str returns
	   exp as 1.  */
	ndigits = prec + 2 + 1
	  + EXP(f) * (mp_bases[ABS(p->base)].chars_per_limb + (EXP(f)>=0));
	ndigits = MAX (ndigits, 1);
	break;

      case DOPRNT_CONV_SCIENTIFIC:
	/* precision is digits after the radix point, and there's one digit
	   before */
	ndigits = prec + 1;
	break;

      default:
	ASSERT (0);
	/*FALLTHRU*/

      case DOPRNT_CONV_GENERAL:
	/* precision is total digits, but be sure to ask mpf_get_str for at
	   least 1, not 0 */
	ndigits = MAX (prec, 1);
	break;
      }
    }
  TRACE (printf ("  ndigits %d\n", ndigits));

  s = mpf_get_str (NULL, &exp, p->base, ndigits, f);
  len = strlen (s);
  free_ptr = s;
  free_size = len + 1;
  TRACE (printf ("  s   %s\n", s);
	 printf ("  exp %ld\n", exp);
	 printf ("  len %d\n", len));

  /* For fixed mode check the ndigits formed above was in fact enough for
     the integer part plus p->prec after the radix point. */
  ASSERT ((p->conv == DOPRNT_CONV_FIXED && p->prec > -1)
	  ? ndigits >= MAX (1, exp + p->prec + 2) : 1);

  sign = p->sign;
  if (s[0] == '-')
    {
      sign = s[0];
      s++, len--;
    }
  signlen = (sign != '\0');
  TRACE (printf ("  sign %c  signlen %d\n", sign, signlen));

  switch (p->conv) {
  case DOPRNT_CONV_FIXED:
    if (prec <= -1)
      prec = MAX (0, len-exp);   /* retain all digits */

    /* Truncate if necessary so fraction will be at most prec digits. */
    ASSERT (prec >= 0);
    newlen = exp + prec;
    if (newlen < 0)
      {
	/* first non-zero digit is below target prec, and at least one zero
	   digit in between, so print zero */
	len = 0;
	exp = 0;
      }
    else if (len <= newlen)
      {
	/* already got few enough digits */
      }
    else
      {
	/* discard excess digits and round to nearest */

	const char  *num_to_text = (p->base >= 0
				    ? "0123456789abcdefghijklmnopqrstuvwxyz"
				    : "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
	int  base = ABS(p->base);
	int  n;

	ASSERT (base <= 36);

	len = newlen;
	n = DIGIT_VALUE (s[len]);
	TRACE (printf ("  rounding with %d\n", n));
	if (n >= (base + 1) / 2)
	  {
	    /* propagate a carry */
	    for (;;)
	      {
		if (len == 0)
		  {
		    s[0] = '1';
		    len = 1;
		    exp++;
		    break;
		  }
		n = DIGIT_VALUE (s[len-1]);
		ASSERT (n >= 0 && n < base);
		n++;
		if (n != base)
		  {
		    TRACE (printf ("  storing now %d\n", n));
		    s[len-1] = num_to_text[n];
		    break;
		  }
		len--;
	      }
	  }
	else
	  {
	    /* truncate only, strip any trailing zeros now exposed */
	    while (len > 0 && s[len-1] == '0')
	      len--;
	  }

	/* Can have newlen==0, in which case the truncate was just to check
	   for a carry turning it into "1".  If we're left with len==0 then
	   adjust exp to match.  */
	if (len == 0)
	  exp = 0;
      }

  fixed:
    ASSERT (len == 0 ? exp == 0 : 1);
    if (exp <= 0)
      {
	TRACE (printf ("  fixed 0.000sss\n"));
	intlen = 0;
	intzeros = 1;
	fraczeros = -exp;
	fraclen = len;
      }
    else
      {
	TRACE (printf ("  fixed sss.sss or sss000\n"));
	intlen = MIN (len, exp);
	intzeros = exp - intlen;
	fraczeros = 0;
	fraclen = len - intlen;
      }
    explen = 0;
    break;

  case DOPRNT_CONV_SCIENTIFIC:
    {
      long int expval;
      char  expsign;

      if (prec <= -1)
	prec = MAX (0, len-1);   /* retain all digits */

    scientific:
      TRACE (printf ("  scientific s.sss\n"));

      intlen = MIN (1, len);
      intzeros = (intlen == 0 ? 1 : 0);
      fraczeros = 0;
      fraclen = len - intlen;

      expval = (exp-intlen);
      if (p->exptimes4)
	expval <<= 2;

      /* Split out the sign since %o or %x in expfmt give negatives as twos
	 complement, not with a sign. */
      expsign = (expval >= 0 ? '+' : '-');
      expval = ABS (expval);

#if HAVE_VSNPRINTF
      explen = snprintf (exponent, sizeof(exponent),
			 p->expfmt, expsign, expval);
      /* test for < sizeof-1 since a glibc 2.0.x return of sizeof-1 might
	 mean truncation */
      ASSERT (explen >= 0 && explen < sizeof(exponent)-1);
#else
      sprintf (exponent, p->expfmt, expsign, expval);
      explen = strlen (exponent);
      ASSERT (explen < sizeof(exponent));
#endif
      TRACE (printf ("  expfmt %s gives %s\n", p->expfmt, exponent));
    }
    break;

  default:
    ASSERT (0);
    /*FALLTHRU*/  /* to stop variables looking uninitialized */

  case DOPRNT_CONV_GENERAL:
    /* The exponent for "scientific" will be exp-1, choose scientific if
       this is < -4 or >= prec (and minimum 1 for prec).  For f==0 will have
       exp==0 and get the desired "fixed".  This rule follows glibc.  For
       fixed there's no need to truncate, the desired ndigits will already
       be as required.  */
    if (exp-1 < -4 || exp-1 >= MAX (1, prec))
      goto scientific;
    else
      goto fixed;
  }

  TRACE (printf ("  intlen %d intzeros %d fraczeros %d fraclen %d\n",
		 intlen, intzeros, fraczeros, fraclen));
  ASSERT (p->prec <= -1
	  ? intlen + fraclen == strlen (s)
	  : intlen + fraclen <= strlen (s));

  if (p->showtrailing)
    {
      /* Pad to requested precision with trailing zeros, for general this is
	 all digits, for fixed and scientific just the fraction.  */
      preczeros = prec - (fraczeros + fraclen
			  + (p->conv == DOPRNT_CONV_GENERAL
			     ? intlen + intzeros : 0));
      preczeros = MAX (0, preczeros);
    }
  else
    preczeros = 0;
  TRACE (printf ("  prec=%d showtrailing=%d, pad with preczeros %d\n",
		 prec, p->showtrailing, preczeros));

  /* radix point if needed, or if forced */
  pointlen = ((fraczeros + fraclen + preczeros) != 0 || p->showpoint != 0)
    ? strlen (point) : 0;
  TRACE (printf ("  point |%s|  pointlen %d\n", point, pointlen));

  /* Notice the test for a non-zero value is done after any truncation for
     DOPRNT_CONV_FIXED. */
  showbase = NULL;
  showbaselen = 0;
  switch (p->showbase) {
  default:
    ASSERT (0);
    /*FALLTHRU*/
  case DOPRNT_SHOWBASE_NO:
    break;
  case DOPRNT_SHOWBASE_NONZERO:
    if (intlen == 0 && fraclen == 0)
      break;
    /*FALLTHRU*/
  case DOPRNT_SHOWBASE_YES:
    switch (p->base) {
    case 16:  showbase = "0x"; showbaselen = 2; break;
    case -16: showbase = "0X"; showbaselen = 2; break;
    case 8:   showbase = "0";  showbaselen = 1; break;
    }
    break;
  }
  TRACE (printf ("  showbase %s showbaselen %d\n",
		 showbase == NULL ? "" : showbase, showbaselen));

  /* left over field width */
  justlen = p->width - (signlen + showbaselen + intlen + intzeros + pointlen
			+ fraczeros + fraclen + preczeros + explen);
  TRACE (printf ("  justlen %d fill 0x%X\n", justlen, p->fill));

  justify = p->justify;
  if (justlen <= 0) /* no justifying if exceed width */
    justify = DOPRNT_JUSTIFY_NONE;

  TRACE (printf ("  justify type %d  intlen %d pointlen %d fraclen %d\n",
		 justify, intlen, pointlen, fraclen));

  if (justify == DOPRNT_JUSTIFY_RIGHT)         /* pad for right */
    DOPRNT_REPS (p->fill, justlen);

  if (signlen)                                 /* sign */
    DOPRNT_REPS (sign, 1);

  DOPRNT_MEMORY_MAYBE (showbase, showbaselen); /* base */

  if (justify == DOPRNT_JUSTIFY_INTERNAL)      /* pad for internal */
    DOPRNT_REPS (p->fill, justlen);

  DOPRNT_MEMORY (s, intlen);                   /* integer */
  DOPRNT_REPS_MAYBE ('0', intzeros);

  DOPRNT_MEMORY_MAYBE (point, pointlen);       /* point */

  DOPRNT_REPS_MAYBE ('0', fraczeros);          /* frac */
  DOPRNT_MEMORY_MAYBE (s+intlen, fraclen);

  DOPRNT_REPS_MAYBE ('0', preczeros);          /* prec */

  DOPRNT_MEMORY_MAYBE (exponent, explen);      /* exp */

  if (justify == DOPRNT_JUSTIFY_LEFT)          /* pad for left */
    DOPRNT_REPS (p->fill, justlen);

 done:
  (*__gmp_free_func) (free_ptr, free_size);
  return retval;

 error:
  retval = -1;
  goto done;
}
