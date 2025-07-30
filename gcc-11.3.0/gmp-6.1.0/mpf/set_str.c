/* mpf_set_str (dest, string, base) -- Convert the string STRING
   in base BASE to a float in dest.  If BASE is zero, the leading characters
   of STRING is used to figure out the base.

Copyright 1993-1997, 2000-2003, 2005, 2007, 2008, 2011, 2013 Free Software
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

/*
  This still needs work, as suggested by some FIXME comments.
  1. Don't depend on superfluous mantissa digits.
  2. Allocate temp space more cleverly.
  3. Use mpn_div_q instead of mpn_lshift+mpn_divrem.
*/

#define _GNU_SOURCE    /* for DECIMAL_POINT in langinfo.h */

#include "config.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#if HAVE_LANGINFO_H
#include <langinfo.h>  /* for nl_langinfo */
#endif

#if HAVE_LOCALE_H
#include <locale.h>    /* for localeconv */
#endif

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"


#define digit_value_tab __gmp_digit_value_tab

/* Compute base^exp and return the most significant prec limbs in rp[].
   Put the count of omitted low limbs in *ign.
   Return the actual size (which might be less than prec).  */
static mp_size_t
mpn_pow_1_highpart (mp_ptr rp, mp_size_t *ignp,
		    mp_limb_t base, mp_exp_t exp,
		    mp_size_t prec, mp_ptr tp)
{
  mp_size_t ign;		/* counts number of ignored low limbs in r */
  mp_size_t off;		/* keeps track of offset where value starts */
  mp_ptr passed_rp = rp;
  mp_size_t rn;
  int cnt;
  int i;

  rp[0] = base;
  rn = 1;
  off = 0;
  ign = 0;
  count_leading_zeros (cnt, exp);
  for (i = GMP_LIMB_BITS - cnt - 2; i >= 0; i--)
    {
      mpn_sqr (tp, rp + off, rn);
      rn = 2 * rn;
      rn -= tp[rn - 1] == 0;
      ign <<= 1;

      off = 0;
      if (rn > prec)
	{
	  ign += rn - prec;
	  off = rn - prec;
	  rn = prec;
	}
      MP_PTR_SWAP (rp, tp);

      if (((exp >> i) & 1) != 0)
	{
	  mp_limb_t cy;
	  cy = mpn_mul_1 (rp, rp + off, rn, base);
	  rp[rn] = cy;
	  rn += cy != 0;
	  off = 0;
	}
    }

  if (rn > prec)
    {
      ign += rn - prec;
      rp += rn - prec;
      rn = prec;
    }

  MPN_COPY_INCR (passed_rp, rp + off, rn);
  *ignp = ign;
  return rn;
}

int
mpf_set_str (mpf_ptr x, const char *str, int base)
{
  size_t str_size;
  char *s, *begs;
  size_t i, j;
  int c;
  int negative;
  char *dotpos = 0;
  const char *expptr;
  int exp_base;
  const char  *point = GMP_DECIMAL_POINT;
  size_t      pointlen = strlen (point);
  const unsigned char *digit_value;
  TMP_DECL;

  c = (unsigned char) *str;

  /* Skip whitespace.  */
  while (isspace (c))
    c = (unsigned char) *++str;

  negative = 0;
  if (c == '-')
    {
      negative = 1;
      c = (unsigned char) *++str;
    }

  /* Default base to decimal.  */
  if (base == 0)
    base = 10;

  exp_base = base;

  if (base < 0)
    {
      exp_base = 10;
      base = -base;
    }

  digit_value = digit_value_tab;
  if (base > 36)
    {
      /* For bases > 36, use the collating sequence
	 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.  */
      digit_value += 208;
      if (base > 62)
	return -1;		/* too large base */
    }

  /* Require at least one digit, possibly after an initial decimal point.  */
  if (digit_value[c] >= base)
    {
      /* not a digit, must be a decimal point */
      for (i = 0; i < pointlen; i++)
	if (str[i] != point[i])
	  return -1;
      if (digit_value[(unsigned char) str[pointlen]] >= base)
	return -1;
    }

  /* Locate exponent part of the input.  Look from the right of the string,
     since the exponent is usually a lot shorter than the mantissa.  */
  expptr = NULL;
  str_size = strlen (str);
  for (i = str_size - 1; i > 0; i--)
    {
      c = (unsigned char) str[i];
      if (c == '@' || (base <= 10 && (c == 'e' || c == 'E')))
	{
	  expptr = str + i + 1;
	  str_size = i;
	  break;
	}
    }

  TMP_MARK;
  s = begs = (char *) TMP_ALLOC (str_size + 1);

  /* Loop through mantissa, converting it from ASCII to raw byte values.  */
  for (i = 0; i < str_size; i++)
    {
      c = (unsigned char) *str;
      if (!isspace (c))
	{
	  int dig;

	  for (j = 0; j < pointlen; j++)
	    if (str[j] != point[j])
	      goto not_point;
	  if (1)
	    {
	      if (dotpos != 0)
		{
		  /* already saw a decimal point, another is invalid */
		  TMP_FREE;
		  return -1;
		}
	      dotpos = s;
	      str += pointlen - 1;
	      i += pointlen - 1;
	    }
	  else
	    {
	    not_point:
	      dig = digit_value[c];
	      if (dig >= base)
		{
		  TMP_FREE;
		  return -1;
		}
	      *s++ = dig;
	    }
	}
      c = (unsigned char) *++str;
    }

  str_size = s - begs;

  {
    long exp_in_base;
    mp_size_t ra, ma, rn, mn;
    int cnt;
    mp_ptr mp, tp, rp;
    mp_exp_t exp_in_limbs;
    mp_size_t prec = PREC(x) + 1;
    int divflag;
    mp_size_t madj, radj;

#if 0
    size_t n_chars_needed;

    /* This breaks things like 0.000...0001.  To safely ignore superfluous
       digits, we need to skip over leading zeros.  */
    /* Just consider the relevant leading digits of the mantissa.  */
    LIMBS_PER_DIGIT_IN_BASE (n_chars_needed, prec, base);
    if (str_size > n_chars_needed)
      str_size = n_chars_needed;
#endif

    LIMBS_PER_DIGIT_IN_BASE (ma, str_size, base);
    mp = TMP_ALLOC_LIMBS (ma);
    mn = mpn_set_str (mp, (unsigned char *) begs, str_size, base);

    if (mn == 0)
      {
	SIZ(x) = 0;
	EXP(x) = 0;
	TMP_FREE;
	return 0;
      }

    madj = 0;
    /* Ignore excess limbs in MP,MSIZE.  */
    if (mn > prec)
      {
	madj = mn - prec;
	mp += mn - prec;
	mn = prec;
      }

    if (expptr != 0)
      {
	/* Scan and convert the exponent, in base exp_base.  */
	long dig, minus, plusminus;
	c = (unsigned char) *expptr;
	minus = -(long) (c == '-');
	plusminus = minus | -(long) (c == '+');
	expptr -= plusminus;			/* conditional increment */
	c = (unsigned char) *expptr++;
	dig = digit_value[c];
	if (dig >= exp_base)
	  {
	    TMP_FREE;
	    return -1;
	  }
	exp_in_base = dig;
	c = (unsigned char) *expptr++;
	dig = digit_value[c];
	while (dig < exp_base)
	  {
	    exp_in_base = exp_in_base * exp_base;
	    exp_in_base += dig;
	    c = (unsigned char) *expptr++;
	    dig = digit_value[c];
	  }
	exp_in_base = (exp_in_base ^ minus) - minus; /* conditional negation */
      }
    else
      exp_in_base = 0;
    if (dotpos != 0)
      exp_in_base -= s - dotpos;
    divflag = exp_in_base < 0;
    exp_in_base = ABS (exp_in_base);

    if (exp_in_base == 0)
      {
	MPN_COPY (PTR(x), mp, mn);
	SIZ(x) = negative ? -mn : mn;
	EXP(x) = mn + madj;
	TMP_FREE;
	return 0;
      }

    ra = 2 * (prec + 1);
    rp = TMP_ALLOC_LIMBS (ra);
    tp = TMP_ALLOC_LIMBS (ra);
    rn = mpn_pow_1_highpart (rp, &radj, (mp_limb_t) base, exp_in_base, prec, tp);

    if (divflag)
      {
#if 0
	/* FIXME: Should use mpn_div_q here.  */
	...
	mpn_div_q (tp, mp, mn, rp, rn, scratch);
	...
#else
	mp_ptr qp;
	mp_limb_t qlimb;
	if (mn < rn)
	  {
	    /* Pad out MP,MSIZE for current divrem semantics.  */
	    mp_ptr tmp = TMP_ALLOC_LIMBS (rn + 1);
	    MPN_ZERO (tmp, rn - mn);
	    MPN_COPY (tmp + rn - mn, mp, mn);
	    mp = tmp;
	    madj -= rn - mn;
	    mn = rn;
	  }
	if ((rp[rn - 1] & GMP_NUMB_HIGHBIT) == 0)
	  {
	    mp_limb_t cy;
	    count_leading_zeros (cnt, rp[rn - 1]);
	    cnt -= GMP_NAIL_BITS;
	    mpn_lshift (rp, rp, rn, cnt);
	    cy = mpn_lshift (mp, mp, mn, cnt);
	    if (cy)
	      mp[mn++] = cy;
	  }

	qp = TMP_ALLOC_LIMBS (prec + 1);
	qlimb = mpn_divrem (qp, prec - (mn - rn), mp, mn, rp, rn);
	tp = qp;
	exp_in_limbs = qlimb + (mn - rn) + (madj - radj);
	rn = prec;
	if (qlimb != 0)
	  {
	    tp[prec] = qlimb;
	    /* Skip the least significant limb not to overrun the destination
	       variable.  */
	    tp++;
	  }
#endif
      }
    else
      {
	tp = TMP_ALLOC_LIMBS (rn + mn);
	if (rn > mn)
	  mpn_mul (tp, rp, rn, mp, mn);
	else
	  mpn_mul (tp, mp, mn, rp, rn);
	rn += mn;
	rn -= tp[rn - 1] == 0;
	exp_in_limbs = rn + madj + radj;

	if (rn > prec)
	  {
	    tp += rn - prec;
	    rn = prec;
	    exp_in_limbs += 0;
	  }
      }

    MPN_COPY (PTR(x), tp, rn);
    SIZ(x) = negative ? -rn : rn;
    EXP(x) = exp_in_limbs;
    TMP_FREE;
    return 0;
  }
}
