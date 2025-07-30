/*
  Name:     imrat.c
  Purpose:  Arbitrary precision rational arithmetic routines.
  Author:   M. J. Fromberger <http://spinning-yarns.org/michael/>

  Copyright (C) 2002-2007 Michael J. Fromberger, All Rights Reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
 */

#include "imrat.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#define TEMP(K) (temp + (K))
#define SETUP(E, C) \
do{if((res = (E)) != MP_OK) goto CLEANUP; ++(C);}while(0)

/* Argument checking:
   Use CHECK() where a return value is required; NRCHECK() elsewhere */
#define CHECK(TEST)   assert(TEST)
#define NRCHECK(TEST) assert(TEST)

/* Reduce the given rational, in place, to lowest terms and canonical form.
   Zero is represented as 0/1, one as 1/1.  Signs are adjusted so that the sign
   of the numerator is definitive. */
static mp_result s_rat_reduce(mp_rat r);

/* Common code for addition and subtraction operations on rationals. */
static mp_result s_rat_combine(mp_rat a, mp_rat b, mp_rat c,
			       mp_result (*comb_f)(mp_int, mp_int, mp_int));

mp_result mp_rat_init(mp_rat r)
{
  mp_result res;

  if ((res = mp_int_init(MP_NUMER_P(r))) != MP_OK)
    return res;
  if ((res = mp_int_init(MP_DENOM_P(r))) != MP_OK) {
    mp_int_clear(MP_NUMER_P(r));
    return res;
  }

  return mp_int_set_value(MP_DENOM_P(r), 1);
}

mp_rat mp_rat_alloc(void)
{
  mp_rat out = malloc(sizeof(*out));

  if (out != NULL) {
    if (mp_rat_init(out) != MP_OK) {
      free(out);
      return NULL;
    }
  }

  return out;
}

mp_result mp_rat_reduce(mp_rat r) {
  return s_rat_reduce(r);
}

mp_result mp_rat_init_size(mp_rat r, mp_size n_prec, mp_size d_prec)
{
  mp_result res;

  if ((res = mp_int_init_size(MP_NUMER_P(r), n_prec)) != MP_OK)
    return res;
  if ((res = mp_int_init_size(MP_DENOM_P(r), d_prec)) != MP_OK) {
    mp_int_clear(MP_NUMER_P(r));
    return res;
  }
  
  return mp_int_set_value(MP_DENOM_P(r), 1);
}

mp_result mp_rat_init_copy(mp_rat r, mp_rat old)
{
  mp_result res;

  if ((res = mp_int_init_copy(MP_NUMER_P(r), MP_NUMER_P(old))) != MP_OK)
    return res;
  if ((res = mp_int_init_copy(MP_DENOM_P(r), MP_DENOM_P(old))) != MP_OK) 
    mp_int_clear(MP_NUMER_P(r));
  
  return res;
}

mp_result mp_rat_set_value(mp_rat r, mp_small numer, mp_small denom)
{
  mp_result res;

  if (denom == 0)
    return MP_UNDEF;

  if ((res = mp_int_set_value(MP_NUMER_P(r), numer)) != MP_OK)
    return res;
  if ((res = mp_int_set_value(MP_DENOM_P(r), denom)) != MP_OK)
    return res;

  return s_rat_reduce(r);
}

mp_result mp_rat_set_uvalue(mp_rat r, mp_usmall numer, mp_usmall denom)
{
  mp_result res;

  if (denom == 0)
    return MP_UNDEF;

  if ((res = mp_int_set_uvalue(MP_NUMER_P(r), numer)) != MP_OK)
    return res;
  if ((res = mp_int_set_uvalue(MP_DENOM_P(r), denom)) != MP_OK)
    return res;

  return s_rat_reduce(r);
}

void      mp_rat_clear(mp_rat r)
{
  mp_int_clear(MP_NUMER_P(r));
  mp_int_clear(MP_DENOM_P(r));

}

void      mp_rat_free(mp_rat r)
{
  NRCHECK(r != NULL);
  
  if (r->num.digits != NULL)
    mp_rat_clear(r);

  free(r);
}

mp_result mp_rat_numer(mp_rat r, mp_int z)
{
  return mp_int_copy(MP_NUMER_P(r), z);
}

mp_int mp_rat_numer_ref(mp_rat r)
{
  return MP_NUMER_P(r);
}


mp_result mp_rat_denom(mp_rat r, mp_int z)
{
  return mp_int_copy(MP_DENOM_P(r), z);
}

mp_int    mp_rat_denom_ref(mp_rat r)
{
  return MP_DENOM_P(r);
}

mp_sign   mp_rat_sign(mp_rat r)
{
  return MP_SIGN(MP_NUMER_P(r));
}

mp_result mp_rat_copy(mp_rat a, mp_rat c)
{
  mp_result res;

  if ((res = mp_int_copy(MP_NUMER_P(a), MP_NUMER_P(c))) != MP_OK)
    return res;
  
  res = mp_int_copy(MP_DENOM_P(a), MP_DENOM_P(c));
  return res;
}

void      mp_rat_zero(mp_rat r)
{
  mp_int_zero(MP_NUMER_P(r));
  mp_int_set_value(MP_DENOM_P(r), 1);
  
}

mp_result mp_rat_abs(mp_rat a, mp_rat c)
{
  mp_result res;

  if ((res = mp_int_abs(MP_NUMER_P(a), MP_NUMER_P(c))) != MP_OK)
    return res;
  
  res = mp_int_abs(MP_DENOM_P(a), MP_DENOM_P(c));
  return res;
}

mp_result mp_rat_neg(mp_rat a, mp_rat c)
{
  mp_result res;

  if ((res = mp_int_neg(MP_NUMER_P(a), MP_NUMER_P(c))) != MP_OK)
    return res;

  res = mp_int_copy(MP_DENOM_P(a), MP_DENOM_P(c));
  return res;
}

mp_result mp_rat_recip(mp_rat a, mp_rat c)
{
  mp_result res;

  if (mp_rat_compare_zero(a) == 0)
    return MP_UNDEF;

  if ((res = mp_rat_copy(a, c)) != MP_OK)
    return res;

  mp_int_swap(MP_NUMER_P(c), MP_DENOM_P(c));

  /* Restore the signs of the swapped elements */
  {
    mp_sign tmp = MP_SIGN(MP_NUMER_P(c));

    MP_SIGN(MP_NUMER_P(c)) = MP_SIGN(MP_DENOM_P(c));
    MP_SIGN(MP_DENOM_P(c)) = tmp;
  }

  return MP_OK;
}

mp_result mp_rat_add(mp_rat a, mp_rat b, mp_rat c)
{
  return s_rat_combine(a, b, c, mp_int_add);

}

mp_result mp_rat_sub(mp_rat a, mp_rat b, mp_rat c)
{
  return s_rat_combine(a, b, c, mp_int_sub);

}

mp_result mp_rat_mul(mp_rat a, mp_rat b, mp_rat c)
{
  mp_result res;

  if ((res = mp_int_mul(MP_NUMER_P(a), MP_NUMER_P(b), MP_NUMER_P(c))) != MP_OK)
    return res;

  if (mp_int_compare_zero(MP_NUMER_P(c)) != 0) {
    if ((res = mp_int_mul(MP_DENOM_P(a), MP_DENOM_P(b), MP_DENOM_P(c))) != MP_OK)
      return res;
  }

  return s_rat_reduce(c);
}

mp_result mp_rat_div(mp_rat a, mp_rat b, mp_rat c)
{
  mp_result res = MP_OK;

  if (mp_rat_compare_zero(b) == 0)
    return MP_UNDEF;

  if (c == a || c == b) {
    mpz_t tmp;

    if ((res = mp_int_init(&tmp)) != MP_OK) return res;
    if ((res = mp_int_mul(MP_NUMER_P(a), MP_DENOM_P(b), &tmp)) != MP_OK) 
      goto CLEANUP;
    if ((res = mp_int_mul(MP_DENOM_P(a), MP_NUMER_P(b), MP_DENOM_P(c))) != MP_OK)
      goto CLEANUP;
    res = mp_int_copy(&tmp, MP_NUMER_P(c));

  CLEANUP:
    mp_int_clear(&tmp);
  }
  else {
    if ((res = mp_int_mul(MP_NUMER_P(a), MP_DENOM_P(b), MP_NUMER_P(c))) != MP_OK)
      return res;
    if ((res = mp_int_mul(MP_DENOM_P(a), MP_NUMER_P(b), MP_DENOM_P(c))) != MP_OK)
      return res;
  }

  if (res != MP_OK)
    return res;
  else
    return s_rat_reduce(c);
}

mp_result mp_rat_add_int(mp_rat a, mp_int b, mp_rat c)
{
  mpz_t tmp;
  mp_result res;

  if ((res = mp_int_init_copy(&tmp, b)) != MP_OK)
    return res;

  if ((res = mp_int_mul(&tmp, MP_DENOM_P(a), &tmp)) != MP_OK)
    goto CLEANUP;

  if ((res = mp_rat_copy(a, c)) != MP_OK)
    goto CLEANUP;

  if ((res = mp_int_add(MP_NUMER_P(c), &tmp, MP_NUMER_P(c))) != MP_OK)
    goto CLEANUP;

  res = s_rat_reduce(c);

 CLEANUP:
  mp_int_clear(&tmp);
  return res;
}

mp_result mp_rat_sub_int(mp_rat a, mp_int b, mp_rat c)
{
  mpz_t tmp;
  mp_result res;

  if ((res = mp_int_init_copy(&tmp, b)) != MP_OK)
    return res;

  if ((res = mp_int_mul(&tmp, MP_DENOM_P(a), &tmp)) != MP_OK)
    goto CLEANUP;

  if ((res = mp_rat_copy(a, c)) != MP_OK)
    goto CLEANUP;

  if ((res = mp_int_sub(MP_NUMER_P(c), &tmp, MP_NUMER_P(c))) != MP_OK)
    goto CLEANUP;

  res = s_rat_reduce(c);

 CLEANUP:
  mp_int_clear(&tmp);
  return res;
}

mp_result mp_rat_mul_int(mp_rat a, mp_int b, mp_rat c)
{
  mp_result res;

  if ((res = mp_rat_copy(a, c)) != MP_OK)
    return res;

  if ((res = mp_int_mul(MP_NUMER_P(c), b, MP_NUMER_P(c))) != MP_OK)
    return res;

  return s_rat_reduce(c);
}

mp_result mp_rat_div_int(mp_rat a, mp_int b, mp_rat c)
{
  mp_result res;

  if (mp_int_compare_zero(b) == 0)
    return MP_UNDEF;

  if ((res = mp_rat_copy(a, c)) != MP_OK)
    return res;

  if ((res = mp_int_mul(MP_DENOM_P(c), b, MP_DENOM_P(c))) != MP_OK)
    return res;

  return s_rat_reduce(c);
}

mp_result mp_rat_expt(mp_rat a, mp_small b, mp_rat c)
{
  mp_result  res;

  /* Special cases for easy powers. */
  if (b == 0)
    return mp_rat_set_value(c, 1, 1);
  else if(b == 1)
    return mp_rat_copy(a, c);

  /* Since rationals are always stored in lowest terms, it is not necessary to
     reduce again when raising to an integer power. */
  if ((res = mp_int_expt(MP_NUMER_P(a), b, MP_NUMER_P(c))) != MP_OK)
    return res;

  return mp_int_expt(MP_DENOM_P(a), b, MP_DENOM_P(c));
}

int       mp_rat_compare(mp_rat a, mp_rat b)
{
  /* Quick check for opposite signs.  Works because the sign of the numerator
     is always definitive. */
  if (MP_SIGN(MP_NUMER_P(a)) != MP_SIGN(MP_NUMER_P(b))) {
    if (MP_SIGN(MP_NUMER_P(a)) == MP_ZPOS)
      return 1;
    else
      return -1;
  }
  else {
    /* Compare absolute magnitudes; if both are positive, the answer stands,
       otherwise it needs to be reflected about zero. */
    int cmp = mp_rat_compare_unsigned(a, b);

    if (MP_SIGN(MP_NUMER_P(a)) == MP_ZPOS)
      return cmp;
    else
      return -cmp;
  }
}

int       mp_rat_compare_unsigned(mp_rat a, mp_rat b)
{
  /* If the denominators are equal, we can quickly compare numerators without
     multiplying.  Otherwise, we actually have to do some work. */
  if (mp_int_compare_unsigned(MP_DENOM_P(a), MP_DENOM_P(b)) == 0)
    return mp_int_compare_unsigned(MP_NUMER_P(a), MP_NUMER_P(b));

  else {
    mpz_t  temp[2];
    mp_result res;
    int  cmp = INT_MAX, last = 0;

    /* t0 = num(a) * den(b), t1 = num(b) * den(a) */
    SETUP(mp_int_init_copy(TEMP(last), MP_NUMER_P(a)), last);
    SETUP(mp_int_init_copy(TEMP(last), MP_NUMER_P(b)), last);

    if ((res = mp_int_mul(TEMP(0), MP_DENOM_P(b), TEMP(0))) != MP_OK ||
	(res = mp_int_mul(TEMP(1), MP_DENOM_P(a), TEMP(1))) != MP_OK)
      goto CLEANUP;
    
    cmp = mp_int_compare_unsigned(TEMP(0), TEMP(1));
    
  CLEANUP:
    while (--last >= 0)
      mp_int_clear(TEMP(last));

    return cmp;
  }
}

int       mp_rat_compare_zero(mp_rat r)
{
  return mp_int_compare_zero(MP_NUMER_P(r));
}

int       mp_rat_compare_value(mp_rat r, mp_small n, mp_small d)
{
  mpq_t tmp;
  mp_result res;
  int  out = INT_MAX;

  if ((res = mp_rat_init(&tmp)) != MP_OK)
    return out;
  if ((res = mp_rat_set_value(&tmp, n, d)) != MP_OK)
    goto CLEANUP;
  
  out = mp_rat_compare(r, &tmp);
  
 CLEANUP:
  mp_rat_clear(&tmp);
  return out;
}

int       mp_rat_is_integer(mp_rat r)
{
  return (mp_int_compare_value(MP_DENOM_P(r), 1) == 0);
}

mp_result mp_rat_to_ints(mp_rat r, mp_small *num, mp_small *den)
{
  mp_result res;

  if ((res = mp_int_to_int(MP_NUMER_P(r), num)) != MP_OK)
    return res;

  res = mp_int_to_int(MP_DENOM_P(r), den);
  return res;
}

mp_result mp_rat_to_string(mp_rat r, mp_size radix, char *str, int limit)
{
  char *start;
  int   len;
  mp_result res;

  /* Write the numerator.  The sign of the rational number is written by the
     underlying integer implementation. */
  if ((res = mp_int_to_string(MP_NUMER_P(r), radix, str, limit)) != MP_OK)
    return res;

  /* If the value is zero, don't bother writing any denominator */
  if (mp_int_compare_zero(MP_NUMER_P(r)) == 0)
    return MP_OK;
  
  /* Locate the end of the numerator, and make sure we are not going to exceed
     the limit by writing a slash. */
  len = strlen(str);
  start = str + len;
  limit -= len;
  if(limit == 0)
    return MP_TRUNC;

  *start++ = '/';
  limit -= 1;
  
  res = mp_int_to_string(MP_DENOM_P(r), radix, start, limit);
  return res;
}

mp_result mp_rat_to_decimal(mp_rat r, mp_size radix, mp_size prec,
                            mp_round_mode round, char *str, int limit)
{
  mpz_t temp[3];
  mp_result res;
  char *start = str;
  int len, lead_0, left = limit, last = 0;
    
  SETUP(mp_int_init_copy(TEMP(last), MP_NUMER_P(r)), last);
  SETUP(mp_int_init(TEMP(last)), last);
  SETUP(mp_int_init(TEMP(last)), last);

  /* Get the unsigned integer part by dividing denominator into the absolute
     value of the numerator. */
  mp_int_abs(TEMP(0), TEMP(0));
  if ((res = mp_int_div(TEMP(0), MP_DENOM_P(r), TEMP(0), TEMP(1))) != MP_OK)
    goto CLEANUP;

  /* Now:  T0 = integer portion, unsigned;
           T1 = remainder, from which fractional part is computed. */

  /* Count up leading zeroes after the radix point. */
  for (lead_0 = 0; lead_0 < prec && mp_int_compare(TEMP(1), MP_DENOM_P(r)) < 0; 
      ++lead_0) {
    if ((res = mp_int_mul_value(TEMP(1), radix, TEMP(1))) != MP_OK)
      goto CLEANUP;
  }

  /* Multiply remainder by a power of the radix sufficient to get the right
     number of significant figures. */
  if (prec > lead_0) {
    if ((res = mp_int_expt_value(radix, prec - lead_0, TEMP(2))) != MP_OK)
      goto CLEANUP;
    if ((res = mp_int_mul(TEMP(1), TEMP(2), TEMP(1))) != MP_OK)
      goto CLEANUP;
  }
  if ((res = mp_int_div(TEMP(1), MP_DENOM_P(r), TEMP(1), TEMP(2))) != MP_OK)
    goto CLEANUP;

  /* Now:  T1 = significant digits of fractional part;
           T2 = leftovers, to use for rounding. 

     At this point, what we do depends on the rounding mode.  The default is
     MP_ROUND_DOWN, for which everything is as it should be already.
  */
  switch (round) {
    int cmp;

  case MP_ROUND_UP:
    if (mp_int_compare_zero(TEMP(2)) != 0) {
      if (prec == 0)
	res = mp_int_add_value(TEMP(0), 1, TEMP(0));
      else
	res = mp_int_add_value(TEMP(1), 1, TEMP(1));
    }
    break;

  case MP_ROUND_HALF_UP:
  case MP_ROUND_HALF_DOWN:
    if ((res = mp_int_mul_pow2(TEMP(2), 1, TEMP(2))) != MP_OK)
      goto CLEANUP;

    cmp = mp_int_compare(TEMP(2), MP_DENOM_P(r));    

    if (round == MP_ROUND_HALF_UP)
      cmp += 1;

    if (cmp > 0) {
      if (prec == 0)
	res = mp_int_add_value(TEMP(0), 1, TEMP(0));
      else
	res = mp_int_add_value(TEMP(1), 1, TEMP(1));
    }
    break;
    
  case MP_ROUND_DOWN:
    break;  /* No action required */

  default: 
    return MP_BADARG; /* Invalid rounding specifier */
  }

  /* The sign of the output should be the sign of the numerator, but if all the
     displayed digits will be zero due to the precision, a negative shouldn't
     be shown. */
  if (MP_SIGN(MP_NUMER_P(r)) == MP_NEG &&
      (mp_int_compare_zero(TEMP(0)) != 0 ||
       mp_int_compare_zero(TEMP(1)) != 0)) {
    *start++ = '-';
    left -= 1;
  }

  if ((res = mp_int_to_string(TEMP(0), radix, start, left)) != MP_OK)
    goto CLEANUP;
  
  len = strlen(start);
  start += len;
  left -= len;
  
  if (prec == 0) 
    goto CLEANUP;
  
  *start++ = '.';
  left -= 1;
  
  if (left < prec + 1) {
    res = MP_TRUNC;
    goto CLEANUP;
  }

  memset(start, '0', lead_0 - 1);
  left -= lead_0;
  start += lead_0 - 1;

  res = mp_int_to_string(TEMP(1), radix, start, left);

 CLEANUP:
  while (--last >= 0)
    mp_int_clear(TEMP(last));
  
  return res;
}

mp_result mp_rat_string_len(mp_rat r, mp_size radix)
{
  mp_result n_len, d_len = 0;

  n_len = mp_int_string_len(MP_NUMER_P(r), radix);

  if (mp_int_compare_zero(MP_NUMER_P(r)) != 0)
    d_len = mp_int_string_len(MP_DENOM_P(r), radix);

  /* Though simplistic, this formula is correct.  Space for the sign flag is
     included in n_len, and the space for the NUL that is counted in n_len
     counts for the separator here.  The space for the NUL counted in d_len
     counts for the final terminator here. */

  return n_len + d_len;

}

mp_result mp_rat_decimal_len(mp_rat r, mp_size radix, mp_size prec)
{
  int  z_len, f_len;

  z_len = mp_int_string_len(MP_NUMER_P(r), radix);
  
  if (prec == 0)
    f_len = 1; /* terminator only */
  else
    f_len = 1 + prec + 1; /* decimal point, digits, terminator */
  
  return z_len + f_len;
}

mp_result mp_rat_read_string(mp_rat r, mp_size radix, const char *str)
{
  return mp_rat_read_cstring(r, radix, str, NULL);
}

mp_result mp_rat_read_cstring(mp_rat r, mp_size radix, const char *str, 
			      char **end)
{
  mp_result res;
  char *endp;

  if ((res = mp_int_read_cstring(MP_NUMER_P(r), radix, str, &endp)) != MP_OK &&
      (res != MP_TRUNC))
    return res;

  /* Skip whitespace between numerator and (possible) separator */
  while (isspace((unsigned char) *endp))
    ++endp;
  
  /* If there is no separator, we will stop reading at this point. */
  if (*endp != '/') {
    mp_int_set_value(MP_DENOM_P(r), 1);
    if (end != NULL)
      *end = endp;
    return res;
  }
  
  ++endp; /* skip separator */
  if ((res = mp_int_read_cstring(MP_DENOM_P(r), radix, endp, end)) != MP_OK)
    return res;
  
  /* Make sure the value is well-defined */
  if (mp_int_compare_zero(MP_DENOM_P(r)) == 0)
    return MP_UNDEF;

  /* Reduce to lowest terms */
  return s_rat_reduce(r);
}

/* Read a string and figure out what format it's in.  The radix may be supplied
   as zero to use "default" behaviour.

   This function will accept either a/b notation or decimal notation.
 */
mp_result mp_rat_read_ustring(mp_rat r, mp_size radix, const char *str, 
			      char **end)
{
  char      *endp;
  mp_result  res;

  if (radix == 0)
    radix = 10;  /* default to decimal input */

  if ((res = mp_rat_read_cstring(r, radix, str, &endp)) != MP_OK) {
    if (res == MP_TRUNC) {
      if (*endp == '.')
	res = mp_rat_read_cdecimal(r, radix, str, &endp);
    }
    else
      return res;
  }

  if (end != NULL)
    *end = endp;

  return res;
}

mp_result mp_rat_read_decimal(mp_rat r, mp_size radix, const char *str)
{
  return mp_rat_read_cdecimal(r, radix, str, NULL);
}

mp_result mp_rat_read_cdecimal(mp_rat r, mp_size radix, const char *str, 
			       char **end)
{
  mp_result res;
  mp_sign   osign;
  char *endp;

  while (isspace((unsigned char) *str))
    ++str;
  
  switch (*str) {
  case '-':
    osign = MP_NEG;
    break;
  default:
    osign = MP_ZPOS;
  }
  
  if ((res = mp_int_read_cstring(MP_NUMER_P(r), radix, str, &endp)) != MP_OK &&
     (res != MP_TRUNC))
    return res;

  /* This needs to be here. */
  (void) mp_int_set_value(MP_DENOM_P(r), 1);

  if (*endp != '.') {
    if (end != NULL)
      *end = endp;
    return res;
  }

  /* If the character following the decimal point is whitespace or a sign flag,
     we will consider this a truncated value.  This special case is because
     mp_int_read_string() will consider whitespace or sign flags to be valid
     starting characters for a value, and we do not want them following the
     decimal point.

     Once we have done this check, it is safe to read in the value of the
     fractional piece as a regular old integer.
  */
  ++endp;
  if (*endp == '\0') {
    if (end != NULL)
      *end = endp;
    return MP_OK;
  }
  else if(isspace((unsigned char) *endp) || *endp == '-' || *endp == '+') {
    return MP_TRUNC;
  }
  else {
    mpz_t  frac;
    mp_result save_res;
    char  *save = endp;
    int    num_lz = 0;

    /* Make a temporary to hold the part after the decimal point. */
    if ((res = mp_int_init(&frac)) != MP_OK)
      return res;
    
    if ((res = mp_int_read_cstring(&frac, radix, endp, &endp)) != MP_OK &&
       (res != MP_TRUNC))
      goto CLEANUP;

    /* Save this response for later. */
    save_res = res;

    if (mp_int_compare_zero(&frac) == 0)
      goto FINISHED;

    /* Discard trailing zeroes (somewhat inefficiently) */
    while (mp_int_divisible_value(&frac, radix))
      if ((res = mp_int_div_value(&frac, radix, &frac, NULL)) != MP_OK)
	goto CLEANUP;
    
    /* Count leading zeros after the decimal point */
    while (save[num_lz] == '0')
      ++num_lz;

    /* Find the least power of the radix that is at least as large as the
       significant value of the fractional part, ignoring leading zeroes.  */
    (void) mp_int_set_value(MP_DENOM_P(r), radix); 
    
    while (mp_int_compare(MP_DENOM_P(r), &frac) < 0) {
      if ((res = mp_int_mul_value(MP_DENOM_P(r), radix, MP_DENOM_P(r))) != MP_OK)
	goto CLEANUP;
    }
    
    /* Also shift by enough to account for leading zeroes */
    while (num_lz > 0) {
      if ((res = mp_int_mul_value(MP_DENOM_P(r), radix, MP_DENOM_P(r))) != MP_OK)
	goto CLEANUP;

      --num_lz;
    }

    /* Having found this power, shift the numerator leftward that many, digits,
       and add the nonzero significant digits of the fractional part to get the
       result. */
    if ((res = mp_int_mul(MP_NUMER_P(r), MP_DENOM_P(r), MP_NUMER_P(r))) != MP_OK)
      goto CLEANUP;
    
    { /* This addition needs to be unsigned. */
      MP_SIGN(MP_NUMER_P(r)) = MP_ZPOS;
      if ((res = mp_int_add(MP_NUMER_P(r), &frac, MP_NUMER_P(r))) != MP_OK)
	goto CLEANUP;

      MP_SIGN(MP_NUMER_P(r)) = osign;
    }
    if ((res = s_rat_reduce(r)) != MP_OK)
      goto CLEANUP;

    /* At this point, what we return depends on whether reading the fractional
       part was truncated or not.  That information is saved from when we
       called mp_int_read_string() above. */
  FINISHED:
    res = save_res;
    if (end != NULL)
      *end = endp;

  CLEANUP:
    mp_int_clear(&frac);

    return res;
  }
}

/* Private functions for internal use.  Make unchecked assumptions about format
   and validity of inputs. */

static mp_result s_rat_reduce(mp_rat r)
{
  mpz_t gcd;
  mp_result res = MP_OK;

  if (mp_int_compare_zero(MP_NUMER_P(r)) == 0) {
    mp_int_set_value(MP_DENOM_P(r), 1);
    return MP_OK;
  }

  /* If the greatest common divisor of the numerator and denominator is greater
     than 1, divide it out. */
  if ((res = mp_int_init(&gcd)) != MP_OK)
    return res;

  if ((res = mp_int_gcd(MP_NUMER_P(r), MP_DENOM_P(r), &gcd)) != MP_OK)
    goto CLEANUP;

  if (mp_int_compare_value(&gcd, 1) != 0) {
    if ((res = mp_int_div(MP_NUMER_P(r), &gcd, MP_NUMER_P(r), NULL)) != MP_OK)
      goto CLEANUP;
    if ((res = mp_int_div(MP_DENOM_P(r), &gcd, MP_DENOM_P(r), NULL)) != MP_OK)
      goto CLEANUP;
  }

  /* Fix up the signs of numerator and denominator */
  if (MP_SIGN(MP_NUMER_P(r)) == MP_SIGN(MP_DENOM_P(r)))
    MP_SIGN(MP_NUMER_P(r)) = MP_SIGN(MP_DENOM_P(r)) = MP_ZPOS;
  else {
    MP_SIGN(MP_NUMER_P(r)) = MP_NEG;
    MP_SIGN(MP_DENOM_P(r)) = MP_ZPOS;
  }

 CLEANUP:
  mp_int_clear(&gcd);

  return res;
}

static mp_result s_rat_combine(mp_rat a, mp_rat b, mp_rat c, 
			       mp_result (*comb_f)(mp_int, mp_int, mp_int))
{
  mp_result res;

  /* Shortcut when denominators are already common */
  if (mp_int_compare(MP_DENOM_P(a), MP_DENOM_P(b)) == 0) {
    if ((res = (comb_f)(MP_NUMER_P(a), MP_NUMER_P(b), MP_NUMER_P(c))) != MP_OK)
      return res;
    if ((res = mp_int_copy(MP_DENOM_P(a), MP_DENOM_P(c))) != MP_OK)
      return res;
    
    return s_rat_reduce(c);
  }
  else {
    mpz_t  temp[2];
    int    last = 0;

    SETUP(mp_int_init_copy(TEMP(last), MP_NUMER_P(a)), last);
    SETUP(mp_int_init_copy(TEMP(last), MP_NUMER_P(b)), last);
    
    if ((res = mp_int_mul(TEMP(0), MP_DENOM_P(b), TEMP(0))) != MP_OK)
      goto CLEANUP;
    if ((res = mp_int_mul(TEMP(1), MP_DENOM_P(a), TEMP(1))) != MP_OK)
      goto CLEANUP;
    if ((res = (comb_f)(TEMP(0), TEMP(1), MP_NUMER_P(c))) != MP_OK)
      goto CLEANUP;

    res = mp_int_mul(MP_DENOM_P(a), MP_DENOM_P(b), MP_DENOM_P(c));

  CLEANUP:
    while (--last >= 0) 
      mp_int_clear(TEMP(last));

    if (res == MP_OK)
      return s_rat_reduce(c);
    else
      return res;
  }
}

/* Here there be dragons */
