/*
  Name:     imrat.h
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

#ifndef IMRAT_H_
#define IMRAT_H_

#include "imath.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mpq {
  mpz_t   num;    /* Numerator         */
  mpz_t   den;    /* Denominator, <> 0 */
} mpq_t, *mp_rat;

#define MP_NUMER_P(Q)  (&((Q)->num)) /* Pointer to numerator   */
#define MP_DENOM_P(Q)  (&((Q)->den)) /* Pointer to denominator */

/* Rounding constants */
typedef enum {
  MP_ROUND_DOWN,
  MP_ROUND_HALF_UP,
  MP_ROUND_UP,
  MP_ROUND_HALF_DOWN
} mp_round_mode;

mp_result mp_rat_init(mp_rat r);
mp_rat    mp_rat_alloc(void);
mp_result mp_rat_reduce(mp_rat r);
mp_result mp_rat_init_size(mp_rat r, mp_size n_prec, mp_size d_prec);
mp_result mp_rat_init_copy(mp_rat r, mp_rat old);
mp_result mp_rat_set_value(mp_rat r, mp_small numer, mp_small denom);
mp_result mp_rat_set_uvalue(mp_rat r, mp_usmall numer, mp_usmall denom);
void      mp_rat_clear(mp_rat r);
void      mp_rat_free(mp_rat r);
mp_result mp_rat_numer(mp_rat r, mp_int z);             /* z = num(r)  */
mp_int    mp_rat_numer_ref(mp_rat r);                   /* &num(r)     */
mp_result mp_rat_denom(mp_rat r, mp_int z);             /* z = den(r)  */
mp_int    mp_rat_denom_ref(mp_rat r);                   /* &den(r)     */
mp_sign   mp_rat_sign(mp_rat r);

mp_result mp_rat_copy(mp_rat a, mp_rat c);              /* c = a       */
void      mp_rat_zero(mp_rat r);                        /* r = 0       */
mp_result mp_rat_abs(mp_rat a, mp_rat c);               /* c = |a|     */
mp_result mp_rat_neg(mp_rat a, mp_rat c);               /* c = -a      */
mp_result mp_rat_recip(mp_rat a, mp_rat c);             /* c = 1 / a   */
mp_result mp_rat_add(mp_rat a, mp_rat b, mp_rat c);     /* c = a + b   */
mp_result mp_rat_sub(mp_rat a, mp_rat b, mp_rat c);     /* c = a - b   */
mp_result mp_rat_mul(mp_rat a, mp_rat b, mp_rat c);     /* c = a * b   */
mp_result mp_rat_div(mp_rat a, mp_rat b, mp_rat c);     /* c = a / b   */

mp_result mp_rat_add_int(mp_rat a, mp_int b, mp_rat c); /* c = a + b   */
mp_result mp_rat_sub_int(mp_rat a, mp_int b, mp_rat c); /* c = a - b   */
mp_result mp_rat_mul_int(mp_rat a, mp_int b, mp_rat c); /* c = a * b   */
mp_result mp_rat_div_int(mp_rat a, mp_int b, mp_rat c); /* c = a / b   */
mp_result mp_rat_expt(mp_rat a, mp_small b, mp_rat c);  /* c = a ^ b   */

int       mp_rat_compare(mp_rat a, mp_rat b);           /* a <=> b     */
int       mp_rat_compare_unsigned(mp_rat a, mp_rat b);  /* |a| <=> |b| */
int       mp_rat_compare_zero(mp_rat r);                /* r <=> 0     */
int       mp_rat_compare_value(mp_rat r, mp_small n, mp_small d); /* r <=> n/d */
int       mp_rat_is_integer(mp_rat r);

/* Convert to integers, if representable (returns MP_RANGE if not). */
mp_result mp_rat_to_ints(mp_rat r, mp_small *num, mp_small *den);

/* Convert to nul-terminated string with the specified radix, writing
   at most limit characters including the nul terminator. */
mp_result mp_rat_to_string(mp_rat r, mp_size radix, char *str, int limit);

/* Convert to decimal format in the specified radix and precision,
   writing at most limit characters including a nul terminator. */
mp_result mp_rat_to_decimal(mp_rat r, mp_size radix, mp_size prec,
                            mp_round_mode round, char *str, int limit);

/* Return the number of characters required to represent r in the given
   radix.  May over-estimate. */
mp_result mp_rat_string_len(mp_rat r, mp_size radix);

/* Return the number of characters required to represent r in decimal
   format with the given radix and precision.  May over-estimate. */
mp_result mp_rat_decimal_len(mp_rat r, mp_size radix, mp_size prec);

/* Read zero-terminated string into r */
mp_result mp_rat_read_string(mp_rat r, mp_size radix, const char *str);
mp_result mp_rat_read_cstring(mp_rat r, mp_size radix, const char *str,
			      char **end);
mp_result mp_rat_read_ustring(mp_rat r, mp_size radix, const char *str,
			      char **end);

/* Read zero-terminated string in decimal format into r */
mp_result mp_rat_read_decimal(mp_rat r, mp_size radix, const char *str);
mp_result mp_rat_read_cdecimal(mp_rat r, mp_size radix, const char *str,
			       char **end);

#ifdef __cplusplus
}
#endif
#endif /* IMRAT_H_ */
