/*
  Name:     imath.h
  Purpose:  Arbitrary precision integer arithmetic routines.
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

#ifndef IMATH_H_
#define IMATH_H_

#include <stdint.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char      mp_sign;
typedef unsigned int       mp_size;
typedef int                mp_result;
typedef long               mp_small;  /* must be a signed type */
typedef unsigned long      mp_usmall; /* must be an unsigned type */

/* Force building with uint64_t so that the library builds consistently
 * whether we build from the makefile or by embedding imath in another project.
 */
#undef  USE_64BIT_WORDS
#define USE_64BIT_WORDS
#ifdef  USE_64BIT_WORDS
typedef uint32_t           mp_digit;
typedef uint64_t           mp_word;
#else
typedef uint16_t           mp_digit;
typedef uint32_t           mp_word;
#endif

typedef struct mpz {
  mp_digit    single;
  mp_digit   *digits;
  mp_size     alloc;
  mp_size     used;
  mp_sign     sign;
} mpz_t, *mp_int;

#define MP_DIGITS(Z) ((Z)->digits)
#define MP_ALLOC(Z)  ((Z)->alloc)
#define MP_USED(Z)   ((Z)->used)
#define MP_SIGN(Z)   ((Z)->sign)

extern const mp_result MP_OK;
extern const mp_result MP_FALSE;
extern const mp_result MP_TRUE;
extern const mp_result MP_MEMORY;
extern const mp_result MP_RANGE;
extern const mp_result MP_UNDEF;
extern const mp_result MP_TRUNC;
extern const mp_result MP_BADARG;
extern const mp_result MP_MINERR;

#define MP_DIGIT_BIT    (sizeof(mp_digit) * CHAR_BIT)
#define MP_WORD_BIT     (sizeof(mp_word) * CHAR_BIT)
#define MP_SMALL_MIN    LONG_MIN
#define MP_SMALL_MAX    LONG_MAX
#define MP_USMALL_MIN   ULONG_MIN
#define MP_USMALL_MAX   ULONG_MAX

#ifdef USE_64BIT_WORDS
#  define MP_DIGIT_MAX   (UINT32_MAX * UINT64_C(1))
#  define MP_WORD_MAX    (UINT64_MAX)
#else
#  define MP_DIGIT_MAX   (UINT16_MAX * 1UL)
#  define MP_WORD_MAX    (UINT32_MAX * 1UL)
#endif

#define MP_MIN_RADIX    2
#define MP_MAX_RADIX    36

/* Values with fewer than this many significant digits use the standard
   multiplication algorithm; otherwise, a recursive algorithm is used.  
   Choose a value to suit your platform.
 */
#define MP_MULT_THRESH  22

#define MP_DEFAULT_PREC 8   /* default memory allocation, in digits */

extern const mp_sign   MP_NEG;
extern const mp_sign   MP_ZPOS;

#define mp_int_is_odd(Z)  ((Z)->digits[0] & 1)
#define mp_int_is_even(Z) !((Z)->digits[0] & 1)

mp_result mp_int_init(mp_int z);
mp_int    mp_int_alloc(void);
mp_result mp_int_init_size(mp_int z, mp_size prec);
mp_result mp_int_init_copy(mp_int z, mp_int old);
mp_result mp_int_init_value(mp_int z, mp_small value);
mp_result mp_int_init_uvalue(mp_int z, mp_usmall uvalue);
mp_result mp_int_set_value(mp_int z, mp_small value);
mp_result mp_int_set_uvalue(mp_int z, mp_usmall uvalue);
void      mp_int_clear(mp_int z);
void      mp_int_free(mp_int z);

mp_result mp_int_copy(mp_int a, mp_int c);           /* c = a     */
void      mp_int_swap(mp_int a, mp_int c);           /* swap a, c */
void      mp_int_zero(mp_int z);                     /* z = 0     */
mp_result mp_int_abs(mp_int a, mp_int c);            /* c = |a|   */
mp_result mp_int_neg(mp_int a, mp_int c);            /* c = -a    */
mp_result mp_int_add(mp_int a, mp_int b, mp_int c);  /* c = a + b */
mp_result mp_int_add_value(mp_int a, mp_small value, mp_int c);
mp_result mp_int_sub(mp_int a, mp_int b, mp_int c);  /* c = a - b */
mp_result mp_int_sub_value(mp_int a, mp_small value, mp_int c);
mp_result mp_int_mul(mp_int a, mp_int b, mp_int c);  /* c = a * b */
mp_result mp_int_mul_value(mp_int a, mp_small value, mp_int c);
mp_result mp_int_mul_pow2(mp_int a, mp_small p2, mp_int c);
mp_result mp_int_sqr(mp_int a, mp_int c);            /* c = a * a */
mp_result mp_int_div(mp_int a, mp_int b,             /* q = a / b */
		     mp_int q, mp_int r);            /* r = a % b */
mp_result mp_int_div_value(mp_int a, mp_small value, /* q = a / value */
			   mp_int q, mp_small *r);   /* r = a % value */
mp_result mp_int_div_pow2(mp_int a, mp_small p2,     /* q = a / 2^p2  */
			  mp_int q, mp_int r);       /* r = q % 2^p2  */
mp_result mp_int_mod(mp_int a, mp_int m, mp_int c);  /* c = a % m */
#define   mp_int_mod_value(A, V, R) mp_int_div_value((A), (V), 0, (R))
mp_result mp_int_expt(mp_int a, mp_small b, mp_int c);         /* c = a^b */
mp_result mp_int_expt_value(mp_small a, mp_small b, mp_int c); /* c = a^b */
mp_result mp_int_expt_full(mp_int a, mp_int b, mp_int c);      /* c = a^b */

int       mp_int_compare(mp_int a, mp_int b);          /* a <=> b     */
int       mp_int_compare_unsigned(mp_int a, mp_int b); /* |a| <=> |b| */
int       mp_int_compare_zero(mp_int z);                 /* a <=> 0  */
int       mp_int_compare_value(mp_int z, mp_small v);    /* a <=> v  */
int       mp_int_compare_uvalue(mp_int z, mp_usmall uv); /* a <=> uv */

/* Returns true if v|a, false otherwise (including errors) */
int       mp_int_divisible_value(mp_int a, mp_small v);

/* Returns k >= 0 such that z = 2^k, if one exists; otherwise < 0 */
int       mp_int_is_pow2(mp_int z);

mp_result mp_int_exptmod(mp_int a, mp_int b, mp_int m,
			 mp_int c);                    /* c = a^b (mod m) */
mp_result mp_int_exptmod_evalue(mp_int a, mp_small value,
				mp_int m, mp_int c);   /* c = a^v (mod m) */
mp_result mp_int_exptmod_bvalue(mp_small value, mp_int b,
				mp_int m, mp_int c);   /* c = v^b (mod m) */
mp_result mp_int_exptmod_known(mp_int a, mp_int b,
			       mp_int m, mp_int mu,
			       mp_int c);              /* c = a^b (mod m) */
mp_result mp_int_redux_const(mp_int m, mp_int c);

mp_result mp_int_invmod(mp_int a, mp_int m, mp_int c); /* c = 1/a (mod m) */

mp_result mp_int_gcd(mp_int a, mp_int b, mp_int c);    /* c = gcd(a, b)   */

mp_result mp_int_egcd(mp_int a, mp_int b, mp_int c,    /* c = gcd(a, b)   */
		      mp_int x, mp_int y);             /* c = ax + by     */

mp_result mp_int_lcm(mp_int a, mp_int b, mp_int c);    /* c = lcm(a, b)   */

mp_result mp_int_root(mp_int a, mp_small b, mp_int c); /* c = floor(a^{1/b}) */
#define   mp_int_sqrt(a, c) mp_int_root(a, 2, c)       /* c = floor(sqrt(a)) */

/* Convert to a small int, if representable; else MP_RANGE */
mp_result mp_int_to_int(mp_int z, mp_small *out);
mp_result mp_int_to_uint(mp_int z, mp_usmall *out);

/* Convert to nul-terminated string with the specified radix, writing at
   most limit characters including the nul terminator  */
mp_result mp_int_to_string(mp_int z, mp_size radix,
			   char *str, int limit);

/* Return the number of characters required to represent
   z in the given radix.  May over-estimate. */
mp_result mp_int_string_len(mp_int z, mp_size radix);

/* Read zero-terminated string into z */
mp_result mp_int_read_string(mp_int z, mp_size radix, const char *str);
mp_result mp_int_read_cstring(mp_int z, mp_size radix, const char *str,
			      char **end);

/* Return the number of significant bits in z */
mp_result mp_int_count_bits(mp_int z);

/* Convert z to two's complement binary, writing at most limit bytes */
mp_result mp_int_to_binary(mp_int z, unsigned char *buf, int limit);

/* Read a two's complement binary value into z from the given buffer */
mp_result mp_int_read_binary(mp_int z, unsigned char *buf, int len);

/* Return the number of bytes required to represent z in binary. */
mp_result mp_int_binary_len(mp_int z);

/* Convert z to unsigned binary, writing at most limit bytes */
mp_result mp_int_to_unsigned(mp_int z, unsigned char *buf, int limit);

/* Read an unsigned binary value into z from the given buffer */
mp_result mp_int_read_unsigned(mp_int z, unsigned char *buf, int len);

/* Return the number of bytes required to represent z as unsigned output */
mp_result mp_int_unsigned_len(mp_int z);

/* Return a statically allocated string describing error code res */
const char *mp_error_string(mp_result res);

#if DEBUG
void      s_print(char *tag, mp_int z);
void      s_print_buf(char *tag, mp_digit *buf, mp_size num);
#endif

#ifdef __cplusplus
}
#endif
#endif /* end IMATH_H_ */
