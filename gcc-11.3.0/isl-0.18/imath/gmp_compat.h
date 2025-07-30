/*
  Name:     gmp_compat.h
  Purpose:  Provide GMP compatiable routines for imath library
  Author:   David Peixotto

  Copyright (c) 2012 Qualcomm Innovation Center, Inc. All rights reserved.

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

#ifndef IMATH_GMP_COMPAT_H_
#define IMATH_GMP_COMPAT_H_
#include "imath.h"
#include "imrat.h"
#include <stddef.h>

#define GMPZAPI(fun) impz_ ## fun
#define GMPQAPI(fun) impq_ ## fun

#ifdef __cplusplus
extern "C" {
#endif
/*************************************************************************
 *
 * Functions with direct translations
 *
 *************************************************************************/
/* gmp: mpq_clear */
void GMPQAPI(clear)(mp_rat x);

/* gmp: mpq_cmp */
int GMPQAPI(cmp)(mp_rat op1, mp_rat op2);

/* gmp: mpq_init */
void GMPQAPI(init)(mp_rat x);

/* gmp: mpq_mul */
void GMPQAPI(mul)(mp_rat product, mp_rat multiplier, mp_rat multiplicand);

/* gmp: mpq_set */
void GMPQAPI(set)(mp_rat rop, mp_rat op);

/* gmp: mpz_abs */
void GMPZAPI(abs)(mp_int rop, mp_int op);

/* gmp: mpz_add */
void GMPZAPI(add)(mp_int rop, mp_int op1, mp_int op2);

/* gmp: mpz_clear */
void GMPZAPI(clear)(mp_int x);

/* gmp: mpz_cmp_si */
int GMPZAPI(cmp_si)(mp_int op1, long op2);

/* gmp: mpz_cmpabs */
int GMPZAPI(cmpabs)(mp_int op1, mp_int op2);

/* gmp: mpz_cmp */
int GMPZAPI(cmp)(mp_int op1, mp_int op2);

/* gmp: mpz_init */
void GMPZAPI(init)(mp_int x);

/* gmp: mpz_mul */
void GMPZAPI(mul)(mp_int rop, mp_int op1, mp_int op2);

/* gmp: mpz_neg */
void GMPZAPI(neg)(mp_int rop, mp_int op);

/* gmp: mpz_set_si */
void GMPZAPI(set_si)(mp_int rop, long op);

/* gmp: mpz_set */
void GMPZAPI(set)(mp_int rop, mp_int op);

/* gmp: mpz_sub */
void GMPZAPI(sub)(mp_int rop, mp_int op1, mp_int op2);

/* gmp: mpz_swap */
void GMPZAPI(swap)(mp_int rop1, mp_int rop2);

/* gmp: mpq_sgn */
int GMPQAPI(sgn)(mp_rat op);

/* gmp: mpz_sgn */
int GMPZAPI(sgn)(mp_int op);

/* gmp: mpq_set_ui */
void GMPQAPI(set_ui)(mp_rat rop, unsigned long op1, unsigned long op2);

/* gmp: mpz_set_ui */
void GMPZAPI(set_ui)(mp_int rop, unsigned long op);

/* gmp: mpq_den_ref */
mp_int GMPQAPI(denref)(mp_rat op);

/* gmp: mpq_num_ref */
mp_int GMPQAPI(numref)(mp_rat op);

/* gmp: mpq_canonicalize */
void GMPQAPI(canonicalize)(mp_rat op);

/*************************************************************************
 *
 * Functions that can be implemented as a combination of imath functions
 *
 *************************************************************************/
/* gmp: mpz_addmul */
void GMPZAPI(addmul)(mp_int rop, mp_int op1, mp_int op2);

/* gmp: mpz_divexact */
void GMPZAPI(divexact)(mp_int q, mp_int n, mp_int d);

/* gmp: mpz_divisible_p */
int GMPZAPI(divisible_p)(mp_int n, mp_int d);

/* gmp: mpz_submul */
void GMPZAPI(submul)(mp_int rop, mp_int op1, mp_int op2);

/* gmp: mpz_add_ui */
void GMPZAPI(add_ui)(mp_int rop, mp_int op1, unsigned long op2);

/* gmp: mpz_divexact_ui */
void GMPZAPI(divexact_ui)(mp_int q, mp_int n, unsigned long d);

/* gmp: mpz_mul_ui */
void GMPZAPI(mul_ui)(mp_int rop, mp_int op1, unsigned long op2);

/* gmp: mpz_pow_ui */
void GMPZAPI(pow_ui)(mp_int rop, mp_int base, unsigned long exp);

/* gmp: mpz_sub_ui */
void GMPZAPI(sub_ui)(mp_int rop, mp_int op1, unsigned long op2);

/* gmp: mpz_fdiv_q_ui */
unsigned long GMPZAPI(fdiv_q_ui)(mp_int q, mp_int n, unsigned long d);

/* gmp: mpz_sizeinbase */
size_t GMPZAPI(sizeinbase)(mp_int op, int base);

/*************************************************************************
 *
 * Functions with different behavior in corner cases
 *
 *************************************************************************/
/* gmp: mpz_gcd */
/* gmp: When op1 = 0 and op2 = 0, return 0.*/
void GMPZAPI(gcd)(mp_int rop, mp_int op1, mp_int op2);

/* gmp: mpz_get_str */
/* gmp: If str is NULL then allocate space using the default allocator. */
char* GMPZAPI(get_str)(char *str, int radix, mp_int op);

/* gmp: mpq_get_str */
/* gmp: If str is NULL then allocate space using the default allocator. */
/* gmp: If value is a whole number do not print denomenator. */
/* TODO: Need to handle 0 values better. GMP prints 0/4 instead of 0.*/
char* GMPQAPI(get_str)(char *str, int radix, mp_rat op);

/* gmp: mpz_set_str */
/* gmp: Allow and ignore spaces in string. */
int GMPZAPI(set_str)(mp_int rop, char *str, int base);

/* gmp: mpq_set_str */
int GMPQAPI(set_str)(mp_rat rop, char *str, int base);

/* gmp: mpz_get_ui */
/* gmp: Return least significant bits if value is too big for a long. */
unsigned long GMPZAPI(get_ui)(mp_int op);

/* gmp: mpz_get_si */
/* gmp: Return least significant bits if value is too bit for a long. */
/* gmp: If value is too big for long, return the least significant
        (8*sizeof(long)-1) bits from the op and set the sign bit according to
        the sign of the op. */
long GMPZAPI(get_si)(mp_int op);

/* gmp: mpz_lcm */
/* gmp: When op1 = 0 or op2 = 0, return 0.*/
/* gmp: The resutl of lcm(a,b) is always positive. */
void GMPZAPI(lcm)(mp_int rop, mp_int op1, mp_int op2);

/* gmp: mpz_mul_2exp */
/* gmp: allow big values for op2 when op1 == 0 */
void GMPZAPI(mul_2exp)(mp_int rop, mp_int op1, unsigned long op2);

/*************************************************************************
 *
 * Functions needing expanded functionality
 *
 *************************************************************************/
/* gmp: mpz_cdiv_q */
void GMPZAPI(cdiv_q)(mp_int q, mp_int n, mp_int d);

/* gmp: mpz_fdiv_q */
void GMPZAPI(fdiv_q)(mp_int q, mp_int n, mp_int d);

/* gmp: mpz_fdiv_r */
void GMPZAPI(fdiv_r)(mp_int r, mp_int n, mp_int d);

/* gmp: mpz_tdiv_q */
void GMPZAPI(tdiv_q)(mp_int q, mp_int n, mp_int d);

/* gmp: mpz_export */
void* GMPZAPI(export)(void *rop, size_t *countp, int order, size_t size, int endian, size_t nails, mp_int op);

/* gmp: mpz_import */
void GMPZAPI(import)(mp_int rop, size_t count, int order, size_t size, int endian, size_t nails, const void* op);

#ifdef __cplusplus
}
#endif
#endif /* end IMATH_GMP_COMPAT_H_ */
