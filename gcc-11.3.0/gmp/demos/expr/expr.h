/* Header for expression evaluation.

Copyright 2000-2002, 2004 Free Software Foundation, Inc.

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


#ifndef __EXPR_H__
#define __EXPR_H__

#define MPEXPR_RESULT_OK            0
#define MPEXPR_RESULT_BAD_VARIABLE  1
#define MPEXPR_RESULT_BAD_TABLE     2
#define MPEXPR_RESULT_PARSE_ERROR   3
#define MPEXPR_RESULT_NOT_UI        4


/* basic types */
#define MPEXPR_TYPE_NARY(n)       ((n) * 0x0100)
#define MPEXPR_TYPE_MASK_ARGCOUNT MPEXPR_TYPE_NARY(0xF)
#define MPEXPR_TYPE_0ARY          MPEXPR_TYPE_NARY(0)
#define MPEXPR_TYPE_UNARY         MPEXPR_TYPE_NARY(1)
#define MPEXPR_TYPE_BINARY        MPEXPR_TYPE_NARY(2)
#define MPEXPR_TYPE_TERNARY       MPEXPR_TYPE_NARY(3)

/* options for all */
#define MPEXPR_TYPE_LAST_UI       0x0010
#define MPEXPR_TYPE_RESULT_INT    0x0020
#define MPEXPR_TYPE_MASK_ARGSTYLE 0x0030

#define MPEXPR_TYPE_UNARY_UI     (MPEXPR_TYPE_UNARY   | MPEXPR_TYPE_LAST_UI)
#define MPEXPR_TYPE_I_UNARY      (MPEXPR_TYPE_UNARY   | MPEXPR_TYPE_RESULT_INT)
#define MPEXPR_TYPE_I_UNARY_UI   (MPEXPR_TYPE_I_UNARY | MPEXPR_TYPE_LAST_UI)
#define MPEXPR_TYPE_BINARY_UI    (MPEXPR_TYPE_BINARY  | MPEXPR_TYPE_LAST_UI)
#define MPEXPR_TYPE_I_BINARY     (MPEXPR_TYPE_BINARY  | MPEXPR_TYPE_RESULT_INT)
#define MPEXPR_TYPE_I_BINARY_UI  (MPEXPR_TYPE_I_BINARY| MPEXPR_TYPE_LAST_UI)
#define MPEXPR_TYPE_TERNARY_UI   (MPEXPR_TYPE_TERNARY | MPEXPR_TYPE_LAST_UI)
#define MPEXPR_TYPE_I_TERNARY    (MPEXPR_TYPE_TERNARY | MPEXPR_TYPE_RESULT_INT)
#define MPEXPR_TYPE_I_TERNARY_UI (MPEXPR_TYPE_I_TERNARY|MPEXPR_TYPE_LAST_UI)

/* 0ary with options */
#define MPEXPR_TYPE_CONSTANT      (MPEXPR_TYPE_0ARY | 0x0040)

/* unary options */
#define MPEXPR_TYPE_PREFIX        0x0040

/* binary options */
#define MPEXPR_TYPE_RIGHTASSOC    0x0040
#define MPEXPR_TYPE_PAIRWISE      0x0080

#define MPEXPR_TYPE_MASK_SPECIAL  0x000F

/* unary specials */
#define MPEXPR_TYPE_NEW_TABLE     (MPEXPR_TYPE_UNARY | 0x001)
#define MPEXPR_TYPE_DONE          (MPEXPR_TYPE_UNARY | 0x002)
#define MPEXPR_TYPE_VARIABLE      (MPEXPR_TYPE_UNARY | 0x003)
#define MPEXPR_TYPE_LOGICAL_NOT   (MPEXPR_TYPE_UNARY | 0x004)
#define MPEXPR_TYPE_CLOSEPAREN    (MPEXPR_TYPE_UNARY | 0x005)
#define MPEXPR_TYPE_OPENPAREN     (MPEXPR_TYPE_CLOSEPAREN | MPEXPR_TYPE_PREFIX)

/* binary specials */
#define MPEXPR_TYPE_LOGICAL_AND   (MPEXPR_TYPE_BINARY | 0x001)
#define MPEXPR_TYPE_LOGICAL_OR    (MPEXPR_TYPE_BINARY | 0x002)
#define MPEXPR_TYPE_ARGSEP        (MPEXPR_TYPE_BINARY | 0x003)
#define MPEXPR_TYPE_QUESTION      (MPEXPR_TYPE_BINARY | 0x004)
#define MPEXPR_TYPE_COLON         (MPEXPR_TYPE_BINARY | 0x005)
#define MPEXPR_TYPE_MAX           (MPEXPR_TYPE_BINARY | 0x006)
#define MPEXPR_TYPE_MIN           (MPEXPR_TYPE_BINARY | 0x007)
#define MPEXPR_TYPE_MASK_CMP      0x008
#define MPEXPR_TYPE_MASK_CMP_LT   0x001
#define MPEXPR_TYPE_MASK_CMP_EQ   0x002
#define MPEXPR_TYPE_MASK_CMP_GT   0x004
#define MPEXPR_TYPE_CMP_LT       (MPEXPR_TYPE_BINARY | MPEXPR_TYPE_MASK_CMP \
				  | MPEXPR_TYPE_MASK_CMP_LT)
#define MPEXPR_TYPE_CMP_EQ       (MPEXPR_TYPE_BINARY | MPEXPR_TYPE_MASK_CMP \
				  | MPEXPR_TYPE_MASK_CMP_EQ)
#define MPEXPR_TYPE_CMP_GT       (MPEXPR_TYPE_BINARY | MPEXPR_TYPE_MASK_CMP \
				  | MPEXPR_TYPE_MASK_CMP_GT)
#define MPEXPR_TYPE_CMP_LE       (MPEXPR_TYPE_CMP_LT | MPEXPR_TYPE_MASK_CMP_EQ)
#define MPEXPR_TYPE_CMP_NE       (MPEXPR_TYPE_CMP_LT | MPEXPR_TYPE_MASK_CMP_GT)
#define MPEXPR_TYPE_CMP_GE       (MPEXPR_TYPE_CMP_GT | MPEXPR_TYPE_MASK_CMP_EQ)

/* parse options */
#define MPEXPR_TYPE_WHOLEWORD      0x1000
#define MPEXPR_TYPE_OPERATOR       0x2000


#ifdef __cplusplus
extern "C" {
#endif

typedef void (*mpexpr_fun_t) (void);

struct mpexpr_operator_t {
  const char   *name;
  mpexpr_fun_t fun;
  int          type;
  int          precedence;
};


int mpf_expr_a (const struct mpexpr_operator_t *, mpf_ptr, int,
		unsigned long, const char *, size_t, mpf_srcptr [26]);
int mpf_expr (mpf_ptr, int, const char *, ...);

int mpq_expr_a (const struct mpexpr_operator_t *, mpq_ptr,
		int, const char *, size_t, mpq_srcptr [26]);
int mpq_expr (mpq_ptr, int, const char *, ...);

int mpz_expr_a (const struct mpexpr_operator_t *, mpz_ptr, int,
		const char *, size_t, mpz_srcptr [26]);
int mpz_expr (mpz_ptr, int, const char *, ...);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
