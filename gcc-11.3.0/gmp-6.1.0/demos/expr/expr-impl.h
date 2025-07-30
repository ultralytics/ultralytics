/* Implementation specifics for expression evaluation.

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

#include <stdarg.h>

#include "expr.h"


#define isasciidigit(c)   (isascii (c) && isdigit (c))
#define isasciicsym(c)    (isascii (c) && (isalnum(c) || (c) == '_'))

#define isasciidigit_in_base(c,base)                    \
  (isascii (c)                                          \
   && ((isdigit (c) && (c)-'0' < (base))                \
       || (isupper (c) && (c)-'A'+10 < (base))          \
       || (islower (c) && (c)-'a'+10 < (base))))


union mpX_t {
  mpz_t   z;
  mpq_t   q;
  mpf_t   f;
};

typedef union mpX_t *mpX_ptr;
typedef const union mpX_t *mpX_srcptr;

typedef void (*mpexpr_fun_one_t) (mpX_ptr);
typedef unsigned long (*mpexpr_fun_ui_one_t) (mpX_ptr);

typedef void (*mpexpr_fun_0ary_t) (mpX_ptr);
typedef int  (*mpexpr_fun_i_0ary_t) (void);

typedef void (*mpexpr_fun_unary_t) (mpX_ptr, mpX_srcptr);
typedef void (*mpexpr_fun_unary_ui_t) (mpX_ptr, unsigned long);
typedef int  (*mpexpr_fun_i_unary_t) (mpX_srcptr);
typedef int  (*mpexpr_fun_i_unary_ui_t) (unsigned long);

typedef void (*mpexpr_fun_binary_t) (mpX_ptr, mpX_srcptr, mpX_srcptr);
typedef void (*mpexpr_fun_binary_ui_t) (mpX_ptr, mpX_srcptr, unsigned long);
typedef int  (*mpexpr_fun_i_binary_t) (mpX_srcptr, mpX_srcptr);
typedef int  (*mpexpr_fun_i_binary_ui_t) (mpX_srcptr, unsigned long);

typedef void (*mpexpr_fun_ternary_t) (mpX_ptr, mpX_srcptr, mpX_srcptr, mpX_srcptr);
typedef void (*mpexpr_fun_ternary_ui_t) (mpX_ptr, mpX_srcptr, mpX_srcptr, unsigned long);
typedef int (*mpexpr_fun_i_ternary_t) (mpX_srcptr, mpX_srcptr, mpX_srcptr);
typedef int (*mpexpr_fun_i_ternary_ui_t) (mpX_srcptr, mpX_srcptr, unsigned long);

typedef size_t (*mpexpr_fun_number_t) (mpX_ptr, const char *str, size_t len, int base);
typedef void (*mpexpr_fun_swap_t) (mpX_ptr, mpX_ptr);
typedef unsigned long (*mpexpr_fun_get_ui_t) (mpX_srcptr);
typedef void (*mpexpr_fun_set_si_t) (mpX_srcptr, long);

struct mpexpr_control_t {
  const struct mpexpr_operator_t  *op;
  int                             argcount;
};

#define MPEXPR_VARIABLES  26

struct mpexpr_parse_t {
  const struct mpexpr_operator_t  *table;

  mpX_ptr                         res;
  int                             base;
  unsigned long                   prec;
  const char                      *e;
  size_t                          elen;
  mpX_srcptr                      *var;
  int                             error_code;

  int                             token;
  const struct mpexpr_operator_t  *token_op;

  union mpX_t                     *data_stack;
  int                             data_top;
  int                             data_alloc;
  int                             data_inited;

  struct mpexpr_control_t         *control_stack;
  int                             control_top;
  int                             control_alloc;

  mpexpr_fun_0ary_t               mpX_clear;
  mpexpr_fun_i_unary_t            mpX_ulong_p;
  mpexpr_fun_get_ui_t             mpX_get_ui;
  mpexpr_fun_unary_ui_t           mpX_init;
  mpexpr_fun_number_t             mpX_number;
  mpexpr_fun_unary_t              mpX_set;
  mpexpr_fun_unary_t              mpX_set_or_swap;
  mpexpr_fun_set_si_t             mpX_set_si;
  mpexpr_fun_swap_t               mpX_swap;
};


int mpexpr_evaluate (struct mpexpr_parse_t *p);
int mpexpr_va_to_var (void *var[], va_list ap);
size_t mpexpr_mpz_number (mpz_ptr res, const char *e, size_t elen, int base);
