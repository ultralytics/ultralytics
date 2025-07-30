/* mpq expression evaluation

Copyright 2000-2002 Free Software Foundation, Inc.

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

#include <stdio.h>
#include <string.h>
#include "gmp.h"
#include "expr-impl.h"


/* Change this to "#define TRACE(x) x" to get some traces. */
#define TRACE(x)


static void
e_mpq_pow_ui (mpq_ptr r, mpq_srcptr b, unsigned long e)
{
  mpz_pow_ui (mpq_numref(r), mpq_numref(b), e);
  mpz_pow_ui (mpq_denref(r), mpq_denref(b), e);
}

/* Wrapped because mpq_sgn is a macro. */
static int
e_mpq_sgn (mpq_srcptr x)
{
  return mpq_sgn (x);
}

/* Wrapped because mpq_equal only guarantees a non-zero return, whereas we
   want 1 or 0 for == and !=. */
static int
e_mpq_equal (mpq_srcptr x, mpq_srcptr y)
{
  return mpq_equal (x, y) != 0;
}
static int
e_mpq_notequal (mpq_srcptr x, mpq_srcptr y)
{
  return ! mpq_equal (x, y);
}

static void
e_mpq_num (mpq_ptr w, mpq_srcptr x)
{
  if (w != x)
    mpz_set (mpq_numref(w), mpq_numref(x));
  mpz_set_ui (mpq_denref(w), 1L);
}
static void
e_mpq_den (mpq_ptr w, mpq_srcptr x)
{
  if (w == x)
    mpz_swap (mpq_numref(w), mpq_denref(w));
  else
    mpz_set (mpq_numref(w), mpq_denref(x));
  mpz_set_ui (mpq_denref(w), 1L);
}


static const struct mpexpr_operator_t  _mpq_expr_standard_table[] = {

  { "**",  (mpexpr_fun_t) e_mpq_pow_ui,
    MPEXPR_TYPE_BINARY_UI | MPEXPR_TYPE_RIGHTASSOC,                   220 },

  { "!",   (mpexpr_fun_t) e_mpq_sgn,
    MPEXPR_TYPE_LOGICAL_NOT | MPEXPR_TYPE_PREFIX,                     210 },
  { "-",   (mpexpr_fun_t) mpq_neg,
    MPEXPR_TYPE_UNARY | MPEXPR_TYPE_PREFIX,                           210 },

  { "*",   (mpexpr_fun_t) mpq_mul,           MPEXPR_TYPE_BINARY,      200 },
  { "/",   (mpexpr_fun_t) mpq_div,           MPEXPR_TYPE_BINARY,      200 },

  { "+",   (mpexpr_fun_t) mpq_add,           MPEXPR_TYPE_BINARY,      190 },
  { "-",   (mpexpr_fun_t) mpq_sub,           MPEXPR_TYPE_BINARY,      190 },

  { "<<",  (mpexpr_fun_t) mpq_mul_2exp,      MPEXPR_TYPE_BINARY_UI,   180 },
  { ">>",  (mpexpr_fun_t) mpq_div_2exp,      MPEXPR_TYPE_BINARY_UI,   180 },

  { "<=",  (mpexpr_fun_t) mpq_cmp,           MPEXPR_TYPE_CMP_LE,      170 },
  { "<",   (mpexpr_fun_t) mpq_cmp,           MPEXPR_TYPE_CMP_LT,      170 },
  { ">=",  (mpexpr_fun_t) mpq_cmp,           MPEXPR_TYPE_CMP_GE,      170 },
  { ">",   (mpexpr_fun_t) mpq_cmp,           MPEXPR_TYPE_CMP_GT,      170 },

  { "==",  (mpexpr_fun_t) e_mpq_equal,       MPEXPR_TYPE_I_BINARY,    160 },
  { "!=",  (mpexpr_fun_t) e_mpq_notequal,    MPEXPR_TYPE_I_BINARY,    160 },

  { "&&",  (mpexpr_fun_t) e_mpq_sgn,         MPEXPR_TYPE_LOGICAL_AND, 120 },
  { "||",  (mpexpr_fun_t) e_mpq_sgn,         MPEXPR_TYPE_LOGICAL_OR,  110 },

  { ":",   NULL,                             MPEXPR_TYPE_COLON,       101 },
  { "?",   (mpexpr_fun_t) e_mpq_sgn,         MPEXPR_TYPE_QUESTION,    100 },

  { ")",   (mpexpr_fun_t) e_mpq_sgn,         MPEXPR_TYPE_CLOSEPAREN,    4 },
  { "(",   (mpexpr_fun_t) e_mpq_sgn,         MPEXPR_TYPE_OPENPAREN,     3 },
  { ",",   (mpexpr_fun_t) e_mpq_sgn,         MPEXPR_TYPE_ARGSEP,        2 },
  { "$",   NULL,                             MPEXPR_TYPE_VARIABLE,      1 },

  { "abs",  (mpexpr_fun_t) mpq_abs,          MPEXPR_TYPE_UNARY            },
  { "cmp",  (mpexpr_fun_t) mpq_cmp,          MPEXPR_TYPE_I_BINARY         },
  { "den",  (mpexpr_fun_t) e_mpq_den,        MPEXPR_TYPE_UNARY            },
  { "max",  (mpexpr_fun_t) mpq_cmp,  MPEXPR_TYPE_MAX | MPEXPR_TYPE_PAIRWISE },
  { "min",  (mpexpr_fun_t) mpq_cmp,  MPEXPR_TYPE_MIN | MPEXPR_TYPE_PAIRWISE },
  { "num",  (mpexpr_fun_t) e_mpq_num,        MPEXPR_TYPE_UNARY            },
  { "sgn",  (mpexpr_fun_t) e_mpq_sgn,        MPEXPR_TYPE_I_UNARY          },

  { NULL }
};

const struct mpexpr_operator_t * const mpq_expr_standard_table
= _mpq_expr_standard_table;


int
mpq_expr (mpq_ptr res, int base, const char *e, ...)
{
  mpq_srcptr  var[MPEXPR_VARIABLES];
  va_list     ap;
  int         ret;
  va_start (ap, e);

  TRACE (printf ("mpq_expr(): base %d, %s\n", base, e));
  ret = mpexpr_va_to_var ((void **) var, ap);
  va_end (ap);

  if (ret != MPEXPR_RESULT_OK)
    return ret;

  return mpq_expr_a (mpq_expr_standard_table, res, base, e, strlen(e), var);
}
