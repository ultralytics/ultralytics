/* mpz expression evaluation, simple part

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

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include "gmp.h"
#include "expr-impl.h"


/* Change this to "#define TRACE(x) x" to get some traces. */
#define TRACE(x)


/* These are macros, so need function wrappers. */
static int
e_mpz_sgn (mpz_srcptr x)
{
  return mpz_sgn (x);
}
static int
e_mpz_odd_p (mpz_srcptr x)
{
  return mpz_odd_p (x);
}
static int
e_mpz_even_p (mpz_srcptr x)
{
  return mpz_even_p (x);
}

/* These wrapped because MPEXPR_TYPE_I_ functions are expected to return
   "int" whereas these return "unsigned long".  */
static void
e_mpz_hamdist (mpz_ptr w, mpz_srcptr x, mpz_srcptr y)
{
  mpz_set_ui (w, mpz_hamdist (x, y));
}
static void
e_mpz_popcount (mpz_ptr w, mpz_srcptr x)
{
  mpz_set_ui (w, mpz_popcount (x));
}
static void
e_mpz_scan0 (mpz_ptr w, mpz_srcptr x, unsigned long start)
{
  mpz_set_ui (w, mpz_scan0 (x, start));
}
static void
e_mpz_scan1 (mpz_ptr w, mpz_srcptr x, unsigned long start)
{
  mpz_set_ui (w, mpz_scan1 (x, start));
}

/* These wrapped because they're in-place whereas MPEXPR_TYPE_BINARY_UI
   expects a separate source and destination.  Actually the parser will
   normally pass w==x anyway.  */
static void
e_mpz_setbit (mpz_ptr w, mpz_srcptr x, unsigned long n)
{
  if (w != x)
    mpz_set (w, x);
  mpz_setbit (w, n);
}
static void
e_mpz_clrbit (mpz_ptr w, mpz_srcptr x, unsigned long n)
{
  if (w != x)
    mpz_set (w, x);
  mpz_clrbit (w, n);
}

static const struct mpexpr_operator_t  _mpz_expr_standard_table[] = {

  { "**",  (mpexpr_fun_t) mpz_pow_ui,
    MPEXPR_TYPE_BINARY_UI | MPEXPR_TYPE_RIGHTASSOC,                  220 },

  { "~",   (mpexpr_fun_t) mpz_com,
    MPEXPR_TYPE_UNARY | MPEXPR_TYPE_PREFIX,                          210 },
  { "!",   (mpexpr_fun_t) e_mpz_sgn,
    MPEXPR_TYPE_LOGICAL_NOT | MPEXPR_TYPE_PREFIX,                    210 },
  { "-",   (mpexpr_fun_t) mpz_neg,
    MPEXPR_TYPE_UNARY | MPEXPR_TYPE_PREFIX,                          210 },

  { "*",   (mpexpr_fun_t) mpz_mul,          MPEXPR_TYPE_BINARY,      200 },
  { "/",   (mpexpr_fun_t) mpz_tdiv_q,       MPEXPR_TYPE_BINARY,      200 },
  { "%",   (mpexpr_fun_t) mpz_tdiv_r,       MPEXPR_TYPE_BINARY,      200 },

  { "+",   (mpexpr_fun_t) mpz_add,          MPEXPR_TYPE_BINARY,      190 },
  { "-",   (mpexpr_fun_t) mpz_sub,          MPEXPR_TYPE_BINARY,      190 },

  { "<<",  (mpexpr_fun_t) mpz_mul_2exp,     MPEXPR_TYPE_BINARY_UI,   180 },
  { ">>",  (mpexpr_fun_t) mpz_tdiv_q_2exp,  MPEXPR_TYPE_BINARY_UI,   180 },

  { "<=",  (mpexpr_fun_t) mpz_cmp,          MPEXPR_TYPE_CMP_LE,      170 },
  { "<",   (mpexpr_fun_t) mpz_cmp,          MPEXPR_TYPE_CMP_LT,      170 },
  { ">=",  (mpexpr_fun_t) mpz_cmp,          MPEXPR_TYPE_CMP_GE,      170 },
  { ">",   (mpexpr_fun_t) mpz_cmp,          MPEXPR_TYPE_CMP_GT,      170 },

  { "==",  (mpexpr_fun_t) mpz_cmp,          MPEXPR_TYPE_CMP_EQ,      160 },
  { "!=",  (mpexpr_fun_t) mpz_cmp,          MPEXPR_TYPE_CMP_NE,      160 },

  { "&",   (mpexpr_fun_t) mpz_and,          MPEXPR_TYPE_BINARY,      150 },
  { "^",   (mpexpr_fun_t) mpz_xor,          MPEXPR_TYPE_BINARY,      140 },
  { "|",   (mpexpr_fun_t) mpz_ior,          MPEXPR_TYPE_BINARY,      130 },
  { "&&",  (mpexpr_fun_t) e_mpz_sgn, MPEXPR_TYPE_LOGICAL_AND, 120 },
  { "||",  (mpexpr_fun_t) e_mpz_sgn, MPEXPR_TYPE_LOGICAL_OR,  110 },

  { ":",   NULL,                            MPEXPR_TYPE_COLON,       101 },
  { "?",   (mpexpr_fun_t) e_mpz_sgn, MPEXPR_TYPE_QUESTION,    100 },

  { ")",   NULL,                            MPEXPR_TYPE_CLOSEPAREN,   4 },
  { "(",   NULL,                            MPEXPR_TYPE_OPENPAREN,    3 },
  { ",",   NULL,                            MPEXPR_TYPE_ARGSEP,       2 },
  { "$",   NULL,                            MPEXPR_TYPE_VARIABLE,     1 },

  { "abs",       (mpexpr_fun_t) mpz_abs,           MPEXPR_TYPE_UNARY         },
  { "bin",       (mpexpr_fun_t) mpz_bin_ui,        MPEXPR_TYPE_BINARY_UI     },
  { "clrbit",    (mpexpr_fun_t) e_mpz_clrbit,      MPEXPR_TYPE_BINARY_UI     },
  { "cmp",       (mpexpr_fun_t) mpz_cmp,           MPEXPR_TYPE_I_BINARY      },
  { "cmpabs",    (mpexpr_fun_t) mpz_cmpabs,        MPEXPR_TYPE_I_BINARY      },
  { "congruent_p",(mpexpr_fun_t)mpz_congruent_p,   MPEXPR_TYPE_I_TERNARY     },
  { "divisible_p",(mpexpr_fun_t)mpz_divisible_p,   MPEXPR_TYPE_I_BINARY      },
  { "even_p",    (mpexpr_fun_t) e_mpz_even_p,      MPEXPR_TYPE_I_UNARY       },
  { "fib",       (mpexpr_fun_t) mpz_fib_ui,        MPEXPR_TYPE_UNARY_UI      },
  { "fac",       (mpexpr_fun_t) mpz_fac_ui,        MPEXPR_TYPE_UNARY_UI      },
  { "gcd",       (mpexpr_fun_t) mpz_gcd,           MPEXPR_TYPE_BINARY
						   | MPEXPR_TYPE_PAIRWISE    },
  { "hamdist",   (mpexpr_fun_t) e_mpz_hamdist,     MPEXPR_TYPE_BINARY        },
  { "invert",    (mpexpr_fun_t) mpz_invert,        MPEXPR_TYPE_BINARY        },
  { "jacobi",    (mpexpr_fun_t) mpz_jacobi,        MPEXPR_TYPE_I_BINARY      },
  { "kronecker", (mpexpr_fun_t) mpz_kronecker,     MPEXPR_TYPE_I_BINARY      },
  { "lcm",       (mpexpr_fun_t) mpz_lcm,           MPEXPR_TYPE_BINARY
						   | MPEXPR_TYPE_PAIRWISE    },
  { "lucnum",    (mpexpr_fun_t) mpz_lucnum_ui,     MPEXPR_TYPE_UNARY_UI      },
  { "max",       (mpexpr_fun_t) mpz_cmp,           MPEXPR_TYPE_MAX
						   | MPEXPR_TYPE_PAIRWISE    },
  { "min",       (mpexpr_fun_t) mpz_cmp,           MPEXPR_TYPE_MIN
						   | MPEXPR_TYPE_PAIRWISE    },
  { "nextprime", (mpexpr_fun_t) mpz_nextprime,     MPEXPR_TYPE_UNARY         },
  { "odd_p",     (mpexpr_fun_t) e_mpz_odd_p,       MPEXPR_TYPE_I_UNARY       },
  { "perfect_power_p", (mpexpr_fun_t)mpz_perfect_power_p, MPEXPR_TYPE_I_UNARY},
  { "perfect_square_p",(mpexpr_fun_t)mpz_perfect_square_p,MPEXPR_TYPE_I_UNARY},
  { "popcount",  (mpexpr_fun_t) e_mpz_popcount,    MPEXPR_TYPE_UNARY         },
  { "powm",      (mpexpr_fun_t) mpz_powm,          MPEXPR_TYPE_TERNARY       },
  { "probab_prime_p",  (mpexpr_fun_t)mpz_probab_prime_p,  MPEXPR_TYPE_I_UNARY},
  { "root",      (mpexpr_fun_t) mpz_root,          MPEXPR_TYPE_BINARY_UI     },
  { "scan0",     (mpexpr_fun_t) e_mpz_scan0,       MPEXPR_TYPE_BINARY_UI     },
  { "scan1",     (mpexpr_fun_t) e_mpz_scan1,       MPEXPR_TYPE_BINARY_UI     },
  { "setbit",    (mpexpr_fun_t) e_mpz_setbit,      MPEXPR_TYPE_BINARY_UI     },
  { "tstbit",    (mpexpr_fun_t) mpz_tstbit,        MPEXPR_TYPE_I_BINARY_UI   },
  { "sgn",       (mpexpr_fun_t) e_mpz_sgn,         MPEXPR_TYPE_I_UNARY       },
  { "sqrt",      (mpexpr_fun_t) mpz_sqrt,          MPEXPR_TYPE_UNARY         },
  { NULL }
};

/* The table is available globally only through a pointer, so the table size
   can change without breaking binary compatibility. */
const struct mpexpr_operator_t * const mpz_expr_standard_table
= _mpz_expr_standard_table;


int
mpz_expr (mpz_ptr res, int base, const char *e, ...)
{
  mpz_srcptr  var[MPEXPR_VARIABLES];
  va_list     ap;
  int         ret;
  va_start (ap, e);

  TRACE (printf ("mpz_expr(): base %d, %s\n", base, e));
  ret = mpexpr_va_to_var ((void **) var, ap);
  va_end (ap);

  if (ret != MPEXPR_RESULT_OK)
    return ret;

  return mpz_expr_a (mpz_expr_standard_table, res, base, e, strlen(e), var);
}
