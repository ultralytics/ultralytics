/* mpq expression evaluation

Copyright 2000, 2001, 2004 Free Software Foundation, Inc.

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
#include "gmp.h"
#include "expr-impl.h"


static int
e_mpq_ulong_p (mpq_srcptr q)
{
  return mpz_fits_ulong_p (mpq_numref (q))
    && mpz_cmp_ui (mpq_denref (q), 1L) == 0;
}

/* get value as a ui, on the assumption it fits */
static int
e_mpq_get_ui_fits (mpq_srcptr q)
{
  return mpz_get_ui (mpq_numref (q));
}

static void
e_mpq_set_si1 (mpq_ptr q, long num)
{
  mpq_set_si (q, num, 1L);
}

/* The same as mpz, but putting the result in the numerator.  Negatives and
   fractions aren't parsed here because '-' and '/' are operators. */
static size_t
e_mpq_number (mpq_ptr res, const char *e, size_t elen, int base)
{
  mpz_set_ui (mpq_denref (res), 1L);
  return mpexpr_mpz_number (mpq_numref (res), e, elen, base);
}


/* ignoring prec */
static void
e_mpq_init (mpq_ptr q, unsigned long prec)
{
  mpq_init (q);
}

int
mpq_expr_a (const struct mpexpr_operator_t *table,
            mpq_ptr res, int base,
            const char *e, size_t elen,
            mpq_srcptr var[26])
{
  struct mpexpr_parse_t  p;

  p.table = table;
  p.res = (mpX_ptr) res;
  p.base = base;
  p.e = e;
  p.elen = elen;
  p.var = (mpX_srcptr *) var;

  p.mpX_clear       = (mpexpr_fun_one_t)      mpq_clear;
  p.mpX_ulong_p     = (mpexpr_fun_i_unary_t)  e_mpq_ulong_p;
  p.mpX_get_ui      = (mpexpr_fun_get_ui_t)   e_mpq_get_ui_fits;
  p.mpX_init        = (mpexpr_fun_unary_ui_t) e_mpq_init;
  p.mpX_number      = (mpexpr_fun_number_t)   e_mpq_number;
  p.mpX_set         = (mpexpr_fun_unary_t)    mpq_set;
  p.mpX_set_or_swap = (mpexpr_fun_unary_t)    mpq_swap;
  p.mpX_set_si      = (mpexpr_fun_set_si_t)   e_mpq_set_si1;
  p.mpX_swap        = (mpexpr_fun_swap_t)     mpq_swap;

  return mpexpr_evaluate (&p);
}
