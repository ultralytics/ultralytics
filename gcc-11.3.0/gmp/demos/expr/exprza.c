/* mpz expression evaluation

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

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include "gmp.h"
#include "expr-impl.h"


/* No need to parse '-' since that's handled as an operator.
   This function also by mpq_expr_a, so it's not static.  */
size_t
mpexpr_mpz_number (mpz_ptr res, const char *e, size_t elen, int base)
{
  char    *edup;
  size_t  i, ret;
  int     base_effective = (base == 0 ? 10 : base);
  void    *(*allocate_func) (size_t);
  void    (*free_func) (void *, size_t);

  i = 0;
  if (e[i] == '0')
    {
      i++;
      if (e[i] == 'x' || e[i] == 'b')
        i++;
    }

  for ( ; i < elen; i++)
    if (! isasciidigit_in_base (e[i], base_effective))
      break;

  mp_get_memory_functions (&allocate_func, NULL, &free_func);
  edup = (*allocate_func) (i+1);
  memcpy (edup, e, i);
  edup[i] = '\0';

  if (mpz_set_str (res, edup, base) == 0)
    ret = i;
  else
    ret = 0;

  (*free_func) (edup, i+1);
  return ret;
}

/* ignoring prec */
static void
e_mpz_init (mpz_ptr z, unsigned long prec)
{
  mpz_init (z);
}

int
mpz_expr_a (const struct mpexpr_operator_t *table,
            mpz_ptr res, int base,
            const char *e, size_t elen,
            mpz_srcptr var[26])
{
  struct mpexpr_parse_t  p;

  p.table = table;
  p.res = (mpX_ptr) res;
  p.base = base;
  p.e = e;
  p.elen = elen;
  p.var = (mpX_srcptr *) var;

  p.mpX_clear       = (mpexpr_fun_one_t)      mpz_clear;
  p.mpX_ulong_p     = (mpexpr_fun_i_unary_t)  mpz_fits_ulong_p;
  p.mpX_get_ui      = (mpexpr_fun_get_ui_t)   mpz_get_ui;
  p.mpX_init        = (mpexpr_fun_unary_ui_t) e_mpz_init;
  p.mpX_number      = (mpexpr_fun_number_t)   mpexpr_mpz_number;
  p.mpX_set         = (mpexpr_fun_unary_t)    mpz_set;
  p.mpX_set_or_swap = (mpexpr_fun_unary_t)    mpz_swap;
  p.mpX_set_si      = (mpexpr_fun_set_si_t)   mpz_set_si;
  p.mpX_swap        = (mpexpr_fun_swap_t)     mpz_swap;

  return mpexpr_evaluate (&p);
}
