/* mpf expression evaluation

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


/* Future: Bitwise "&", "|" and "&" could be done, if desired.  Not sure
   those functions would be much value though.  */


#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "gmp.h"
#include "expr-impl.h"


/* Change this to "#define TRACE(x) x" to get some traces. */
#define TRACE(x)


static size_t
e_mpf_number (mpf_ptr res, const char *e, size_t elen, int base)
{
  char    *edup;
  size_t  i, ret, extra=0;
  int     mant_base, exp_base;
  void    *(*allocate_func) (size_t);
  void    (*free_func) (void *, size_t);

  TRACE (printf ("mpf_number base=%d \"%.*s\"\n", base, (int) elen, e));

  /* mpf_set_str doesn't currently accept 0x for hex in base==0, so do it
     here instead.  FIXME: Would prefer to let mpf_set_str handle this.  */
  if (base == 0 && elen >= 2 && e[0] == '0' && (e[1] == 'x' || e[1] == 'X'))
    {
      base = 16;
      extra = 2;
      e += extra;
      elen -= extra;
    }

  if (base == 0)
    mant_base = 10;
  else if (base < 0)
    mant_base = -base;
  else
    mant_base = base;

  /* exponent in decimal if base is negative */
  if (base < 0)
    exp_base = 10;
  else if (base == 0)
    exp_base = 10;
  else
    exp_base = base;

#define IS_EXPONENT(c) \
  (c == '@' || (base <= 10 && base >= -10 && (e[i] == 'e' || e[i] == 'E')))

  i = 0;
  for (;;)
    {
      if (i >= elen)
        goto parsed;
      if (e[i] == '.')
        break;
      if (IS_EXPONENT (e[i]))
        goto exponent;
      if (! isasciidigit_in_base (e[i], mant_base))
        goto parsed;
      i++;
    }

  /* fraction */
  i++;
  for (;;)
    {
      if (i >= elen)
        goto parsed;
      if (IS_EXPONENT (e[i]))
        goto exponent;
      if (! isasciidigit_in_base (e[i], mant_base))
        goto parsed;
      i++;
    }

 exponent:
  i++;
  if (i >= elen)
    goto parsed;
  if (e[i] == '-')
    i++;
  for (;;)
    {
      if (i >= elen)
        goto parsed;
      if (! isasciidigit_in_base (e[i], exp_base))
        break;
      i++;
    }

 parsed:
  TRACE (printf ("  parsed i=%u \"%.*s\"\n", i, (int) i, e));

  mp_get_memory_functions (&allocate_func, NULL, &free_func);
  edup = (*allocate_func) (i+1);
  memcpy (edup, e, i);
  edup[i] = '\0';

  if (mpf_set_str (res, edup, base) == 0)
    ret = i + extra;
  else
    ret = 0;

  (*free_func) (edup, i+1);
  return ret;
}

static int
e_mpf_ulong_p (mpf_srcptr f)
{
  return mpf_integer_p (f) && mpf_fits_ulong_p (f);
}

/* Don't want to change the precision of w, can only do an actual swap when
   w and x have the same precision.  */
static void
e_mpf_set_or_swap (mpf_ptr w, mpf_ptr x)
{
  if (mpf_get_prec (w) == mpf_get_prec (x))
    mpf_swap (w, x);
  else
    mpf_set (w, x);
}


int
mpf_expr_a (const struct mpexpr_operator_t *table,
            mpf_ptr res, int base, unsigned long prec,
            const char *e, size_t elen,
            mpf_srcptr var[26])
{
  struct mpexpr_parse_t  p;

  p.table = table;
  p.res = (mpX_ptr) res;
  p.base = base;
  p.prec = prec;
  p.e = e;
  p.elen = elen;
  p.var = (mpX_srcptr *) var;

  p.mpX_clear       = (mpexpr_fun_one_t)      mpf_clear;
  p.mpX_ulong_p     = (mpexpr_fun_i_unary_t)  e_mpf_ulong_p;
  p.mpX_get_ui      = (mpexpr_fun_get_ui_t)   mpf_get_ui;
  p.mpX_init        = (mpexpr_fun_unary_ui_t) mpf_init2;
  p.mpX_number      = (mpexpr_fun_number_t)   e_mpf_number;
  p.mpX_set         = (mpexpr_fun_unary_t)    mpf_set;
  p.mpX_set_or_swap = (mpexpr_fun_unary_t)    e_mpf_set_or_swap;
  p.mpX_set_si      = (mpexpr_fun_set_si_t)   mpf_set_si;
  p.mpX_swap        = (mpexpr_fun_swap_t)     mpf_swap;

  return mpexpr_evaluate (&p);
}
