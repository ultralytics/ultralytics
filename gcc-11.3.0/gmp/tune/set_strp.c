/* mpn_set_str_subquad -- mpn_set_str forced to the sub-quadratic case.

Copyright 2002 Free Software Foundation, Inc.

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

#define TUNE_PROGRAM_BUILD  1   /* for gmp-impl.h */

#include "gmp.h"
#include "gmp-impl.h"

void
mpn_pre_set_str (mp_ptr wp, unsigned char *str, size_t str_len, powers_t *powtab, mp_ptr tp)
{
  if (BELOW_THRESHOLD (str_len, set_str_dc_threshold))
    mpn_bc_set_str (wp, str, str_len, powtab->base);
  else
    mpn_dc_set_str (wp, str, str_len, powtab, tp);
}
