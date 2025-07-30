/* double mpf_get_d_2exp (signed long int *exp, mpf_t src).

Copyright 2001-2004 Free Software Foundation, Inc.

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

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"


double
mpf_get_d_2exp (signed long int *exp2, mpf_srcptr src)
{
  mp_size_t size, abs_size;
  mp_srcptr ptr;
  int cnt;
  long exp;

  size = SIZ(src);
  if (UNLIKELY (size == 0))
    {
      *exp2 = 0;
      return 0.0;
    }

  ptr = PTR(src);
  abs_size = ABS (size);
  count_leading_zeros (cnt, ptr[abs_size - 1]);
  cnt -= GMP_NAIL_BITS;

  exp = EXP(src) * GMP_NUMB_BITS - cnt;
  *exp2 = exp;

  return mpn_get_d (ptr, abs_size, (mp_size_t) 0,
                    (long) - (abs_size * GMP_NUMB_BITS - cnt));
}
