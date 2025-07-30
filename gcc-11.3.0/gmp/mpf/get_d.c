/* double mpf_get_d (mpf_t src) -- return SRC truncated to a double.

Copyright 1996, 2001-2004 Free Software Foundation, Inc.

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

double
mpf_get_d (mpf_srcptr src)
{
  mp_size_t  size, abs_size;
  long       exp;

  size = SIZ (src);
  if (UNLIKELY (size == 0))
    return 0.0;

  abs_size = ABS (size);
  exp = (EXP (src) - abs_size) * GMP_NUMB_BITS;
  return mpn_get_d (PTR (src), abs_size, size, exp);
}
