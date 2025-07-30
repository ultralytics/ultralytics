/* mpn_random -- Generate random numbers.

Copyright 2001, 2002 Free Software Foundation, Inc.

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

void
mpn_random (mp_ptr ptr, mp_size_t size)
{
  gmp_randstate_ptr  rands;

  /* FIXME: Is size==0 supposed to be allowed? */
  ASSERT (size >= 0);

  if (size == 0)
    return;

  rands = RANDS;
  _gmp_rand (ptr, rands, size * GMP_NUMB_BITS);

  /* Make sure the most significant limb is non-zero.  */
  while (ptr[size-1] == 0)
    _gmp_rand (&ptr[size-1], rands, GMP_NUMB_BITS);
}
