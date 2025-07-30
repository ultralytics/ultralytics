/* mpz_array_init (array, array_size, size_per_elem) --

Copyright 1991, 1993-1995, 2000-2002, 2012 Free Software Foundation, Inc.

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
mpz_array_init (mpz_ptr arr, mp_size_t arr_size, mp_size_t nbits)
{
  mp_ptr p;
  mp_size_t i;
  mp_size_t nlimbs;

  nlimbs = nbits / GMP_NUMB_BITS + 1;
  p = __GMP_ALLOCATE_FUNC_LIMBS (arr_size * nlimbs);

  for (i = 0; i < arr_size; i++)
    {
      ALLOC (&arr[i]) = nlimbs + 1; /* Yes, lie a little... */
      SIZ (&arr[i]) = 0;
      PTR (&arr[i]) = p + i * nlimbs;
    }
}
