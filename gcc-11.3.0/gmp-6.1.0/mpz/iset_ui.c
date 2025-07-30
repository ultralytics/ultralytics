/* mpz_init_set_ui(dest,val) -- Make a new multiple precision in DEST and
   assign VAL to the new number.

Copyright 1991, 1993-1995, 2000-2002, 2004, 2012 Free Software Foundation, Inc.

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
mpz_init_set_ui (mpz_ptr dest, unsigned long int val)
{
  mp_size_t size;

#if BITS_PER_ULONG > GMP_NUMB_BITS  /* avoid warnings about shift amount */
  if (val > GMP_NUMB_MAX)
    {
      ALLOC (dest) = 2;
      PTR (dest) = __GMP_ALLOCATE_FUNC_LIMBS (2);
      PTR (dest)[1] = val >> GMP_NUMB_BITS;
      size = 2;
    }
  else
#endif
    {
      ALLOC (dest) = 1;
      PTR (dest) = __GMP_ALLOCATE_FUNC_LIMBS (1);

      size = val != 0;
    }
  PTR (dest)[0] = val & GMP_NUMB_MASK;

  SIZ (dest) = size;
}
