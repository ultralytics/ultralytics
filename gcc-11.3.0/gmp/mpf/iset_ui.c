/* mpf_init_set_ui() -- Initialize a float and assign it from an unsigned int.

Copyright 1993-1995, 2000, 2001, 2003, 2004 Free Software Foundation, Inc.

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
mpf_init_set_ui (mpf_ptr r, unsigned long int val)
{
  mp_size_t prec = __gmp_default_fp_limb_precision;
  mp_size_t size;

  r->_mp_prec = prec;
  r->_mp_d = __GMP_ALLOCATE_FUNC_LIMBS (prec + 1);
  r->_mp_d[0] = val & GMP_NUMB_MASK;
  size = (val != 0);

#if BITS_PER_ULONG > GMP_NUMB_BITS
  val >>= GMP_NUMB_BITS;
  r->_mp_d[1] = val;
  size += (val != 0);
#endif

  r->_mp_size = size;
  r->_mp_exp = size;
}
