/* mpf_get_ui -- mpf to ulong conversion

Copyright 2001, 2002, 2004 Free Software Foundation, Inc.

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


/* Any fraction bits are truncated, meaning simply discarded.

   For values bigger than a ulong, the low bits are returned (the low
   absolute value bits actually), like mpz_get_ui, but this isn't
   documented.

   Notice this is equivalent to mpz_set_f + mpz_get_ui.


   Implementation:

   The limb just above the radix point for us to extract is ptr[size-exp].

   We need to check that the size-exp index falls in our available data
   range, 0 to size-1 inclusive.  We test this without risk of an overflow
   involving exp by requiring size>=exp (giving size-exp >= 0) and exp>0
   (giving size-exp <= size-1).

   Notice if size==0 there's no fetch, since of course size>=exp and exp>0
   can only be true if size>0.  So there's no special handling for size==0,
   it comes out as 0 the same as any other time we have no data at our
   target index.

   For nails, the second limb above the radix point is also required, this
   is ptr[size-exp+1].

   Again we need to check that size-exp+1 falls in our data range, 0 to
   size-1 inclusive.  We test without risk of overflow by requiring
   size+1>=exp (giving size-exp+1 >= 0) and exp>1 (giving size-exp+1 <=
   size-1).

   And again if size==0 these second fetch conditions are not satisfied
   either since size+1>=exp and exp>1 are only true if size>0.

   The code is arranged with exp>0 wrapping the exp>1 test since exp>1 is
   mis-compiled by alpha gcc prior to version 3.4.  It re-writes it as
   exp-1>0, which is incorrect when exp==MP_EXP_T_MIN.  By having exp>0
   tested first we ensure MP_EXP_T_MIN doesn't reach exp>1.  */

unsigned long
mpf_get_ui (mpf_srcptr f) __GMP_NOTHROW
{
  mp_size_t size;
  mp_exp_t exp;
  mp_srcptr fp;
  mp_limb_t fl;

  exp = EXP (f);
  size = SIZ (f);
  fp = PTR (f);

  fl = 0;
  if (exp > 0)
    {
      /* there are some limbs above the radix point */

      size = ABS (size);
      if (size >= exp)
        fl = fp[size-exp];

#if BITS_PER_ULONG > GMP_NUMB_BITS
      if (exp > 1 && size+1 >= exp)
        fl += (fp[size-exp+1] << GMP_NUMB_BITS);
#endif
    }

  return (unsigned long) fl;
}
