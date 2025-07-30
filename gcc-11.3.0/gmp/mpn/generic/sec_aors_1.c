/* mpn_sec_add_1, mpn_sec_sub_1

   Contributed to the GNU project by Niels Möller

Copyright 2013, 2014 Free Software Foundation, Inc.

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

#if OPERATION_sec_add_1
#define FNAME mpn_sec_add_1
#define FNAME_itch mpn_sec_add_1_itch
#define OP_N mpn_add_n
#endif
#if OPERATION_sec_sub_1
#define FNAME mpn_sec_sub_1
#define FNAME_itch mpn_sec_sub_1_itch
#define OP_N mpn_sub_n
#endif

/* It's annoying to that we need scratch space */
mp_size_t
FNAME_itch (mp_size_t n)
{
  return n;
}

mp_limb_t
FNAME (mp_ptr rp, mp_srcptr ap, mp_size_t n, mp_limb_t b, mp_ptr scratch)
{
  scratch[0] = b;
  MPN_ZERO (scratch + 1, n-1);
  return OP_N (rp, ap, scratch, n);
}
