/* mpn_scan1 -- Scan from a given bit position for the next set bit.

Copyright 1994, 1996, 2001, 2002, 2004 Free Software Foundation, Inc.

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

/* Argument constraints:
   1. U must sooner or later have a limb != 0.
 */

mp_bitcnt_t
mpn_scan1 (mp_srcptr up, mp_bitcnt_t starting_bit)
{
  mp_size_t starting_word;
  mp_limb_t alimb;
  int cnt;
  mp_srcptr p;

  /* Start at the word implied by STARTING_BIT.  */
  starting_word = starting_bit / GMP_NUMB_BITS;
  p = up + starting_word;
  alimb = *p++;

  /* Mask off any bits before STARTING_BIT in the first limb.  */
  alimb &= - (mp_limb_t) 1 << (starting_bit % GMP_NUMB_BITS);

  while (alimb == 0)
    alimb = *p++;

  count_trailing_zeros (cnt, alimb);
  return (p - up - 1) * GMP_NUMB_BITS + cnt;
}
