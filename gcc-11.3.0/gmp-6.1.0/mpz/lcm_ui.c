/* mpz_lcm_ui -- least common multiple of mpz and ulong.

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
#include "longlong.h"


void
mpz_lcm_ui (mpz_ptr r, mpz_srcptr u, unsigned long v)
{
  mp_size_t      usize;
  mp_srcptr      up;
  mp_ptr         rp;
  unsigned long  g;
  mp_limb_t      c;

#if BITS_PER_ULONG > GMP_NUMB_BITS  /* avoid warnings about shift amount */
  if (v > GMP_NUMB_MAX)
    {
      mpz_t vz;
      mp_limb_t vlimbs[2];
      vlimbs[0] = v & GMP_NUMB_MASK;
      vlimbs[1] = v >> GMP_NUMB_BITS;
      PTR(vz) = vlimbs;
      SIZ(vz) = 2;
      mpz_lcm (r, u, vz);
      return;
    }
#endif

  /* result zero if either operand zero */
  usize = SIZ(u);
  if (usize == 0 || v == 0)
    {
      SIZ(r) = 0;
      return;
    }
  usize = ABS(usize);

  MPZ_REALLOC (r, usize+1);

  up = PTR(u);
  g = (unsigned long) mpn_gcd_1 (up, usize, (mp_limb_t) v);
  v /= g;

  rp = PTR(r);
  c = mpn_mul_1 (rp, up, usize, (mp_limb_t) v);
  rp[usize] = c;
  usize += (c != 0);
  SIZ(r) = usize;
}
