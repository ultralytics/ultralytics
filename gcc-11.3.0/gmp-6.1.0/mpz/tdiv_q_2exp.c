/* mpz_tdiv_q_2exp -- Divide an integer by 2**CNT.  Round the quotient
   towards -infinity.

Copyright 1991, 1993, 1994, 1996, 2001, 2002, 2012 Free Software Foundation,
Inc.

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
mpz_tdiv_q_2exp (mpz_ptr r, mpz_srcptr u, mp_bitcnt_t cnt)
{
  mp_size_t un, rn;
  mp_size_t limb_cnt;
  mp_ptr rp;
  mp_srcptr up;

  un = SIZ(u);
  limb_cnt = cnt / GMP_NUMB_BITS;
  rn = ABS (un) - limb_cnt;

  if (rn <= 0)
    rn = 0;
  else
    {
      rp = MPZ_REALLOC (r, rn);
      up = PTR(u) + limb_cnt;

      cnt %= GMP_NUMB_BITS;
      if (cnt != 0)
	{
	  mpn_rshift (rp, up, rn, cnt);
	  rn -= rp[rn - 1] == 0;
	}
      else
	{
	  MPN_COPY_INCR (rp, up, rn);
	}
    }

  SIZ(r) = un >= 0 ? rn : -rn;
}
