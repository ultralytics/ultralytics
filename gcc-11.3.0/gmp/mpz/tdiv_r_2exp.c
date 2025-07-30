/* mpz_tdiv_r_2exp -- Divide an integer by 2**CNT and produce a remainder.

Copyright 1991, 1993-1995, 2001, 2002, 2012 Free Software Foundation, Inc.

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
mpz_tdiv_r_2exp (mpz_ptr res, mpz_srcptr in, mp_bitcnt_t cnt)
{
  mp_size_t in_size = ABSIZ (in);
  mp_size_t res_size;
  mp_size_t limb_cnt = cnt / GMP_NUMB_BITS;
  mp_srcptr in_ptr = PTR (in);

  if (in_size > limb_cnt)
    {
      /* The input operand is (probably) greater than 2**CNT.  */
      mp_limb_t x;

      x = in_ptr[limb_cnt] & (((mp_limb_t) 1 << cnt % GMP_NUMB_BITS) - 1);
      if (x != 0)
	{
	  res_size = limb_cnt + 1;
	  MPZ_REALLOC (res, res_size);

	  PTR (res)[limb_cnt] = x;
	}
      else
	{
	  res_size = limb_cnt;
	  MPN_NORMALIZE (in_ptr, res_size);

	  MPZ_REALLOC (res, res_size);

	  limb_cnt = res_size;
	}
    }
  else
    {
      /* The input operand is smaller than 2**CNT.  We perform a no-op,
	 apart from that we might need to copy IN to RES.  */
      res_size = in_size;
      MPZ_REALLOC (res, res_size);

      limb_cnt = res_size;
    }

  if (res != in)
    MPN_COPY (PTR (res), PTR (in), limb_cnt);
  SIZ (res) = SIZ (in) >= 0 ? res_size : -res_size;
}
