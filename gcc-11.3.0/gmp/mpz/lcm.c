/* mpz_lcm -- mpz/mpz least common multiple.

Copyright 1996, 2000, 2001, 2005, 2012 Free Software Foundation, Inc.

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
mpz_lcm (mpz_ptr r, mpz_srcptr u, mpz_srcptr v)
{
  mpz_t g;
  mp_size_t usize, vsize;
  TMP_DECL;

  usize = SIZ (u);
  vsize = SIZ (v);
  if (usize == 0 || vsize == 0)
    {
      SIZ (r) = 0;
      return;
    }
  usize = ABS (usize);
  vsize = ABS (vsize);

  if (vsize == 1 || usize == 1)
    {
      mp_limb_t  vl, gl, c;
      mp_srcptr  up;
      mp_ptr     rp;

      if (usize == 1)
	{
	  usize = vsize;
	  MPZ_SRCPTR_SWAP (u, v);
	}

      MPZ_REALLOC (r, usize+1);

      up = PTR(u);
      vl = PTR(v)[0];
      gl = mpn_gcd_1 (up, usize, vl);
      vl /= gl;

      rp = PTR(r);
      c = mpn_mul_1 (rp, up, usize, vl);
      rp[usize] = c;
      usize += (c != 0);
      SIZ(r) = usize;
      return;
    }

  TMP_MARK;
  MPZ_TMP_INIT (g, usize); /* v != 0 implies |gcd(u,v)| <= |u| */

  mpz_gcd (g, u, v);
  mpz_divexact (g, u, g);
  mpz_mul (r, g, v);

  SIZ (r) = ABS (SIZ (r));	/* result always positive */

  TMP_FREE;
}
