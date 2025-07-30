/* matrix22_mul1_inverse_vector.c

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2008, 2010 Free Software Foundation, Inc.

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

/* Sets (r;b) = M^{-1}(a;b), with M^{-1} = (u11, -u01; -u10, u00) from
   the left. Uses three buffers, to avoid a copy. */
mp_size_t
mpn_matrix22_mul1_inverse_vector (const struct hgcd_matrix1 *M,
				  mp_ptr rp, mp_srcptr ap, mp_ptr bp, mp_size_t n)
{
  mp_limb_t h0, h1;

  /* Compute (r;b) <-- (u11 a - u01 b; -u10 a + u00 b) as

     r  = u11 * a
     r -= u01 * b
     b *= u00
     b -= u10 * a
  */

  h0 =    mpn_mul_1 (rp, ap, n, M->u[1][1]);
  h1 = mpn_submul_1 (rp, bp, n, M->u[0][1]);
  ASSERT (h0 == h1);

  h0 =    mpn_mul_1 (bp, bp, n, M->u[0][0]);
  h1 = mpn_submul_1 (bp, ap, n, M->u[1][0]);
  ASSERT (h0 == h1);

  n -= (rp[n-1] | bp[n-1]) == 0;
  return n;
}
