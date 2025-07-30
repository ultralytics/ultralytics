/* gmp_urandomm_ui -- uniform random number 0 to N-1 for ulong N.

Copyright 2003, 2004 Free Software Foundation, Inc.

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


/* If n is a power of 2 then the test ret<n is always true and the loop is
   unnecessary, but there's no need to add special code for this.  Just get
   the "bits" calculation correct and let it go through normally.

   If n is 1 then will have bits==0 and _gmp_rand will produce no output and
   we always return 0.  Again there seems no need for a special case, just
   initialize a[0]=0 and let it go through normally.  */

#define MAX_URANDOMM_ITER  80

unsigned long
gmp_urandomm_ui (gmp_randstate_ptr rstate, unsigned long n)
{
  mp_limb_t      a[LIMBS_PER_ULONG];
  unsigned long  ret, bits, leading;
  int            i;

  if (UNLIKELY (n == 0))
    DIVIDE_BY_ZERO;

  /* start with zeros, since if bits==0 then _gmp_rand will store nothing at
     all (bits==0 arises when n==1), or if bits <= GMP_NUMB_BITS then it
     will store only a[0].  */
  a[0] = 0;
#if LIMBS_PER_ULONG > 1
  a[1] = 0;
#endif

  count_leading_zeros (leading, (mp_limb_t) n);
  bits = GMP_LIMB_BITS - leading - (POW2_P(n) != 0);

  for (i = 0; i < MAX_URANDOMM_ITER; i++)
    {
      _gmp_rand (a, rstate, bits);
#if LIMBS_PER_ULONG == 1
      ret = a[0];
#else
      ret = a[0] | (a[1] << GMP_NUMB_BITS);
#endif
      if (LIKELY (ret < n))   /* usually one iteration suffices */
        goto done;
    }

  /* Too many iterations, there must be something degenerate about the
     rstate algorithm.  Return r%n.  */
  ret -= n;
  ASSERT (ret < n);

 done:
  return ret;
}
