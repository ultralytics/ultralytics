/* mpz_divisible_ui_p -- mpz by ulong divisibility test.

Copyright 2000-2002 Free Software Foundation, Inc.

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


int
mpz_divisible_ui_p (mpz_srcptr a, unsigned long d)
{
  mp_size_t  asize;
  mp_ptr     ap;
  unsigned   twos;

  asize = SIZ(a);
  if (UNLIKELY (d == 0))
    return (asize == 0);

  if (asize == 0)  /* 0 divisible by any d */
    return 1;

  /* For nails don't try to be clever if d is bigger than a limb, just fake
     up an mpz_t and go to the main mpz_divisible_p.  */
  if (d > GMP_NUMB_MAX)
    {
      mp_limb_t  dlimbs[2];
      mpz_t      dz;
      ALLOC(dz) = 2;
      PTR(dz) = dlimbs;
      mpz_set_ui (dz, d);
      return mpz_divisible_p (a, dz);
    }

  ap = PTR(a);
  asize = ABS(asize);  /* ignore sign of a */

  if (ABOVE_THRESHOLD (asize, BMOD_1_TO_MOD_1_THRESHOLD))
    return mpn_mod_1 (ap, asize, (mp_limb_t) d) == 0;

  if (! (d & 1))
    {
      /* Strip low zero bits to get odd d required by modexact.  If d==e*2^n
	 and a is divisible by 2^n and by e, then it's divisible by d. */

      if ((ap[0] & LOW_ZEROS_MASK (d)) != 0)
	return 0;

      count_trailing_zeros (twos, (mp_limb_t) d);
      d >>= twos;
    }

  return mpn_modexact_1_odd (ap, asize, (mp_limb_t) d) == 0;
}
