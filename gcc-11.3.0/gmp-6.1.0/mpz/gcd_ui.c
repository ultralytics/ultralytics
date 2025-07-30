/* mpz_gcd_ui -- Calculate the greatest common divisor of two integers.

Copyright 1994, 1996, 1999-2004 Free Software Foundation, Inc.

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

#include <stdio.h> /* for NULL */
#include "gmp.h"
#include "gmp-impl.h"

unsigned long int
mpz_gcd_ui (mpz_ptr w, mpz_srcptr u, unsigned long int v)
{
  mp_size_t un;
  mp_limb_t res;

#if BITS_PER_ULONG > GMP_NUMB_BITS  /* avoid warnings about shift amount */
  if (v > GMP_NUMB_MAX)
    {
      mpz_t vz;
      mp_limb_t vlimbs[2];
      vlimbs[0] = v & GMP_NUMB_MASK;
      vlimbs[1] = v >> GMP_NUMB_BITS;
      PTR(vz) = vlimbs;
      SIZ(vz) = 2;
      mpz_gcd (w, u, vz);
      /* because v!=0 we will have w<=v hence fitting a ulong */
      ASSERT (mpz_fits_ulong_p (w));
      return mpz_get_ui (w);
    }
#endif

  un = ABSIZ(u);

  if (un == 0)
    res = v;
  else if (v == 0)
    {
      if (w != NULL)
	{
	  if (u != w)
	    {
	      MPZ_REALLOC (w, un);
	      MPN_COPY (PTR(w), PTR(u), un);
	    }
	  SIZ(w) = un;
	}
      /* Return u if it fits a ulong, otherwise 0. */
      res = PTR(u)[0];
      return (un == 1 && res <= ULONG_MAX ? res : 0);
    }
  else
    res = mpn_gcd_1 (PTR(u), un, (mp_limb_t) v);

  if (w != NULL)
    {
      PTR(w)[0] = res;
      SIZ(w) = res != 0;
    }
  return res;
}
