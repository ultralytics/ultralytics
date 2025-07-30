/* mpz_powm_sec(res,base,exp,mod) -- Set R to (U^E) mod M.

   Contributed to the GNU project by Torbjorn Granlund.

Copyright 1991, 1993, 1994, 1996, 1997, 2000-2002, 2005, 2008, 2009, 2012 Free
Software Foundation, Inc.

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
mpz_powm_sec (mpz_ptr r, mpz_srcptr b, mpz_srcptr e, mpz_srcptr m)
{
  mp_size_t n;
  mp_ptr rp, tp;
  mp_srcptr bp, ep, mp;
  mp_size_t rn, bn, es, en;
  TMP_DECL;

  n = ABSIZ(m);

  mp = PTR(m);

  if (UNLIKELY ((n == 0) || (mp[0] % 2 == 0)))
    DIVIDE_BY_ZERO;

  es = SIZ(e);
  if (UNLIKELY (es <= 0))
    {
      if (es == 0)
	{
	  /* b^0 mod m,  b is anything and m is non-zero.
	     Result is 1 mod m, i.e., 1 or 0 depending on if m = 1.  */
	  SIZ(r) = n != 1 || mp[0] != 1;
	  PTR(r)[0] = 1;
	  return;
	}
      DIVIDE_BY_ZERO;
    }
  en = es;

  bn = ABSIZ(b);

  if (UNLIKELY (bn == 0))
    {
      SIZ(r) = 0;
      return;
    }

  TMP_MARK;
  tp = TMP_ALLOC_LIMBS (n + mpn_sec_powm_itch (bn, en * GMP_NUMB_BITS, n));

  rp = tp;  tp += n;

  bp = PTR(b);
  ep = PTR(e);

  mpn_sec_powm (rp, bp, bn, ep, en * GMP_NUMB_BITS, mp, n, tp);

  rn = n;

  MPN_NORMALIZE (rp, rn);

  if ((ep[0] & 1) && SIZ(b) < 0 && rn != 0)
    {
      mpn_sub (rp, PTR(m), n, rp, rn);
      rn = n;
      MPN_NORMALIZE (rp, rn);
    }

  MPZ_REALLOC (r, rn);
  SIZ(r) = rn;
  MPN_COPY (PTR(r), rp, rn);

  TMP_FREE;
}
