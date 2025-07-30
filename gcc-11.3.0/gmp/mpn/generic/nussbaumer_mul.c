/* mpn_nussbaumer_mul -- Multiply {ap,an} and {bp,bn} using
   Nussbaumer's negacyclic convolution.

   Contributed to the GNU project by Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009 Free Software Foundation, Inc.

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

/* Multiply {ap,an} by {bp,bn}, and put the result in {pp, an+bn} */
void
mpn_nussbaumer_mul (mp_ptr pp,
		    mp_srcptr ap, mp_size_t an,
		    mp_srcptr bp, mp_size_t bn)
{
  mp_size_t rn;
  mp_ptr tp;
  TMP_DECL;

  ASSERT (an >= bn);
  ASSERT (bn > 0);

  TMP_MARK;

  if ((ap == bp) && (an == bn))
    {
      rn = mpn_sqrmod_bnm1_next_size (2*an);
      tp = TMP_ALLOC_LIMBS (mpn_sqrmod_bnm1_itch (rn, an));
      mpn_sqrmod_bnm1 (pp, rn, ap, an, tp);
    }
  else
    {
      rn = mpn_mulmod_bnm1_next_size (an + bn);
      tp = TMP_ALLOC_LIMBS (mpn_mulmod_bnm1_itch (rn, an, bn));
      mpn_mulmod_bnm1 (pp, rn, ap, an, bp, bn, tp);
    }

  TMP_FREE;
}
