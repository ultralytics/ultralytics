/* mpn_redc_2.  Set rp[] <- up[]/R^n mod mp[].  Clobber up[].
   mp[] is n limbs; up[] is 2n limbs.

   THIS IS AN INTERNAL FUNCTION WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH THIS FUNCTION THROUGH DOCUMENTED INTERFACES.

Copyright (C) 2000-2002, 2004, 2008, 2012 Free Software Foundation, Inc.

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


#if GMP_NAIL_BITS != 0
you lose
#endif

/* For testing purposes, define our own mpn_addmul_2 if there is none already
   available.  */
#ifndef HAVE_NATIVE_mpn_addmul_2
#undef mpn_addmul_2
static mp_limb_t
mpn_addmul_2 (mp_ptr rp, mp_srcptr up, mp_size_t n, mp_srcptr vp)
{
  rp[n] = mpn_addmul_1 (rp, up, n, vp[0]);
  return mpn_addmul_1 (rp + 1, up, n, vp[1]);
}
#endif

#if defined (__GNUC__) && ! defined (NO_ASM) \
  && defined (__ia64) && W_TYPE_SIZE == 64
#define umul2low(ph, pl, uh, ul, vh, vl) \
  do {									\
    mp_limb_t _ph, _pl;							\
    __asm__ ("xma.hu %0 = %3, %5, f0\n\t"				\
	     "xma.l %1 = %3, %5, f0\n\t"				\
	     ";;\n\t"							\
	     "xma.l %0 = %3, %4, %0\n\t"				\
	     ";;\n\t"							\
	     "xma.l %0 = %2, %5, %0"					\
	     : "=&f" (ph), "=&f" (pl)					\
	     : "f" (uh), "f" (ul), "f" (vh), "f" (vl));			\
  } while (0)
#endif

#ifndef umul2low
#define umul2low(ph, pl, uh, ul, vh, vl) \
  do {									\
    mp_limb_t _ph, _pl;							\
    umul_ppmm (_ph, _pl, ul, vl);					\
    (ph) = _ph + (ul) * (vh) + (uh) * (vl);				\
    (pl) = _pl;								\
  } while (0)
#endif

mp_limb_t
mpn_redc_2 (mp_ptr rp, mp_ptr up, mp_srcptr mp, mp_size_t n, mp_srcptr mip)
{
  mp_limb_t q[2];
  mp_size_t j;
  mp_limb_t upn;
  mp_limb_t cy;

  ASSERT (n > 0);
  ASSERT_MPN (up, 2*n);

  if ((n & 1) != 0)
    {
      up[0] = mpn_addmul_1 (up, mp, n, (up[0] * mip[0]) & GMP_NUMB_MASK);
      up++;
    }

  for (j = n - 2; j >= 0; j -= 2)
    {
      umul2low (q[1], q[0], mip[1], mip[0], up[1], up[0]);
      upn = up[n];		/* mpn_addmul_2 overwrites this */
      up[1] = mpn_addmul_2 (up, mp, n, q);
      up[0] = up[n];
      up[n] = upn;
      up += 2;
    }

  cy = mpn_add_n (rp, up, up - n, n);
  return cy;
}
