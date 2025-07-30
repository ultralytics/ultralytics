/* Helper function for high degree Toom-Cook algorithms.

   Contributed to the GNU project by Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009, 2010 Free Software Foundation, Inc.

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

/* Gets {pp,n} and (sign?-1:1)*{np,n}. Computes at once:
     {pp,n} <- ({pp,n}+{np,n})/2^{ps+1}
     {pn,n} <- ({pp,n}-{np,n})/2^{ns+1}
   Finally recompose them obtaining:
     {pp,n+off} <- {pp,n}+{np,n}*2^{off*GMP_NUMB_BITS}
*/
void
mpn_toom_couple_handling (mp_ptr pp, mp_size_t n, mp_ptr np,
			  int nsign, mp_size_t off, int ps, int ns)
{
  if (nsign) {
#ifdef HAVE_NATIVE_mpn_rsh1sub_n
    mpn_rsh1sub_n (np, pp, np, n);
#else
    mpn_sub_n (np, pp, np, n);
    mpn_rshift (np, np, n, 1);
#endif
  } else {
#ifdef HAVE_NATIVE_mpn_rsh1add_n
    mpn_rsh1add_n (np, pp, np, n);
#else
    mpn_add_n (np, pp, np, n);
    mpn_rshift (np, np, n, 1);
#endif
  }

#ifdef HAVE_NATIVE_mpn_rsh1sub_n
  if (ps == 1)
    mpn_rsh1sub_n (pp, pp, np, n);
  else
#endif
  {
    mpn_sub_n (pp, pp, np, n);
    if (ps > 0)
      mpn_rshift (pp, pp, n, ps);
  }
  if (ns > 0)
    mpn_rshift (np, np, n, ns);
  pp[n] = mpn_add_n (pp+off, pp+off, np, n-off);
  ASSERT_NOCARRY (mpn_add_1(pp+n, np+n-off, off, pp[n]) );
}
