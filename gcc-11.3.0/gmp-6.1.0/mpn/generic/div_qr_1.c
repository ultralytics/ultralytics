/* mpn_div_qr_1 -- mpn by limb division.

   Contributed to the GNU project by Niels Möller and Torbjörn Granlund

Copyright 1991, 1993, 1994, 1996, 1998-2000, 2002, 2003, 2013 Free Software
Foundation, Inc.

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

#ifndef DIV_QR_1_NORM_THRESHOLD
#define DIV_QR_1_NORM_THRESHOLD 3
#endif
#ifndef DIV_QR_1_UNNORM_THRESHOLD
#define DIV_QR_1_UNNORM_THRESHOLD 3
#endif

#if GMP_NAIL_BITS > 0
#error Nail bits not supported
#endif

/* Divides {up, n} by d. Writes the n-1 low quotient limbs at {qp,
 * n-1}, and the high quote limb at *qh. Returns remainder. */
mp_limb_t
mpn_div_qr_1 (mp_ptr qp, mp_limb_t *qh, mp_srcptr up, mp_size_t n,
	      mp_limb_t d)
{
  unsigned cnt;
  mp_limb_t uh;

  ASSERT (n > 0);
  ASSERT (d > 0);

  if (d & GMP_NUMB_HIGHBIT)
    {
      /* Normalized case */
      mp_limb_t dinv, q;

      uh = up[--n];

      q = (uh >= d);
      *qh = q;
      uh -= (-q) & d;

      if (BELOW_THRESHOLD (n, DIV_QR_1_NORM_THRESHOLD))
	{
	  cnt = 0;
	plain:
	  while (n > 0)
	    {
	      mp_limb_t ul = up[--n];
	      udiv_qrnnd (qp[n], uh, uh, ul, d);
	    }
	  return uh >> cnt;
	}
      invert_limb (dinv, d);
      return mpn_div_qr_1n_pi1 (qp, up, n, uh, d, dinv);
    }
  else
    {
      /* Unnormalized case */
      mp_limb_t dinv, ul;

      if (! UDIV_NEEDS_NORMALIZATION
	  && BELOW_THRESHOLD (n, DIV_QR_1_UNNORM_THRESHOLD))
	{
	  uh = up[--n];
	  udiv_qrnnd (*qh, uh, CNST_LIMB(0), uh, d);
	  cnt = 0;
	  goto plain;
	}

      count_leading_zeros (cnt, d);
      d <<= cnt;

#if HAVE_NATIVE_div_qr_1u_pi1
      /* FIXME: Call loop doing on-the-fly normalization */
#endif

      /* Shift up front, use qp area for shifted copy. A bit messy,
	 since we have only n-1 limbs available, and shift the high
	 limb manually. */
      uh = up[--n];
      ul = (uh << cnt) | mpn_lshift (qp, up, n, cnt);
      uh >>= (GMP_LIMB_BITS - cnt);

      if (UDIV_NEEDS_NORMALIZATION
	  && BELOW_THRESHOLD (n, DIV_QR_1_UNNORM_THRESHOLD))
	{
	  udiv_qrnnd (*qh, uh, uh, ul, d);
	  up = qp;
	  goto plain;
	}
      invert_limb (dinv, d);

      udiv_qrnnd_preinv (*qh, uh, uh, ul, d, dinv);
      return mpn_div_qr_1n_pi1 (qp, qp, n, uh, d, dinv) >> cnt;
    }
}
