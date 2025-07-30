/* mpz_cdiv_q_ui -- Division rounding the quotient towards +infinity.  The
   remainder gets the opposite sign as the denominator.  In order to make it
   always fit into the return type, the negative of the true remainder is
   returned.

Copyright 1994, 1996, 1999, 2001, 2002, 2004, 2012 Free Software Foundation,
Inc.

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

unsigned long int
mpz_cdiv_q_ui (mpz_ptr quot, mpz_srcptr dividend, unsigned long int divisor)
{
  mp_size_t ns, nn, qn;
  mp_ptr np, qp;
  mp_limb_t rl;

  if (UNLIKELY (divisor == 0))
    DIVIDE_BY_ZERO;

  ns = SIZ(dividend);
  if (ns == 0)
    {
      SIZ(quot) = 0;
      return 0;
    }

  nn = ABS(ns);
  qp = MPZ_REALLOC (quot, nn);
  np = PTR(dividend);

#if BITS_PER_ULONG > GMP_NUMB_BITS  /* avoid warnings about shift amount */
  if (divisor > GMP_NUMB_MAX)
    {
      mp_limb_t dp[2], rp[2];

      if (nn == 1)		/* tdiv_qr requirements; tested above for 0 */
	{
	  qp[0] = 0;
	  rl = np[0];
	  qn = 1;		/* a white lie, fixed below */
	}
      else
	{
	  dp[0] = divisor & GMP_NUMB_MASK;
	  dp[1] = divisor >> GMP_NUMB_BITS;
	  mpn_tdiv_qr (qp, rp, (mp_size_t) 0, np, nn, dp, (mp_size_t) 2);
	  rl = rp[0] + (rp[1] << GMP_NUMB_BITS);
	  qn = nn - 2 + 1;
	}

      if (rl != 0 && ns >= 0)
	{
	  mpn_incr_u (qp, (mp_limb_t) 1);
	  rl = divisor - rl;
	}

      qn -= qp[qn - 1] == 0; qn -= qn != 0 && qp[qn - 1] == 0;
    }
  else
#endif
    {
      rl = mpn_divrem_1 (qp, (mp_size_t) 0, np, nn, (mp_limb_t) divisor);

      if (rl != 0 && ns >= 0)
	{
	  mpn_incr_u (qp, (mp_limb_t) 1);
	  rl = divisor - rl;
	}

      qn = nn - (qp[nn - 1] == 0);
    }

  SIZ(quot) = ns >= 0 ? qn : -qn;
  return rl;
}
