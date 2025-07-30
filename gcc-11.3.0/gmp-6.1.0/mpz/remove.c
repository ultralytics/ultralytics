/* mpz_remove -- divide out a factor and return its multiplicity.

Copyright 1998-2002, 2012 Free Software Foundation, Inc.

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

mp_bitcnt_t
mpz_remove (mpz_ptr dest, mpz_srcptr src, mpz_srcptr f)
{
  mp_bitcnt_t pwr;
  mp_srcptr fp;
  mp_size_t sn, fn, afn;
  mp_limb_t fp0;

  sn = SIZ (src);
  fn = SIZ (f);
  fp = PTR (f);
  afn = ABS (fn);
  fp0 = fp[0];

  if (UNLIKELY ((afn <= (fp0 == 1)) /* mpz_cmpabs_ui (f, 1) <= 0 */
		| (sn == 0)))
    {
      /*  f = 0 or f = +- 1 or src = 0 */
      if (afn == 0)
	DIVIDE_BY_ZERO;
      mpz_set (dest, src);
      return 0;
    }

  if ((fp0 & 1) != 0)
    { /* f is odd */
      mp_ptr dp;
      mp_size_t dn;

      dn = ABS (sn);
      dp = MPZ_REALLOC (dest, dn);

      pwr = mpn_remove (dp, &dn, PTR(src), dn, PTR(f), afn, ~(mp_bitcnt_t) 0);

      SIZ (dest) = ((pwr & (fn < 0)) ^ (sn < 0)) ? -dn : dn;
    }
  else if (afn == (fp0 == 2))
    { /* mpz_cmpabs_ui (f, 2) == 0 */
      pwr = mpz_scan1 (src, 0);
      mpz_div_2exp (dest, src, pwr);
      if (pwr & (fn < 0)) /*((pwr % 2 == 1) && (SIZ (f) < 0))*/
	mpz_neg (dest, dest);
    }
  else
    { /* f != +-2 */
      mpz_t x, rem;

      mpz_init (rem);
      mpz_init (x);

      pwr = 0;
      mpz_tdiv_qr (x, rem, src, f);
      if (SIZ (rem) == 0)
	{
	  mpz_t fpow[GMP_LIMB_BITS];		/* Really MP_SIZE_T_BITS */
	  int p;

#if WANT_ORIGINAL_DEST
	  mp_ptr dp;
	  dp = PTR (dest);
#endif
      /* We could perhaps compute mpz_scan1(src,0)/mpz_scan1(f,0).  It is an
	 upper bound of the result we're seeking.  We could also shift down the
	 operands so that they become odd, to make intermediate values
	 smaller.  */
	  mpz_init_set (fpow[0], f);
	  mpz_swap (dest, x);

	  p = 1;
      /* Divide by f, f^2 ... f^(2^k) until we get a remainder for f^(2^k).  */
	  while (ABSIZ (dest) >= 2 * ABSIZ (fpow[p - 1]) - 1)
	    {
	      mpz_init (fpow[p]);
	      mpz_mul (fpow[p], fpow[p - 1], fpow[p - 1]);
	      mpz_tdiv_qr (x, rem, dest, fpow[p]);
	      if (SIZ (rem) != 0) {
		mpz_clear (fpow[p]);
		break;
	      }
	      mpz_swap (dest, x);
	      p++;
	    }

	  pwr = ((mp_bitcnt_t)1 << p) - 1;

      /* Divide by f^(2^(k-1)), f^(2^(k-2)), ..., f for all divisors that give
	 a zero remainder.  */
	  while (--p >= 0)
	    {
	      mpz_tdiv_qr (x, rem, dest, fpow[p]);
	      if (SIZ (rem) == 0)
		{
		  pwr += (mp_bitcnt_t)1 << p;
		  mpz_swap (dest, x);
		}
	      mpz_clear (fpow[p]);
	    }

#if WANT_ORIGINAL_DEST
	  if (PTR (x) == dp) {
	    mpz_swap (dest, x);
	    mpz_set (dest, x);
	  }
#endif
	}
      else
	mpz_set (dest, src);

      mpz_clear (x);
      mpz_clear (rem);
    }

  return pwr;
}
