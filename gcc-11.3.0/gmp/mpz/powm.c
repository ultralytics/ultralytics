/* mpz_powm(res,base,exp,mod) -- Set R to (U^E) mod M.

   Contributed to the GNU project by Torbjorn Granlund.

Copyright 1991, 1993, 1994, 1996, 1997, 2000-2002, 2005, 2008, 2009, 2011, 2012
Free Software Foundation, Inc.

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


/* TODO

 * Improve handling of buffers.  It is pretty ugly now.

 * For even moduli, we compute a binvert of its odd part both here and in
   mpn_powm.  How can we avoid this recomputation?
*/

/*
  b ^ e mod m   res
  0   0     0    ?
  0   e     0    ?
  0   0     m    ?
  0   e     m    0
  b   0     0    ?
  b   e     0    ?
  b   0     m    1 mod m
  b   e     m    b^e mod m
*/

#define HANDLE_NEGATIVE_EXPONENT 1

void
mpz_powm (mpz_ptr r, mpz_srcptr b, mpz_srcptr e, mpz_srcptr m)
{
  mp_size_t n, nodd, ncnt;
  int cnt;
  mp_ptr rp, tp;
  mp_srcptr bp, ep, mp;
  mp_size_t rn, bn, es, en, itch;
  mpz_t new_b;			/* note: value lives long via 'b' */
  TMP_DECL;

  n = ABSIZ(m);
  if (UNLIKELY (n == 0))
    DIVIDE_BY_ZERO;

  mp = PTR(m);

  TMP_MARK;

  es = SIZ(e);
  if (UNLIKELY (es <= 0))
    {
      if (es == 0)
	{
	  /* b^0 mod m,  b is anything and m is non-zero.
	     Result is 1 mod m, i.e., 1 or 0 depending on if m = 1.  */
	  SIZ(r) = n != 1 || mp[0] != 1;
	  PTR(r)[0] = 1;
	  TMP_FREE;	/* we haven't really allocated anything here */
	  return;
	}
#if HANDLE_NEGATIVE_EXPONENT
      MPZ_TMP_INIT (new_b, n + 1);

      if (UNLIKELY (! mpz_invert (new_b, b, m)))
	DIVIDE_BY_ZERO;
      b = new_b;
      es = -es;
#else
      DIVIDE_BY_ZERO;
#endif
    }
  en = es;

  bn = ABSIZ(b);

  if (UNLIKELY (bn == 0))
    {
      SIZ(r) = 0;
      TMP_FREE;
      return;
    }

  ep = PTR(e);

  /* Handle (b^1 mod m) early, since mpn_pow* do not handle that case.  */
  if (UNLIKELY (en == 1 && ep[0] == 1))
    {
      rp = TMP_ALLOC_LIMBS (n);
      bp = PTR(b);
      if (bn >= n)
	{
	  mp_ptr qp = TMP_ALLOC_LIMBS (bn - n + 1);
	  mpn_tdiv_qr (qp, rp, 0L, bp, bn, mp, n);
	  rn = n;
	  MPN_NORMALIZE (rp, rn);

	  if (SIZ(b) < 0 && rn != 0)
	    {
	      mpn_sub (rp, mp, n, rp, rn);
	      rn = n;
	      MPN_NORMALIZE (rp, rn);
	    }
	}
      else
	{
	  if (SIZ(b) < 0)
	    {
	      mpn_sub (rp, mp, n, bp, bn);
	      rn = n;
	      rn -= (rp[rn - 1] == 0);
	    }
	  else
	    {
	      MPN_COPY (rp, bp, bn);
	      rn = bn;
	    }
	}
      goto ret;
    }

  /* Remove low zero limbs from M.  This loop will terminate for correctly
     represented mpz numbers.  */
  ncnt = 0;
  while (UNLIKELY (mp[0] == 0))
    {
      mp++;
      ncnt++;
    }
  nodd = n - ncnt;
  cnt = 0;
  if (mp[0] % 2 == 0)
    {
      mp_ptr newmp = TMP_ALLOC_LIMBS (nodd);
      count_trailing_zeros (cnt, mp[0]);
      mpn_rshift (newmp, mp, nodd, cnt);
      nodd -= newmp[nodd - 1] == 0;
      mp = newmp;
      ncnt++;
    }

  if (ncnt != 0)
    {
      /* We will call both mpn_powm and mpn_powlo.  */
      /* rp needs n, mpn_powlo needs 4n, the 2 mpn_binvert might need more */
      mp_size_t n_largest_binvert = MAX (ncnt, nodd);
      mp_size_t itch_binvert = mpn_binvert_itch (n_largest_binvert);
      itch = 3 * n + MAX (itch_binvert, 2 * n);
    }
  else
    {
      /* We will call just mpn_powm.  */
      mp_size_t itch_binvert = mpn_binvert_itch (nodd);
      itch = n + MAX (itch_binvert, 2 * n);
    }
  tp = TMP_ALLOC_LIMBS (itch);

  rp = tp;  tp += n;

  bp = PTR(b);
  mpn_powm (rp, bp, bn, ep, en, mp, nodd, tp);

  rn = n;

  if (ncnt != 0)
    {
      mp_ptr r2, xp, yp, odd_inv_2exp;
      unsigned long t;
      int bcnt;

      if (bn < ncnt)
	{
	  mp_ptr newbp = TMP_ALLOC_LIMBS (ncnt);
	  MPN_COPY (newbp, bp, bn);
	  MPN_ZERO (newbp + bn, ncnt - bn);
	  bp = newbp;
	}

      r2 = tp;

      if (bp[0] % 2 == 0)
	{
	  if (en > 1)
	    {
	      MPN_ZERO (r2, ncnt);
	      goto zero;
	    }

	  ASSERT (en == 1);
	  t = (ncnt - (cnt != 0)) * GMP_NUMB_BITS + cnt;

	  /* Count number of low zero bits in B, up to 3.  */
	  bcnt = (0x1213 >> ((bp[0] & 7) << 1)) & 0x3;
	  /* Note that ep[0] * bcnt might overflow, but that just results
	     in a missed optimization.  */
	  if (ep[0] * bcnt >= t)
	    {
	      MPN_ZERO (r2, ncnt);
	      goto zero;
	    }
	}

      mpn_powlo (r2, bp, ep, en, ncnt, tp + ncnt);

    zero:
      if (nodd < ncnt)
	{
	  mp_ptr newmp = TMP_ALLOC_LIMBS (ncnt);
	  MPN_COPY (newmp, mp, nodd);
	  MPN_ZERO (newmp + nodd, ncnt - nodd);
	  mp = newmp;
	}

      odd_inv_2exp = tp + n;
      mpn_binvert (odd_inv_2exp, mp, ncnt, tp + 2 * n);

      mpn_sub (r2, r2, ncnt, rp, nodd > ncnt ? ncnt : nodd);

      xp = tp + 2 * n;
      mpn_mullo_n (xp, odd_inv_2exp, r2, ncnt);

      if (cnt != 0)
	xp[ncnt - 1] &= (CNST_LIMB(1) << cnt) - 1;

      yp = tp;
      if (ncnt > nodd)
	mpn_mul (yp, xp, ncnt, mp, nodd);
      else
	mpn_mul (yp, mp, nodd, xp, ncnt);

      mpn_add (rp, yp, n, rp, nodd);

      ASSERT (nodd + ncnt >= n);
      ASSERT (nodd + ncnt <= n + 1);
    }

  MPN_NORMALIZE (rp, rn);

  if ((ep[0] & 1) && SIZ(b) < 0 && rn != 0)
    {
      mpn_sub (rp, PTR(m), n, rp, rn);
      rn = n;
      MPN_NORMALIZE (rp, rn);
    }

 ret:
  MPZ_REALLOC (r, rn);
  SIZ(r) = rn;
  MPN_COPY (PTR(r), rp, rn);

  TMP_FREE;
}
