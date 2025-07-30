/* mpn_mu_divappr_q, mpn_preinv_mu_divappr_q.

   Compute Q = floor(N / D) + e.  N is nn limbs, D is dn limbs and must be
   normalized, and Q must be nn-dn limbs, 0 <= e <= 4.  The requirement that Q
   is nn-dn limbs (and not nn-dn+1 limbs) was put in place in order to allow us
   to let N be unmodified during the operation.

   Contributed to the GNU project by Torbjorn Granlund.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GMP RELEASE.

Copyright 2005-2007, 2009, 2010 Free Software Foundation, Inc.

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


/*
   The idea of the algorithm used herein is to compute a smaller inverted value
   than used in the standard Barrett algorithm, and thus save time in the
   Newton iterations, and pay just a small price when using the inverted value
   for developing quotient bits.  This algorithm was presented at ICMS 2006.
*/

/* CAUTION: This code and the code in mu_div_qr.c should be edited in sync.

 Things to work on:

  * The itch/scratch scheme isn't perhaps such a good idea as it once seemed,
    demonstrated by the fact that the mpn_invertappr function's scratch needs
    mean that we need to keep a large allocation long after it is needed.
    Things are worse as mpn_mul_fft does not accept any scratch parameter,
    which means we'll have a large memory hole while in mpn_mul_fft.  In
    general, a peak scratch need in the beginning of a function isn't
    well-handled by the itch/scratch scheme.
*/

#ifdef STAT
#undef STAT
#define STAT(x) x
#else
#define STAT(x)
#endif

#include <stdlib.h>		/* for NULL */
#include "gmp.h"
#include "gmp-impl.h"


mp_limb_t
mpn_mu_divappr_q (mp_ptr qp,
		  mp_srcptr np,
		  mp_size_t nn,
		  mp_srcptr dp,
		  mp_size_t dn,
		  mp_ptr scratch)
{
  mp_size_t qn, in;
  mp_limb_t cy, qh;
  mp_ptr ip, tp;

  ASSERT (dn > 1);

  qn = nn - dn;

  /* If Q is smaller than D, truncate operands. */
  if (qn + 1 < dn)
    {
      np += dn - (qn + 1);
      nn -= dn - (qn + 1);
      dp += dn - (qn + 1);
      dn = qn + 1;
    }

  /* Compute the inverse size.  */
  in = mpn_mu_divappr_q_choose_in (qn, dn, 0);
  ASSERT (in <= dn);

#if 1
  /* This alternative inverse computation method gets slightly more accurate
     results.  FIXMEs: (1) Temp allocation needs not analysed (2) itch function
     not adapted (3) mpn_invertappr scratch needs not met.  */
  ip = scratch;
  tp = scratch + in + 1;

  /* compute an approximate inverse on (in+1) limbs */
  if (dn == in)
    {
      MPN_COPY (tp + 1, dp, in);
      tp[0] = 1;
      mpn_invertappr (ip, tp, in + 1, tp + in + 1);
      MPN_COPY_INCR (ip, ip + 1, in);
    }
  else
    {
      cy = mpn_add_1 (tp, dp + dn - (in + 1), in + 1, 1);
      if (UNLIKELY (cy != 0))
	MPN_ZERO (ip, in);
      else
	{
	  mpn_invertappr (ip, tp, in + 1, tp + in + 1);
	  MPN_COPY_INCR (ip, ip + 1, in);
	}
    }
#else
  /* This older inverse computation method gets slightly worse results than the
     one above.  */
  ip = scratch;
  tp = scratch + in;

  /* Compute inverse of D to in+1 limbs, then round to 'in' limbs.  Ideally the
     inversion function should do this automatically.  */
  if (dn == in)
    {
      tp[in + 1] = 0;
      MPN_COPY (tp + in + 2, dp, in);
      mpn_invertappr (tp, tp + in + 1, in + 1, NULL);
    }
  else
    {
      mpn_invertappr (tp, dp + dn - (in + 1), in + 1, NULL);
    }
  cy = mpn_sub_1 (tp, tp, in + 1, GMP_NUMB_HIGHBIT);
  if (UNLIKELY (cy != 0))
    MPN_ZERO (tp + 1, in);
  MPN_COPY (ip, tp + 1, in);
#endif

  qh = mpn_preinv_mu_divappr_q (qp, np, nn, dp, dn, ip, in, scratch + in);

  return qh;
}

mp_limb_t
mpn_preinv_mu_divappr_q (mp_ptr qp,
			 mp_srcptr np,
			 mp_size_t nn,
			 mp_srcptr dp,
			 mp_size_t dn,
			 mp_srcptr ip,
			 mp_size_t in,
			 mp_ptr scratch)
{
  mp_size_t qn;
  mp_limb_t cy, cx, qh;
  mp_limb_t r;
  mp_size_t tn, wn;

#define rp           scratch
#define tp           (scratch + dn)
#define scratch_out  (scratch + dn + tn)

  qn = nn - dn;

  np += qn;
  qp += qn;

  qh = mpn_cmp (np, dp, dn) >= 0;
  if (qh != 0)
    mpn_sub_n (rp, np, dp, dn);
  else
    MPN_COPY (rp, np, dn);

  if (qn == 0)
    return qh;			/* Degenerate use.  Should we allow this? */

  while (qn > 0)
    {
      if (qn < in)
	{
	  ip += in - qn;
	  in = qn;
	}
      np -= in;
      qp -= in;

      /* Compute the next block of quotient limbs by multiplying the inverse I
	 by the upper part of the partial remainder R.  */
      mpn_mul_n (tp, rp + dn - in, ip, in);		/* mulhi  */
      cy = mpn_add_n (qp, tp + in, rp + dn - in, in);	/* I's msb implicit */
      ASSERT_ALWAYS (cy == 0);

      qn -= in;
      if (qn == 0)
	break;

      /* Compute the product of the quotient block and the divisor D, to be
	 subtracted from the partial remainder combined with new limbs from the
	 dividend N.  We only really need the low dn limbs.  */

      if (BELOW_THRESHOLD (in, MUL_TO_MULMOD_BNM1_FOR_2NXN_THRESHOLD))
	mpn_mul (tp, dp, dn, qp, in);		/* dn+in limbs, high 'in' cancels */
      else
	{
	  tn = mpn_mulmod_bnm1_next_size (dn + 1);
	  mpn_mulmod_bnm1 (tp, tn, dp, dn, qp, in, scratch_out);
	  wn = dn + in - tn;			/* number of wrapped limbs */
	  if (wn > 0)
	    {
	      cy = mpn_sub_n (tp, tp, rp + dn - wn, wn);
	      cy = mpn_sub_1 (tp + wn, tp + wn, tn - wn, cy);
	      cx = mpn_cmp (rp + dn - in, tp + dn, tn - dn) < 0;
	      ASSERT_ALWAYS (cx >= cy);
	      mpn_incr_u (tp, cx - cy);
	    }
	}

      r = rp[dn - in] - tp[dn];

      /* Subtract the product from the partial remainder combined with new
	 limbs from the dividend N, generating a new partial remainder R.  */
      if (dn != in)
	{
	  cy = mpn_sub_n (tp, np, tp, in);	/* get next 'in' limbs from N */
	  cy = mpn_sub_nc (tp + in, rp, tp + in, dn - in, cy);
	  MPN_COPY (rp, tp, dn);		/* FIXME: try to avoid this */
	}
      else
	{
	  cy = mpn_sub_n (rp, np, tp, in);	/* get next 'in' limbs from N */
	}

      STAT (int i; int err = 0;
	    static int errarr[5]; static int err_rec; static int tot);

      /* Check the remainder R and adjust the quotient as needed.  */
      r -= cy;
      while (r != 0)
	{
	  /* We loop 0 times with about 69% probability, 1 time with about 31%
	     probability, 2 times with about 0.6% probability, if inverse is
	     computed as recommended.  */
	  mpn_incr_u (qp, 1);
	  cy = mpn_sub_n (rp, rp, dp, dn);
	  r -= cy;
	  STAT (err++);
	}
      if (mpn_cmp (rp, dp, dn) >= 0)
	{
	  /* This is executed with about 76% probability.  */
	  mpn_incr_u (qp, 1);
	  cy = mpn_sub_n (rp, rp, dp, dn);
	  STAT (err++);
	}

      STAT (
	    tot++;
	    errarr[err]++;
	    if (err > err_rec)
	      err_rec = err;
	    if (tot % 0x10000 == 0)
	      {
		for (i = 0; i <= err_rec; i++)
		  printf ("  %d(%.1f%%)", errarr[i], 100.0*errarr[i]/tot);
		printf ("\n");
	      }
	    );
    }

  /* FIXME: We should perhaps be somewhat more elegant in our rounding of the
     quotient.  For now, just make sure the returned quotient is >= the real
     quotient; add 3 with saturating arithmetic.  */
  qn = nn - dn;
  cy += mpn_add_1 (qp, qp, qn, 3);
  if (cy != 0)
    {
      if (qh != 0)
	{
	  /* Return a quotient of just 1-bits, with qh set.  */
	  mp_size_t i;
	  for (i = 0; i < qn; i++)
	    qp[i] = GMP_NUMB_MAX;
	}
      else
	{
	  /* Propagate carry into qh.  */
	  qh = 1;
	}
    }

  return qh;
}

/* In case k=0 (automatic choice), we distinguish 3 cases:
   (a) dn < qn:         in = ceil(qn / ceil(qn/dn))
   (b) dn/3 < qn <= dn: in = ceil(qn / 2)
   (c) qn < dn/3:       in = qn
   In all cases we have in <= dn.
 */
mp_size_t
mpn_mu_divappr_q_choose_in (mp_size_t qn, mp_size_t dn, int k)
{
  mp_size_t in;

  if (k == 0)
    {
      mp_size_t b;
      if (qn > dn)
	{
	  /* Compute an inverse size that is a nice partition of the quotient.  */
	  b = (qn - 1) / dn + 1;	/* ceil(qn/dn), number of blocks */
	  in = (qn - 1) / b + 1;	/* ceil(qn/b) = ceil(qn / ceil(qn/dn)) */
	}
      else if (3 * qn > dn)
	{
	  in = (qn - 1) / 2 + 1;	/* b = 2 */
	}
      else
	{
	  in = (qn - 1) / 1 + 1;	/* b = 1 */
	}
    }
  else
    {
      mp_size_t xn;
      xn = MIN (dn, qn);
      in = (xn - 1) / k + 1;
    }

  return in;
}

mp_size_t
mpn_mu_divappr_q_itch (mp_size_t nn, mp_size_t dn, int mua_k)
{
  mp_size_t qn, in, itch_local, itch_out, itch_invapp;

  qn = nn - dn;
  if (qn + 1 < dn)
    {
      dn = qn + 1;
    }
  in = mpn_mu_divappr_q_choose_in (qn, dn, mua_k);

  itch_local = mpn_mulmod_bnm1_next_size (dn + 1);
  itch_out = mpn_mulmod_bnm1_itch (itch_local, dn, in);
  itch_invapp = mpn_invertappr_itch (in + 1) + in + 2; /* 3in + 4 */

  ASSERT (dn + itch_local + itch_out >= itch_invapp);
  return in + MAX (dn + itch_local + itch_out, itch_invapp);
}
