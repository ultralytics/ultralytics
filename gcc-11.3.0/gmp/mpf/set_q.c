/* mpf_set_q (mpf_t rop, mpq_t op) -- Convert the rational op to the float rop.

Copyright 1996, 1999, 2001, 2002, 2004, 2005 Free Software Foundation, Inc.

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

#include <stdio.h>  /* for NULL */
#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"


/* As usual the aim is to produce PREC(r) limbs, with the high non-zero.
   The basic mpn_tdiv_qr produces a quotient of nsize-dsize+1 limbs, with
   either the high or second highest limb non-zero.  We arrange for
   nsize-dsize+1 to equal prec+1, hence giving either prec or prec+1 result
   limbs at PTR(r).

   nsize-dsize+1 == prec+1 is achieved by adjusting num(q), either dropping
   low limbs if it's too big, or padding with low zeros if it's too small.
   The full given den(q) is always used.

   We cannot truncate den(q), because even when it's much bigger than prec
   the last limbs can still influence the final quotient.  Often they don't,
   but we leave optimization of that to a prospective quotient-only mpn
   division.

   Not done:

   If den(q) is a power of 2 then we may end up with low zero limbs on the
   result.  But nothing is done about this, since it should be unlikely on
   random data, and can be left to an application to call mpf_div_2exp if it
   might occur with any frequency.

   Enhancements:

   The high quotient limb is non-zero when high{np,dsize} >= {dp,dsize}.  We
   could make that comparison and use qsize==prec instead of qsize==prec+1,
   to save one limb in the division.

   Future:

   If/when mpn_tdiv_qr supports its qxn parameter we can use that instead of
   padding n with zeros in temporary space.

   If/when a quotient-only division exists it can be used here immediately.
   remp is only to satisfy mpn_tdiv_qr, the remainder is not used.  */

void
mpf_set_q (mpf_t r, mpq_srcptr q)
{
  mp_srcptr np, dp;
  mp_size_t prec, nsize, dsize, qsize, prospective_qsize, tsize, zeros;
  mp_size_t sign_quotient, high_zero;
  mp_ptr qp, tp, remp;
  mp_exp_t exp;
  TMP_DECL;

  ASSERT (SIZ(&q->_mp_den) > 0);  /* canonical q */

  nsize = SIZ (&q->_mp_num);
  dsize = SIZ (&q->_mp_den);

  if (UNLIKELY (nsize == 0))
    {
      SIZ (r) = 0;
      EXP (r) = 0;
      return;
    }

  TMP_MARK;

  prec = PREC (r);
  qp = PTR (r);

  sign_quotient = nsize;
  nsize = ABS (nsize);
  np = PTR (&q->_mp_num);
  dp = PTR (&q->_mp_den);

  prospective_qsize = nsize - dsize + 1;  /* q from using given n,d sizes */
  exp = prospective_qsize;                /* ie. number of integer limbs */
  qsize = prec + 1;                       /* desired q */

  zeros = qsize - prospective_qsize;   /* n zeros to get desired qsize */
  tsize = nsize + zeros;               /* possible copy of n */

  if (WANT_TMP_DEBUG)
    {
      /* separate alloc blocks, for malloc debugging */
      remp = TMP_ALLOC_LIMBS (dsize);
      tp = NULL;
      if (zeros > 0)
        tp = TMP_ALLOC_LIMBS (tsize);
    }
  else
    {
      /* one alloc with a conditionalized size, for efficiency */
      mp_size_t size = dsize + (zeros > 0 ? tsize : 0);
      remp = TMP_ALLOC_LIMBS (size);
      tp = remp + dsize;
    }

  if (zeros > 0)
    {
      /* pad n with zeros into temporary space */
      MPN_ZERO (tp, zeros);
      MPN_COPY (tp+zeros, np, nsize);
      np = tp;
      nsize = tsize;
    }
  else
    {
      /* shorten n to get desired qsize */
      nsize += zeros;
      np -= zeros;
    }

  ASSERT (nsize-dsize+1 == qsize);
  mpn_tdiv_qr (qp, remp, (mp_size_t) 0, np, nsize, dp, dsize);

  /* strip possible zero high limb */
  high_zero = (qp[qsize-1] == 0);
  qsize -= high_zero;
  exp -= high_zero;

  EXP (r) = exp;
  SIZ (r) = sign_quotient >= 0 ? qsize : -qsize;

  TMP_FREE;
}
