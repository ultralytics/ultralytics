/* double mpq_get_d (mpq_t src) -- mpq to double, rounding towards zero.

Copyright 1995, 1996, 2001-2005 Free Software Foundation, Inc.

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


/* All that's needed is to get the high 53 bits of the quotient num/den,
   rounded towards zero.  More than 53 bits is fine, any excess is ignored
   by mpn_get_d.

   N_QLIMBS is how many quotient limbs we need to satisfy the mantissa of a
   double, assuming the highest of those limbs is non-zero.  The target
   qsize for mpn_tdiv_qr is then 1 more than this, since that function may
   give a zero in the high limb (and non-zero in the second highest).

   The use of 8*sizeof(double) in N_QLIMBS is an overestimate of the
   mantissa bits, but it gets the same result as the true value (53 or 48 or
   whatever) when rounded up to a multiple of GMP_NUMB_BITS, for non-nails.

   Enhancements:

   Use the true mantissa size in the N_QLIMBS formula, to save a divide step
   in nails.

   Examine the high limbs of num and den to see if the highest 1 bit of the
   quotient will fall high enough that just N_QLIMBS-1 limbs is enough to
   get the necessary bits, thereby saving a division step.

   Bit shift either num or den to arrange for the above condition on the
   high 1 bit of the quotient, to save a division step always.  A shift to
   save a division step is definitely worthwhile with mpn_tdiv_qr, though we
   may want to reassess this on big num/den when a quotient-only division
   exists.

   Maybe we could estimate the final exponent using nsize-dsize (and
   possibly the high limbs of num and den), so as to detect overflow and
   return infinity or zero quickly.  Overflow is never very helpful to an
   application, and can therefore probably be regarded as abnormal, but we
   may still like to optimize it if the conditions are easy.  (This would
   only be for float formats we know, unknown formats are not important and
   can be left to mpn_get_d.)

   Future:

   If/when mpn_tdiv_qr supports its qxn parameter we can use that instead of
   padding n with zeros in temporary space.

   If/when a quotient-only division exists it can be used here immediately.
   remp is only to satisfy mpn_tdiv_qr, the remainder is not used.

   Alternatives:

   An alternative algorithm, that may be faster:
   0. Let n be somewhat larger than the number of significant bits in a double.
   1. Extract the most significant n bits of the denominator, and an equal
      number of bits from the numerator.
   2. Interpret the extracted numbers as integers, call them a and b
      respectively, and develop n bits of the fractions ((a + 1) / b) and
      (a / (b + 1)) using mpn_divrem.
   3. If the computed values are identical UP TO THE POSITION WE CARE ABOUT,
      we are done.  If they are different, repeat the algorithm from step 1,
      but first let n = n * 2.
   4. If we end up using all bits from the numerator and denominator, fall
      back to a plain division.
   5. Just to make life harder, The computation of a + 1 and b + 1 above
      might give carry-out...  Needs special handling.  It might work to
      subtract 1 in both cases instead.

   Not certain if this approach would be faster than a quotient-only
   division.  Presumably such optimizations are the sort of thing we would
   like to have helping everywhere that uses a quotient-only division. */

double
mpq_get_d (mpq_srcptr src)
{
  double res;
  mp_srcptr np, dp;
  mp_ptr remp, tp;
  mp_size_t nsize = SIZ(NUM(src));
  mp_size_t dsize = SIZ(DEN(src));
  mp_size_t qsize, prospective_qsize, zeros, chop, tsize;
  mp_size_t sign_quotient = nsize;
  long exp;
#define N_QLIMBS (1 + (sizeof (double) + GMP_LIMB_BYTES-1) / GMP_LIMB_BYTES)
  mp_limb_t qarr[N_QLIMBS + 1];
  mp_ptr qp = qarr;
  TMP_DECL;

  ASSERT (dsize > 0);    /* canonical src */

  /* mpn_get_d below requires a non-zero operand */
  if (UNLIKELY (nsize == 0))
    return 0.0;

  TMP_MARK;
  nsize = ABS (nsize);
  dsize = ABS (dsize);
  np = PTR(NUM(src));
  dp = PTR(DEN(src));

  prospective_qsize = nsize - dsize + 1;   /* from using given n,d */
  qsize = N_QLIMBS + 1;                    /* desired qsize */

  zeros = qsize - prospective_qsize;       /* padding n to get qsize */
  exp = (long) -zeros * GMP_NUMB_BITS;     /* relative to low of qp */

  chop = MAX (-zeros, 0);                  /* negative zeros means shorten n */
  np += chop;
  nsize -= chop;
  zeros += chop;                           /* now zeros >= 0 */

  tsize = nsize + zeros;                   /* size for possible copy of n */

  if (WANT_TMP_DEBUG)
    {
      /* separate blocks, for malloc debugging */
      remp = TMP_ALLOC_LIMBS (dsize);
      tp = (zeros > 0 ? TMP_ALLOC_LIMBS (tsize) : NULL);
    }
  else
    {
      /* one block with conditionalized size, for efficiency */
      remp = TMP_ALLOC_LIMBS (dsize + (zeros > 0 ? tsize : 0));
      tp = remp + dsize;
    }

  /* zero extend n into temporary space, if necessary */
  if (zeros > 0)
    {
      MPN_ZERO (tp, zeros);
      MPN_COPY (tp+zeros, np, nsize);
      np = tp;
      nsize = tsize;
    }

  ASSERT (qsize == nsize - dsize + 1);
  mpn_tdiv_qr (qp, remp, (mp_size_t) 0, np, nsize, dp, dsize);

  /* strip possible zero high limb */
  qsize -= (qp[qsize-1] == 0);

  res = mpn_get_d (qp, qsize, sign_quotient, exp);
  TMP_FREE;
  return res;
}
