/* mpn_toom_interpolate_7pts -- Interpolate for toom44, 53, 62.

   Contributed to the GNU project by Niels Möller.
   Improvements by Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2006, 2007, 2009, 2014 Free Software Foundation, Inc.

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

#define BINVERT_3 MODLIMB_INVERSE_3

#define BINVERT_9 \
  ((((GMP_NUMB_MAX / 9) << (6 - GMP_NUMB_BITS % 6)) * 8 & GMP_NUMB_MAX) | 0x39)

#define BINVERT_15 \
  ((((GMP_NUMB_MAX >> (GMP_NUMB_BITS % 4)) / 15) * 14 * 16 & GMP_NUMB_MAX) + 15)

/* For the various mpn_divexact_byN here, fall back to using either
   mpn_pi1_bdiv_q_1 or mpn_divexact_1.  The former has less overhead and is
   many faster if it is native.  For now, since mpn_divexact_1 is native on
   several platforms where mpn_pi1_bdiv_q_1 does not yet exist, do not use
   mpn_pi1_bdiv_q_1 unconditionally.  FIXME.  */

/* For odd divisors, mpn_divexact_1 works fine with two's complement. */
#ifndef mpn_divexact_by3
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1
#define mpn_divexact_by3(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,3,BINVERT_3,0)
#else
#define mpn_divexact_by3(dst,src,size) mpn_divexact_1(dst,src,size,3)
#endif
#endif

#ifndef mpn_divexact_by9
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1
#define mpn_divexact_by9(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,9,BINVERT_9,0)
#else
#define mpn_divexact_by9(dst,src,size) mpn_divexact_1(dst,src,size,9)
#endif
#endif

#ifndef mpn_divexact_by15
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1
#define mpn_divexact_by15(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,15,BINVERT_15,0)
#else
#define mpn_divexact_by15(dst,src,size) mpn_divexact_1(dst,src,size,15)
#endif
#endif

/* Interpolation for toom4, using the evaluation points 0, infinity,
   1, -1, 2, -2, 1/2. More precisely, we want to compute
   f(2^(GMP_NUMB_BITS * n)) for a polynomial f of degree 6, given the
   seven values

     w0 = f(0),
     w1 = f(-2),
     w2 = f(1),
     w3 = f(-1),
     w4 = f(2)
     w5 = 64 * f(1/2)
     w6 = limit at infinity of f(x) / x^6,

   The result is 6*n + w6n limbs. At entry, w0 is stored at {rp, 2n },
   w2 is stored at { rp + 2n, 2n+1 }, and w6 is stored at { rp + 6n,
   w6n }. The other values are 2n + 1 limbs each (with most
   significant limbs small). f(-1) and f(-1/2) may be negative, signs
   determined by the flag bits. Inputs are destroyed.

   Needs (2*n + 1) limbs of temporary storage.
*/

void
mpn_toom_interpolate_7pts (mp_ptr rp, mp_size_t n, enum toom7_flags flags,
			   mp_ptr w1, mp_ptr w3, mp_ptr w4, mp_ptr w5,
			   mp_size_t w6n, mp_ptr tp)
{
  mp_size_t m;
  mp_limb_t cy;

  m = 2*n + 1;
#define w0 rp
#define w2 (rp + 2*n)
#define w6 (rp + 6*n)

  ASSERT (w6n > 0);
  ASSERT (w6n <= 2*n);

  /* Using formulas similar to Marco Bodrato's

     W5 = W5 + W4
     W1 =(W4 - W1)/2
     W4 = W4 - W0
     W4 =(W4 - W1)/4 - W6*16
     W3 =(W2 - W3)/2
     W2 = W2 - W3

     W5 = W5 - W2*65      May be negative.
     W2 = W2 - W6 - W0
     W5 =(W5 + W2*45)/2   Now >= 0 again.
     W4 =(W4 - W2)/3
     W2 = W2 - W4

     W1 = W5 - W1         May be negative.
     W5 =(W5 - W3*8)/9
     W3 = W3 - W5
     W1 =(W1/15 + W5)/2   Now >= 0 again.
     W5 = W5 - W1

     where W0 = f(0), W1 = f(-2), W2 = f(1), W3 = f(-1),
	   W4 = f(2), W5 = f(1/2), W6 = f(oo),

     Note that most intermediate results are positive; the ones that
     may be negative are represented in two's complement. We must
     never shift right a value that may be negative, since that would
     invalidate the sign bit. On the other hand, divexact by odd
     numbers work fine with two's complement.
  */

  mpn_add_n (w5, w5, w4, m);
  if (flags & toom7_w1_neg)
    {
#ifdef HAVE_NATIVE_mpn_rsh1add_n
      mpn_rsh1add_n (w1, w1, w4, m);
#else
      mpn_add_n (w1, w1, w4, m);  ASSERT (!(w1[0] & 1));
      mpn_rshift (w1, w1, m, 1);
#endif
    }
  else
    {
#ifdef HAVE_NATIVE_mpn_rsh1sub_n
      mpn_rsh1sub_n (w1, w4, w1, m);
#else
      mpn_sub_n (w1, w4, w1, m);  ASSERT (!(w1[0] & 1));
      mpn_rshift (w1, w1, m, 1);
#endif
    }
  mpn_sub (w4, w4, m, w0, 2*n);
  mpn_sub_n (w4, w4, w1, m);  ASSERT (!(w4[0] & 3));
  mpn_rshift (w4, w4, m, 2); /* w4>=0 */

  tp[w6n] = mpn_lshift (tp, w6, w6n, 4);
  mpn_sub (w4, w4, m, tp, w6n+1);

  if (flags & toom7_w3_neg)
    {
#ifdef HAVE_NATIVE_mpn_rsh1add_n
      mpn_rsh1add_n (w3, w3, w2, m);
#else
      mpn_add_n (w3, w3, w2, m);  ASSERT (!(w3[0] & 1));
      mpn_rshift (w3, w3, m, 1);
#endif
    }
  else
    {
#ifdef HAVE_NATIVE_mpn_rsh1sub_n
      mpn_rsh1sub_n (w3, w2, w3, m);
#else
      mpn_sub_n (w3, w2, w3, m);  ASSERT (!(w3[0] & 1));
      mpn_rshift (w3, w3, m, 1);
#endif
    }

  mpn_sub_n (w2, w2, w3, m);

  mpn_submul_1 (w5, w2, m, 65);
  mpn_sub (w2, w2, m, w6, w6n);
  mpn_sub (w2, w2, m, w0, 2*n);

  mpn_addmul_1 (w5, w2, m, 45);  ASSERT (!(w5[0] & 1));
  mpn_rshift (w5, w5, m, 1);
  mpn_sub_n (w4, w4, w2, m);

  mpn_divexact_by3 (w4, w4, m);
  mpn_sub_n (w2, w2, w4, m);

  mpn_sub_n (w1, w5, w1, m);
  mpn_lshift (tp, w3, m, 3);
  mpn_sub_n (w5, w5, tp, m);
  mpn_divexact_by9 (w5, w5, m);
  mpn_sub_n (w3, w3, w5, m);

  mpn_divexact_by15 (w1, w1, m);
  mpn_add_n (w1, w1, w5, m);  ASSERT (!(w1[0] & 1));
  mpn_rshift (w1, w1, m, 1); /* w1>=0 now */
  mpn_sub_n (w5, w5, w1, m);

  /* These bounds are valid for the 4x4 polynomial product of toom44,
   * and they are conservative for toom53 and toom62. */
  ASSERT (w1[2*n] < 2);
  ASSERT (w2[2*n] < 3);
  ASSERT (w3[2*n] < 4);
  ASSERT (w4[2*n] < 3);
  ASSERT (w5[2*n] < 2);

  /* Addition chain. Note carries and the 2n'th limbs that need to be
   * added in.
   *
   * Special care is needed for w2[2n] and the corresponding carry,
   * since the "simple" way of adding it all together would overwrite
   * the limb at wp[2*n] and rp[4*n] (same location) with the sum of
   * the high half of w3 and the low half of w4.
   *
   *         7    6    5    4    3    2    1    0
   *    |    |    |    |    |    |    |    |    |
   *                  ||w3 (2n+1)|
   *             ||w4 (2n+1)|
   *        ||w5 (2n+1)|        ||w1 (2n+1)|
   *  + | w6 (w6n)|        ||w2 (2n+1)| w0 (2n) |  (share storage with r)
   *  -----------------------------------------------
   *  r |    |    |    |    |    |    |    |    |
   *        c7   c6   c5   c4   c3                 Carries to propagate
   */

  cy = mpn_add_n (rp + n, rp + n, w1, m);
  MPN_INCR_U (w2 + n + 1, n , cy);
  cy = mpn_add_n (rp + 3*n, rp + 3*n, w3, n);
  MPN_INCR_U (w3 + n, n + 1, w2[2*n] + cy);
  cy = mpn_add_n (rp + 4*n, w3 + n, w4, n);
  MPN_INCR_U (w4 + n, n + 1, w3[2*n] + cy);
  cy = mpn_add_n (rp + 5*n, w4 + n, w5, n);
  MPN_INCR_U (w5 + n, n + 1, w4[2*n] + cy);
  if (w6n > n + 1)
    {
      cy = mpn_add_n (rp + 6*n, rp + 6*n, w5 + n, n + 1);
      MPN_INCR_U (rp + 7*n + 1, w6n - n - 1, cy);
    }
  else
    {
      ASSERT_NOCARRY (mpn_add_n (rp + 6*n, rp + 6*n, w5 + n, w6n));
#if WANT_ASSERT
      {
	mp_size_t i;
	for (i = w6n; i <= n; i++)
	  ASSERT (w5[n + i] == 0);
      }
#endif
    }
}
