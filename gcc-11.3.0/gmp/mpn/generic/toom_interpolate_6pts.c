/* mpn_toom_interpolate_6pts -- Interpolate for toom43, 52

   Contributed to the GNU project by Marco Bodrato.

   THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.  IT IS ONLY
   SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT IT WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2009, 2010, 2012 Free Software Foundation, Inc.

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

/* For odd divisors, mpn_divexact_1 works fine with two's complement. */
#ifndef mpn_divexact_by3
#if HAVE_NATIVE_mpn_pi1_bdiv_q_1 && MODLIMB_INVERSE_3
#define mpn_divexact_by3(dst,src,size) mpn_pi1_bdiv_q_1(dst,src,size,3,MODLIMB_INVERSE_3,0)
#else
#define mpn_divexact_by3(dst,src,size) mpn_divexact_1(dst,src,size,3)
#endif
#endif

/* Interpolation for Toom-3.5, using the evaluation points: infinity,
   1, -1, 2, -2. More precisely, we want to compute
   f(2^(GMP_NUMB_BITS * n)) for a polynomial f of degree 5, given the
   six values

     w5 = f(0),
     w4 = f(-1),
     w3 = f(1)
     w2 = f(-2),
     w1 = f(2),
     w0 = limit at infinity of f(x) / x^5,

   The result is stored in {pp, 5*n + w0n}. At entry, w5 is stored at
   {pp, 2n}, w3 is stored at {pp + 2n, 2n+1}, and w0 is stored at
   {pp + 5n, w0n}. The other values are 2n + 1 limbs each (with most
   significant limbs small). f(-1) and f(-2) may be negative, signs
   determined by the flag bits. All intermediate results are positive.
   Inputs are destroyed.

   Interpolation sequence was taken from the paper: "Integer and
   Polynomial Multiplication: Towards Optimal Toom-Cook Matrices".
   Some slight variations were introduced: adaptation to "gmp
   instruction set", and a final saving of an operation by interlacing
   interpolation and recomposition phases.
*/

void
mpn_toom_interpolate_6pts (mp_ptr pp, mp_size_t n, enum toom6_flags flags,
			   mp_ptr w4, mp_ptr w2, mp_ptr w1,
			   mp_size_t w0n)
{
  mp_limb_t cy;
  /* cy6 can be stored in w1[2*n], cy4 in w4[0], embankment in w2[0] */
  mp_limb_t cy4, cy6, embankment;

  ASSERT( n > 0 );
  ASSERT( 2*n >= w0n && w0n > 0 );

#define w5  pp					/* 2n   */
#define w3  (pp + 2 * n)			/* 2n+1 */
#define w0  (pp + 5 * n)			/* w0n  */

  /* Interpolate with sequence:
     W2 =(W1 - W2)>>2
     W1 =(W1 - W5)>>1
     W1 =(W1 - W2)>>1
     W4 =(W3 - W4)>>1
     W2 =(W2 - W4)/3
     W3 = W3 - W4 - W5
     W1 =(W1 - W3)/3
     // Last steps are mixed with recomposition...
     W2 = W2 - W0<<2
     W4 = W4 - W2
     W3 = W3 - W1
     W2 = W2 - W0
  */

  /* W2 =(W1 - W2)>>2 */
  if (flags & toom6_vm2_neg)
    mpn_add_n (w2, w1, w2, 2 * n + 1);
  else
    mpn_sub_n (w2, w1, w2, 2 * n + 1);
  mpn_rshift (w2, w2, 2 * n + 1, 2);

  /* W1 =(W1 - W5)>>1 */
  w1[2*n] -= mpn_sub_n (w1, w1, w5, 2*n);
  mpn_rshift (w1, w1, 2 * n + 1, 1);

  /* W1 =(W1 - W2)>>1 */
#if HAVE_NATIVE_mpn_rsh1sub_n
  mpn_rsh1sub_n (w1, w1, w2, 2 * n + 1);
#else
  mpn_sub_n (w1, w1, w2, 2 * n + 1);
  mpn_rshift (w1, w1, 2 * n + 1, 1);
#endif

  /* W4 =(W3 - W4)>>1 */
  if (flags & toom6_vm1_neg)
    {
#if HAVE_NATIVE_mpn_rsh1add_n
      mpn_rsh1add_n (w4, w3, w4, 2 * n + 1);
#else
      mpn_add_n (w4, w3, w4, 2 * n + 1);
      mpn_rshift (w4, w4, 2 * n + 1, 1);
#endif
    }
  else
    {
#if HAVE_NATIVE_mpn_rsh1sub_n
      mpn_rsh1sub_n (w4, w3, w4, 2 * n + 1);
#else
      mpn_sub_n (w4, w3, w4, 2 * n + 1);
      mpn_rshift (w4, w4, 2 * n + 1, 1);
#endif
    }

  /* W2 =(W2 - W4)/3 */
  mpn_sub_n (w2, w2, w4, 2 * n + 1);
  mpn_divexact_by3 (w2, w2, 2 * n + 1);

  /* W3 = W3 - W4 - W5 */
  mpn_sub_n (w3, w3, w4, 2 * n + 1);
  w3[2 * n] -= mpn_sub_n (w3, w3, w5, 2 * n);

  /* W1 =(W1 - W3)/3 */
  mpn_sub_n (w1, w1, w3, 2 * n + 1);
  mpn_divexact_by3 (w1, w1, 2 * n + 1);

  /*
    [1 0 0 0 0 0;
     0 1 0 0 0 0;
     1 0 1 0 0 0;
     0 1 0 1 0 0;
     1 0 1 0 1 0;
     0 0 0 0 0 1]

    pp[] prior to operations:
     |_H w0__|_L w0__|______||_H w3__|_L w3__|_H w5__|_L w5__|

    summation scheme for remaining operations:
     |______________5|n_____4|n_____3|n_____2|n______|n______|pp
     |_H w0__|_L w0__|______||_H w3__|_L w3__|_H w5__|_L w5__|
				    || H w4  | L w4  |
		    || H w2  | L w2  |
	    || H w1  | L w1  |
			    ||-H w1  |-L w1  |
		     |-H w0  |-L w0 ||-H w2  |-L w2  |
  */
  cy = mpn_add_n (pp + n, pp + n, w4, 2 * n + 1);
  MPN_INCR_U (pp + 3 * n + 1, n, cy);

  /* W2 -= W0<<2 */
#if HAVE_NATIVE_mpn_sublsh_n || HAVE_NATIVE_mpn_sublsh2_n_ip1
#if HAVE_NATIVE_mpn_sublsh2_n_ip1
  cy = mpn_sublsh2_n_ip1 (w2, w0, w0n);
#else
  cy = mpn_sublsh_n (w2, w2, w0, w0n, 2);
#endif
#else
  /* {W4,2*n+1} is now free and can be overwritten. */
  cy = mpn_lshift(w4, w0, w0n, 2);
  cy+= mpn_sub_n(w2, w2, w4, w0n);
#endif
  MPN_DECR_U (w2 + w0n, 2 * n + 1 - w0n, cy);

  /* W4L = W4L - W2L */
  cy = mpn_sub_n (pp + n, pp + n, w2, n);
  MPN_DECR_U (w3, 2 * n + 1, cy);

  /* W3H = W3H + W2L */
  cy4 = w3[2 * n] + mpn_add_n (pp + 3 * n, pp + 3 * n, w2, n);
  /* W1L + W2H */
  cy = w2[2 * n] + mpn_add_n (pp + 4 * n, w1, w2 + n, n);
  MPN_INCR_U (w1 + n, n + 1, cy);

  /* W0 = W0 + W1H */
  if (LIKELY (w0n > n))
    cy6 = w1[2 * n] + mpn_add_n (w0, w0, w1 + n, n);
  else
    cy6 = mpn_add_n (w0, w0, w1 + n, w0n);

  /*
    summation scheme for the next operation:
     |...____5|n_____4|n_____3|n_____2|n______|n______|pp
     |...w0___|_w1_w2_|_H w3__|_L w3__|_H w5__|_L w5__|
		     ...-w0___|-w1_w2 |
  */
  /* if(LIKELY(w0n>n)) the two operands below DO overlap! */
  cy = mpn_sub_n (pp + 2 * n, pp + 2 * n, pp + 4 * n, n + w0n);

  /* embankment is a "dirty trick" to avoid carry/borrow propagation
     beyond allocated memory */
  embankment = w0[w0n - 1] - 1;
  w0[w0n - 1] = 1;
  if (LIKELY (w0n > n)) {
    if (cy4 > cy6)
      MPN_INCR_U (pp + 4 * n, w0n + n, cy4 - cy6);
    else
      MPN_DECR_U (pp + 4 * n, w0n + n, cy6 - cy4);
    MPN_DECR_U (pp + 3 * n + w0n, 2 * n, cy);
    MPN_INCR_U (w0 + n, w0n - n, cy6);
  } else {
    MPN_INCR_U (pp + 4 * n, w0n + n, cy4);
    MPN_DECR_U (pp + 3 * n + w0n, 2 * n, cy + cy6);
  }
  w0[w0n - 1] += embankment;

#undef w5
#undef w3
#undef w0

}
