/* mpn_gcdext -- Extended Greatest Common Divisor.

Copyright 1996, 1998, 2000-2005, 2008, 2009 Free Software Foundation, Inc.

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

#ifndef GCDEXT_1_USE_BINARY
#define GCDEXT_1_USE_BINARY 0
#endif

#ifndef GCDEXT_1_BINARY_METHOD
#define GCDEXT_1_BINARY_METHOD 2
#endif

#ifndef USE_ZEROTAB
#define USE_ZEROTAB 1
#endif

#if GCDEXT_1_USE_BINARY

#if USE_ZEROTAB
static unsigned char zerotab[0x40] = {
  6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
  4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
  5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
  4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0
};
#endif

mp_limb_t
mpn_gcdext_1 (mp_limb_signed_t *sp, mp_limb_signed_t *tp,
	      mp_limb_t u, mp_limb_t v)
{
  /* Maintain

     U = t1 u + t0 v
     V = s1 u + s0 v

     where U, V are the inputs (without any shared power of two),
     and the matrix has determinant ± 2^{shift}.
  */
  mp_limb_t s0 = 1;
  mp_limb_t t0 = 0;
  mp_limb_t s1 = 0;
  mp_limb_t t1 = 1;
  mp_limb_t ug;
  mp_limb_t vg;
  mp_limb_t ugh;
  mp_limb_t vgh;
  unsigned zero_bits;
  unsigned shift;
  unsigned i;
#if GCDEXT_1_BINARY_METHOD == 2
  mp_limb_t det_sign;
#endif

  ASSERT (u > 0);
  ASSERT (v > 0);

  count_trailing_zeros (zero_bits, u | v);
  u >>= zero_bits;
  v >>= zero_bits;

  if ((u & 1) == 0)
    {
      count_trailing_zeros (shift, u);
      u >>= shift;
      t1 <<= shift;
    }
  else if ((v & 1) == 0)
    {
      count_trailing_zeros (shift, v);
      v >>= shift;
      s0 <<= shift;
    }
  else
    shift = 0;

#if GCDEXT_1_BINARY_METHOD == 1
  while (u != v)
    {
      unsigned count;
      if (u > v)
	{
	  u -= v;
#if USE_ZEROTAB
	  count = zerotab [u & 0x3f];
	  u >>= count;
	  if (UNLIKELY (count == 6))
	    {
	      unsigned c;
	      do
		{
		  c = zerotab[u & 0x3f];
		  u >>= c;
		  count += c;
		}
	      while (c == 6);
	    }
#else
	  count_trailing_zeros (count, u);
	  u >>= count;
#endif
	  t0 += t1; t1 <<= count;
	  s0 += s1; s1 <<= count;
	}
      else
	{
	  v -= u;
#if USE_ZEROTAB
	  count = zerotab [v & 0x3f];
	  v >>= count;
	  if (UNLIKELY (count == 6))
	    {
	      unsigned c;
	      do
		{
		  c = zerotab[v & 0x3f];
		  v >>= c;
		  count += c;
		}
	      while (c == 6);
	    }
#else
	  count_trailing_zeros (count, v);
	  v >>= count;
#endif
	  t1 += t0; t0 <<= count;
	  s1 += s0; s0 <<= count;
	}
      shift += count;
    }
#else
# if GCDEXT_1_BINARY_METHOD == 2
  u >>= 1;
  v >>= 1;

  det_sign = 0;

  while (u != v)
    {
      unsigned count;
      mp_limb_t d =  u - v;
      mp_limb_t vgtu = LIMB_HIGHBIT_TO_MASK (d);
      mp_limb_t sx;
      mp_limb_t tx;

      /* When v <= u (vgtu == 0), the updates are:

	   (u; v)   <-- ( (u - v) >> count; v)    (det = +(1<<count) for corr. M factor)
	   (t1, t0) <-- (t1 << count, t0 + t1)

	 and when v > 0, the updates are

	   (u; v)   <-- ( (v - u) >> count; u)    (det = -(1<<count))
	   (t1, t0) <-- (t0 << count, t0 + t1)

	 and similarly for s1, s0
      */

      /* v <-- min (u, v) */
      v += (vgtu & d);

      /* u <-- |u - v| */
      u = (d ^ vgtu) - vgtu;

      /* Number of trailing zeros is the same no matter if we look at
       * d or u, but using d gives more parallelism. */
#if USE_ZEROTAB
      count = zerotab[d & 0x3f];
      if (UNLIKELY (count == 6))
	{
	  unsigned c = 6;
	  do
	    {
	      d >>= c;
	      c = zerotab[d & 0x3f];
	      count += c;
	    }
	  while (c == 6);
	}
#else
      count_trailing_zeros (count, d);
#endif
      det_sign ^= vgtu;

      tx = vgtu & (t0 - t1);
      sx = vgtu & (s0 - s1);
      t0 += t1;
      s0 += s1;
      t1 += tx;
      s1 += sx;

      count++;
      u >>= count;
      t1 <<= count;
      s1 <<= count;
      shift += count;
    }
  u = (u << 1) + 1;
# else /* GCDEXT_1_BINARY_METHOD == 2 */
#  error Unknown GCDEXT_1_BINARY_METHOD
# endif
#endif

  /* Now u = v = g = gcd (u,v). Compute U/g and V/g */
  ug = t0 + t1;
  vg = s0 + s1;

  ugh = ug/2 + (ug & 1);
  vgh = vg/2 + (vg & 1);

  /* Now ±2^{shift} g = s0 U - t0 V. Get rid of the power of two, using
     s0 U - t0 V = (s0 + V/g) U - (t0 + U/g) V. */
  for (i = 0; i < shift; i++)
    {
      mp_limb_t mask = - ( (s0 | t0) & 1);

      s0 /= 2;
      t0 /= 2;
      s0 += mask & vgh;
      t0 += mask & ugh;
    }
  /* FIXME: Try simplifying this condition. */
  if ( (s0 > 1 && 2*s0 >= vg) || (t0 > 1 && 2*t0 >= ug) )
    {
      s0 -= vg;
      t0 -= ug;
    }
#if GCDEXT_1_BINARY_METHOD == 2
  /* Conditional negation. */
  s0 = (s0 ^ det_sign) - det_sign;
  t0 = (t0 ^ det_sign) - det_sign;
#endif
  *sp = s0;
  *tp = -t0;

  return u << zero_bits;
}

#else /* !GCDEXT_1_USE_BINARY */


/* FIXME: Takes two single-word limbs. It could be extended to a
 * function that accepts a bignum for the first input, and only
 * returns the first co-factor. */

mp_limb_t
mpn_gcdext_1 (mp_limb_signed_t *up, mp_limb_signed_t *vp,
	      mp_limb_t a, mp_limb_t b)
{
  /* Maintain

     a =  u0 A + v0 B
     b =  u1 A + v1 B

     where A, B are the original inputs.
  */
  mp_limb_signed_t u0 = 1;
  mp_limb_signed_t v0 = 0;
  mp_limb_signed_t u1 = 0;
  mp_limb_signed_t v1 = 1;

  ASSERT (a > 0);
  ASSERT (b > 0);

  if (a < b)
    goto divide_by_b;

  for (;;)
    {
      mp_limb_t q;

      q = a / b;
      a -= q * b;

      if (a == 0)
	{
	  *up = u1;
	  *vp = v1;
	  return b;
	}
      u0 -= q * u1;
      v0 -= q * v1;

    divide_by_b:
      q = b / a;
      b -= q * a;

      if (b == 0)
	{
	  *up = u0;
	  *vp = v0;
	  return a;
	}
      u1 -= q * u0;
      v1 -= q * v0;
    }
}
#endif /* !GCDEXT_1_USE_BINARY */
