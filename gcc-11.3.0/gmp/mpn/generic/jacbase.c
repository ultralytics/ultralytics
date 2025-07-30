/* mpn_jacobi_base -- limb/limb Jacobi symbol with restricted arguments.

   THIS INTERFACE IS PRELIMINARY AND MIGHT DISAPPEAR OR BE SUBJECT TO
   INCOMPATIBLE CHANGES IN A FUTURE RELEASE OF GMP.

Copyright 1999-2002, 2010 Free Software Foundation, Inc.

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


/* Use the simple loop by default.  The generic count_trailing_zeros is not
   very fast, and the extra trickery of method 3 has proven to be less use
   than might have been though.  */
#ifndef JACOBI_BASE_METHOD
#define JACOBI_BASE_METHOD  2
#endif


/* Use count_trailing_zeros.  */
#if JACOBI_BASE_METHOD == 1
#define PROCESS_TWOS_ANY                                \
  {                                                     \
    mp_limb_t  twos;                                    \
    count_trailing_zeros (twos, a);                     \
    result_bit1 ^= JACOBI_TWOS_U_BIT1 (twos, b);        \
    a >>= twos;                                         \
  }
#define PROCESS_TWOS_EVEN  PROCESS_TWOS_ANY
#endif

/* Use a simple loop.  A disadvantage of this is that there's a branch on a
   50/50 chance of a 0 or 1 low bit.  */
#if JACOBI_BASE_METHOD == 2
#define PROCESS_TWOS_EVEN               \
  {                                     \
    int  two;                           \
    two = JACOBI_TWO_U_BIT1 (b);        \
    do                                  \
      {                                 \
	a >>= 1;                        \
	result_bit1 ^= two;             \
	ASSERT (a != 0);                \
      }                                 \
    while ((a & 1) == 0);               \
  }
#define PROCESS_TWOS_ANY        \
  if ((a & 1) == 0)             \
    PROCESS_TWOS_EVEN;
#endif

/* Process one bit arithmetically, then a simple loop.  This cuts the loop
   condition down to a 25/75 chance, which should branch predict better.
   The CPU will need a reasonable variable left shift.  */
#if JACOBI_BASE_METHOD == 3
#define PROCESS_TWOS_EVEN               \
  {                                     \
    int  two, mask, shift;              \
					\
    two = JACOBI_TWO_U_BIT1 (b);        \
    mask = (~a & 2);                    \
    a >>= 1;                            \
					\
    shift = (~a & 1);                   \
    a >>= shift;                        \
    result_bit1 ^= two ^ (two & mask);  \
					\
    while ((a & 1) == 0)                \
      {                                 \
	a >>= 1;                        \
	result_bit1 ^= two;             \
	ASSERT (a != 0);                \
      }                                 \
  }
#define PROCESS_TWOS_ANY                \
  {                                     \
    int  two, mask, shift;              \
					\
    two = JACOBI_TWO_U_BIT1 (b);        \
    shift = (~a & 1);                   \
    a >>= shift;                        \
					\
    mask = shift << 1;                  \
    result_bit1 ^= (two & mask);        \
					\
    while ((a & 1) == 0)                \
      {                                 \
	a >>= 1;                        \
	result_bit1 ^= two;             \
	ASSERT (a != 0);                \
      }                                 \
  }
#endif

#if JACOBI_BASE_METHOD < 4
/* Calculate the value of the Jacobi symbol (a/b) of two mp_limb_t's, but
   with a restricted range of inputs accepted, namely b>1, b odd.

   The initial result_bit1 is taken as a parameter for the convenience of
   mpz_kronecker_ui() et al.  The sign changes both here and in those
   routines accumulate nicely in bit 1, see the JACOBI macros.

   The return value here is the normal +1, 0, or -1.  Note that +1 and -1
   have bit 1 in the "BIT1" sense, which could be useful if the caller is
   accumulating it into some extended calculation.

   Duplicating the loop body to avoid the MP_LIMB_T_SWAP(a,b) would be
   possible, but a couple of tests suggest it's not a significant speedup,
   and may even be a slowdown, so what's here is good enough for now. */

int
mpn_jacobi_base (mp_limb_t a, mp_limb_t b, int result_bit1)
{
  ASSERT (b & 1);  /* b odd */
  ASSERT (b != 1);

  if (a == 0)
    return 0;

  PROCESS_TWOS_ANY;
  if (a == 1)
    goto done;

  if (a >= b)
    goto a_gt_b;

  for (;;)
    {
      result_bit1 ^= JACOBI_RECIP_UU_BIT1 (a, b);
      MP_LIMB_T_SWAP (a, b);

    a_gt_b:
      do
	{
	  /* working on (a/b), a,b odd, a>=b */
	  ASSERT (a & 1);
	  ASSERT (b & 1);
	  ASSERT (a >= b);

	  if ((a -= b) == 0)
	    return 0;

	  PROCESS_TWOS_EVEN;
	  if (a == 1)
	    goto done;
	}
      while (a >= b);
    }

 done:
  return JACOBI_BIT1_TO_PN (result_bit1);
}
#endif

#if JACOBI_BASE_METHOD == 4
/* Computes (a/b) for odd b > 1 and any a. The initial bit is taken as a
 * parameter. We have no need for the convention that the sign is in
 * bit 1, internally we use bit 0. */

/* FIXME: Could try table-based count_trailing_zeros. */
int
mpn_jacobi_base (mp_limb_t a, mp_limb_t b, int bit)
{
  int c;

  ASSERT (b & 1);
  ASSERT (b > 1);

  if (a == 0)
    /* This is the only line which depends on b > 1 */
    return 0;

  bit >>= 1;

  /* Below, we represent a and b shifted right so that the least
     significant one bit is implicit. */

  b >>= 1;

  count_trailing_zeros (c, a);
  bit ^= c & (b ^ (b >> 1));

  /* We may have c==GMP_LIMB_BITS-1, so we can't use a>>c+1. */
  a >>= c;
  a >>= 1;

  do
    {
      mp_limb_t t = a - b;
      mp_limb_t bgta = LIMB_HIGHBIT_TO_MASK (t);

      if (t == 0)
	return 0;

      /* If b > a, invoke reciprocity */
      bit ^= (bgta & a & b);

      /* b <-- min (a, b) */
      b += (bgta & t);

      /* a <-- |a - b| */
      a = (t ^ bgta) - bgta;

      /* Number of trailing zeros is the same no matter if we look at
       * t or a, but using t gives more parallelism. */
      count_trailing_zeros (c, t);
      c ++;
      /* (2/b) = -1 if b = 3 or 5 mod 8 */
      bit ^= c & (b ^ (b >> 1));
      a >>= c;
    }
  while (b > 0);

  return 1-2*(bit & 1);
}
#endif /* JACOBI_BASE_METHOD == 4 */
