/* mpn_mod_34lsub1 -- remainder modulo 2^(GMP_NUMB_BITS*3/4)-1.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2000-2002 Free Software Foundation, Inc.

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


/* Calculate a remainder from {p,n} divided by 2^(GMP_NUMB_BITS*3/4)-1.
   The remainder is not fully reduced, it's any limb value congruent to
   {p,n} modulo that divisor.

   This implementation is only correct when GMP_NUMB_BITS is a multiple of
   4.

   FIXME: If GMP_NAIL_BITS is some silly big value during development then
   it's possible the carry accumulators c0,c1,c2 could overflow.

   General notes:

   The basic idea is to use a set of N accumulators (N=3 in this case) to
   effectively get a remainder mod 2^(GMP_NUMB_BITS*N)-1 followed at the end
   by a reduction to GMP_NUMB_BITS*N/M bits (M=4 in this case) for a
   remainder mod 2^(GMP_NUMB_BITS*N/M)-1.  N and M are chosen to give a good
   set of small prime factors in 2^(GMP_NUMB_BITS*N/M)-1.

   N=3 M=4 suits GMP_NUMB_BITS==32 and GMP_NUMB_BITS==64 quite well, giving
   a few more primes than a single accumulator N=1 does, and for no extra
   cost (assuming the processor has a decent number of registers).

   For strange nailified values of GMP_NUMB_BITS the idea would be to look
   for what N and M give good primes.  With GMP_NUMB_BITS not a power of 2
   the choices for M may be opened up a bit.  But such things are probably
   best done in separate code, not grafted on here.  */

#if GMP_NUMB_BITS % 4 == 0

#define B1  (GMP_NUMB_BITS / 4)
#define B2  (B1 * 2)
#define B3  (B1 * 3)

#define M1  ((CNST_LIMB(1) << B1) - 1)
#define M2  ((CNST_LIMB(1) << B2) - 1)
#define M3  ((CNST_LIMB(1) << B3) - 1)

#define LOW0(n)      ((n) & M3)
#define HIGH0(n)     ((n) >> B3)

#define LOW1(n)      (((n) & M2) << B1)
#define HIGH1(n)     ((n) >> B2)

#define LOW2(n)      (((n) & M1) << B2)
#define HIGH2(n)     ((n) >> B1)

#define PARTS0(n)    (LOW0(n) + HIGH0(n))
#define PARTS1(n)    (LOW1(n) + HIGH1(n))
#define PARTS2(n)    (LOW2(n) + HIGH2(n))

#define ADD(c,a,val)                    \
  do {                                  \
    mp_limb_t  new_c;                   \
    ADDC_LIMB (new_c, a, a, val);       \
    (c) += new_c;                       \
  } while (0)

mp_limb_t
mpn_mod_34lsub1 (mp_srcptr p, mp_size_t n)
{
  mp_limb_t  c0 = 0;
  mp_limb_t  c1 = 0;
  mp_limb_t  c2 = 0;
  mp_limb_t  a0, a1, a2;

  ASSERT (n >= 1);
  ASSERT (n/3 < GMP_NUMB_MAX);

  a0 = a1 = a2 = 0;
  c0 = c1 = c2 = 0;

  while ((n -= 3) >= 0)
    {
      ADD (c0, a0, p[0]);
      ADD (c1, a1, p[1]);
      ADD (c2, a2, p[2]);
      p += 3;
    }

  if (n != -3)
    {
      ADD (c0, a0, p[0]);
      if (n != -2)
	ADD (c1, a1, p[1]);
    }

  return
    PARTS0 (a0) + PARTS1 (a1) + PARTS2 (a2)
    + PARTS1 (c0) + PARTS2 (c1) + PARTS0 (c2);
}

#endif
