/* mpz_n_pow_ui -- mpn raised to ulong.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2001, 2002, 2005, 2012 Free Software Foundation, Inc.

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


/* Change this to "#define TRACE(x) x" for some traces. */
#define TRACE(x)


/* Use this to test the mul_2 code on a CPU without a native version of that
   routine.  */
#if 0
#define mpn_mul_2  refmpn_mul_2
#define HAVE_NATIVE_mpn_mul_2  1
#endif


/* mpz_pow_ui and mpz_ui_pow_ui want to share almost all of this code.
   ui_pow_ui doesn't need the mpn_mul based powering loop or the tests on
   bsize==2 or >2, but separating that isn't easy because there's shared
   code both before and after (the size calculations and the powers of 2
   handling).

   Alternatives:

   It would work to just use the mpn_mul powering loop for 1 and 2 limb
   bases, but the current separate loop allows mul_1 and mul_2 to be done
   in-place, which might help cache locality a bit.  If mpn_mul was relaxed
   to allow source==dest when vn==1 or 2 then some pointer twiddling might
   let us get the same effect in one loop.

   The initial powering for bsize==1 into blimb or blimb:blimb_low doesn't
   form the biggest possible power of b that fits, only the biggest power of
   2 power, ie. b^(2^n).  It'd be possible to choose a bigger power, perhaps
   using mp_bases[b].big_base for small b, and thereby get better value
   from mpn_mul_1 or mpn_mul_2 in the bignum powering.  It's felt that doing
   so would be more complicated than it's worth, and could well end up being
   a slowdown for small e.  For big e on the other hand the algorithm is
   dominated by mpn_sqr so there wouldn't much of a saving.  The current
   code can be viewed as simply doing the first few steps of the powering in
   a single or double limb where possible.

   If r==b, and blow_twos==0, and r must be realloc'ed, then the temporary
   copy made of b is unnecessary.  We could just use the old alloc'ed block
   and free it at the end.  But arranging this seems like a lot more trouble
   than it's worth.  */


/* floor(sqrt(GMP_NUMB_MAX)), ie. the biggest value that can be squared in
   a limb without overflowing.
   FIXME: This formula is an underestimate when GMP_NUMB_BITS is odd. */

#define GMP_NUMB_HALFMAX  (((mp_limb_t) 1 << GMP_NUMB_BITS/2) - 1)


/* The following are for convenience, they update the size and check the
   alloc.  */

#define MPN_SQR(dst, alloc, src, size)          \
  do {                                          \
    ASSERT (2*(size) <= (alloc));               \
    mpn_sqr (dst, src, size);                   \
    (size) *= 2;                                \
    (size) -= ((dst)[(size)-1] == 0);           \
  } while (0)

#define MPN_MUL(dst, alloc, src, size, src2, size2)     \
  do {                                                  \
    mp_limb_t  cy;                                      \
    ASSERT ((size) + (size2) <= (alloc));               \
    cy = mpn_mul (dst, src, size, src2, size2);         \
    (size) += (size2) - (cy == 0);                      \
  } while (0)

#define MPN_MUL_2(ptr, size, alloc, mult)       \
  do {                                          \
    mp_limb_t  cy;                              \
    ASSERT ((size)+2 <= (alloc));               \
    cy = mpn_mul_2 (ptr, ptr, size, mult);      \
    (size)++;                                   \
    (ptr)[(size)] = cy;                         \
    (size) += (cy != 0);                        \
  } while (0)

#define MPN_MUL_1(ptr, size, alloc, limb)       \
  do {                                          \
    mp_limb_t  cy;                              \
    ASSERT ((size)+1 <= (alloc));               \
    cy = mpn_mul_1 (ptr, ptr, size, limb);      \
    (ptr)[size] = cy;                           \
    (size) += (cy != 0);                        \
  } while (0)

#define MPN_LSHIFT(ptr, size, alloc, shift)     \
  do {                                          \
    mp_limb_t  cy;                              \
    ASSERT ((size)+1 <= (alloc));               \
    cy = mpn_lshift (ptr, ptr, size, shift);    \
    (ptr)[size] = cy;                           \
    (size) += (cy != 0);                        \
  } while (0)

#define MPN_RSHIFT_OR_COPY(dst, src, size, shift)       \
  do {                                                  \
    if ((shift) == 0)                                   \
      MPN_COPY (dst, src, size);                        \
    else                                                \
      {                                                 \
        mpn_rshift (dst, src, size, shift);             \
        (size) -= ((dst)[(size)-1] == 0);               \
      }                                                 \
  } while (0)


/* ralloc and talloc are only wanted for ASSERTs, after the initial space
   allocations.  Avoid writing values to them in a normal build, to ensure
   the compiler lets them go dead.  gcc already figures this out itself
   actually.  */

#define SWAP_RP_TP                                      \
  do {                                                  \
    MP_PTR_SWAP (rp, tp);                               \
    ASSERT_CODE (MP_SIZE_T_SWAP (ralloc, talloc));      \
  } while (0)


void
mpz_n_pow_ui (mpz_ptr r, mp_srcptr bp, mp_size_t bsize, unsigned long int e)
{
  mp_ptr         rp;
  mp_size_t      rtwos_limbs, ralloc, rsize;
  int            rneg, i, cnt, btwos, r_bp_overlap;
  mp_limb_t      blimb, rl;
  mp_bitcnt_t    rtwos_bits;
#if HAVE_NATIVE_mpn_mul_2
  mp_limb_t      blimb_low, rl_high;
#else
  mp_limb_t      b_twolimbs[2];
#endif
  TMP_DECL;

  TRACE (printf ("mpz_n_pow_ui rp=0x%lX bp=0x%lX bsize=%ld e=%lu (0x%lX)\n",
		 PTR(r), bp, bsize, e, e);
	 mpn_trace ("b", bp, bsize));

  ASSERT (bsize == 0 || bp[ABS(bsize)-1] != 0);
  ASSERT (MPN_SAME_OR_SEPARATE2_P (PTR(r), ALLOC(r), bp, ABS(bsize)));

  /* b^0 == 1, including 0^0 == 1 */
  if (e == 0)
    {
      PTR(r)[0] = 1;
      SIZ(r) = 1;
      return;
    }

  /* 0^e == 0 apart from 0^0 above */
  if (bsize == 0)
    {
      SIZ(r) = 0;
      return;
    }

  /* Sign of the final result. */
  rneg = (bsize < 0 && (e & 1) != 0);
  bsize = ABS (bsize);
  TRACE (printf ("rneg %d\n", rneg));

  r_bp_overlap = (PTR(r) == bp);

  /* Strip low zero limbs from b. */
  rtwos_limbs = 0;
  for (blimb = *bp; blimb == 0; blimb = *++bp)
    {
      rtwos_limbs += e;
      bsize--; ASSERT (bsize >= 1);
    }
  TRACE (printf ("trailing zero rtwos_limbs=%ld\n", rtwos_limbs));

  /* Strip low zero bits from b. */
  count_trailing_zeros (btwos, blimb);
  blimb >>= btwos;
  rtwos_bits = e * btwos;
  rtwos_limbs += rtwos_bits / GMP_NUMB_BITS;
  rtwos_bits %= GMP_NUMB_BITS;
  TRACE (printf ("trailing zero btwos=%d rtwos_limbs=%ld rtwos_bits=%lu\n",
		 btwos, rtwos_limbs, rtwos_bits));

  TMP_MARK;

  rl = 1;
#if HAVE_NATIVE_mpn_mul_2
  rl_high = 0;
#endif

  if (bsize == 1)
    {
    bsize_1:
      /* Power up as far as possible within blimb.  We start here with e!=0,
	 but if e is small then we might reach e==0 and the whole b^e in rl.
	 Notice this code works when blimb==1 too, reaching e==0.  */

      while (blimb <= GMP_NUMB_HALFMAX)
	{
	  TRACE (printf ("small e=0x%lX blimb=0x%lX rl=0x%lX\n",
			 e, blimb, rl));
	  ASSERT (e != 0);
	  if ((e & 1) != 0)
	    rl *= blimb;
	  e >>= 1;
	  if (e == 0)
	    goto got_rl;
	  blimb *= blimb;
	}

#if HAVE_NATIVE_mpn_mul_2
      TRACE (printf ("single power, e=0x%lX b=0x%lX rl=0x%lX\n",
		     e, blimb, rl));

      /* Can power b once more into blimb:blimb_low */
      bsize = 2;
      ASSERT (e != 0);
      if ((e & 1) != 0)
	{
	  umul_ppmm (rl_high, rl, rl, blimb << GMP_NAIL_BITS);
	  rl >>= GMP_NAIL_BITS;
	}
      e >>= 1;
      umul_ppmm (blimb, blimb_low, blimb, blimb << GMP_NAIL_BITS);
      blimb_low >>= GMP_NAIL_BITS;

    got_rl:
      TRACE (printf ("double power e=0x%lX blimb=0x%lX:0x%lX rl=0x%lX:%lX\n",
		     e, blimb, blimb_low, rl_high, rl));

      /* Combine left-over rtwos_bits into rl_high:rl to be handled by the
	 final mul_1 or mul_2 rather than a separate lshift.
	 - rl_high:rl mustn't be 1 (since then there's no final mul)
	 - rl_high mustn't overflow
	 - rl_high mustn't change to non-zero, since mul_1+lshift is
	 probably faster than mul_2 (FIXME: is this true?)  */

      if (rtwos_bits != 0
	  && ! (rl_high == 0 && rl == 1)
	  && (rl_high >> (GMP_NUMB_BITS-rtwos_bits)) == 0)
	{
	  mp_limb_t  new_rl_high = (rl_high << rtwos_bits)
	    | (rl >> (GMP_NUMB_BITS-rtwos_bits));
	  if (! (rl_high == 0 && new_rl_high != 0))
	    {
	      rl_high = new_rl_high;
	      rl <<= rtwos_bits;
	      rtwos_bits = 0;
	      TRACE (printf ("merged rtwos_bits, rl=0x%lX:%lX\n",
			     rl_high, rl));
	    }
	}
#else
    got_rl:
      TRACE (printf ("small power e=0x%lX blimb=0x%lX rl=0x%lX\n",
		     e, blimb, rl));

      /* Combine left-over rtwos_bits into rl to be handled by the final
	 mul_1 rather than a separate lshift.
	 - rl mustn't be 1 (since then there's no final mul)
	 - rl mustn't overflow	*/

      if (rtwos_bits != 0
	  && rl != 1
	  && (rl >> (GMP_NUMB_BITS-rtwos_bits)) == 0)
	{
	  rl <<= rtwos_bits;
	  rtwos_bits = 0;
	  TRACE (printf ("merged rtwos_bits, rl=0x%lX\n", rl));
	}
#endif
    }
  else if (bsize == 2)
    {
      mp_limb_t  bsecond = bp[1];
      if (btwos != 0)
	blimb |= (bsecond << (GMP_NUMB_BITS - btwos)) & GMP_NUMB_MASK;
      bsecond >>= btwos;
      if (bsecond == 0)
	{
	  /* Two limbs became one after rshift. */
	  bsize = 1;
	  goto bsize_1;
	}

      TRACE (printf ("bsize==2 using b=0x%lX:%lX", bsecond, blimb));
#if HAVE_NATIVE_mpn_mul_2
      blimb_low = blimb;
#else
      bp = b_twolimbs;
      b_twolimbs[0] = blimb;
      b_twolimbs[1] = bsecond;
#endif
      blimb = bsecond;
    }
  else
    {
      if (r_bp_overlap || btwos != 0)
	{
	  mp_ptr tp = TMP_ALLOC_LIMBS (bsize);
	  MPN_RSHIFT_OR_COPY (tp, bp, bsize, btwos);
	  bp = tp;
	  TRACE (printf ("rshift or copy bp,bsize, new bsize=%ld\n", bsize));
	}
#if HAVE_NATIVE_mpn_mul_2
      /* in case 3 limbs rshift to 2 and hence use the mul_2 loop below */
      blimb_low = bp[0];
#endif
      blimb = bp[bsize-1];

      TRACE (printf ("big bsize=%ld  ", bsize);
	     mpn_trace ("b", bp, bsize));
    }

  /* At this point blimb is the most significant limb of the base to use.

     Each factor of b takes (bsize*BPML-cnt) bits and there's e of them; +1
     limb to round up the division; +1 for multiplies all using an extra
     limb over the true size; +2 for rl at the end; +1 for lshift at the
     end.

     The size calculation here is reasonably accurate.  The base is at least
     half a limb, so in 32 bits the worst case is 2^16+1 treated as 17 bits
     when it will power up as just over 16, an overestimate of 17/16 =
     6.25%.  For a 64-bit limb it's half that.

     If e==0 then blimb won't be anything useful (though it will be
     non-zero), but that doesn't matter since we just end up with ralloc==5,
     and that's fine for 2 limbs of rl and 1 of lshift.  */

  ASSERT (blimb != 0);
  count_leading_zeros (cnt, blimb);
  ralloc = (bsize*GMP_NUMB_BITS - cnt + GMP_NAIL_BITS) * e / GMP_NUMB_BITS + 5;
  TRACE (printf ("ralloc %ld, from bsize=%ld blimb=0x%lX cnt=%d\n",
		 ralloc, bsize, blimb, cnt));
  rp = MPZ_REALLOC (r, ralloc + rtwos_limbs);

  /* Low zero limbs resulting from powers of 2. */
  MPN_ZERO (rp, rtwos_limbs);
  rp += rtwos_limbs;

  if (e == 0)
    {
      /* Any e==0 other than via bsize==1 or bsize==2 is covered at the
	 start. */
      rp[0] = rl;
      rsize = 1;
#if HAVE_NATIVE_mpn_mul_2
      rp[1] = rl_high;
      rsize += (rl_high != 0);
#endif
      ASSERT (rp[rsize-1] != 0);
    }
  else
    {
      mp_ptr     tp;
      mp_size_t  talloc;

      /* In the mpn_mul_1 or mpn_mul_2 loops or in the mpn_mul loop when the
	 low bit of e is zero, tp only has to hold the second last power
	 step, which is half the size of the final result.  There's no need
	 to round up the divide by 2, since ralloc includes a +2 for rl
	 which not needed by tp.  In the mpn_mul loop when the low bit of e
	 is 1, tp must hold nearly the full result, so just size it the same
	 as rp.  */

      talloc = ralloc;
#if HAVE_NATIVE_mpn_mul_2
      if (bsize <= 2 || (e & 1) == 0)
	talloc /= 2;
#else
      if (bsize <= 1 || (e & 1) == 0)
	talloc /= 2;
#endif
      TRACE (printf ("talloc %ld\n", talloc));
      tp = TMP_ALLOC_LIMBS (talloc);

      /* Go from high to low over the bits of e, starting with i pointing at
	 the bit below the highest 1 (which will mean i==-1 if e==1).  */
      count_leading_zeros (cnt, (mp_limb_t) e);
      i = GMP_LIMB_BITS - cnt - 2;

#if HAVE_NATIVE_mpn_mul_2
      if (bsize <= 2)
	{
	  mp_limb_t  mult[2];

	  /* Any bsize==1 will have been powered above to be two limbs. */
	  ASSERT (bsize == 2);
	  ASSERT (blimb != 0);

	  /* Arrange the final result ends up in r, not in the temp space */
	  if ((i & 1) == 0)
	    SWAP_RP_TP;

	  rp[0] = blimb_low;
	  rp[1] = blimb;
	  rsize = 2;

	  mult[0] = blimb_low;
	  mult[1] = blimb;

	  for ( ; i >= 0; i--)
	    {
	      TRACE (printf ("mul_2 loop i=%d e=0x%lX, rsize=%ld ralloc=%ld talloc=%ld\n",
			     i, e, rsize, ralloc, talloc);
		     mpn_trace ("r", rp, rsize));

	      MPN_SQR (tp, talloc, rp, rsize);
	      SWAP_RP_TP;
	      if ((e & (1L << i)) != 0)
		MPN_MUL_2 (rp, rsize, ralloc, mult);
	    }

	  TRACE (mpn_trace ("mul_2 before rl, r", rp, rsize));
	  if (rl_high != 0)
	    {
	      mult[0] = rl;
	      mult[1] = rl_high;
	      MPN_MUL_2 (rp, rsize, ralloc, mult);
	    }
	  else if (rl != 1)
	    MPN_MUL_1 (rp, rsize, ralloc, rl);
	}
#else
      if (bsize == 1)
	{
	  /* Arrange the final result ends up in r, not in the temp space */
	  if ((i & 1) == 0)
	    SWAP_RP_TP;

	  rp[0] = blimb;
	  rsize = 1;

	  for ( ; i >= 0; i--)
	    {
	      TRACE (printf ("mul_1 loop i=%d e=0x%lX, rsize=%ld ralloc=%ld talloc=%ld\n",
			     i, e, rsize, ralloc, talloc);
		     mpn_trace ("r", rp, rsize));

	      MPN_SQR (tp, talloc, rp, rsize);
	      SWAP_RP_TP;
	      if ((e & (1L << i)) != 0)
		MPN_MUL_1 (rp, rsize, ralloc, blimb);
	    }

	  TRACE (mpn_trace ("mul_1 before rl, r", rp, rsize));
	  if (rl != 1)
	    MPN_MUL_1 (rp, rsize, ralloc, rl);
	}
#endif
      else
	{
	  int  parity;

	  /* Arrange the final result ends up in r, not in the temp space */
	  ULONG_PARITY (parity, e);
	  if (((parity ^ i) & 1) != 0)
	    SWAP_RP_TP;

	  MPN_COPY (rp, bp, bsize);
	  rsize = bsize;

	  for ( ; i >= 0; i--)
	    {
	      TRACE (printf ("mul loop i=%d e=0x%lX, rsize=%ld ralloc=%ld talloc=%ld\n",
			     i, e, rsize, ralloc, talloc);
		     mpn_trace ("r", rp, rsize));

	      MPN_SQR (tp, talloc, rp, rsize);
	      SWAP_RP_TP;
	      if ((e & (1L << i)) != 0)
		{
		  MPN_MUL (tp, talloc, rp, rsize, bp, bsize);
		  SWAP_RP_TP;
		}
	    }
	}
    }

  ASSERT (rp == PTR(r) + rtwos_limbs);
  TRACE (mpn_trace ("end loop r", rp, rsize));
  TMP_FREE;

  /* Apply any partial limb factors of 2. */
  if (rtwos_bits != 0)
    {
      MPN_LSHIFT (rp, rsize, ralloc, (unsigned) rtwos_bits);
      TRACE (mpn_trace ("lshift r", rp, rsize));
    }

  rsize += rtwos_limbs;
  SIZ(r) = (rneg ? -rsize : rsize);
}
