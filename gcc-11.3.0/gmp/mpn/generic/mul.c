/* mpn_mul -- Multiply two natural numbers.

   Contributed to the GNU project by Torbjorn Granlund.

Copyright 1991, 1993, 1994, 1996, 1997, 1999-2003, 2005-2007, 2009, 2010, 2012,
2014 Free Software Foundation, Inc.

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


#ifndef MUL_BASECASE_MAX_UN
#define MUL_BASECASE_MAX_UN 500
#endif

/* Areas where the different toom algorithms can be called (extracted
   from the t-toom*.c files, and ignoring small constant offsets):

   1/6  1/5 1/4 4/13 1/3 3/8 2/5 5/11 1/2 3/5 2/3 3/4 4/5   1 vn/un
                                        4/7              6/7
				       6/11
                                       |--------------------| toom22 (small)
                                                           || toom22 (large)
                                                       |xxxx| toom22 called
                      |-------------------------------------| toom32
                                         |xxxxxxxxxxxxxxxx| | toom32 called
                                               |------------| toom33
                                                          |x| toom33 called
             |---------------------------------|            | toom42
	              |xxxxxxxxxxxxxxxxxxxxxxxx|            | toom42 called
                                       |--------------------| toom43
                                               |xxxxxxxxxx|   toom43 called
         |-----------------------------|                      toom52 (unused)
                                                   |--------| toom44
						   |xxxxxxxx| toom44 called
                              |--------------------|        | toom53
                                        |xxxxxx|              toom53 called
    |-------------------------|                               toom62 (unused)
                                           |----------------| toom54 (unused)
                      |--------------------|                  toom63
	                      |xxxxxxxxx|                   | toom63 called
                          |---------------------------------| toom6h
						   |xxxxxxxx| toom6h called
                                  |-------------------------| toom8h (32 bit)
                 |------------------------------------------| toom8h (64 bit)
						   |xxxxxxxx| toom8h called
*/

#define TOOM33_OK(an,bn) (6 + 2 * an < 3 * bn)
#define TOOM44_OK(an,bn) (12 + 3 * an < 4 * bn)

/* Multiply the natural numbers u (pointed to by UP, with UN limbs) and v
   (pointed to by VP, with VN limbs), and store the result at PRODP.  The
   result is UN + VN limbs.  Return the most significant limb of the result.

   NOTE: The space pointed to by PRODP is overwritten before finished with U
   and V, so overlap is an error.

   Argument constraints:
   1. UN >= VN.
   2. PRODP != UP and PRODP != VP, i.e. the destination must be distinct from
      the multiplier and the multiplicand.  */

/*
  * The cutoff lines in the toomX2 and toomX3 code are now exactly between the
    ideal lines of the surrounding algorithms.  Is that optimal?

  * The toomX3 code now uses a structure similar to the one of toomX2, except
    that it loops longer in the unbalanced case.  The result is that the
    remaining area might have un < vn.  Should we fix the toomX2 code in a
    similar way?

  * The toomX3 code is used for the largest non-FFT unbalanced operands.  It
    therefore calls mpn_mul recursively for certain cases.

  * Allocate static temp space using THRESHOLD variables (except for toom44
    when !WANT_FFT).  That way, we can typically have no TMP_ALLOC at all.

  * We sort ToomX2 algorithms together, assuming the toom22, toom32, toom42
    have the same vn threshold.  This is not true, we should actually use
    mul_basecase for slightly larger operands for toom32 than for toom22, and
    even larger for toom42.

  * That problem is even more prevalent for toomX3.  We therefore use special
    THRESHOLD variables there.

  * Is our ITCH allocation correct?
*/

#define ITCH (16*vn + 100)

mp_limb_t
mpn_mul (mp_ptr prodp,
	 mp_srcptr up, mp_size_t un,
	 mp_srcptr vp, mp_size_t vn)
{
  ASSERT (un >= vn);
  ASSERT (vn >= 1);
  ASSERT (! MPN_OVERLAP_P (prodp, un+vn, up, un));
  ASSERT (! MPN_OVERLAP_P (prodp, un+vn, vp, vn));

  if (un == vn)
    {
      if (up == vp)
	mpn_sqr (prodp, up, un);
      else
	mpn_mul_n (prodp, up, vp, un);
    }
  else if (vn < MUL_TOOM22_THRESHOLD)
    { /* plain schoolbook multiplication */

      /* Unless un is very large, or else if have an applicable mpn_mul_N,
	 perform basecase multiply directly.  */
      if (un <= MUL_BASECASE_MAX_UN
#if HAVE_NATIVE_mpn_mul_2
	  || vn <= 2
#else
	  || vn == 1
#endif
	  )
	mpn_mul_basecase (prodp, up, un, vp, vn);
      else
	{
	  /* We have un >> MUL_BASECASE_MAX_UN > vn.  For better memory
	     locality, split up[] into MUL_BASECASE_MAX_UN pieces and multiply
	     these pieces with the vp[] operand.  After each such partial
	     multiplication (but the last) we copy the most significant vn
	     limbs into a temporary buffer since that part would otherwise be
	     overwritten by the next multiplication.  After the next
	     multiplication, we add it back.  This illustrates the situation:

                                                    -->vn<--
                                                      |  |<------- un ------->|
                                                         _____________________|
                                                        X                    /|
                                                      /XX__________________/  |
                                    _____________________                     |
                                   X                    /                     |
                                 /XX__________________/                       |
               _____________________                                          |
              /                    /                                          |
            /____________________/                                            |
	    ==================================================================

	    The parts marked with X are the parts whose sums are copied into
	    the temporary buffer.  */

	  mp_limb_t tp[MUL_TOOM22_THRESHOLD_LIMIT];
	  mp_limb_t cy;
	  ASSERT (MUL_TOOM22_THRESHOLD <= MUL_TOOM22_THRESHOLD_LIMIT);

	  mpn_mul_basecase (prodp, up, MUL_BASECASE_MAX_UN, vp, vn);
	  prodp += MUL_BASECASE_MAX_UN;
	  MPN_COPY (tp, prodp, vn);		/* preserve high triangle */
	  up += MUL_BASECASE_MAX_UN;
	  un -= MUL_BASECASE_MAX_UN;
	  while (un > MUL_BASECASE_MAX_UN)
	    {
	      mpn_mul_basecase (prodp, up, MUL_BASECASE_MAX_UN, vp, vn);
	      cy = mpn_add_n (prodp, prodp, tp, vn); /* add back preserved triangle */
	      mpn_incr_u (prodp + vn, cy);
	      prodp += MUL_BASECASE_MAX_UN;
	      MPN_COPY (tp, prodp, vn);		/* preserve high triangle */
	      up += MUL_BASECASE_MAX_UN;
	      un -= MUL_BASECASE_MAX_UN;
	    }
	  if (un > vn)
	    {
	      mpn_mul_basecase (prodp, up, un, vp, vn);
	    }
	  else
	    {
	      ASSERT (un > 0);
	      mpn_mul_basecase (prodp, vp, vn, up, un);
	    }
	  cy = mpn_add_n (prodp, prodp, tp, vn); /* add back preserved triangle */
	  mpn_incr_u (prodp + vn, cy);
	}
    }
  else if (BELOW_THRESHOLD (vn, MUL_TOOM33_THRESHOLD))
    {
      /* Use ToomX2 variants */
      mp_ptr scratch;
      TMP_SDECL; TMP_SMARK;

#define ITCH_TOOMX2 (9 * vn / 2 + GMP_NUMB_BITS * 2)
      scratch = TMP_SALLOC_LIMBS (ITCH_TOOMX2);
      ASSERT (mpn_toom22_mul_itch ((5*vn-1)/4, vn) <= ITCH_TOOMX2); /* 5vn/2+ */
      ASSERT (mpn_toom32_mul_itch ((7*vn-1)/4, vn) <= ITCH_TOOMX2); /* 7vn/6+ */
      ASSERT (mpn_toom42_mul_itch (3 * vn - 1, vn) <= ITCH_TOOMX2); /* 9vn/2+ */
#undef ITCH_TOOMX2

      /* FIXME: This condition (repeated in the loop below) leaves from a vn*vn
	 square to a (3vn-1)*vn rectangle.  Leaving such a rectangle is hardly
	 wise; we would get better balance by slightly moving the bound.  We
	 will sometimes end up with un < vn, like in the X3 arm below.  */
      if (un >= 3 * vn)
	{
	  mp_limb_t cy;
	  mp_ptr ws;

	  /* The maximum ws usage is for the mpn_mul result.  */
	  ws = TMP_SALLOC_LIMBS (4 * vn);

	  mpn_toom42_mul (prodp, up, 2 * vn, vp, vn, scratch);
	  un -= 2 * vn;
	  up += 2 * vn;
	  prodp += 2 * vn;

	  while (un >= 3 * vn)
	    {
	      mpn_toom42_mul (ws, up, 2 * vn, vp, vn, scratch);
	      un -= 2 * vn;
	      up += 2 * vn;
	      cy = mpn_add_n (prodp, prodp, ws, vn);
	      MPN_COPY (prodp + vn, ws + vn, 2 * vn);
	      mpn_incr_u (prodp + vn, cy);
	      prodp += 2 * vn;
	    }

	  /* vn <= un < 3vn */

	  if (4 * un < 5 * vn)
	    mpn_toom22_mul (ws, up, un, vp, vn, scratch);
	  else if (4 * un < 7 * vn)
	    mpn_toom32_mul (ws, up, un, vp, vn, scratch);
	  else
	    mpn_toom42_mul (ws, up, un, vp, vn, scratch);

	  cy = mpn_add_n (prodp, prodp, ws, vn);
	  MPN_COPY (prodp + vn, ws + vn, un);
	  mpn_incr_u (prodp + vn, cy);
	}
      else
	{
	  if (4 * un < 5 * vn)
	    mpn_toom22_mul (prodp, up, un, vp, vn, scratch);
	  else if (4 * un < 7 * vn)
	    mpn_toom32_mul (prodp, up, un, vp, vn, scratch);
	  else
	    mpn_toom42_mul (prodp, up, un, vp, vn, scratch);
	}
      TMP_SFREE;
    }
  else if (BELOW_THRESHOLD ((un + vn) >> 1, MUL_FFT_THRESHOLD) ||
	   BELOW_THRESHOLD (3 * vn, MUL_FFT_THRESHOLD))
    {
      /* Handle the largest operands that are not in the FFT range.  The 2nd
	 condition makes very unbalanced operands avoid the FFT code (except
	 perhaps as coefficient products of the Toom code.  */

      if (BELOW_THRESHOLD (vn, MUL_TOOM44_THRESHOLD) || !TOOM44_OK (un, vn))
	{
	  /* Use ToomX3 variants */
	  mp_ptr scratch;
	  TMP_DECL; TMP_MARK;

#define ITCH_TOOMX3 (4 * vn + GMP_NUMB_BITS)
	  scratch = TMP_ALLOC_LIMBS (ITCH_TOOMX3);
	  ASSERT (mpn_toom33_mul_itch ((7*vn-1)/6, vn) <= ITCH_TOOMX3); /* 7vn/2+ */
	  ASSERT (mpn_toom43_mul_itch ((3*vn-1)/2, vn) <= ITCH_TOOMX3); /* 9vn/4+ */
	  ASSERT (mpn_toom32_mul_itch ((7*vn-1)/4, vn) <= ITCH_TOOMX3); /* 7vn/6+ */
	  ASSERT (mpn_toom53_mul_itch ((11*vn-1)/6, vn) <= ITCH_TOOMX3); /* 11vn/3+ */
	  ASSERT (mpn_toom42_mul_itch ((5*vn-1)/2, vn) <= ITCH_TOOMX3); /* 15vn/4+ */
	  ASSERT (mpn_toom63_mul_itch ((5*vn-1)/2, vn) <= ITCH_TOOMX3); /* 15vn/4+ */
#undef ITCH_TOOMX3

	  if (2 * un >= 5 * vn)
	    {
	      mp_limb_t cy;
	      mp_ptr ws;

	      /* The maximum ws usage is for the mpn_mul result.  */
	      ws = TMP_ALLOC_LIMBS (7 * vn >> 1);

	      if (BELOW_THRESHOLD (vn, MUL_TOOM42_TO_TOOM63_THRESHOLD))
		mpn_toom42_mul (prodp, up, 2 * vn, vp, vn, scratch);
	      else
		mpn_toom63_mul (prodp, up, 2 * vn, vp, vn, scratch);
	      un -= 2 * vn;
	      up += 2 * vn;
	      prodp += 2 * vn;

	      while (2 * un >= 5 * vn)	/* un >= 2.5vn */
		{
		  if (BELOW_THRESHOLD (vn, MUL_TOOM42_TO_TOOM63_THRESHOLD))
		    mpn_toom42_mul (ws, up, 2 * vn, vp, vn, scratch);
		  else
		    mpn_toom63_mul (ws, up, 2 * vn, vp, vn, scratch);
		  un -= 2 * vn;
		  up += 2 * vn;
		  cy = mpn_add_n (prodp, prodp, ws, vn);
		  MPN_COPY (prodp + vn, ws + vn, 2 * vn);
		  mpn_incr_u (prodp + vn, cy);
		  prodp += 2 * vn;
		}

	      /* vn / 2 <= un < 2.5vn */

	      if (un < vn)
		mpn_mul (ws, vp, vn, up, un);
	      else
		mpn_mul (ws, up, un, vp, vn);

	      cy = mpn_add_n (prodp, prodp, ws, vn);
	      MPN_COPY (prodp + vn, ws + vn, un);
	      mpn_incr_u (prodp + vn, cy);
	    }
	  else
	    {
	      if (6 * un < 7 * vn)
		mpn_toom33_mul (prodp, up, un, vp, vn, scratch);
	      else if (2 * un < 3 * vn)
		{
		  if (BELOW_THRESHOLD (vn, MUL_TOOM32_TO_TOOM43_THRESHOLD))
		    mpn_toom32_mul (prodp, up, un, vp, vn, scratch);
		  else
		    mpn_toom43_mul (prodp, up, un, vp, vn, scratch);
		}
	      else if (6 * un < 11 * vn)
		{
		  if (4 * un < 7 * vn)
		    {
		      if (BELOW_THRESHOLD (vn, MUL_TOOM32_TO_TOOM53_THRESHOLD))
			mpn_toom32_mul (prodp, up, un, vp, vn, scratch);
		      else
			mpn_toom53_mul (prodp, up, un, vp, vn, scratch);
		    }
		  else
		    {
		      if (BELOW_THRESHOLD (vn, MUL_TOOM42_TO_TOOM53_THRESHOLD))
			mpn_toom42_mul (prodp, up, un, vp, vn, scratch);
		      else
			mpn_toom53_mul (prodp, up, un, vp, vn, scratch);
		    }
		}
	      else
		{
		  if (BELOW_THRESHOLD (vn, MUL_TOOM42_TO_TOOM63_THRESHOLD))
		    mpn_toom42_mul (prodp, up, un, vp, vn, scratch);
		  else
		    mpn_toom63_mul (prodp, up, un, vp, vn, scratch);
		}
	    }
	  TMP_FREE;
	}
      else
	{
	  mp_ptr scratch;
	  TMP_DECL; TMP_MARK;

	  if (BELOW_THRESHOLD (vn, MUL_TOOM6H_THRESHOLD))
	    {
	      scratch = TMP_SALLOC_LIMBS (mpn_toom44_mul_itch (un, vn));
	      mpn_toom44_mul (prodp, up, un, vp, vn, scratch);
	    }
	  else if (BELOW_THRESHOLD (vn, MUL_TOOM8H_THRESHOLD))
	    {
	      scratch = TMP_SALLOC_LIMBS (mpn_toom6h_mul_itch (un, vn));
	      mpn_toom6h_mul (prodp, up, un, vp, vn, scratch);
	    }
	  else
	    {
	      scratch = TMP_ALLOC_LIMBS (mpn_toom8h_mul_itch (un, vn));
	      mpn_toom8h_mul (prodp, up, un, vp, vn, scratch);
	    }
	  TMP_FREE;
	}
    }
  else
    {
      if (un >= 8 * vn)
	{
	  mp_limb_t cy;
	  mp_ptr ws;
	  TMP_DECL; TMP_MARK;

	  /* The maximum ws usage is for the mpn_mul result.  */
	  ws = TMP_BALLOC_LIMBS (9 * vn >> 1);

	  mpn_fft_mul (prodp, up, 3 * vn, vp, vn);
	  un -= 3 * vn;
	  up += 3 * vn;
	  prodp += 3 * vn;

	  while (2 * un >= 7 * vn)	/* un >= 3.5vn  */
	    {
	      mpn_fft_mul (ws, up, 3 * vn, vp, vn);
	      un -= 3 * vn;
	      up += 3 * vn;
	      cy = mpn_add_n (prodp, prodp, ws, vn);
	      MPN_COPY (prodp + vn, ws + vn, 3 * vn);
	      mpn_incr_u (prodp + vn, cy);
	      prodp += 3 * vn;
	    }

	  /* vn / 2 <= un < 3.5vn */

	  if (un < vn)
	    mpn_mul (ws, vp, vn, up, un);
	  else
	    mpn_mul (ws, up, un, vp, vn);

	  cy = mpn_add_n (prodp, prodp, ws, vn);
	  MPN_COPY (prodp + vn, ws + vn, un);
	  mpn_incr_u (prodp + vn, cy);

	  TMP_FREE;
	}
      else
	mpn_fft_mul (prodp, up, un, vp, vn);
    }

  return prodp[un + vn - 1];	/* historic */
}
