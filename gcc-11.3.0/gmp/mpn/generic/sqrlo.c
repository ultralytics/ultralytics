/* mpn_sqrlo -- squares an n-limb number and returns the low n limbs
   of the result.

   Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

   THIS IS (FOR NOW) AN INTERNAL FUNCTION.  IT IS ONLY SAFE TO REACH THIS
   FUNCTION THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST GUARANTEED
   THAT IT'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2004, 2005, 2009, 2010, 2012, 2015 Free Software Foundation, Inc.

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

#if TUNE_PROGRAM_BUILD || WANT_FAT_BINARY
#define MAYBE_range_basecase 1
#define MAYBE_range_toom22   1
#else
#define MAYBE_range_basecase                                           \
  ((SQRLO_DC_THRESHOLD == 0 ? SQRLO_BASECASE_THRESHOLD : SQRLO_DC_THRESHOLD) < SQR_TOOM2_THRESHOLD*36/(36-11))
#define MAYBE_range_toom22                                             \
  ((SQRLO_DC_THRESHOLD == 0 ? SQRLO_BASECASE_THRESHOLD : SQRLO_DC_THRESHOLD) < SQR_TOOM3_THRESHOLD*36/(36-11) )
#endif

/*  THINK: The DC strategy uses different constants in different Toom's
	 ranges. Something smoother?
*/

/*
  Compute the least significant half of the product {xy,n}*{yp,n}, or
  formally {rp,n} = {xy,n}*{yp,n} Mod (B^n).

  Above the given threshold, the Divide and Conquer strategy is used.
  The operand is split in two, and a full square plus a mullo
  is used to obtain the final result. The more natural strategy is to
  split in two halves, but this is far from optimal when a
  sub-quadratic multiplication is used.

  Mulders suggests an unbalanced split in favour of the full product,
  split n = n1 + n2, where an = n1 <= n2 = (1-a)n; i.e. 0 < a <= 1/2.

  To compute the value of a, we assume that the cost of mullo for a
  given size ML(n) is a fraction of the cost of a full product with
  same size M(n), and the cost M(n)=n^e for some exponent 1 < e <= 2;
  then we can write:

  ML(n) = 2*ML(an) + M((1-a)n) => k*M(n) = 2*k*M(n)*a^e + M(n)*(1-a)^e

  Given a value for e, want to minimise the value of k, i.e. the
  function k=(1-a)^e/(1-2*a^e).

  With e=2, the exponent for schoolbook multiplication, the minimum is
  given by the values a=1-a=1/2.

  With e=log(3)/log(2), the exponent for Karatsuba (aka toom22),
  Mulders compute (1-a) = 0.694... and we approximate a with 11/36.

  Other possible approximations follow:
  e=log(5)/log(3) [Toom-3] -> a ~= 9/40
  e=log(7)/log(4) [Toom-4] -> a ~= 7/39
  e=log(11)/log(6) [Toom-6] -> a ~= 1/8
  e=log(15)/log(8) [Toom-8] -> a ~= 1/10

  The values above where obtained with the following trivial commands
  in the gp-pari shell:

fun(e,a)=(1-a)^e/(1-2*a^e)
mul(a,b,c)={local(m,x,p);if(b-c<1/10000,(b+c)/2,m=1;x=b;forstep(p=c,b,(b-c)/8,if(fun(a,p)<m,m=fun(a,p);x=p));mul(a,(b+x)/2,(c+x)/2))}
contfracpnqn(contfrac(mul(log(2*2-1)/log(2),1/2,0),5))
contfracpnqn(contfrac(mul(log(3*2-1)/log(3),1/2,0),5))
contfracpnqn(contfrac(mul(log(4*2-1)/log(4),1/2,0),5))
contfracpnqn(contfrac(mul(log(6*2-1)/log(6),1/2,0),3))
contfracpnqn(contfrac(mul(log(8*2-1)/log(8),1/2,0),3))

  ,
  |\
  | \
  +----,
  |    |
  |    |
  |    |\
  |    | \
  +----+--`
  ^ n2 ^n1^

  For an actual implementation, the assumption that M(n)=n^e is
  incorrect, as a consequence also the assumption that ML(n)=k*M(n)
  with a constant k is wrong.

  But theory suggest us two things:
  - the best the multiplication product is (lower e), the more k
    approaches 1, and a approaches 0.

  - A value for a smaller than optimal is probably less bad than a
    bigger one: e.g. let e=log(3)/log(2), a=0.3058_ the optimal
    value, and k(a)=0.808_ the mul/mullo speed ratio. We get
    k(a+1/6)=0.929_ but k(a-1/6)=0.865_.
*/

static mp_size_t
mpn_sqrlo_itch (mp_size_t n)
{
  return 2*n;
}

/*
    mpn_dc_sqrlo requires a scratch space of 2*n limbs at tp.
    It accepts tp == rp.
*/
static void
mpn_dc_sqrlo (mp_ptr rp, mp_srcptr xp, mp_size_t n, mp_ptr tp)
{
  mp_size_t n2, n1;
  ASSERT (n >= 2);
  ASSERT (! MPN_OVERLAP_P (rp, n, xp, n));
  ASSERT (MPN_SAME_OR_SEPARATE2_P(rp, n, tp, 2*n));

  /* Divide-and-conquer */

  /* We need fractional approximation of the value 0 < a <= 1/2
     giving the minimum in the function k=(1-a)^e/(1-2*a^e).
  */
  if (MAYBE_range_basecase && BELOW_THRESHOLD (n, SQR_TOOM2_THRESHOLD*36/(36-11)))
    n1 = n >> 1;
  else if (MAYBE_range_toom22 && BELOW_THRESHOLD (n, SQR_TOOM3_THRESHOLD*36/(36-11)))
    n1 = n * 11 / (size_t) 36;	/* n1 ~= n*(1-.694...) */
  else if (BELOW_THRESHOLD (n, SQR_TOOM4_THRESHOLD*40/(40-9)))
    n1 = n * 9 / (size_t) 40;	/* n1 ~= n*(1-.775...) */
  else if (BELOW_THRESHOLD (n, SQR_TOOM8_THRESHOLD*10/9))
    n1 = n * 7 / (size_t) 39;	/* n1 ~= n*(1-.821...) */
  /* n1 = n * 4 / (size_t) 31;	// n1 ~= n*(1-.871...) [TOOM66] */
  else
    n1 = n / (size_t) 10;		/* n1 ~= n*(1-.899...) [TOOM88] */

  n2 = n - n1;

  /* Split as x = x1 2^(n2 GMP_NUMB_BITS) + x0 */

  /* x0 ^ 2 */
  mpn_sqr (tp, xp, n2);
  MPN_COPY (rp, tp, n2);

  /* x1 * x0 * 2^(n2 GMP_NUMB_BITS) */
  if (BELOW_THRESHOLD (n1, MULLO_BASECASE_THRESHOLD))
    mpn_mul_basecase (tp + n, xp + n2, n1, xp, n1);
  else if (BELOW_THRESHOLD (n1, MULLO_DC_THRESHOLD))
    mpn_mullo_basecase (tp + n, xp + n2, xp, n1);
  else
    mpn_mullo_n (tp + n, xp + n2, xp, n1);
  /* mpn_dc_mullo_n (tp + n, xp + n2, xp, n1, tp + n); */
#if HAVE_NATIVE_mpn_addlsh1_n
  mpn_addlsh1_n (rp + n2, tp + n2, tp + n, n1);
#else
  mpn_lshift (rp + n2, tp + n, n1, 1);
  mpn_add_n (rp + n2, rp + n2, tp + n2, n1);
#endif
}

/* Avoid zero allocations when MULLO_BASECASE_THRESHOLD is 0.  */
#define SQR_BASECASE_ALLOC \
 (SQRLO_BASECASE_THRESHOLD_LIMIT == 0 ? 1 : 2*SQRLO_BASECASE_THRESHOLD_LIMIT)

/* FIXME: This function should accept a temporary area; dc_sqrlo
   accepts a pointer tp, and handle the case tp == rp, do the same here.
*/

void
mpn_sqrlo (mp_ptr rp, mp_srcptr xp, mp_size_t n)
{
  ASSERT (n >= 1);
  ASSERT (! MPN_OVERLAP_P (rp, n, xp, n));

  if (BELOW_THRESHOLD (n, SQRLO_BASECASE_THRESHOLD))
    {
      /* FIXME: smarter criteria? */
#if HAVE_NATIVE_mpn_mullo_basecase || ! HAVE_NATIVE_mpn_sqr_basecase
      /* mullo computes as many products as sqr, but directly writes
	 on the result area. */
      mpn_mullo_basecase (rp, xp, xp, n);
#else
      /* Allocate workspace of fixed size on stack: fast! */
      mp_limb_t tp[SQR_BASECASE_ALLOC];
      mpn_sqr_basecase (tp, xp, n);
      MPN_COPY (rp, tp, n);
#endif
    }
  else if (BELOW_THRESHOLD (n, SQRLO_DC_THRESHOLD))
    {
      mpn_sqrlo_basecase (rp, xp, n);
    }
  else
    {
      mp_ptr tp;
      TMP_DECL;
      TMP_MARK;
      tp = TMP_ALLOC_LIMBS (mpn_sqrlo_itch (n));
      if (BELOW_THRESHOLD (n, SQRLO_SQR_THRESHOLD))
	{
	  mpn_dc_sqrlo (rp, xp, n, tp);
	}
      else
	{
	  /* For really large operands, use plain mpn_mul_n but throw away upper n
	     limbs of result.  */
#if !TUNE_PROGRAM_BUILD && (SQRLO_SQR_THRESHOLD > SQR_FFT_THRESHOLD)
	  mpn_fft_mul (tp, xp, n, xp, n);
#else
	  mpn_sqr (tp, xp, n);
#endif
	  MPN_COPY (rp, tp, n);
	}
      TMP_FREE;
    }
}
