/* mpfr_round_raw_generic, mpfr_round_raw2, mpfr_round_raw, mpfr_prec_round,
   mpfr_can_round, mpfr_can_round_raw -- various rounding functions

Copyright 1999-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include "mpfr-impl.h"

#define mpfr_round_raw_generic mpfr_round_raw
#define flag 0
#define use_inexp 1
#include "round_raw_generic.c"

#define mpfr_round_raw_generic mpfr_round_raw_2
#define flag 1
#define use_inexp 0
#include "round_raw_generic.c"

/* Seems to be unused. Remove comment to implement it.
#define mpfr_round_raw_generic mpfr_round_raw_3
#define flag 1
#define use_inexp 1
#include "round_raw_generic.c"
*/

#define mpfr_round_raw_generic mpfr_round_raw_4
#define flag 0
#define use_inexp 0
#include "round_raw_generic.c"

int
mpfr_prec_round (mpfr_ptr x, mpfr_prec_t prec, mpfr_rnd_t rnd_mode)
{
  mp_limb_t *tmp, *xp;
  int carry, inexact;
  mpfr_prec_t nw, ow;
  MPFR_TMP_DECL(marker);

  MPFR_ASSERTN(prec >= MPFR_PREC_MIN && prec <= MPFR_PREC_MAX);

  nw = MPFR_PREC2LIMBS (prec); /* needed allocated limbs */

  /* check if x has enough allocated space for the significand */
  /* Get the number of limbs from the precision.
     (Compatible with all allocation methods) */
  ow = MPFR_LIMB_SIZE (x);
  if (nw > ow)
    {
      /* FIXME: Variable can't be created using custom allocation,
         MPFR_DECL_INIT or GROUP_ALLOC: How to detect? */
      ow = MPFR_GET_ALLOC_SIZE(x);
      if (nw > ow)
       {
         /* Realloc significand */
         mpfr_limb_ptr tmpx = (mpfr_limb_ptr) (*__gmp_reallocate_func)
           (MPFR_GET_REAL_PTR(x), MPFR_MALLOC_SIZE(ow), MPFR_MALLOC_SIZE(nw));
         MPFR_SET_MANT_PTR(x, tmpx); /* mant ptr must be set
                                        before alloc size */
         MPFR_SET_ALLOC_SIZE(x, nw); /* new number of allocated limbs */
       }
    }

  if (MPFR_UNLIKELY( MPFR_IS_SINGULAR(x) ))
    {
      MPFR_PREC(x) = prec; /* Special value: need to set prec */
      if (MPFR_IS_NAN(x))
        MPFR_RET_NAN;
      MPFR_ASSERTD(MPFR_IS_INF(x) || MPFR_IS_ZERO(x));
      return 0; /* infinity and zero are exact */
    }

  /* x is a non-zero real number */

  MPFR_TMP_MARK(marker);
  tmp = MPFR_TMP_LIMBS_ALLOC (nw);
  xp = MPFR_MANT(x);
  carry = mpfr_round_raw (tmp, xp, MPFR_PREC(x), MPFR_IS_NEG(x),
                          prec, rnd_mode, &inexact);
  MPFR_PREC(x) = prec;

  if (MPFR_UNLIKELY(carry))
    {
      mpfr_exp_t exp = MPFR_EXP (x);

      if (MPFR_UNLIKELY(exp == __gmpfr_emax))
        (void) mpfr_overflow(x, rnd_mode, MPFR_SIGN(x));
      else
        {
          MPFR_ASSERTD (exp < __gmpfr_emax);
          MPFR_SET_EXP (x, exp + 1);
          xp[nw - 1] = MPFR_LIMB_HIGHBIT;
          if (nw - 1 > 0)
            MPN_ZERO(xp, nw - 1);
        }
    }
  else
    MPN_COPY(xp, tmp, nw);

  MPFR_TMP_FREE(marker);
  return inexact;
}

/* assumption: GMP_NUMB_BITS is a power of 2 */

/* assuming b is an approximation to x in direction rnd1 with error at
   most 2^(MPFR_EXP(b)-err), returns 1 if one is able to round exactly
   x to precision prec with direction rnd2, and 0 otherwise.

   Side effects: none.
*/

int
mpfr_can_round (mpfr_srcptr b, mpfr_exp_t err, mpfr_rnd_t rnd1,
                mpfr_rnd_t rnd2, mpfr_prec_t prec)
{
  if (MPFR_UNLIKELY(MPFR_IS_SINGULAR(b)))
    return 0; /* We cannot round if Zero, Nan or Inf */
  else
    return mpfr_can_round_raw (MPFR_MANT(b), MPFR_LIMB_SIZE(b),
                               MPFR_SIGN(b), err, rnd1, rnd2, prec);
}

int
mpfr_can_round_raw (const mp_limb_t *bp, mp_size_t bn, int neg, mpfr_exp_t err,
                    mpfr_rnd_t rnd1, mpfr_rnd_t rnd2, mpfr_prec_t prec)
{
  mpfr_prec_t prec2;
  mp_size_t k, k1, tn;
  int s, s1;
  mp_limb_t cc, cc2;
  mp_limb_t *tmp;
  mp_limb_t cy = 0, tmp_hi;
  int res;
  MPFR_TMP_DECL(marker);

  /* Since mpfr_can_round is a function in the API, use MPFR_ASSERTN.
     The specification makes sense only for prec >= 1. */
  MPFR_ASSERTN (prec >= 1);

  MPFR_ASSERTD(bp[bn - 1] & MPFR_LIMB_HIGHBIT);

  MPFR_ASSERT_SIGN(neg);
  neg = MPFR_IS_NEG_SIGN(neg);

  /* Transform RNDD and RNDU to Zero / Away */
  MPFR_ASSERTD (neg == 0 || neg == 1);
  if (rnd1 != MPFR_RNDN)
    rnd1 = MPFR_IS_LIKE_RNDZ(rnd1, neg) ? MPFR_RNDZ : MPFR_RNDA;
  if (rnd2 != MPFR_RNDN)
    rnd2 = MPFR_IS_LIKE_RNDZ(rnd2, neg) ? MPFR_RNDZ : MPFR_RNDA;

  /* For err < prec (+1 for rnd1=RNDN), we can never round correctly, since
     the error is at least 2*ulp(b) >= ulp(round(b)).
     However for err = prec (+1 for rnd1=RNDN), we can round correctly in some
     rare cases where ulp(b) = 1/2*ulp(U) [see below for the definition of U],
     which implies rnd1 = RNDZ or RNDN, and rnd2 = RNDA or RNDN. */

  if (MPFR_UNLIKELY (err < prec + (rnd1 == MPFR_RNDN) ||
                     (err == prec + (rnd1 == MPFR_RNDN) &&
                      (rnd1 == MPFR_RNDA ||
                       rnd2 == MPFR_RNDZ))))
    return 0;  /* can't round */

  /* As a consequence... */
  MPFR_ASSERTD (err >= prec);

  /* The bound c on the error |x-b| is: c = 2^(MPFR_EXP(b)-err) <= b/2.
   * So, we now know that x and b have the same sign. By symmetry,
   * assume x > 0 and b > 0. We have: L <= x <= U, where, depending
   * on rnd1:
   *   MPFR_RNDN: L = b-c, U = b+c
   *   MPFR_RNDZ: L = b,   U = b+c
   *   MPFR_RNDA: L = b-c, U = b
   *
   * We can round x iff round(L,prec,rnd2) = round(U,prec,rnd2).
   */

  if (MPFR_UNLIKELY (prec > (mpfr_prec_t) bn * GMP_NUMB_BITS))
    { /* Then prec > PREC(b): we can round:
         (i) in rounding to the nearest as long as err >= prec + 2.
             When err = prec + 1 and b is not a power
             of two (so that a change of binade cannot occur), then one
             can round to nearest thanks to the even rounding rule (in the
             target precision prec, the significand of b ends with a 0).
             When err = prec + 1 and b is a power of two, when rnd1 = RNDZ one
             can round too.
         (ii) in directed rounding mode iff rnd1 is compatible with rnd2
              and err >= prec + 1, unless b = 2^k and rnd1 = RNDA or RNDN in
              which case we need err >= prec + 2.
      */
      if ((rnd1 == rnd2 || rnd2 == MPFR_RNDN) && err >= prec + 1)
        {
          if (rnd1 != MPFR_RNDZ &&
              err == prec + 1 &&
              mpfr_powerof2_raw2 (bp, bn))
            return 0;
          else
            return 1;
        }
      return 0;
    }

  /* now prec <= bn * GMP_NUMB_BITS */

  if (MPFR_UNLIKELY (err > (mpfr_prec_t) bn * GMP_NUMB_BITS))
    {
      /* we distinguish the case where b is a power of two:
         rnd1 rnd2 can round?
         RNDZ RNDZ ok
         RNDZ RNDA no
         RNDZ RNDN ok
         RNDA RNDZ no
         RNDA RNDA ok except when err = prec + 1
         RNDA RNDN ok except when err = prec + 1
         RNDN RNDZ no
         RNDN RNDA no
         RNDN RNDN ok except when err = prec + 1 */
      if (mpfr_powerof2_raw2 (bp, bn))
        {
          if ((rnd2 == MPFR_RNDZ || rnd2 == MPFR_RNDA) && rnd1 != rnd2)
            return 0;
          else if (rnd1 == MPFR_RNDZ)
            return 1; /* RNDZ RNDZ and RNDZ RNDN */
          else
            return err > prec + 1;
        }

      /* now the general case where b is not a power of two:
         rnd1 rnd2 can round?
         RNDZ RNDZ ok
         RNDZ RNDA except when b is representable in precision 'prec'
         RNDZ RNDN except when b is the middle of two representable numbers in
                   precision 'prec' and b ends with 'xxx0[1]',
                   or b is representable in precision 'prec'
                   and err = prec + 1 and b ends with '1'.
         RNDA RNDZ except when b is representable in precision 'prec'
         RNDA RNDA ok
         RNDA RNDN except when b is the middle of two representable numbers in
                   precision 'prec' and b ends with 'xxx1[1]',
                   or b is representable in precision 'prec'
                   and err = prec + 1 and b ends with '1'.
         RNDN RNDZ except when b is representable in precision 'prec'
         RNDN RNDA except when b is representable in precision 'prec'
         RNDN RNDN except when b is the middle of two representable numbers in
                   precision 'prec', or b is representable in precision 'prec'
                   and err = prec + 1 and b ends with '1'. */
      if (rnd2 == MPFR_RNDN)
        {
          if (err == prec + 1 && (bp[0] & 1))
            return 0; /* err == prec + 1 implies prec = bn * GMP_NUMB_BITS */
          if (prec < (mpfr_prec_t) bn * GMP_NUMB_BITS)
            {
              k1 = MPFR_PREC2LIMBS (prec + 1);
              MPFR_UNSIGNED_MINUS_MODULO(s1, prec + 1);
              if (((bp[bn - k1] >> s1) & 1) &&
                  mpfr_round_raw2 (bp, bn, neg, MPFR_RNDA, prec + 1) == 0)
                { /* b is the middle of two representable numbers */
                  if (rnd1 == MPFR_RNDN)
                    return 0;
                  k1 = MPFR_PREC2LIMBS (prec);
                  MPFR_UNSIGNED_MINUS_MODULO(s1, prec);
                  return (rnd1 == MPFR_RNDZ) ^
                    (((bp[bn - k1] >> s1) & 1) == 0);
                }
            }
          return 1;
        }
      else if (rnd1 == rnd2)
        {
          if (rnd1 == MPFR_RNDN && prec < (mpfr_prec_t) bn * GMP_NUMB_BITS)
            {
              /* then rnd2 = RNDN, and for prec = bn * GMP_NUMB_BITS we cannot
                 have b the middle of two representable numbers */
              k1 = MPFR_PREC2LIMBS (prec + 1);
              MPFR_UNSIGNED_MINUS_MODULO(s1, prec + 1);
              if (((bp[bn - k1] >> s1) & 1) &&
                  mpfr_round_raw2 (bp, bn, neg, MPFR_RNDA, prec + 1) == 0)
                /* b is representable in precision prec+1 and ends with a 1 */
                return 0;
              else
                return 1;
            }
          else
            return 1;
        }
      else
        return mpfr_round_raw2 (bp, bn, neg, MPFR_RNDA, prec) != 0;
    }

  /* now err <= bn * GMP_NUMB_BITS */

  /* warning: if k = m*GMP_NUMB_BITS, consider limb m-1 and not m */
  k = (err - 1) / GMP_NUMB_BITS;
  MPFR_UNSIGNED_MINUS_MODULO(s, err);
  /* the error corresponds to bit s in limb k, the most significant limb
     being limb 0; in memory, limb k is bp[bn-1-k]. */

  k1 = (prec - 1) / GMP_NUMB_BITS;
  MPFR_UNSIGNED_MINUS_MODULO(s1, prec);
  /* the least significant bit is bit s1 in limb k1 */

  /* We don't need to consider the k1 most significant limbs.
     They will be considered later only to detect when subtracting
     the error bound yields a change of binade.
     Warning! The number with updated bn may no longer be normalized. */
  k -= k1;
  bn -= k1;
  prec2 = prec - (mpfr_prec_t) k1 * GMP_NUMB_BITS;

  /* We can decide of the correct rounding if rnd2(b-eps) and rnd2(b+eps)
     give the same result to the target precision 'prec', i.e., if when
     adding or subtracting (1 << s) in bp[bn-1-k], it does not change the
     rounding in direction 'rnd2' at ulp-position bp[bn-1] >> s1, taking also
     into account the possible change of binade. */
  MPFR_TMP_MARK(marker);
  tn = bn;
  k++; /* since we work with k+1 everywhere */
  tmp = MPFR_TMP_LIMBS_ALLOC (tn);
  if (bn > k)
    MPN_COPY (tmp, bp, bn - k);

  MPFR_ASSERTD (k > 0);

  switch (rnd1)
    {
    case MPFR_RNDZ:
      /* rnd1 = Round to Zero */
      cc = (bp[bn - 1] >> s1) & 1;
      /* mpfr_round_raw2 returns 1 if one should add 1 at ulp(b,prec),
         and 0 otherwise */
      cc ^= mpfr_round_raw2 (bp, bn, neg, rnd2, prec2);
      /* cc is the new value of bit s1 in bp[bn-1] after rounding 'rnd2' */

      /* now round b + 2^(MPFR_EXP(b)-err) */
      cy = mpn_add_1 (tmp + bn - k, bp + bn - k, k, MPFR_LIMB_ONE << s);
      /* propagate carry up to most significant limb */
      for (tn = 0; tn + 1 < k1 && cy != 0; tn ++)
        cy = ~bp[bn + tn] == 0;
      if (cy == 0 && err == prec)
        {
          res = 0;
          goto end;
        }
      if (MPFR_UNLIKELY(cy))
        {
          /* when a carry occurs, we have b < 2^h <= b+c, we can round iff:
             rnd2 = RNDZ: never, since b and b+c round to different values;
             rnd2 = RNDA: when b+c is an exact power of two, and err > prec
                          (since for err = prec, b = 2^h - 1/2*ulp(2^h) is
                          exactly representable and thus rounds to itself);
             rnd2 = RNDN: whenever cc = 0, since err >= prec implies
                          c <= ulp(b) = 1/2*ulp(2^h), thus b+c rounds to 2^h,
                          and b+c >= 2^h implies that bit 'prec' of b is 1,
                          thus cc = 0 means that b is rounded to 2^h too. */
          res = (rnd2 == MPFR_RNDZ) ? 0
            : (rnd2 == MPFR_RNDA) ? (err > prec && k == bn && tmp[0] == 0)
            : cc == 0;
          goto end;
        }
      break;
    case MPFR_RNDN:
      /* rnd1 = Round to nearest */

      /* first round b+2^(MPFR_EXP(b)-err) */
      cy = mpn_add_1 (tmp + bn - k, bp + bn - k, k, MPFR_LIMB_ONE << s);
      /* propagate carry up to most significant limb */
      for (tn = 0; tn + 1 < k1 && cy != 0; tn ++)
        cy = ~bp[bn + tn] == 0;
      cc = (tmp[bn - 1] >> s1) & 1; /* gives 0 when cc=1 */
      cc ^= mpfr_round_raw2 (tmp, bn, neg, rnd2, prec2);
      /* cc is the new value of bit s1 in bp[bn-1]+eps after rounding 'rnd2' */
      if (MPFR_UNLIKELY (cy != 0))
        {
          /* when a carry occurs, we have b-c < b < 2^h <= b+c, we can round
             iff:
             rnd2 = RNDZ: never, since b-c and b+c round to different values;
             rnd2 = RNDA: when b+c is an exact power of two, and
                          err > prec + 1 (since for err <= prec + 1,
                          b-c <= 2^h - 1/2*ulp(2^h) is exactly representable
                          and thus rounds to itself);
             rnd2 = RNDN: whenever err > prec + 1, since for err = prec + 1,
                          b+c rounds to 2^h, and b-c rounds to nextbelow(2^h).
                          For err > prec + 1, c <= 1/4*ulp(b) <= 1/8*ulp(2^h),
                          thus
                          2^h - 1/4*ulp(b) <= b-c < b+c <= 2^h + 1/8*ulp(2^h),
                          therefore both b-c and b+c round to 2^h. */
          res = (rnd2 == MPFR_RNDZ) ? 0
            : (rnd2 == MPFR_RNDA) ? (err > prec + 1 && k == bn && tmp[0] == 0)
            : err > prec + 1;
          goto end;
        }
    subtract_eps:
      /* now round b-2^(MPFR_EXP(b)-err), this happens for
         rnd1 = RNDN or RNDA */
      MPFR_ASSERTD(rnd1 == MPFR_RNDN || rnd1 == MPFR_RNDA);
      cy = mpn_sub_1 (tmp + bn - k, bp + bn - k, k, MPFR_LIMB_ONE << s);
      /* propagate the potential borrow up to the most significant limb
         (it cannot propagate further since the most significant limb is
         at least MPFR_LIMB_HIGHBIT).
         Note: we use the same limb tmp[bn-1] to subtract. */
      tmp_hi = tmp[bn - 1];
      for (tn = 0; tn < k1 && cy != 0; tn ++)
        cy = mpn_sub_1 (&tmp_hi, bp + bn + tn, 1, cy);
      /* We have an exponent decrease when tn = k1 and
         tmp[bn-1] < MPFR_LIMB_HIGHBIT:
         b-c < 2^h <= b (for RNDA) or b+c (for RNDN).
         Then we surely cannot round when rnd2 = RNDZ, since b or b+c round to
         a value >= 2^h, and b-c rounds to a value < 2^h.
         We also surely cannot round when (rnd1,rnd2) = (RNDN,RNDA), since
         b-c rounds to a value <= 2^h, and b+c > 2^h rounds to a value > 2^h.
         It thus remains:
         (rnd1,rnd2) = (RNDA,RNDA), (RNDA,RNDN) and (RNDN,RNDN).
         For (RNDA,RNDA) we can round only when b-c and b round to 2^h, which
         implies b = 2^h and err > prec (which is true in that case):
         a necessary condition is that cc = 0.
         For (RNDA,RNDN) we can round only when b-c and b round to 2^h, which
         implies b-c >= 2^h - 1/4*ulp(2^h), and b <= 2^h + 1/2*ulp(2^h);
         since ulp(2^h) = ulp(b), this implies c <= 3/4*ulp(b), thus
         err > prec.
         For (RNDN,RNDN) we can round only when b-c and b+c round to 2^h,
         which implies b-c >= 2^h - 1/4*ulp(2^h), and
         b+c <= 2^h + 1/2*ulp(2^h);
         since ulp(2^h) = ulp(b), this implies 2*c <= 3/4*ulp(b), thus
         err > prec+1.
      */
      if (tn == k1 && tmp_hi < MPFR_LIMB_HIGHBIT) /* exponent decrease */
        {
          if (rnd2 == MPFR_RNDZ || (rnd1 == MPFR_RNDN && rnd2 == MPFR_RNDA) ||
              cc != 0 /* b or b+c does not round to 2^h */)
            {
              res = 0;
              goto end;
            }
          /* in that case since the most significant bit of tmp is 0, we
             should consider one more bit; res = 0 when b-c does not round
             to 2^h. */
          res = mpfr_round_raw2 (tmp, bn, neg, rnd2, prec2 + 1) != 0;
          goto end;
        }
      if (err == prec + (rnd1 == MPFR_RNDN))
        {
          /* No exponent increase nor decrease, thus we have |U-L| = ulp(b).
             For rnd2 = RNDZ or RNDA, either [L,U] contains one representable
             number in the target precision, and then L and U round
             differently; or both L and U are representable: they round
             differently too; thus in all cases we cannot round.
             For rnd2 = RNDN, the only case where we can round is when the
             middle of [L,U] (i.e. b) is representable, and ends with a 0. */
          res = (rnd2 == MPFR_RNDN && (((bp[bn - 1] >> s1) & 1) == 0) &&
                 mpfr_round_raw2 (bp, bn, neg, MPFR_RNDZ, prec2) ==
                 mpfr_round_raw2 (bp, bn, neg, MPFR_RNDA, prec2));
          goto end;
        }
      break;
    default:
      /* rnd1 = Round away */
      MPFR_ASSERTD (rnd1 == MPFR_RNDA);
      cc = (bp[bn - 1] >> s1) & 1;
      /* the mpfr_round_raw2() call below returns whether one should add 1 or
         not for rounding */
      cc ^= mpfr_round_raw2 (bp, bn, neg, rnd2, prec2);
      /* cc is the new value of bit s1 in bp[bn-1]+eps after rounding 'rnd2' */

      goto subtract_eps;
    }

  cc2 = (tmp[bn - 1] >> s1) & 1;
  res = cc == (cc2 ^ mpfr_round_raw2 (tmp, bn, neg, rnd2, prec2));

 end:
  MPFR_TMP_FREE(marker);
  return res;
}
