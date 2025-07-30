/* mpfr_round_raw_generic -- Generic rounding function

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

#ifndef flag
# error "ERROR: flag must be defined (0 / 1)"
#endif
#ifndef use_inexp
# error "ERROR: use_enexp must be defined (0 / 1)"
#endif
#ifndef mpfr_round_raw_generic
# error "ERROR: mpfr_round_raw_generic must be defined"
#endif

/*
 * If flag = 0, puts in y the value of xp (with precision xprec and
 * sign 1 if negative=0, -1 otherwise) rounded to precision yprec and
 * direction rnd_mode. Supposes x is not zero nor NaN nor +/- Infinity
 * (i.e. *xp != 0). In that case, the return value is a possible carry
 * (0 or 1) that may happen during the rounding, in which case the result
 * is a power of two.
 *
 * If inexp != NULL, put in *inexp the inexact flag of the rounding (0, 1, -1).
 * In case of even rounding when rnd = MPFR_RNDN, put MPFR_EVEN_INEX (2) or
 * -MPFR_EVEN_INEX (-2) in *inexp.
 *
 * If flag = 1, just returns whether one should add 1 or not for rounding.
 *
 * Note: yprec may be < MPFR_PREC_MIN; in particular, it may be equal
 * to 1. In this case, the even rounding is done away from 0, which is
 * a natural generalization. Indeed, a number with 1-bit precision can
 * be seen as a subnormal number with more precision.
 */

int
mpfr_round_raw_generic(
#if flag == 0
                       mp_limb_t *yp,
#endif
                       const mp_limb_t *xp, mpfr_prec_t xprec,
                       int neg, mpfr_prec_t yprec, mpfr_rnd_t rnd_mode
#if use_inexp != 0
                       , int *inexp
#endif
                       )
{
  mp_size_t xsize, nw;
  mp_limb_t himask, lomask, sb;
  int rw;
#if flag == 0
  int carry;
#endif
#if use_inexp == 0
  int *inexp;
#endif

  if (use_inexp)
    MPFR_ASSERTD(inexp != ((int*) 0));
  MPFR_ASSERTD(neg == 0 || neg == 1);

  if (flag && !use_inexp &&
      (xprec <= yprec || MPFR_IS_LIKE_RNDZ (rnd_mode, neg)))
    return 0;

  xsize = MPFR_PREC2LIMBS (xprec);
  nw = yprec / GMP_NUMB_BITS;
  rw = yprec & (GMP_NUMB_BITS - 1);

  if (MPFR_UNLIKELY(xprec <= yprec))
    { /* No rounding is necessary. */
      /* if yp=xp, maybe an overlap: MPN_COPY_DECR is ok when src <= dst */
      if (MPFR_LIKELY(rw))
        nw++;
      MPFR_ASSERTD(nw >= 1);
      MPFR_ASSERTD(nw >= xsize);
      if (use_inexp)
        *inexp = 0;
#if flag == 0
      MPN_COPY_DECR(yp + (nw - xsize), xp, xsize);
      MPN_ZERO(yp, nw - xsize);
#endif
      return 0;
    }

  if (use_inexp || !MPFR_IS_LIKE_RNDZ(rnd_mode, neg))
    {
      mp_size_t k = xsize - nw - 1;

      if (MPFR_LIKELY(rw))
        {
          nw++;
          lomask = MPFR_LIMB_MASK (GMP_NUMB_BITS - rw);
          himask = ~lomask;
        }
      else
        {
          lomask = ~(mp_limb_t) 0;
          himask = ~(mp_limb_t) 0;
        }
      MPFR_ASSERTD(k >= 0);
      sb = xp[k] & lomask;  /* First non-significant bits */
      /* Rounding to nearest ? */
      if (MPFR_LIKELY( rnd_mode == MPFR_RNDN) )
        {
          /* Rounding to nearest */
          mp_limb_t rbmask = MPFR_LIMB_ONE << (GMP_NUMB_BITS - 1 - rw);
          if (sb & rbmask) /* rounding bit */
            sb &= ~rbmask; /* it is 1, clear it */
          else
            {
              /* Rounding bit is 0, behave like rounding to 0 */
              goto rnd_RNDZ;
            }
          while (MPFR_UNLIKELY(sb == 0) && k > 0)
            sb = xp[--k];
          /* rounding to nearest, with rounding bit = 1 */
          if (MPFR_UNLIKELY(sb == 0)) /* Even rounding. */
            {
              /* sb == 0 && rnd_mode == MPFR_RNDN */
              sb = xp[xsize - nw] & (himask ^ (himask << 1));
              if (sb == 0)
                {
                  if (use_inexp)
                    *inexp = 2*MPFR_EVEN_INEX*neg-MPFR_EVEN_INEX;
                  /* ((neg!=0)^(sb!=0)) ? MPFR_EVEN_INEX  : -MPFR_EVEN_INEX;*/
                  /* Since neg = 0 or 1 and sb=0*/
#if flag == 1
                  return 0 /*sb != 0 && rnd_mode != MPFR_RNDZ */;
#else
                  MPN_COPY_INCR(yp, xp + xsize - nw, nw);
                  yp[0] &= himask;
                  return 0;
#endif
                }
              else
                {
                  /* sb != 0 && rnd_mode == MPFR_RNDN */
                  if (use_inexp)
                    *inexp = MPFR_EVEN_INEX-2*MPFR_EVEN_INEX*neg;
                  /*((neg!=0)^(sb!=0))? MPFR_EVEN_INEX  : -MPFR_EVEN_INEX; */
                  /*Since neg= 0 or 1 and sb != 0 */
                  goto rnd_RNDN_add_one_ulp;
                }
            }
          else /* sb != 0  && rnd_mode == MPFR_RNDN*/
            {
              if (use_inexp)
                /* *inexp = (neg == 0) ? 1 : -1; but since neg = 0 or 1 */
                *inexp = 1-2*neg;
            rnd_RNDN_add_one_ulp:
#if flag == 1
              return 1; /*sb != 0 && rnd_mode != MPFR_RNDZ;*/
#else
              carry = mpn_add_1 (yp, xp + xsize - nw, nw,
                                 rw ?
                                 MPFR_LIMB_ONE << (GMP_NUMB_BITS - rw)
                                 : MPFR_LIMB_ONE);
              yp[0] &= himask;
              return carry;
#endif
            }
        }
      /* Rounding to Zero ? */
      else if (MPFR_IS_LIKE_RNDZ(rnd_mode, neg))
        {
          /* rnd_mode == MPFR_RNDZ */
        rnd_RNDZ:
          while (MPFR_UNLIKELY(sb == 0) && k > 0)
            sb = xp[--k];
          if (use_inexp)
            /* rnd_mode == MPFR_RNDZ and neg = 0 or 1 */
            /* (neg != 0) ^ (rnd_mode != MPFR_RNDZ)) ? 1 : -1);*/
            *inexp = MPFR_UNLIKELY(sb == 0) ? 0 : (2*neg-1);
#if flag == 1
          return 0; /*sb != 0 && rnd_mode != MPFR_RNDZ;*/
#else
          MPN_COPY_INCR(yp, xp + xsize - nw, nw);
          yp[0] &= himask;
          return 0;
#endif
        }
      else
        {
          /* rnd_mode = Away */
          while (MPFR_UNLIKELY(sb == 0) && k > 0)
            sb = xp[--k];
          if (MPFR_UNLIKELY(sb == 0))
            {
              /* sb = 0 && rnd_mode != MPFR_RNDZ */
              if (use_inexp)
                /* (neg != 0) ^ (rnd_mode != MPFR_RNDZ)) ? 1 : -1);*/
                *inexp = 0;
#if flag == 1
              return 0;
#else
              MPN_COPY_INCR(yp, xp + xsize - nw, nw);
              yp[0] &= himask;
              return 0;
#endif
            }
          else
            {
              /* sb != 0 && rnd_mode != MPFR_RNDZ */
              if (use_inexp)
                /* (neg != 0) ^ (rnd_mode != MPFR_RNDZ)) ? 1 : -1);*/
                *inexp = 1-2*neg;
#if flag == 1
              return 1;
#else
              carry = mpn_add_1(yp, xp + xsize - nw, nw,
                                rw ? MPFR_LIMB_ONE << (GMP_NUMB_BITS - rw)
                                : 1);
              yp[0] &= himask;
              return carry;
#endif
            }
        }
    }
  else
    {
      /* Roundind mode = Zero / No inexact flag */
#if flag == 1
      return 0 /*sb != 0 && rnd_mode != MPFR_RNDZ*/;
#else
      if (MPFR_LIKELY(rw))
        {
          nw++;
          himask = ~MPFR_LIMB_MASK (GMP_NUMB_BITS - rw);
        }
      else
        himask = ~(mp_limb_t) 0;
      MPN_COPY_INCR(yp, xp + xsize - nw, nw);
      yp[0] &= himask;
      return 0;
#endif
    }
}

#undef flag
#undef use_inexp
#undef mpfr_round_raw_generic
