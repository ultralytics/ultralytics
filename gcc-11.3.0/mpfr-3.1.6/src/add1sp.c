/* mpfr_add1sp -- internal function to perform a "real" addition
   All the op must have the same precision

Copyright 2004-2017 Free Software Foundation, Inc.
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

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* Check if we have to check the result of mpfr_add1sp with mpfr_add1 */
#ifdef MPFR_WANT_ASSERT
# if MPFR_WANT_ASSERT >= 2

int mpfr_add1sp2 (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
int mpfr_add1sp (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mpfr_rnd_t rnd_mode)
{
  mpfr_t tmpa, tmpb, tmpc;
  int inexb, inexc, inexact, inexact2;

  mpfr_init2 (tmpa, MPFR_PREC (a));
  mpfr_init2 (tmpb, MPFR_PREC (b));
  mpfr_init2 (tmpc, MPFR_PREC (c));

  inexb = mpfr_set (tmpb, b, MPFR_RNDN);
  MPFR_ASSERTN (inexb == 0);

  inexc = mpfr_set (tmpc, c, MPFR_RNDN);
  MPFR_ASSERTN (inexc == 0);

  inexact2 = mpfr_add1 (tmpa, tmpb, tmpc, rnd_mode);
  inexact  = mpfr_add1sp2 (a, b, c, rnd_mode);

  if (mpfr_cmp (tmpa, a) || inexact != inexact2)
    {
      fprintf (stderr, "add1 & add1sp return different values for %s\n"
               "Prec_a = %lu, Prec_b = %lu, Prec_c = %lu\nB = ",
               mpfr_print_rnd_mode (rnd_mode),
               (unsigned long) MPFR_PREC (a),
               (unsigned long) MPFR_PREC (b),
               (unsigned long) MPFR_PREC (c));
      mpfr_fprint_binary (stderr, tmpb);
      fprintf (stderr, "\nC = ");
      mpfr_fprint_binary (stderr, tmpc);
      fprintf (stderr, "\n\nadd1  : ");
      mpfr_fprint_binary (stderr, tmpa);
      fprintf (stderr, "\nadd1sp: ");
      mpfr_fprint_binary (stderr, a);
      fprintf (stderr, "\nInexact sp = %d | Inexact = %d\n",
               inexact, inexact2);
      MPFR_ASSERTN (0);
    }
  mpfr_clears (tmpa, tmpb, tmpc, (mpfr_ptr) 0);
  return inexact;
}
#  define mpfr_add1sp mpfr_add1sp2
# endif
#endif

/* Debugging support */
#ifdef DEBUG
# undef DEBUG
# define DEBUG(x) (x)
#else
# define DEBUG(x) /**/
#endif

/* compute sign(b) * (|b| + |c|)
   Returns 0 iff result is exact,
   a negative value when the result is less than the exact value,
   a positive value otherwise. */
int
mpfr_add1sp (mpfr_ptr a, mpfr_srcptr b, mpfr_srcptr c, mpfr_rnd_t rnd_mode)
{
  mpfr_uexp_t d;
  mpfr_prec_t p;
  unsigned int sh;
  mp_size_t n;
  mp_limb_t *ap, *cp;
  mpfr_exp_t bx;
  mp_limb_t limb;
  int inexact;
  MPFR_TMP_DECL(marker);

  MPFR_TMP_MARK(marker);

  MPFR_ASSERTD(MPFR_PREC(a) == MPFR_PREC(b) && MPFR_PREC(b) == MPFR_PREC(c));
  MPFR_ASSERTD(MPFR_IS_PURE_FP(b));
  MPFR_ASSERTD(MPFR_IS_PURE_FP(c));
  MPFR_ASSERTD(MPFR_GET_EXP(b) >= MPFR_GET_EXP(c));

  /* Read prec and num of limbs */
  p = MPFR_PREC(b);
  n = MPFR_PREC2LIMBS (p);
  MPFR_UNSIGNED_MINUS_MODULO(sh, p);
  bx = MPFR_GET_EXP(b);
  d = (mpfr_uexp_t) (bx - MPFR_GET_EXP(c));

  DEBUG (printf ("New add1sp with diff=%lu\n", (unsigned long) d));

  if (MPFR_UNLIKELY(d == 0))
    {
      /* d==0 */
      DEBUG( mpfr_print_mant_binary("C= ", MPFR_MANT(c), p) );
      DEBUG( mpfr_print_mant_binary("B= ", MPFR_MANT(b), p) );
      bx++;                                /* exp + 1 */
      ap = MPFR_MANT(a);
      limb = mpn_add_n(ap, MPFR_MANT(b), MPFR_MANT(c), n);
      DEBUG( mpfr_print_mant_binary("A= ", ap, p) );
      MPFR_ASSERTD(limb != 0);             /* There must be a carry */
      limb = ap[0];                        /* Get LSB (In fact, LSW) */
      mpn_rshift(ap, ap, n, 1);            /* Shift mantissa A */
      ap[n-1] |= MPFR_LIMB_HIGHBIT;        /* Set MSB */
      ap[0]   &= ~MPFR_LIMB_MASK(sh);      /* Clear LSB bit */
      if (MPFR_LIKELY((limb&(MPFR_LIMB_ONE<<sh)) == 0)) /* Check exact case */
        { inexact = 0; goto set_exponent; }
      /* Zero: Truncate
         Nearest: Even Rule => truncate or add 1
         Away: Add 1 */
      if (MPFR_LIKELY(rnd_mode==MPFR_RNDN))
        {
          if (MPFR_LIKELY((ap[0]&(MPFR_LIMB_ONE<<sh))==0))
            { inexact = -1; goto set_exponent; }
          else
            goto add_one_ulp;
        }
      MPFR_UPDATE_RND_MODE(rnd_mode, MPFR_IS_NEG(b));
      if (rnd_mode==MPFR_RNDZ)
        { inexact = -1; goto set_exponent; }
      else
        goto add_one_ulp;
    }
  else if (MPFR_UNLIKELY (d >= p))
    {
      if (MPFR_LIKELY (d > p))
        {
          /* d > p : Copy B in A */
          /* Away:    Add 1
             Nearest: Trunc
             Zero:    Trunc */
          if (MPFR_LIKELY (rnd_mode==MPFR_RNDN
                           || MPFR_IS_LIKE_RNDZ (rnd_mode, MPFR_IS_NEG (b))))
            {
            copy_set_exponent:
              ap = MPFR_MANT (a);
              MPN_COPY (ap, MPFR_MANT(b), n);
              inexact = -1;
              goto set_exponent;
            }
          else
            {
            copy_add_one_ulp:
              ap = MPFR_MANT(a);
              MPN_COPY (ap, MPFR_MANT(b), n);
              goto add_one_ulp;
            }
        }
      else
        {
          /* d==p : Copy B in A */
          /* Away:    Add 1
             Nearest: Even Rule if C is a power of 2, else Add 1
             Zero:    Trunc */
          if (MPFR_LIKELY(rnd_mode==MPFR_RNDN))
            {
              /* Check if C was a power of 2 */
              cp = MPFR_MANT(c);
              if (MPFR_UNLIKELY(cp[n-1] == MPFR_LIMB_HIGHBIT))
                {
                  mp_size_t k = n-1;
                  do {
                    k--;
                  } while (k>=0 && cp[k]==0);
                  if (MPFR_UNLIKELY(k<0))
                    /* Power of 2: Even rule */
                    if ((MPFR_MANT (b)[0]&(MPFR_LIMB_ONE<<sh))==0)
                      goto copy_set_exponent;
                }
              /* Not a Power of 2 */
              goto copy_add_one_ulp;
            }
          else if (MPFR_IS_LIKE_RNDZ (rnd_mode, MPFR_IS_NEG (b)))
            goto copy_set_exponent;
          else
            goto copy_add_one_ulp;
        }
    }
  else
    {
      mp_limb_t mask;
      mp_limb_t bcp, bcp1; /* Cp and C'p+1 */

      /* General case: 1 <= d < p */
      cp = MPFR_TMP_LIMBS_ALLOC (n);

      /* Shift c in temporary allocated place */
      {
        mpfr_uexp_t dm;
        mp_size_t m;

        dm = d % GMP_NUMB_BITS;
        m = d / GMP_NUMB_BITS;
        if (MPFR_UNLIKELY(dm == 0))
          {
            /* dm = 0 and m > 0: Just copy */
            MPFR_ASSERTD(m!=0);
            MPN_COPY(cp, MPFR_MANT(c)+m, n-m);
            MPN_ZERO(cp+n-m, m);
          }
        else if (MPFR_LIKELY(m == 0))
          {
            /* dm >=1 and m == 0: just shift */
            MPFR_ASSERTD(dm >= 1);
            mpn_rshift(cp, MPFR_MANT(c), n, dm);
          }
        else
          {
            /* dm > 0 and m > 0: shift and zero  */
            mpn_rshift(cp, MPFR_MANT(c)+m, n-m, dm);
            MPN_ZERO(cp+n-m, m);
          }
      }

      DEBUG( mpfr_print_mant_binary("Before", MPFR_MANT(c), p) );
      DEBUG( mpfr_print_mant_binary("B=    ", MPFR_MANT(b), p) );
      DEBUG( mpfr_print_mant_binary("After ", cp, p) );

      /* Compute bcp=Cp and bcp1=C'p+1 */
      if (MPFR_LIKELY (sh > 0))
        {
          /* Try to compute them from C' rather than C */
          bcp = (cp[0] & (MPFR_LIMB_ONE<<(sh-1))) ;
          if (MPFR_LIKELY(cp[0]&MPFR_LIMB_MASK(sh-1)))
            bcp1 = 1;
          else
            {
              /* We can't compute C'p+1 from C'. Compute it from C */
              /* Start from bit x=p-d+sh in mantissa C
                 (+sh since we have already looked sh bits in C'!) */
              mpfr_prec_t x = p-d+sh-1;
              if (MPFR_LIKELY(x>p))
                /* We are already looked at all the bits of c, so C'p+1 = 0*/
                bcp1 = 0;
              else
                {
                  mp_limb_t *tp = MPFR_MANT(c);
                  mp_size_t kx = n-1 - (x / GMP_NUMB_BITS);
                  mpfr_prec_t sx = GMP_NUMB_BITS-1-(x%GMP_NUMB_BITS);
                  DEBUG (printf ("(First) x=%lu Kx=%ld Sx=%lu\n",
                                 (unsigned long) x, (long) kx,
                                 (unsigned long) sx));
                  /* Looks at the last bits of limb kx (if sx=0 does nothing)*/
                  if (tp[kx] & MPFR_LIMB_MASK(sx))
                    bcp1 = 1;
                  else
                    {
                      /*kx += (sx==0);*/
                      /*If sx==0, tp[kx] hasn't been checked*/
                      do {
                        kx--;
                      } while (kx>=0 && tp[kx]==0);
                      bcp1 = (kx >= 0);
                    }
                }
            }
        }
      else /* sh == 0 */
        {
          /* Compute Cp and C'p+1 from C with sh=0 */
          mp_limb_t *tp = MPFR_MANT(c);
          /* Start from bit x=p-d in mantissa C */
          mpfr_prec_t  x = p-d;
          mp_size_t   kx = n-1 - (x / GMP_NUMB_BITS);
          mpfr_prec_t sx = GMP_NUMB_BITS-1-(x%GMP_NUMB_BITS);
          MPFR_ASSERTD(p >= d);
          bcp = tp[kx] & (MPFR_LIMB_ONE<<sx);
          /* Looks at the last bits of limb kx (If sx=0, does nothing)*/
          if (tp[kx]&MPFR_LIMB_MASK(sx))
            bcp1 = 1;
          else
            {
              do {
                kx--;
              } while (kx>=0 && tp[kx]==0);
              bcp1 = (kx>=0);
            }
        }
      DEBUG (printf("sh=%u Cp=%lu C'p+1=%lu\n", sh,
                    (unsigned long) bcp, (unsigned long) bcp1));

      /* Clean shifted C' */
      mask = ~MPFR_LIMB_MASK(sh);
      cp[0] &= mask;

      /* Add the mantissa c from b in a */
      ap = MPFR_MANT(a);
      limb = mpn_add_n (ap, MPFR_MANT(b), cp, n);
      DEBUG( mpfr_print_mant_binary("Add=  ", ap, p) );

      /* Check for overflow */
      if (MPFR_UNLIKELY (limb))
        {
          limb = ap[0] & (MPFR_LIMB_ONE<<sh); /* Get LSB */
          mpn_rshift (ap, ap, n, 1);          /* Shift mantissa*/
          bx++;                               /* Fix exponent */
          ap[n-1] |= MPFR_LIMB_HIGHBIT;       /* Set MSB */
          ap[0]   &= mask;                    /* Clear LSB bit */
          bcp1    |= bcp;                     /* Recompute C'p+1 */
          bcp      = limb;                    /* Recompute Cp */
          DEBUG (printf ("(Overflow) Cp=%lu C'p+1=%lu\n",
                         (unsigned long) bcp, (unsigned long) bcp1));
          DEBUG (mpfr_print_mant_binary ("Add=  ", ap, p));
        }

      /* Round:
          Zero: Truncate but could be exact.
          Away: Add 1 if Cp or C'p+1 !=0
          Nearest: Truncate but could be exact if Cp==0
                   Add 1 if C'p+1 !=0,
                   Even rule else */
      if (MPFR_LIKELY(rnd_mode == MPFR_RNDN))
        {
          if (MPFR_LIKELY(bcp == 0))
            { inexact = MPFR_LIKELY(bcp1) ? -1 : 0; goto set_exponent; }
          else if (MPFR_UNLIKELY(bcp1==0) && (ap[0]&(MPFR_LIMB_ONE<<sh))==0)
            { inexact = -1; goto set_exponent; }
          else
            goto add_one_ulp;
        }
      MPFR_UPDATE_RND_MODE(rnd_mode, MPFR_IS_NEG(b));
      if (rnd_mode == MPFR_RNDZ)
        {
          inexact = MPFR_LIKELY(bcp || bcp1) ? -1 : 0;
          goto set_exponent;
        }
      else
        {
          if (MPFR_UNLIKELY(bcp==0 && bcp1==0))
            { inexact = 0; goto set_exponent; }
          else
            goto add_one_ulp;
        }
    }
  MPFR_ASSERTN(0);

 add_one_ulp:
  /* add one unit in last place to a */
  DEBUG( printf("AddOneUlp\n") );
  if (MPFR_UNLIKELY( mpn_add_1(ap, ap, n, MPFR_LIMB_ONE<<sh) ))
    {
      /* Case 100000x0 = 0x1111x1 + 1*/
      DEBUG( printf("Pow of 2\n") );
      bx++;
      ap[n-1] = MPFR_LIMB_HIGHBIT;
    }
  inexact = 1;

 set_exponent:
  if (MPFR_UNLIKELY(bx > __gmpfr_emax)) /* Check for overflow */
    {
      DEBUG( printf("Overflow\n") );
      MPFR_TMP_FREE(marker);
      MPFR_SET_SAME_SIGN(a,b);
      return mpfr_overflow(a, rnd_mode, MPFR_SIGN(a));
    }
  MPFR_SET_EXP (a, bx);
  MPFR_SET_SAME_SIGN(a,b);

  MPFR_TMP_FREE(marker);
  MPFR_RET (inexact * MPFR_INT_SIGN (a));
}
