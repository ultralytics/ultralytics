/* mpfr_cmp2 -- exponent shift when subtracting two numbers.

Copyright 1999-2004, 2006-2017 Free Software Foundation, Inc.
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

/* If |b| != |c|, puts the number of canceled bits when one subtracts |c|
   from |b| in *cancel. Returns the sign of the difference (-1, 0, 1).

   Assumes neither of b or c is NaN, +/- infinity, or +/- 0.

   In other terms, if |b| != |c|, mpfr_cmp2 (b, c) returns
   EXP(max(|b|,|c|)) - EXP(|b| - |c|).
*/

int
mpfr_cmp2 (mpfr_srcptr b, mpfr_srcptr c, mpfr_prec_t *cancel)
{
  mp_limb_t *bp, *cp, bb, cc = 0, lastc = 0, dif, high_dif = 0;
  mp_size_t bn, cn;
  mpfr_uexp_t diff_exp;
  mpfr_prec_t res = 0;
  int sign;

  /* b=c should not happen, since cmp2 is called only from agm (with
     different variables) and from sub1 (if b=c, then sub1sp would be
     called instead). So, no need for a particular optimization here. */

  /* the cases b=0 or c=0 are also treated apart in agm and sub
     (which calls sub1) */
  MPFR_ASSERTD (MPFR_IS_PURE_FP(b));
  MPFR_ASSERTD (MPFR_IS_PURE_FP(c));

  if (MPFR_GET_EXP (b) >= MPFR_GET_EXP (c))
    {
      sign = 1;
      diff_exp = (mpfr_uexp_t) MPFR_GET_EXP (b) - MPFR_GET_EXP (c);

      bp = MPFR_MANT(b);
      cp = MPFR_MANT(c);

      bn = (MPFR_PREC(b) - 1) / GMP_NUMB_BITS;
      cn = (MPFR_PREC(c) - 1) / GMP_NUMB_BITS; /* # of limbs of c minus 1 */

      if (MPFR_UNLIKELY( diff_exp == 0 ))
        {
          while (bn >= 0 && cn >= 0 && bp[bn] == cp[cn])
            {
              bn--;
              cn--;
              res += GMP_NUMB_BITS;
            }

          if (MPFR_UNLIKELY (bn < 0))
            {
              if (MPFR_LIKELY (cn < 0)) /* b = c */
                return 0;

              bp = cp;
              bn = cn;
              cn = -1;
              sign = -1;
            }

          if (MPFR_UNLIKELY (cn < 0))
            /* c discards exactly the upper part of b */
            {
              unsigned int z;

              MPFR_ASSERTD (bn >= 0);

              while (bp[bn] == 0)
                {
                  if (--bn < 0) /* b = c */
                    return 0;
                  res += GMP_NUMB_BITS;
                }

              count_leading_zeros(z, bp[bn]); /* bp[bn] <> 0 */
              *cancel = res + z;
              return sign;
            }

          MPFR_ASSERTD (bn >= 0);
          MPFR_ASSERTD (cn >= 0);
          MPFR_ASSERTD (bp[bn] != cp[cn]);
          if (bp[bn] < cp[cn])
            {
              mp_limb_t *tp;
              mp_size_t tn;

              tp = bp; bp = cp; cp = tp;
              tn = bn; bn = cn; cn = tn;
              sign = -1;
            }
        }
    } /* MPFR_EXP(b) >= MPFR_EXP(c) */
  else /* MPFR_EXP(b) < MPFR_EXP(c) */
    {
      sign = -1;
      diff_exp = (mpfr_uexp_t) MPFR_GET_EXP (c) - MPFR_GET_EXP (b);

      bp = MPFR_MANT(c);
      cp = MPFR_MANT(b);

      bn = (MPFR_PREC(c) - 1) / GMP_NUMB_BITS;
      cn = (MPFR_PREC(b) - 1) / GMP_NUMB_BITS;
    }

  /* now we have removed the identical upper limbs of b and c
     (can happen only when diff_exp = 0), and after the possible
     swap, we have |b| > |c|: bp[bn] > cc, bn >= 0, cn >= 0,
     diff_exp = EXP(b) - EXP(c).
  */

  if (MPFR_LIKELY (diff_exp < GMP_NUMB_BITS))
    {
      cc = cp[cn] >> diff_exp;
      /* warning: a shift by GMP_NUMB_BITS may give wrong results */
      if (diff_exp)
        lastc = cp[cn] << (GMP_NUMB_BITS - diff_exp);
      cn--;
    }
  else
    diff_exp -= GMP_NUMB_BITS; /* cc = 0 */

  dif = bp[bn--] - cc; /* necessarily dif >= 1 */
  MPFR_ASSERTD(dif >= 1);

  /* now high_dif = 0, dif >= 1, lastc is the neglected part of cp[cn+1] */

  while (MPFR_UNLIKELY ((cn >= 0 || lastc != 0)
                        && (high_dif == 0) && (dif == 1)))
    { /* dif=1 implies diff_exp = 0 or 1 */
      bb = (bn >= 0) ? bp[bn] : 0;
      cc = lastc;
      if (cn >= 0)
        {
          if (diff_exp == 0)
            {
              cc += cp[cn];
            }
          else /* diff_exp = 1 */
            {
              cc += cp[cn] >> 1;
              lastc = cp[cn] << (GMP_NUMB_BITS - 1);
            }
        }
      else
        lastc = 0;
      high_dif = 1 - mpn_sub_n (&dif, &bb, &cc, 1);
      bn--;
      cn--;
      res += GMP_NUMB_BITS;
    }

  /* (cn<0 and lastc=0) or (high_dif,dif)<>(0,1) */

  if (MPFR_UNLIKELY (high_dif != 0)) /* high_dif == 1 */
    {
      res--;
      MPFR_ASSERTD (res >= 0);
      if (dif != 0)
        {
          *cancel = res;
          return sign;
        }
    }
  else /* high_dif == 0 */
    {
      unsigned int z;

      count_leading_zeros(z, dif); /* dif > 1 here */
      res += z;
      if (MPFR_LIKELY(dif != (MPFR_LIMB_ONE << (GMP_NUMB_BITS - z - 1))))
        { /* dif is not a power of two */
          *cancel = res;
          return sign;
        }
    }

  /* now result is res + (low(b) < low(c)) */
  while (MPFR_UNLIKELY (bn >= 0 && (cn >= 0 || lastc != 0)))
    {
      if (diff_exp >= GMP_NUMB_BITS)
        diff_exp -= GMP_NUMB_BITS;
      else
        {
          cc = lastc;
          if (cn >= 0)
            {
              cc += cp[cn] >> diff_exp;
              if (diff_exp != 0)
                lastc = cp[cn] << (GMP_NUMB_BITS - diff_exp);
            }
          else
            lastc = 0;
          cn--;
        }
      if (bp[bn] != cc)
        {
          *cancel = res + (bp[bn] < cc);
          return sign;
        }
      bn--;
    }

  if (bn < 0)
    {
      if (lastc != 0)
        res++;
      else
        {
          while (cn >= 0 && cp[cn] == 0)
            cn--;
          if (cn >= 0)
            res++;
        }
    }

  *cancel = res;
  return sign;
}
