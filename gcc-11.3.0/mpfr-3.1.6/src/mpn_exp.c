/* mpfr_mpn_exp -- auxiliary function for mpfr_get_str and mpfr_set_str

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


#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* this function computes an approximation of b^e in {a, n}, with exponent
   stored in exp_r. The computed value is rounded toward zero (truncated).
   It returns an integer f such that the final error is bounded by 2^f ulps,
   that is:
   a*2^exp_r <= b^e <= 2^exp_r (a + 2^f),
   where a represents {a, n}, i.e. the integer
   a[0] + a[1]*B + ... + a[n-1]*B^(n-1) where B=2^GMP_NUMB_BITS

   Return -1 is the result is exact.
   Return -2 if an overflow occurred in the computation of exp_r.
*/

long
mpfr_mpn_exp (mp_limb_t *a, mpfr_exp_t *exp_r, int b, mpfr_exp_t e, size_t n)
{
  mp_limb_t *c, B;
  mpfr_exp_t f, h;
  int i;
  unsigned long t; /* number of bits in e */
  unsigned long bits;
  size_t n1;
  unsigned int error;           /* (number - 1) of loop a^2b inexact */
                                 /* error == t means no error */
  int err_s_a2 = 0;
  int err_s_ab = 0;              /* number of error when shift A^2, AB */
  MPFR_TMP_DECL(marker);

  MPFR_ASSERTN(e > 0);
  MPFR_ASSERTN((2 <= b) && (b <= 62));

  MPFR_TMP_MARK(marker);

  /* initialization of a, b, f, h */

  /* normalize the base */
  B = (mp_limb_t) b;
  count_leading_zeros (h, B);

  bits = GMP_NUMB_BITS - h;

  B = B << h;
  h = - h;

  /* allocate space for A and set it to B */
  c = MPFR_TMP_LIMBS_ALLOC (2 * n);
  a [n - 1] = B;
  MPN_ZERO (a, n - 1);
  /* initial exponent for A: invariant is A = {a, n} * 2^f */
  f = h - (n - 1) * GMP_NUMB_BITS;

  /* determine number of bits in e */
  count_leading_zeros (t, (mp_limb_t) e);

  t = GMP_NUMB_BITS - t; /* number of bits of exponent e */

  error = t; /* error <= GMP_NUMB_BITS */

  MPN_ZERO (c, 2 * n);

  for (i = t - 2; i >= 0; i--)
    {

      /* determine precision needed */
      bits = n * GMP_NUMB_BITS - mpn_scan1 (a, 0);
      n1 = (n * GMP_NUMB_BITS - bits) / GMP_NUMB_BITS;

      /* square of A : {c+2n1, 2(n-n1)} = {a+n1, n-n1}^2 */
      mpn_sqr_n (c + 2 * n1, a + n1, n - n1);

      /* set {c+n, 2n1-n} to 0 : {c, n} = {a, n}^2*K^n */

      /* check overflow on f */
      if (MPFR_UNLIKELY(f < MPFR_EXP_MIN/2 || f > MPFR_EXP_MAX/2))
        {
        overflow:
          MPFR_TMP_FREE(marker);
          return -2;
        }
      /* FIXME: Could f = 2*f + n * GMP_NUMB_BITS be used? */
      f = 2*f;
      MPFR_SADD_OVERFLOW (f, f, n * GMP_NUMB_BITS,
                          mpfr_exp_t, mpfr_uexp_t,
                          MPFR_EXP_MIN, MPFR_EXP_MAX,
                          goto overflow, goto overflow);
      if ((c[2*n - 1] & MPFR_LIMB_HIGHBIT) == 0)
        {
          /* shift A by one bit to the left */
          mpn_lshift (a, c + n, n, 1);
          a[0] |= mpn_lshift (c + n - 1, c + n - 1, 1, 1);
          f --;
          if (error != t)
            err_s_a2 ++;
        }
      else
        MPN_COPY (a, c + n, n);

      if ((error == t) && (2 * n1 <= n) &&
          (mpn_scan1 (c + 2 * n1, 0) < (n - 2 * n1) * GMP_NUMB_BITS))
        error = i;

      if (e & ((mpfr_exp_t) 1 << i))
        {
          /* multiply A by B */
          c[2 * n - 1] = mpn_mul_1 (c + n - 1, a, n, B);
          f += h + GMP_NUMB_BITS;
          if ((c[2 * n - 1] & MPFR_LIMB_HIGHBIT) == 0)
            { /* shift A by one bit to the left */
              mpn_lshift (a, c + n, n, 1);
              a[0] |= mpn_lshift (c + n - 1, c + n - 1, 1, 1);
              f --;
            }
          else
            {
              MPN_COPY (a, c + n, n);
              if (error != t)
                err_s_ab ++;
            }
          if ((error == t) && (c[n - 1] != 0))
            error = i;
        }
    }

  MPFR_TMP_FREE(marker);

  *exp_r = f;

  if (error == t)
    return -1; /* result is exact */
  else /* error <= t-2 <= GMP_NUMB_BITS-2
          err_s_ab, err_s_a2 <= t-1       */
    {
      /* if there are p loops after the first inexact result, with
         j shifts in a^2 and l shifts in a*b, then the final error is
         at most 2^(p+ceil((j+1)/2)+l+1)*ulp(res).
         This is bounded by 2^(5/2*t-1/2) where t is the number of bits of e.
      */
      error = error + err_s_ab + err_s_a2 / 2 + 3; /* <= 5t/2-1/2 */
#if 0
      if ((error - 1) >= ((n * GMP_NUMB_BITS - 1) / 2))
        error = n * GMP_NUMB_BITS; /* result is completely wrong:
                                         this is very unlikely since error is
                                         at most 5/2*log_2(e), and
                                         n * GMP_NUMB_BITS is at least
                                         3*log_2(e) */
#endif
      return error;
    }
}
