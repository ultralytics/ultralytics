/* mpfr_zeta_ui -- compute the Riemann Zeta function for integer argument.

Copyright 2005-2017 Free Software Foundation, Inc.
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

int
mpfr_zeta_ui (mpfr_ptr z, unsigned long m, mpfr_rnd_t r)
{
  MPFR_ZIV_DECL (loop);

  MPFR_LOG_FUNC
    (("m=%lu rnd=%d prec=%Pu", m, r, mpfr_get_prec (z)),
     ("z[%Pu]=%.*Rg", mpfr_get_prec (z), mpfr_log_prec, z));

  if (m == 0)
    {
      return mpfr_set_si_2exp (z, -1, -1, r);
    }
  else if (m == 1)
    {
      MPFR_SET_INF (z);
      MPFR_SET_POS (z);
      mpfr_set_divby0 ();
      return 0;
    }
  else /* m >= 2 */
    {
      mpfr_prec_t p = MPFR_PREC(z);
      unsigned long n, k, err, kbits;
      mpz_t d, t, s, q;
      mpfr_t y;
      int inex;
      MPFR_SAVE_EXPO_DECL (expo);

      if (r == MPFR_RNDA)
        r = MPFR_RNDU; /* since the result is always positive */

      MPFR_SAVE_EXPO_MARK (expo);

      if (m >= p) /* 2^(-m) < ulp(1) = 2^(1-p). This means that
                     2^(-m) <= 1/2*ulp(1). We have 3^(-m)+4^(-m)+... < 2^(-m)
                     i.e. zeta(m) < 1+2*2^(-m) for m >= 3 */
        {
          if (m == 2) /* necessarily p=2 */
            inex = mpfr_set_ui_2exp (z, 13, -3, r);
          else if (r == MPFR_RNDZ || r == MPFR_RNDD ||
                   (r == MPFR_RNDN && m > p))
            {
              mpfr_set_ui (z, 1, r);
              inex = -1;
            }
          else
            {
              mpfr_set_ui (z, 1, r);
              mpfr_nextabove (z);
              inex = 1;
            }
          goto end;
        }

      /* now treat also the case where zeta(m) - (1+1/2^m) < 1/2*ulp(1),
         and the result is either 1+2^(-m) or 1+2^(-m)+2^(1-p). */
      mpfr_init2 (y, 31);

      if (m >= p / 2) /* otherwise 4^(-m) > 2^(-p) */
        {
          /* the following is a lower bound for log(3)/log(2) */
          mpfr_set_str_binary (y, "1.100101011100000000011010001110");
          mpfr_mul_ui (y, y, m, MPFR_RNDZ); /* lower bound for log2(3^m) */
          if (mpfr_cmp_ui (y, p + 2) >= 0)
            {
              mpfr_clear (y);
              mpfr_set_ui (z, 1, MPFR_RNDZ);
              mpfr_div_2ui (z, z, m, MPFR_RNDZ);
              mpfr_add_ui (z, z, 1, MPFR_RNDZ);
              if (r != MPFR_RNDU)
                inex = -1;
              else
                {
                  mpfr_nextabove (z);
                  inex = 1;
                }
              goto end;
            }
        }

      mpz_init (s);
      mpz_init (d);
      mpz_init (t);
      mpz_init (q);

      p += MPFR_INT_CEIL_LOG2(p); /* account of the n term in the error */

      p += MPFR_INT_CEIL_LOG2(p) + 15; /* initial value */

      MPFR_ZIV_INIT (loop, p);
      for(;;)
        {
          /* 0.39321985067869744 = log(2)/log(3+sqrt(8)) */
          n = 1 + (unsigned long) (0.39321985067869744 * (double) p);
          err = n + 4;

          mpfr_set_prec (y, p);

          /* computation of the d[k] */
          mpz_set_ui (s, 0);
          mpz_set_ui (t, 1);
          mpz_mul_2exp (t, t, 2 * n - 1); /* t[n] */
          mpz_set (d, t);
          for (k = n; k > 0; k--)
            {
              count_leading_zeros (kbits, k);
              kbits = GMP_NUMB_BITS - kbits;
              /* if k^m is too large, use mpz_tdiv_q */
              if (m * kbits > 2 * GMP_NUMB_BITS)
                {
                  /* if we know in advance that k^m > d, then floor(d/k^m) will
                     be zero below, so there is no need to compute k^m */
                  kbits = (kbits - 1) * m + 1;
                  /* k^m has at least kbits bits */
                  if (kbits > mpz_sizeinbase (d, 2))
                    mpz_set_ui (q, 0);
                  else
                    {
                      mpz_ui_pow_ui (q, k, m);
                      mpz_tdiv_q (q, d, q);
                    }
                }
              else /* use several mpz_tdiv_q_ui calls */
                {
                  unsigned long km = k, mm = m - 1;
                  while (mm > 0 && km < ULONG_MAX / k)
                    {
                      km *= k;
                      mm --;
                    }
                  mpz_tdiv_q_ui (q, d, km);
                  while (mm > 0)
                    {
                      km = k;
                      mm --;
                      while (mm > 0 && km < ULONG_MAX / k)
                        {
                          km *= k;
                          mm --;
                        }
                      mpz_tdiv_q_ui (q, q, km);
                    }
                }
              if (k % 2)
                mpz_add (s, s, q);
              else
                mpz_sub (s, s, q);

              /* we have d[k] = sum(t[i], i=k+1..n)
                 with t[i] = n*(n+i-1)!*4^i/(n-i)!/(2i)!
                 t[k-1]/t[k] = k*(2k-1)/(n-k+1)/(n+k-1)/2 */
#if (GMP_NUMB_BITS == 32)
#define KMAX 46341 /* max k such that k*(2k-1) < 2^32 */
#elif (GMP_NUMB_BITS == 64)
#define KMAX 3037000500
#endif
#ifdef KMAX
              if (k <= KMAX)
                mpz_mul_ui (t, t, k * (2 * k - 1));
              else
#endif
                {
                  mpz_mul_ui (t, t, k);
                  mpz_mul_ui (t, t, 2 * k - 1);
                }
              mpz_fdiv_q_2exp (t, t, 1);
              /* Warning: the test below assumes that an unsigned long
                 has no padding bits. */
              if (n < 1UL << ((sizeof(unsigned long) * CHAR_BIT) / 2))
                /* (n - k + 1) * (n + k - 1) < n^2 */
                mpz_divexact_ui (t, t, (n - k + 1) * (n + k - 1));
              else
                {
                  mpz_divexact_ui (t, t, n - k + 1);
                  mpz_divexact_ui (t, t, n + k - 1);
                }
              mpz_add (d, d, t);
            }

          /* multiply by 1/(1-2^(1-m)) = 1 + 2^(1-m) + 2^(2-m) + ... */
          mpz_fdiv_q_2exp (t, s, m - 1);
          do
            {
              err ++;
              mpz_add (s, s, t);
              mpz_fdiv_q_2exp (t, t, m - 1);
            }
          while (mpz_cmp_ui (t, 0) > 0);

          /* divide by d[n] */
          mpz_mul_2exp (s, s, p);
          mpz_tdiv_q (s, s, d);
          mpfr_set_z (y, s, MPFR_RNDN);
          mpfr_div_2ui (y, y, p, MPFR_RNDN);

          err = MPFR_INT_CEIL_LOG2 (err);

          if (MPFR_LIKELY(MPFR_CAN_ROUND (y, p - err, MPFR_PREC(z), r)))
            break;

          MPFR_ZIV_NEXT (loop, p);
        }
      MPFR_ZIV_FREE (loop);

      mpz_clear (d);
      mpz_clear (t);
      mpz_clear (q);
      mpz_clear (s);
      inex = mpfr_set (z, y, r);
      mpfr_clear (y);

    end:
      MPFR_LOG_VAR (z);
      MPFR_LOG_MSG (("inex = %d before mpfr_check_range\n", inex));
      MPFR_SAVE_EXPO_FREE (expo);
      return mpfr_check_range (z, inex, r);
    }
}
