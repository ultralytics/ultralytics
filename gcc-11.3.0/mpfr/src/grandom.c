/* mpfr_grandom (rop1, rop2, state, rnd_mode) -- Generate up to two
   pseudorandom real numbers according to a standard normal gaussian
   distribution and round it to the precision of rop1, rop2 according
   to the given rounding mode.

Copyright 2011-2017 Free Software Foundation, Inc.
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


/* #define MPFR_NEED_LONGLONG_H */
#include "mpfr-impl.h"


int
mpfr_grandom (mpfr_ptr rop1, mpfr_ptr rop2, gmp_randstate_t rstate,
              mpfr_rnd_t rnd)
{
  int inex1, inex2, s1, s2;
  mpz_t x, y, xp, yp, t, a, b, s;
  mpfr_t sfr, l, r1, r2;
  mpfr_prec_t tprec, tprec0;

  inex2 = inex1 = 0;

  if (rop2 == NULL) /* only one output requested. */
    {
      tprec0 = MPFR_PREC (rop1);
    }
  else
    {
      tprec0 = MAX (MPFR_PREC (rop1), MPFR_PREC (rop2));
    }

  tprec0 += 11;

  /* We use "Marsaglia polar method" here (cf.
     George Marsaglia, Normal (Gaussian) random variables for supercomputers
     The Journal of Supercomputing, Volume 5, Number 1, 49–55
     DOI: 10.1007/BF00155857).

     First we draw uniform x and y in [0,1] using mpz_urandomb (in
     fixed precision), and scale them to [-1, 1].
  */

  mpz_init (xp);
  mpz_init (yp);
  mpz_init (x);
  mpz_init (y);
  mpz_init (t);
  mpz_init (s);
  mpz_init (a);
  mpz_init (b);
  mpfr_init2 (sfr, MPFR_PREC_MIN);
  mpfr_init2 (l, MPFR_PREC_MIN);
  mpfr_init2 (r1, MPFR_PREC_MIN);
  if (rop2 != NULL)
    mpfr_init2 (r2, MPFR_PREC_MIN);

  mpz_set_ui (xp, 0);
  mpz_set_ui (yp, 0);

  for (;;)
    {
      tprec = tprec0;
      do
        {
          mpz_urandomb (xp, rstate, tprec);
          mpz_urandomb (yp, rstate, tprec);
          mpz_mul (a, xp, xp);
          mpz_mul (b, yp, yp);
          mpz_add (s, a, b);
        }
      while (mpz_sizeinbase (s, 2) > tprec * 2); /* x^2 + y^2 <= 2^{2tprec} */

      for (;;)
        {
          /* FIXME: compute s as s += 2x + 2y + 2 */
          mpz_add_ui (a, xp, 1);
          mpz_add_ui (b, yp, 1);
          mpz_mul (a, a, a);
          mpz_mul (b, b, b);
          mpz_add (s, a, b);
          if ((mpz_sizeinbase (s, 2) <= 2 * tprec) ||
              ((mpz_sizeinbase (s, 2) == 2 * tprec + 1) &&
               (mpz_scan1 (s, 0) == 2 * tprec)))
            goto yeepee;
          /* Extend by 32 bits */
          mpz_mul_2exp (xp, xp, 32);
          mpz_mul_2exp (yp, yp, 32);
          mpz_urandomb (x, rstate, 32);
          mpz_urandomb (y, rstate, 32);
          mpz_add (xp, xp, x);
          mpz_add (yp, yp, y);
          tprec += 32;

          mpz_mul (a, xp, xp);
          mpz_mul (b, yp, yp);
          mpz_add (s, a, b);
          if (mpz_sizeinbase (s, 2) > tprec * 2)
            break;
        }
    }
 yeepee:

  /* FIXME: compute s with s -= 2x + 2y + 2 */
  mpz_mul (a, xp, xp);
  mpz_mul (b, yp, yp);
  mpz_add (s, a, b);
  /* Compute the signs of the output */
  mpz_urandomb (x, rstate, 2);
  s1 = mpz_tstbit (x, 0);
  s2 = mpz_tstbit (x, 1);
  for (;;)
    {
      /* s = xp^2 + yp^2 (loop invariant) */
      mpfr_set_prec (sfr, 2 * tprec);
      mpfr_set_prec (l, tprec);
      mpfr_set_z (sfr, s, MPFR_RNDN); /* exact */
      mpfr_mul_2si (sfr, sfr, -2 * tprec, MPFR_RNDN); /* exact */
      mpfr_log (l, sfr, MPFR_RNDN);
      mpfr_neg (l, l, MPFR_RNDN);
      mpfr_mul_2si (l, l, 1, MPFR_RNDN);
      mpfr_div (l, l, sfr, MPFR_RNDN);
      mpfr_sqrt (l, l, MPFR_RNDN);

      mpfr_set_prec (r1, tprec);
      mpfr_mul_z (r1, l, xp, MPFR_RNDN);
      mpfr_div_2ui (r1, r1, tprec, MPFR_RNDN); /* exact */
      if (s1)
        mpfr_neg (r1, r1, MPFR_RNDN);
      if (MPFR_CAN_ROUND (r1, tprec - 2, MPFR_PREC (rop1), rnd))
        {
          if (rop2 != NULL)
            {
              mpfr_set_prec (r2, tprec);
              mpfr_mul_z (r2, l, yp, MPFR_RNDN);
              mpfr_div_2ui (r2, r2, tprec, MPFR_RNDN); /* exact */
              if (s2)
                mpfr_neg (r2, r2, MPFR_RNDN);
              if (MPFR_CAN_ROUND (r2, tprec - 2, MPFR_PREC (rop2), rnd))
                break;
            }
          else
            break;
        }
      /* Extend by 32 bits */
      mpz_mul_2exp (xp, xp, 32);
      mpz_mul_2exp (yp, yp, 32);
      mpz_urandomb (x, rstate, 32);
      mpz_urandomb (y, rstate, 32);
      mpz_add (xp, xp, x);
      mpz_add (yp, yp, y);
      tprec += 32;
      mpz_mul (a, xp, xp);
      mpz_mul (b, yp, yp);
      mpz_add (s, a, b);
    }
  inex1 = mpfr_set (rop1, r1, rnd);
  if (rop2 != NULL)
    {
      inex2 = mpfr_set (rop2, r2, rnd);
      inex2 = mpfr_check_range (rop2, inex2, rnd);
    }
  inex1 = mpfr_check_range (rop1, inex1, rnd);

  if (rop2 != NULL)
    mpfr_clear (r2);
  mpfr_clear (r1);
  mpfr_clear (l);
  mpfr_clear (sfr);
  mpz_clear (b);
  mpz_clear (a);
  mpz_clear (s);
  mpz_clear (t);
  mpz_clear (y);
  mpz_clear (x);
  mpz_clear (yp);
  mpz_clear (xp);

  return INEX (inex1, inex2);
}
