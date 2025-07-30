/* mpc_pow -- Raise a complex number to the power of another complex number.

Copyright (C) 2009, 2010, 2011, 2012 INRIA

This file is part of GNU MPC.

GNU MPC is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

GNU MPC is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see http://www.gnu.org/licenses/ .
*/

#include <stdio.h> /* for MPC_ASSERT */
#include "mpc-impl.h"

/* Return non-zero iff c+i*d is an exact square (a+i*b)^2,
   with a, b both of the form m*2^e with m, e integers.
   If so, returns in a+i*b the corresponding square root, with a >= 0.
   The variables a, b must not overlap with c, d.

   We have c = a^2 - b^2 and d = 2*a*b.

   If one of a, b is exact, then both are (see algorithms.tex).

   Case 1: a <> 0 and b <> 0.
   Let a = m*2^e and b = n*2^f with m, e, n, f integers, m and n odd
   (we will treat apart the case a = 0 or b = 0).
   Then 2*a*b = m*n*2^(e+f+1), thus necessarily e+f >= -1.
   Assume e < 0, then f >= 0, then a^2 - b^2 = m^2*2^(2e) - n^2*2^(2f) cannot
   be an integer, since n^2*2^(2f) is an integer, and m^2*2^(2e) is not.
   Similarly when f < 0 (and thus e >= 0).
   Thus we have e, f >= 0, and a, b are both integers.
   Let A = 2a^2, then eliminating b between c = a^2 - b^2 and d = 2*a*b
   gives A^2 - 2c*A - d^2 = 0, which has solutions c +/- sqrt(c^2+d^2).
   We thus need c^2+d^2 to be a square, and c + sqrt(c^2+d^2) --- the solution
   we are interested in --- to be two times a square. Then b = d/(2a) is
   necessarily an integer.

   Case 2: a = 0. Then d is necessarily zero, thus it suffices to check
   whether c = -b^2, i.e., if -c is a square.

   Case 3: b = 0. Then d is necessarily zero, thus it suffices to check
   whether c = a^2, i.e., if c is a square.
*/
static int
mpc_perfect_square_p (mpz_t a, mpz_t b, mpz_t c, mpz_t d)
{
  if (mpz_cmp_ui (d, 0) == 0) /* case a = 0 or b = 0 */
    {
      /* necessarily c < 0 here, since we have already considered the case
         where x is real non-negative and y is real */
      MPC_ASSERT (mpz_cmp_ui (c, 0) < 0);
      mpz_neg (b, c);
      if (mpz_perfect_square_p (b)) /* case 2 above */
        {
          mpz_sqrt (b, b);
          mpz_set_ui (a, 0);
          return 1; /* c + i*d = (0 + i*b)^2 */
        }
    }
  else /* both a and b are non-zero */
    {
      if (mpz_divisible_2exp_p (d, 1) == 0)
        return 0; /* d must be even */
      mpz_mul (a, c, c);
      mpz_addmul (a, d, d); /* c^2 + d^2 */
      if (mpz_perfect_square_p (a))
        {
          mpz_sqrt (a, a);
          mpz_add (a, c, a); /* c + sqrt(c^2+d^2) */
          if (mpz_divisible_2exp_p (a, 1))
            {
              mpz_tdiv_q_2exp (a, a, 1);
              if (mpz_perfect_square_p (a))
                {
                  mpz_sqrt (a, a);
                  mpz_tdiv_q_2exp (b, d, 1); /* d/2 */
                  mpz_divexact (b, b, a); /* d/(2a) */
                  return 1;
                }
            }
        }
    }
  return 0; /* not a square */
}

/* fix the sign of Re(z) or Im(z) in case it is zero,
   and Re(x) is zero.
   sign_eps is 0 if Re(x) = +0, 1 if Re(x) = -0
   sign_a is the sign bit of Im(x).
   Assume y is an integer (does nothing otherwise).
*/
static void
fix_sign (mpc_ptr z, int sign_eps, int sign_a, mpfr_srcptr y)
{
  int ymod4 = -1;
  mpfr_exp_t ey;
  mpz_t my;
  unsigned long int t;

  mpz_init (my);

  ey = mpfr_get_z_exp (my, y);
  /* normalize so that my is odd */
  t = mpz_scan1 (my, 0);
  ey += (mpfr_exp_t) t;
  mpz_tdiv_q_2exp (my, my, t);
  /* y = my*2^ey */

  /* compute y mod 4 (in case y is an integer) */
  if (ey >= 2)
    ymod4 = 0;
  else if (ey == 1)
    ymod4 = mpz_tstbit (my, 0) * 2; /* correct if my < 0 */
  else if (ey == 0)
    {
      ymod4 = mpz_tstbit (my, 1) * 2 + mpz_tstbit (my, 0);
      if (mpz_cmp_ui (my , 0) < 0)
        ymod4 = 4 - ymod4;
    }
  else /* y is not an integer */
    goto end;

  if (mpfr_zero_p (mpc_realref(z)))
    {
      /* we assume y is always integer in that case (FIXME: prove it):
         (eps+I*a)^y = +0 + I*a^y for y = 1 mod 4 and sign_eps = 0
         (eps+I*a)^y = -0 - I*a^y for y = 3 mod 4 and sign_eps = 0 */
      MPC_ASSERT (ymod4 == 1 || ymod4 == 3);
      if ((ymod4 == 3 && sign_eps == 0) ||
          (ymod4 == 1 && sign_eps == 1))
        mpfr_neg (mpc_realref(z), mpc_realref(z), GMP_RNDZ);
    }
  else if (mpfr_zero_p (mpc_imagref(z)))
    {
      /* we assume y is always integer in that case (FIXME: prove it):
         (eps+I*a)^y =  a^y - 0*I for y = 0 mod 4 and sign_a = sign_eps
         (eps+I*a)^y =  -a^y +0*I for y = 2 mod 4 and sign_a = sign_eps */
      MPC_ASSERT (ymod4 == 0 || ymod4 == 2);
      if ((ymod4 == 0 && sign_a == sign_eps) ||
          (ymod4 == 2 && sign_a != sign_eps))
        mpfr_neg (mpc_imagref(z), mpc_imagref(z), GMP_RNDZ);
    }

 end:
  mpz_clear (my);
}

/* If x^y is exactly representable (with maybe a larger precision than z),
   round it in z and return the (mpc) inexact flag in [0, 10].

   If x^y is not exactly representable, return -1.

   If intermediate computations lead to numbers of more than maxprec bits,
   then abort and return -2 (in that case, to avoid loops, mpc_pow_exact
   should be called again with a larger value of maxprec).

   Assume one of Re(x) or Im(x) is non-zero, and y is non-zero (y is real).

   Warning: z and x might be the same variable, same for Re(z) or Im(z) and y.

   In case -1 or -2 is returned, z is not modified.
*/
static int
mpc_pow_exact (mpc_ptr z, mpc_srcptr x, mpfr_srcptr y, mpc_rnd_t rnd,
               mpfr_prec_t maxprec)
{
  mpfr_exp_t ec, ed, ey;
  mpz_t my, a, b, c, d, u;
  unsigned long int t;
  int ret = -2;
  int sign_rex = mpfr_signbit (mpc_realref(x));
  int sign_imx = mpfr_signbit (mpc_imagref(x));
  int x_imag = mpfr_zero_p (mpc_realref(x));
  int z_is_y = 0;
  mpfr_t copy_of_y;

  if (mpc_realref (z) == y || mpc_imagref (z) == y)
    {
      z_is_y = 1;
      mpfr_init2 (copy_of_y, mpfr_get_prec (y));
      mpfr_set (copy_of_y, y, GMP_RNDN);
    }

  mpz_init (my);
  mpz_init (a);
  mpz_init (b);
  mpz_init (c);
  mpz_init (d);
  mpz_init (u);

  ey = mpfr_get_z_exp (my, y);
  /* normalize so that my is odd */
  t = mpz_scan1 (my, 0);
  ey += (mpfr_exp_t) t;
  mpz_tdiv_q_2exp (my, my, t);
  /* y = my*2^ey with my odd */

  if (x_imag)
    {
      mpz_set_ui (c, 0);
      ec = 0;
    }
  else
    ec = mpfr_get_z_exp (c, mpc_realref(x));
  if (mpfr_zero_p (mpc_imagref(x)))
    {
      mpz_set_ui (d, 0);
      ed = ec;
    }
  else
    {
      ed = mpfr_get_z_exp (d, mpc_imagref(x));
      if (x_imag)
        ec = ed;
    }
  /* x = c*2^ec + I * d*2^ed */
  /* equalize the exponents of x */
  if (ec < ed)
    {
      mpz_mul_2exp (d, d, (unsigned long int) (ed - ec));
      if ((mpfr_prec_t) mpz_sizeinbase (d, 2) > maxprec)
        goto end;
    }
  else if (ed < ec)
    {
      mpz_mul_2exp (c, c, (unsigned long int) (ec - ed));
      if ((mpfr_prec_t) mpz_sizeinbase (c, 2) > maxprec)
        goto end;
      ec = ed;
    }
  /* now ec=ed and x = (c + I * d) * 2^ec */

  /* divide by two if possible */
  if (mpz_cmp_ui (c, 0) == 0)
    {
      t = mpz_scan1 (d, 0);
      mpz_tdiv_q_2exp (d, d, t);
      ec += (mpfr_exp_t) t;
    }
  else if (mpz_cmp_ui (d, 0) == 0)
    {
      t = mpz_scan1 (c, 0);
      mpz_tdiv_q_2exp (c, c, t);
      ec += (mpfr_exp_t) t;
    }
  else /* neither c nor d is zero */
    {
      unsigned long v;
      t = mpz_scan1 (c, 0);
      v = mpz_scan1 (d, 0);
      if (v < t)
        t = v;
      mpz_tdiv_q_2exp (c, c, t);
      mpz_tdiv_q_2exp (d, d, t);
      ec += (mpfr_exp_t) t;
    }

  /* now either one of c, d is odd */

  while (ey < 0)
    {
      /* check if x is a square */
      if (ec & 1)
        {
          mpz_mul_2exp (c, c, 1);
          mpz_mul_2exp (d, d, 1);
          ec --;
        }
      /* now ec is even */
      if (mpc_perfect_square_p (a, b, c, d) == 0)
        break;
      mpz_swap (a, c);
      mpz_swap (b, d);
      ec /= 2;
      ey ++;
    }

  if (ey < 0)
    {
      ret = -1; /* not representable */
      goto end;
    }

  /* Now ey >= 0, it thus suffices to check that x^my is representable.
     If my > 0, this is always true. If my < 0, we first try to invert
     (c+I*d)*2^ec.
  */
  if (mpz_cmp_ui (my, 0) < 0)
    {
      /* If my < 0, 1 / (c + I*d) = (c - I*d)/(c^2 + d^2), thus a sufficient
         condition is that c^2 + d^2 is a power of two, assuming |c| <> |d|.
         Assume a prime p <> 2 divides c^2 + d^2,
         then if p does not divide c or d, 1 / (c + I*d) cannot be exact.
         If p divides both c and d, then we can write c = p*c', d = p*d',
         and 1 / (c + I*d) = 1/p * 1/(c' + I*d'). This shows that if 1/(c+I*d)
         is exact, then 1/(c' + I*d') is exact too, and we are back to the
         previous case. In conclusion, a necessary and sufficient condition
         is that c^2 + d^2 is a power of two.
      */
      /* FIXME: we could first compute c^2+d^2 mod a limb for example */
      mpz_mul (a, c, c);
      mpz_addmul (a, d, d);
      t = mpz_scan1 (a, 0);
      if (mpz_sizeinbase (a, 2) != 1 + t) /* a is not a power of two */
        {
          ret = -1; /* not representable */
          goto end;
        }
      /* replace (c,d) by (c/(c^2+d^2), -d/(c^2+d^2)) */
      mpz_neg (d, d);
      ec = -ec - (mpfr_exp_t) t;
      mpz_neg (my, my);
    }

  /* now ey >= 0 and my >= 0, and we want to compute
     [(c + I * d) * 2^ec] ^ (my * 2^ey).

     We first compute [(c + I * d) * 2^ec]^my, then square ey times. */
  t = mpz_sizeinbase (my, 2) - 1;
  mpz_set (a, c);
  mpz_set (b, d);
  ed = ec;
  /* invariant: (a + I*b) * 2^ed = ((c + I*d) * 2^ec)^trunc(my/2^t) */
  while (t-- > 0)
    {
      unsigned long int v, w;
      /* square a + I*b */
      mpz_mul (u, a, b);
      mpz_mul (a, a, a);
      mpz_submul (a, b, b);
      mpz_mul_2exp (b, u, 1);
      ed *= 2;
      if (mpz_tstbit (my, t)) /* multiply by c + I*d */
        {
          mpz_mul (u, a, c);
          mpz_submul (u, b, d); /* ac-bd */
          mpz_mul (b, b, c);
          mpz_addmul (b, a, d); /* bc+ad */
          mpz_swap (a, u);
          ed += ec;
        }
      /* remove powers of two in (a,b) */
      if (mpz_cmp_ui (a, 0) == 0)
        {
          w = mpz_scan1 (b, 0);
          mpz_tdiv_q_2exp (b, b, w);
          ed += (mpfr_exp_t) w;
        }
      else if (mpz_cmp_ui (b, 0) == 0)
        {
          w = mpz_scan1 (a, 0);
          mpz_tdiv_q_2exp (a, a, w);
          ed += (mpfr_exp_t) w;
        }
      else
        {
          w = mpz_scan1 (a, 0);
          v = mpz_scan1 (b, 0);
          if (v < w)
            w = v;
          mpz_tdiv_q_2exp (a, a, w);
          mpz_tdiv_q_2exp (b, b, w);
          ed += (mpfr_exp_t) w;
        }
      if (   (mpfr_prec_t) mpz_sizeinbase (a, 2) > maxprec
          || (mpfr_prec_t) mpz_sizeinbase (b, 2) > maxprec)
        goto end;
    }
  /* now a+I*b = (c+I*d)^my */

  while (ey-- > 0)
    {
      unsigned long sa, sb;

      /* square a + I*b */
      mpz_mul (u, a, b);
      mpz_mul (a, a, a);
      mpz_submul (a, b, b);
      mpz_mul_2exp (b, u, 1);
      ed *= 2;

      /* divide by largest 2^n possible, to avoid many loops for e.g.,
         (2+2*I)^16777216 */
      sa = mpz_scan1 (a, 0);
      sb = mpz_scan1 (b, 0);
      sa = (sa <= sb) ? sa : sb;
      mpz_tdiv_q_2exp (a, a, sa);
      mpz_tdiv_q_2exp (b, b, sa);
      ed += (mpfr_exp_t) sa;

      if (   (mpfr_prec_t) mpz_sizeinbase (a, 2) > maxprec
          || (mpfr_prec_t) mpz_sizeinbase (b, 2) > maxprec)
        goto end;
    }

  ret = mpfr_set_z (mpc_realref(z), a, MPC_RND_RE(rnd));
  ret = MPC_INEX(ret, mpfr_set_z (mpc_imagref(z), b, MPC_RND_IM(rnd)));
  mpfr_mul_2si (mpc_realref(z), mpc_realref(z), ed, MPC_RND_RE(rnd));
  mpfr_mul_2si (mpc_imagref(z), mpc_imagref(z), ed, MPC_RND_IM(rnd));

 end:
  mpz_clear (my);
  mpz_clear (a);
  mpz_clear (b);
  mpz_clear (c);
  mpz_clear (d);
  mpz_clear (u);

  if (ret >= 0 && x_imag)
    fix_sign (z, sign_rex, sign_imx, (z_is_y) ? copy_of_y : y);

  if (z_is_y)
    mpfr_clear (copy_of_y);

  return ret;
}

/* Return 1 if y*2^k is an odd integer, 0 otherwise.
   Adapted from MPFR, file pow.c.

   Examples: with k=0, check if y is an odd integer,
             with k=1, check if y is half-an-integer,
             with k=-1, check if y/2 is an odd integer.
*/
#define MPFR_LIMB_HIGHBIT ((mp_limb_t) 1 << (BITS_PER_MP_LIMB - 1))
static int
is_odd (mpfr_srcptr y, mpfr_exp_t k)
{
  mpfr_exp_t expo;
  mpfr_prec_t prec;
  mp_size_t yn;
  mp_limb_t *yp;

  expo = mpfr_get_exp (y) + k;
  if (expo <= 0)
    return 0;  /* |y| < 1 and not 0 */

  prec = mpfr_get_prec (y);
  if ((mpfr_prec_t) expo > prec)
    return 0;  /* y is a multiple of 2^(expo-prec), thus not odd */

  /* 0 < expo <= prec:
     y = 1xxxxxxxxxt.zzzzzzzzzzzzzzzzzz[000]
          expo bits   (prec-expo) bits

     We have to check that:
     (a) the bit 't' is set
     (b) all the 'z' bits are zero
  */

  prec = ((prec - 1) / BITS_PER_MP_LIMB + 1) * BITS_PER_MP_LIMB - expo;
  /* number of z+0 bits */

  yn = prec / BITS_PER_MP_LIMB;
  /* yn is the index of limb containing the 't' bit */

  yp = y->_mpfr_d;
  /* if expo is a multiple of BITS_PER_MP_LIMB, t is bit 0 */
  if (expo % BITS_PER_MP_LIMB == 0 ? (yp[yn] & 1) == 0
      : yp[yn] << ((expo % BITS_PER_MP_LIMB) - 1) != MPFR_LIMB_HIGHBIT)
    return 0;
  while (--yn >= 0)
    if (yp[yn] != 0)
      return 0;
  return 1;
}

/* Put in z the value of x^y, rounded according to 'rnd'.
   Return the inexact flag in [0, 10]. */
int
mpc_pow (mpc_ptr z, mpc_srcptr x, mpc_srcptr y, mpc_rnd_t rnd)
{
  int ret = -2, loop, x_real, x_imag, y_real, z_real = 0, z_imag = 0;
  mpc_t t, u;
  mpfr_prec_t p, pr, pi, maxprec;
  int saved_underflow, saved_overflow;
  
  /* save the underflow or overflow flags from MPFR */
  saved_underflow = mpfr_underflow_p ();
  saved_overflow = mpfr_overflow_p ();

  x_real = mpfr_zero_p (mpc_imagref(x));
  y_real = mpfr_zero_p (mpc_imagref(y));

  if (y_real && mpfr_zero_p (mpc_realref(y))) /* case y zero */
    {
      if (x_real && mpfr_zero_p (mpc_realref(x)))
        {
          /* we define 0^0 to be (1, +0) since the real part is
             coherent with MPFR where 0^0 gives 1, and the sign of the
             imaginary part cannot be determined                       */
          mpc_set_ui_ui (z, 1, 0, MPC_RNDNN);
          return 0;
        }
      else /* x^0 = 1 +/- i*0 even for x=NaN see algorithms.tex for the
              sign of zero */
        {
          mpfr_t n;
          int inex, cx1;
          int sign_zi;
          /* cx1 < 0 if |x| < 1
             cx1 = 0 if |x| = 1
             cx1 > 0 if |x| > 1
          */
          mpfr_init (n);
          inex = mpc_norm (n, x, GMP_RNDN);
          cx1 = mpfr_cmp_ui (n, 1);
          if (cx1 == 0 && inex != 0)
            cx1 = -inex;

          sign_zi = (cx1 < 0 && mpfr_signbit (mpc_imagref (y)) == 0)
            || (cx1 == 0
                && mpfr_signbit (mpc_imagref (x)) != mpfr_signbit (mpc_realref (y)))
            || (cx1 > 0 && mpfr_signbit (mpc_imagref (y)));

          /* warning: mpc_set_ui_ui does not set Im(z) to -0 if Im(rnd)=RNDD */
          ret = mpc_set_ui_ui (z, 1, 0, rnd);

          if (MPC_RND_IM (rnd) == GMP_RNDD || sign_zi)
            mpc_conj (z, z, MPC_RNDNN);

          mpfr_clear (n);
          return ret;
        }
    }

  if (!mpc_fin_p (x) || !mpc_fin_p (y))
    {
      /* special values: exp(y*log(x)) */
      mpc_init2 (u, 2);
      mpc_log (u, x, MPC_RNDNN);
      mpc_mul (u, u, y, MPC_RNDNN);
      ret = mpc_exp (z, u, rnd);
      mpc_clear (u);
      goto end;
    }

  if (x_real) /* case x real */
    {
      if (mpfr_zero_p (mpc_realref(x))) /* x is zero */
        {
          /* special values: exp(y*log(x)) */
          mpc_init2 (u, 2);
          mpc_log (u, x, MPC_RNDNN);
          mpc_mul (u, u, y, MPC_RNDNN);
          ret = mpc_exp (z, u, rnd);
          mpc_clear (u);
          goto end;
        }

      /* Special case 1^y = 1 */
      if (mpfr_cmp_ui (mpc_realref(x), 1) == 0)
        {
          int s1, s2;
          s1 = mpfr_signbit (mpc_realref (y));
          s2 = mpfr_signbit (mpc_imagref (x));

          ret = mpc_set_ui (z, +1, rnd);
          /* the sign of the zero imaginary part is known in some cases (see
             algorithm.tex). In such cases we have
             (x +s*0i)^(y+/-0i) = x^y + s*sign(y)*0i
             where s = +/-1.  We extend here this rule to fix the sign of the
             zero part.

             Note that the sign must also be set explicitly when rnd=RNDD
             because mpfr_set_ui(z_i, 0, rnd) always sets z_i to +0.
          */
          if (MPC_RND_IM (rnd) == GMP_RNDD || s1 != s2)
            mpc_conj (z, z, MPC_RNDNN);
          goto end;
        }

      /* x^y is real when:
         (a) x is real and y is integer
         (b) x is real non-negative and y is real */
      if (y_real && (mpfr_integer_p (mpc_realref(y)) ||
                     mpfr_cmp_ui (mpc_realref(x), 0) >= 0))
        {
          int s1, s2;
          s1 = mpfr_signbit (mpc_realref (y));
          s2 = mpfr_signbit (mpc_imagref (x));

          ret = mpfr_pow (mpc_realref(z), mpc_realref(x), mpc_realref(y), MPC_RND_RE(rnd));
          ret = MPC_INEX(ret, mpfr_set_ui (mpc_imagref(z), 0, MPC_RND_IM(rnd)));

          /* the sign of the zero imaginary part is known in some cases
             (see algorithm.tex). In such cases we have (x +s*0i)^(y+/-0i)
             = x^y + s*sign(y)*0i where s = +/-1.
             We extend here this rule to fix the sign of the zero part.

             Note that the sign must also be set explicitly when rnd=RNDD
             because mpfr_set_ui(z_i, 0, rnd) always sets z_i to +0.
          */
          if (MPC_RND_IM(rnd) == GMP_RNDD || s1 != s2)
            mpfr_neg (mpc_imagref(z), mpc_imagref(z), MPC_RND_IM(rnd));
          goto end;
        }

      /* (-1)^(n+I*t) is real for n integer and t real */
      if (mpfr_cmp_si (mpc_realref(x), -1) == 0 && mpfr_integer_p (mpc_realref(y)))
        z_real = 1;

      /* for x real, x^y is imaginary when:
         (a) x is negative and y is half-an-integer
         (b) x = -1 and Re(y) is half-an-integer
      */
      if ((mpfr_cmp_ui (mpc_realref(x), 0) < 0) && is_odd (mpc_realref(y), 1)
         && (y_real || mpfr_cmp_si (mpc_realref(x), -1) == 0))
        z_imag = 1;
    }
  else /* x non real */
    /* I^(t*I) and (-I)^(t*I) are real for t real,
       I^(n+t*I) and (-I)^(n+t*I) are real for n even and t real, and
       I^(n+t*I) and (-I)^(n+t*I) are imaginary for n odd and t real
       (s*I)^n is real for n even and imaginary for n odd */
    if ((mpc_cmp_si_si (x, 0, 1) == 0 || mpc_cmp_si_si (x, 0, -1) == 0 ||
         (mpfr_cmp_ui (mpc_realref(x), 0) == 0 && y_real)) &&
        mpfr_integer_p (mpc_realref(y)))
      { /* x is I or -I, and Re(y) is an integer */
        if (is_odd (mpc_realref(y), 0))
          z_imag = 1; /* Re(y) odd: z is imaginary */
        else
          z_real = 1; /* Re(y) even: z is real */
      }
    else /* (t+/-t*I)^(2n) is imaginary for n odd and real for n even */
      if (mpfr_cmpabs (mpc_realref(x), mpc_imagref(x)) == 0 && y_real &&
          mpfr_integer_p (mpc_realref(y)) && is_odd (mpc_realref(y), 0) == 0)
        {
          if (is_odd (mpc_realref(y), -1)) /* y/2 is odd */
            z_imag = 1;
          else
            z_real = 1;
        }

  pr = mpfr_get_prec (mpc_realref(z));
  pi = mpfr_get_prec (mpc_imagref(z));
  p = (pr > pi) ? pr : pi;
  p += 12; /* experimentally, seems to give less than 10% of failures in
              Ziv's strategy; probably wrong now since q is not computed */
  if (p < 64)
    p = 64;
  mpc_init2 (u, p);
  mpc_init2 (t, p);
  pr += MPC_RND_RE(rnd) == GMP_RNDN;
  pi += MPC_RND_IM(rnd) == GMP_RNDN;
  maxprec = MPC_MAX_PREC (z);
  x_imag = mpfr_zero_p (mpc_realref(x));
  for (loop = 0;; loop++)
    {
      int ret_exp;
      mpfr_exp_t dr, di;
      mpfr_prec_t q;

      mpc_log (t, x, MPC_RNDNN);
      mpc_mul (t, t, y, MPC_RNDNN);

      /* Compute q such that |Re (y log x)|, |Im (y log x)| < 2^q.
         We recompute it at each loop since we might get different
         bounds if the precision is not enough. */
      q = mpfr_get_exp (mpc_realref(t)) > 0 ? mpfr_get_exp (mpc_realref(t)) : 0;
      if (mpfr_get_exp (mpc_imagref(t)) > (mpfr_exp_t) q)
        q = mpfr_get_exp (mpc_imagref(t));

      mpfr_clear_overflow ();
      mpfr_clear_underflow ();
      ret_exp = mpc_exp (u, t, MPC_RNDNN);
      if (mpfr_underflow_p () || mpfr_overflow_p ()) {
         /* under- and overflow flags are set by mpc_exp */
         mpc_set (z, u, MPC_RNDNN);
         ret = ret_exp;
         goto exact;
      }

      /* Since the error bound is global, we have to take into account the
         exponent difference between the real and imaginary parts. We assume
         either the real or the imaginary part of u is not zero.
      */
      dr = mpfr_zero_p (mpc_realref(u)) ? mpfr_get_exp (mpc_imagref(u))
        : mpfr_get_exp (mpc_realref(u));
      di = mpfr_zero_p (mpc_imagref(u)) ? dr : mpfr_get_exp (mpc_imagref(u));
      if (dr > di)
        {
          di = dr - di;
          dr = 0;
        }
      else
        {
          dr = di - dr;
          di = 0;
        }
      /* the term -3 takes into account the factor 4 in the complex error
         (see algorithms.tex) plus one due to the exponent difference: if
         z = a + I*b, where the relative error on z is at most 2^(-p), and
         EXP(a) = EXP(b) + k, the relative error on b is at most 2^(k-p) */
      if ((z_imag || (p > q + 3 + dr && mpfr_can_round (mpc_realref(u), p - q - 3 - dr, GMP_RNDN, GMP_RNDZ, pr))) &&
          (z_real || (p > q + 3 + di && mpfr_can_round (mpc_imagref(u), p - q - 3 - di, GMP_RNDN, GMP_RNDZ, pi))))
        break;

      /* if Re(u) is not known to be zero, assume it is a normal number, i.e.,
         neither zero, Inf or NaN, otherwise we might enter an infinite loop */
      MPC_ASSERT (z_imag || mpfr_number_p (mpc_realref(u)));
      /* idem for Im(u) */
      MPC_ASSERT (z_real || mpfr_number_p (mpc_imagref(u)));

      if (ret == -2) /* we did not yet call mpc_pow_exact, or it aborted
                        because intermediate computations had > maxprec bits */
        {
          /* check exact cases (see algorithms.tex) */
          if (y_real)
            {
              maxprec *= 2;
              ret = mpc_pow_exact (z, x, mpc_realref(y), rnd, maxprec);
              if (ret != -1 && ret != -2)
                goto exact;
            }
          p += dr + di + 64;
        }
      else
        p += p / 2;
      mpc_set_prec (t, p);
      mpc_set_prec (u, p);
    }

  if (z_real)
    {
      /* When the result is real (see algorithm.tex for details),
         Im(x^y) =
         + sign(imag(y))*0i,               if |x| > 1
         + sign(imag(x))*sign(real(y))*0i, if |x| = 1
         - sign(imag(y))*0i,               if |x| < 1
      */
      mpfr_t n;
      int inex, cx1;
      int sign_zi, sign_rex, sign_imx;
      /* cx1 < 0 if |x| < 1
         cx1 = 0 if |x| = 1
         cx1 > 0 if |x| > 1
      */

      sign_rex = mpfr_signbit (mpc_realref (x));
      sign_imx = mpfr_signbit (mpc_imagref (x));
      mpfr_init (n);
      inex = mpc_norm (n, x, GMP_RNDN);
      cx1 = mpfr_cmp_ui (n, 1);
      if (cx1 == 0 && inex != 0)
        cx1 = -inex;

      sign_zi = (cx1 < 0 && mpfr_signbit (mpc_imagref (y)) == 0)
        || (cx1 == 0 && sign_imx != mpfr_signbit (mpc_realref (y)))
        || (cx1 > 0 && mpfr_signbit (mpc_imagref (y)));

      /* copy RE(y) to n since if z==y we will destroy Re(y) below */
      mpfr_set_prec (n, mpfr_get_prec (mpc_realref (y)));
      mpfr_set (n, mpc_realref (y), GMP_RNDN);
      ret = mpfr_set (mpc_realref(z), mpc_realref(u), MPC_RND_RE(rnd));
      if (y_real && (x_real || x_imag))
        {
          /* FIXME: with y_real we assume Im(y) is really 0, which is the case
             for example when y comes from pow_fr, but in case Im(y) is +0 or
             -0, we might get different results */
          mpfr_set_ui (mpc_imagref (z), 0, MPC_RND_IM (rnd));
          fix_sign (z, sign_rex, sign_imx, n);
          ret = MPC_INEX(ret, 0); /* imaginary part is exact */
        }
      else
        {
          ret = MPC_INEX (ret, mpfr_set_ui (mpc_imagref (z), 0, MPC_RND_IM (rnd)));
          /* warning: mpfr_set_ui does not set Im(z) to -0 if Im(rnd) = RNDD */
          if (MPC_RND_IM (rnd) == GMP_RNDD || sign_zi)
            mpc_conj (z, z, MPC_RNDNN);
        }

      mpfr_clear (n);
    }
  else if (z_imag)
    {
      ret = mpfr_set (mpc_imagref(z), mpc_imagref(u), MPC_RND_IM(rnd));
      /* if z is imaginary and y real, then x cannot be real */
      if (y_real && x_imag)
        {
          int sign_rex = mpfr_signbit (mpc_realref (x));

          /* If z overlaps with y we set Re(z) before checking Re(y) below,
             but in that case y=0, which was dealt with above. */
          mpfr_set_ui (mpc_realref (z), 0, MPC_RND_RE (rnd));
          /* Note: fix_sign only does something when y is an integer,
             then necessarily y = 1 or 3 (mod 4), and in that case the
             sign of Im(x) is irrelevant. */
          fix_sign (z, sign_rex, 0, mpc_realref (y));
          ret = MPC_INEX(0, ret);
        }
      else
        ret = MPC_INEX(mpfr_set_ui (mpc_realref(z), 0, MPC_RND_RE(rnd)), ret);
    }
  else
    ret = mpc_set (z, u, rnd);
 exact:
  mpc_clear (t);
  mpc_clear (u);

  /* restore underflow and overflow flags from MPFR */
  if (saved_underflow)
    mpfr_set_underflow ();
  if (saved_overflow)
    mpfr_set_overflow ();

 end:
  return ret;
}
