/* mpc_mul -- Multiply two complex numbers

Copyright (C) 2002, 2004, 2005, 2008, 2009, 2010, 2011, 2012 INRIA

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

#include <stdio.h>    /* for MPC_ASSERT */
#include "mpc-impl.h"

#define mpz_add_si(z,x,y) do { \
   if (y >= 0) \
      mpz_add_ui (z, x, (long int) y); \
   else \
      mpz_sub_ui (z, x, (long int) (-y)); \
   } while (0);

/* compute z=x*y when x has an infinite part */
static int
mul_infinite (mpc_ptr z, mpc_srcptr x, mpc_srcptr y)
{
   /* Let x=xr+i*xi and y=yr+i*yi; extract the signs of the operands */
   int xrs = mpfr_signbit (mpc_realref (x)) ? -1 : 1;
   int xis = mpfr_signbit (mpc_imagref (x)) ? -1 : 1;
   int yrs = mpfr_signbit (mpc_realref (y)) ? -1 : 1;
   int yis = mpfr_signbit (mpc_imagref (y)) ? -1 : 1;

   int u, v;

   /* compute the sign of
      u = xrs * yrs * xr * yr - xis * yis * xi * yi
      v = xrs * yis * xr * yi + xis * yrs * xi * yr
      +1 if positive, -1 if negatiye, 0 if NaN */
   if (  mpfr_nan_p (mpc_realref (x)) || mpfr_nan_p (mpc_imagref (x))
      || mpfr_nan_p (mpc_realref (y)) || mpfr_nan_p (mpc_imagref (y))) {
      u = 0;
      v = 0;
   }
   else if (mpfr_inf_p (mpc_realref (x))) {
      /* x = (+/-inf) xr + i*xi */
      u = (   mpfr_zero_p (mpc_realref (y))
           || (mpfr_inf_p (mpc_imagref (x)) && mpfr_zero_p (mpc_imagref (y)))
           || (mpfr_zero_p (mpc_imagref (x)) && mpfr_inf_p (mpc_imagref (y)))
           || (   (mpfr_inf_p (mpc_imagref (x)) || mpfr_inf_p (mpc_imagref (y)))
              && xrs*yrs == xis*yis)
           ? 0 : xrs * yrs);
      v = (   mpfr_zero_p (mpc_imagref (y))
           || (mpfr_inf_p (mpc_imagref (x)) && mpfr_zero_p (mpc_realref (y)))
           || (mpfr_zero_p (mpc_imagref (x)) && mpfr_inf_p (mpc_realref (y)))
           || (   (mpfr_inf_p (mpc_imagref (x)) || mpfr_inf_p (mpc_imagref (x)))
               && xrs*yis != xis*yrs)
           ? 0 : xrs * yis);
   }
   else {
      /* x = xr + i*(+/-inf) with |xr| != inf */
      u = (   mpfr_zero_p (mpc_imagref (y))
           || (mpfr_zero_p (mpc_realref (x)) && mpfr_inf_p (mpc_realref (y)))
           || (mpfr_inf_p (mpc_realref (y)) && xrs*yrs == xis*yis)
           ? 0 : -xis * yis);
      v = (   mpfr_zero_p (mpc_realref (y))
           || (mpfr_zero_p (mpc_realref (x)) && mpfr_inf_p (mpc_imagref (y)))
           || (mpfr_inf_p (mpc_imagref (y)) && xrs*yis != xis*yrs)
           ? 0 : xis * yrs);
   }

   if (u == 0 && v == 0) {
      /* Naive result is NaN+i*NaN. Obtain an infinity using the algorithm
         given in Annex G.5.1 of the ISO C99 standard */
      int xr = (mpfr_zero_p (mpc_realref (x)) || mpfr_nan_p (mpc_realref (x)) ? 0
                : (mpfr_inf_p (mpc_realref (x)) ? 1 : 0));
      int xi = (mpfr_zero_p (mpc_imagref (x)) || mpfr_nan_p (mpc_imagref (x)) ? 0
                : (mpfr_inf_p (mpc_imagref (x)) ? 1 : 0));
      int yr = (mpfr_zero_p (mpc_realref (y)) || mpfr_nan_p (mpc_realref (y)) ? 0 : 1);
      int yi = (mpfr_zero_p (mpc_imagref (y)) || mpfr_nan_p (mpc_imagref (y)) ? 0 : 1);
      if (mpc_inf_p (y)) {
         yr = mpfr_inf_p (mpc_realref (y)) ? 1 : 0;
         yi = mpfr_inf_p (mpc_imagref (y)) ? 1 : 0;
      }

      u = xrs * xr * yrs * yr - xis * xi * yis * yi;
      v = xrs * xr * yis * yi + xis * xi * yrs * yr;
   }

   if (u == 0)
      mpfr_set_nan (mpc_realref (z));
   else
      mpfr_set_inf (mpc_realref (z), u);

   if (v == 0)
      mpfr_set_nan (mpc_imagref (z));
   else
      mpfr_set_inf (mpc_imagref (z), v);

   return MPC_INEX (0, 0); /* exact */
}


/* compute z = x*y for Im(y) == 0 */
static int
mul_real (mpc_ptr z, mpc_srcptr x, mpc_srcptr y, mpc_rnd_t rnd)
{
   int xrs, xis, yrs, yis;
   int inex;

   /* save signs of operands */
   xrs = MPFR_SIGNBIT (mpc_realref (x));
   xis = MPFR_SIGNBIT (mpc_imagref (x));
   yrs = MPFR_SIGNBIT (mpc_realref (y));
   yis = MPFR_SIGNBIT (mpc_imagref (y));

   inex = mpc_mul_fr (z, x, mpc_realref (y), rnd);
   /* Signs of zeroes may be wrong. Their correction does not change the
      inexact flag. */
   if (mpfr_zero_p (mpc_realref (z)))
      mpfr_setsign (mpc_realref (z), mpc_realref (z), MPC_RND_RE(rnd) == GMP_RNDD
                     || (xrs != yrs && xis == yis), GMP_RNDN);
   if (mpfr_zero_p (mpc_imagref (z)))
      mpfr_setsign (mpc_imagref (z), mpc_imagref (z), MPC_RND_IM (rnd) == GMP_RNDD
                     || (xrs != yis && xis != yrs), GMP_RNDN);

   return inex;
}


/* compute z = x*y for Re(y) == 0, and Im(x) != 0 and Im(y) != 0 */
static int
mul_imag (mpc_ptr z, mpc_srcptr x, mpc_srcptr y, mpc_rnd_t rnd)
{
   int sign;
   int inex_re, inex_im;
   int overlap = z == x || z == y;
   mpc_t rop;

   if (overlap)
      mpc_init3 (rop, MPC_PREC_RE (z), MPC_PREC_IM (z));
   else
      rop [0] = z[0];

   sign =    (MPFR_SIGNBIT (mpc_realref (y)) != MPFR_SIGNBIT (mpc_imagref (x)))
          && (MPFR_SIGNBIT (mpc_imagref (y)) != MPFR_SIGNBIT (mpc_realref (x)));

   inex_re = -mpfr_mul (mpc_realref (rop), mpc_imagref (x), mpc_imagref (y),
                        INV_RND (MPC_RND_RE (rnd)));
   mpfr_neg (mpc_realref (rop), mpc_realref (rop), GMP_RNDN); /* exact */
   inex_im = mpfr_mul (mpc_imagref (rop), mpc_realref (x), mpc_imagref (y),
                       MPC_RND_IM (rnd));
   mpc_set (z, rop, MPC_RNDNN);

   /* Sign of zeroes may be wrong (note that Re(z) cannot be zero) */
   if (mpfr_zero_p (mpc_imagref (z)))
      mpfr_setsign (mpc_imagref (z), mpc_imagref (z), MPC_RND_IM (rnd) == GMP_RNDD
                     || sign, GMP_RNDN);

   if (overlap)
      mpc_clear (rop);

   return MPC_INEX (inex_re, inex_im);
}


static int
mpfr_fmma (mpfr_ptr z, mpfr_srcptr a, mpfr_srcptr b, mpfr_srcptr c,
           mpfr_srcptr d, int sign, mpfr_rnd_t rnd)
{
   /* Computes z = ab+cd if sign >= 0, or z = ab-cd if sign < 0.
      Assumes that a, b, c, d are finite and non-zero; so any multiplication
      of two of them yielding an infinity is an overflow, and a
      multiplication yielding 0 is an underflow.
      Assumes further that z is distinct from a, b, c, d. */

   int inex;
   mpfr_t u, v;

   /* u=a*b, v=sign*c*d exactly */
   mpfr_init2 (u, mpfr_get_prec (a) + mpfr_get_prec (b));
   mpfr_init2 (v, mpfr_get_prec (c) + mpfr_get_prec (d));
   mpfr_mul (u, a, b, GMP_RNDN);
   mpfr_mul (v, c, d, GMP_RNDN);
   if (sign < 0)
      mpfr_neg (v, v, GMP_RNDN);

   /* tentatively compute z as u+v; here we need z to be distinct
      from a, b, c, d to not lose the latter */
   inex = mpfr_add (z, u, v, rnd);

   if (mpfr_inf_p (z)) {
      /* replace by "correctly rounded overflow" */
      mpfr_set_si (z, (mpfr_signbit (z) ? -1 : 1), GMP_RNDN);
      inex = mpfr_mul_2ui (z, z, mpfr_get_emax (), rnd);
   }
   else if (mpfr_zero_p (u) && !mpfr_zero_p (v)) {
      /* exactly u underflowed, determine inexact flag */
      inex = (mpfr_signbit (u) ? 1 : -1);
   }
   else if (mpfr_zero_p (v) && !mpfr_zero_p (u)) {
      /* exactly v underflowed, determine inexact flag */
      inex = (mpfr_signbit (v) ? 1 : -1);
   }
   else if (mpfr_nan_p (z) || (mpfr_zero_p (u) && mpfr_zero_p (v))) {
      /* In the first case, u and v are infinities with opposite signs.
         In the second case, u and v are zeroes; their sum may be 0 or the
         least representable number, with a sign to be determined.
         Redo the computations with mpz_t exponents */
      mpfr_exp_t ea, eb, ec, ed;
      mpz_t eu, ev;
         /* cheat to work around the const qualifiers */

      /* Normalise the input by shifting and keep track of the shifts in
         the exponents of u and v */
      ea = mpfr_get_exp (a);
      eb = mpfr_get_exp (b);
      ec = mpfr_get_exp (c);
      ed = mpfr_get_exp (d);

      mpfr_set_exp ((mpfr_ptr) a, (mpfr_prec_t) 0);
      mpfr_set_exp ((mpfr_ptr) b, (mpfr_prec_t) 0);
      mpfr_set_exp ((mpfr_ptr) c, (mpfr_prec_t) 0);
      mpfr_set_exp ((mpfr_ptr) d, (mpfr_prec_t) 0);

      mpz_init (eu);
      mpz_init (ev);
      mpz_set_si (eu, (long int) ea);
      mpz_add_si (eu, eu, (long int) eb);
      mpz_set_si (ev, (long int) ec);
      mpz_add_si (ev, ev, (long int) ed);

      /* recompute u and v and move exponents to eu and ev */
      mpfr_mul (u, a, b, GMP_RNDN);
      /* exponent of u is non-positive */
      mpz_sub_ui (eu, eu, (unsigned long int) (-mpfr_get_exp (u)));
      mpfr_set_exp (u, (mpfr_prec_t) 0);
      mpfr_mul (v, c, d, GMP_RNDN);
      if (sign < 0)
         mpfr_neg (v, v, GMP_RNDN);
      mpz_sub_ui (ev, ev, (unsigned long int) (-mpfr_get_exp (v)));
      mpfr_set_exp (v, (mpfr_prec_t) 0);

      if (mpfr_nan_p (z)) {
         mpfr_exp_t emax = mpfr_get_emax ();
         int overflow;
         /* We have a = ma * 2^ea with 1/2 <= |ma| < 1 and ea <= emax, and
            analogously for b. So eu <= 2*emax, and eu > emax since we have
            an overflow. The same holds for ev. Shift u and v by as much as
            possible so that one of them has exponent emax and the
            remaining exponents in eu and ev are the same. Then carry out
            the addition. Shifting u and v prevents an underflow. */
         if (mpz_cmp (eu, ev) >= 0) {
            mpfr_set_exp (u, emax);
            mpz_sub_ui (eu, eu, (long int) emax);
            mpz_sub (ev, ev, eu);
            mpfr_set_exp (v, (mpfr_exp_t) mpz_get_ui (ev));
               /* remaining common exponent is now in eu */
         }
         else {
            mpfr_set_exp (v, emax);
            mpz_sub_ui (ev, ev, (long int) emax);
            mpz_sub (eu, eu, ev);
            mpfr_set_exp (u, (mpfr_exp_t) mpz_get_ui (eu));
            mpz_set (eu, ev);
               /* remaining common exponent is now also in eu */
         }
         inex = mpfr_add (z, u, v, rnd);
            /* Result is finite since u and v have different signs. */
         overflow = mpfr_mul_2ui (z, z, mpz_get_ui (eu), rnd);
         if (overflow)
            inex = overflow;
      }
      else {
         int underflow;
         /* Addition of two zeroes with same sign. We have a = ma * 2^ea
            with 1/2 <= |ma| < 1 and ea >= emin and similarly for b.
            So 2*emin < 2*emin+1 <= eu < emin < 0, and analogously for v. */
         mpfr_exp_t emin = mpfr_get_emin ();
         if (mpz_cmp (eu, ev) <= 0) {
            mpfr_set_exp (u, emin);
            mpz_add_ui (eu, eu, (unsigned long int) (-emin));
            mpz_sub (ev, ev, eu);
            mpfr_set_exp (v, (mpfr_exp_t) mpz_get_si (ev));
         }
         else {
            mpfr_set_exp (v, emin);
            mpz_add_ui (ev, ev, (unsigned long int) (-emin));
            mpz_sub (eu, eu, ev);
            mpfr_set_exp (u, (mpfr_exp_t) mpz_get_si (eu));
            mpz_set (eu, ev);
         }
         inex = mpfr_add (z, u, v, rnd);
         mpz_neg (eu, eu);
         underflow = mpfr_div_2ui (z, z, mpz_get_ui (eu), rnd);
         if (underflow)
            inex = underflow;
      }

      mpz_clear (eu);
      mpz_clear (ev);

      mpfr_set_exp ((mpfr_ptr) a, ea);
      mpfr_set_exp ((mpfr_ptr) b, eb);
      mpfr_set_exp ((mpfr_ptr) c, ec);
      mpfr_set_exp ((mpfr_ptr) d, ed);
         /* works also when some of a, b, c, d are not all distinct */
   }

   mpfr_clear (u);
   mpfr_clear (v);

   return inex;
}


int
mpc_mul_naive (mpc_ptr z, mpc_srcptr x, mpc_srcptr y, mpc_rnd_t rnd)
{
   /* computes z=x*y by the schoolbook method, where x and y are assumed
      to be finite and without zero parts                                */
   int overlap, inex;
   mpc_t rop;

   MPC_ASSERT (   mpfr_regular_p (mpc_realref (x)) && mpfr_regular_p (mpc_imagref (x))
               && mpfr_regular_p (mpc_realref (y)) && mpfr_regular_p (mpc_imagref (y)));
   overlap = (z == x) || (z == y);
   if (overlap)
      mpc_init3 (rop, MPC_PREC_RE (z), MPC_PREC_IM (z));
   else
      rop [0] = z [0];

   inex = MPC_INEX (mpfr_fmma (mpc_realref (rop), mpc_realref (x), mpc_realref (y), mpc_imagref (x),
                               mpc_imagref (y), -1, MPC_RND_RE (rnd)),
                    mpfr_fmma (mpc_imagref (rop), mpc_realref (x), mpc_imagref (y), mpc_imagref (x),
                               mpc_realref (y), +1, MPC_RND_IM (rnd)));

   mpc_set (z, rop, MPC_RNDNN);
   if (overlap)
      mpc_clear (rop);

   return inex;
}


int
mpc_mul_karatsuba (mpc_ptr rop, mpc_srcptr op1, mpc_srcptr op2, mpc_rnd_t rnd)
{
   /* computes rop=op1*op2 by a Karatsuba algorithm, where op1 and op2
      are assumed to be finite and without zero parts                  */
  mpfr_srcptr a, b, c, d;
  int mul_i, ok, inexact, mul_a, mul_c, inex_re = 0, inex_im = 0, sign_x, sign_u;
  mpfr_t u, v, w, x;
  mpfr_prec_t prec, prec_re, prec_u, prec_v, prec_w;
  mpfr_rnd_t rnd_re, rnd_u;
  int overlap;
     /* true if rop == op1 or rop == op2 */
  mpc_t result;
     /* overlap is quite difficult to handle, because we have to tentatively
        round the variable u in the end to either the real or the imaginary
        part of rop (it is not possible to tell now whether the real or
        imaginary part is used). If this fails, we have to start again and
        need the correct values of op1 and op2.
        So we just create a new variable for the result in this case. */
  int loop;
  const int MAX_MUL_LOOP = 1;

  overlap = (rop == op1) || (rop == op2);
  if (overlap)
     mpc_init3 (result, MPC_PREC_RE (rop), MPC_PREC_IM (rop));
  else
     result [0] = rop [0];

  a = mpc_realref(op1);
  b = mpc_imagref(op1);
  c = mpc_realref(op2);
  d = mpc_imagref(op2);

  /* (a + i*b) * (c + i*d) = [ac - bd] + i*[ad + bc] */

  mul_i = 0; /* number of multiplications by i */
  mul_a = 1; /* implicit factor for a */
  mul_c = 1; /* implicit factor for c */

  if (mpfr_cmp_abs (a, b) < 0)
    {
      MPFR_SWAP (a, b);
      mul_i ++;
      mul_a = -1; /* consider i * (a+i*b) = -b + i*a */
    }

  if (mpfr_cmp_abs (c, d) < 0)
    {
      MPFR_SWAP (c, d);
      mul_i ++;
      mul_c = -1; /* consider -d + i*c instead of c + i*d */
    }

  /* find the precision and rounding mode for the new real part */
  if (mul_i % 2)
    {
      prec_re = MPC_PREC_IM(rop);
      rnd_re = MPC_RND_IM(rnd);
    }
  else /* mul_i = 0 or 2 */
    {
      prec_re = MPC_PREC_RE(rop);
      rnd_re = MPC_RND_RE(rnd);
    }

  if (mul_i)
    rnd_re = INV_RND(rnd_re);

  /* now |a| >= |b| and |c| >= |d| */
  prec = MPC_MAX_PREC(rop);

  mpfr_init2 (v, prec_v = mpfr_get_prec (a) + mpfr_get_prec (d));
  mpfr_init2 (w, prec_w = mpfr_get_prec (b) + mpfr_get_prec (c));
  mpfr_init2 (u, 2);
  mpfr_init2 (x, 2);

  inexact = mpfr_mul (v, a, d, GMP_RNDN);
  if (inexact) {
     /* over- or underflow */
    ok = 0;
    goto clear;
  }
  if (mul_a == -1)
    mpfr_neg (v, v, GMP_RNDN);

  inexact = mpfr_mul (w, b, c, GMP_RNDN);
  if (inexact) {
     /* over- or underflow */
     ok = 0;
     goto clear;
  }
  if (mul_c == -1)
    mpfr_neg (w, w, GMP_RNDN);

  /* compute sign(v-w) */
  sign_x = mpfr_cmp_abs (v, w);
  if (sign_x > 0)
    sign_x = 2 * mpfr_sgn (v) - mpfr_sgn (w);
  else if (sign_x == 0)
    sign_x = mpfr_sgn (v) - mpfr_sgn (w);
  else
    sign_x = mpfr_sgn (v) - 2 * mpfr_sgn (w);

   sign_u = mul_a * mpfr_sgn (a) * mul_c * mpfr_sgn (c);

   if (sign_x * sign_u < 0)
    {  /* swap inputs */
      MPFR_SWAP (a, c);
      MPFR_SWAP (b, d);
      mpfr_swap (v, w);
      { int tmp; tmp = mul_a; mul_a = mul_c; mul_c = tmp; }
      sign_x = - sign_x;
    }

   /* now sign_x * sign_u >= 0 */
   loop = 0;
   do
     {
        loop++;
         /* the following should give failures with prob. <= 1/prec */
         prec += mpc_ceil_log2 (prec) + 3;

         mpfr_set_prec (u, prec_u = prec);
         mpfr_set_prec (x, prec);

         /* first compute away(b +/- a) and store it in u */
         inexact = (mul_a == -1 ?
                    ROUND_AWAY (mpfr_sub (u, b, a, MPFR_RNDA), u) :
                    ROUND_AWAY (mpfr_add (u, b, a, MPFR_RNDA), u));

         /* then compute away(+/-c - d) and store it in x */
         inexact |= (mul_c == -1 ?
                     ROUND_AWAY (mpfr_add (x, c, d, MPFR_RNDA), x) :
                     ROUND_AWAY (mpfr_sub (x, c, d, MPFR_RNDA), x));
         if (mul_c == -1)
           mpfr_neg (x, x, GMP_RNDN);

         if (inexact == 0)
            mpfr_prec_round (u, prec_u = 2 * prec, GMP_RNDN);

         /* compute away(u*x) and store it in u */
         inexact |= ROUND_AWAY (mpfr_mul (u, u, x, MPFR_RNDA), u);
            /* (a+b)*(c-d) */

	 /* if all computations are exact up to here, it may be that
	    the real part is exact, thus we need if possible to
	    compute v - w exactly */
	 if (inexact == 0)
	   {
	     mpfr_prec_t prec_x;
             /* v and w are different from 0, so mpfr_get_exp is safe to use */
             prec_x = SAFE_ABS (mpfr_exp_t, mpfr_get_exp (v) - mpfr_get_exp (w))
                      + MPC_MAX (prec_v, prec_w) + 1;
                      /* +1 is necessary for a potential carry */
	     /* ensure we do not use a too large precision */
	     if (prec_x > prec_u)
               prec_x = prec_u;
	     if (prec_x > prec)
	       mpfr_prec_round (x, prec_x, GMP_RNDN);
	   }

         rnd_u = (sign_u > 0) ? GMP_RNDU : GMP_RNDD;
         inexact |= mpfr_sub (x, v, w, rnd_u); /* ad - bc */

         /* in case u=0, ensure that rnd_u rounds x away from zero */
         if (mpfr_sgn (u) == 0)
           rnd_u = (mpfr_sgn (x) > 0) ? GMP_RNDU : GMP_RNDD;
         inexact |= mpfr_add (u, u, x, rnd_u); /* ac - bd */

         ok = inexact == 0 ||
           mpfr_can_round (u, prec_u - 3, rnd_u, GMP_RNDZ,
                           prec_re + (rnd_re == GMP_RNDN));
         /* this ensures both we can round correctly and determine the correct
            inexact flag (for rounding to nearest) */
     }
   while (!ok && loop <= MAX_MUL_LOOP);
      /* after MAX_MUL_LOOP rounds, use mpc_naive instead */

   if (ok) {
      /* if inexact is zero, then u is exactly ac-bd, otherwise fix the sign
         of the inexact flag for u, which was rounded away from ac-bd */
      if (inexact != 0)
      inexact = mpfr_sgn (u);

      if (mul_i == 0)
      {
         inex_re = mpfr_set (mpc_realref(result), u, MPC_RND_RE(rnd));
         if (inex_re == 0)
            {
            inex_re = inexact; /* u is rounded away from 0 */
            inex_im = mpfr_add (mpc_imagref(result), v, w, MPC_RND_IM(rnd));
            }
         else
            inex_im = mpfr_add (mpc_imagref(result), v, w, MPC_RND_IM(rnd));
      }
      else if (mul_i == 1) /* (x+i*y)/i = y - i*x */
      {
         inex_im = mpfr_neg (mpc_imagref(result), u, MPC_RND_IM(rnd));
         if (inex_im == 0)
            {
            inex_im = -inexact; /* u is rounded away from 0 */
            inex_re = mpfr_add (mpc_realref(result), v, w, MPC_RND_RE(rnd));
            }
         else
            inex_re = mpfr_add (mpc_realref(result), v, w, MPC_RND_RE(rnd));
      }
      else /* mul_i = 2, z/i^2 = -z */
      {
         inex_re = mpfr_neg (mpc_realref(result), u, MPC_RND_RE(rnd));
         if (inex_re == 0)
            {
            inex_re = -inexact; /* u is rounded away from 0 */
            inex_im = -mpfr_add (mpc_imagref(result), v, w,
                                 INV_RND(MPC_RND_IM(rnd)));
            mpfr_neg (mpc_imagref(result), mpc_imagref(result), MPC_RND_IM(rnd));
            }
         else
            {
            inex_im = -mpfr_add (mpc_imagref(result), v, w,
                                 INV_RND(MPC_RND_IM(rnd)));
            mpfr_neg (mpc_imagref(result), mpc_imagref(result), MPC_RND_IM(rnd));
            }
      }

      mpc_set (rop, result, MPC_RNDNN);
   }

clear:
   mpfr_clear (u);
   mpfr_clear (v);
   mpfr_clear (w);
   mpfr_clear (x);
   if (overlap)
      mpc_clear (result);

   if (ok)
      return MPC_INEX(inex_re, inex_im);
   else
      return mpc_mul_naive (rop, op1, op2, rnd);
}


int
mpc_mul (mpc_ptr a, mpc_srcptr b, mpc_srcptr c, mpc_rnd_t rnd)
{
   /* Conforming to ISO C99 standard (G.5.1 multiplicative operators),
      infinities are treated specially if both parts are NaN when computed
      naively. */
   if (mpc_inf_p (b))
      return mul_infinite (a, b, c);
   if (mpc_inf_p (c))
      return mul_infinite (a, c, b);

   /* NaN contamination of both parts in result */
   if (mpfr_nan_p (mpc_realref (b)) || mpfr_nan_p (mpc_imagref (b))
       || mpfr_nan_p (mpc_realref (c)) || mpfr_nan_p (mpc_imagref (c))) {
      mpfr_set_nan (mpc_realref (a));
      mpfr_set_nan (mpc_imagref (a));
      return MPC_INEX (0, 0);
   }

   /* check for real multiplication */
   if (mpfr_zero_p (mpc_imagref (b)))
      return mul_real (a, c, b, rnd);
   if (mpfr_zero_p (mpc_imagref (c)))
      return mul_real (a, b, c, rnd);

   /* check for purely imaginary multiplication */
   if (mpfr_zero_p (mpc_realref (b)))
      return mul_imag (a, c, b, rnd);
   if (mpfr_zero_p (mpc_realref (c)))
      return mul_imag (a, b, c, rnd);

   /* If the real and imaginary part of one argument have a very different */
   /* exponent, it is not reasonable to use Karatsuba multiplication.      */
   if (   SAFE_ABS (mpfr_exp_t,
                     mpfr_get_exp (mpc_realref (b)) - mpfr_get_exp (mpc_imagref (b)))
         > (mpfr_exp_t) MPC_MAX_PREC (b) / 2
      || SAFE_ABS (mpfr_exp_t,
                     mpfr_get_exp (mpc_realref (c)) - mpfr_get_exp (mpc_imagref (c)))
         > (mpfr_exp_t) MPC_MAX_PREC (c) / 2)
      return mpc_mul_naive (a, b, c, rnd);
   else
      return ((MPC_MAX_PREC(a)
               <= (mpfr_prec_t) MUL_KARATSUBA_THRESHOLD * BITS_PER_MP_LIMB)
            ? mpc_mul_naive : mpc_mul_karatsuba) (a, b, c, rnd);
}
