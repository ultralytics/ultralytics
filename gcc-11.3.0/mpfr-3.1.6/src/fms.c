/* mpfr_fms -- Floating multiply-subtract

Copyright 2001-2002, 2004, 2006-2017 Free Software Foundation, Inc.
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

/* The fused-multiply-subtract (fms) of x, y and z is defined by:
   fms(x,y,z)= x*y - z
   Note: this is neither in IEEE754R, nor in LIA-2, but both the
   PowerPC and the Itanium define fms as x*y - z.
*/

int
mpfr_fms (mpfr_ptr s, mpfr_srcptr x, mpfr_srcptr y, mpfr_srcptr z,
          mpfr_rnd_t rnd_mode)
{
  int inexact;
  mpfr_t u;
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_GROUP_DECL(group);

  MPFR_LOG_FUNC
    (("x[%Pu]=%.*Rg y[%Pu]=%.*Rg  z[%Pu]=%.*Rg rnd=%d",
      mpfr_get_prec (x), mpfr_log_prec, x,
      mpfr_get_prec (y), mpfr_log_prec, y,
      mpfr_get_prec (z), mpfr_log_prec, z, rnd_mode),
     ("s[%Pu]=%.*Rg inexact=%d",
      mpfr_get_prec (s), mpfr_log_prec, s, inexact));

  /* particular cases */
  if (MPFR_UNLIKELY( MPFR_IS_SINGULAR(x) ||
                     MPFR_IS_SINGULAR(y) ||
                     MPFR_IS_SINGULAR(z) ))
    {
      if (MPFR_IS_NAN(x) || MPFR_IS_NAN(y) || MPFR_IS_NAN(z))
        {
          MPFR_SET_NAN(s);
          MPFR_RET_NAN;
        }
      /* now neither x, y or z is NaN */
      else if (MPFR_IS_INF(x) || MPFR_IS_INF(y))
        {
          /* cases Inf*0-z, 0*Inf-z, Inf-Inf */
          if ((MPFR_IS_ZERO(y)) ||
              (MPFR_IS_ZERO(x)) ||
              (MPFR_IS_INF(z) &&
               ((MPFR_MULT_SIGN(MPFR_SIGN(x), MPFR_SIGN(y))) == MPFR_SIGN(z))))
            {
              MPFR_SET_NAN(s);
              MPFR_RET_NAN;
            }
          else if (MPFR_IS_INF(z)) /* case Inf-Inf already checked above */
            {
              MPFR_SET_INF(s);
              MPFR_SET_OPPOSITE_SIGN(s, z);
              MPFR_RET(0);
            }
          else /* z is finite */
            {
              MPFR_SET_INF(s);
              MPFR_SET_SIGN(s, MPFR_MULT_SIGN(MPFR_SIGN(x) , MPFR_SIGN(y)));
              MPFR_RET(0);
            }
        }
      /* now x and y are finite */
      else if (MPFR_IS_INF(z))
        {
          MPFR_SET_INF(s);
          MPFR_SET_OPPOSITE_SIGN(s, z);
          MPFR_RET(0);
        }
      else if (MPFR_IS_ZERO(x) || MPFR_IS_ZERO(y))
        {
          if (MPFR_IS_ZERO(z))
            {
              int sign_p;
              sign_p = MPFR_MULT_SIGN( MPFR_SIGN(x) , MPFR_SIGN(y) );
              MPFR_SET_SIGN(s,(rnd_mode != MPFR_RNDD ?
                               ((MPFR_IS_NEG_SIGN(sign_p) && MPFR_IS_POS(z))
                                ? -1 : 1) :
                               ((MPFR_IS_POS_SIGN(sign_p) && MPFR_IS_NEG(z))
                                ? 1 : -1)));
              MPFR_SET_ZERO(s);
              MPFR_RET(0);
            }
          else
            return mpfr_neg (s, z, rnd_mode);
        }
      else /* necessarily z is zero here */
        {
          MPFR_ASSERTD(MPFR_IS_ZERO(z));
          return mpfr_mul (s, x, y, rnd_mode);
        }
    }

  /* If we take prec(u) >= prec(x) + prec(y), the product u <- x*y
     is exact, except in case of overflow or underflow. */
  MPFR_SAVE_EXPO_MARK (expo);
  MPFR_GROUP_INIT_1 (group, MPFR_PREC(x) + MPFR_PREC(y), u);

  if (MPFR_UNLIKELY (mpfr_mul (u, x, y, MPFR_RNDN)))
    {
      /* overflow or underflow - this case is regarded as rare, thus
         does not need to be very efficient (even if some tests below
         could have been done earlier).
         It is an overflow iff u is an infinity (since MPFR_RNDN was used).
         Alternatively, we could test the overflow flag, but in this case,
         mpfr_clear_flags would have been necessary. */
      if (MPFR_IS_INF (u))  /* overflow */
        {
          /* Let's eliminate the obvious case where x*y and z have the
             same sign. No possible cancellation -> real overflow.
             Also, we know that |z| < 2^emax. If E(x) + E(y) >= emax+3,
             then |x*y| >= 2^(emax+1), and |x*y - z| >= 2^emax. This case
             is also an overflow. */
          if (MPFR_SIGN (u) != MPFR_SIGN (z) ||
              MPFR_GET_EXP (x) + MPFR_GET_EXP (y) >= __gmpfr_emax + 3)
            {
              MPFR_GROUP_CLEAR (group);
              MPFR_SAVE_EXPO_FREE (expo);
              return mpfr_overflow (s, rnd_mode, - MPFR_SIGN (z));
            }

          /* E(x) + E(y) <= emax+2, therefore |x*y| < 2^(emax+2), and
             (x/4)*y does not overflow (let's recall that the result
             is exact with an unbounded exponent range). It does not
             underflow either, because x*y overflows and the exponent
             range is large enough. */
          inexact = mpfr_div_2ui (u, x, 2, MPFR_RNDN);
          MPFR_ASSERTN (inexact == 0);
          inexact = mpfr_mul (u, u, y, MPFR_RNDN);
          MPFR_ASSERTN (inexact == 0);

          /* Now, we need to subtract z/4... But it may underflow! */
          {
            mpfr_t zo4;
            mpfr_srcptr zz;
            MPFR_BLOCK_DECL (flags);

            if (MPFR_GET_EXP (u) > MPFR_GET_EXP (z) &&
                MPFR_GET_EXP (u) - MPFR_GET_EXP (z) > MPFR_PREC (u))
              {
                /* |z| < ulp(u)/2, therefore one can use z instead of z/4. */
                zz = z;
              }
            else
              {
                mpfr_init2 (zo4, MPFR_PREC (z));
                if (mpfr_div_2ui (zo4, z, 2, MPFR_RNDZ))
                  {
                    /* The division by 4 underflowed! */
                    MPFR_ASSERTN (0); /* TODO... */
                  }
                zz = zo4;
              }

            /* Let's recall that u = x*y/4 and zz = z/4 (or z if the
               following subtraction would give the same result). */
            MPFR_BLOCK (flags, inexact = mpfr_sub (s, u, zz, rnd_mode));
            /* u and zz have the same sign, so that an overflow
               is not possible. But an underflow is theoretically
               possible! */
            if (MPFR_UNDERFLOW (flags))
              {
                MPFR_ASSERTN (zz != z);
                MPFR_ASSERTN (0); /* TODO... */
                mpfr_clears (zo4, u, (mpfr_ptr) 0);
              }
            else
              {
                int inex2;

                if (zz != z)
                  mpfr_clear (zo4);
                MPFR_GROUP_CLEAR (group);
                MPFR_ASSERTN (! MPFR_OVERFLOW (flags));
                inex2 = mpfr_mul_2ui (s, s, 2, rnd_mode);
                if (inex2)  /* overflow */
                  {
                    inexact = inex2;
                    MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
                  }
                goto end;
              }
          }
        }
      else  /* underflow: one has |xy| < 2^(emin-1). */
        {
          unsigned long scale = 0;
          mpfr_t scaled_z;
          mpfr_srcptr new_z;
          mpfr_exp_t diffexp;
          mpfr_prec_t pzs;
          int xy_underflows;

          /* Let's scale z so that ulp(z) > 2^emin and ulp(s) > 2^emin
             (the + 1 on MPFR_PREC (s) is necessary because the exponent
             of the result can be EXP(z) - 1). */
          diffexp = MPFR_GET_EXP (z) - __gmpfr_emin;
          pzs = MAX (MPFR_PREC (z), MPFR_PREC (s) + 1);
          if (diffexp <= pzs)
            {
              mpfr_uexp_t uscale;
              mpfr_t scaled_v;
              MPFR_BLOCK_DECL (flags);

              uscale = (mpfr_uexp_t) pzs - diffexp + 1;
              MPFR_ASSERTN (uscale > 0);
              MPFR_ASSERTN (uscale <= ULONG_MAX);
              scale = uscale;
              mpfr_init2 (scaled_z, MPFR_PREC (z));
              inexact = mpfr_mul_2ui (scaled_z, z, scale, MPFR_RNDN);
              MPFR_ASSERTN (inexact == 0);  /* TODO: overflow case */
              new_z = scaled_z;
              /* Now we need to recompute u = xy * 2^scale. */
              MPFR_BLOCK (flags,
                          if (MPFR_GET_EXP (x) < MPFR_GET_EXP (y))
                            {
                              mpfr_init2 (scaled_v, MPFR_PREC (x));
                              mpfr_mul_2ui (scaled_v, x, scale, MPFR_RNDN);
                              mpfr_mul (u, scaled_v, y, MPFR_RNDN);
                            }
                          else
                            {
                              mpfr_init2 (scaled_v, MPFR_PREC (y));
                              mpfr_mul_2ui (scaled_v, y, scale, MPFR_RNDN);
                              mpfr_mul (u, x, scaled_v, MPFR_RNDN);
                            });
              mpfr_clear (scaled_v);
              MPFR_ASSERTN (! MPFR_OVERFLOW (flags));
              xy_underflows = MPFR_UNDERFLOW (flags);
            }
          else
            {
              new_z = z;
              xy_underflows = 1;
            }

          if (xy_underflows)
            {
              /* Let's replace xy by sign(xy) * 2^(emin-1). */
              MPFR_PREC (u) = MPFR_PREC_MIN;
              mpfr_setmin (u, __gmpfr_emin);
              MPFR_SET_SIGN (u, MPFR_MULT_SIGN (MPFR_SIGN (x),
                                                MPFR_SIGN (y)));
            }

          {
            MPFR_BLOCK_DECL (flags);

            MPFR_BLOCK (flags, inexact = mpfr_sub (s, u, new_z, rnd_mode));
            MPFR_GROUP_CLEAR (group);
            if (scale != 0)
              {
                int inex2;

                mpfr_clear (scaled_z);
                /* Here an overflow is theoretically possible, in which case
                   the result may be wrong, hence the assert. An underflow
                   is not possible, but let's check that anyway. */
                MPFR_ASSERTN (! MPFR_OVERFLOW (flags));  /* TODO... */
                MPFR_ASSERTN (! MPFR_UNDERFLOW (flags));  /* not possible */
                inex2 = mpfr_div_2ui (s, s, scale, MPFR_RNDN);
                /* FIXME: this seems incorrect. MPFR_RNDN -> rnd_mode?
                   Also, handle the double rounding case:
                   s / 2^scale = 2^(emin - 2) in MPFR_RNDN. */
                if (inex2)  /* underflow */
                  inexact = inex2;
              }
          }

          /* FIXME/TODO: I'm not sure that the following is correct.
             Check for possible spurious exceptions due to intermediate
             computations. */
          MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
          goto end;
        }
    }

  inexact = mpfr_sub (s, u, z, rnd_mode);
  MPFR_GROUP_CLEAR (group);
  MPFR_SAVE_EXPO_UPDATE_FLAGS (expo, __gmpfr_flags);
 end:
  MPFR_SAVE_EXPO_FREE (expo);
  return mpfr_check_range (s, inexact, rnd_mode);
}
