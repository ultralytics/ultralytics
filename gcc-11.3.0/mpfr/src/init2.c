/* mpfr_init2 -- initialize a floating-point number with given precision

Copyright 2001-2017 Free Software Foundation, Inc.
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

void
mpfr_init2 (mpfr_ptr x, mpfr_prec_t p)
{
  mp_size_t xsize;
  mpfr_limb_ptr tmp;

  /* Check if we can represent the number of limbs
   * associated to the maximum of mpfr_prec_t*/
  MPFR_ASSERTN( MP_SIZE_T_MAX >= (MPFR_PREC_MAX/MPFR_BYTES_PER_MP_LIMB) );

  /* Check for correct GMP_NUMB_BITS and MPFR_BYTES_PER_MP_LIMB */
  MPFR_ASSERTN( GMP_NUMB_BITS == MPFR_BYTES_PER_MP_LIMB * CHAR_BIT
                && sizeof(mp_limb_t) == MPFR_BYTES_PER_MP_LIMB );

  MPFR_ASSERTN (mp_bits_per_limb == GMP_NUMB_BITS);

  /* Check for correct EXP NAN, ZERO & INF in both mpfr.h and mpfr-impl.h */
  MPFR_ASSERTN( __MPFR_EXP_NAN  == MPFR_EXP_NAN  );
  MPFR_ASSERTN( __MPFR_EXP_ZERO == MPFR_EXP_ZERO );
  MPFR_ASSERTN( __MPFR_EXP_INF  == MPFR_EXP_INF  );

  MPFR_ASSERTN( MPFR_EMAX_MAX <= (MPFR_EXP_MAX >> 1)  );
  MPFR_ASSERTN( MPFR_EMIN_MIN >= -(MPFR_EXP_MAX >> 1) );

  /* p=1 is not allowed since the rounding to nearest even rule requires at
     least two bits of mantissa: the neighbours of 3/2 are 1*2^0 and 1*2^1,
     which both have an odd mantissa */
  MPFR_ASSERTN(p >= MPFR_PREC_MIN && p <= MPFR_PREC_MAX);

  xsize = MPFR_PREC2LIMBS (p);
  tmp   = (mpfr_limb_ptr) (*__gmp_allocate_func)(MPFR_MALLOC_SIZE(xsize));

  MPFR_PREC(x) = p;                /* Set prec */
  MPFR_EXP (x) = MPFR_EXP_INVALID; /* make sure that the exp field has a
                                      valid value in the C point of view */
  MPFR_SET_POS(x);                 /* Set a sign */
  MPFR_SET_MANT_PTR(x, tmp);       /* Set Mantissa ptr */
  MPFR_SET_ALLOC_SIZE(x, xsize);   /* Fix alloc size of Mantissa */
  MPFR_SET_NAN(x);                 /* initializes to NaN */
}

#ifdef MPFR_USE_OWN_MPFR_TMP_ALLOC
static unsigned char mpfr_stack_tab[8000000];
unsigned char *mpfr_stack = mpfr_stack_tab;
#endif
