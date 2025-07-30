/* mpn_mod_1(dividend_ptr, dividend_size, divisor_limb) --
   Divide (DIVIDEND_PTR,,DIVIDEND_SIZE) by DIVISOR_LIMB.
   Return the single-limb remainder.
   There are no constraints on the value of the divisor.

Copyright 1991, 1993, 1994, 1999, 2000, 2002, 2007-2009, 2012 Free Software
Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include "gmp.h"
#include "gmp-impl.h"
#include "longlong.h"


/* The size where udiv_qrnnd_preinv should be used rather than udiv_qrnnd,
   meaning the quotient size where that should happen, the quotient size
   being how many udiv divisions will be done.

   The default is to use preinv always, CPUs where this doesn't suit have
   tuned thresholds.  Note in particular that preinv should certainly be
   used if that's the only division available (USE_PREINV_ALWAYS).  */

#ifndef MOD_1_NORM_THRESHOLD
#define MOD_1_NORM_THRESHOLD  0
#endif

#ifndef MOD_1_UNNORM_THRESHOLD
#define MOD_1_UNNORM_THRESHOLD  0
#endif

#ifndef MOD_1U_TO_MOD_1_1_THRESHOLD
#define MOD_1U_TO_MOD_1_1_THRESHOLD  MP_SIZE_T_MAX /* default is not to use mpn_mod_1s */
#endif

#ifndef MOD_1N_TO_MOD_1_1_THRESHOLD
#define MOD_1N_TO_MOD_1_1_THRESHOLD  MP_SIZE_T_MAX /* default is not to use mpn_mod_1s */
#endif

#ifndef MOD_1_1_TO_MOD_1_2_THRESHOLD
#define MOD_1_1_TO_MOD_1_2_THRESHOLD  10
#endif

#ifndef MOD_1_2_TO_MOD_1_4_THRESHOLD
#define MOD_1_2_TO_MOD_1_4_THRESHOLD  20
#endif

#if TUNE_PROGRAM_BUILD && !HAVE_NATIVE_mpn_mod_1_1p
/* Duplicates declarations in tune/speed.h */
mp_limb_t mpn_mod_1_1p_1 (mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t [4]);
mp_limb_t mpn_mod_1_1p_2 (mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t [4]);

void mpn_mod_1_1p_cps_1 (mp_limb_t [4], mp_limb_t);
void mpn_mod_1_1p_cps_2 (mp_limb_t [4], mp_limb_t);

#undef mpn_mod_1_1p
#define mpn_mod_1_1p(ap, n, b, pre)			     \
  (mod_1_1p_method == 1 ? mpn_mod_1_1p_1 (ap, n, b, pre)     \
   : (mod_1_1p_method == 2 ? mpn_mod_1_1p_2 (ap, n, b, pre)  \
      : __gmpn_mod_1_1p (ap, n, b, pre)))

#undef mpn_mod_1_1p_cps
#define mpn_mod_1_1p_cps(pre, b)				\
  (mod_1_1p_method == 1 ? mpn_mod_1_1p_cps_1 (pre, b)		\
   : (mod_1_1p_method == 2 ? mpn_mod_1_1p_cps_2 (pre, b)	\
      : __gmpn_mod_1_1p_cps (pre, b)))
#endif /* TUNE_PROGRAM_BUILD && !HAVE_NATIVE_mpn_mod_1_1p */


/* The comments in mpn/generic/divrem_1.c apply here too.

   As noted in the algorithms section of the manual, the shifts in the loop
   for the unnorm case can be avoided by calculating r = a%(d*2^n), followed
   by a final (r*2^n)%(d*2^n).  In fact if it happens that a%(d*2^n) can
   skip a division where (a*2^n)%(d*2^n) can't then there's the same number
   of divide steps, though how often that happens depends on the assumed
   distributions of dividend and divisor.  In any case this idea is left to
   CPU specific implementations to consider.  */

static mp_limb_t
mpn_mod_1_unnorm (mp_srcptr up, mp_size_t un, mp_limb_t d)
{
  mp_size_t  i;
  mp_limb_t  n1, n0, r;
  mp_limb_t  dummy;
  int cnt;

  ASSERT (un > 0);
  ASSERT (d != 0);

  d <<= GMP_NAIL_BITS;

  /* Skip a division if high < divisor.  Having the test here before
     normalizing will still skip as often as possible.  */
  r = up[un - 1] << GMP_NAIL_BITS;
  if (r < d)
    {
      r >>= GMP_NAIL_BITS;
      un--;
      if (un == 0)
	return r;
    }
  else
    r = 0;

  /* If udiv_qrnnd doesn't need a normalized divisor, can use the simple
     code above. */
  if (! UDIV_NEEDS_NORMALIZATION
      && BELOW_THRESHOLD (un, MOD_1_UNNORM_THRESHOLD))
    {
      for (i = un - 1; i >= 0; i--)
	{
	  n0 = up[i] << GMP_NAIL_BITS;
	  udiv_qrnnd (dummy, r, r, n0, d);
	  r >>= GMP_NAIL_BITS;
	}
      return r;
    }

  count_leading_zeros (cnt, d);
  d <<= cnt;

  n1 = up[un - 1] << GMP_NAIL_BITS;
  r = (r << cnt) | (n1 >> (GMP_LIMB_BITS - cnt));

  if (UDIV_NEEDS_NORMALIZATION
      && BELOW_THRESHOLD (un, MOD_1_UNNORM_THRESHOLD))
    {
      mp_limb_t nshift;
      for (i = un - 2; i >= 0; i--)
	{
	  n0 = up[i] << GMP_NAIL_BITS;
	  nshift = (n1 << cnt) | (n0 >> (GMP_NUMB_BITS - cnt));
	  udiv_qrnnd (dummy, r, r, nshift, d);
	  r >>= GMP_NAIL_BITS;
	  n1 = n0;
	}
      udiv_qrnnd (dummy, r, r, n1 << cnt, d);
      r >>= GMP_NAIL_BITS;
      return r >> cnt;
    }
  else
    {
      mp_limb_t inv, nshift;
      invert_limb (inv, d);

      for (i = un - 2; i >= 0; i--)
	{
	  n0 = up[i] << GMP_NAIL_BITS;
	  nshift = (n1 << cnt) | (n0 >> (GMP_NUMB_BITS - cnt));
	  udiv_rnnd_preinv (r, r, nshift, d, inv);
	  r >>= GMP_NAIL_BITS;
	  n1 = n0;
	}
      udiv_rnnd_preinv (r, r, n1 << cnt, d, inv);
      r >>= GMP_NAIL_BITS;
      return r >> cnt;
    }
}

static mp_limb_t
mpn_mod_1_norm (mp_srcptr up, mp_size_t un, mp_limb_t d)
{
  mp_size_t  i;
  mp_limb_t  n0, r;
  mp_limb_t  dummy;

  ASSERT (un > 0);

  d <<= GMP_NAIL_BITS;

  ASSERT (d & GMP_LIMB_HIGHBIT);

  /* High limb is initial remainder, possibly with one subtract of
     d to get r<d.  */
  r = up[un - 1] << GMP_NAIL_BITS;
  if (r >= d)
    r -= d;
  r >>= GMP_NAIL_BITS;
  un--;
  if (un == 0)
    return r;

  if (BELOW_THRESHOLD (un, MOD_1_NORM_THRESHOLD))
    {
      for (i = un - 1; i >= 0; i--)
	{
	  n0 = up[i] << GMP_NAIL_BITS;
	  udiv_qrnnd (dummy, r, r, n0, d);
	  r >>= GMP_NAIL_BITS;
	}
      return r;
    }
  else
    {
      mp_limb_t  inv;
      invert_limb (inv, d);
      for (i = un - 1; i >= 0; i--)
	{
	  n0 = up[i] << GMP_NAIL_BITS;
	  udiv_rnnd_preinv (r, r, n0, d, inv);
	  r >>= GMP_NAIL_BITS;
	}
      return r;
    }
}

mp_limb_t
mpn_mod_1 (mp_srcptr ap, mp_size_t n, mp_limb_t b)
{
  ASSERT (n >= 0);
  ASSERT (b != 0);

  /* Should this be handled at all?  Rely on callers?  Note un==0 is currently
     required by mpz/fdiv_r_ui.c and possibly other places.  */
  if (n == 0)
    return 0;

  if (UNLIKELY ((b & GMP_NUMB_HIGHBIT) != 0))
    {
      if (BELOW_THRESHOLD (n, MOD_1N_TO_MOD_1_1_THRESHOLD))
	{
	  return mpn_mod_1_norm (ap, n, b);
	}
      else
	{
	  mp_limb_t pre[4];
	  mpn_mod_1_1p_cps (pre, b);
	  return mpn_mod_1_1p (ap, n, b, pre);
	}
    }
  else
    {
      if (BELOW_THRESHOLD (n, MOD_1U_TO_MOD_1_1_THRESHOLD))
	{
	  return mpn_mod_1_unnorm (ap, n, b);
	}
      else if (BELOW_THRESHOLD (n, MOD_1_1_TO_MOD_1_2_THRESHOLD))
	{
	  mp_limb_t pre[4];
	  mpn_mod_1_1p_cps (pre, b);
	  return mpn_mod_1_1p (ap, n, b << pre[1], pre);
	}
      else if (BELOW_THRESHOLD (n, MOD_1_2_TO_MOD_1_4_THRESHOLD) || UNLIKELY (b > GMP_NUMB_MASK / 4))
	{
	  mp_limb_t pre[5];
	  mpn_mod_1s_2p_cps (pre, b);
	  return mpn_mod_1s_2p (ap, n, b << pre[1], pre);
	}
      else
	{
	  mp_limb_t pre[7];
	  mpn_mod_1s_4p_cps (pre, b);
	  return mpn_mod_1s_4p (ap, n, b << pre[1], pre);
	}
    }
}
