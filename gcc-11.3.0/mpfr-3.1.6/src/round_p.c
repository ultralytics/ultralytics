/* mpfr_round_p -- check if an approximation is roundable.

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

#include "mpfr-impl.h"

/* Check against mpfr_can_round? */
#ifdef MPFR_WANT_ASSERT
# if MPFR_WANT_ASSERT >= 2
int mpfr_round_p_2 (mp_limb_t *, mp_size_t, mpfr_exp_t, mpfr_prec_t);
int
mpfr_round_p (mp_limb_t *bp, mp_size_t bn, mpfr_exp_t err0, mpfr_prec_t prec)
{
  int i1, i2;

  MPFR_ASSERTN(bp[bn - 1] & MPFR_LIMB_HIGHBIT);

  i1 = mpfr_round_p_2 (bp, bn, err0, prec);

  /* Note: since revision 10747, mpfr_can_round_raw is supposed to be always
     correct, whereas mpfr_round_p_2 might return 0 in some cases where one
     could round, for example with err0=67 and prec=54:
     b = 1111101101010001100011111011100010100011101111011011101111111111
     thus we cannot compare i1 and i2, we only can check that we don't have
     i1 <> 0 and i2 = 0.
  */
  i2 = mpfr_can_round_raw (bp, bn, MPFR_SIGN_POS, err0,
                           MPFR_RNDN, MPFR_RNDZ, prec);
  if (i1 && (i2 == 0))
    {
      fprintf (stderr, "mpfr_round_p(%d) != mpfr_can_round(%d)!\n"
               "bn = %lu, err0 = %ld, prec = %lu\nbp = ", i1, i2,
               (unsigned long) bn, (long) err0, (unsigned long) prec);
      gmp_fprintf (stderr, "%NX\n", bp, bn);
      MPFR_ASSERTN (0);
    }

  return i1;
}
# define mpfr_round_p mpfr_round_p_2
# endif
#endif

/*
 * Assuming {bp, bn} is an approximation of a non-singular number
 * with error at most equal to 2^(EXP(b)-err0) (`err0' bits of b are known)
 * of direction unknown, check if we can round b toward zero with
 * precision prec.
 */
int
mpfr_round_p (mp_limb_t *bp, mp_size_t bn, mpfr_exp_t err0, mpfr_prec_t prec)
{
  mpfr_prec_t err;
  mp_size_t k, n;
  mp_limb_t tmp, mask;
  int s;

  MPFR_ASSERTD(bp[bn - 1] & MPFR_LIMB_HIGHBIT);

  err = (mpfr_prec_t) bn * GMP_NUMB_BITS;
  if (MPFR_UNLIKELY (err0 <= 0 || (mpfr_uexp_t) err0 <= prec || prec >= err))
    return 0;  /* can't round */
  err = MIN (err, (mpfr_uexp_t) err0);

  k = prec / GMP_NUMB_BITS;
  s = GMP_NUMB_BITS - prec%GMP_NUMB_BITS;
  n = err / GMP_NUMB_BITS - k;

  MPFR_ASSERTD (n >= 0);
  MPFR_ASSERTD (bn > k);

  /* Check first limb */
  bp += bn-1-k;
  tmp = *bp--;
  mask = s == GMP_NUMB_BITS ? MP_LIMB_T_MAX : MPFR_LIMB_MASK (s);
  tmp &= mask;

  if (MPFR_LIKELY (n == 0))
    {
      /* prec and error are in the same limb */
      s = GMP_NUMB_BITS - err % GMP_NUMB_BITS;
      MPFR_ASSERTD (s < GMP_NUMB_BITS);
      tmp  >>= s;
      mask >>= s;
      return tmp != 0 && tmp != mask;
    }
  else if (MPFR_UNLIKELY (tmp == 0))
    {
      /* Check if all (n-1) limbs are 0 */
      while (--n)
        if (*bp-- != 0)
          return 1;
      /* Check if final error limb is 0 */
      s = GMP_NUMB_BITS - err % GMP_NUMB_BITS;
      if (s == GMP_NUMB_BITS)
        return 0;
      tmp = *bp >> s;
      return tmp != 0;
    }
  else if (MPFR_UNLIKELY (tmp == mask))
    {
      /* Check if all (n-1) limbs are 11111111111111111 */
      while (--n)
        if (*bp-- != MP_LIMB_T_MAX)
          return 1;
      /* Check if final error limb is 0 */
      s = GMP_NUMB_BITS - err % GMP_NUMB_BITS;
      if (s == GMP_NUMB_BITS)
        return 0;
      tmp = *bp >> s;
      return tmp != (MP_LIMB_T_MAX >> s);
    }
  else
    {
      /* First limb is different from 000000 or 1111111 */
      return 1;
    }
}
