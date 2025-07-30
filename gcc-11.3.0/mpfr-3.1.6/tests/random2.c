/* mpfr_random2 -- Generate a positive random mpfr_t of specified size, with
   long runs of consecutive ones and zeros in the binary representation.

Copyright 1999, 2001-2004, 2006-2017 Free Software Foundation, Inc.
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

#include "mpfr-test.h"

#define LOGBITS_PER_BLOCK 4
#if GMP_NUMB_BITS < 32
#define BITS_PER_RANDCALL GMP_NUMB_BITS
#else
#define BITS_PER_RANDCALL 32
#endif

void
mpfr_random2 (mpfr_ptr x, mp_size_t size, mpfr_exp_t exp,
              gmp_randstate_t rstate)
{
  mp_size_t xn, k, ri;
  unsigned long sh;
  mp_limb_t *xp;
  mp_limb_t elimb, ran, acc;
  int ran_nbits, bit_pos, nb;

  if (MPFR_UNLIKELY(size == 0))
    {
      MPFR_SET_ZERO (x);
      MPFR_SET_POS (x);
      return ;
    }
  else if (size > 0)
    {
      MPFR_SET_POS (x);
    }
  else
    {
      MPFR_SET_NEG (x);
      size = -size;
    }

  xn = MPFR_LIMB_SIZE (x);
  xp = MPFR_MANT (x);
  if (size > xn)
    size = xn;
  k = xn - size;

  /* Code extracted from GMP, function mpn_random2, to avoid the use
     of GMP's internal random state in MPFR */

  mpfr_rand_raw (&elimb, rstate, BITS_PER_RANDCALL);
  ran = elimb;

  /* Start off at a random bit position in the most significant limb.  */
  bit_pos = GMP_NUMB_BITS - 1;
  ran >>= 6;                            /* Ideally   log2(GMP_NUMB_BITS) */
  ran_nbits = BITS_PER_RANDCALL - 6;    /* Ideally - log2(GMP_NUMB_BITS) */

  /* Bit 0 of ran chooses string of ones/string of zeroes.
     Make most significant limb be non-zero by setting bit 0 of RAN.  */
  ran |= 1;
  ri = xn - 1;

  acc = 0;
  while (ri >= k)
    {
      if (ran_nbits < LOGBITS_PER_BLOCK + 1)
        {
          mpfr_rand_raw (&elimb, rstate, BITS_PER_RANDCALL);
          ran = elimb;
          ran_nbits = BITS_PER_RANDCALL;
        }

      nb = (ran >> 1) % (1 << LOGBITS_PER_BLOCK) + 1;
      if ((ran & 1) != 0)
        {
          /* Generate a string of nb ones.  */
          if (nb > bit_pos)
            {
              xp[ri--] = acc | (((mp_limb_t) 2 << bit_pos) - 1);
              bit_pos += GMP_NUMB_BITS;
              bit_pos -= nb;
              acc = ((~(mp_limb_t) 1) << bit_pos) & GMP_NUMB_MASK;
            }
          else
            {
              bit_pos -= nb;
              acc |= (((mp_limb_t) 2 << nb) - 2) << bit_pos;
            }
        }
      else
        {
          /* Generate a string of nb zeroes.  */
          if (nb > bit_pos)
            {
              xp[ri--] = acc;
              acc = 0;
              bit_pos += GMP_NUMB_BITS;
            }
          bit_pos -= nb;
        }
      ran_nbits -= LOGBITS_PER_BLOCK + 1;
      ran >>= LOGBITS_PER_BLOCK + 1;
    }

  /* Set mandatory most significant bit.  */
  /* xp[xn - 1] |= MPFR_LIMB_HIGHBIT; */

  if (k != 0)
    {
      /* Clear last limbs */
      MPN_ZERO (xp, k);
    }
  else
    {
      /* Mask off non significant bits in the low limb.  */
      MPFR_UNSIGNED_MINUS_MODULO (sh, MPFR_PREC (x));
      xp[0] &= ~MPFR_LIMB_MASK (sh);
    }

  /* Generate random exponent.  */
  mpfr_rand_raw (&elimb, RANDS, GMP_NUMB_BITS);
  exp = ABS (exp);
  MPFR_SET_EXP (x, elimb % (2 * exp + 1) - exp);

  return ;
}
