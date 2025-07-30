/* mpz_rrandomb -- Generate a positive random mpz_t of specified bit size, with
   long runs of consecutive ones and zeros in the binary representation.
   Meant for testing of other MP routines.

Copyright 2000-2002, 2004, 2012 Free Software Foundation, Inc.

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

static void gmp_rrandomb (mp_ptr, gmp_randstate_t, mp_bitcnt_t);

void
mpz_rrandomb (mpz_ptr x, gmp_randstate_t rstate, mp_bitcnt_t nbits)
{
  mp_size_t nl;
  mp_ptr xp;

  nl = BITS_TO_LIMBS (nbits);
  if (nbits != 0)
    {
      xp = MPZ_NEWALLOC (x, nl);
      gmp_rrandomb (xp, rstate, nbits);
    }

  SIZ(x) = nl;
}

/* Ask _gmp_rand for 32 bits per call unless that's more than a limb can hold.
   Thus, we get the same random number sequence in the common cases.
   FIXME: We should always generate the same random number sequence!  */
#if GMP_NUMB_BITS < 32
#define BITS_PER_RANDCALL GMP_NUMB_BITS
#else
#define BITS_PER_RANDCALL 32
#endif

static void
gmp_rrandomb (mp_ptr rp, gmp_randstate_t rstate, mp_bitcnt_t nbits)
{
  mp_bitcnt_t bi;
  mp_limb_t ranm;		/* buffer for random bits */
  unsigned cap_chunksize, chunksize;
  mp_size_t i;

  /* Set entire result to 111..1  */
  i = BITS_TO_LIMBS (nbits) - 1;
  rp[i] = GMP_NUMB_MAX >> (GMP_NUMB_BITS - (nbits % GMP_NUMB_BITS)) % GMP_NUMB_BITS;
  for (i = i - 1; i >= 0; i--)
    rp[i] = GMP_NUMB_MAX;

  _gmp_rand (&ranm, rstate, BITS_PER_RANDCALL);
  cap_chunksize = nbits / (ranm % 4 + 1);
  cap_chunksize += cap_chunksize == 0; /* make it at least 1 */

  bi = nbits;

  for (;;)
    {
      _gmp_rand (&ranm, rstate, BITS_PER_RANDCALL);
      chunksize = 1 + ranm % cap_chunksize;
      bi = (bi < chunksize) ? 0 : bi - chunksize;

      if (bi == 0)
	break;			/* low chunk is ...1 */

      rp[bi / GMP_NUMB_BITS] ^= CNST_LIMB (1) << bi % GMP_NUMB_BITS;

      _gmp_rand (&ranm, rstate, BITS_PER_RANDCALL);
      chunksize = 1 + ranm % cap_chunksize;
      bi = (bi < chunksize) ? 0 : bi - chunksize;

      mpn_incr_u (rp + bi / GMP_NUMB_BITS, CNST_LIMB (1) << bi % GMP_NUMB_BITS);

      if (bi == 0)
	break;			/* low chunk is ...0 */
    }
}
