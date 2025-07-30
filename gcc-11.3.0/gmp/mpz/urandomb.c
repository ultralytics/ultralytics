/* mpz_urandomb (rop, state, n) -- Generate a uniform pseudorandom
   integer in the range 0 to 2^N - 1, inclusive, using STATE as the
   random state previously initialized by a call to gmp_randinit().

Copyright 1999, 2000, 2002 Free Software Foundation, Inc.

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

void
mpz_urandomb (mpz_ptr rop, gmp_randstate_t rstate, mp_bitcnt_t nbits)
{
  mp_ptr rp;
  mp_size_t size;

  size = BITS_TO_LIMBS (nbits);
  rp = MPZ_NEWALLOC (rop, size);

  _gmp_rand (rp, rstate, nbits);
  MPN_NORMALIZE (rp, size);
  SIZ (rop) = size;
}
