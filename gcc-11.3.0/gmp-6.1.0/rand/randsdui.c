/* gmp_randseed_ui (state, seed) -- Set initial seed SEED in random
   state STATE.

Copyright 2000, 2002 Free Software Foundation, Inc.

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
gmp_randseed_ui (gmp_randstate_t rstate,
                 unsigned long int seed)
{
  mpz_t zseed;
  mp_limb_t zlimbs[LIMBS_PER_ULONG];

  MPZ_FAKE_UI (zseed, zlimbs, seed);
  gmp_randseed (rstate, zseed);
}
