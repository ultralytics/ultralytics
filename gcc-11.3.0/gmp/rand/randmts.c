/* Mersenne Twister pseudo-random number generator functions.

Copyright 2002, 2003, 2013, 2014 Free Software Foundation, Inc.

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
#include "randmt.h"


/* Calculate (b^e) mod (2^n-k) for e=1074888996, n=19937 and k=20023,
   needed by the seeding function below.  */
static void
mangle_seed (mpz_ptr r)
{
  mpz_t          t, b;
  unsigned long  e = 0x40118124;
  unsigned long  bit = 0x20000000;

  mpz_init2 (t, 19937L);
  mpz_init_set (b, r);

  do
    {
      mpz_mul (r, r, r);

    reduce:
      for (;;)
        {
          mpz_tdiv_q_2exp (t, r, 19937L);
          if (SIZ (t) == 0)
            break;
          mpz_tdiv_r_2exp (r, r, 19937L);
          mpz_addmul_ui (r, t, 20023L);
        }

      if ((e & bit) != 0)
        {
          e ^= bit;
          mpz_mul (r, r, b);
          goto reduce;
        }

      bit >>= 1;
    }
  while (bit != 0);

  mpz_clear (t);
  mpz_clear (b);
}


/* Seeding function.  Uses powering modulo a non-Mersenne prime to obtain
   a permutation of the input seed space.  The modulus is 2^19937-20023,
   which is probably prime.  The power is 1074888996.  In order to avoid
   seeds 0 and 1 generating invalid or strange output, the input seed is
   first manipulated as follows:

     seed1 = seed mod (2^19937-20027) + 2

   so that seed1 lies between 2 and 2^19937-20026 inclusive. Then the
   powering is performed as follows:

     seed2 = (seed1^1074888996) mod (2^19937-20023)

   and then seed2 is used to bootstrap the buffer.

   This method aims to give guarantees that:
     a) seed2 will never be zero,
     b) seed2 will very seldom have a very low population of ones in its
	binary representation, and
     c) every seed between 0 and 2^19937-20028 (inclusive) will yield a
	different sequence.

   CAVEATS:

   The period of the seeding function is 2^19937-20027.  This means that
   with seeds 2^19937-20027, 2^19937-20026, ... the exact same sequences
   are obtained as with seeds 0, 1, etc.; it also means that seed -1
   produces the same sequence as seed 2^19937-20028, etc.
 */

static void
randseed_mt (gmp_randstate_t rstate, mpz_srcptr seed)
{
  int i;
  size_t cnt;

  gmp_rand_mt_struct *p;
  mpz_t mod;    /* Modulus.  */
  mpz_t seed1;  /* Intermediate result.  */

  p = (gmp_rand_mt_struct *) RNG_STATE (rstate);

  mpz_init2 (mod, 19938L);
  mpz_init2 (seed1, 19937L);

  mpz_setbit (mod, 19937L);
  mpz_sub_ui (mod, mod, 20027L);
  mpz_mod (seed1, seed, mod);	/* Reduce `seed' modulo `mod'.  */
  mpz_clear (mod);
  mpz_add_ui (seed1, seed1, 2L);	/* seed1 is now ready.  */
  mangle_seed (seed1);	/* Perform the mangling by powering.  */

  /* Copy the last bit into bit 31 of mt[0] and clear it.  */
  p->mt[0] = (mpz_tstbit (seed1, 19936L) != 0) ? 0x80000000 : 0;
  mpz_clrbit (seed1, 19936L);

  /* Split seed1 into N-1 32-bit chunks.  */
  mpz_export (&p->mt[1], &cnt, -1, sizeof (p->mt[1]), 0,
              8 * sizeof (p->mt[1]) - 32, seed1);
  mpz_clear (seed1);
  cnt++;
  ASSERT (cnt <= N);
  while (cnt < N)
    p->mt[cnt++] = 0;

  /* Warm the generator up if necessary.  */
  if (WARM_UP != 0)
    for (i = 0; i < WARM_UP / N; i++)
      __gmp_mt_recalc_buffer (p->mt);

  p->mti = WARM_UP % N;
}


static const gmp_randfnptr_t Mersenne_Twister_Generator = {
  randseed_mt,
  __gmp_randget_mt,
  __gmp_randclear_mt,
  __gmp_randiset_mt
};

/* Initialize MT-specific data.  */
void
gmp_randinit_mt (gmp_randstate_t rstate)
{
  __gmp_randinit_mt_noseed (rstate);
  RNG_FNPTR (rstate) = (void *) &Mersenne_Twister_Generator;
}
