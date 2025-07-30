/* primesieve (BIT_ARRAY, N) -- Fills the BIT_ARRAY with a mask for primes up to N.

Contributed to the GNU project by Marco Bodrato.

THE FUNCTION IN THIS FILE IS INTERNAL WITH A MUTABLE INTERFACE.
IT IS ONLY SAFE TO REACH IT THROUGH DOCUMENTED INTERFACES.
IN FACT, IT IS ALMOST GUARANTEED THAT IT WILL CHANGE OR
DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 2010-2012 Free Software Foundation, Inc.

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

/**************************************************************/
/* Section macros: common macros, for mswing/fac/bin (&sieve) */
/**************************************************************/

#define LOOP_ON_SIEVE_CONTINUE(prime,end,sieve)			\
    __max_i = (end);						\
								\
    do {							\
      ++__i;							\
      if (((sieve)[__index] & __mask) == 0)			\
	{							\
	  (prime) = id_to_n(__i)

#define LOOP_ON_SIEVE_BEGIN(prime,start,end,off,sieve)		\
  do {								\
    mp_limb_t __mask, __index, __max_i, __i;			\
								\
    __i = (start)-(off);					\
    __index = __i / GMP_LIMB_BITS;				\
    __mask = CNST_LIMB(1) << (__i % GMP_LIMB_BITS);		\
    __i += (off);						\
								\
    LOOP_ON_SIEVE_CONTINUE(prime,end,sieve)

#define LOOP_ON_SIEVE_STOP					\
	}							\
      __mask = __mask << 1 | __mask >> (GMP_LIMB_BITS-1);	\
      __index += __mask & 1;					\
    }  while (__i <= __max_i)					\

#define LOOP_ON_SIEVE_END					\
    LOOP_ON_SIEVE_STOP;						\
  } while (0)

/*********************************************************/
/* Section sieve: sieving functions and tools for primes */
/*********************************************************/

#if 0
static mp_limb_t
bit_to_n (mp_limb_t bit) { return (bit*3+4)|1; }
#endif

/* id_to_n (x) = bit_to_n (x-1) = (id*3+1)|1*/
static mp_limb_t
id_to_n  (mp_limb_t id)  { return id*3+1+(id&1); }

/* n_to_bit (n) = ((n-1)&(-CNST_LIMB(2)))/3U-1 */
static mp_limb_t
n_to_bit (mp_limb_t n) { return ((n-5)|1)/3U; }

#if 0
static mp_size_t
primesieve_size (mp_limb_t n) { return n_to_bit(n) / GMP_LIMB_BITS + 1; }
#endif

#if GMP_LIMB_BITS > 61
#define SIEVE_SEED CNST_LIMB(0x3294C9E069128480)
#define SEED_LIMIT 202
#else
#if GMP_LIMB_BITS > 30
#define SIEVE_SEED CNST_LIMB(0x69128480)
#define SEED_LIMIT 114
#else
#if GMP_LIMB_BITS > 15
#define SIEVE_SEED CNST_LIMB(0x8480)
#define SEED_LIMIT 54
#else
#if GMP_LIMB_BITS > 7
#define SIEVE_SEED CNST_LIMB(0x80)
#define SEED_LIMIT 34
#else
#define SIEVE_SEED CNST_LIMB(0x0)
#define SEED_LIMIT 24
#endif /* 7 */
#endif /* 15 */
#endif /* 30 */
#endif /* 61 */

static void
first_block_primesieve (mp_ptr bit_array, mp_limb_t n)
{
  mp_size_t bits, limbs;

  ASSERT (n > 4);

  bits  = n_to_bit(n);
  limbs = bits / GMP_LIMB_BITS + 1;

  /* FIXME: We can skip 5 too, filling with a 5-part pattern. */
  MPN_ZERO (bit_array, limbs);
  bit_array[0] = SIEVE_SEED;

  if ((bits + 1) % GMP_LIMB_BITS != 0)
    bit_array[limbs-1] |= MP_LIMB_T_MAX << ((bits + 1) % GMP_LIMB_BITS);

  if (n > SEED_LIMIT) {
    mp_limb_t mask, index, i;

    ASSERT (n > 49);

    mask = 1;
    index = 0;
    i = 1;
    do {
      if ((bit_array[index] & mask) == 0)
	{
	  mp_size_t step, lindex;
	  mp_limb_t lmask;
	  unsigned  maskrot;

	  step = id_to_n(i);
/*	  lindex = n_to_bit(id_to_n(i)*id_to_n(i)); */
	  lindex = i*(step+1)-1+(-(i&1)&(i+1));
/*	  lindex = i*(step+1+(i&1))-1+(i&1); */
	  if (lindex > bits)
	    break;

	  step <<= 1;
	  maskrot = step % GMP_LIMB_BITS;

	  lmask = CNST_LIMB(1) << (lindex % GMP_LIMB_BITS);
	  do {
	    bit_array[lindex / GMP_LIMB_BITS] |= lmask;
	    lmask = lmask << maskrot | lmask >> (GMP_LIMB_BITS - maskrot);
	    lindex += step;
	  } while (lindex <= bits);

/*	  lindex = n_to_bit(id_to_n(i)*bit_to_n(i)); */
	  lindex = i*(i*3+6)+(i&1);

	  lmask = CNST_LIMB(1) << (lindex % GMP_LIMB_BITS);
	  for ( ; lindex <= bits; lindex += step) {
	    bit_array[lindex / GMP_LIMB_BITS] |= lmask;
	    lmask = lmask << maskrot | lmask >> (GMP_LIMB_BITS - maskrot);
	  };
	}
      mask = mask << 1 | mask >> (GMP_LIMB_BITS-1);
      index += mask & 1;
      i++;
    } while (1);
  }
}

static void
block_resieve (mp_ptr bit_array, mp_size_t limbs, mp_limb_t offset,
		      mp_srcptr sieve, mp_limb_t sieve_bits)
{
  mp_size_t bits, step;

  ASSERT (limbs > 0);

  bits = limbs * GMP_LIMB_BITS - 1;

  /* FIXME: We can skip 5 too, filling with a 5-part pattern. */
  MPN_ZERO (bit_array, limbs);

  LOOP_ON_SIEVE_BEGIN(step,0,sieve_bits,0,sieve);
  {
    mp_size_t lindex;
    mp_limb_t lmask;
    unsigned  maskrot;

/*  lindex = n_to_bit(id_to_n(i)*id_to_n(i)); */
    lindex = __i*(step+1)-1+(-(__i&1)&(__i+1));
/*  lindex = __i*(step+1+(__i&1))-1+(__i&1); */
    if (lindex > bits + offset)
      break;

    step <<= 1;
    maskrot = step % GMP_LIMB_BITS;

    if (lindex < offset)
      lindex += step * ((offset - lindex - 1) / step + 1);

    lindex -= offset;

    lmask = CNST_LIMB(1) << (lindex % GMP_LIMB_BITS);
    for ( ; lindex <= bits; lindex += step) {
      bit_array[lindex / GMP_LIMB_BITS] |= lmask;
      lmask = lmask << maskrot | lmask >> (GMP_LIMB_BITS - maskrot);
    };

/*  lindex = n_to_bit(id_to_n(i)*bit_to_n(i)); */
    lindex = __i*(__i*3+6)+(__i&1);
    if (lindex > bits + offset)
      continue;

    if (lindex < offset)
      lindex += step * ((offset - lindex - 1) / step + 1);

    lindex -= offset;

    lmask = CNST_LIMB(1) << (lindex % GMP_LIMB_BITS);
    for ( ; lindex <= bits; lindex += step) {
      bit_array[lindex / GMP_LIMB_BITS] |= lmask;
      lmask = lmask << maskrot | lmask >> (GMP_LIMB_BITS - maskrot);
    };
  }
  LOOP_ON_SIEVE_END;
}

#define BLOCK_SIZE 2048

/* Fills bit_array with the characteristic function of composite
   numbers up to the parameter n. I.e. a bit set to "1" represent a
   composite, a "0" represent a prime.

   The primesieve_size(n) limbs pointed to by bit_array are
   overwritten. The returned value counts prime integers in the
   interval [4, n]. Note that n > 4.

   Even numbers and multiples of 3 are excluded "a priori", only
   numbers equivalent to +/- 1 mod 6 have their bit in the array.

   Once sieved, if the bit b is ZERO it represent a prime, the
   represented prime is bit_to_n(b), if the LSbit is bit 0, or
   id_to_n(b), if you call "1" the first bit.
 */

mp_limb_t
gmp_primesieve (mp_ptr bit_array, mp_limb_t n)
{
  mp_size_t size;
  mp_limb_t bits;

  ASSERT (n > 4);

  bits = n_to_bit(n);
  size = bits / GMP_LIMB_BITS + 1;

  if (size > BLOCK_SIZE * 2) {
    mp_size_t off;
    off = BLOCK_SIZE + (size % BLOCK_SIZE);
    first_block_primesieve (bit_array, id_to_n (off * GMP_LIMB_BITS));
    for ( ; off < size; off += BLOCK_SIZE)
      block_resieve (bit_array + off, BLOCK_SIZE, off * GMP_LIMB_BITS, bit_array, off * GMP_LIMB_BITS - 1);
  } else {
    first_block_primesieve (bit_array, n);
  }

  if ((bits + 1) % GMP_LIMB_BITS != 0)
    bit_array[size-1] |= MP_LIMB_T_MAX << ((bits + 1) % GMP_LIMB_BITS);


  return size * GMP_LIMB_BITS - mpn_popcount (bit_array, size);
}

#undef BLOCK_SIZE
#undef SEED_LIMIT
#undef SIEVE_SEED
#undef LOOP_ON_SIEVE_END
#undef LOOP_ON_SIEVE_STOP
#undef LOOP_ON_SIEVE_BEGIN
#undef LOOP_ON_SIEVE_CONTINUE
