/* Test mpz_sizeinbase.

Copyright 2001, 2002 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include <stdio.h>
#include <stdlib.h>
#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


#if 0
  /* Disabled due to the bogosity of trying to fake an _mp_d pointer to
     below an object.  Has been seen to fail on a hppa system and on ia64.  */


/* Create a fake mpz consisting of just a single 1 bit, with totbits being
   the total number of bits, inclusive of that 1 bit.  */
void
mpz_fake_bits (mpz_ptr z, unsigned long totbits)
{
  static mp_limb_t  n;
  unsigned long     zero_bits, zero_limbs;

  zero_bits = totbits - 1;
  zero_limbs = zero_bits / GMP_NUMB_BITS;
  zero_bits %= GMP_NUMB_BITS;

  SIZ(z) = zero_limbs + 1;
  PTR(z) = (&n) - (SIZ(z) - 1);
  n = CNST_LIMB(1) << zero_bits;

  ASSERT_ALWAYS (mpz_sizeinbase (z, 2) == totbits);
}


/* This was seen to fail on a GNU/Linux powerpc32 with gcc 2.95.2,
   apparently due to a doubtful value of mp_bases[10].chars_per_bit_exactly
   (0X1.34413509F79FDP-2 whereas 0X1.34413509F79FFP-2 is believed correct).
   Presumably this is a glibc problem when gcc converts the decimal string
   in mp_bases.c, or maybe it's only a function of the rounding mode during
   compilation.  */
void
check_sample (void)
{
  unsigned long  totbits = 198096465;
  int        base = 10;
  size_t     want = 59632979;
  size_t     got;
  mpz_t      z;

  mpz_fake_bits (z, totbits);
  got = mpz_sizeinbase (z, base);
  if (got != want)
    {
      printf ("mpz_sizeinbase\n");
      printf ("  base    %d\n",  base);
      printf ("  totbits %lu\n", totbits);
      printf ("  got     %u\n",  got);
      printf ("  want    %u\n",  want);
      abort ();
    }
}
#endif

int
main (void)
{
  tests_start ();

  /* check_sample (); */

  tests_end ();
  exit (0);
}
