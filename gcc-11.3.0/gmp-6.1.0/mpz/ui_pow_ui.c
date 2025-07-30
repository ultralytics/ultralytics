/* mpz_ui_pow_ui -- ulong raised to ulong.

Copyright 2001, 2002 Free Software Foundation, Inc.

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
mpz_ui_pow_ui (mpz_ptr r, unsigned long b, unsigned long e)
{
#if GMP_NAIL_BITS != 0
  if (b > GMP_NUMB_MAX)
    {
      mp_limb_t bb[2];
      bb[0] = b & GMP_NUMB_MASK;
      bb[1] = b >> GMP_NUMB_BITS;
      mpz_n_pow_ui (r, bb, (mp_size_t) 2, e);
    }
  else
#endif
    {
#ifdef _LONG_LONG_LIMB
      /* i386 gcc 2.95.3 doesn't recognise blimb can be eliminated when
	 mp_limb_t is an unsigned long, so only use a separate blimb when
	 necessary.  */
      mp_limb_t  blimb = b;
      mpz_n_pow_ui (r, &blimb, (mp_size_t) (b != 0), e);
#else
      mpz_n_pow_ui (r, &b,     (mp_size_t) (b != 0), e);
#endif
    }
}
