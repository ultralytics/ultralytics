/* mpz_set_f (dest_integer, src_float) -- Assign DEST_INTEGER from SRC_FLOAT.

Copyright 1996, 2001, 2012 Free Software Foundation, Inc.

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
mpz_set_f (mpz_ptr w, mpf_srcptr u)
{
  mp_ptr    wp, up;
  mp_size_t size;
  mp_exp_t  exp;

  /* abs(u)<1 truncates to zero */
  exp = EXP (u);
  if (exp <= 0)
    {
      SIZ(w) = 0;
      return;
    }

  wp = MPZ_REALLOC (w, exp);
  up = PTR(u);

  size = SIZ (u);
  SIZ(w) = (size >= 0 ? exp : -exp);
  size = ABS (size);

  if (exp > size)
    {
      /* pad with low zeros to get a total "exp" many limbs */
      mp_size_t  zeros = exp - size;
      MPN_ZERO (wp, zeros);
      wp += zeros;
    }
  else
    {
      /* exp<=size, truncate to the high "exp" many limbs */
      up += (size - exp);
      size = exp;
    }

  MPN_COPY (wp, up, size);
}
