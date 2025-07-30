/* mpq_inv(dest,src) -- invert a rational number, i.e. set DEST to SRC
   with the numerator and denominator swapped.

Copyright 1991, 1994, 1995, 2000, 2001, 2012 Free Software Foundation, Inc.

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
mpq_inv (mpq_ptr dest, mpq_srcptr src)
{
  mp_size_t num_size = SIZ(NUM(src));
  mp_size_t den_size = SIZ(DEN(src));

  if (num_size < 0)
    {
      num_size = -num_size;
      den_size = -den_size;
    }
  else if (UNLIKELY (num_size == 0))
    DIVIDE_BY_ZERO;

  SIZ(DEN(dest)) = num_size;
  SIZ(NUM(dest)) = den_size;

  /* If dest == src we may just swap the numerator and denominator;
     we ensured that the new denominator is positive.  */

  if (dest == src)
    {
      MP_PTR_SWAP (PTR(NUM(dest)), PTR(DEN(dest)));
      MP_SIZE_T_SWAP (ALLOC(NUM(dest)), ALLOC(DEN(dest)));
    }
  else
    {
      mp_ptr dp;

      den_size = ABS (den_size);
      dp = MPZ_NEWALLOC (NUM(dest), den_size);
      MPN_COPY (dp, PTR(DEN(src)), den_size);

      dp = MPZ_NEWALLOC (DEN(dest), num_size);
      MPN_COPY (dp, PTR(NUM(src)), num_size);
    }
}
