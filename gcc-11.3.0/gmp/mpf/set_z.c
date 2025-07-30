/* mpf_set_z -- Assign a float from an integer.

Copyright 1996, 2001, 2004 Free Software Foundation, Inc.

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
mpf_set_z (mpf_ptr r, mpz_srcptr u)
{
  mp_ptr rp, up;
  mp_size_t size, asize;
  mp_size_t prec;

  prec = PREC (r) + 1;
  size = SIZ (u);
  asize = ABS (size);
  rp = PTR (r);
  up = PTR (u);

  EXP (r) = asize;

  if (asize > prec)
    {
      up += asize - prec;
      asize = prec;
    }

  SIZ (r) = size >= 0 ? asize : -asize;
  MPN_COPY (rp, up, asize);
}
