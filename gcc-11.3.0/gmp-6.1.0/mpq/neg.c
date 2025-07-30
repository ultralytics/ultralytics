/* mpq_neg -- negate a rational.

Copyright 2000, 2001, 2012 Free Software Foundation, Inc.

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

#define __GMP_FORCE_mpq_neg 1

#include "gmp.h"
#include "gmp-impl.h"


void
mpq_neg (mpq_ptr dst, mpq_srcptr src)
{
  mp_size_t  num_size = SIZ(NUM(src));

  if (src != dst)
    {
      mp_size_t  size;
      mp_ptr dp;

      size = ABS(num_size);
      dp = MPZ_NEWALLOC (NUM(dst), size);
      MPN_COPY (dp, PTR(NUM(src)), size);

      size = SIZ(DEN(src));
      dp = MPZ_NEWALLOC (DEN(dst), size);
      SIZ(DEN(dst)) = size;
      MPN_COPY (dp, PTR(DEN(src)), size);
    }

  SIZ(NUM(dst)) = -num_size;
}
