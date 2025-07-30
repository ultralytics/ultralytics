/* mpz_sizeinbase(x, base) -- return an approximation to the number of
   character the integer X would have printed in base BASE.  The
   approximation is never too small.

Copyright 1991, 1993-1995, 2001, 2002 Free Software Foundation, Inc.

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
#include "longlong.h"

size_t
mpz_sizeinbase (mpz_srcptr x, int base) __GMP_NOTHROW
{
  size_t  result;
  MPN_SIZEINBASE (result, PTR(x), ABSIZ(x), base);
  return result;
}
