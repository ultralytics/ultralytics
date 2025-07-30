/* mpf_cmp_z -- Compare a float with an integer.

Copyright 2015 Free Software Foundation, Inc.

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

int
mpf_cmp_z (mpf_srcptr u, mpz_srcptr v) __GMP_NOTHROW
{
  mpf_t vf;
  mp_size_t size;

  SIZ (vf) = size = SIZ (v);
  EXP (vf) = size = ABS (size);
  /* PREC (vf) = size; */ 
  PTR (vf) = PTR (v);

  return mpf_cmp (u, vf);
}
