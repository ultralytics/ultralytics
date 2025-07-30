/* mpn_sec_mul.

   Contributed to the GNU project by Torbjörn Granlund.

Copyright 2013 Free Software Foundation, Inc.

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
mpn_sec_mul (mp_ptr rp,
	     mp_srcptr ap, mp_size_t an,
	     mp_srcptr bp, mp_size_t bn,
	     mp_ptr tp)
{
  mpn_mul_basecase (rp, ap, an, bp, bn);
}

mp_size_t
mpn_sec_mul_itch (mp_size_t an, mp_size_t bn)
{
  return 0;
}
