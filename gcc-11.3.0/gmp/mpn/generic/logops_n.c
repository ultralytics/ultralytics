/* mpn_and_n, mpn_ior_n, etc -- mpn logical operations.

Copyright 2009 Free Software Foundation, Inc.

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

#ifdef OPERATION_and_n
#define func __MPN(and_n)
#define call mpn_and_n
#endif

#ifdef OPERATION_andn_n
#define func __MPN(andn_n)
#define call mpn_andn_n
#endif

#ifdef OPERATION_nand_n
#define func __MPN(nand_n)
#define call mpn_nand_n
#endif

#ifdef OPERATION_ior_n
#define func __MPN(ior_n)
#define call mpn_ior_n
#endif

#ifdef OPERATION_iorn_n
#define func __MPN(iorn_n)
#define call mpn_iorn_n
#endif

#ifdef OPERATION_nior_n
#define func __MPN(nior_n)
#define call mpn_nior_n
#endif

#ifdef OPERATION_xor_n
#define func __MPN(xor_n)
#define call mpn_xor_n
#endif

#ifdef OPERATION_xnor_n
#define func __MPN(xnor_n)
#define call mpn_xnor_n
#endif

void
func (mp_ptr rp, mp_srcptr up, mp_srcptr vp, mp_size_t n)
{
  call (rp, up, vp, n);
}
