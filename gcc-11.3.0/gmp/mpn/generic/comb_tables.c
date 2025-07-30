/* Const tables shared among combinatoric functions.

   THE CONTENTS OF THIS FILE ARE FOR INTERNAL USE AND ARE ALMOST CERTAIN TO
   BE SUBJECT TO INCOMPATIBLE CHANGES IN FUTURE GNU MP RELEASES.

Copyright 2012 Free Software Foundation, Inc.

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

/* Entry i contains (i!/2^t) where t is chosen such that the parenthesis
   is an odd integer. */
const mp_limb_t __gmp_oddfac_table[] = { ONE_LIMB_ODD_FACTORIAL_TABLE, ONE_LIMB_ODD_FACTORIAL_EXTTABLE };

/* Entry i contains ((2i+1)!!/2^t) where t is chosen such that the parenthesis
   is an odd integer. */
const mp_limb_t __gmp_odd2fac_table[] = { ONE_LIMB_ODD_DOUBLEFACTORIAL_TABLE };

/* Entry i contains 2i-popc(2i). */
const unsigned char __gmp_fac2cnt_table[] = { TABLE_2N_MINUS_POPC_2N };

const mp_limb_t __gmp_limbroots_table[] = { NTH_ROOT_NUMB_MASK_TABLE };
