/* mpn/generic/mod_1_1.c method 2.

Copyright 2011 Free Software Foundation, Inc.

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

#undef MOD_1_1P_METHOD
#define MOD_1_1P_METHOD 2
#undef mpn_mod_1_1p
#undef mpn_mod_1_1p_cps
#define mpn_mod_1_1p mpn_mod_1_1p_2
#define mpn_mod_1_1p_cps mpn_mod_1_1p_cps_2

#include "mpn/generic/mod_1_1.c"
