/* mpn/generic/div_qr_1, using tuned threshold and method.

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

#define TUNE_PROGRAM_BUILD 1

#include "gmp.h"
#include "gmp-impl.h"

mp_limb_t mpn_div_qr_1n_pi1_1 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, mp_limb_t);
mp_limb_t mpn_div_qr_1n_pi1_2 (mp_ptr, mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t, mp_limb_t);

#if !HAVE_NATIVE_mpn_div_qr_1n_pi1
#define __gmpn_div_qr_1n_pi1 \
  (div_qr_1n_pi1_method == 1 ? mpn_div_qr_1n_pi1_1 : mpn_div_qr_1n_pi1_2)
#endif

#undef mpn_div_qr_1
#define mpn_div_qr_1 mpn_div_qr_1_tune

#include "mpn/generic/div_qr_1.c"
