/* MPFR internal constant FP numbers

Copyright 2005-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

#include "mpfr-impl.h"

static const mp_limb_t __gmpfr_limb1[1] = {MPFR_LIMB_HIGHBIT};
const mpfr_t __gmpfr_one = {{2, MPFR_SIGN_POS, 1, (mp_limb_t*)__gmpfr_limb1}};
const mpfr_t __gmpfr_two = {{2, MPFR_SIGN_POS, 2, (mp_limb_t*)__gmpfr_limb1}};
const mpfr_t __gmpfr_four ={{2, MPFR_SIGN_POS, 3, (mp_limb_t*)__gmpfr_limb1}};
