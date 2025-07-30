dnl  mc68020 mpn_umul_ppmm -- limb by limb multiplication

dnl  Copyright 1999-2001 Free Software Foundation, Inc.

dnl  This file is part of the GNU MP Library.
dnl
dnl  The GNU MP Library is free software; you can redistribute it and/or modify
dnl  it under the terms of either:
dnl
dnl    * the GNU Lesser General Public License as published by the Free
dnl      Software Foundation; either version 3 of the License, or (at your
dnl      option) any later version.
dnl
dnl  or
dnl
dnl    * the GNU General Public License as published by the Free Software
dnl      Foundation; either version 2 of the License, or (at your option) any
dnl      later version.
dnl
dnl  or both in parallel, as here.
dnl
dnl  The GNU MP Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
dnl  for more details.
dnl
dnl  You should have received copies of the GNU General Public License and the
dnl  GNU Lesser General Public License along with the GNU MP Library.  If not,
dnl  see https://www.gnu.org/licenses/.

include(`../config.m4')


C mp_limb_t mpn_umul_ppmm (mp_limb_t *lp, mp_limb_t x, mp_limb_t y);
C

PROLOGUE(mpn_umul_ppmm)
	movel	M(sp,4), a0	C lp
	movel	M(sp,8), d1	C x
	movel	M(sp,12), d0	C y
	mulul	d0, d0:d1
	movel	d1, M(a0)	C low
	rts
EPILOGUE(mpn_umul_ppmm)
