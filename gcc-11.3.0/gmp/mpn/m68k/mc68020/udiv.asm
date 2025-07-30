dnl  mc68020 mpn_udiv_qrnnd -- 2x1 limb division

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


C mp_limb_t mpn_udiv_qrnnd (mp_limb_t *rp,
C                           mp_limb_t nh, mp_limb_t nl, mp_limb_t d);
C

PROLOGUE(mpn_udiv_qrnnd)
	movel	M(sp,4), a0	C rp
	movel	M(sp,8), d1	C nh
	movel	M(sp,12), d0	C nl
	divul	M(sp,16), d1:d0
	movel	d1, M(a0)	C r
	rts
EPILOGUE(mpn_udiv_qrnnd)
