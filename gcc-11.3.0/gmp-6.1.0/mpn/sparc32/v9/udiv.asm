dnl  SPARC v9 32-bit mpn_udiv_qrnnd - division support for longlong.h.

dnl  Copyright 2002, 2003 Free Software Foundation, Inc.

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

C INPUT PARAMETERS
C rem_ptr	o0
C n1		o1
C n0		o2
C d		o3

ASM_START()
PROLOGUE(mpn_udiv_qrnnd)
	sllx	%o1, 32, %g1		C shift upper dividend limb
	srl	%o2, 0, %g2		C zero extend lower dividend limb
	srl	%o3, 0, %g3		C zero extend divisor
	or	%g2, %g1, %g1		C assemble 64-bit dividend
	udivx	%g1, %g3, %g1
	mulx	%g1, %g3, %g4
	sub	%g2, %g4, %g2
	st	%g2, [%o0]		C store remainder
	retl
	mov	%g1, %o0		C return quotient
EPILOGUE(mpn_udiv_qrnnd)
