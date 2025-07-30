dnl  mc68020 mpn_mul_1 -- mpn by limb multiply

dnl  Copyright 1992, 1994, 1996, 1999-2002 Free Software Foundation, Inc.

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

C         cycles/limb
C 68040:     24

C INPUT PARAMETERS
C res_ptr	(sp + 4)
C s1_ptr	(sp + 8)
C s1_size	(sp + 12)
C s2_limb	(sp + 16)


define(res_ptr, `a0')
define(s1_ptr,  `a1')
define(s1_size, `d2')
define(s2_limb, `d4')


PROLOGUE(mpn_mul_1)

C Save used registers on the stack.
	moveml	d2-d4, M(-,sp)

C	movel	d2, M(-,sp)
C	movel	d3, M(-,sp)
C	movel	d4, M(-,sp)

C Copy the arguments to registers.  Better use movem?
	movel	M(sp,16), res_ptr
	movel	M(sp,20), s1_ptr
	movel	M(sp,24), s1_size
	movel	M(sp,28), s2_limb

	eorw	#1, s1_size
	clrl	d1
	lsrl	#1, s1_size
	bcc	L(L1)
	subql	#1, s1_size
	subl	d0, d0		C (d0,cy) <= (0,0)

L(Loop):
	movel	M(s1_ptr,+), d3
	mulul	s2_limb, d1:d3
	addxl	d0, d3
	movel	d3, M(res_ptr,+)
L(L1):	movel	M(s1_ptr,+), d3
	mulul	s2_limb, d0:d3
	addxl	d1, d3
	movel	d3, M(res_ptr,+)

	dbf	s1_size, L(Loop)
	clrl	d3
	addxl	d3, d0
	subl	#0x10000, s1_size
	bcc	L(Loop)

C Restore used registers from stack frame.
	moveml	M(sp,+), d2-d4

C	movel	M(sp,+),d4
C	movel	M(sp,+),d3
C	movel	M(sp,+),d2

	rts

EPILOGUE(mpn_mul_1)
