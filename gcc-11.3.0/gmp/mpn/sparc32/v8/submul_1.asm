dnl  SPARC v8 mpn_submul_1 -- Multiply a limb vector with a limb and
dnl  subtract the result from a second limb vector.

dnl  Copyright 1992-1994, 2000 Free Software Foundation, Inc.

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
C res_ptr	o0
C s1_ptr	o1
C size		o2
C s2_limb	o3

ASM_START()
PROLOGUE(mpn_submul_1)
	sub	%g0,%o2,%o2		C negate ...
	sll	%o2,2,%o2		C ... and scale size
	sub	%o1,%o2,%o1		C o1 is offset s1_ptr
	sub	%o0,%o2,%g1		C g1 is offset res_ptr

	mov	0,%o0			C clear cy_limb

L(loop):
	ld	[%o1+%o2],%o4
	ld	[%g1+%o2],%g2
	umul	%o4,%o3,%o5
	rd	%y,%g3
	addcc	%o5,%o0,%o5
	addx	%g3,0,%o0
	subcc	%g2,%o5,%g2
	addx	%o0,0,%o0
	st	%g2,[%g1+%o2]

	addcc	%o2,4,%o2
	bne	L(loop)
	 nop

	retl
	 nop
EPILOGUE(mpn_submul_1)
