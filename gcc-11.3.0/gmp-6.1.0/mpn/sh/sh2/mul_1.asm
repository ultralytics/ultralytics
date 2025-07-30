dnl  SH2 mpn_mul_1 -- Multiply a limb vector with a limb and store the result
dnl  in a second limb vector.

dnl  Copyright 1995, 2000, 2011 Free Software Foundation, Inc.

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
C res_ptr	r4
C s1_ptr	r5
C size		r6
C s2_limb	r7

changecom(blah)			C disable # to make all C comments below work

ASM_START()
PROLOGUE(mpn_mul_1)
	mov	#0,r2		C cy_limb = 0
	mov	#0,r0		C Keep r0 = 0 for entire loop
	clrt

L(top):	mov.l	@r5+,r3
	dmulu.l	r3,r7
	sts	macl,r1
	addc	r2,r1
	sts	mach,r2
	addc	r0,r2		C propagate carry to cy_limb (dt clobbers T)
	dt	r6
	mov.l	r1,@r4
	bf.s	L(top)
	add	#4,r4

	rts
	mov	r2,r0
EPILOGUE()
