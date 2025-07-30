dnl  ARM mpn_com.

dnl  Copyright 2003, 2012 Free Software Foundation, Inc.

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

C	     cycles/limb
C StrongARM	 ?
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 2.0
C Cortex-A15	 1.75

define(`rp', `r0')
define(`up', `r1')
define(`n',  `r2')

ASM_START()
PROLOGUE(mpn_com)
	tst	n, #1
	beq	L(skip1)
	ldr	r3, [up], #4
	mvn	r3, r3
	str	r3, [rp], #4
L(skip1):
	tst	n, #2
	beq	L(skip2)
	ldmia	up!, { r3, r12 }		C load 2 limbs
	mvn	r3, r3
	mvn	r12, r12
	stmia	rp!, { r3, r12 }		C store 2 limbs
L(skip2):
	bics	n, n, #3
	beq	L(rtn)
	stmfd	sp!, { r7, r8, r9 }		C save regs on stack

L(top):	ldmia	up!, { r3, r8, r9, r12 }	C load 4 limbs
	subs	n, n, #4
	mvn	r3, r3
	mvn	r8, r8
	mvn	r9, r9
	mvn	r12, r12
	stmia	rp!, { r3, r8, r9, r12 }	C store 4 limbs
	bne	L(top)

	ldmfd	sp!, { r7, r8, r9 }		C restore regs from stack
L(rtn):	bx	lr
EPILOGUE()
