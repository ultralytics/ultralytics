dnl  ARM mpn_copyi.

dnl  Contributed to the GNU project by Robert Harley and Torbjörn Granlund.

dnl  Copyright 2003, 2012, 2013 Free Software Foundation, Inc.

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
C Cortex-A9	 1.25-1.5
C Cortex-A15	 1.25

C TODO
C  * Consider wider unrolling.  Analogous 8-way code runs 10% faster on both A9
C    and A15.  But it probably slows things down for 8 <= n < a few dozen.

define(`rp', `r0')
define(`up', `r1')
define(`n',  `r2')

ASM_START()
PROLOGUE(mpn_copyi)
	tst	n, #1
	beq	L(skip1)
	ldr	r3, [up], #4
	str	r3, [rp], #4
L(skip1):
	tst	n, #2
	beq	L(skip2)
	ldmia	up!, { r3,r12 }
	stmia	rp!, { r3,r12 }
L(skip2):
	bics	n, n, #3
	beq	L(rtn)

	push	{ r4-r5 }
	subs	n, n, #4
	ldmia	up!, { r3,r4,r5,r12 }
	beq	L(end)

L(top):	subs	n, n, #4
	stmia	rp!, { r3,r4,r5,r12 }
	ldmia	up!, { r3,r4,r5,r12 }
	bne	L(top)

L(end):	stm	rp, { r3,r4,r5,r12 }
	pop	{ r4-r5 }
L(rtn):	bx	lr
EPILOGUE()
