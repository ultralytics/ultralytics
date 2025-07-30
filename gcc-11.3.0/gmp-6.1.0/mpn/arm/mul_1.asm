dnl  ARM mpn_mul_1 -- Multiply a limb vector with a limb and store the result
dnl  in a second limb vector.
dnl  Contributed by Robert Harley.

dnl  Copyright 1998, 2000, 2001, 2003, 2012 Free Software Foundation, Inc.

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
C StrongARM	6-8
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 4.75
C Cortex-A15	 ?

C We should rewrite this along the lines of addmul_1.asm.  That should save a
C cycle on StrongARM, and several cycles on XScale.

define(`rp',`r0')
define(`up',`r1')
define(`n',`r2')
define(`vl',`r3')


ASM_START()
PROLOGUE(mpn_mul_1)
	stmfd	sp!, { r8, r9, lr }
	ands	r12, n, #1
	beq	L(skip1)
	ldr	lr, [up], #4
	umull	r9, r12, lr, vl
	str	r9, [rp], #4
L(skip1):
	tst	n, #2
	beq	L(skip2)
	mov	r8, r12
	ldmia	up!, { r12, lr }
	mov	r9, #0
	umlal	r8, r9, r12, vl
	mov	r12, #0
	umlal	r9, r12, lr, vl
	stmia	rp!, { r8, r9 }
L(skip2):
	bics	n, n, #3
	beq	L(rtn)
	stmfd	sp!, { r6, r7 }

L(top):	mov	r6, r12
	ldmia	up!, { r8, r9, r12, lr }
	ldr	r7, [rp, #12]			C cache allocate
	mov	r7, #0
	umlal	r6, r7, r8, vl
	mov	r8, #0
	umlal	r7, r8, r9, vl
	mov	r9, #0
	umlal	r8, r9, r12, vl
	mov	r12, #0
	umlal	r9, r12, lr, vl
	subs	n, n, #4
	stmia	rp!, { r6, r7, r8, r9 }
	bne	L(top)

	ldmfd	sp!, { r6, r7 }

L(rtn):	mov	r0, r12
	ldmfd	sp!, { r8, r9, pc }
EPILOGUE()
