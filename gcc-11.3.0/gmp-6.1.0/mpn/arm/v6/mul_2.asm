dnl  ARM mpn_mul_2.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2012 Free Software Foundation, Inc.

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
C StrongARM:	 -
C XScale	 -
C ARM11		 5.25
C Cortex-A7	 3.13
C Cortex-A8	 5
C Cortex-A9	 2.25
C Cortex-A15	 2.5

C TODO
C  * This is a trivial edit of the addmul_2 code.  Check for simplifications,
C    and possible speedups to 2.0 c/l.

define(`rp',`r0')
define(`up',`r1')
define(`n', `r2')
define(`vp',`r3')

define(`v0',`r6')
define(`v1',`r7')
define(`u0',`r3')
define(`u1',`r9')

define(`cya',`r8')
define(`cyb',`r12')


ASM_START()
PROLOGUE(mpn_mul_2)
	push	{ r4, r5, r6, r7, r8, r9 }

	ldm	vp, { v0, v1 }
	mov	cya, #0
	mov	cyb, #0

	tst	n, #1
	beq	L(evn)
L(odd):	mov	r5, #0
	ldr	u0, [up, #0]
	mov	r4, #0
	tst	n, #2
	beq	L(fi1)
L(fi3):	sub	up, up, #12
	sub	rp, rp, #16
	b	L(lo3)
L(fi1):	sub	n, n, #1
	sub	up, up, #4
	sub	rp, rp, #8
	b	L(lo1)
L(evn):	mov	r4, #0
	ldr	u1, [up, #0]
	mov	r5, #0
	tst	n, #2
	bne	L(fi2)
L(fi0):	sub	up, up, #8
	sub	rp, rp, #12
	b	L(lo0)
L(fi2):	subs	n, n, #2
	sub	rp, rp, #4
	bls	L(end)

	ALIGN(16)
L(top):	ldr	u0, [up, #4]
	umaal	r4, cya, u1, v0
	str	r4, [rp, #4]
	mov	r4, #0
	umaal	r5, cyb, u1, v1
L(lo1):	ldr	u1, [up, #8]
	umaal	r5, cya, u0, v0
	str	r5, [rp, #8]
	mov	r5, #0
	umaal	r4, cyb, u0, v1
L(lo0):	ldr	u0, [up, #12]
	umaal	r4, cya, u1, v0
	str	r4, [rp, #12]
	mov	r4, #0
	umaal	r5, cyb, u1, v1
L(lo3):	ldr	u1, [up, #16]!
	umaal	r5, cya, u0, v0
	str	r5, [rp, #16]!
	mov	r5, #0
	umaal	r4, cyb, u0, v1
	subs	n, n, #4
	bhi	L(top)

L(end):	umaal	r4, cya, u1, v0
	ldr	u0, [up, #4]
	umaal	r5, cyb, u1, v1
	str	r4, [rp, #4]
	umaal	r5, cya, u0, v0
	umaal	cya, cyb, u0, v1
	str	r5, [rp, #8]
	str	cya, [rp, #12]
	mov	r0, cyb

	pop	{ r4, r5, r6, r7, r8, r9 }
	bx	r14
EPILOGUE()
