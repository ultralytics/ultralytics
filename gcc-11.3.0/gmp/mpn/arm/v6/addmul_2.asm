dnl  ARM mpn_addmul_2.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2012, 2013, 2015 Free Software Foundation, Inc.

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
C ARM11		 4.68
C Cortex-A7	 3.625
C Cortex-A8	 4
C Cortex-A9	 2.25
C Cortex-A15	 2.5

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
PROLOGUE(mpn_addmul_2)
	push	{ r4-r9 }

	ldrd	v0, v1, [vp, #0]
	mov	cya, #0
	mov	cyb, #0

	tst	n, #1
	beq	L(evn)

L(odd):	ldr	u1, [up, #0]
	ldr	r4, [rp, #0]
	tst	n, #2
	beq	L(fi1)
L(fi3):	sub	up, up, #8
	sub	rp, rp, #8
	b	L(lo3)
L(fi1):	sub	n, n, #1
	b	L(top)

L(evn):	ldr	u0, [up, #0]
	ldr	r5, [rp, #0]
	tst	n, #2
	bne	L(fi2)
L(fi0):	sub	up, up, #4
	sub	rp, rp, #4
	b	L(lo0)
L(fi2):	sub	up, up, #12
	sub	rp, rp, #12
	b	L(lo2)

	ALIGN(16)
L(top):	ldr	r5, [rp, #4]
	umaal	r4, cya, u1, v0
	ldr	u0, [up, #4]
	umaal	r5, cyb, u1, v1
	str	r4, [rp, #0]
L(lo0):	ldr	r4, [rp, #8]
	umaal	r5, cya, u0, v0
	ldr	u1, [up, #8]
	umaal	r4, cyb, u0, v1
	str	r5, [rp, #4]
L(lo3):	ldr	r5, [rp, #12]
	umaal	r4, cya, u1, v0
	ldr	u0, [up, #12]
	umaal	r5, cyb, u1, v1
	str	r4, [rp, #8]
L(lo2):	ldr	r4, [rp, #16]!
	umaal	r5, cya, u0, v0
	ldr	u1, [up, #16]!
	umaal	r4, cyb, u0, v1
	str	r5, [rp, #-4]
	subs	n, n, #4
	bhi	L(top)

L(end):	umaal	r4, cya, u1, v0
	umaal	cya, cyb, u1, v1
	str	r4, [rp, #0]
	str	cya, [rp, #4]
	mov	r0, cyb

	pop	{ r4-r9 }
	bx	r14
EPILOGUE()
