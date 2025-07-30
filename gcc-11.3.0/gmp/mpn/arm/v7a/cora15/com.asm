dnl  ARM mpn_com optimised for A15.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2013 Free Software Foundation, Inc.

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

C            cycles/limb
C StrongARM	 ?
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	2.5
C Cortex-A15	1.0

C This is great A15 core register code, but it is a bit large.
C We use FEEDIN_VARIANT 1 to save some space, but use 8-way unrolling.

C Architecture requirements:
C v5	-
C v5t	-
C v5te	ldrd strd
C v6	-
C v6t2	-
C v7a	-

define(`FEEDIN_VARIANT', 1)	C alternatives: 0 1 2
define(`UNROLL', 4x2)		C alternatives: 4 4x2

define(`rp', `r0')
define(`up', `r1')
define(`n',  `r2')

ASM_START()
PROLOGUE(mpn_com)
	push	{ r4-r5,r8-r9 }

ifelse(FEEDIN_VARIANT,0,`
	ands	r12, n, #3
	mov	n, n, lsr #2
	beq	L(b00a)
	tst	r12, #1
	beq	L(bx0)
	ldr	r5, [up], #4
	mvn	r9, r5
	str	r9, [rp], #4
	tst	r12, #2
	beq	L(b00)
L(bx0):	ldrd	r4, r5, [up, #0]
	sub	rp, rp, #8
	b	L(lo)
L(b00):	tst	n, n
	beq	L(wd1)
L(b00a):ldrd	r4, r5, [up], #-8
	sub	rp, rp, #16
	b	L(mid)
')
ifelse(FEEDIN_VARIANT,1,`
	and	r12, n, #3
	mov	n, n, lsr #2
	tst	r12, #1
	beq	L(bx0)
	ldr	r5, [up], #4
	mvn	r9, r5
	str	r9, [rp], #4
L(bx0):	tst	r12, #2
	beq	L(b00)
	ldrd	r4, r5, [up, #0]
	sub	rp, rp, #8
	b	L(lo)
L(b00):	tst	n, n
	beq	L(wd1)
	ldrd	r4, r5, [up], #-8
	sub	rp, rp, #16
	b	L(mid)
')
ifelse(FEEDIN_VARIANT,2,`
	ands	r12, n, #3
	mov	n, n, lsr #2
	beq	L(b00)
	cmp	r12, #2
	bcc	L(b01)
	beq	L(b10)

L(b11):	ldr	r5, [up], #4
	mvn	r9, r5
	ldrd	r4, r5, [up, #0]
	str	r9, [rp], #-4
	b	L(lo)

L(b00):	ldrd	r4, r5, [up], #-8
	sub	rp, rp, #16
	b	L(mid)

L(b01):	ldr	r5, [up], #-4
	mvn	r9, r5
	str	r9, [rp], #-12
	tst	n, n
	beq	L(wd1)
L(gt1):	ldrd	r4, r5, [up, #8]
	b	L(mid)

L(b10):	ldrd	r4, r5, [up]
	sub	rp, rp, #8
	b	L(lo)
')
	ALIGN(16)
ifelse(UNROLL,4,`
L(top):	ldrd	r4, r5, [up, #8]
	strd	r8, r9, [rp, #8]
L(mid):	mvn	r8, r4
	mvn	r9, r5
	ldrd	r4, r5, [up, #16]!
	strd	r8, r9, [rp, #16]!
	sub	n, n, #1
L(lo):	mvn	r8, r4
	mvn	r9, r5
	tst	n, n
	bne	L(top)
')
ifelse(UNROLL,4x2,`
L(top):	ldrd	r4, r5, [up, #8]
	strd	r8, r9, [rp, #8]
L(mid):	mvn	r8, r4
	mvn	r9, r5
	ldrd	r4, r5, [up, #16]
	strd	r8, r9, [rp, #16]
	mvn	r8, r4
	mvn	r9, r5
	sub	n, n, #2
	tst	n, n
	bmi	L(dne)
	ldrd	r4, r5, [up, #24]
	strd	r8, r9, [rp, #24]
	mvn	r8, r4
	mvn	r9, r5
	ldrd	r4, r5, [up, #32]!
	strd	r8, r9, [rp, #32]!
L(lo):	mvn	r8, r4
	mvn	r9, r5
	tst	n, n
	bne	L(top)
')

L(end):	strd	r8, r9, [rp, #8]
L(wd1):	pop	{ r4-r5,r8-r9 }
	bx	r14
ifelse(UNROLL,4x2,`
L(dne):	strd	r8, r9, [rp, #24]
	pop	{ r4-r5,r8-r9 }
	bx	r14
')
EPILOGUE()
