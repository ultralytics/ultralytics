dnl  ARM mpn_mod_1s_2p

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
C StrongARM	 -
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 4.25
C Cortex-A15	 3

define(`ap', `r0')
define(`n',  `r1')
define(`d',  `r2')
define(`cps',`r3')

ASM_START()
PROLOGUE(mpn_mod_1s_2p)
	push	{r4-r10}
	tst	n, #1
	add	r7, r3, #8
	ldmia	r7, {r7, r8, r12}	C load B1, B2, B3
	add	ap, ap, n, lsl #2	C put ap at operand end
	beq	L(evn)

L(odd):	subs	n, n, #1
	beq	L(1)
	ldmdb	ap!, {r4,r6,r9}
	mov	r10, #0
	umlal	r4, r10, r6, r7
	umlal	r4, r10, r9, r8
	b	L(com)

L(evn):	ldmdb	ap!, {r4,r10}
L(com):	subs	n, n, #2
	ble	L(end)
	ldmdb	ap!, {r5,r6}
	b	L(mid)

L(top):	mov	r9, #0
	umlal	r5, r9, r6, r7		C B1
	umlal	r5, r9, r4, r8		C B2
	ldmdb	ap!, {r4,r6}
	umlal	r5, r9, r10, r12	C B3
	ble	L(xit)
	mov	r10, #0
	umlal	r4, r10, r6, r7		C B1
	umlal	r4, r10, r5, r8		C B2
	ldmdb	ap!, {r5,r6}
	umlal	r4, r10, r9, r12	C B3
L(mid):	subs	n, n, #4
	bge	L(top)

	mov	r9, #0
	umlal	r5, r9, r6, r7		C B1
	umlal	r5, r9, r4, r8		C B2
	umlal	r5, r9, r10, r12	C B3
	mov	r4, r5

L(end):	movge	   r9, r10		C executed iff coming via xit
	ldr	r6, [r3, #4]		C cps[1] = cnt
	mov	r5, #0
	umlal	r4, r5, r9, r7
	mov	r7, r5, lsl r6
L(x):	rsb	r1, r6, #32
	orr	r8, r7, r4, lsr r1
	mov	r9, r4, lsl r6
	ldr	r5, [r3, #0]
	add	r0, r8, #1
	umull	r12, r1, r8, r5
	adds	r4, r12, r9
	adc	r1, r1, r0
	mul	r5, r2, r1
	sub	r9, r9, r5
	cmp	r9, r4
	addhi	r9, r9, r2
	cmp	r2, r9
	subls	r9, r9, r2
	mov	r0, r9, lsr r6
	pop	{r4-r10}
	bx	r14

L(xit):	mov	r10, #0
	umlal	r4, r10, r6, r7		C B1
	umlal	r4, r10, r5, r8		C B2
	umlal	r4, r10, r9, r12	C B3
	b	L(end)

L(1):	ldr	r6, [r3, #4]		C cps[1] = cnt
	ldr	r4, [ap, #-4]		C ap[0]
	mov	r7, #0
	b	L(x)
EPILOGUE()

PROLOGUE(mpn_mod_1s_2p_cps)
	push	{r4-r8, r14}
	clz	r4, r1
	mov	r5, r1, lsl r4		C b <<= cnt
	mov	r6, r0			C r6 = cps
	mov	r0, r5
	bl	mpn_invert_limb
	rsb	r3, r4, #32
	mov	r3, r0, lsr r3
	mov	r2, #1
	orr	r3, r3, r2, lsl r4
	rsb	r1, r5, #0
	mul	r2, r1, r3
	umull	r3, r12, r2, r0
	add	r12, r2, r12
	mvn	r12, r12
	mul	r1, r5, r12
	cmp	r1, r3
	addhi	r1, r1, r5
	umull	r12, r7, r1, r0
	add	r7, r1, r7
	mvn	r7, r7
	mul	r3, r5, r7
	cmp	r3, r12
	addhi	r3, r3, r5
	mov	r5, r2, lsr r4
	mov	r7, r1, lsr r4
	mov	r8, r3, lsr r4
	stmia	r6, {r0,r4,r5,r7,r8}	C fill cps
	pop	{r4-r8, pc}
EPILOGUE()
