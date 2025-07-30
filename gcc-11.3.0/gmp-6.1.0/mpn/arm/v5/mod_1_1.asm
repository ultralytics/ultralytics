dnl  ARM mpn_mod_1_1p

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
C Cortex-A9	 7
C Cortex-A15	 6

define(`ap', `r0')
define(`n',  `r1')
define(`d',  `r2')
define(`cps',`r3')

ASM_START()
PROLOGUE(mpn_mod_1_1p)
	push	{r4-r10}
	add	r0, r0, r1, asl #2
	ldr	r5, [r0, #-4]!
	ldr	r12, [r0, #-4]!
	subs	r1, r1, #2
	ble	L(4)
	ldr	r8, [r3, #12]
	mov	r4, r12
	mov	r10, r5
	umull	r7, r5, r10, r8
	sub	r1, r1, #1
	b	L(mid)

L(top):	adds	r12, r6, r7
	adcs	r10, r4, r5
	sub	r1, r1, #1
	mov	r6, #0
	movcs	r6, r8
	umull	r7, r5, r10, r8
	adds	r4, r12, r6
	subcs	r4, r4, r2
L(mid):	ldr	r6, [r0, #-4]!
	teq	r1, #0
	bne	L(top)

	adds	r12, r6, r7
	adcs	r5, r4, r5
	subcs	r5, r5, r2
L(4):	ldr	r1, [r3, #4]
	cmp	r1, #0
	beq	L(7)
	ldr	r4, [r3, #8]
	umull	r0, r6, r5, r4
	adds	r12, r0, r12
	addcs	r6, r6, #1
	rsb	r0, r1, #32
	mov	r0, r12, lsr r0
	orr	r5, r0, r6, asl r1
	mov	r12, r12, asl r1
	b	L(8)
L(7):	cmp	r5, r2
	subcs	r5, r5, r2
L(8):	ldr	r0, [r3, #0]
	umull	r4, r3, r5, r0
	add	r5, r5, #1
	adds	r0, r4, r12
	adc	r5, r3, r5
	mul	r5, r2, r5
	sub	r12, r12, r5
	cmp	r12, r0
	addhi	r12, r12, r2
	cmp	r2, r12
	subls	r12, r12, r2
	mov	r0, r12, lsr r1
	pop	{r4-r10}
	bx	r14
EPILOGUE()

PROLOGUE(mpn_mod_1_1p_cps)
	stmfd	sp!, {r4, r5, r6, r14}
	mov	r5, r0
	clz	r4, r1
	mov	r0, r1, asl r4
	rsb	r6, r0, #0
	bl	mpn_invert_limb
	str	r0, [r5, #0]
	str	r4, [r5, #4]
	cmp	r4, #0
	beq	L(2)
	rsb	r1, r4, #32
	mov	r3, #1
	mov	r3, r3, asl r4
	orr	r3, r3, r0, lsr r1
	mul	r3, r6, r3
	mov	r4, r3, lsr r4
	str	r4, [r5, #8]
L(2):	mul	r0, r6, r0
	str	r0, [r5, #12]
	ldmfd	sp!, {r4, r5, r6, pc}
EPILOGUE()
