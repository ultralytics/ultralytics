dnl  ARM v4 mpn_modexact_1c_odd

dnl  Contributed to the GNU project by Torbjorn Granlund.

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

C               cycles/limb       cycles/limb
C               norm    unorm    modexact_1c_odd
C StrongARM	 ?
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	10	12
C Cortex-A15	 9	 9

C Architecture requirements:
C v5	-
C v5t	-
C v5te	-
C v6	-
C v6t2	-
C v7a	-

define(`rp', `r0')
define(`up', `r1')
define(`n',  `r2')
define(`d',  `r3')

define(`cy', `r7')
define(`cnt', `r6')
define(`tnc', `r8')

ASM_START()
PROLOGUE(mpn_divexact_1)
	tst	d, #1
	push	{r4-r9}
	mov	cnt, #0
	bne	L(inv)

C count trailing zeros
	movs	r4, d, lsl #16
	moveq	d, d, lsr #16
	moveq	cnt, #16
	tst	d, #0xff
	moveq	d, d, lsr #8
	addeq	cnt, cnt, #8
	LEA(	r4, ctz_tab)
	and	r5, d, #0xff
	ldrb	r4, [r4, r5]
	mov	d, d, lsr r4
	add	cnt, cnt, r4

C binvert limb
L(inv):	LEA(	r4, binvert_limb_table)
	and	r12, d, #254
	ldrb	r4, [r4, r12, lsr #1]
	mul	r12, r4, r4
	mul	r12, d, r12
	rsb	r12, r12, r4, lsl #1
	mul	r4, r12, r12
	mul	r4, d, r4
	rsb	r4, r4, r12, lsl #1	C r4 = inverse

	tst	cnt, cnt
	ldr	r5, [up], #4		C up[0]
	mov	cy, #0
	bne	L(unnorm)

L(norm):
	subs	n, n, #1		C set carry as side-effect
	beq	L(end)

	ALIGN(16)
L(top):	sbcs	cy, r5, cy
	ldr	r5, [up], #4
	sub	n, n, #1
	mul	r9, r4, cy
	tst	n, n
	umull	r12, cy, d, r9
	str	r9, [rp], #4
	bne	L(top)

L(end):	sbc	cy, r5, cy
	mul	r9, r4, cy
	str	r9, [rp]
	pop	{r4-r9}
	bx	r14

L(unnorm):
	rsb	tnc, cnt, #32
	mov	r5, r5, lsr cnt
	subs	n, n, #1		C set carry as side-effect
	beq	L(edu)

	ALIGN(16)
L(tpu):	ldr	r12, [up], #4
	orr	r9, r5, r12, lsl tnc
	mov	r5, r12, lsr cnt
	sbcs	cy, r9, cy		C critical path ->cy->cy->
	sub	n, n, #1
	mul	r9, r4, cy		C critical path ->cy->r9->
	tst	n, n
	umull	r12, cy, d, r9		C critical path ->r9->cy->
	str	r9, [rp], #4
	bne	L(tpu)

L(edu):	sbc	cy, r5, cy
	mul	r9, r4, cy
	str	r9, [rp]
	pop	{r4-r9}
	bx	r14
EPILOGUE()

	RODATA
ctz_tab:
	.byte	8,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0
	.byte	5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0
	.byte	6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0
	.byte	5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0
	.byte	7,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0
	.byte	5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0
	.byte	6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0
	.byte	5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0
