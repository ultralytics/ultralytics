dnl  ARM mpn_submul_1 optimised for A15.

dnl  Copyright 2012, 2013 Free Software Foundation, Inc.

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

C	     cycles/limb		best
C StrongARM:     -
C XScale	 ?
C Cortex-A7	 ?
C Cortex-A8	 ?
C Cortex-A9	 5.75			3.75
C Cortex-A15	 2.32			this

C This code uses umlal and umaal for adding in the rp[] data, keeping the
C recurrency path separate from any multiply instructions.  It performs well on
C A15, but not quite at the multiply bandwidth like the corresponding addmul_1
C code.
C
C We don't use r12 due to ldrd and strd limitations.
C
C This loop complements U on the fly,
C   U' = B^n - 1 - U
C and then uses that
C   R - U*v = R + U'*v + v - B^n v

C Architecture requirements:
C v5	-
C v5t	-
C v5te	ldrd strd
C v6	umaal
C v6t2	-
C v7a	-

define(`rp', `r0')
define(`up', `r1')
define(`n',  `r2')
define(`v0', `r3')

define(`w0', `r10') define(`w1', `r11')
define(`u0', `r8')  define(`u1', `r9')

ASM_START()
PROLOGUE(mpn_submul_1)
	sub	sp, sp, #32
	strd	r10, r11, [sp, #24]
	strd	r8, r9, [sp, #16]
	strd	r6, r7, [sp, #8]
	strd	r4, r5, [sp, #0]
C	push	{ r4-r11 }

	ands	r6, n, #3
	sub	n, n, #3
	beq	L(b00)
	cmp	r6, #2
	bcc	L(b01)
	beq	L(b10)

L(b11):	mov	r6, #0
	ldr	u1, [up], #-4
	ldr	w1, [rp], #-16
	mvn	u1, u1
	adds	r7, v0, #0
	b	L(mid)

L(b00):	ldrd	u0, u1, [up]
	ldrd	w0, w1, [rp], #-12
	mvn	u0, u0
	mvn	u1, u1
	mov	r6, v0
	umaal	w0, r6, u0, v0
	cmn	r13, #0			C carry clear
	mov	r7, #0
	str	w0, [rp, #12]
	b	L(mid)

L(b10):	ldrd	u0, u1, [up], #8
	ldrd	w0, w1, [rp]
	mvn	u0, u0
	mvn	u1, u1
	mov	r4, v0
	umaal	w0, r4, u0, v0
	mov	r5, #0
	str	w0, [rp], #-4
	umlal	w1, r5, u1, v0
	adds	n, n, #0
	bmi	L(end)
	b	L(top)

L(b01):	ldr	u1, [up], #4
	ldr	w1, [rp], #-8
	mvn	u1, u1
	mov	r5, v0
	mov	r4, #0
	umaal	w1, r5, u1, v0
	tst	n, n
	bmi	L(end)

C	ALIGN(16)
L(top):	ldrd	u0, u1, [up, #0]
	adcs	r4, r4, w1
	mvn	u0, u0
	ldrd	w0, w1, [rp, #12]
	mvn	u1, u1
	mov	r6, #0
	umlal	w0, r6, u0, v0		C 1 2
	adcs	r5, r5, w0
	mov	r7, #0
	strd	r4, r5, [rp, #8]
L(mid):	umaal	w1, r7, u1, v0		C 2 3
	ldrd	u0, u1, [up, #8]
	add	up, up, #16
	adcs	r6, r6, w1
	mvn	u0, u0
	ldrd	w0, w1, [rp, #20]
	mvn	u1, u1
	mov	r4, #0
	umlal	w0, r4, u0, v0		C 3 4
	adcs	r7, r7, w0
	mov	r5, #0
	strd	r6, r7, [rp, #16]!
	sub	n, n, #4
	umlal	w1, r5, u1, v0		C 0 1
	tst	n, n
	bpl	L(top)

L(end):	adcs	r4, r4, w1
	str	r4, [rp, #8]
	adc	r0, r5, #0
	sub	r0, v0, r0
	pop	{ r4-r11 }
	bx	r14
EPILOGUE()
