dnl  ARM mpn_addmul_1 optimised for A15.

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
C Cortex-A9	 6			3.25
C Cortex-A15	 2			this

C This code uses umlal for adding in the rp[] data, keeping the recurrency path
C separate from any multiply instructions.  It performs well on A15, at umlal's
C bandwidth.
C
C An A9 variant should perhaps stick to 3-way unrolling, and use ldm and stm
C for all loads and stores.  Alternatively, it could do 2-way or 4-way, but
C then alignment aware code will be necessary (adding O(1) bookkeeping
C overhead).
C
C We don't use r12 due to ldrd and strd limitations.

C Architecture requirements:
C v5	-
C v5t	-
C v5te	ldrd strd
C v6	-
C v6t2	-
C v7a	-

define(`rp', `r0')
define(`up', `r1')
define(`n',  `r2')
define(`v0', `r3')

define(`w0', `r10') define(`w1', `r11')
define(`u0', `r8')  define(`u1', `r9')

ASM_START()
PROLOGUE(mpn_addmul_1)
	push	{ r4-r11 }

	ands	r6, n, #3
	sub	n, n, #3
	beq	L(b00)
	cmp	r6, #2
	bcc	L(b01)
	beq	L(b10)

L(b11):	mov	r6, #0
	cmn	r13, #0			C carry clear
	ldr	u1, [up], #-4
	ldr	w1, [rp], #-4
	mov	r7, #0
	b	L(mid)

L(b00):	ldrd	u0, u1, [up]
	ldrd	w0, w1, [rp]
	mov	r6, #0
	umlal	w0, r6, u0, v0
	cmn	r13, #0			C carry clear
	mov	r7, #0
	str	w0, [rp]
	b	L(mid)

L(b10):	ldrd	u0, u1, [up], #8
	ldrd	w0, w1, [rp]
	mov	r4, #0
	umlal	w0, r4, u0, v0
	cmn	r13, #0			C carry clear
	mov	r5, #0
	str	w0, [rp], #8
	umlal	w1, r5, u1, v0
	tst	n, n
	bmi	L(end)
	b	L(top)

L(b01):	mov	r4, #0
	ldr	u1, [up], #4
	ldr	w1, [rp], #4
	mov	r5, #0
	umlal	w1, r5, u1, v0
	tst	n, n
	bmi	L(end)

	ALIGN(16)
L(top):	ldrd	u0, u1, [up, #0]
	adcs	r4, r4, w1
	ldrd	w0, w1, [rp, #0]
	mov	r6, #0
	umlal	w0, r6, u0, v0		C 1 2
	adcs	r5, r5, w0
	mov	r7, #0
	strd	r4, r5, [rp, #-4]
L(mid):	umlal	w1, r7, u1, v0		C 2 3
	ldrd	u0, u1, [up, #8]
	adcs	r6, r6, w1
	ldrd	w0, w1, [rp, #8]
	mov	r4, #0
	umlal	w0, r4, u0, v0		C 3 4
	adcs	r7, r7, w0
	mov	r5, #0
	strd	r6, r7, [rp, #4]
	umlal	w1, r5, u1, v0		C 0 1
	sub	n, n, #4
	add	up, up, #16
	add	rp, rp, #16
	tst	n, n
	bpl	L(top)

L(end):	adcs	r4, r4, w1
	str	r4, [rp, #-4]
	adc	r0, r5, #0
	pop	{ r4-r11 }
	bx	r14
EPILOGUE()
