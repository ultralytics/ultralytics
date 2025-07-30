dnl  ARM mpn_addmul_3.

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

C	     cycles/limb
C StrongARM:	 -
C XScale	 -
C ARM11		 4.33
C Cortex-A7	 3.23
C Cortex-A8	 3.19
C Cortex-A9	 2.125
C Cortex-A15	 2

C TODO
C  * Use a fast path for n <= KARATSUBA_MUL_THRESHOLD using a jump table,
C    avoiding the current multiply.
C  * Start the first multiply or multiplies early.

define(`rp',`r0')
define(`up',`r1')
define(`n', `r2')
define(`vp',`r3')

define(`v0',`r4')  define(`v1',`r5')  define(`v2',`r6')
define(`u0',`r3')  define(`u1',`r14')
define(`w0',`r7')  define(`w1',`r8')  define(`w2',`r9')
define(`cy0',`r10')  define(`cy1',`r11') define(`cy2',`r12')


ASM_START()
PROLOGUE(mpn_addmul_3)
	push	{ r4-r11, r14 }

	ldr	w0, =0xaaaaaaab		C 3^{-1} mod 2^32
	ldm	vp, { v0,v1,v2 }
	mov	cy0, #0
	mov	cy1, #0
	mov	cy2, #0

C Tricky n mod 6
	mul	w0, w0, n		C n * 3^{-1} mod 2^32
	and	w0, w0, #0xc0000001	C pseudo-CRT mod 3,2
	sub	n, n, #3
ifdef(`PIC',`
	add	pc, pc, w0, ror $28
	nop
	b	L(b0)
	b	L(b2)
	b	L(b4)
	.word	0xe7f000f0	C udf
	b	L(b3)
	b	L(b5)
	b	L(b1)
',`
	ldr	pc, [pc, w0, ror $28]
	nop
	.word	L(b0), L(b2), L(b4), 0, L(b3), L(b5), L(b1)
')

L(b5):	add	up, up, #-8
	ldr	w1, [rp, #0]
	ldr	w2, [rp, #4]
	ldr	u1, [up, #8]
	b	L(lo5)

L(b4):	add	rp, rp, #-4
	add	up, up, #-12
	ldr	w2, [rp, #4]
	ldr	w0, [rp, #8]
	ldr	u0, [up, #12]
	b	L(lo4)

L(b3):	add	rp, rp, #-8
	add	up, up, #-16
	ldr	w0, [rp, #8]
	ldr	w1, [rp, #12]
	ldr	u1, [up, #16]
	b	L(lo3)

L(b1):	add	rp, rp, #8
	ldr	w2, [rp, #-8]
	ldr	w0, [rp, #-4]
	ldr	u1, [up, #0]
	b	L(lo1)

L(b0):	add	rp, rp, #4
	add	up, up, #-4
	ldr	w0, [rp, #-4]
	ldr	w1, [rp, #0]
	ldr	u0, [up, #4]
	b	L(lo0)

L(b2):	add	rp, rp, #12
	add	up, up, #4
	ldr	w1, [rp, #-12]
	ldr	w2, [rp, #-8]
	ldr	u0, [up, #-4]

	ALIGN(16)
L(top):	ldr	w0, [rp, #-4]
	umaal	w1, cy0, u0, v0
	ldr	u1, [up, #0]
	umaal	w2, cy1, u0, v1
	str	w1, [rp, #-12]
	umaal	w0, cy2, u0, v2
L(lo1):	ldr	w1, [rp, #0]
	umaal	w2, cy0, u1, v0
	ldr	u0, [up, #4]
	umaal	w0, cy1, u1, v1
	str	w2, [rp, #-8]
	umaal	w1, cy2, u1, v2
L(lo0):	ldr	w2, [rp, #4]
	umaal	w0, cy0, u0, v0
	ldr	u1, [up, #8]
	umaal	w1, cy1, u0, v1
	str	w0, [rp, #-4]
	umaal	w2, cy2, u0, v2
L(lo5):	ldr	w0, [rp, #8]
	umaal	w1, cy0, u1, v0
	ldr	u0, [up, #12]
	umaal	w2, cy1, u1, v1
	str	w1, [rp, #0]
	umaal	w0, cy2, u1, v2
L(lo4):	ldr	w1, [rp, #12]
	umaal	w2, cy0, u0, v0
	ldr	u1, [up, #16]
	umaal	w0, cy1, u0, v1
	str	w2, [rp, #4]
	umaal	w1, cy2, u0, v2
L(lo3):	ldr	w2, [rp, #16]
	umaal	w0, cy0, u1, v0
	ldr	u0, [up, #20]
	umaal	w1, cy1, u1, v1
	str	w0, [rp, #8]
	umaal	w2, cy2, u1, v2
L(lo2):	subs	n, n, #6
	add	up, up, #24
	add	rp, rp, #24
	bge	L(top)

L(end):	umaal	w1, cy0, u0, v0
	ldr	u1, [up, #0]
	umaal	w2, cy1, u0, v1
	str	w1, [rp, #-12]
	mov	w0, #0
	umaal	w0, cy2, u0, v2
	umaal	w2, cy0, u1, v0
	umaal	w0, cy1, u1, v1
	str	w2, [rp, #-8]
	umaal	cy1, cy2, u1, v2
	adds	w0, w0, cy0
	str	w0, [rp, #-4]
	adcs	w1, cy1, #0
	str	w1, [rp, #0]
	adc	r0, cy2, #0

	pop	{ r4-r11, pc }
EPILOGUE()
