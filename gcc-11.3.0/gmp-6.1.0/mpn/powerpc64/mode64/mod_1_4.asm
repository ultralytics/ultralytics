dnl  PowerPC-64 mpn_mod_1s_4p

dnl  Copyright 2010, 2011 Free Software Foundation, Inc.

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

C                   cycles/limb
C POWER3/PPC630          ?
C POWER4/PPC970          9
C POWER5                 9
C POWER6                13
C POWER7                3.5

C TODO
C  * Optimise, in particular the cps function.  This was compiler-generated and
C    then hand optimised.

C INPUT PARAMETERS
define(`ap',  `r3')
define(`n',   `r4')
define(`d',   `r5')
define(`cps', `r6')

ASM_START()

EXTERN_FUNC(mpn_invert_limb)

PROLOGUE(mpn_mod_1s_4p)
	std	r23, -72(r1)
	ld	r23, 48(cps)
	std	r24, -64(r1)
	std	r25, -56(r1)
	ld	r24, 32(cps)
	ld	r25, 24(cps)
	std	r26, -48(r1)
	std	r27, -40(r1)
	ld	r26, 16(cps)
	std	r28, -32(r1)
	std	r29, -24(r1)
	std	r30, -16(r1)
	std	r31, -8(r1)
	ld	r30, 40(cps)

	rldicl.	r0, n, 0,62
	sldi	r31, n, 3
	add	ap, ap, r31		C make ap point at end of operand

	cmpdi	cr7, r0, 2
	beq	cr0, L(b00)
	blt	cr7, L(b01)
	beq	cr7, L(b10)

L(b11):	ld	r11, -16(ap)
	ld	r9, -8(ap)
	ld	r0, -24(ap)
	mulhdu	r27, r11, r26
	mulld	r8, r11, r26
	mulhdu	r11, r9, r25
	mulld	r9, r9, r25
	addc	r31, r8, r0
	addze	r10, r27
	addc	r0, r9, r31
	adde	r9, r11, r10
	addi	ap, ap, -40
	b	L(6)

	ALIGN(16)
L(b00):	ld	r11, -24(ap)
	ld	r10, -16(ap)
	ld	r9, -8(ap)
	ld	r0, -32(ap)
	mulld	r8, r11, r26
	mulhdu	r7, r10, r25
	mulhdu	r27, r11, r26
	mulhdu	r11, r9, r24
	mulld	r10, r10, r25
	mulld	r9, r9, r24
	addc	r31, r8, r0
	addze	r0, r27
	addc	r8, r31, r10
	adde	r10, r0, r7
	addc	r0, r9, r8
	adde	r9, r11, r10
	addi	ap, ap, -48
	b	L(6)

	ALIGN(16)
L(b01):	li	r9, 0
	ld	r0, -8(ap)
	addi	ap, ap, -24
	b	L(6)

	ALIGN(16)
L(b10):	ld	r9, -8(ap)
	ld	r0, -16(ap)
	addi	ap, ap, -32

	ALIGN(16)
L(6):	addi	r10, n, 3
	srdi	r7, r10, 2
	mtctr	r7
	bdz	L(end)

	ALIGN(16)
L(top):	ld	r31, -16(ap)
	ld	r10, -8(ap)
	ld	r11, 8(ap)
	ld	r12, 0(ap)
	mulld	r29, r0, r30		C rl * B4modb
	mulhdu	r0,  r0, r30		C rl * B4modb
	mulhdu	r27, r10, r26
	mulld	r10, r10, r26
	mulhdu	r7, r9, r23		C rh * B5modb
	mulld	r9, r9, r23		C rh * B5modb
	mulhdu	r28, r11, r24
	mulld	r11, r11, r24
	mulhdu	r4, r12, r25
	mulld	r12, r12, r25
	addc	r8, r10, r31
	addze	r10, r27
	addi	ap, ap, -32
	addc	r27, r8, r12
	adde	r12, r10, r4
	addc	r11, r27, r11
	adde	r31, r12, r28
	addc	r12, r11, r29
	adde	r4, r31, r0
	addc	r0, r9, r12
	adde	r9, r7, r4
	bdnz	L(top)

L(end):
ifdef(`HAVE_LIMB_LITTLE_ENDIAN',
`	lwz	r3, 8(cps)',
`	lwz	r3, 12(cps)')
	mulld	r10, r9, r26
	mulhdu	r9, r9, r26
	addc	r11, r0, r10
	addze	r9, r9
	ld	r10, 0(cps)
	subfic	r8, r3, 64
	sld	r9, r9, r3
	srd	r8, r11, r8
	sld	r11, r11, r3
	or	r9, r8, r9
	mulld	r0, r9, r10
	mulhdu	r10, r9, r10
	addi	r9, r9, 1
	addc	r8, r0, r11
	adde	r0, r10, r9
	mulld	r0, r0, d
	subf	r0, r0, r11
	cmpld	cr7, r8, r0
	bge	cr7, L(9)
	add	r0, r0, d
L(9):	cmpld	cr7, r0, d
	bge-	cr7, L(16)
L(10):	srd	r3, r0, r3
	ld	r23, -72(r1)
	ld	r24, -64(r1)
	ld	r25, -56(r1)
	ld	r26, -48(r1)
	ld	r27, -40(r1)
	ld	r28, -32(r1)
	ld	r29, -24(r1)
	ld	r30, -16(r1)
	ld	r31, -8(r1)
	blr

L(16):	subf	r0, d, r0
	b	L(10)
EPILOGUE()

PROLOGUE(mpn_mod_1s_4p_cps,toc)
	mflr	r0
	std	r29, -24(r1)
	std	r30, -16(r1)
	mr	r29, r3
	std	r0, 16(r1)
	std	r31, -8(r1)
	stdu	r1, -144(r1)
	cntlzd	r31, r4
	sld	r30, r4, r31
	mr	r3, r30
	CALL(	mpn_invert_limb)
	subfic	r9, r31, 64
	li	r10, 1
	sld	r10, r10, r31
	srd	r9, r3, r9
	neg	r0, r30
	or	r10, r10, r9
	mulld	r10, r10, r0
	mulhdu	r11, r10, r3
	nor	r11, r11, r11
	subf	r11, r10, r11
	mulld	r11, r11, r30
	mulld	r0, r10, r3
	cmpld	cr7, r0, r11
	bge	cr7, L(18)
	add	r11, r11, r30
L(18):	mulhdu	r9, r11, r3
	add	r9, r11, r9
	nor	r9, r9, r9
	mulld	r9, r9, r30
	mulld	r0, r11, r3
	cmpld	cr7, r0, r9
	bge	cr7, L(19)
	add	r9, r9, r30
L(19):	mulhdu	r0, r9, r3
	add	r0, r9, r0
	nor	r0, r0, r0
	mulld	r0, r0, r30
	mulld	r8, r9, r3
	cmpld	cr7, r8, r0
	bge	cr7, L(20)
	add	r0, r0, r30
L(20):	mulhdu	r8, r0, r3
	add	r8, r0, r8
	nor	r8, r8, r8
	mulld	r8, r8, r30
	mulld	r7, r0, r3
	cmpld	cr7, r7, r8
	bge	cr7, L(21)
	add	r8, r8, r30
L(21):	srd	r0, r0, r31
	addi	r1, r1, 144
	srd	r8, r8, r31
	srd	r10, r10, r31
	srd	r11, r11, r31
	std	r0, 40(r29)
	std	r31, 8(r29)
	srd	r9, r9, r31
	ld	r0, 16(r1)
	ld	r30, -16(r1)
	std	r8, 48(r29)
	std	r3, 0(r29)
	mtlr	r0
	ld	r31, -8(r1)
	std	r10, 16(r29)
	std	r11, 24(r29)
	std	r9, 32(r29)
	ld	r29, -24(r1)
	blr
EPILOGUE()
